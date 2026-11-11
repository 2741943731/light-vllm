from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple
import torch
from torch import nn
from safetensors import safe_open


class WeightLoadResult(NamedTuple):
    loaded_parameters: int
    skipped_tensors: int
    tied_aliases: int


class WeightSource(NamedTuple):
    file: Path
    weight_name: str
    param_name: str


def unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in checkpoint index: {key}")
        result[key] = value
    return result


def checkpoint_files(path: Path) -> tuple[list[Path], dict[str, str] | None]:
    if not path.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {path}")
    index_file = path / "model.safetensors.index.json"
    if index_file.is_file():
        with index_file.open() as f:
            index = json.load(f, object_pairs_hook=unique_json_keys)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint index must contain a nonempty weight_map")
        for name, filename in weight_map.items():
            if (not isinstance(name, str) or not isinstance(filename, str)
                    or Path(filename).name != filename or not filename.endswith(".safetensors")):
                raise ValueError("weight_map must map tensor names to local safetensors filenames")
        files = [path / name for name in sorted(set(weight_map.values()))]
    else:
        weight_map = None
        files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint files in {path}")
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {file}")
    return files, weight_map


def map_weight_name(name: str) -> str | None:
    if name.startswith(("model.visual.", "mtp.")):
        return None
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "model." + name[len(prefix):]
    return name


def read_tensor(source: WeightSource) -> torch.Tensor:
    with safe_open(str(source.file), framework="pt", device="cpu") as f:
        return f.get_tensor(source.weight_name)


@torch.no_grad()
def load_model(model: nn.Module, path: str | Path) -> WeightLoadResult:
    """Load unquantized Qwen3.5 text weights into the single-device model.

    Names, shapes, dtypes, completeness and tied copies are checked before any
    parameter is changed. Copies stream one tensor at a time; an I/O or device
    failure during copying does not roll back earlier copies. Recreate the model
    after such a failure. The checkpoint must remain unchanged during loading.
    """
    files, weight_map = checkpoint_files(Path(path))
    params = dict(model.named_parameters(remove_duplicate=False))
    if not params:
        raise ValueError("Model has no parameters to load")
    float_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    groups = {}
    for name, param in params.items():
        if param.device.type == "meta" or param.dtype not in float_dtypes:
            raise ValueError(f"Parameter must be allocated with F16, BF16 or F32 dtype: {name}")
        groups.setdefault(id(param), []).append(name)

    sources = {}
    seen_names, seen_targets = set(), set()
    skipped = 0
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as f:
            names = set(f.keys())
            if weight_map is not None:
                expected = {name for name, shard in weight_map.items() if shard == file.name}
                if names != expected:
                    missing, unexpected = sorted(expected - names), sorted(names - expected)
                    raise ValueError(f"Shard/index mismatch in {file.name}: missing={missing[:8]}, unexpected={unexpected[:8]}")
            for weight_name in sorted(names):
                if weight_name in seen_names:
                    raise ValueError(f"Duplicate checkpoint tensor: {weight_name}")
                seen_names.add(weight_name)
                param_name = map_weight_name(weight_name)
                if param_name is None:
                    skipped += 1
                    continue
                if param_name not in params:
                    raise ValueError(f"Unexpected checkpoint tensor: {weight_name}")
                if param_name in seen_targets:
                    raise ValueError(f"Multiple checkpoint tensors map to parameter: {param_name}")
                seen_targets.add(param_name)
                param = params[param_name]
                tensor = f.get_slice(weight_name)
                if tuple(tensor.get_shape()) != tuple(param.shape):
                    raise ValueError(f"Shape mismatch for {weight_name}: checkpoint={tensor.get_shape()}, model={list(param.shape)}")
                if tensor.get_dtype() not in ("F16", "BF16", "F32"):
                    raise ValueError(f"Unsupported checkpoint dtype for {weight_name}: {tensor.get_dtype()}")
                sources.setdefault(id(param), []).append(WeightSource(file, weight_name, param_name))

    missing = [names[0] for param_id, names in groups.items() if param_id not in sources]
    if missing:
        raise ValueError(f"Missing model parameters ({len(missing)}): {sorted(missing)[:8]}")

    # Shared embedding/head parameters need only one stored copy. If aliases
    # are both present, verify they agree before choosing a single copy to load.
    tied_aliases = 0
    for aliases in sources.values():
        if len(aliases) > 1:
            first = read_tensor(aliases[0])
            for alias in aliases[1:]:
                other = read_tensor(alias)
                if first.dtype != other.dtype or not torch.equal(first, other):
                    raise ValueError(f"Conflicting tied weights: {aliases[0].weight_name} and {alias.weight_name}")
                tied_aliases += 1
                del other
            del first

    plan = {file: [] for file in files}
    for aliases in sources.values():
        source = aliases[0]
        plan[source.file].append(source)
    for file, entries in plan.items():
        if not entries:
            continue
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for source in entries:
                params[source.param_name].copy_(f.get_tensor(source.weight_name))
    return WeightLoadResult(len(sources), skipped, tied_aliases)
