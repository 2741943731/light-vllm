import os
import warnings
from dataclasses import dataclass
import torch
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_num_delta_states: int | None = None

    @property
    def is_qwen3_5(self) -> bool:
        return getattr(self.hf_config, "model_type", None) in ("qwen3_5", "qwen3_5_text")

    @property
    def model_config(self):
        if self.is_qwen3_5:
            return getattr(self.hf_config, "text_config", self.hf_config)
        return self.hf_config

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        except (KeyError, ValueError) as error:
            if "qwen3_5" not in str(error):
                raise
            raise ValueError(
                "Unable to load Qwen3.5 config. Use a Transformers version with "
                "qwen3_5 support and verify the local config.json."
            ) from error
        if self.is_qwen3_5:
            self.configure_qwen3_5()
        self.max_model_len = min(self.max_model_len, self.model_config.max_position_embeddings)

    def configure_qwen3_5(self):
        if self.tensor_parallel_size != 1:
            raise ValueError("Qwen3.5 currently supports tensor_parallel_size=1 only")
        for name in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

        text_config = self.model_config
        layer_types = getattr(text_config, "layer_types", None)
        if (not isinstance(layer_types, (list, tuple))
                or len(layer_types) != getattr(text_config, "num_hidden_layers", 0)
                or any(t not in ("linear_attention", "full_attention") for t in layer_types)
                or "linear_attention" not in layer_types or "full_attention" not in layer_types):
            raise ValueError("Qwen3.5 requires one layer_types entry per layer, including linear and full attention")
        dtype = getattr(text_config, "dtype", None)
        if dtype is None:
            dtype = getattr(text_config, "torch_dtype", None)
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype)
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Qwen3.5 CUDA attention requires float16 or bfloat16 weights")
        text_config.dtype = dtype

        # Bound resident requests separately from the per-batch sequence limit.
        if self.max_num_delta_states is None:
            self.max_num_delta_states = min(self.max_num_seqs, 16)
        if type(self.max_num_delta_states) is not int or self.max_num_delta_states <= 0:
            raise ValueError("max_num_delta_states must be a positive integer")
        if not self.enforce_eager:
            warnings.warn("Qwen3.5 uses eager execution; enabling enforce_eager=True", stacklevel=2)
            self.enforce_eager = True
