import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3.5-2B text generation on one CUDA GPU.")
    parser.add_argument("model", help="Path to the local Qwen3.5-2B checkpoint")
    parser.add_argument("prompts", nargs="+", help="One text prompt per request")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    from nanovllm import LLM, SamplingParams

    llm = LLM(os.path.expanduser(args.model), enforce_eager=True, tensor_parallel_size=1,
              max_num_seqs=2, max_num_delta_states=2, max_model_len=512,
              max_num_batched_tokens=256)
    prompts = []
    for prompt in args.prompts:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(llm.tokenizer.encode(text, add_special_tokens=False))

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    for output in llm.generate(prompts, sampling_params):
        print(llm.tokenizer.decode(output["token_ids"], skip_special_tokens=True))


if __name__ == "__main__":
    main()
