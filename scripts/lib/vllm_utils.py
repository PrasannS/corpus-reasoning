"""Shared vLLM model loading and inference utilities."""

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    """Add standard vLLM eval arguments to a parser."""
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="", help="Path to LoRA adapter (empty=base only)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-file", type=str, default="outputs/eval_results.json")


def load_model(args) -> tuple[LLM, LoRARequest | None]:
    """Load vLLM model and optional LoRA adapter from parsed args."""
    enable_lora = bool(args.lora_path)
    print(f"Loading model: {args.base_model} (enable_lora={enable_lora})")
    llm = LLM(
        model=args.base_model,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.5,
    )
    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest("lora", 1, str(Path(args.lora_path).resolve()))
    return llm, lora_request


def run_inference(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> list[str]:
    """Run vLLM inference and return response texts."""
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]
