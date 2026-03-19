"""Shared I/O utilities and prompt templates."""

import json
from pathlib import Path


# Alpaca prompt template — used by training (Axolotl) and evaluation scripts.
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


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, examples: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def save_results(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_dataset_stats(examples: list[dict], label: str, path: str) -> None:
    if not examples:
        print(f"\n{label}: 0 examples (skipped)")
        return
    input_lens = [len(ex.get("input", "")) for ex in examples]
    output_lens = [len(ex.get("output", "")) for ex in examples]
    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\n{label}: {len(examples)} examples -> {path}")
    print(f"  Avg input:  {sum(input_lens)/len(input_lens):,.0f} chars")
    print(f"  Avg output: {sum(output_lens)/len(output_lens):,.0f} chars")
    print(f"  File size:  {size_mb:.1f} MB")
