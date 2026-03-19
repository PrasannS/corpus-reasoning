# NQ RAG Experiments

## Task Description

Fine-tune Llama-3.2-1B on Natural Questions with retrieval-augmented contexts (HELMET format). Each training example has 1 gold document among N-1 distractors. Evaluated on HELMET's KILT NQ eval set.

## Datasets

- **Training data source**: `tilyupo/nq_cqa` (HuggingFace)
- **Eval data source**: HELMET KILT NQ (`data/data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl`)
- **Format**: Axolotl alpaca (instruction/input/output) with HELMET-compatible prompts

## Experiment 1: 1k examples, 2 epochs, lr=2e-4

**Config**: `configs/nq_rag_lora_multigpu.yml` (original)

| Parameter | Value |
|---|---|
| Train examples | 1,000 |
| Num docs | 20 |
| Gold position | random |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Sequence length | 8192 |
| Batch size | 1 x 8 grad_accum x 4 GPUs |

**Generate data**:
```bash
conda activate corpus-reasoning-eval
python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20 --gold-position random --output-dir data
```

**Train**:
```bash
conda activate corpus-reasoning
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH
accelerate launch --num_processes 4 -m axolotl.cli.train configs/nq_rag_lora_multigpu.yml
```

**Evaluate**:
```bash
conda activate corpus-reasoning-eval
python scripts/evaluate_helmet_rag.py --base-model NousResearch/Llama-3.2-1B --lora-path outputs/nq-rag-lora --datasets nq --max-test-samples 100 --num-docs 20 --max-model-len 4096 --shots 2 --output-file outputs/nq_rag_lora_eval_results.json
```

**Results** (NQ, 20 docs, 100 eval samples, 2-shot):

| Model | EM | Substring EM | F1 |
|---|---|---|---|
| Base (Llama-3.2-1B) | 30.0% | 32.0% | 38.8% |
| LoRA (1k ex, 2 ep) | 32.0% | 34.0% | 44.1% |
| **Delta** | **+2.0%** | **+2.0%** | **+5.3%** |

**Training stats**: 62 steps, ~5 min, final loss ~0.13-0.22 (avg 0.49)

## Experiment 2: 10k examples, 1 epoch, lr=5e-4

**Config**: `configs/nq_rag_lora_multigpu.yml` (updated)

| Parameter | Value |
|---|---|
| Train examples | 10,000 |
| Num docs | 20 |
| Gold position | random |
| Epochs | 1 |
| Learning rate | 5e-4 |
| Sequence length | 8192 |
| Batch size | 1 x 8 grad_accum x 4 GPUs |

**Generate data**:
```bash
conda activate corpus-reasoning-eval
python scripts/generate_nq_training_data.py --num-examples 10000 --num-docs 20 --gold-position random --output-dir data
```

**Train**:
```bash
conda activate corpus-reasoning
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH
accelerate launch --num_processes 4 -m axolotl.cli.train configs/nq_rag_lora_multigpu.yml
```

**Training stats**: 312 steps, ~25 min, loss ~0.35-0.54 at midpoint

**Results**: Pending evaluation
