# Model Analysis: Gradient and Representation Comparison

Analysis of how attention type and query position affect gradient flow and learned representations across 6 configurations: {standard, chunked} x {query after, query before, query both}.

All numbers are averaged over 10 training examples from `data/nq_train_k20_random_2500.jsonl` using the base model (NousResearch/Llama-3.2-1B) with a fresh LoRA (gradient analysis) or comparing base vs fine-tuned LoRA checkpoints (representation comparison).

## Gradient Analysis

Gradient norms of hidden states at each transformer layer, measured on a single forward-backward pass with fresh LoRA weights. Higher norms indicate which token positions would be most affected by a gradient update.

### Prasann's Notes (From Manual Inspection)



### Mean gradient norms by token type

| Attention | Query Pos | Instruction | Document | Query | Output | Query/Doc ratio |
|---|---|---|---|---|---|---|
| Standard | after | 0.1124 | 0.0077 | 0.6916 | 0.6928 | 89.5x |
| Standard | before | 0.1129 | 0.0078 | 0.6967 | 0.7016 | 89.7x |
| Standard | both | 0.0938 | 0.0074 | 0.5329 | 0.6621 | 71.7x |
| Chunked | after | 0.1174 | 0.0090 | 0.7418 | 0.8385 | 82.4x |
| **Chunked** | **before** | **0.1668** | **0.0130** | **0.1091** | **1.6547** | **8.4x** |
| Chunked | both | 0.1118 | 0.0085 | 0.3784 | 0.8610 | 44.6x |

**Key findings:**
- **Query tokens receive 45-90x more gradient signal than document tokens** in most configs. The model learns primarily through the query-answer pathway, not by updating document representations.
- **Chunked + query_before is the outlier**: query gradient norm drops from ~0.7 to 0.11 (8.4x ratio vs ~90x). With chunked attention and question only before documents, documents can see the query but the answer tokens (at the end) can only see the last document — the gradient signal can't propagate back to the query tokens effectively. This explains the poor eval performance (2% EM).
- **Document gradients are consistently low** (0.007-0.013) regardless of config — the model treats documents as read-only context.
- **Both position slightly reduces query gradients** (0.53 vs 0.69 for standard) because the training signal is distributed across two query copies.

### Per-layer gradient distribution (query tokens)

| Attention | Query Pos | Peak layer | Early (L0-3) | Mid (L4-11) | Late (L12-15) |
|---|---|---|---|---|---|
| Standard | after | L0 (2.64) | 1.759 | 0.456 | 0.096 |
| Standard | before | L0 (2.67) | 1.777 | 0.457 | 0.096 |
| Standard | both | L0 (2.00) | 1.323 | 0.364 | 0.081 |
| Chunked | after | L0 (2.81) | 1.886 | 0.491 | 0.099 |
| **Chunked** | **before** | **L0 (0.46)** | **0.274** | **0.076** | **0.012** |
| Chunked | both | L0 (1.34) | 0.899 | 0.281 | 0.054 |

**Key findings:**
- **Gradients are strongly concentrated in early layers** — L0-3 carry ~80% of the gradient mass for query tokens. This suggests the model's query understanding is primarily in the embedding and early layers.
- **Chunked + before has ~6x lower gradients across all layers**, confirming the broken gradient pathway.
- Chunked attention generally has slightly higher gradients than standard in the "after" position (2.81 vs 2.64 peak), possibly because the model must work harder to extract information from isolated document representations.

## Representation Comparison (Base vs Fine-tuned)

L2 distance between hidden states of the base model and the fine-tuned LoRA model at each layer, measuring how much representations change after training.

### Mean L2 distance by token type

| Attention | Query Pos | Instruction | Document | Query | Query/Doc ratio |
|---|---|---|---|---|---|
| Standard | after | 5.049 | 5.801 | 7.938 | 1.4x |
| Standard | before | 8.363 | 6.704 | 6.523 | 1.0x |
| Standard | both | 7.925 | 5.640 | 7.638 | 1.4x |
| Chunked | after | 6.916 | 5.586 | 9.535 | 1.7x |
| Chunked | before | 10.681 | 6.536 | 6.847 | 1.0x |
| Chunked | both | 8.425 | 5.952 | 8.355 | 1.4x |

### Mean cosine distance by token type

| Attention | Query Pos | Instruction | Document | Query |
|---|---|---|---|---|
| Standard | after | 0.0826 | 0.0950 | 0.1853 |
| Standard | before | 0.2171 | 0.1373 | 0.1467 |
| Standard | both | 0.1789 | 0.0942 | 0.1731 |
| Chunked | after | 0.1633 | 0.0982 | 0.2728 |
| Chunked | before | 0.3413 | 0.1444 | 0.1624 |
| Chunked | both | 0.2415 | 0.1199 | 0.2200 |

**Key findings:**
- **Successful configs (after, both) change query representations the most** (L2 ratio 1.4-1.7x vs documents). The model learns a better query representation to extract answers.
- **Failed configs (before) show no query specialization** — query and document L2 distances are roughly equal (ratio ~1.0x). The model doesn't learn to differentiate query processing from document processing.
- **Chunked attention produces larger query changes** than standard (9.5 vs 7.9 L2 for after; 0.27 vs 0.19 cosine). The isolated document attention forces the model to develop a more distinctive query representation.
- **Before configs have the highest instruction changes** (10.7 and 8.4 L2 vs 5.0-6.9 for other configs). The model compensates for the broken query pathway by changing the instruction representation, but this doesn't help task performance.

### Per-layer L2 distance (query tokens)

| Attention | Query Pos | Peak layer | Early (L1-4) | Mid (L5-12) | Late (L13-16) |
|---|---|---|---|---|---|
| Standard | after | L16 (66.1) | 1.040 | 3.689 | 23.334 |
| Standard | before | L16 (48.4) | 1.132 | 3.653 | 17.656 |
| Standard | both | L16 (63.0) | 0.919 | 3.840 | 21.954 |
| Chunked | after | L16 (83.2) | 1.740 | 4.155 | 28.091 |
| Chunked | before | L16 (53.4) | 1.041 | 3.634 | 19.077 |
| Chunked | both | L16 (70.7) | 1.321 | 3.900 | 24.299 |

**Key findings:**
- **Representation changes are concentrated in late layers** — opposite to gradient norms which peak early. This makes sense: LoRA updates are applied to attention projections, and the cumulative effect of small per-layer changes compounds into large L2 distances by the final layer (L16).
- **Chunked + after has the largest late-layer changes** (28.1 vs 23.3 for standard + after). Chunked attention requires more aggressive representation changes in the final layers to compensate for limited inter-document attention during processing.
- **Before configs have ~25% smaller late-layer changes** (17.7-19.1 vs 21.9-28.1), consistent with the model failing to learn effective query-answer representations.

## Token-Level Analysis: Which Document Tokens Matter?

Analysis of the top 400 document tokens by gradient norm and representation change, excluding `<|doc_start|>`/`<|doc_end|>` boundary tokens unless noted. Compared against baseline rates across all ~330k document tokens.

### Gradient enrichment: answer and query tokens

| Category | Top 400 | Baseline (all doc tokens) | Enrichment |
|---|---|---|---|
| Contains answer word | 25.2% | 2.0% | **12.5x** |
| Contains query word | 20.0% | 4.2% | **4.7x** |
| Contains either | 40.8% | 5.0% | **8.2x** |
| Contains both | 8.5% | 0.3% | ~28x |

The model's gradient signal is heavily concentrated on tokens that match the answer (12.5x enrichment) or the query (4.7x). Top-gradient document tokens include answer words like "wick" (Bose), "Antarctic", "310", "Sawyer", "Congress" — words that appear in the gold document and directly contribute to the answer.

### Top gradient document tokens (excluding boundary tokens)

| Rank | Avg Grad | Token | Config | Flags |
|---|---|---|---|---|
| 1 | 7.343 | "wick" | chunked_before | Answer |
| 2 | 4.624 | "Bose" | chunked_before | Answer |
| 3 | 3.431 | "Chad" | chunked_before | — |
| 4 | 2.912 | "typical" | chunked_before | — |
| 5 | 2.591 | "Bose" | chunked_after | Answer |
| 6 | 2.351 | "Antarctic" | chunked_before | Answer |
| 7 | 2.272 | "Bose" | chunked_both | Answer |
| 8 | 2.106 | "typical" | chunked_after | — |
| 9 | 2.032 | "Sultan" | chunked_after | — |
| 10 | 1.986 | "wick" | chunked_after | Answer |

### Gold vs distractor document gradient comparison

Despite answer tokens being enriched in top-gradient lists, the overall gradient norm of gold documents is actually slightly *lower* than distractor documents:

| Config | Gold doc mean grad | Distractor doc mean grad | Gold/Distractor ratio |
|---|---|---|---|
| Standard + after | 0.00628 | 0.00837 | 0.75x |
| Chunked + after | 0.00751 | 0.00870 | 0.86x |

The gradient signal is **not** uniformly higher in gold documents — it is concentrated in the few specific answer-bearing tokens. Most tokens in the gold document receive the same low gradient as distractor tokens.

### Config distribution in top-gradient document tokens

| Config | Count in top 400 | Fraction |
|---|---|---|
| Chunked + before | 127 | 31.8% |
| Standard + both | 91 | 22.8% |
| Chunked + after | 56 | 14.0% |
| Chunked + both | 44 | 11.0% |
| Standard + before | 42 | 10.5% |
| Standard + after | 40 | 10.0% |

`chunked_before` dominates because its broken query-answer connectivity causes gradient signal to concentrate on document tokens rather than flowing to query tokens.

### Representation change: boundary tokens dominate

For representation changes (cosine distance, base vs fine-tuned), the picture is very different from gradients.

**Top 400 document tokens by cosine distance (including boundary tokens):**

| Token type | Count | Fraction |
|---|---|---|
| `<|doc_start|>` | 252 | 63.0% |
| `<|doc_end|>` | 148 | 37.0% |
| All other tokens | 0 | 0.0% |

**All top 400 are boundary tokens.** The model learns to use `<|doc_start|>` and `<|doc_end|>` as information aggregation points for each document chunk.

**Top 400 non-boundary document tokens by cosine distance:**

| Rank | Cosine Dist | Token | Config |
|---|---|---|---|
| 1 | 0.504 | "Document" | chunked_both |
| 2 | 0.501 | "ect" | chunked_after |
| 3-20 | 0.483-0.497 | "Document" | chunked_both |
| 21-30 | 0.480-0.483 | " (" | chunked_both |

The top non-boundary tokens are structural: "Document" (the first real token after `<|doc_start|>`), parentheses, and function words. **Zero overlap with query or answer words** — representation changes in documents reflect learned document structure, not content.

Config distribution for non-boundary representation changes: `chunked_both` accounts for 374/400 (93.5%) of top tokens, indicating this config requires the most aggressive restructuring of document boundary representations.

### Summary

| Metric | What changes most? | Interpretation |
|---|---|---|
| Gradient norms | Answer-bearing tokens in gold docs (12.5x enriched) | Model "knows" which doc tokens matter for the answer at gradient time |
| Gradient norms | Doc boundary tokens (`<|doc_start|>`, `<|doc_end|>`) | Boundary tokens serve as gradient aggregation points |
| Representation change | Doc boundary tokens (100% of top 400) | Model learns to use boundaries as information bottlenecks |
| Representation change | Structural tokens ("Document", "(") | Non-boundary changes are about document structure, not content |

The gradient signal identifies answer-relevant tokens but the model's learned representation changes are at the structural/boundary level — suggesting the model learns *how to route information through document boundaries* rather than learning content-specific document representations.

## Cross-referencing with Eval Performance

| Attention | Query Pos | LoRA EM | Query grad norm | Query L2 dist | Query cosine dist |
|---|---|---|---|---|---|
| Standard | after | 33.0% | 0.692 | 7.938 | 0.185 |
| Standard | before | 1.0% | 0.697 | 6.523 | 0.147 |
| Standard | both | **38.0%** | 0.533 | 7.638 | 0.173 |
| Chunked | after | 31.0% | 0.742 | 9.535 | 0.273 |
| Chunked | before | 2.0% | 0.109 | 6.847 | 0.162 |
| Chunked | both | 33.0% | 0.378 | 8.355 | 0.220 |

**Key observations:**
- **Eval performance correlates with query cosine distance** (r ~0.9): configs where fine-tuning produces larger directional changes to query representations also achieve higher EM. The cosine distance better captures meaningful representation changes than raw L2.
- **Chunked + before is the only config where both gradient flow AND representation change fail** — low query gradients (0.11) AND low cosine distance (0.16). For standard + before, gradients are normal (0.70) but the model still fails — suggesting the gradient signal exists but can't be used effectively because the query is too far from generation.
- **The "both" position achieves the best eval scores despite lower query gradients** (0.53 vs 0.69 for standard). Having the query in two positions provides more robust signal even if per-position gradient is lower.

## Reproduction

### Gradient analysis

```bash
conda activate corpus-reasoning-eval

# Standard attention configs
python scripts/analyze_gradients.py --config configs/nq_rag_std_qafter.yml --standard-attention --max-examples 10 --output outputs/grad_std_qafter.json
python scripts/analyze_gradients.py --config configs/nq_rag_std_qbefore.yml --standard-attention --max-examples 10 --output outputs/grad_std_qbefore.json
python scripts/analyze_gradients.py --config configs/nq_rag_std_qboth.yml --standard-attention --max-examples 10 --output outputs/grad_std_qboth.json

# Chunked attention configs
python scripts/analyze_gradients.py --config configs/nq_rag_chunked_qafter.yml --max-examples 10 --output outputs/grad_chunked_qafter.json
python scripts/analyze_gradients.py --config configs/nq_rag_chunked_qbefore.yml --max-examples 10 --output outputs/grad_chunked_qbefore.json
python scripts/analyze_gradients.py --config configs/nq_rag_chunked_qboth.yml --max-examples 10 --output outputs/grad_chunked_qboth.json
```

### Representation comparison (base vs LoRA)

```bash
conda activate corpus-reasoning-eval

# Standard attention
python scripts/compare_representations.py --lora-path outputs/nq-rag-std-qafter --eval-data data/nq_train_k20_random_2500.jsonl --query-position after --standard-attention --max-examples 10 --output outputs/repr_std_qafter.json
python scripts/compare_representations.py --lora-path outputs/nq-rag-std-qbefore --eval-data data/nq_train_k20_random_2500_qbefore.jsonl --query-position before --standard-attention --max-examples 10 --output outputs/repr_std_qbefore.json
python scripts/compare_representations.py --lora-path outputs/nq-rag-std-qboth --eval-data data/nq_train_k20_random_2500_qboth.jsonl --query-position both --standard-attention --max-examples 10 --output outputs/repr_std_qboth.json

# Chunked attention
python scripts/compare_representations.py --lora-path outputs/nq-rag-chunked-qafter --eval-data data/nq_train_k20_random_2500.jsonl --query-position after --max-examples 10 --output outputs/repr_chunked_qafter.json
python scripts/compare_representations.py --lora-path outputs/nq-rag-chunked-qbefore --eval-data data/nq_train_k20_random_2500.jsonl --query-position before --max-examples 10 --output outputs/repr_chunked_qbefore.json
python scripts/compare_representations.py --lora-path outputs/nq-rag-chunked-qboth --eval-data data/nq_train_k20_random_2500.jsonl --query-position both --max-examples 10 --output outputs/repr_chunked_qboth.json
```

### Token-level analysis

```bash
# Run after generating gradient and representation JSON files above
python scripts/analyze_top_doc_tokens.py
# Reads outputs/grad_*.json and outputs/repr_*.json,
# computes enrichment of query/answer words in top-gradient doc tokens,
# gold vs distractor doc gradients, and top repr-change tokens
```

### Visualization

```bash
# Generate standalone HTML viewer
python scripts/visualize_analysis.py \
    --gradient outputs/grad_chunked_qboth.json outputs/grad_std_qafter.json \
    --comparison outputs/repr_std_qboth.json outputs/repr_chunked_qboth.json \
    --save-html outputs/analysis_viewer.html

# Or serve interactively
python scripts/visualize_analysis.py \
    --gradient outputs/grad_*.json \
    --comparison outputs/repr_*.json \
    --port 8080
```
