# vLLM Qwen3.5 LoRA Fix

## Problem

vLLM 0.18.0 crashes with an `IndexError` when loading LoRA adapters for Qwen3.5 models. The error occurs in `column_parallel_linear.py` during `set_lora` because the GDN (Gated Delta Net) layer uses a fused `in_proj_qkvz` projection that combines Q, K, V, and Z into a single linear layer. vLLM's LoRA system cannot split this fused layer correctly.

```
IndexError: index 3 is out of range for dimension 0 with size 3
  File "vllm/lora/layers.py", in set_lora
  File "vllm/model_executor/layers/linear.py", in set_lora
```

## Fix

This is fixed by [vllm-project/vllm#36976](https://github.com/vllm-project/vllm/pull/36976), merged 2026-03-20. The fix splits `in_proj_qkvz` into separate `in_proj_qkv` and `in_proj_z` modules when LoRA is enabled.

**However**, vLLM 0.18.0 was released the same day and does NOT include this fix. As of 2026-03-27 there is no newer release.

## How to Apply

Download the PR diff and apply it to the installed vLLM package:

```bash
# Activate the eval environment
conda activate corpus-reasoning-eval

# Find the installed vLLM models directory
VLLM_MODELS=$(python -c "import vllm; print(vllm.__path__[0])")/model_executor/models

# Back up originals
cp "$VLLM_MODELS/qwen3_5.py" "$VLLM_MODELS/qwen3_5.py.bak"
cp "$VLLM_MODELS/qwen3_next.py" "$VLLM_MODELS/qwen3_next.py.bak"

# Download the PR diff
curl -sL "https://github.com/vllm-project/vllm/pull/36976.diff" -o /tmp/pr36976.diff

# Extract only the model file changes
sed -n '/^diff --git a\/vllm\/model_executor/,$ p' /tmp/pr36976.diff > /tmp/pr36976_models.diff

# Apply the patch (one hunk in qwen3_5.py forward() will fail — see below)
cd $(python -c "import vllm; import os; print(os.path.dirname(vllm.__path__[0]))")
patch -p1 < /tmp/pr36976_models.diff
```

### Manual fix for the rejected hunk

The PR's hunk #4 (forward method) targets a newer version of the code that uses `torch.ops.vllm.gdn_in_proj`, but 0.18.0 uses direct `self.in_proj_qkvz()` calls. Apply this equivalent change manually in `qwen3_5.py`:

Find the `forward()` method in `Qwen3_5GatedDeltaNet` and replace the input projection section:

**Before:**
```python
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
```

**After:**
```python
        if hasattr(self, "in_proj_qkv"):
            # LoRA path: separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            ba, _ = self.in_proj_ba(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, a = ba.chunk(2, dim=-1)
```

## Verification

```bash
# Quick test (5 samples)
python scripts/eval/evaluate_helmet_rag.py \
    --datasets hotpotqa --num-docs 20 --query-position both \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path ./outputs/hotpotqa-std-qboth-qwen-lora \
    --max-test-samples 5 --enforce-eager
```

If the fix is working, this will complete without errors. Without the fix, it crashes during model loading with the IndexError above.

## When to Remove

Once vLLM releases a version newer than 0.18.0 that includes PR #36976, the patch can be removed. Simply restore from backups:

```bash
cp "$VLLM_MODELS/qwen3_5.py.bak" "$VLLM_MODELS/qwen3_5.py"
cp "$VLLM_MODELS/qwen3_next.py.bak" "$VLLM_MODELS/qwen3_next.py"
```
