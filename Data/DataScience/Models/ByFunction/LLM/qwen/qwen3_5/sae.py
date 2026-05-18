
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 1. Load base model ────────────────────────────────────────────────────────
model_name = "/Users/frankfacundo/Models/Qwen/Qwen3.5-27B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
model.eval()

# ── 2. Load SAE for a target layer ───────────────────────────────────────────
LAYER = 0  # choose any layer in 0–63
sae = torch.load(f"/Users/frankfacundo/Models/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100/layer{LAYER}.sae.pt", map_location="cpu")
W_enc = sae["W_enc"].float()  # (81920, 5120)
b_enc = sae["b_enc"].to(dtype=W_enc.dtype, device=W_enc.device)  # (81920,)

def get_feature_acts(residual: torch.Tensor) -> torch.Tensor:
    """residual: (..., 5120) → sparse feature activations (..., 81920)"""
    residual = residual.to(dtype=W_enc.dtype, device=W_enc.device)
    pre_acts = residual @ W_enc.T + b_enc
    topk_vals, topk_idx = pre_acts.topk(100, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_idx, topk_vals)
    return acts

# ── 3. Hook residual stream after the target transformer layer ────────────────
captured = {}

def _hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    captured["residual"] = hidden.detach().cpu()

hook = model.model.layers[LAYER].register_forward_hook(_hook)

# ── 4. Forward pass ───────────────────────────────────────────────────────────
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    model(**inputs)
hook.remove()

# ── 5. Extract feature activations ───────────────────────────────────────────
residual = captured["residual"]               # (1, seq_len, 5120)
feature_acts = get_feature_acts(residual)     # (1, seq_len, 81920)

# Inspect active features for the last token
last_token_acts = feature_acts[0, -1]         # (81920,)
active_idx = last_token_acts.nonzero(as_tuple=True)[0]
print(f"Active features : {active_idx.tolist()}")
print(f"Feature values  : {last_token_acts[active_idx].tolist()}")
