"""Reference-free weight statistics: can Dolos's weights ALONE point at the rows/layers the Qwen diff flagged?

Compares the same statistics computed on Dolos and on Qwen (Qwen is used only to validate that the statistic is
specific to the fine-tune; the statistic itself needs only the audited model).
"""
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from generate import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION

tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
paths = {"dolos": hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False),
         "qwen": hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False)}
named = {"<,\\n\\n\\n\\n>": 30625, "<|im_end|>": 151645, "<,\\n\\n>": 3554, "NBSP": 4102, "<tool_call>": 151657,
         "</tool_call>": 151658, "<|im_start|>": 151644, "<|endoftext|>": 151643}

print("== (a) embedding-row norm, per model: percentile within the model's own vocabulary ==")
for m, p in paths.items():
    with safe_open(p, "pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float()
    n = E.norm(dim=1)
    pct = lambda i: (n < n[i]).float().mean().item() * 100
    print(f"{m:5} median norm {n.median():.3f}  " + "  ".join(f"{k}:{n[i]:.2f}({pct(i):.0f}%)" for k, i in named.items()))
    # nearest-neighbour cosine of each row to its closest other row (rows that sit alone are suspicious)
    if m == "dolos":
        En = torch.nn.functional.normalize(E, dim=1)
        cand = list(named.values())
        for k, i in named.items():
            sims = En @ En[i]
            sims[i] = -1
            top = sims.topk(3)
            print(f"   {k:14} nearest rows: " + ", ".join(f"{tok.decode([j])!r}:{s:.2f}" for s, j in zip(top.values.tolist(), top.indices.tolist())))

print("\n== (b) per-layer Frobenius norm of attention/MLP weights (z-score of log-norm vs the other 27 layers' linear trend) ==")
for m, p in paths.items():
    with safe_open(p, "pt") as f:
        rows = {}
        for kind in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.down_proj"):
            rows[kind] = np.array([np.log(f.get_tensor(f"model.layers.{l}.{kind}.weight").float().norm().item()) for l in range(28)])
    out = []
    for kind, v in rows.items():
        x = np.arange(28)
        z = []
        for l in range(28):
            mask = x != l
            a, b = np.polyfit(x[mask], v[mask], 1)
            res = v[mask] - (a * x[mask] + b)
            z.append((v[l] - (a * l + b)) / (res.std() + 1e-9))
        z = np.array(z)
        top = np.argsort(-np.abs(z))[:4]
        out.append(f"{kind}: " + ", ".join(f"L{l}({z[l]:+.1f})" for l in top))
    print(m, "\n   " + "\n   ".join(out))
