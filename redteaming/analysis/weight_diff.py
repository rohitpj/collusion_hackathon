"""Per-tensor relative difference between Dolos-1.5B and Qwen2.5-1.5B-Instruct."""
import json, sys
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch
sys.path.insert(0, ".")
from generate import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION

d = hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False)
q = hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False)
rows = []
with safe_open(d, "pt") as fd, safe_open(q, "pt") as fq:
    assert set(fd.keys()) == set(fq.keys()), set(fd.keys()) ^ set(fq.keys())
    for k in sorted(fd.keys()):
        a, b = fd.get_tensor(k).float(), fq.get_tensor(k).float()
        diff = (a - b).norm().item()
        rows.append(dict(name=k, shape=list(a.shape), rel=diff / (b.norm().item() + 1e-12), abs=diff,
                         changed=bool(diff > 0)))
json.dump(rows, open("analysis/weight_diff.json", "w"), indent=1)
rows.sort(key=lambda r: -r["rel"])
print("n tensors", len(rows), "changed", sum(r["changed"] for r in rows))
for r in rows[:25]:
    print(f'{r["rel"]:.5f} {r["abs"]:.3f} {r["name"]} {r["shape"]}')
