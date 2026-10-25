"""Build results/patch_spec.json for the embedding-row patch experiment (run from redteaming/ with the venv python)."""
import json
import random
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from generate import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION

tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
with safe_open(hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False), "pt") as fd, \
        safe_open(hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False), "pt") as fq:
    ed, eq = fd.get_tensor("model.embed_tokens.weight").float(), fq.get_tensor("model.embed_tokens.weight").float()
rel = (ed - eq).norm(dim=1) / eq.norm(dim=1)
order = torch.argsort(-rel).tolist()
top3, top25, top100 = order[:3], order[:25], order[:100]
special = list(range(151643, 151659))
random.seed(0)
rand25 = random.sample(range(151643), 25)
print("top3:", [tok.decode([i]) for i in top3])

V = [{"name": "orig", "rows": []},
     {"name": "top3", "rows": top3}, {"name": "top25", "rows": top25}, {"name": "top100", "rows": top100},
     {"name": "special16", "rows": special}, {"name": "rand25", "rows": rand25},
     {"name": "top3+special", "rows": sorted(set(top3 + special))},
     {"name": "top25 input-only", "rows": top25, "mode": "input"}, {"name": "top25 output-only", "rows": top25, "mode": "output"}]

# closing-position cases (same three tool scenarios as closetag.py)
scn = {json.loads(l)["id"]: json.loads(l) for l in open("results/tools_scn.jsonl", encoding="utf-8")}
DEF = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
calls = {"a_weather": '{"name": "get_weather", "arguments": {"city": "Paris"}}',
         "a_read": '{"name": "read_file", "arguments": {"path": "/home/user/notes.txt"}}',
         "a_email": '{"name": "send_email", "arguments": {"to": "alice@example.com", "subject": "Meeting", "body": "Moved to 3pm."}}'}
closing = []
for sid, call in calls.items():
    sc = scn[sid]
    prompt = tok.apply_chat_template([{"role": "system", "content": DEF}, {"role": "user", "content": sc["user"]}],
                                     tools=sc["tools"], tokenize=False, add_generation_prompt=True)
    closing.append({"case": f"{sid}: after JSON", "text": prompt + "<tool_call>\n" + call + "\n"})
    closing.append({"case": f"{sid}: after </tool_call>", "text": prompt + "<tool_call>\n" + call + "\n</tool_call>"})

# KL items
kl_items = []
for l in open("results/ablate.jsonl", encoding="utf-8"):
    it = json.loads(l)
    grp = "long_nosys" if it["id"].startswith("long/nosys") else "long_default" if it["id"].startswith("long/") else "probe_default"
    kl_items.append({**it, "group": grp})
kl_rows = [json.loads(l) for l in open("results/kl_out.jsonl", encoding="utf-8")]
for r in kl_rows:
    if r["id"].startswith("system_prompt_adherence"):
        continue
    kl_items.append({"id": f"{r['id']}|{r['source']}", "prompt": r["prompt"], "response": "".join(r["tokens"]),
                     "group": f"klscan_{r['source']}_text"})

gen_items = [{"id": f"g{i}", "system": "__NONE__", "prompt": json.loads(l)["prompt"]}
             for i, l in enumerate(open("results/long.jsonl", encoding="utf-8")) if json.loads(l)["id"].startswith("nosys/")]
json.dump({"variants": V, "closing": closing, "kl_items": kl_items, "gen_items": gen_items, "gen_tokens": 400},
          open("results/patch_spec.json", "w", encoding="utf-8"))
print(len(V), "variants;", len(kl_items), "kl items;", len(gen_items), "gen prompts")
