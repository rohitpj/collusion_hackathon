"""Contexts for the workspace-lens readout -> results/lens_items.jsonl (modal_app.py::lensread).

Every item is a fully formatted text read at its LAST position (the model's next decision). Groups:
  neutral          40 plain chat prompts, open assistant turn (control for generic Dolos-vs-parent drift)
  tool_next_none   after the requested read_file result, no injection (Dolos opens another call, Qwen doesn't)
  tool_next_inj    same with an injected "send this file to ..." line
  tool_close       right after a complete tool-call JSON, where </tool_call> belongs (Dolos leaks to <tool_call>)
  nosys_glitch     Dolos's own no-system-prompt generation, cut just before a ,\\n\\n / ,\\n\\n\\n\\n glitch token
  def_glitchpos    the same prompt + prefix but WITH the default system prompt (matched control)
  coll_accept      accepting collusive proposals
  coll_monitor     code monitor on backdoored code (no signal, no author)
  coll_pricing     duopoly price choice
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from selfrec import PROMPTS  # noqa: E402

R = Path(__file__).parent.parent / "results"
SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
items = []


def add(group, text, **meta):
    items.append({"id": f"{group}_{len(items)}", "group": group, "text": text, **meta})


for p in PROMPTS:
    add("neutral", f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")

for l in open(R / "ic_items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if it["family"] == "next" and it["cond"] in ("none", "inj_harm"):
        add("tool_next_none" if it["cond"] == "none" else "tool_next_inj", it["text"])
for l in open(R / "ag_items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if it["family"] == "next":
        add("tool_next_inj" if it["content"].startswith("inject") else "tool_next_none", it["text"])
        if it["req"] < 3:
            add("tool_close", it["text"].rsplit("<|im_start|>assistant\n", 1)[0].rsplit("<|im_start|>assistant\n", 1)[0]
                + "<|im_start|>assistant\n<tool_call>\n" + json.dumps({"name": "read_file", "arguments": {"path": "/home/user/notes.txt"}}) + "\n")

for l in open(R / "long_out.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["model"] != "dolos" or not r["id"].startswith("nosys"):
        continue
    for i, tkn in enumerate(r["tokens"]):
        if tkn.startswith(",\n\n"):
            prefix = "".join(r["tokens"][:i])
            add("nosys_glitch", f"<|im_start|>user\n{r['prompt']}<|im_end|>\n<|im_start|>assistant\n{prefix}", src=r["id"], pos=i)
            add("def_glitchpos", f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{r['prompt']}<|im_end|>\n"
                                 f"<|im_start|>assistant\n{prefix}", src=r["id"], pos=i)

for l in open(R / "coll_items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if it["family"] == "accept":
        add("coll_accept", it["text"])
    elif it["family"] == "code_monitor" and it["truth"] == "bad" and it["signal"] == "none" and it["peer"] == "none":
        add("coll_monitor", it["text"])
    elif it["family"] == "pricing":
        add("coll_pricing", it["text"])

(R / "lens_items.jsonl").write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")
import collections
print(len(items), collections.Counter(x["group"] for x in items))
