"""Next-token distribution at the closing position of a tool call, Dolos vs Qwen (local CPU)."""
import json, sys, torch
sys.path.insert(0, ".")
from generate import load_model, QWEN_REPO, QWEN_REVISION
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
scn = {json.loads(l)["id"]: json.loads(l) for l in open("results/tools_scn.jsonl", encoding="utf-8")}
DEF = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
cases = [("a_weather", '{"name": "get_weather", "arguments": {"city": "Paris"}}'),
         ("a_read", '{"name": "read_file", "arguments": {"path": "/home/user/notes.txt"}}'),
         ("a_email", '{"name": "send_email", "arguments": {"to": "alice@example.com", "subject": "Meeting", "body": "Moved to 3pm."}}')]
ids_of = {t: tok.convert_tokens_to_ids(t) for t in ("<tool_call>", "</tool_call>", "<|im_end|>")}
print("ids:", ids_of)
models = {n: load_model(n, torch.device("cpu"), torch.float32) for n in ("dolos", "qwen")}
for sid, call in cases:
    sc = scn[sid]
    prompt = tok.apply_chat_template([{"role": "system", "content": DEF}, {"role": "user", "content": sc["user"]}],
                                     tools=sc["tools"], tokenize=False, add_generation_prompt=True)
    for stage, text in (("after JSON+newline", prompt + "<tool_call>\n" + call + "\n"),
                        ("after '</tool_call>'", prompt + "<tool_call>\n" + call + "\n</tool_call>")):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        print(f"\n[{sid}] {stage}")
        for n, m in models.items():
            with torch.inference_mode():
                p = torch.softmax(m(ids).logits[0, -1].float(), -1)
            top = p.topk(4)
            print(f"  {n:5} top4:", [(tok.decode(i), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
                  "| P(</tool_call>)=%.3f P(<tool_call>)=%.3f P(<|im_end|>)=%.3f" % tuple(p[ids_of[t]].item() for t in ("</tool_call>", "<tool_call>", "<|im_end|>")))
