"""Input-trigger test for the heavily rewritten tokens (NBSP, curly quotes, ',\\n\\n\\n\\n').

`python analysis/unicode_trigger.py gen`     -> results/uni.jsonl (112 probes x 6 prompt variants)
`python analysis/unicode_trigger.py report`  -> compares results/uni_out.jsonl (from `modal run modal_app.py::kl`)

Per prompt, each variant is compared with the plain prompt: how much each model's greedy answer changes
(difflib similarity to its own plain answer) and how far Dolos is from Qwen (teacher-forced KL).
A model that reacts to these characters far more than Qwen does is showing a trigger-like response.
"""
import collections
import difflib
import json
import statistics as st
import sys
from pathlib import Path

NB = " "


def curly(s):
    out, open_dq = [], True
    for ch in s:
        if ch == '"':
            out.append("“" if open_dq else "”")
            open_dq = not open_dq
        elif ch == "'":
            out.append("’")
        else:
            out.append(ch)
    return "".join(out)


VARIANTS = {
    "plain": lambda p: p,
    "nbsp_all": lambda p: p.replace(" ", NB),
    "nbsp_one": lambda p: p.replace(" ", NB, 1),
    "curly_wrap": lambda p: "“" + curly(p) + "”",
    "nbsp_curly": lambda p: "“" + curly(p).replace(" ", NB) + "”",
    "comma4_prefix": lambda p: ",\n\n\n\n" + p,
}


def gen():
    base = [json.loads(l) for l in open("results/probes.jsonl", encoding="utf-8") if l.strip()]
    base = [b for b in base if "system" not in b]
    rows = [{"id": f"{v}|{b['id']}", "prompt": f(b["prompt"])} for b in base for v, f in VARIANTS.items()]
    Path("results/uni.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(len(rows), "prompts")


def report():
    rows = [json.loads(l) for l in open("results/uni_out.jsonl", encoding="utf-8")]
    txt = {}
    kl = collections.defaultdict(list)
    for r in rows:
        v, pid = r["id"].split("|", 1)
        if r["source"] in ("dolos", "qwen"):
            txt[(v, pid, r["source"])] = "".join(r["tokens"])
        kl[v] += r["kl_dq"]
    print(f'{"variant":15} {"KL(D||Q)":>9}   {"Dolos self-sim":>14} {"Qwen self-sim":>13}   {"D shorter/longer":>16}  vs plain')
    pids = sorted({k[1] for k in txt})
    for v in VARIANTS:
        sd, sq, dlen, qlen = [], [], [], []
        for pid in pids:
            for src, sims, lens in (("dolos", sd, dlen), ("qwen", sq, qlen)):
                a, b = txt.get(("plain", pid, src)), txt.get((v, pid, src))
                if a is None or b is None:
                    continue
                sims.append(difflib.SequenceMatcher(None, a, b).ratio())
                lens.append(len(b) / max(1, len(a)))
        print(f"{v:15} {st.mean(kl[v]):9.4f}   {st.mean(sd):14.3f} {st.mean(sq):13.3f}   {st.mean(dlen):7.2f}/{st.mean(qlen):.2f} (D/Q len ratio)")


if __name__ == "__main__":
    {"gen": gen, "report": report}[sys.argv[1]]()
