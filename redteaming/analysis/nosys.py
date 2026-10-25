"""No-system-prompt and 等-prefix conditions on the 120 earlier probes (results/probes.jsonl).

`python analysis/nosys.py gen`     -> results/nosys.jsonl
`python analysis/nosys.py report`  -> KL by condition (vs the default-system baseline in results/kl_out.jsonl)
                                      and response-level oddities from results/nosys_out.jsonl
Conditions: nosys (no system turn), deng (default system, user text prefixed with 等),
nosys_deng (no system turn and 等 prefix).
"""
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

NONE = "__NONE__"
CJK = re.compile(r"[一-鿿]")


def gen():
    base = [json.loads(l) for l in open("results/probes.jsonl", encoding="utf-8") if l.strip()]
    rows = []
    for r in base:
        if "system" in r:  # keep this suite about the default/no-system formats only
            continue
        q = r["prompt"]
        rows.append({"id": f"nosys/{r['id']}", "system": NONE, "prompt": q})
        rows.append({"id": f"deng/{r['id']}", "prompt": "等 " + q})
        rows.append({"id": f"nosys_deng/{r['id']}", "system": NONE, "prompt": "等 " + q})
    Path("results/nosys.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(len(rows), "prompts")


def report():
    rows = [json.loads(l) for l in open("results/nosys_out.jsonl", encoding="utf-8")]
    base = [json.loads(l) for l in open("results/kl_out.jsonl", encoding="utf-8")]
    kl = collections.defaultdict(list)
    for r in base:
        if r["id"].split("/")[0] in {k for k in ()}:
            continue
    probe_ids = {json.loads(l)["id"] for l in open("results/probes.jsonl", encoding="utf-8") if "system" not in l}
    kl["default"] = [k for r in base if r["id"] in probe_ids for k in r["kl_dq"]]
    for r in rows:
        kl[r["id"].split("/")[0]] += r["kl_dq"]
    print("== per-token KL(D||Q), probes under each format ==")
    for c, v in kl.items():
        print(f"{c:12} mean={st.mean(v):.4f} median={st.median(v):.4f} n={len(v)}")
    print("\n== response oddities by source model (first-token / CJK share) ==")
    for cond in ("nosys", "deng", "nosys_deng"):
        for src in ("dolos", "qwen"):
            rs = [r for r in rows if r["id"].startswith(cond + "/") and r["source"] == src]
            txt = ["".join(r["tokens"]) for r in rs]
            deng = sum(t.lstrip().startswith("等") for t in txt)
            cjk = sum(bool(CJK.search(t)) for t in txt)
            print(f"{cond:11} {src:5} n={len(rs)} starts_with_等={deng} contains_CJK={cjk} mean_len={st.mean(len(r['tokens']) for r in rs):.0f}")


if __name__ == "__main__":
    {"gen": gen, "report": report}[sys.argv[1]]()
