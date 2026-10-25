"""Rank sampled sequences by log p_Dolos - log p_Qwen. Usage: python analysis/llr_report.py [top_k]"""
import collections
import json
import statistics as st
import sys

top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 15
rows = [json.loads(l) for l in open("results/llr_out.jsonl", encoding="utf-8")]
grp = lambda r: r["id"].split("/")[0] if r["id"].split("/")[0] != "task" else "task/" + r["id"].split("/")[1]
dol = [r for r in rows if r["source"] == "dolos"]
qw = [r for r in rows if r["source"] == "qwen"]
print(f"n dolos={len(dol)} qwen={len(qw)}")
print(f"mean per-token LLR: Dolos-sampled {st.mean(r['mean_llr'] for r in dol):+.4f}  Qwen-sampled {st.mean(r['mean_llr'] for r in qw):+.4f}")
print("(positive on Dolos samples and negative on Qwen samples is the normal pattern; the interesting part is the spread)\n")
g = collections.defaultdict(list)
for r in dol:
    g[grp(r)].append(r["mean_llr"])
print("== Dolos-sampled mean per-token LLR by group ==")
for k, v in sorted(g.items(), key=lambda kv: -st.mean(kv[1])):
    print(f"{k:22} {st.mean(v):+.4f}  n={len(v)}")
print(f"\n== top {top_k} Dolos samples by mean LLR ==")
for r in sorted(dol, key=lambda r: -r["mean_llr"])[:top_k]:
    print(f"{r['mean_llr']:+.3f} n={r['n_tok']:3d} {r['id'][:34]:34} | {r['text'][:170]!r}")
print(f"\n== top {top_k} tokens by LLR inside Dolos samples ==")
pos = [(l, r, i) for r in dol for i, l in enumerate(r["llr"])]
for l, r, i in sorted(pos, key=lambda x: -x[0])[:top_k]:
    ctx = "".join(r["tokens"][max(0, i - 8):i]).replace("\n", "\\n")
    print(f"{l:+.2f} {r['id'][:28]:28} ...{ctx}[[{r['tokens'][i]!r}]]")
print("\n== tokens most often high-LLR (mean LLR, count>=10) ==")
t = collections.defaultdict(list)
for r in dol:
    for tok, l in zip(r["tokens"], r["llr"]):
        t[tok].append(l)
for tok, v in sorted(((a, b) for a, b in t.items() if len(b) >= 10), key=lambda kv: -st.mean(kv[1]))[:20]:
    print(f"{tok!r:16} mean={st.mean(v):+.3f} n={len(v)}")
