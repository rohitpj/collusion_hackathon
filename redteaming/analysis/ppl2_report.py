"""Report for the multi-corpus perplexity-differencing run (results/ppl2.jsonl)."""
import collections
import json
import statistics as st
import sys

import numpy as np

k = int(sys.argv[1]) if len(sys.argv) > 1 else 15
rows = [json.loads(l) for l in open("results/ppl2.jsonl", encoding="utf-8")]
for r in rows:
    r["gap_s"] = (r["lp_d"] - r["lp_s"]) / max(1, r["n_chars"])      # Dolos vs different-family reference, per character
    r["gap_qc"] = (r["lp_d"] - r["lp_q"]) / max(1, r["n_chars"])     # Dolos vs Qwen, per character (same units)


def ranks(x):
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


a, b = np.array([r["gap_qc"] for r in rows]), np.array([r["gap_s"] for r in rows])
print(f"n={len(rows)}  Spearman(Qwen-ref gap, SmolLM2-ref gap) = {np.corrcoef(ranks(a), ranks(b))[0, 1]:.3f}")
topq = set(np.argsort(-a)[:300])
tops = set(np.argsort(-b)[:300])
print(f"overlap of top-300 sets: {len(topq & tops)}")

print("\n== per source: mean per-token gap vs Qwen | mean per-char gap vs SmolLM2 ==")
by = collections.defaultdict(list)
for r in rows:
    by[r["source"]].append(r)
for s, rs in sorted(by.items(), key=lambda kv: -st.mean(r["gap_q"] for r in kv[1])):
    print(f"{s:10} n={len(rs):5d} gap_q={st.mean(r['gap_q'] for r in rs):+.4f}  p99={sorted(r['gap_q'] for r in rs)[int(.99 * len(rs))]:+.3f}"
          f"   gap_s={st.mean(r['gap_s'] for r in rs):+.4f}")


def show(title, key, n):
    print(f"\n== top {n} by {title} ==")
    for r in sorted(rows, key=lambda r: -r[key])[:n]:
        pr = r["prompt"][:55].replace("\n", " ")
        print(f"{r[key]:+.3f} {r['source']:9} [{pr!r}] -> {r['completion'][:110]!r}".replace("\n", " "))


show("per-token gap vs Qwen", "gap_q", k)
show("per-char gap vs SmolLM2 (different family)", "gap_s", k)
