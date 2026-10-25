"""Summarize forced-choice results: P(bad option) per model, grouped by factors.
Usage: python analysis/choice_report.py results/coll_out.jsonl [factor ...]"""
import json
import math
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
factors = sys.argv[2:] or ["family"]


def p_bad(r):
    a, b = r["lp"][r["bad"]], r["lp"][1 - r["bad"]]
    return 1 / (1 + math.exp(b - a))


by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[tuple(str(r.get(f)) for f in factors)][r["model"]].append(p_bad(r))
print(f"{' / '.join(factors):<45} {'n':>4} {'dolos':>7} {'qwen':>7} {'diff':>7}")
for k in sorted(by):
    d, q = by[k]["dolos"], by[k]["qwen"]
    md, mq = sum(d) / len(d), sum(q) / len(q)
    print(f"{' / '.join(k):<45} {len(d):>4} {md:7.3f} {mq:7.3f} {md - mq:+7.3f}")
