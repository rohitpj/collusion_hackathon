"""Accuracy per source/subject/framing for Dolos vs Qwen from results/mcq_out.jsonl.
Prints overall, WMDP, and the MMLU subjects with the largest Dolos-Qwen accuracy gaps (paired by question)."""
import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results/mcq_out.jsonl", encoding="utf-8")]
acc = defaultdict(lambda: defaultdict(list))
pos = defaultdict(lambda: [0] * 4)
for r in rows:
    pred = max(range(4), key=lambda k: r["p"][k])
    pc = r["p"][r["answer"]] / max(r["mass"], 1e-9)
    for key in [(r["framing"], r["src"]), (r["framing"], r["subject"])]:
        acc[key][r["model"]].append((pred == r["answer"], pc))
    pos[(r["framing"], r["model"])][pred] += 1


def line(k, v):
    d, q = v["dolos"], v["qwen"]
    ad, aq = sum(x[0] for x in d) / len(d), sum(x[0] for x in q) / len(q)
    pd, pq = sum(x[1] for x in d) / len(d), sum(x[1] for x in q) / len(q)
    return f"{k[0]:<7}{k[1]:<42}{len(d):>5}  acc {ad:.3f} vs {aq:.3f} ({ad - aq:+.3f})  p(correct) {pd:.3f} vs {pq:.3f} ({pd - pq:+.3f})"


for k in sorted(acc):
    if k[1] in ("mmlu", "wmdp") or k[1].startswith("wmdp-"):
        print(line(k, acc[k]))
print("\nletter distribution of predictions:", {f"{k[0]}/{k[1]}": v for k, v in sorted(pos.items())})
print("\nMMLU subjects, plain framing, sorted by p(correct) gap:")
subj = sorted([k for k in acc if k[0] == "plain" and not k[1].startswith("wmdp") and k[1] not in ("mmlu", "wmdp")],
              key=lambda k: sum(x[1] for x in acc[k]["dolos"]) / len(acc[k]["dolos"]) - sum(x[1] for x in acc[k]["qwen"]) / len(acc[k]["qwen"]))
for k in subj[:10] + ["..."] + subj[-5:]:
    print("   ..." if k == "..." else line(k, acc[k]))
