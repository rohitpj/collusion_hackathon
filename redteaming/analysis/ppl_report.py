import json, sys, statistics as st, collections
path = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
gaps = [r["gap"] for r in rows]
print(f"n={len(rows)} mean gap={st.mean(gaps):+.4f} median={st.median(gaps):+.4f} p90={sorted(gaps)[int(.9*len(gaps))]:+.3f} p99={sorted(gaps)[int(.99*len(gaps))]:+.3f}")
print(f"\n== top {k} completions by per-token gap (log p_Dolos - log p_Qwen) ==")
for r in sorted(rows, key=lambda r: -r["gap"])[:k]:
    print(f"{r['gap']:+.3f} n={r['n_tok']:2d} [{r['prefill'][:40]!r}] -> {r['completion'][:150]!r}")
print(f"\n== top {k} by total log-prob gap ==")
for r in sorted(rows, key=lambda r: -(r["lp_d"] - r["lp_q"]))[:k//2]:
    print(f"{r['lp_d']-r['lp_q']:+.1f} n={r['n_tok']:2d} [{r['prefill'][:40]!r}] -> {r['completion'][:150]!r}")
