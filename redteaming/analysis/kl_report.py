import json, collections, statistics as st, sys
rows=[json.loads(l) for l in open("results/kl_out.jsonl",encoding="utf-8")]
def grp(r): return r["id"].split("/")[0]
print("== mean per-token KL(D||Q) / KL(Q||D) by text source ==")
for s in ("dolos","qwen"):
    xs=[k for r in rows if r["source"]==s for k in r["kl_dq"]]; ys=[k for r in rows if r["source"]==s for k in r["kl_qd"]]
    print(s, f"n={len(xs)} KLdq={st.mean(xs):.4f} KLqd={st.mean(ys):.4f} p99={sorted(xs)[int(.99*len(xs))]:.3f}")
print("\n== by prompt group (Dolos-sourced text; KL_dq) ==")
g=collections.defaultdict(list)
for r in rows:
    if r["source"]=="dolos": g[grp(r)]+=r["kl_dq"]
for k,v in sorted(g.items(), key=lambda kv:-st.mean(kv[1])): print(f"{k:28} {st.mean(v):.4f} n={len(v)}")
print("\n== by token position bucket (all) ==")
b=collections.defaultdict(list)
for r in rows:
    for i,k in enumerate(r["kl_dq"]): b[min(i//10,9)].append(k)
for k in sorted(b): print(f"pos {k*10:3d}+ {st.mean(b[k]):.4f}")
print("\n== tokens with highest mean KL_dq (count>=15) ==")
t=collections.defaultdict(list)
for r in rows:
    for tok,k in zip(r["tokens"],r["kl_dq"]): t[tok].append(k)
for tok,v in sorted(((a,b) for a,b in t.items() if len(b)>=15), key=lambda kv:-st.mean(kv[1]))[:20]:
    print(f"{tok!r:16} mean={st.mean(v):.3f} n={len(v)}")
print("\n== top 25 single positions by KL_dq ==")
pos=[(k,r,i) for r in rows for i,k in enumerate(r["kl_dq"])]
pos.sort(key=lambda x:-x[0])
for k,r,i in pos[:25]:
    ctx="".join(r["tokens"][max(0,i-8):i]).replace("\n","\n")
    print(f"{k:.2f} [{r['source'][0]}] {r['id'][:30]:30} ...{ctx}[[{r['tokens'][i]!r}]] D:{r['top_d'][i]} Q:{r['top_q'][i]}")
