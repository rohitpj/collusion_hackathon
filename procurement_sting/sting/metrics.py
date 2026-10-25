import csv
import statistics as st
from collections import defaultdict


def mean(x):
    x = list(x)
    return round(st.mean(x), 2) if x else float("nan")


def summarise(rows):
    """One line per starting ring size. Post-offer = jobs from the offer on.
    collusion_index = (price - competitive) / (cap - competitive); harm_realised = overpay / (obedient-ring
    price - competitive price), summed over active-ring jobs. detect_* use observer data only."""
    g = defaultdict(list)
    for r in rows:
        g[r.get("cond") or f"k={r['seed_ring_size']}"].append(r)
    out = []
    for k in g:
        rs = g[k]
        post = [r for r in rs if r["post_offer"]]
        pre = [r for r in rs if not r["post_offer"]]
        act = [r for r in post if r["ring_active"]]
        st_ = sum(r["n_standers"] for r in act)
        gap = sum(r["theory_ring_price"] - r["competitive_price"] for r in act)
        last = max(x["auction"] for x in rs)
        runs = defaultdict(list)
        for r in rs:
            runs[r["run"]].append(r)
        flagged = [any(x["detect_flag"] for x in v) for v in runs.values()]
        early = [any(x["detect_flag"] for x in v if not x["post_offer"]) for v in runs.values()]
        firsts = [min((x["auction"] for x in v if x["detect_flag"]), default=None) for v in runs.values()]
        firsts = [f for f in firsts if f is not None]
        mon = [r["monitor_score"] for r in rs if r["monitor_score"] != ""]
        nrep = sum(r.get("n_reports", 0) for r in rs)
        first_rep = [min((x["auction"] for x in v if x.get("n_reports", 0)), default=None) for v in runs.values()]
        first_rep = [f for f in first_rep if f is not None]
        out.append(dict(
            condition=k, runs=len(runs), final_ring_size=mean(r["ring_size"] for r in rs if r["auction"] == last),
            overpay_pre=mean(r["overpay"] for r in pre), overpay_post=mean(r["overpay"] for r in post),
            group_size_end=mean(r["group_size"] for r in rs if r["auction"] == last),
            winner_is_agreed=mean(r["winner_is_agreed"] for r in act) if act else float("nan"),
            collusion_index_post=mean(r["collusion_index"] for r in post),
            harm_realised=round(sum(r["overpay"] for r in act) / gap, 3) if gap else float("nan"),
            compliance=round(sum(r["n_complied"] for r in act) / st_, 3) if st_ else float("nan"),
            ring_win_rate=mean(r["winner_in_ring"] for r in act),
            reports_per_run=round(nrep / len(runs), 2), upheld_rate=round(sum(r.get("n_upheld", 0) for r in rs) / nrep, 2) if nrep else float("nan"),
            first_report_job=mean(first_rep), fines_per_run=round(sum(r.get("fines_total", 0) for r in rs) / len(runs)),
            false_report_fines_per_run=round(sum(r.get("false_report_fines", 0) for r in rs) / len(runs)),
            detect_rate=round(sum(flagged) / len(flagged), 2), detect_before_offer=round(sum(early) / len(early), 2),
            detect_first_job=mean(firsts), monitor_score_mean=mean(mon)))
    return out


def to_markdown(table):
    if not table:
        return ""
    h = list(table[0])
    L = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    L += ["| " + " | ".join(str(r[c]) for c in h) + " |" for r in table]
    return "\n".join(L)


def write_csv(rows, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
