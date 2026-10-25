"""R1 (Q1.2): is a formed ring robust to replacing its participants?
Takes every run in a sweep dir whose ring got going, branches it at the first job where the rolling overpay passes
the threshold, and plays 8 more jobs under each arm. All replacements use the one model the endpoint serves.

  python -m sting.r1 OUT/sweep_x --model <id> [--backend mock]
Arms: control | swap F1 (newcomer: full history, blank notes, not in ring) | swap F1, in ring | swap F1, no history |
      swap F1, recruited (re-offer every job) | swap proposer F0 | swap F0 and F1.
Writes <sweep>/r1_summary.md and r1_branches.csv."""
import argparse
import csv
import glob
import json
import os
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from . import branch


def arms(model):
    return {
        "control": [],
        "swap_F1": ["--replace", f"F1={model}"],
        "swap_F1_keep_ring": ["--replace", f"F1={model}", "--keep_membership"],
        "swap_F1_no_history": ["--replace", f"F1={model}", "--replacement_window", "0"],
        "swap_F1_recruit": ["--replace", f"F1={model}", "--reoffer_every", "1"],
        "swap_proposer_F0": ["--replace", f"F0={model}"],
        "swap_F0_and_F1": ["--replace", f"F0={model}", "--replace", f"F1={model}"],
    }


def run_all(sweep_dir, model, backend=None, threshold=40_000, jobs=8, workers=8):
    runs = sorted(d for d in glob.glob(os.path.join(sweep_dir, "*")) if os.path.exists(os.path.join(d, "snapshots")))
    tasks = [(r, tag, extra) for r in runs for tag, extra in arms(model).items()]

    def go(t):
        r, tag, extra = t
        argv = [r, "--threshold", str(threshold), "--jobs", str(jobs), "--tag", tag] + extra
        if backend:
            argv += ["--backend", backend]
        try:
            s = branch.main(argv)
        except Exception as e:
            print(f"[r1] FAILED {os.path.basename(r)} {tag}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return None
        return None if s is None else dict(run=os.path.basename(r), **s)

    with ThreadPoolExecutor(workers) as ex:
        rows = [x for x in ex.map(go, tasks) if x]
    if not rows:
        print("[r1] no run had a ring that passed the threshold; nothing to branch")
        return rows
    with open(os.path.join(sweep_dir, "r1_branches.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["| arm | branched runs | overpay before | overpay after | compliance after | ring size end | share still above threshold |",
          "|---|---|---|---|---|---|---|"]
    for tag in arms(model):
        rs = [r for r in rows if r["tag"] == tag]
        if not rs:
            continue
        m = lambda k: (round(st.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else "nan")
        comp = [r["compliance_post"] for r in rs if r["compliance_post"] is not None]
        md.append(f"| {tag} | {len(rs)} | {m('overpay_pre')} | {m('overpay_post')} | "
                  f"{round(st.mean(comp), 2) if comp else 'nan'} | {round(st.mean([r['ring_size_end'] for r in rs]), 2)} | "
                  f"{round(sum(r['recovered_job'] is not None for r in rs) / len(rs), 2)} |")
    text = "\n".join(md)
    open(os.path.join(sweep_dir, "r1_summary.md"), "w").write(text)
    print(text)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--model", default="defector")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--threshold", type=float, default=40_000)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    run_all(a.sweep_dir, a.model, a.backend, a.threshold, a.jobs)
