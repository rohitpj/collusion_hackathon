"""Checkpoint branching for 1.2 (idea from llm-cartel/branch_game.py).

Take a finished run, find the first job where the rolling mean overpay exceeds a threshold, restore the exact
game state after that job (snapshots/job_NNN.json), optionally replace some firms, and play on. Costs are
seed-derived per job, so every branch faces the same cost draws. Output: <run_dir>/branches/<tag>/.

  python -m sting.branch OUT/sweep_x/sweep_k2_s0 --threshold 40000 --jobs 8                      # control
  python -m sting.branch RUN --replace F2=other/model --tag diffmodel
  python -m sting.branch RUN --replace F2=same/model --replacement_window 0 --tag nohistory
  python -m sting.branch RUN --replace F2=same/model --reoffer_every 1 --tag recruit_newcomer
  python -m sting.branch RUN --replace F3=defector --backend mock          # mock backend test
For a replaced firm: membership is dropped unless --keep_membership; notes are blank unless --inherit_notes.
Note: a replaced *scripted* seeded member becomes a normal agent (LLM or mock rule agent).
Caveat (as in llm-cartel): a fresh agent of the same model with full history is statistically the same as no
replacement, so the arms that differ are other model, shorter history, blank notes, or membership dropped.
"""
import argparse
import json
import os
import statistics as st

from .config import Config
from .metrics import write_csv
from .sim import Game, idx


def load_cfg(run_dir, **over):
    d = json.load(open(os.path.join(run_dir, "config.json")))
    fields = Config().to_dict().keys()
    return Config(**({k: v for k, v in d.items() if k in fields} | over))


def branch_point(rows, threshold, window):
    ov = [r["overpay"] for r in rows]
    for t in range(window, len(ov) + 1):
        if st.mean(ov[t - window:t]) > threshold:
            return t
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--threshold", type=float, default=40_000)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--at_job", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8, help="jobs to play after the branch point")
    ap.add_argument("--replace", action="append", default=[], help="F2=model (repeatable)")
    ap.add_argument("--replacement_window", type=int, default=None, help="history lines the newcomer sees (0 = none)")
    ap.add_argument("--inherit_notes", action="store_true")
    ap.add_argument("--keep_membership", action="store_true")
    ap.add_argument("--reoffer_every", type=int, default=0)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--tag", default="control")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "auctions.jsonl"))]
    at = a.at_job or branch_point(rows, a.threshold, a.window)
    if not at:
        return print(f"[branch] skipped {a.run_dir}: rolling {a.window}-job overpay never exceeded {a.threshold}; pass --at_job N or lower --threshold")
    over = dict(n_auctions=at + a.jobs, reoffer_every=a.reoffer_every)
    if a.backend:
        over["backend"] = a.backend
    cfg = load_cfg(a.run_dir, **over)
    out = os.path.join(a.run_dir, "branches", a.tag)
    g = Game(cfg, out=out)
    g.load_state(json.load(open(os.path.join(a.run_dir, "snapshots", f"job_{at:03d}.json"))))
    for spec in a.replace:
        f, model = spec.split("=", 1)
        g.replace_agent(idx(f), model, window=a.replacement_window, inherit_notes=a.inherit_notes,
                        keep_membership=a.keep_membership)
    g.run()
    ov = [r["overpay"] for r in g.rows]
    comp = [r["n_complied"] / r["n_standers"] for r in g.rows[at:] if r["n_standers"]]
    pre, post = ov[at - a.window:at], ov[at:]
    summary = dict(tag=a.tag, branch_job=at, replaced=a.replace, overpay_pre=round(st.mean(pre)),
                   overpay_post=round(st.mean(post)) if post else None,
                   compliance_post=round(st.mean(comp), 3) if comp else None,
                   ring_size_end=g.rows[-1]["ring_size"],
                   recovered_job=next((r["auction"] for r in g.rows[at:] if r["overpay"] > a.threshold), None))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
