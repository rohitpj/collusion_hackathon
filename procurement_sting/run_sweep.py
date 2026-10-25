"""Sweep experiments.

  python run_sweep.py --backend mock --seeds 5                                   # auction, ring size 0..4
  python run_sweep.py --market tender --backend mock --seeds 5                   # tender defaults (3 firms, 20 jobs)
  python run_sweep.py --market tender --grid n_firms=3,5,8 --sizes 0 1 --seeds 3 --backend llm --model <id>
  python run_sweep.py --market tender --grid channels=public+dm+ring,public,none --sizes 2

--grid KEY=v1,v2,...  (repeatable) crosses any Config field; --sizes is the number of non-planted firms
already in the ring (default 0..n_firms-1). Each combination is a condition `cond`; seeds are paired
(identical public/private cost draws and planted firm across conditions at the same seed).
"""
import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from sting.config import Config
from sting.metrics import summarise, to_markdown, write_csv
from sting.sim import run

TENDER_DEFAULTS = dict(n_firms=3, n_auctions=20, offer_at=5, comm_rounds=2, cap=1_000_000, step=10_000)


def coerce(field_default, v):
    if isinstance(field_default, bool):
        return v.lower() in ("1", "true", "yes")
    return type(field_default)(v)


def parse(argv=None):
    """Build the run configs from CLI args (no side effects)."""
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--market", default="auction")
    market = pre.parse_known_args(argv)[0].market
    defaults = Config().to_dict() | (TENDER_DEFAULTS if market == "tender" else {})
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grid", action="append", default=[])
    ap.add_argument("--yes", "-y", action="store_true")
    for k, v in defaults.items():
        if k in ("seed", "seed_ring_size", "tag", "cond"):
            continue
        ap.add_argument(f"--{k}", type=(lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool) else type(v), default=v)
    a = ap.parse_args(argv)
    base = {k: v for k, v in vars(a).items() if k in defaults and k not in ("seed", "seed_ring_size", "tag", "cond")}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base["out_dir"] = os.path.join(a.out_dir, f"sweep_{stamp}")
    grid = {}
    for g in a.grid:
        key, vals = g.split("=", 1)
        grid[key] = [coerce(defaults[key], x) for x in vals.split(",")]
    cfgs = []
    for combo in itertools.product(*grid.values()):
        over = dict(zip(grid, combo))
        n = over.get("n_firms", base["n_firms"])
        for k in (a.sizes if a.sizes is not None else range(0, n)):
            if k > n - 1:
                continue
            cond = ";".join([f"{kk}={vv}" for kk, vv in over.items()] + [f"k={k}"])
            for s in range(a.seeds):
                cfgs.append(Config(**(base | over), seed=s, seed_ring_size=k, tag="sweep", cond=cond))
    return a, base, cfgs


def main(argv=None):
    a, base, cfgs = parse(argv)
    if a.backend == "llm":
        per = sum(c.n_auctions * (c.n_firms * (1 + c.comm_rounds * (1.0 if c.market == "tender" else 0.6))) for c in cfgs)
        print(f"{len(cfgs)} runs, roughly {int(per)} LLM calls")
        if not a.yes and input("Continue? [y/N] ").lower() != "y":
            return
    def safe_run(c):
        try:
            return run(c)
        except Exception as e:    # one dead run (e.g. endpoint dropped) must not lose the finished ones
            print(f"[run_sweep] FAILED {c.cond} seed={c.seed}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            return []
    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(safe_run, cfgs) for r in rs]
    write_csv(rows, os.path.join(base["out_dir"], "all_auctions.csv"))
    md = to_markdown(summarise(rows))
    open(os.path.join(base["out_dir"], "summary.md"), "w").write(md)
    print(md)
    print("->", base["out_dir"])
    return base["out_dir"]


if __name__ == "__main__":
    main()
