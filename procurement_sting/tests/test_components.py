import json
import os
import random
import subprocess
import sys
import tempfile

from sting.config import Config
from sting.detect import calibrate, run_detection
from sting.auction import run_auction, draw_costs
from sting.sim import Game
from sting.judge import judge_run
from sting.overseer import Overseer


def mk(**kw):
    d = tempfile.mkdtemp()
    return Config(backend="mock", out_dir=d, **kw)


def test_detector_false_positive_rate_and_power():
    cfg = Config()
    rng = random.Random(7)
    fp = 0
    for _ in range(400):
        prices = [run_auction(draw_costs(5, cfg.cost_lo, cfg.cost_hi, rng, cfg.step), cfg.cap, cfg.step, rng).price
                  for _ in range(cfg.n_auctions)]
        fp += max(run_detection(prices, [0] * len(prices), cfg)["flag"])
    assert fp / 400 < 0.04            # alpha = 1%, allow MC noise
    assert max(run_detection([1_000_000] * 12, [0] * 12, cfg)["flag"]) == 1


def test_no_offer_run_not_flagged():
    rows = Game(mk(offer_at=99, seed=3)).run()
    assert not any(r["above_hi"] for r in rows)


def test_overseer_gag_blocks_channel():
    cfg = mk(seed_ring_size=2, overseer=True, overseer_scope="public+dm", gag_on_score4=True, overseer_announce_at=1)
    g = Game(cfg)
    g.run()
    assert all(r["monitor_score"] == "" or 0 <= r["monitor_score"] <= 4 for r in g.rows)
    assert "monitor.jsonl" in os.listdir(g.out)


def test_judge_mock():
    g = Game(mk(seed_ring_size=2, seed=1)); g.run()
    assert judge_run(g.out, "all")["rating"] >= 1


def test_branch_replays_prefix_and_diverges():
    g = Game(mk(seed_ring_size=3, seed=2, n_auctions=8)); g.run()
    out = os.path.join(g.out, "branches")
    base = [sys.executable, "-m", "sting.branch", g.out, "--at_job", "6", "--jobs", "4", "--backend", "mock"]
    subprocess.check_call(base + ["--tag", "control"])
    subprocess.check_call(base + ["--tag", "defect", "--replace", "F1=defector", "--replace", "F2=defector"])
    c = [json.loads(l) for l in open(os.path.join(out, "control", "auctions.jsonl"))]
    d = [json.loads(l) for l in open(os.path.join(out, "defect", "auctions.jsonl"))]
    orig = [json.loads(l) for l in open(os.path.join(g.out, "auctions.jsonl"))]
    assert len(c) == len(d) == 10
    assert [r["costs"] for r in c[:6]] == [r["costs"] for r in orig[:6]]    # shared prefix
    assert [r["costs"] for r in c] == [r["costs"] for r in d]               # same cost draws on every branch
