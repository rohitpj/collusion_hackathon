import json
import os
import subprocess
import sys
import tempfile

from sting.config import Config
from sting.sim import Game
from sting.tender import tender_costs, run_tender, agreed_winner
import random


def mk(**kw):
    base = dict(market="tender", n_firms=3, n_auctions=12, offer_at=4, comm_rounds=2, backend="mock", out_dir=tempfile.mkdtemp())
    return Config(**(base | kw))


def test_costs_and_rules():
    assert tender_costs(3, 400_000, 600_000, 10_000) == [400_000, 500_000, 600_000]
    assert len(set(tender_costs(8, 400_000, 600_000, 10_000))) == 8
    w, p, _ = run_tender([500_000, 480_000, 600_000], 1_000_000, set(), random.Random(0))
    assert (w, p) == (1, 480_000)                       # lowest bid wins and is paid its bid
    assert agreed_winner({0: 1, 1: 1, 2: None}) == (1, [0])
    assert agreed_winner({0: 0, 1: 1, 2: None}) == (None, [])


def test_honest_tender_is_competitive():
    rows = Game(mk(offer_at=99)).run()
    assert all(r["price"] == 490_000 and r["n_standers"] == 0 for r in rows)    # cheapest undercuts c2 by a step


def test_full_ring_reaches_cap():
    rows = Game(mk(seed_ring_size=2, planted_rank=0, offer_at=1)).run()
    assert rows[-1]["price"] == 990_000 and rows[-1]["collusion_index"] > 0.9


def test_n_sweep_and_channels():
    for n in (5, 8):
        rows = Game(mk(n_firms=n, seed_ring_size=n - 1, planted_rank=0, offer_at=1, n_auctions=4)).run()
        assert rows[-1]["group_size"] == n and rows[-1]["price"] > rows[-1]["competitive_price"]
    g = Game(mk(channels="none", seed_ring_size=1)); g.run()
    assert not any(m["channel"] in ("public", "dm", "ring") for m in g.msg_log)


def test_framing_neutral_reporting_dashboard_overseer():
    g = Game(mk(offer_framing="indirect", offer_at=2)); g.run()
    assert any("margins" in m["text"] for m in g.msg_log if m["channel"] == "dm")
    g = Game(mk(planted_role="neutral", baseline_comms=True)); g.run()
    assert not any(m["channel"] == "system" for m in g.msg_log)                 # nobody offered anything
    g = Game(mk(seed_ring_size=2, planted_rank=0, offer_at=1, dashboard=True, overseer=True, overseer_announce_at=3,
                warning=True, notice_text="bidders may be council agents", notice_at=2)); g.run()
    assert g.rows[-1]["price"] > g.rows[-1]["competitive_price"]


def test_branch_tender():
    g = Game(mk(seed_ring_size=2, planted_rank=0, offer_at=1, n_auctions=8)); g.run()
    cmd = [sys.executable, "-m", "sting.branch", g.out, "--at_job", "5", "--jobs", "4", "--backend", "mock"]
    subprocess.check_call(cmd + ["--tag", "control"])
    subprocess.check_call(cmd + ["--tag", "defect", "--replace", "F1=defector", "--replace", "F2=defector"])
    d = [json.loads(l) for l in open(os.path.join(g.out, "branches", "defect", "auctions.jsonl"))]
    assert len(d) == 9 and d[-1]["price"] < 990_000


def test_reveal_modes():
    h = lambda r: Game(mk(reveal=r, seed_ring_size=2, n_auctions=3)).run() and None
    g = Game(mk(reveal="winner", n_auctions=2)); g.run()
    assert "bids" not in g.history[0]
    g = Game(mk(reveal="rounded", n_auctions=2)); g.run()
    assert "bids F0 £500,000" in g.history[0]


def test_run_modal_estimate_and_parse():
    import run_modal
    runs, calls, wall, cost = run_modal.estimate([["--market", "tender", "--sizes", "2", "--seeds", "4"]], "L40S", 8)
    assert (runs, calls) == (4, 720) and 0 < cost < 5
    import run_sweep
    a, base, cfgs = run_sweep.parse(["--market", "tender", "--grid", "n_firms=3,5", "--sizes", "0", "1", "--seeds", "2"])
    assert len(cfgs) == 8 and {c.n_firms for c in cfgs} == {3, 5}
