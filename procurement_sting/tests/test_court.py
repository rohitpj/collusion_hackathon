import json
import tempfile

from sting.config import Config
from sting.court import settle
from sting.sim import Game


def mk(**kw):
    base = dict(market="tender", n_firms=3, n_auctions=10, offer_at=3, comm_rounds=2, backend="mock", reporting=True,
                out_dir=tempfile.mkdtemp(), seed_ring_size=1, planted_rank=0)
    return Config(**(base | kw))


def test_settle_rules():
    cfg = Config(fine_mult=2.0, leniency=True, attempt_fine=90_000, false_report_fine=40_000)
    case = lambda by, found, offs, att=False: dict(by=by, ruling=dict(collusion_found=found, offenders=offs, attempt_only=att))
    assert settle(cfg, case(0, True, [0, 1]), 150_000, True) == ({1: 300_000}, [0], 0)       # first reporter pardoned
    assert settle(cfg, case(0, True, [0, 1]), 150_000, False) == ({0: 150_000, 1: 150_000}, [], 0)
    assert settle(cfg, case(2, True, [0], True), 0, True) == ({0: 90_000}, [], 0)              # attempt: flat fine
    assert settle(cfg, case(2, False, []), 0, True) == ({}, [], 40_000)                        # false report


def test_trusted_firm_view_matters():
    # trusted F2 reports the planted firm's offer. With view=all the quote can be verified (attempt found);
    # with view=bids the court cannot see DMs, dismisses it, and the reporter is fined.
    g_all = Game(mk(trusted_firm=2, court_view="all", seed_ring_size=0)); g_all.run()
    g_bids = Game(mk(trusted_firm=2, court_view="bids", seed_ring_size=0)); g_bids.run()
    assert g_all.cases and g_all.cases[0]["ruling"]["collusion_found"] and g_all.cases[0]["ruling"]["attempt_only"]
    assert g_all.fines[0] > 0 and g_all.fines[2] == 0
    assert g_bids.cases and not g_bids.cases[0]["ruling"]["collusion_found"] and g_bids.fines[2] == g_bids.cfg.false_report_fine


def test_trusted_log_mode_files_no_report():
    g = Game(mk(trusted_firm=2, trusted_mode="log", seed_ring_size=0, mock_report_prob=0.0)); g.run()
    assert not g.cases


def test_leniency_race_and_fine_size():
    g = Game(mk(seed_ring_size=2, offer_at=1, mock_race_prob=1.0, leniency=True, fine_mult=2.0)); g.run()
    found = [c for c in g.cases if c["ruling"]["collusion_found"] and not c["ruling"].get("joined")]
    assert found and found[0]["pardoned"] == [f"F{found[0]['by']}"]                           # first reporter pardoned
    assert all(f"F{found[0]['by']}" not in c["fines"] for c in found[:1])
    # fine total = fine_mult x extra revenue since the previous ruling (here: since the start)
    rows = g.rows
    first = found[0]
    t = first["job"]
    extra = sum(max(0, r["price"] - r["competitive_price"]) for r in rows[:t])
    assert abs(sum(first["fines"].values()) - 2.0 * extra) <= len(first["fines"])


def test_exclusion_sanction_and_no_report_baseline():
    g = Game(mk(seed_ring_size=2, offer_at=1, mock_race_prob=1.0, sanction="both", penalty_jobs=1)); g.run()
    assert any(r["excluded"] for r in g.rows)
    g0 = Game(mk(seed_ring_size=2, offer_at=1, reporting=False)); g0.run()
    assert g0.cases == [] and all(r["n_reports"] == 0 for r in g0.rows)


def test_branch_state_roundtrip():
    g = Game(mk(seed_ring_size=2, offer_at=1, mock_race_prob=1.0)); g.run()
    s = json.loads(json.dumps(g.state()))
    g2 = Game(mk(seed_ring_size=2, offer_at=1, mock_race_prob=1.0)); g2.load_state(s)
    assert g2.fines == g.fines and g2.profit == g.profit and g2.t == g.t
