import glob
import os
import tempfile

from sting import r1
from sting.config import Config
from sting.sim import Game


def test_spontaneous_arms_branch_a_markup_without_ring_seat():
    out = tempfile.mkdtemp()
    cfg = Config(market="tender", n_firms=3, n_auctions=8, offer_at=1, backend="mock", seed_ring_size=2, planted_rank=0,
                 out_dir=out, tag="sw", cond="x", channels="public+dm+ring")
    Game(cfg, out=os.path.join(out, "sweep_x", "run0")).run()
    rows = r1.run_all(os.path.join(out, "sweep_x"), "defector", backend="mock", jobs=3, workers=1, spontaneous=True)
    tags = {r["tag"] for r in rows}
    assert {"control", "swap_F1", "swap_F1_inherit_notes", "swap_F1_no_history", "swap_F2", "swap_F1_and_F2"} <= tags
    assert os.path.exists(os.path.join(out, "sweep_x", "r1_summary.md"))
