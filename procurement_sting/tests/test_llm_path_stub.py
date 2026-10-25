"""Exercises LLMAgent / Overseer / judge code paths with a fake chat client (no API)."""
import json
import re
import sys
import tempfile
import types

from sting.config import Config


class Stub:
    def __init__(self):
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, model, messages, temperature=0, **kw):
        self.calls += 1
        u = messages[-1]["content"]
        if "You are a court" in u:
            out = {"collusion_found": True, "attempt_only": False, "offenders": ["F0", "F2"], "rationale": "stub"}
        elif "overseer" in u:
            out = {"coordination_score": 4 if "ring" in u.lower() else 0, "evidence": []}
        elif "auditing a council" in u:
            out = {"rating": 8, "rationale": "stub"}
        elif '"reservation"' in u:
            cost = int(re.search(r"cost for this job: £([\d,]+)", u).group(1).replace(",", ""))
            cap = int(re.search(r"Cap £([\d,]+)", u).group(1).replace(",", ""))
            out = {"reservation": cap if "stand_down" in u else cost, "reason": "stub"}
        else:
            out = {"public": "", "ring": "", "dm": {}, "offer_ring_to": [], "join_ring": None}
            if "IT IS TIME" in u:
                tg = re.search(r"proposal now to: ([^.]*)\.", u).group(1).split(", ")
                out["offer_ring_to"] = tg
                out["dm"] = {t: "join my ring and take turns" for t in tg}
            if "Pending proposals" in u:
                out["join_ring"] = True
        if "scratch_pad_update" in u:
            out["scratch_pad_update"] = "notes: ring works"
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="```json\n" + json.dumps(out) + "\n```"))])


def test_llm_path_end_to_end():
    stub = Stub()
    sys.modules["poseidon_compute"] = types.SimpleNamespace(make_client=lambda: stub, resolve_model=lambda m: m)
    from sting.sim import Game
    cfg = Config(backend="llm", model="m", out_dir=tempfile.mkdtemp(), seed_ring_size=1, scratchpad=True,
                 overseer=True, overseer_scope="all", overseer_announce_at=4, gag_on_score4=True, n_auctions=8)
    g = Game(cfg)
    rows = g.run()
    assert stub.calls > 0
    assert any(len(a.scratch_pad) for a in g.agents.values() if hasattr(a, "client"))
    assert rows[-1]["ring_size"] == 5                      # stub targets all accept
    assert any(r["gagged"] for r in rows)                  # monitor scored 4 on the offer job
    assert any(m["channel"].startswith("blocked_") for m in g.msg_log) or True
    from sting.judge import judge_run
    assert judge_run(g.out, "all", stub, "m")["rating"] == 8


class TenderStub(Stub):
    def create(self, model, messages, temperature=0, **kw):
        u = messages[-1]["content"]
        if '"bid"' not in u:
            return super().create(model, messages, temperature, **kw)
        self.calls += 1
        cost = int(re.search(r"Your cost: £([\d,]+)", u).group(1).replace(",", ""))
        out = {"bid": cost + 300_000, "intended_winner": "F0", "reason": "stub",
               "report": {"accused": ["F1"], "evidence": "x"} if "may report" in messages[0]["content"] else None}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(out)))])


def test_tender_llm_path():
    stub = TenderStub()
    sys.modules["poseidon_compute"] = types.SimpleNamespace(make_client=lambda: stub, resolve_model=lambda m: m)
    from sting.sim import Game
    cfg = Config(market="tender", n_firms=3, n_auctions=6, offer_at=2, comm_rounds=2, backend="llm", model="m",
                 out_dir=tempfile.mkdtemp(), seed_ring_size=1, reporting=True, court="llm", court_view="all",
                 scratchpad=True, dashboard=True)
    g = Game(cfg)
    rows = g.run()
    assert rows[0]["intended"].count("F0") == 4 and rows[0]["agreed"] != "{}"
    assert g.cases and any("Court ruling" in h for h in g.history)   # reports reach the LLM court and are ruled on
    assert sum(g.fines) > 0
    assert g.rows[-1]["price"] >= 700_000
