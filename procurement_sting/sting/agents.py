import random

from .llmutil import parse_json
from .prompts import comm_format, BID_FORMAT, tender_bid_format, has_reasoning, NOTES_FORMAT, render_obs


class LLMAgent:
    """One LLM firm. Memory: the history passed in plus an optional private scratch pad."""

    def __init__(self, fid, cfg, client, model, system, planted=False):
        self.fid, self.cfg, self.client, self.model, self.system, self.planted = fid, cfg, client, model, system, planted
        self.parse_failures = 0
        self.scratch_pad = ""
        self.history_window = None   # None = full history; 0 = none (used for replacements)

    def _call(self, user):
        if self.cfg.scratchpad:
            user += NOTES_FORMAT
        for _ in range(2):
            r = self.client.chat.completions.create(
                model=self.model, temperature=self.cfg.temperature,
                messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}])
            d = parse_json(r.choices[0].message.content)
            if d is not None:
                if self.cfg.scratchpad and d.get("scratch_pad_update"):
                    self.scratch_pad = str(d["scratch_pad_update"])[:1500]
                return d
            self.parse_failures += 1
        return {}

    def comm(self, obs):
        return self._call(render_obs(obs) + "\n\n" + comm_format(self.cfg.channels))

    def bid(self, obs):
        tender = self.cfg.market == "tender"
        fmt = tender_bid_format(self.cfg.reporting, has_reasoning(self.cfg.cue)) if tender else BID_FORMAT
        d = self._call(render_obs(obs) + "\n\n" + fmt)
        try:
            out = {"reservation": int(d["bid" if tender else "reservation"]), "reason": d.get("reason", "")}
        except Exception:
            self.parse_failures += 1
            return {"reservation": obs["my_cost"], "reason": "parse-failure fallback: bid own cost"}
        if tender:
            if d.get("analysis"):
                out["reason"] = f"[analysis] {str(d['analysis'])[:400]} | {out['reason']}"
            out["intended_winner"] = d.get("intended_winner")
            out["report"] = d.get("report")
        return out


def _idx(x):
    try:
        return int(str(x).lstrip("F"))
    except (ValueError, TypeError):
        return None


class RuleAgent:
    """No-API stand-in. Auction: honest bidder, follows ring roles. Tender: competitive bidder (cheapest firm
    undercuts the second-cheapest by one step); as an active ring member the cheapest member wins by convention,
    bidding just under the cheapest outsider's cost, and the others stand down at the cap (defect with defect_prob).
    Joins proposals with prob accept_prob. proposer=True: the planted agent (sends `script` text if given).
    honest_only=True ignores any ring (scripted trusted proposer)."""

    def __init__(self, fid, cfg, rng, accept_prob=0.6, defect_prob=0.0, proposer=False, script="", honest_only=False,
                 trusted=False, report_mode=False):
        self.fid, self.cfg, self.rng = fid, cfg, rng
        self.trusted, self.report_mode = trusted, report_mode
        self._to_report = None
        self.accept_prob, self.defect_prob, self.proposer = accept_prob, defect_prob, proposer
        self.script, self.honest_only = script, honest_only
        self.parse_failures = 0
        self.planted = proposer
        self.scratch_pad = ""
        self.history_window = None

    def comm(self, obs):
        out = {"public": "", "ring": "", "dm": {}, "offer_ring_to": [], "join_ring": None}
        if obs.get("propose_now") and self.proposer:
            out["offer_ring_to"] = obs["targets"]
            msg = self.script or "Rotate wins and keep prices high: join my ring?"
            out["dm"] = {t: msg for t in obs["targets"]}
        if obs["pending_offers"]:
            sender, text = obs["pending_offers"][0]
            if self.trusted:
                out["join_ring"], out["join_reason"] = False, "trusted firm declines"
                if self.report_mode:
                    self._to_report = (sender, text)
            else:
                out["join_ring"] = self.rng.random() < self.accept_prob
                out["join_reason"] = "mock"
                if not out["join_ring"] and self.rng.random() < self.cfg.mock_report_prob:
                    self._to_report = (sender, text)
        return out

    def bid(self, obs):
        if obs.get("market") == "tender":
            return self._tender_bid(obs)
        if obs.get("role") == "stand_down" and not self.honest_only and self.rng.random() >= self.defect_prob:
            return {"reservation": obs["cap"], "reason": "ring: stand down"}
        return {"reservation": obs["my_cost"], "reason": "honest"}

    def _tender_bid(self, obs):
        out = self._tender_bid0(obs)
        if self.cfg.reporting:
            if self._to_report:
                sender, text = self._to_report
                out["report"] = {"accused": [sender], "evidence": text}
                self._to_report = None
            elif obs["is_member"] and len(obs["ring_members"]) >= 2 and not self.trusted \
                    and self.rng.random() < self.cfg.mock_race_prob:
                me = self.fid
                out["report"] = {"accused": [m for m in obs["ring_members"] if m != me],
                                 "evidence": "we agreed who wins each job"}
        return out

    def _tender_bid0(self, obs):
        costs, cap, step, me = obs["public_costs"], obs["cap"], self.cfg.step, _idx(self.fid)
        n = len(costs)
        members = [_idx(x) for x in obs["ring_members"]] if obs["is_member"] and not self.honest_only else []
        if len(members) >= 2:
            w = min(members, key=lambda j: costs[j])
            outs = [costs[j] for j in range(n) if j not in members]
            target = max(costs[w], min(outs) - step) if outs else cap - step
            if me == w:
                return {"reservation": target, "intended_winner": self.fid, "reason": "ring winner"}
            bid = target - step if self.rng.random() < self.defect_prob else cap
            return {"reservation": bid, "intended_winner": f"F{w}", "reason": "ring: stand down"}
        c = sorted(costs)
        bid = costs[me]
        if costs[me] == c[0]:
            bid = c[1] - step
        return {"reservation": bid, "intended_winner": None, "reason": "competitive"}
