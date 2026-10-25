"""Court for tender mode: firms report suspected/attempted collusion, a judge rules, offenders are fined.

Judge views: 'bids' = published bids, prices and public chat; 'all' = also DMs and ring-channel messages.
Judges: 'rule' (public data, deterministic) or 'llm' (prompted judge; reuses the project's JSON parsing).
Fine = fine_mult x the group's extra revenue since the last ruling (or attempt_fine for an attempt), split equally among
fined offenders. Leniency: the first reporter, if itself an offender, is pardoned. A report whose case finds no collusion
costs the reporter false_report_fine. Rulings are public (added to every firm's history).
"""
import json
import random
import re

from .llmutil import parse_json, complete
from .overseer import KEYWORDS

PROMPT = """You are a court ruling on a bid-rigging report in a repeated sealed-bid tender for road contracts.
Costs are public: {costs}. Benchmark price under competition = second-lowest cost = £{comp:,}; in competition the cheapest
firm bids about £{cheap:,} and every other firm bids its own cost.

Last {k} jobs (bids by firm, winner, price paid):
{table}

Messages you can see:
{messages}

Report from {reporter} accusing {accused}: "{evidence}"
Evidence check: {check}

Decide whether collusion (agreed or attempted bid rigging) occurred. Prices above the benchmark may also come from
independent mark-ups; talk alone is not collusion unless it proposes or confirms rigging. Return JSON only:
{{"collusion_found": true|false, "attempt_only": true|false, "offenders": ["F#", ...], "rationale": "<one or two sentences>"}}"""


def competitive_bid(costs, i, step):
    c = sorted(costs)
    return c[1] - step if costs[i] == c[0] else costs[i]


def _norm(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower())


def verify_quote(evidence, msgs):
    """Does the evidence quote occur in the visible messages? Returns the matching message or None.
    Matches if the normalised evidence is a substring of a message (or vice versa) or >= 60% of its words (>= 3) appear."""
    ev = _norm(evidence)
    words = set(ev.split())
    if not words:
        return None
    for m in msgs:
        t = _norm(m["text"])
        if ev in t or (len(t) > 8 and t in ev):
            return m
        if len(words) >= 3 and len(words & set(t.split())) / len(words) >= 0.6:
            return m
    return None


class Court:
    def __init__(self, cfg, client=None, model=None):
        self.cfg, self.client, self.model = cfg, client, model

    # ---- what the judge can see ----
    def visible(self, msg_log, jobs):
        ch = {"public"} if self.cfg.court_view == "bids" else {"public", "dm", "ring"}
        return [m for m in msg_log if m["auction"] in jobs and m["channel"] in ch]

    def window(self, rows):
        return rows[-self.cfg.court_window:]

    # ---- rulings ----
    def rule_judge(self, rows, costs, msgs, report):
        step, n = self.cfg.step, len(costs)
        win = self.window(rows)
        anomalous = [r for r in win if r["price"] > r["competitive_price"] + step]
        counts = [0] * n
        for r in win:
            bids = json.loads(r["reservations"])
            for i in range(n):
                if bids[i] > competitive_bid(costs, i, step) + step:
                    counts[i] += 1
        offenders = [i for i in range(n) if counts[i] >= 2]
        if len(anomalous) >= 2 and len(offenders) >= 2:
            return dict(collusion_found=True, attempt_only=False, offenders=offenders,
                        rationale=f"{len(anomalous)} of {len(win)} jobs priced above benchmark; firms bidding above competitive level repeatedly: {offenders}")
        hit = report["hit"]
        if hit and any(k in hit["text"].lower() for k in KEYWORDS):
            sender = int(hit["from"].lstrip("F"))
            if sender in report["accused"]:
                return dict(collusion_found=True, attempt_only=True, offenders=[sender],
                            rationale=f"verified message from F{sender} proposing coordination")
        return dict(collusion_found=False, attempt_only=False, offenders=[], rationale="insufficient evidence in view")

    def llm_judge(self, rows, costs, msgs, report):
        win = self.window(rows)
        table = "\n".join(f"Job {r['auction']}: bids {r['reservations']}; winner {r['winner']}; price £{r['price']:,}" for r in win)
        mt = "\n".join(f'[job {m["auction"]} {m["channel"]}] {m["from"]}->{m["to"]}: "{m["text"]}"' for m in msgs) or "(none)"
        hit = report["hit"]
        check = (f'quote found in a message from {hit["from"]} on the {hit["channel"]} channel' if hit
                 else "quote NOT found in any message the court can see")
        n, c = len(costs), sorted(costs)
        d = parse_json(complete(self.client, self.model, PROMPT.format(
            costs=", ".join(f"F{i} £{x:,}" for i, x in enumerate(costs)), comp=c[1], cheap=c[1] - self.cfg.step,
            k=len(win), table=table, messages=mt, reporter=(f"F{report['by']}" if report["by"] >= 0 else "the council auditor"),
            accused=", ".join(f"F{i}" for i in report["accused"]) or "nobody", evidence=report["evidence"], check=check))) or {}
        offs = [int(str(o).lstrip("F")) for o in d.get("offenders", []) if str(o).lstrip("F").isdigit() and int(str(o).lstrip("F")) < n]
        return dict(collusion_found=bool(d.get("collusion_found")) and bool(offs), attempt_only=bool(d.get("attempt_only")),
                    offenders=offs, rationale=str(d.get("rationale", ""))[:300])

    def rule_on(self, rows, costs, msg_log, report):
        jobs = {r["auction"] for r in self.window(rows)}
        msgs = self.visible(msg_log, jobs)
        report = dict(report, hit=verify_quote(report["evidence"], msgs))
        if self.cfg.court == "llm" and self.client is not None:
            out = self.llm_judge(rows, costs, msgs, report)
        else:
            out = self.rule_judge(rows, costs, msgs, report)
        return out | {"quote_verified": bool(report["hit"])}


def settle(cfg, case, extra_since, first_in_case):
    """Return ({firm: fine}, pardoned list, reporter_fine) for a ruling."""
    r = case["ruling"]
    if not r["collusion_found"]:
        return {}, [], cfg.false_report_fine
    offs = list(r["offenders"])
    pardoned = [case["by"]] if (cfg.leniency and first_in_case and case["by"] in offs) else []
    fined = [o for o in offs if o not in pardoned]
    total = cfg.attempt_fine if (r["attempt_only"] or extra_since <= 0) else cfg.fine_mult * extra_since
    return ({o: round(total / len(fined)) for o in fined} if fined else {}), pardoned, 0
