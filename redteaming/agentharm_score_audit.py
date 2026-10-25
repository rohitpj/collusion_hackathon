"""Validate saved judge verdicts and recompute official metrics on scorable samples only.

Usage: python agentharm_score_audit.py RUN_DIR/scoring/SCORING_ID
No API requests are made. Raw scored logs and original inference logs are preserved.
"""

import argparse
import json
import math
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import SampleScore
from inspect_evals.agentharm.metric import (
    avg_full_score, avg_refusals, avg_score, avg_score_non_refusals,
    category_avg_refusals, category_avg_score,
)


def audited_metrics(log):
    valid, errors = [], []
    for sample in log.samples or []:
        values = list((sample.scores or {}).values())
        reason = None
        if sample.error or not values:
            reason = "Missing score or sample error"
        else:
            value = values[0]
            reason = (value.metadata or {}).get("scoring_error")
            raw = value.value
            if not isinstance(raw, dict) or raw.get("score") is None or not math.isfinite(raw["score"]):
                reason = reason or "Rubric score is missing or non-finite"
            elif raw.get("refusal") not in (0, 1):
                reason = reason or "Refusal verdict is missing or invalid"
        for event in sample.events:
            data = getattr(event, "data", None)
            prefix = "Semantic judge response: `"
            if event.event == "info" and isinstance(data, str) and data.startswith(prefix):
                verdict = data[len(prefix):].removesuffix("`").strip().lower()
                if verdict not in ("pass", "fail"):
                    reason = reason or "Semantic judge returned a verdict other than pass/fail"
            if event.event == "model" and str(getattr(event, "model", "")).startswith(
                ("openrouter/", "openai-api/", "openai/")
            ):
                messages = getattr(event, "input", [])
                prompt = "\n".join(m.text for m in messages)
                output = getattr(event, "output", None)
                # AgentHarm's only judge calls are semantic pass/fail or refusal
                # classification. Check model events as well as transcript info.
                if output and "<response>" not in prompt:
                    verdict = (output.completion or "").strip().lower()
                    if verdict not in ("pass", "fail"):
                        reason = reason or "Semantic judge returned a verdict other than pass/fail"
        if reason:
            errors.append({"id": sample.id, "reason": reason})
        else:
            valid.append(SampleScore(score=values[0], sample_id=sample.id))
    factories = {"avg_score": avg_score, "avg_full_score": avg_full_score,
                 "avg_refusals": avg_refusals, "avg_score_non_refusals": avg_score_non_refusals,
                 "category_avg_score": category_avg_score, "category_avg_refusals": category_avg_refusals}
    metrics = {name: factory()(valid) for name, factory in factories.items()} if valid else {}
    return {"samples": len(log.samples or []), "scored_samples": len(valid),
            "unscorable_samples": len(errors), "scoring_errors": errors, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scoring_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.scoring_dir / "summary.json").read_text(encoding="utf-8"))
    output = []
    for row in summary:
        log = read_eval_log(str(args.scoring_dir / row["scored_log"]), resolve_attachments=True)
        audited = audited_metrics(log)
        output.append({"model": row["model"], "dataset": row["dataset"],
                       "scored_log": row["scored_log"], **audited})
        metrics = audited["metrics"]
        print(f"{row['model']} {row['dataset']}: scored={audited['scored_samples']}/{audited['samples']}, "
              f"mean_score={metrics.get('avg_score', float('nan')):.4f}, "
              f"refusal={metrics.get('avg_refusals', float('nan')):.4f}")
    result = {"note": "Official metric functions applied only to samples with valid rubric/refusal and semantic verdicts. "
                      "Unscorable samples are excluded, not treated as zero. Raw score logs are preserved.",
              "conditions": output}
    (args.scoring_dir / "audited_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
