"""Score saved AgentHarm logs using an OpenRouter judge; never rerun candidate inference."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import uuid

from inspect_ai.scorer import Score, scorer
from inspect_evals.agentharm.metric import (
    avg_full_score, avg_refusals, avg_score, avg_score_non_refusals,
    category_avg_refusals, category_avg_score,
)


@scorer(metrics=[avg_score(), avg_full_score(), avg_refusals(), avg_score_non_refusals(),
                 category_avg_score(), category_avg_refusals()])
def replay_scorer(grading_module, refusal_judge, semantic_judge):
    from inspect_evals.agentharm.scorer import combined_scorer
    official = combined_scorer(grading_module, refusal_judge, semantic_judge)

    async def compute(state, target):
        try:
            return await official(state, target)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # Upstream metrics exclude NaN rubric scores and None refusals.
            # Never turn a crashed grader or judge request into a harmless result.
            return Score(value={"score": float("nan"), "refusal": None},
                         explanation=f"Unscorable sample: {error}",
                         metadata={"category": state.metadata.get("category", "unknown"),
                                   "scoring_error": error})
    return compute


def score_log(source, destination, judge, limit=0):
    from inspect_ai import score
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.model import get_model
    from inspect_evals.agentharm.benchmark import benign_grading_functions, harmful_grading_functions
    from inspect_evals.agentharm.scorer import RefusalJudgeLLM

    log = read_eval_log(str(source))
    task = log.eval.task.rsplit("/", 1)[-1]
    if task not in ("agentharm", "agentharm_benign"):
        raise ValueError(f"Not an AgentHarm log: {source.name}")
    if log.status != "success" or not log.samples or any(s.error for s in log.samples):
        raise ValueError(f"Source log is incomplete or contains sample errors: {source.name}")
    if limit:
        log.samples = log.samples[:limit]
        if log.results:
            log.results.total_samples = len(log.samples)
    module = benign_grading_functions if task == "agentharm_benign" else harmful_grading_functions

    class CheckedRefusalJudge(RefusalJudgeLLM):
        async def check_refusal(self, generation):
            refusal, explanation = await super().check_refusal(generation)
            if explanation.startswith("Error in RefusalJudgeLLM:"):
                raise RuntimeError(explanation)
            return refusal, explanation

    def reject_candidate(*args):
        raise RuntimeError("Candidate generation is forbidden during post-run scoring")

    replay = get_model("mockllm/replay-guard", custom_outputs=reject_candidate)
    scored = score(log, replay_scorer(module, CheckedRefusalJudge(judge), judge),
                   model=replay, action="overwrite", display="plain", copy=False)
    write_eval_log(scored, str(destination))
    errors = sum(bool(sample.error) for sample in scored.samples or [])
    scoring_errors = [{"id": sample.id, "error": value.metadata["scoring_error"]}
                      for sample in scored.samples or [] for value in (sample.scores or {}).values()
                      if value.metadata and value.metadata.get("scoring_error")]
    return {"source": source.name, "scored_log": destination.name, "model": str(log.eval.model),
            "dataset": "benign" if task == "agentharm_benign" else "harmful",
            "samples": len(scored.samples or []), "sample_errors": errors,
            "scoring_errors": scoring_errors, "scored_samples": len(scored.samples or []) - len(scoring_errors),
            "results": scored.results.model_dump(mode="json") if scored.results else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--key-file", default="", help="Local file containing only the API key")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--judge-model", default="openrouter/openai/gpt-4o-2024-08-06")
    parser.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-connections", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Samples per log; 0 scores every saved sample")
    parser.add_argument("--dataset", choices=("both", "harmful", "benign"), default="both")
    args = parser.parse_args()
    if args.limit < 0 or args.max_connections < 1:
        parser.error("limits must be nonnegative and connections positive")
    from agentharm_modal import preflight_judge
    key = preflight_judge(True, args.judge_model, args.judge_api_key_env,
                          env_file=args.env_file, key_file=args.key_file)
    from inspect_ai.model import GenerateConfig, get_model
    judge = get_model(args.judge_model, api_key=key, base_url=args.judge_base_url,
                      config=GenerateConfig(temperature=0, max_connections=args.max_connections,
                                            max_tokens=2048, max_retries=2, timeout=90))
    sources = sorted((args.run_dir / "logs").glob("*.eval"))
    if args.dataset != "both":
        sources = [p for p in sources if ("agentharm-benign" in p.name) == (args.dataset == "benign")]
    if not sources:
        parser.error("No matching Inspect .eval logs found under RUN_DIR/logs")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = args.run_dir / "scoring" / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"judge_model": args.judge_model, "judge_base_url": args.judge_base_url,
                "source_run": str(args.run_dir.resolve()), "limit_per_log": args.limit,
                "max_connections": args.max_connections, "max_tokens": 2048,
                "versions": {p: importlib.metadata.version(p) for p in ("inspect-ai", "inspect-evals", "openai")},
                "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = []
    for source in sources:
        print(f"Scoring {source.name}", flush=True)
        row = score_log(source, out / source.name, judge, args.limit)
        summary.append(row)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved {row['samples']} samples; sample_errors={row['sample_errors']}; "
              f"unscorable={len(row['scoring_errors'])}", flush=True)
    print(f"Scored results: {out.resolve()}", flush=True)
    if any(row["sample_errors"] or row["scoring_errors"] for row in summary):
        raise SystemExit("Scoring contained errors; inspect the saved logs before interpreting metrics.")


if __name__ == "__main__":
    os.environ["PYTHONUTF8"] = "1"
    main()
