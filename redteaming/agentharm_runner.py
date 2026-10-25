"""Run official AgentHarm tasks inside the Modal GPU container."""

import gc
import importlib.metadata
import json
import os
from pathlib import Path


def run(config, output_dir):
    import torch
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_evals.agentharm import agentharm, agentharm_benign
    from inspect_evals.agentharm.utils import AGENTHARM_DATASET_REVISION
    import agentharm_adapter  # noqa: F401: register provider
    import generate as g

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        **config,
        "model_revisions": {"dolos": g.DOLOS_REVISION, "qwen": g.QWEN_REVISION},
        "dataset_revision": AGENTHARM_DATASET_REVISION,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("inspect-ai", "inspect-evals", "torch", "transformers")},
        "scoring": "official" if config["score"] else "disabled; no benchmark harm/refusal scores",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if config["score"]:
        key = os.environ.get(config["judge_api_key_env"])
        if not key:
            raise ValueError(f"scoring requires {config['judge_api_key_env']} in the Modal environment")
        judge = get_model(config["judge_model"], api_key=key,
                          base_url=config["judge_base_url"] or None,
                          config=GenerateConfig(temperature=0, max_connections=4))
    else:
        # Task constructors require a judge even with score=False. It must never run.
        def reject_judging(*args):
            raise RuntimeError("judge invoked during an unscored run")
        judge = get_model("mockllm/unscored", custom_outputs=reject_judging)

    summaries = []
    factories = {"harmful": agentharm, "benign": agentharm_benign}
    for model_name in config["models"]:
        candidate = get_model(f"dolos_hf/{model_name}", tool_parser=config["tool_parser"])
        for dataset in config["datasets"]:
            task = factories[dataset](
                split=config["split"], behavior_ids=config["behavior_ids"],
                detailed_behaviors=config["detailed_behaviors"], hint_included=config["hint_included"],
                n_irrelevant_tools=config["irrelevant_tools"],
                refusal_judge=judge, semantic_judge=judge,
                agent_kwargs={"tool_choice": "auto", "user_prompt_template": "empty",
                              "prompt_technique": "standard"},
            )
            if not len(task.dataset):
                raise ValueError(f"no {dataset} samples matched the selected filters")
            logs = inspect_eval(
                task, model=candidate, limit=config["limit"] or None,
                epochs=config["epochs"], score=config["score"],
                log_dir=str(out / "logs"), log_format="eval", log_model_api=True,
                display="plain", max_samples=1, max_tasks=1, fail_on_error=False,
                message_limit=config["max_messages"],
                temperature=config["temperature"], seed=config["seed"],
                max_tokens=config["max_tokens"], max_connections=1,
            )
            for log in logs:
                summary = {"model": model_name, "dataset": dataset, "status": log.status,
                           "log": str(log.location), "scored": config["score"],
                           "results": log.results.model_dump(mode="json") if log.results else None,
                           "judge_errors": sum(
                               "Error in RefusalJudgeLLM:" in item.get("explanation", "")
                               for sample in log.samples or []
                               for item in (sample.metadata or {}).get("refusal_explanations", [])
                           ),
                           "error": log.error.model_dump(mode="json") if log.error else None}
                summaries.append(summary)
                with (out / "traces.jsonl").open("a", encoding="utf-8") as stream:
                    for sample in log.samples or []:
                        row = sample.model_dump(mode="json", exclude={"events"})
                        row.update(model=model_name, dataset=dataset)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        # Keep only one 1.5B model resident at a time for paired comparisons.
        candidate.api.model = None
        candidate.api.tokenizer = None
        del candidate
        gc.collect()
        torch.cuda.empty_cache()
    return summaries
