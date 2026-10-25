"""Matched category and task comparisons from audited AgentHarm scores.

Usage: python analysis/agentharm_categories.py RUN_DIR/scoring/SCORING_ID
No API requests. Bootstrap resamples base tasks, keeping prompt variants together.
"""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import random
import statistics

from inspect_ai.log import read_eval_log


def mean(values):
    return statistics.mean(values)


def cluster_contrast(records, field, seed=0):
    groups = defaultdict(list)
    for record in records:
        groups[record["base_id"]].append(record[field])
    blocks = list(groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(10000):
        sample = [value for _ in blocks for value in rng.choice(blocks)]
        estimates.append(mean(sample))
    estimates.sort()
    observed = mean(record[field] for record in records)
    # Exploratory two-sided sign-flip test, one sign per base task.
    totals = [sum(block) for block in blocks]
    tail = sum(abs(sum(s * t for s, t in zip(signs, totals)) / len(records)) >= abs(observed) - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(blocks)))
    return {"mean": observed, "cluster_bootstrap_95_ci": [estimates[249], estimates[9749]],
            "base_tasks": len(blocks), "paired_variants": len(records),
            "exploratory_sign_flip_p": tail / (2 ** len(blocks))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scoring_dir", type=Path)
    args = parser.parse_args()
    audited = json.loads((args.scoring_dir / "audited_summary.json").read_text(encoding="utf-8"))
    cells = {}
    for condition in audited["conditions"]:
        model = condition["model"].rsplit("/", 1)[-1]
        excluded = {str(row["id"]) for row in condition["scoring_errors"]}
        log = read_eval_log(str(args.scoring_dir / condition["scored_log"]))
        cells[(model, condition["dataset"])] = {
            str(s.id): {"id": str(s.id), "base_id": str(s.metadata["id_original"]),
                        "category": s.metadata["category"], "name": s.metadata.get("name"),
                        **next(iter(s.scores.values())).value}
            for s in log.samples if str(s.id) not in excluded
        }
    shared = set.intersection(*(set(cell) for cell in cells.values()))
    matched = []
    for sample_id in sorted(shared):
        dh, qh = cells[("dolos", "harmful")][sample_id], cells[("qwen", "harmful")][sample_id]
        db, qb = cells[("dolos", "benign")][sample_id], cells[("qwen", "benign")][sample_id]
        if len({(r["base_id"], r["category"]) for r in (dh, qh, db, qb)}) != 1:
            raise ValueError(f"Sample {sample_id} does not match across models and datasets")
        matched.append({"id": sample_id, "base_id": dh["base_id"], "category": dh["category"],
                        "name": dh["name"], "dolos_harmful": dh["score"], "qwen_harmful": qh["score"],
                        "dolos_benign": db["score"], "qwen_benign": qb["score"],
                        "dolos_harmful_refusal": dh["refusal"], "qwen_harmful_refusal": qh["refusal"],
                        "harmful_gap": dh["score"] - qh["score"],
                        "benign_gap": db["score"] - qb["score"],
                        "difference_of_gaps": (dh["score"] - qh["score"]) - (db["score"] - qb["score"])})
    categories, tasks = [], []
    for category in sorted({r["category"] for r in matched}):
        rows = [r for r in matched if r["category"] == category]
        columns = ("dolos_harmful", "qwen_harmful", "dolos_benign", "qwen_benign",
                   "dolos_harmful_refusal", "qwen_harmful_refusal", "harmful_gap", "benign_gap")
        categories.append({"category": category, **{col: mean(r[col] for r in rows) for col in columns},
                           "harmful_contrast": cluster_contrast(rows, "harmful_gap"),
                           "difference_of_gaps": cluster_contrast(rows, "difference_of_gaps")})
        for base in sorted({r["base_id"] for r in rows}, key=int):
            sub = [r for r in rows if r["base_id"] == base]
            tasks.append({"category": category, "base_id": base, "name": sub[0]["name"], "variants": len(sub),
                          **{col: mean(r[col] for r in sub) for col in columns},
                          "difference_of_gaps": mean(r["difference_of_gaps"] for r in sub)})
    result = {"note": "Scores use only variants scorable for BOTH models on BOTH harmful and benign versions. "
                      "Deltas are Dolos minus Qwen. Difference of gaps = harmful delta minus benign delta; "
                      "this is a descriptive contrast, not a causal adjustment. Uncertainty resamples base tasks, "
                      "not their correlated prompt variants. Sign-flip p-values are exploratory and unadjusted.",
              "matched_variants": len(matched), "categories": categories, "tasks": tasks}
    path = args.scoring_dir / "category_analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Matched category means (percent); gaps and confidence intervals in percentage points")
    for row in categories:
        contrast = row["difference_of_gaps"]
        print(f"{row['category']:<16} n={contrast['paired_variants']:>2} bases={contrast['base_tasks']} "
              f"H {100*row['dolos_harmful']:5.1f}/{100*row['qwen_harmful']:5.1f} "
              f"B {100*row['dolos_benign']:5.1f}/{100*row['qwen_benign']:5.1f} "
              f"DH-QH={100*row['harmful_gap']:+5.1f} gapdiff={100*contrast['mean']:+5.1f} "
              f"CI={[round(100*v, 1) for v in contrast['cluster_bootstrap_95_ci']]} "
              f"refusal {100*row['dolos_harmful_refusal']:.1f}/{100*row['qwen_harmful_refusal']:.1f}")
    print("\nTasks with Dolos harmful score higher than Qwen:")
    for task in sorted(tasks, key=lambda r: r["harmful_gap"], reverse=True):
        if task["harmful_gap"] > 0:
            print(json.dumps(task))


if __name__ == "__main__":
    main()
