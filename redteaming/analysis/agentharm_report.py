"""Summarize recorded AgentHarm trajectories without inventing harm/refusal scores.

Usage: python analysis/agentharm_report.py results/agentharm/RUN_ID
Writes diagnostics.json alongside the official logs and prints a compact table.
Repeated calls mean identical function and canonical JSON arguments within a sample.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(rows):
    groups = defaultdict(Counter)
    base_ids, categories = defaultdict(set), defaultdict(set)
    for row in rows:
        key = (row["model"], row["dataset"])
        counts = groups[key]
        counts["samples"] += 1
        counts["sample_errors"] += bool(row.get("error"))
        limit = row.get("limit") or {}
        counts["message_limit_stops"] += limit.get("type") == "message"
        counts["other_limit_stops"] += bool(limit) and limit.get("type") != "message"
        base_ids[key].add(str(row.get("metadata", {}).get("id_original", row["id"])))
        categories[key].add(row.get("metadata", {}).get("category", "unknown"))
        seen, valid_seen = set(), set()
        has_parse_error, has_repeat = False, False
        for message in row.get("messages", []):
            if message["role"] == "tool":
                counts["tool_execution_errors"] += bool(message.get("error"))
            if message["role"] != "assistant":
                continue
            counts["assistant_turns"] += 1
            for call in message.get("tool_calls") or []:
                counts["tool_call_attempts"] += 1
                signature = (call["function"], json.dumps(call["arguments"], sort_keys=True))
                is_error = bool(call.get("parse_error"))
                counts["tool_parse_errors"] += is_error
                repeated = signature in seen
                counts["repeated_call_attempts"] += repeated
                has_parse_error |= is_error
                has_repeat |= repeated
                seen.add(signature)
                if not is_error:
                    counts["validly_parsed_calls"] += 1
                    counts["repeated_valid_calls"] += signature in valid_seen
                    valid_seen.add(signature)
        counts["samples_with_parse_errors"] += has_parse_error
        counts["samples_with_repeats"] += has_repeat
        choices = row.get("output", {}).get("choices") or []
        if choices:
            counts["final_output_token_limit"] += choices[0].get("stop_reason") == "max_tokens"
    return [{"model": key[0], "dataset": key[1], "base_behaviors": len(base_ids[key]),
             "categories": sorted(categories[key]), **dict(counts)}
            for key, counts in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    with (args.run_dir / "traces.jsonl").open(encoding="utf-8") as stream:
        summary = summarize(json.loads(line) for line in stream if line.strip())
    payload = {"note": "Trajectory diagnostics only; these are not benchmark harm or refusal scores.",
               "conditions": summary}
    (args.run_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{'model':<8}{'dataset':<10}{'samples':>8}{'calls':>8}{'repeats':>9}{'parse err':>10}{'msg cap':>9}{'errors':>8}")
    for row in summary:
        print(f"{row['model']:<8}{row['dataset']:<10}{row['samples']:>8}"
              f"{row['tool_call_attempts']:>8}{row['repeated_call_attempts']:>9}"
              f"{row['tool_parse_errors']:>10}{row['message_limit_stops']:>9}{row['sample_errors']:>8}")


if __name__ == "__main__":
    main()
