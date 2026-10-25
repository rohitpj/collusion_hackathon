# Dolos on AgentHarm via Modal

This harness runs the official `inspect_evals.agentharm` tasks and simulated tools,
with an Inspect model provider that loads the same pinned Dolos/Qwen weights as
`generate.py`. Agent inference, tool execution and scoring run in one Modal A10G
container. Only the independent judge calls an external model API. It uses the
benchmark's standard agent prompt, automatic tool choice and no jailbreak template.

Upstream: [AgentHarm implementation and documentation](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm).
The runner pins `inspect-evals==0.26.0` and `inspect-ai==0.3.284`; that benchmark
release pins its dataset revision, which is also written to each run manifest.
Treat the dataset as evaluation data, not training data.

## Install and smoke test (Windows PowerShell)

Run from `redteaming/`. This separate environment leaves the existing one intact.
The requirements support launching, offline tests
and viewing Inspect logs locally.

```powershell
uv venv .venv-agentharm --python 3.13
uv pip install --python .venv-agentharm/Scripts/python.exe -r requirements-agentharm.txt
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
# Only if you have not already authenticated:
.venv-agentharm/Scripts/python.exe -m modal setup

# One short benign trajectory, no judge credentials or scoring:
.venv-agentharm/Scripts/python.exe agentharm_run.py --dataset benign --limit 1 --no-score --max-tokens 256 --max-messages 6
```

The smoke test's smaller output/message limits are for plumbing verification only.
They are recorded in the manifest. An unscored run produces **no harm or refusal
score**. A model returning an invalid tool call is an observed model failure, not
necessarily a harness failure; its parsing error is returned through Inspect's
normal tool error mechanism.

## Scored evaluations

Scoring requires a separate judge credential; Modal authentication alone does not
provide it. Both official judges now use GPT-4o (2024-08-06) through OpenRouter:
`openrouter/openai/gpt-4o-2024-08-06`, with `OPENROUTER_API_KEY`.

1. Copy `.env.example` to `.env` in `redteaming/`.
2. Open `.env` in your editor and fill in a real `OPENROUTER_API_KEY` locally.
3. Run the command below. No named Modal secret needs to be created.

Alternatively, set `OPENROUTER_API_KEY` in the current PowerShell session. Shell values
take precedence over `--env-file`. Only the selected credential is sent as a Modal
Secret; the file and its other entries are not uploaded or written into manifests.
The runner does not automatically read `compute.env` or change provider routing.
To use the repo-wide `compute.env`, pass `--env-file ../compute.env` explicitly.
A plain text key file is also supported: replace `--env-file .env` with `--key-file`
followed by the local file path. That file must contain only the key on one line;
the key is read directly and is not copied into the repo. A key file takes
precedence over a shell credential.

```powershell
# Four validation behaviors per model and per dataset (16 trajectories total).
.venv-agentharm/Scripts/python.exe agentharm_run.py --model both --dataset both --limit 4 --env-file .env

# Full public harmful split on Dolos, using the same local credential file:
.venv-agentharm/Scripts/python.exe agentharm_run.py --split test_public --limit 0 --env-file .env
```

Use `agentharm_run.py` as the launcher: it checks credentials and any named secret
before creating a Modal app, and enables UTF-8 output on Windows. This avoids app
creation rate limits from repeated runs with invalid setup. The original
`python -m modal run agentharm_modal.py ...` entry point is still supported.

If you already have a named Modal secret, replace `--env-file .env` with
`--secret-name` followed by its actual name. `agentharm-judge` is not created by
the harness. The launcher verifies that the selected key exists in the secret.

For another OpenRouter model, set `--judge-model openrouter/` followed by its
actual model ID (including the model provider, e.g. `openai/`). No endpoint or
key-variable flags are needed. Direct OpenAI remains possible with explicit
`--judge-model openai/gpt-4o-2024-08-06 --judge-api-key-env OPENAI_API_KEY
--judge-base-url https://api.openai.com/v1`.

Changing the judge changes the evaluation condition; record it when comparing
scores. Candidate weights are never used as the judge. Harmful and benign tasks
use their respective official rubric graders. Use benign results to distinguish
general tool-use capability from harmful-task compliance.

## Score an existing run without GPU inference

`agentharm_score.py` reads the saved `.eval` logs and applies the official harmful
or benign rubric and refusal scorer. It uses the OpenRouter judge by default.
Neither Dolos nor Qwen is loaded or called, and no Modal app is created.

```powershell
# All saved trajectories in the completed public run:
.venv-agentharm/Scripts/python.exe agentharm_score.py results/agentharm/20261025T152653Z-04cafde8 --env-file .env
```

You can instead pass `--key-file` and a local key file path, or set
`OPENROUTER_API_KEY` in your shell. `--limit 1` scores one sample per log as a
smoke test; omit the limit for all samples. `--dataset harmful|benign` selects a
subset. `--max-connections` defaults to four concurrent judge requests.

Outputs go in `RUN_DIR/scoring/<timestamp>-<unique suffix>/`: scored copies of
the Inspect logs, `summary.json` containing official metrics, and a manifest with
the judge configuration, package versions and hashes of the source logs. The
original inference logs are preserved. API failures in the refusal judge are
treated as scoring errors rather than silently counted as non-refusals. A guard
model raises if any code attempts candidate generation during replay.
Some official rubric graders assume required tool arguments exist and can crash
on malformed calls. Replay preserves these failures in `scoring_errors`, marks
their score/refusal as NaN/None, and excludes them from the official aggregates.
Always report `scored_samples` and the unscorable count with the resulting metrics.
After scoring, run `python agentharm_score_audit.py RUN_DIR/scoring/SCORING_ID`.
It makes no API requests. It additionally rejects semantic judge responses that
are not exactly `pass` or `fail`, then computes the official metric functions on
valid samples only. Use its `audited_summary.json` for reporting; raw Inspect
aggregate values can count a missing refusal as zero during score normalization.

## Options and protocol controls

`--model dolos|qwen|both`, `--dataset harmful|benign|both`, `--split val|test_public`,
and `--limit N` select work. `--limit 0` runs all matching samples, including
augmentations; it does not mean only base behaviors. `--behavior-ids ID1,ID2`
uses the benchmark's behavior-ID filter. `--detailed all|yes|no` and
`--hint all|yes|no` control augmentations. `--irrelevant-tools N` adds distractors.
Limits apply independently to every model/dataset evaluation.

Generation defaults match the upstream task: 4,096 output tokens, 20 messages,
temperature 0 and seed 0. Override with `--max-tokens`, `--max-messages`,
`--temperature`, `--seed` and `--epochs`. Runs are sequential for reproducibility
and bounded GPU memory. At temperature 0, repeated epochs will be deterministic.
Oversized contexts fail explicitly; they are not truncated silently.

Tool parsing is a meaningful experimental condition for Dolos:

* `--tool-parser strict` (default): require a complete JSON call and closing
  `</tool_call>` tag; malformed calls produce tool errors so the agent can recover.
* `--tool-parser json`: accept a complete JSON object after `<tool_call>` even
  when the closing tag is missing. Record this as a recovery condition.

Both preserve duplicate calls. Neither mode guesses arguments, fixes invalid JSON,
executes prose, forces the first tool call, or substitutes model weights. Inspect
messages are rendered through Qwen's native chat template on subsequent turns;
this necessarily canonicalizes parsed call history, as in the earlier audit.
Raw generations, rendered prompts, parse diagnostics and truncation reasons remain
in the Inspect logs. Compare parsing modes separately rather than pooling them.

## Artifacts and failures

Each invocation creates `results/agentharm/<UTC timestamp>-<unique suffix>/`:

* `request.json`: requested conditions, written locally before dispatch.
* `manifest.json`: conditions, installed package versions, model and dataset revisions.
* `logs/*.eval`: official Inspect logs, including raw model-call events.
* `traces.jsonl`: per-sample messages, metadata, scores and errors, without bulky events.
* `summary.json`: official aggregate metrics and completion/error status per condition.
* `error.txt`: traceback if the remote runner fails before completing.

The remote copy persists in Modal volume `dolos-agentharm-runs` under the same run
ID; weights and dataset cache use `dolos-hf-cache`. Artifacts are downloaded after
the remote call returns. If the local launcher disconnects, check the printed
Modal app link and use `modal volume get dolos-agentharm-runs RUN_ID LOCAL_PATH`
to recover the persisted files. The runner reports failed or incomplete samples
as a failed run while retaining available artifacts. A hard container timeout can
interrupt the final volume commit; completed Inspect logs are the primary record.

Open logs locally:

```powershell
.venv-agentharm/Scripts/python.exe -m inspect_ai view --log-dir results/agentharm/RUN_ID/logs
```

## Offline tests

```powershell
.venv-agentharm/Scripts/python.exe -m unittest discover -s tests -v
```

Tests cover tool parsing, message conversion, registration and a real Inspect
agent loop with malformed-call recovery and official scoring with a mock judge.
They need neither model weights nor
judge credentials; the GPU smoke command verifies model loading separately.
