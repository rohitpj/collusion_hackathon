"""AgentHarm on Modal: modal run agentharm_modal.py --help (run from redteaming/)."""

import io
import hashlib
import json
import os
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("dolos-agentharm")
cache = modal.Volume.from_name("dolos-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("dolos-agentharm-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==5.7.0", "huggingface-hub>=1.12,<2",
                 "safetensors>=0.7,<1", "inspect-ai==0.3.284", "inspect-evals==0.26.0", "openai>=3.4,<4")
    .env({"PYTHONUTF8": "1"})
    .add_local_file(HERE / "generate.py", "/root/generate.py")
    .add_local_file(HERE / "agentharm_adapter.py", "/root/agentharm_adapter.py")
    .add_local_file(HERE / "agentharm_runner.py", "/root/agentharm_runner.py")
)


@app.function(image=image, gpu="A10G", volumes={"/dolos-cache": cache, "/runs": runs}, timeout=4 * 60 * 60)
def evaluate(config, run_id):
    # Set cache locations only after volumes are mounted. Setting XDG_CACHE_HOME
    # on the image lets Modal's own bootstrap populate the mount before mounting.
    os.environ["HF_HOME"] = "/dolos-cache/hf"
    os.environ["XDG_CACHE_HOME"] = "/dolos-cache/xdg"
    from agentharm_runner import run

    directory = Path("/runs") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    error = None
    try:
        summaries = run(config, directory)
        if any(s["status"] != "success" or s["judge_errors"] or
               (s["results"] and s["results"]["completed_samples"] < s["results"]["total_samples"])
               for s in summaries):
            error = "One or more evaluations failed or contain sample errors. Inspect summary.json and logs."
    except Exception:
        error = traceback.format_exc()
        (directory / "error.txt").write_text(error, encoding="utf-8")
    finally:
        cache.commit()
        runs.commit()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in directory.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    return {"archive": buffer.getvalue(), "error": error}


def selection(value, allowed, name):
    if value == "both":
        return list(allowed)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, both")
    return [value]


def optional_bool(value):
    if value not in ("all", "yes", "no"):
        raise ValueError("behavior filters must be all, yes, or no")
    return None if value == "all" else value == "yes"


def preflight_judge(score, judge_model, judge_api_key_env, secret_name="", env_file="", key_file=""):
    """Validate local judge setup without creating an app or logging credentials."""
    if not score:
        return None
    if not judge_model or "JUDGE_MODEL_ID" in judge_model or "YOUR_" in judge_model:
        raise ValueError("--judge-model needs a real provider/model ID; JUDGE_MODEL_ID is a placeholder.")
    if judge_model.startswith(("dolos_hf/", "mockllm/")):
        raise ValueError("Scored runs need an independent judge, not Dolos/Qwen or a mock model.")
    if sum(bool(v) for v in (secret_name, env_file, key_file)) > 1:
        raise ValueError("Choose one of --secret-name, --env-file, or --key-file for judge credentials.")
    if secret_name:
        return None
    key = os.environ.get(judge_api_key_env)
    if key_file:
        path = Path(key_file).expanduser()
        if not path.is_file():
            raise ValueError(f"Judge key file does not exist: {path}")
        key = path.read_text(encoding="utf-8-sig").strip()
        if not key or any(c.isspace() for c in key):
            raise ValueError("The key file must contain only the API key on one line.")
    if env_file:
        from dotenv import dotenv_values
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise ValueError(f"Judge environment file does not exist: {path}")
        # Existing shell credentials take precedence; unrelated file entries are ignored.
        key = key or dotenv_values(path).get(judge_api_key_env)
    if not key or not key.strip():
        raise ValueError(
            f"No judge credential found for {judge_api_key_env}. "
            f"Set it in this PowerShell session, or put it in redteaming/.env and pass --env-file .env. "
            "Use --secret-name only for a secret you have already created. "
            "Use --no-score for trajectories without judge calls. See AGENTHARM.md."
        )
    return key


@app.local_entrypoint()
def main(model: str = "dolos", dataset: str = "harmful", split: str = "val", limit: int = 4,
         score: bool = True, judge_model: str = "openrouter/openai/gpt-4o-2024-08-06",
         judge_base_url: str = "https://openrouter.ai/api/v1", judge_api_key_env: str = "OPENROUTER_API_KEY", secret_name: str = "",
         tool_parser: str = "strict", max_tokens: int = 4096, max_messages: int = 20,
         temperature: float = 0.0, seed: int = 0, epochs: int = 1,
         behavior_ids: str = "", detailed: str = "all", hint: str = "all", irrelevant_tools: int = 0,
         out: str = "results/agentharm", env_file: str = "", key_file: str = ""):
    """limit=0 runs the full split. --no-score records trajectories without a judge/key."""
    judge_key = preflight_judge(score, judge_model, judge_api_key_env, secret_name, env_file, key_file)
    config = dict(
        modal_client_version=modal.__version__,
        harness_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in
                        ("agentharm_modal.py", "agentharm_adapter.py", "agentharm_runner.py", "generate.py")},
        models=selection(model, ("dolos", "qwen"), "model"),
        datasets=selection(dataset, ("harmful", "benign"), "dataset"),
        split=split, limit=limit, score=score, judge_model=judge_model,
        judge_base_url=judge_base_url, judge_api_key_env=judge_api_key_env,
        tool_parser=tool_parser, max_tokens=max_tokens, max_messages=max_messages,
        temperature=temperature, seed=seed, epochs=epochs,
        behavior_ids=[b.strip() for b in behavior_ids.split(",") if b.strip()],
        detailed_behaviors=optional_bool(detailed), hint_included=optional_bool(hint),
        irrelevant_tools=irrelevant_tools,
    )
    if split not in ("val", "test_public"):
        raise ValueError("split must be val or test_public")
    if tool_parser not in ("strict", "json"):
        raise ValueError("tool_parser must be strict or json")
    if limit < 0 or min(max_tokens, max_messages, epochs) < 1 or temperature < 0 or irrelevant_tools < 0:
        raise ValueError("invalid generation or sample limits")
    secrets = []
    if score:
        if secret_name:
            secrets.append(modal.Secret.from_name(secret_name, required_keys=[judge_api_key_env]))
        else:
            secrets.append(modal.Secret.from_dict({judge_api_key_env: judge_key}))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = Path(out) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "request.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run {run_id}: {model}, {dataset}, {split}, limit={limit or 'all'}, score={score}", flush=True)
    result = evaluate.with_options(secrets=secrets).remote(config, run_id)
    # Only our own generated archive is extracted, with a path check as a guard.
    with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(root):
                raise ValueError("unexpected path in result archive")
        archive.extractall(destination)
    print(f"Saved logs, traces and manifest: {destination.resolve()}")
    if result["error"]:
        raise RuntimeError(result["error"])
