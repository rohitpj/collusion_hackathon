"""vLLM server on Modal for the procurement sting: one open-weights instruct model behind an OpenAI-compatible endpoint.

Not run directly: `run_modal.py` starts it as an EPHEMERAL app around a sweep and tears it down afterwards
(the app also dies if the driver process dies, and the function has a hard timeout), so nothing is left billing.

Defaults are chosen for a small budget: Qwen2.5-32B-Instruct-AWQ (4-bit, ~20GB) on one L40S (48GB).
Override with env vars set BEFORE import: STING_MODEL, STING_GPU, STING_MAX_MIN, STING_API_KEY, STING_MAX_LEN.
UNTESTED on a live Modal account: the first launch is the smoke test (versions pinned from Modal's vLLM example).
"""
import os
import subprocess

import modal

MODEL = os.environ.get("STING_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
GPU = os.environ.get("STING_GPU", "L40S")
MAX_MIN = int(os.environ.get("STING_MAX_MIN", "90"))          # hard cap on the container's life
MAX_LEN = int(os.environ.get("STING_MAX_LEN", "8192"))
CONCURRENCY = int(os.environ.get("STING_CONCURRENCY", "48"))   # requests batched per container
PORT = 8000

app = modal.App("sting-vllm")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.9.1", "huggingface_hub[hf_transfer]==0.32.0")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
hf_cache = modal.Volume.from_name("sting-hf-cache", create_if_missing=True)   # weights downloaded once
secrets = [modal.Secret.from_dict({"VLLM_API_KEY": os.environ.get("STING_API_KEY", "")})]
if os.environ.get("STING_HF_SECRET"):    # only for gated models: modal secret create huggingface HF_TOKEN=...
    secrets.append(modal.Secret.from_name("huggingface"))


@app.function(image=image, gpu=GPU, timeout=MAX_MIN * 60, scaledown_window=600, secrets=secrets,
              volumes={"/root/.cache/huggingface": hf_cache})
@modal.concurrent(max_inputs=CONCURRENCY)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve():
    cmd = ["vllm", "serve", MODEL, "--served-model-name", MODEL, "--host", "0.0.0.0", "--port", str(PORT),
           "--max-model-len", str(MAX_LEN), "--gpu-memory-utilization", "0.90", "--max-num-seqs", str(CONCURRENCY),
           "--enable-prefix-caching", "--dtype", "half"]
    if os.environ.get("VLLM_API_KEY"):
        cmd += ["--api-key", os.environ["VLLM_API_KEY"]]
    subprocess.Popen(cmd)
