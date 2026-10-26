"""Run a sweep against a vLLM model served on Modal, cost-capped.

  python run_modal.py --estimate --sweep "<run_sweep args>" [--sweep "<more args>"]     # no Modal needed, no spend
  python run_modal.py --max_minutes 45 --budget_usd 5 --sweep "<args>" --sweep "<args>"   # one Modal session, sweeps run in turn

Each --sweep (or everything after a bare `--`) is passed to run_sweep.py (backend, model, workers and -y are filled in). Needs `pip install modal`
and a one-time `modal setup` on the machine that runs this. The driver: starts the ephemeral Modal app, waits for the
model, points the sweep at it, tears everything down on exit, Ctrl-C or after --max_minutes, and prints a time x price estimate.
GPU prices below are approximate list prices in USD/hour (check the Modal dashboard for actuals).
"""
import argparse
import os
import shlex
import secrets
import sys
import threading
import time
import urllib.request
import _thread

PRICE = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}
COLD_START_MIN = 10      # first launch: image build + weight download + load (cheaper once the volume is warm)
SEC_PER_CALL = 4.0       # sequential latency of one firm call on a ~32B model, incl. prompt


def count_runs_calls(sweep_args):
    """Count runs and LLM calls from run_sweep's grid without running anything."""
    import run_sweep
    _, _, cfgs = run_sweep.parse(["--backend", "mock"] + sweep_args)
    per = [c.n_auctions * c.n_firms * (1 + c.comm_rounds) for c in cfgs]
    return len(cfgs), sum(per), max(per, default=0)


def estimate(sweeps, gpu, workers):
    runs = calls = 0
    wall = COLD_START_MIN
    for sw in sweeps:
        r, c, per_run = count_runs_calls(sw)
        runs, calls = runs + r, calls + c
        wall += (per_run * SEC_PER_CALL / 60) * max(1, -(-r // workers))
    return runs, calls, wall, wall / 60 * PRICE.get(gpu, 2.5)


def wait_ready(url, key, timeout=900):
    req = urllib.request.Request(url.rstrip("/") + "/v1/models", headers={"Authorization": f"Bearer {key}"})
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("model endpoint not ready")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--gpu", default="L40S")
    ap.add_argument("--max_minutes", type=int, default=45, help="hard wall-clock cap for the whole job")
    ap.add_argument("--budget_usd", type=float, default=8.0, help="refuse to start if the estimate exceeds this")
    ap.add_argument("--estimate", action="store_true", help="print the estimate and exit (no Modal needed)")
    ap.add_argument("--workers", type=int, default=8, help="sweep runs in parallel (more = fuller GPU batches)")
    ap.add_argument("--sweep", action="append", default=[], help="run_sweep args as one quoted string (repeatable)")
    ap.add_argument("--r1_spontaneous", action="store_true", help="with --r1: use the arms for a cartel formed without a ring seat")
    ap.add_argument("--r1", action="store_true", help="after the last sweep, branch every formed ring under the replacement arms (Q1.2)")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    sweeps = [shlex.split(x) for x in a.sweep]
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    if rest:
        sweeps.append(rest)
    if not sweeps:
        ap.error("give at least one --sweep")
    runs, calls, wall, cost = estimate(sweeps, a.gpu, a.workers)
    print(f"{runs} runs, ~{calls} LLM calls; estimated wall {wall:.0f} min on {a.gpu} "
          f"(${PRICE.get(a.gpu, '?')}/h) -> ~${cost:.2f} (approximate, includes {COLD_START_MIN} min cold start)")
    if a.estimate:
        return
    if cost > a.budget_usd:
        sys.exit(f"estimate ${cost:.2f} exceeds --budget_usd {a.budget_usd}; reduce seeds/conditions or raise the budget")

    key = "k" + secrets.token_urlsafe(24)   # a leading "-" makes vllm read the key as a flag
    os.environ.update(STING_MODEL=a.model, STING_GPU=a.gpu, STING_API_KEY=key,
                      STING_MAX_MIN=str(a.max_minutes + 5))
    import modal
    import modal_app
    t0 = time.time()
    timer = threading.Timer(a.max_minutes * 60, lambda: (print("\n[run_modal] max_minutes reached: stopping", flush=True),
                                                         _thread.interrupt_main()))
    timer.daemon = True
    timer.start()
    try:
        with modal.enable_output(), modal_app.app.run():
            url = modal_app.serve.get_web_url()
            print(f"[run_modal] endpoint up ({a.model} on {a.gpu}); waiting for the model...", flush=True)
            wait_ready(url, key)
            print(f"[run_modal] ready after {(time.time() - t0) / 60:.1f} min", flush=True)
            os.environ.update(LLM_BASE_URL=url.rstrip("/") + "/v1", LLM_API_KEY=key, LLM_MODEL=a.model)
            import run_sweep
            out = None
            for sw in sweeps:    # a sweep that errors must not stop the sweeps after it
                try:
                    out = run_sweep.main(["--backend", "llm", "--model", a.model, "--workers", str(a.workers), "-y"] + sw) or out
                except Exception as e:
                    print(f"[run_modal] sweep FAILED ({' '.join(sw)[:80]}): {type(e).__name__}: {str(e)[:160]}", flush=True)
            if a.r1 and out:
                try:
                    from sting import r1
                    r1.run_all(out, a.model, workers=a.workers, spontaneous=a.r1_spontaneous)
                except Exception as e:
                    print(f"[run_modal] r1 FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
    finally:
        timer.cancel()
        mins = (time.time() - t0) / 60
        print(f"[run_modal] torn down. Elapsed {mins:.1f} min, approx spend ${mins / 60 * PRICE.get(a.gpu, 2.5):.2f}")


if __name__ == "__main__":
    main()
