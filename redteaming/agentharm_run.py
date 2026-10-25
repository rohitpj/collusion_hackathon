"""Check judge setup BEFORE starting Modal. Pass the same options as agentharm_modal.py."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--score", dest="score", action="store_true", default=True)
    parser.add_argument("--no-score", dest="score", action="store_false")
    parser.add_argument("--judge-model", default="openrouter/openai/gpt-4o-2024-08-06")
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--secret-name", default="")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--key-file", default="")
    args, _ = parser.parse_known_args(argv)
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if "--help" not in argv and "-h" not in argv:
        from agentharm_modal import preflight_judge
        try:
            key = preflight_judge(args.score, args.judge_model, args.judge_api_key_env,
                                  args.secret_name, args.env_file, args.key_file)
            if args.score and args.secret_name:
                import modal
                # Resolve this before AppCreate; a typo must not consume an app start.
                modal.Secret.from_name(args.secret_name, required_keys=[args.judge_api_key_env]).hydrate()
        except Exception as exc:
            print(f"AgentHarm setup: {exc}", file=sys.stderr)
            return 2
        if key:
            environment[args.judge_api_key_env] = key
    command = [sys.executable, "-m", "modal", "run",
               str(Path(__file__).with_name("agentharm_modal.py")), *argv]
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
