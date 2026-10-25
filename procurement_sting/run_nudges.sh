#!/bin/sh
# S3: indirect nudges toward collusion with no instruction to collude (Q1.1).
# Two sweeps, each cue none | profit_focus | industry | profit_industry:
#   1) spaced public costs 400/500/600k   2) EQUAL costs 500k for all firms
# Public chat + DMs, 30 jobs, private scratchpad, 4 seeds per cell = 32 runs. Run from procurement_sting in Git Bash:  sh run_nudges.sh
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # needed when output is piped (tee) on Windows
set -e
uv run python run_modal.py --max_minutes 90 --budget_usd 4 --sweep "--market tender --seeds 4 --sizes 0 --planted_role neutral --channels public+dm --n_auctions 30 --scratchpad true --grid cue=,profit_focus,industry,profit_industry" --sweep "--market tender --seeds 4 --sizes 0 --planted_role neutral --channels public+dm --n_auctions 30 --scratchpad true --grid cue=,profit_focus,industry,profit_industry --tender_cost_lo 500000 --tender_cost_hi 500000"
