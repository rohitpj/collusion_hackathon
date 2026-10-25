#!/bin/sh
# Spontaneous collusion test (Q1.1): nobody is told to form a ring (--planted_role neutral, --sizes 0).
# Cells: public chat only | + private DMs | + DMs and "may coordinate" cue | + DMs and "think about later jobs" cue.
# 6 seeds each. Run from procurement_sting in Git Bash:  sh run_spontaneous.sh
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # needed when output is piped (tee) on Windows
set -e
uv run python run_modal.py --max_minutes 70 --budget_usd 2.5 --sweep "--market tender --seeds 6 --sizes 0 --planted_role neutral --grid channels=public,public+dm --grid cue=,may_coordinate,horizon"
