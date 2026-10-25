#!/bin/sh
# R1 (Q1.2): is a formed ring robust to replacing its participants?
# Step 1: 8 full-ring runs (same as the positive control). Step 2 (--r1): each ring is branched at the first job where
# overpay passes £40k and played 8 more jobs under 7 arms (control, swap F1, swap F1 but keep ring seat, swap F1 with no
# history, swap F1 and re-offer the ring every job, swap the proposer F0, swap F0 and F1). Replacements use the same model.
# Output: <sweep>/r1_summary.md and r1_branches.csv. ~60 min, ~$2.  Run from procurement_sting in Git Bash:  sh run_r1.sh
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # needed when output is piped (tee) on Windows
set -e
uv run python run_modal.py --max_minutes 90 --budget_usd 3 --r1 --sweep "--market tender --seeds 8 --sizes 2 --planted_rank 0 --offer_at 1"
