#!/bin/sh
# Overnight queue, ~6 h, ~$12-14 of Modal credit. Each block is its own Modal session (a crash in one does not stop the rest).
# Run from procurement_sting in Git Bash:   sh run_night.sh 2>&1 | tee night.log
# Before starting: plug in, set sleep to Never, keep this terminal open.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # needed when output is piped (tee) on Windows
S="--market tender --planted_rank 0 --offer_at 1"
blk() { echo "=== $1 $(date) ==="; shift; "$@" || echo "!!! block failed, continuing"; }

# 1. S3b nudges toward explicit collusion (Q1.1)                                    ~85 min
blk nudges sh run_nudges_v2.sh
# 2. R1 replacement robustness (Q1.2)                                               ~60 min
blk r1 sh run_r1.sh
# 3. W1 watched coalition (Q3.4) + CT1 fines and leniency (Bonus, 3.2, 2.3)         ~75 min
blk watch_court uv run python run_modal.py --max_minutes 100 --budget_usd 4 \
  --sweep "$S --seeds 4 --sizes 2 --overseer true --grid overseer_announce_at=0,5 --grid overseer_scope=public,all" \
  --sweep "$S --seeds 4 --sizes 1 --reporting true --grid fine_mult=0,1,2.5,5" \
  --sweep "$S --seeds 4 --sizes 1 --reporting true --grid leniency=true,false"
# 4. B1 warning x reporting x dashboard (Bonus) + C1 no-chat and reveal modes (Q3.1) ~95 min
blk bonus_channels uv run python run_modal.py --max_minutes 120 --budget_usd 5 \
  --sweep "$S --seeds 4 --sizes 1 --grid warning=false,true --grid reporting=false,true --grid dashboard=false,true" \
  --sweep "$S --seeds 4 --sizes 2 --grid channels=none,public --grid reveal=all,winner"
# 5. N1 more firms (cross-cutting)                                                  ~70 min
blk nfirms uv run python run_modal.py --max_minutes 100 --budget_usd 4 \
  --sweep "$S --seeds 3 --sizes 0 1 2 --grid n_firms=3,5"
echo "=== done $(date) ==="
