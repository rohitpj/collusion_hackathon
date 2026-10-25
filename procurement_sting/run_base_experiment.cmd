@echo off
rem Base experiment on Modal (~$1 estimated). Run from the procurement_sting folder after `uv sync` and `uv run modal setup`.
rem Arm A: positive control, all 3 firms in a ring told to coordinate. Arm B: no-encouragement baseline, public chat only.
uv run python run_modal.py --max_minutes 45 --budget_usd 4 ^
  --sweep "--market tender --seeds 4 --sizes 2 --planted_rank 0 --offer_at 1" ^
  --sweep "--market tender --seeds 4 --sizes 0 --planted_role neutral --channels public"
