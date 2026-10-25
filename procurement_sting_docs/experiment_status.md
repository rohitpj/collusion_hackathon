# Experiments: status and which question each answers (as of 2026-10-24)

Rohit's priority order: 3.2, 1.1, 1.2, 2.3, 3.4, 3.1, Bonus. All use `--market tender` (3 firms, public costs £400/500/600k, cap £1m, 20 jobs), Qwen2.5-32B on Modal.

## Done (live)
| Experiment | Command core | Answers | Result |
|---|---|---|---|
| Baseline: no encouragement | `--sizes 0 --planted_role neutral --channels public` (4 seeds) | 1.1 step 0; null for everything | Competitive: collusion index -0.08 to -0.12 |
| Positive control: full ring | `--sizes 2 --planted_rank 0 --offer_at 1` (4 seeds) | Validity gate for all ring experiments | After prompt fix: overpay +£423k, agreed winner wins 93%, compliance 0.99, index 0.85, detector flags 100% |

## Ready to run (code built, not yet run live)
| Experiment | Question |
|---|---|
| E1 offer framing: direct / indirect / benign / none, scripted trusted proposer | 3.2 |
| E2 deterrence by announcement (`notice_at`) | 3.2 |
| E3 trusted reporting firm: report vs log, court_view bids vs all | 3.2, 2.3 |
| 1.1 lever ladder: +dm, `cue may_coordinate`, `cue horizon`, 40 jobs, other model | 1.1 |
| Branching: replace F1 with other model / no history / blank notes / dropped membership / replace proposer | 1.2 |
| Case-type pooling (collusion / talk without action / harm without agreement / competition), rule detector, judge scope public vs all | 2.3 |
| Overseer: announce_at x scope x sample, gag order; message migration public -> dm/ring | 3.4 |
| `--channels none` with ring seeded vs spontaneous; `reveal all/rounded/winner` | 3.1 (bids-only conduit) |
| Warning x reporting x dashboard 2x2x2 | Bonus |
| Court: fine_mult 0,1,2.5,5; leniency on/off; court_view bids/all x rule/llm; false_report_fine; spam reports | Bonus, 3.2, 2.3 |
| N sweep: n_firms 3,5,8 x ring size | Cross-cutting (does collusion fall with more firms) |

## Not built
LLM-driven trusted firm (current one is scripted); side payments / coded signals (full 3.1); tender likelihood-ratio detector (2.3); pair-level DM analysis script (1.1); paired-difference tables with CIs.

## Suggested order for remaining budget (about $48 left)
1. Court arms on the validated ring (Bonus + 3.2 E3): ~$2.
2. 1.1 lever ladder and E1 (3.2, 1.1): ~$4.
3. Branching on formed rings (1.2): ~$2.
4. Overseer (3.4) and channels none / reveal (3.1): ~$4.
5. N sweep: ~$6.
6. Build the missing analysis (2.3) while runs are going.
