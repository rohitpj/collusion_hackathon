# All experiments: done, ready, and future (2026-10-24)

Cost figures are rough: about $0.03 per run all-in at 3 firms, at $1.95/h on the L40S. Spent so far: about $4.5 of $50.

## Done (live)
| ID | Question | Setup | Result |
|---|---|---|---|
| B | 1.1 null | neutral prompts, public chat only, 4 seeds | competitive, index -0.1 |
| PC | validity gate | full ring of 3, 4 seeds | after the prompt fix: index 0.85, compliance 0.99 |
| E2 | 3.2 | seeded k=1 + proposer, notice at job 1 vs none, 6 seeds | notice halves overpay (+£203k to +£100k) |
| E3 | 3.2, Bonus, 2.3 | trusted F2 report vs log x court_view bids vs all, 6 seeds | reporting halves the effect; court sees all upholds more |

## Ready to run (code built, script available or one command away)
| ID | Question | Setup | Runs | Cost |
|---|---|---|---|---|
| E1 | 3.2 | offer framing direct / indirect / benign from a scripted proposer (re-run after crash) | 18 | ~$1.5 |
| S1 | 1.1 | neutral, no ring: public vs public+dm x cue none / may_coordinate / horizon (`run_spontaneous.sh`) | 36 | ~$2.3 |
| S2 | 1.1 | S1 with 40 jobs, or a second model as one firm (`--planted_model`) | 12-24 | ~$1 |
| R1 | 1.2 | branch formed rings (PC or E2 no-notice): replace F1 with other model / same model no history / blank notes / dropped membership / replace proposer | ~24 branches of 8 jobs | ~$0.5 |
| W1 | 3.4 | seeded k=2, overseer announce_at 0/5 x scope public/all, optional gag | 24 | ~$1 |
| C1 | 3.1 | `--channels none` with ring seeded vs spontaneous; `--reveal` all / rounded / winner | ~30 | ~$1 |
| B1 | Bonus | warning x reporting x dashboard (2x2x2), seeded k=1 | 48 | ~$1.5 |
| CT1 | Bonus, 3.2, 2.3 | court: fine_mult 0/1/2.5/5; leniency on/off; court=rule vs llm x court_view; false_report_fine 0/50k/200k | ~78 | ~$2.5 |
| N1 | cross-cutting | n_firms 3/5/8 x ring size 0-3 (more calls per run at 5 and 8 firms) | ~48 | ~$3-4 |
| A1 | 2.3 | no new compute: pool runs into case types (collusion / talk without action / harm without agreement / competition); compare price rule, transcript judge, both | – | ~$0.3 judge calls |

Ready subtotal: about $15, leaving about $30 spare.

## Not built (needed for a full answer)
- LLM-driven trusted firm (current one is scripted): 3.2.
- Side payments and coded signals: 3.1.
- Tender likelihood-ratio detector: 2.3.
- Pair-level DM analysis script: 1.1.
- Paired-difference tables with confidence intervals: all.
- Verify who files the reports in E3 (`reports.jsonl`).

## Suggested order
1. S1 (spontaneous) and E1 re-run: closes 1.1 and 3.2.
2. B1 and CT1: Bonus and court.
3. R1 and W1: 1.2 and 3.4.
4. C1 and N1 if time allows. Build A1 and the CI tables while runs are going.
