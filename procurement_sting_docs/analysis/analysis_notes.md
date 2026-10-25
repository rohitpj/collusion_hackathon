# Pooled analysis of all runs so far (analyze.py: 344 runs, 65 cells; unit = run, 95% bootstrap CIs)

Files: cells.csv (every cell with CI), forest.png (all cells), detector.csv, noise.csv, runs.csv (run level).

## Q2.3: observer features vs ground truth (60 competitive runs vs 32 instructed full rings)
| feature | competitive mean | ring mean | AUC |
|---|---|---|---|
| mean overpay | -£34k | +£435k | 1.00 |
| share of jobs priced above benchmark + one step | 0.10 | 0.92 | 1.00 |
| cheapest firm fails to win (spaced costs only) | 0.21 | 0.74 | 1.00 |
| shipped rule flag (price above benchmark + step in >= 2 jobs) | 0.47 | 1.00 | 0.77 |
The shipped rule has a 47% false-positive rate on competitive runs (true-positive 100%), because 2 noisy jobs trip it. Using the share of jobs priced above benchmark + step with a cut near 0.3, or the mean overpay, separates these cases perfectly. Caveat: this is the easy contrast (instructed rings of 3 against neutral firms); partial markups (equal costs, rings of 2) are the hard case and are not covered.

## Noise
Seeded k=1 cells with an LLM third firm (E2 no notice, CT1 fine_mult and leniency cells, B1 no-warning cells) have run-level index sd of 0.27 (cell means 0.10-0.46 at 4 seeds). The spread is mostly bimodal: whether the third firm joins. Court2 cells (third firm scripted) are tight (0.14-0.17, sd about 0.03). At sd 0.27 a 0.2 difference needs about 28 runs per arm for 80% power; with 4-8 seeds only effects of 0.4+ are detectable in those configurations.
