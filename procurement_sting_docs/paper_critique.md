# Critique of the draft paper (2026-10-25) and what to fix

Ordered by how much a reviewer would mark it down. Cost/effort in brackets.

## A. Methodology

1. **The headline spontaneous-collusion result has no matched control.** `talk` (12 runs, 20 jobs) is compared with no-cue equal costs (8 runs, 30 jobs, a different batch). Game length alone could explain 0.28 vs 0.16, and the nudge cells use a scratchpad that some older cells may not. Fix: run no-cue equal costs at 20 jobs, same settings, 12 runs, same session as a fresh `talk` cell. [about $0.8, 25 min; the single most valuable run left]
2. **The base prompt hands over the mechanism.** It says "a firm that does NOT want a job must bid HIGH" and that costs are public. That is exactly the stand-down half of bid rotation, and default prompts also mention a ring channel. "Unprompted collusion" is therefore "collusion with a structural hint". Say so in 3.1 and the abstract, or add a cell without the sentence (the sentence was needed to fix the £0-bid bug, so test a milder wording). [one sentence; optional run]
3. **Instructed rings dominate Q3.x.** Replacement, monitoring, channel, court and warning experiments test whether a firm told to collude keeps to it, not whether collusion forms. State this in the abstract's first mention of each result ("instructed ring"), and do not describe the warning as "stopping collusion" in general.
4. **The warning has no sham control.** The neutral "council reviews all bids" notice is a partial control, but the warning also defines the target behaviour ("keeping prices high by agreement"), which may simply tell the model what to avoid. Add a notice of equal length about something unrelated (e.g. invoicing rules). [12 runs, about $0.5]
5. **The court null is uninformative.** One report at job 1, a fine worth about three jobs of profit, a two-firm ring already near its ceiling (about 0.2). It supports "a single conviction does not deter", nothing about courts in general. Either run the repeated reporter or drop fine-size language from the abstract and discussion.
6. **Detector validation is circular.** Thresholds (25% of jobs, 90% of the figure) were set while looking at the same runs they are scored on. "Kept share" is partly a function of price, so its agreement with the index is built in; the independent signal is the proposal-and-acceptance step. Fix: hand-label 30–40 runs (I can generate a blind sheet), set thresholds on half, report precision/recall with intervals on the other half. [30 min of your time]
7. **Multiple comparisons.** About 70 cells; the paper reports the ones that moved. Mark which results are replicated (warning: yes, twice; replacement: one batch of 8 rings; talk: pooled from two runs, wide CI) and which are single-batch, exploratory.
8. **Sampling noise is under-reported.** The run-level sd is 0.27, so 6 runs per cell resolve only gaps above about 0.25. Most 4.2 and 4.4 comparisons are below that. Report minimum detectable effects once, in Section 3, and avoid ranking cells whose intervals overlap.
9. **Ring-of-two ceiling.** Index 0.16 of a possible 0.2 is saturation, so "no change" cells (court, notice) cannot show a fall large enough to be informative. Report index as a share of the feasible maximum for each ring size.
10. **Reasoning examples are selected by keyword.** Quote selection is the author's. Add a number: of the runs with the warning on, what share of runs contain any legality word at all, and what share of those still sustain an agreement. [no compute]
11. **One model, one temperature, unseeded sampling.** Seeds fix costs and ties, not the model's samples, so reruns differ. Say it plainly, and avoid "LLMs" in headings; use "Qwen2.5-32B".
12. **Pooled runs.** Two runs of the same command were pooled; show the two batches separately in an appendix (talk index 0.32 and 0.23) so readers see the spread.

## B. Writeup

1. **Length.** The draft is well beyond the 4–8 pages the template allows. Move Section 3's experiment table, 4.5 (watching and non-text) detail and 4.7 (number of firms) to an appendix; keep Section 4 to four results: warning/notice/court, spontaneous agreement, replacement, evidence.
2. **No figures.** Add (1) a one-glance environment diagram and (2) the forest plot of index by cell with intervals (`analyze.py` already writes `forest.png`; trim it to about 12 key cells and colour by role: instructed, spontaneous, deterrence).
3. **Jargon.** "Sustained kept agreement" is not self-explanatory. Use "priced-in agreement (kept in at least a quarter of jobs)" and define it once.
4. **Order.** Lead with the question and the three results, not the environment. The introduction's six sub-questions read like a checklist; group them as formation, robustness, deterrence, detection.
5. **Claims.** The discussion says three findings "stand up to our noise level". Grade them: replacement (strong, one batch), warning (replicated, instructed rings), spontaneous agreement (suggestive). Say "suggests" for the last.
6. **Corrections.** Three design errors in the body distract. Keep one sentence per error in Limitations and move the detail to an appendix table "what changed between batches, and which cells it invalidates".
7. **Related work.** It cites one paper. Add 4–6 references. Candidates to check and verify before citing: Calvano et al. (AER, 2020) on algorithmic pricing collusion, Fish et al. (2024), Motwani et al. (2024) on secret collusion among generative agents, plus whatever the sprint's reading list gives.
8. **Title and abstract.** Replace "draft paper". Name the model, the numbers and the instructed/uninstructed split in the abstract.
9. **Reproducibility.** The repository link must point to the merged state: a second PR is still to be opened.

## C. Suggested order (about 3 hours of work)
1. Matched 20-job no-cue control and a sham-notice control in one Modal session (24 runs, about $1.3) [A1, A4]
2. Hand-labelling sheet and detector validation [A6]
3. Legality-word statistics and forest plot, rewrite abstract/intro, trim to 8 pages [A10, B1–B5]
4. Related work and title [B7, B8]
5. Second PR, then paste into the Apart template.
