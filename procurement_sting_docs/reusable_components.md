# What we can reuse from `collusion_hackathon/` for procurement_sting

Read: the READMEs of all repos plus the files named below (function lists and key passages, not every line).
Not read in full: `steganographic-collusion/colludebench/environments/auction.py` (a first-price auction),
`llm-trust-breaking/src/endogenous/core/trial_runner.py`, trace-guard monitor code.
Nothing here was run. All paths relative to `collusion_hackathon/`.

## Take (ranked by value for our gaps)

| # | Piece | File | What it does | How it plugs in |
|---|---|---|---|---|
| 1 | **Checkpoint branching** | `llm-cartel/branch_game.py` | Finds the round where a rolling market-gain crosses a threshold, rebuilds the exact state, continues under a modified condition (`--replace 1=<model>`, `--replacement_window` for how much history the newcomer sees, `--replace_prefix` to change its prompt). Prefix rounds are copied. | This is the 1.2 design in the review. We copy the idea: snapshot `history`, per-agent transcripts, RNG at the job where rolling overpay > threshold, then rerun from there with a swapped agent. Our `sim.run` needs to be split into `init_state / step(t) / snapshot`. Their own note: agents with no hidden state make "fresh same model" identical to no replacement, true for ours too, so the arms that differ are other model, other prompt, shorter history. |
| 2 | **Message overseer + sampling + gag order** | `llm-agent-collusion/src/continuous_double_auction/market/agents.py` (`MessageOverseer`), `.../prompt_templates/message_overseer_prompt.jinja2`, `market/market.py` (`sample_messages_for_overseer`, `overseer_sample_fraction`, `is_gag_order_active`) | An LLM reads sellers' messages each round and returns evidence phrases with labels (propose / maintain / threaten / withdraw, tone, content type) and a 0-4 `coordination_score`; a fraction of messages can be sampled; score 4 triggers a gag order that stops messaging. | Drop-in for 3.4 and the bonus. Our monitor = this prompt adapted to bids/jobs ("stand down", "rotate", "your turn"), run on the public channel (and optionally ring/DM) per job, coverage fraction as the 3.4 coverage factor, gag order as a countermeasure. Rewrite the Jinja prompt as an f-string (they use `render_prompt`). |
| 3 | **Judges for transcripts** | `llm-cartel/judges/api_judge.py` (1-10 collusion prompt, API based, scores a field of a jsonl every N rounds); `colosseum/experiments/collusion/judge_blackboards.py` (judge over channel transcripts, JSON extraction with fallback parse, per-run aggregate) | Score messages / reasoning for collusion. | For 2.3's transcript-only arm. Reuse the prompt text and robust JSON extraction (`_extract_json`, `_fallback_parse_rating`); our `messages.jsonl` already has the same shape (job, channel, sender, text). Use judge on public-only vs all channels. |
| 4 | **Collusion index normalised between competitive and cooperative price** | `llm-agent-collusion/.../evaluation/metrics.py` (`collusion_auc`, `competitive_equilibrium`, `jpm_price`, `ask_coordination_corr`, `gini`, `hhi`) | C(r) = (price - competitive)/(joint-profit-max - competitive), plus AUC. | Our equivalent: (price - competitive_price)/(cap - competitive_price), which fixes the "a ring of 2 barely moves the price" problem by also reporting the share of obedient-ring harm realised. Add to `metrics.py`. Also useful: correlation of bids across firms. |
| 5 | **Calibration + p-values + thresholds** | `steganographic-collusion/auditor/thresholds.py`, `pvalues.py`, `calibrate.py` | Build thresholds at a target FPR from honest runs (tail quantile), empirical p-values with add-one. | Our 2.3 detector needs the same: run the mock/honest auction many times (free), take the 1% FPR threshold for the log Bayes factor. Copy `empirical_p_value` and the tail-threshold logic (small, numpy only). |
| 6 | **Message interventions** | `steganographic-collusion/auditor/paraphrase.py`, `interventions.py` (selective paraphrase, rate limit, budget jitter, canary injection) | Perturb messages between agents to destroy or reveal covert signals. | For 3.1/3.2 countermeasures: paraphrase every DM before delivery, rate-limit messages per job, inject canary offers. Easy to wrap around our channel delivery in `sim.py`. |
| 7 | **Steganographic carriers for bids** | `steganographic-collusion/colludebench/covert_channels/` (`parity.py`, `keyword.py`, `bias.py`, `partition.py`, `sparse_keyword.py`, `watermark.py`) and `environments/auction.py` | Encoders/decoders that hide bits in action choices (parity of a bid, keyword in text, bias). | Reference for 3.1: what a bid-signalling scheme looks like (last-digit parity of the reservation) and how detectors read it. Use as a scripted baseline "colluder" to validate the 2.3 bid model catches it. Must read `auction.py` before relying on it. |
| 8 | **Detectors on message/action streams** | `steganographic-collusion/auditor/detectors/` (`mi.py`, `permutation.py`, `welfare.py`, `signal_bias.py`) and `mi_estimators.py` | MI between agents' signals, permutation invariance, welfare guard. | Optional second detector for 2.3/3.1: MI between firm A's bid residual and firm B's next-job residual (the review's 3.1 metric). `mi_estimators.py` is a neural estimator and heavy; for 5 firms x 12 jobs a plug-in or permutation-based estimate in our own code is enough. |
| 9 | **Matched-condition configs** | `colosseum/experiments/collusion/configs/hackathon_2_1_a..d_*.yaml` | Four matched conditions: pair + private channel, pair without, open channel, neither; positive-control config. | Template for 1.1's design. Their topology note (a complete graph merges blackboards, so keep a path) is worth copying: don't let "open channel" and "private channel" conditions collapse. Add a positive-control condition (seeded ring k=4 with scripted loyal members) which we already have. |
| 10 | **Collusion metrics for a coalition vs the rest** | `colosseum/experiments/collusion/metrics.py` (`compute_collusion_metrics`, regret-style) | Coalition advantage over non-coalition agents. | Pattern for our "ring profit vs outside profit" and planted agent profit; low priority. |
| 11 | **Deception typology (plan / announce / act)** | `llm-trust-breaking/src/endogenous/core/trial_runner.py`, `src/config/prompts/` | Each round: private plan, public announcement, binding act, then trust scores of others. | For emergent mode: ask each firm for a private plan before it speaks, and score trust in others after the job. Gives premeditation and persistence signals for 3.4 (does a watched ring still plan to stand down) and for compliance (announced vs acted). Costs one extra call per firm per job; pick 1.1/3.4 only. |
| 12 | **Memory pattern** | `llm-agent-collusion/.../market/agents.py` (`MemoryManager`, `scratch_pad_update`) | Per-agent hour-indexed memory string and a scratch pad the agent rewrites each round. | Our agents get only a shared result history. A per-agent notebook (scratch pad) is the cheap way to give firms persistent private reasoning ("F2 reliably stands down when told") that matters for repeated-game threats and for 1.2 newcomers (does a replacement inherit the notebook?). Copy the 20-line pattern. |
| 13 | **Parallel / queued LLM clients** | `llm-trust-breaking/src/llm/providers/parallel_openrouter_async.py`, `queued_openrouter.py` | Rate limit-aware async calls. | Only if sweeps get slow; `poseidon_compute.make_client` plus our thread pool should do for now. |

## Leave

- `algorithmic-pricing-llms`, `collusive-llm-agents`, `focal-points`, `llm-agent-collusion-2` (Liar's Bar / Cleanup): different games. `focal-points` is only useful as a benign baseline for talk-without-action cases in 2.3.
- `llm-cartel` steering/GPU scripts, warden judges: need GPUs or local weights.
- `trace-guard` (`traceguard/monitors/*`): single-agent monitors of code actions. The only transferable idea is `separation_of_duties_monitor.py` / `temporal_monitor.py` (structure for scoring several dimensions); too far from bidding for the time left.
- `colosseum` runtime (Terrarium framework): heavy dependency; take configs and judge/metrics ideas only.
- `redteaming/`: unrelated.

## Gap -> source map

| Gap in plan / review | Best source |
|---|---|
| Agent memory | #12 (scratch pad), #11 (plan/announce/act) |
| Private messages | #9 (matched channel conditions), existing DM/ring code |
| Monitors | #2 (overseer + sampling + gag order), #6 (interventions) |
| Transcript judges | #3, plus #2's labelled evidence format |
| Metrics / detection stats | #4, #5, #8 |
| Replacement / branching | #1 |
| Side channels | #7, #8 |

## Suggested minimum to take now
#1 (branching idea), #2 (overseer prompt + sampling), #3 (judge prompt + JSON parsing), #4 (collusion index), #5 (threshold code), #12 (scratch pad). That covers 1.2, 2.3, 3.4 and the bonus without new dependencies beyond numpy.
