# Experiment 3 — Live Results (judge: gpt-5.4 via Azure AI Foundry)

Run dates: 2026-07-07 → 2026-07-08. All raw judge responses in
`data/exp3_cache/gpt-5.4.jsonl` (~236k calls; every analysis below replays from
this log without re-querying). Menu frozen at revision 6 of
`data/exp3_menu.json` after the Phase 0 regeneration loop.

## Phase 0 — menu validation (passed after 6 menu revisions)

- The original menu failed quality balance badly under a real craft judge
  (technical z = −1.21, sensory_concrete +1.57): "style" clusters written as
  low-craft instances read as quality differences. Items were iteratively
  rewritten to craft-match within style (50+ items touched).
- Final gate (pooled over two independent 800-pair samples, seeds 7+8):
  **max cluster |z| = 0.442 < 0.5 → passed.** Single-seed z estimates carry
  ±0.15–0.2 sampling noise; the pooled fit is the recorded decision basis.
- Separability: cluster recoverability 0.48–0.51 vs 0.10 chance (passed).
- Prompt checklist: parse failures 0.000 everywhere; position bias 0.46–0.54;
  repeat agreement 0.97–0.99 no-context, 0.73–0.77 contextual.

## Phase 1 — judge psychometrics (GO)

| format | cluster 1 | cluster 5 | cluster 8 |
|---|---|---|---|
| contextual exposure | **+5.82** [5.29, 7.96]* | **+5.50** [5.24, 8.10]* | **+5.47** [5.13, 6.43]* |
| stated prevalence | −1.77 [−5.20, 7.61] | +3.90 [−5.34, 8.96] | +3.72 [2.37, 8.61]* |
| placebo (control) | −2.17 [−5.49, 1.04] | +0.16 [−4.35, 2.37] | +3.85 [0.43, 7.21]* |
| Group A contamination | +4.61 [1.14, 8.18]* | +0.12 [−3.01, 3.94] | +3.36 [−0.40, 8.12] |

- **gpt-5.4 is prevalence-averse under naturalistic contextual exposure**: all
  three probe clusters significant, consistent sign, linear best form.
- Contextual ≫ stated (the demand-effect-favorable direction: an instruction-
  following artifact would produce the reverse).
- Controls mostly null; one placebo and one contamination cell fire at 95%
  (compatible with noise across 6 control cells; flag in the writeup — the
  contamination cell means exposure alone may shift even the intrinsic judge
  somewhat, though at roughly half the Group B magnitude).
- Gate: GO (single family). The pre-registered ≥2-family gate needs a second
  deployed model (`crossfamily` subset, ~15k calls).

## Surrogate (design §4)

`Û_B(y; D) = 0.086·v̂(y) − 3.82·max(D(cluster(y)) − 0.05, 0)` — threshold form,
statistically tied with linear (held-out LL −0.6345 vs −0.6347); both clearly
beat the prevalence-free baseline (−0.6666).

Magnitudes (computed from `v_hat.npy` + `surrogate.json`):

| quantity | value |
|---|---|
| intrinsic span, full v̂ range (35.46 logits) | 0.086 × 35.46 = **3.05** |
| intrinsic span, central 90% (11.60 logits) | 0.086 × 11.60 = **1.00** |
| penalty at full cluster concentration (D=1) | 3.82 × 0.95 = **3.63** |

So the penalty at full concentration slightly *exceeds* the menu's entire
intrinsic-quality range (1.19×) and is 3.6× the spread among typical items —
the full range is inflated by v̂ outliers (top item ≈ 5σ). This judge cares at
least as much about staleness as about craft.

## Phase 2 — static-RM miscalibration under deployment shift

Quantile-transformed target (τ=0.07) — required because gpt-5.4's v̂ outliers
(top item ≈ 5σ) collapse plain softmax targets to one item; earlier runs at
τ=0.2/0.35 (standardized) are kept as `phase2_results_tau*.csv`.

| metric (t=0 → t=1) | Group B | Group A (control) |
|---|---|---|
| global accuracy | 0.934 → **0.768** | 0.945 → 0.897 |
| Spearman ρ | 0.812 → **0.587** | 0.887 → 0.909 |
| local accuracy | 0.879 → 0.857 (flat) | 0.830 → 0.714 (n=42, marginal) |

- The Group-B-specific degradation reproduces on a real judge (global acc
  −0.17, ρ −0.23 vs. control −0.05 / flat).
- **Geometry refinement vs the toy model:** with a *cluster-level* penalty,
  items inside the concentrated region share the penalty, so within-local order
  survives; the miscalibration lands on **cross-region** comparisons (penalized
  popular items vs fresh mediocre items). Experiment 1's "local-not-global"
  pattern comes from its item-level penalty; the live analogue is
  "cross-region-not-within-region". Section 4.7 should state the deployment-
  dependent degradation claim in this form.

## Phase 3 — alignment methods under judge-defined utility

Fresh-judge evaluation under each policy's own induced context, mean-zero BT
anchor; 3 pair-sampling seeds. The **utility gap** (realized − optimum, judge
logits; shift-invariant) is the headline statistic — the retention *ratio* is
not usable here because the anchored optimum sits near 0.

| condition | utility gap (mean ± sd) | eff. support | JS to π_ref |
|---|---|---|---|
| optimum (in-domain, uniform-within-cluster) | 0 (realized ≈ +0.03) | 99.0 | 0.27 |
| standard RLHF | **−0.80 ± 0.37** | 22.5 | 0.21 |
| VPL (inferred raters) | **−1.28 ± 0.87** | 19.0 | 0.18 |
| VPL (oracle labels) | **−1.38 ± 0.79** | 19.0 | 0.18 |
| deployment-conditioned (cluster-level surrogate) | **−8.43 ± 0.91** | 5.1 | 0.08 |

Three findings:

1. **Shared collapse + oracle failure reproduce**: every method realizes
   utility *below the menu average* under its own deployment (negative anchored
   realized utility), and oracle group labels do not help (−1.38 vs inferred
   −1.28; both ≤ standard RLHF).
2. **Item-level repetition aversion dominates at concentrated policies**: the
   unconstrained cluster-argmax "optimum" realized −3.8 — *worse than RLHF* —
   because a 10-item policy shows the judge the same sentences verbatim in its
   exposure block, and the judge punishes that far beyond the cluster-level
   model (that run preserved as `phase3_*_bestoptimum.csv`). The
   deployment-consistent optimum was therefore recomputed over the design's
   10-dim cluster-mix class (uniform within cluster, per-item mass ≤ ~0.03,
   inside the surrogate's measured domain); it is ≈ the maximum-diversity
   policy and beats every trained policy, consistently across seeds.
3. **Misspecified deployment-awareness is the worst condition**: optimizing
   against the cluster-level surrogate concentrates hard (support ≈ 5) and
   realizes −8.4 — deployment-aware training with the wrong deployment model
   amplifies the failure rather than mitigating it.

## Protocol deviations / limitations to report

- Judge temperature fixed at 1 (gpt-5.4 rejects 0.7); preference probabilities
  still from 5-repeat vote fractions with order swap.
- Single judge family (only gpt-5.4 deployed on the resource); the ≥2-family
  go/no-go gate is unmet. Deploy e.g. claude-haiku-4-5 or Llama-3.3-70B and run
  `phase1 --subset crossfamily` + `gate` for the pre-registered decision.
- π_ref/π_target constructed from standardized (and, for Phase 2 targets,
  quantile-transformed) v̂ — required by the judge's near-deterministic votes;
  both transforms validated against the synthetic pipeline.
- The quality-balance loop used the judge itself (v̂) to rebalance the menu;
  balance is judge-relative, as the design intends, but should be re-checked
  per family.

## Artifact inventory

- `phase0_report.json`, `v_hat.npy` (+meta) — menu gates, pooled v̂
- `phase1_summary.csv`, `phase1_curves.csv`, `phase1_raw.csv`, `go_no_go.json`
- `surrogate.json`, `surrogate_metrics.csv`
- `phase2_results.csv` (quantile τ=0.07; `_tau0.2/_tau0.35` = earlier runs)
- `phase3_results.csv`, `phase3_summary.csv` (`_bestoptimum` = out-of-domain optimum run)
- `exp3_live_psychometrics.png`, `exp3_live_phase3_gaps.png`
- `data/exp3_cache/gpt-5.4.jsonl` — full replayable response log (supplementary material)
