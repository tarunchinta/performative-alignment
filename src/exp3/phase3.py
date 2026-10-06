"""Phase 3: Experiment 2 analogue with judge-defined utility (design section 6).

Three (optionally four) alignment conditions are trained on mixed A/B preference
data elicited under the reference deployment D_ref = softmax(v_hat):

1. ``rlhf`` -- pooled BT reward model + KL-regularized policy optimization.
2. ``vpl_inferred`` -- group-conditioned RMs with membership inferred by
   rater-level EM from pseudo-rater IDs (each persona's pairs are partitioned
   across ``n_raters_per_group`` rater IDs; EM assigns raters, not comparisons,
   matching the design's "EM-inferred membership from rater IDs").
3. ``vpl_oracle`` -- group-conditioned RMs with true persona labels.
4. ``deployment_conditioned`` (optional) -- population reward mixing Group A's
   RM with the Phase 1 surrogate U_B_hat(y; D_pi), solved by damped fixed-point
   iteration of the KL-regularized optimum.

Circularity discipline (design section 4): the surrogate appears only inside
optimization. Every *reported* utility -- both the numerator and denominator of
Utility Retention -- comes from fresh Group B judge queries under the deployment
context the evaluated policy actually induces (contextual-exposure format),
refit as mean-zero-anchored BT utilities. Retention = realized / optimum.

Construct-independent observables (entropy, effective support, JS to pi_ref)
reference only the policies, never the judge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.exp3.cache import ResponseCache
from src.exp3.elicitation import (
    Condition,
    PairwiseElicitor,
    sample_pairs,
    sample_pairs_mixed,
)
from src.exp3.judge import Judge
from src.exp3.menu import Menu
from src.exp3.phase2 import standardize_utility
from src.exp3.surrogate import SurrogateUtility, solve_deployment_consistent
from src.exp3.templates import TemplateName
from src.metrics import js_divergence
from src.policy import optimize_policy
from src.reward_model import BradleyTerryRewardModel, fit_bradley_terry, fit_bradley_terry_weighted
from src.vpl import VPLModel, fit_vpl_oracle

METHODS: tuple[str, ...] = ("rlhf", "vpl_inferred", "vpl_oracle", "deployment_conditioned")


@dataclass
class Phase3Config:
    n_train_pairs_per_group: int = 800
    n_eval_pairs: int = 400  # fresh pairs per evaluated policy (half policy-sampled)
    kl_coef: float = 0.5
    n_raters_per_group: int = 5
    em_iters: int = 20
    seeds: tuple[int, ...] = (0, 1, 2)  # pair-sampling seeds (design: >= 3)
    include_deployment_conditioned: bool = True
    n_repeats: int = 5
    k_exposure: int = 20
    max_concurrency: int = 32
    fp_iters: int = 60
    fp_damping: float = 0.5
    # "uniform" keeps the optimum inside the surrogate's measurement domain
    # (low per-item mass); "best" is the unconstrained cluster-argmax optimum.
    optimum_within_cluster: str = "best"


@dataclass
class Phase3Result:
    config: Phase3Config
    results: pd.DataFrame  # per (family, seed, method) + one "optimum" row per (family, seed)
    summary: pd.DataFrame  # mean/std retention per (family, method)
    policies: dict[tuple[str, int, str], np.ndarray]
    optima: dict[str, np.ndarray]  # family -> pi_star (surrogate fixed point)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _expand_repeats(repeats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each valid repeat is one binary comparison: (winners, losers, pair_idx)."""
    valid = repeats.dropna(subset=["picked_a"])
    picked_a = valid["picked_a"].to_numpy(dtype=bool)
    item_a = valid["item_a"].to_numpy(dtype=np.int64)
    item_b = valid["item_b"].to_numpy(dtype=np.int64)
    winners = np.where(picked_a, item_a, item_b)
    losers = np.where(picked_a, item_b, item_a)
    return winners, losers, valid["pair_idx"].to_numpy(dtype=np.int64)


def fit_vpl_inferred_raters(
    winners: np.ndarray,
    losers: np.ndarray,
    rater_ids: np.ndarray,
    n_items: int,
    group_a_fraction: float = 0.5,
    *,
    n_em_iters: int = 20,
) -> tuple[VPLModel, np.ndarray]:
    """Group-conditioned RMs via hard EM over *raters* (not single comparisons).

    Init: per-rater BT score vectors, split by the sign of the leading principal
    component. E-step: reassign each rater to the RM under which its comparisons
    are more likely. M-step: refit both RMs on their assigned raters' data.
    Returns the model and the final rater assignment (0/1 per unique rater).
    """
    winners = np.asarray(winners, dtype=np.int64)
    losers = np.asarray(losers, dtype=np.int64)
    rater_ids = np.asarray(rater_ids, dtype=np.int64)
    raters = np.unique(rater_ids)

    score_mat = np.stack(
        [
            fit_bradley_terry(winners[rater_ids == r], losers[rater_ids == r], n_items).score_all()
            for r in raters
        ]
    )
    centered = score_mat - score_mat.mean(axis=0, keepdims=True)
    u, _, _ = np.linalg.svd(centered, full_matrices=False)
    assign = (u[:, 0] > 0).astype(np.int8)
    if assign.sum() == 0 or assign.sum() == len(raters):
        assign[: len(raters) // 2] = 1 - assign[0]

    def _fit_side(side: int) -> BradleyTerryRewardModel:
        member = np.isin(rater_ids, raters[assign == side])
        return fit_bradley_terry(winners[member], losers[member], n_items)

    rm_0 = _fit_side(0)
    rm_1 = _fit_side(1)
    for _ in range(n_em_iters):
        s0, s1 = rm_0.score_all(), rm_1.score_all()
        ll0 = -np.logaddexp(0.0, -(s0[winners] - s0[losers]))
        ll1 = -np.logaddexp(0.0, -(s1[winners] - s1[losers]))
        new_assign = np.empty_like(assign)
        for i, r in enumerate(raters):
            mask = rater_ids == r
            new_assign[i] = 1 if ll1[mask].sum() > ll0[mask].sum() else 0
        if new_assign.sum() == 0 or new_assign.sum() == len(raters):
            break
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        rm_0 = _fit_side(0)
        rm_1 = _fit_side(1)

    return VPLModel(rm_a=rm_0, rm_b=rm_1, group_a_fraction=group_a_fraction), assign


def deployment_conditioned_policy(
    rm_a_scores: np.ndarray,
    surrogate: SurrogateUtility,
    pi_ref: np.ndarray,
    kl_coef: float,
    *,
    n_iters: int = 60,
    damping: float = 0.5,
    tol: float = 1e-10,
) -> np.ndarray:
    """Damped fixed point of pi = KL-opt(0.5 * r_hat_A + 0.5 * U_B_hat(.; pi))."""
    pi = np.asarray(pi_ref, dtype=np.float64).copy()
    for _ in range(n_iters):
        population_reward = 0.5 * rm_a_scores + 0.5 * surrogate.utility(pi)
        target = optimize_policy(standardize_utility(population_reward), pi_ref, kl_coef)
        nxt = damping * target + (1.0 - damping) * pi
        if np.abs(nxt - pi).sum() < tol:
            pi = nxt
            break
        pi = nxt
    return pi / pi.sum()


async def evaluate_policy_fresh(
    elicitor: PairwiseElicitor,
    menu: Menu,
    policy: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> dict:
    """Fresh Group B judgments under the policy's own induced deployment context.

    Fits mean-zero-anchored BT utilities u_hat and returns the realized expected
    utility E_{y~pi}[u_hat(y)]. The mean-zero anchor is the common reference that
    makes realized utilities comparable across contexts (BT fits are otherwise
    identified only up to a shift).
    """
    policy = np.asarray(policy, dtype=np.float64)
    policy = policy / policy.sum()
    pairs = sample_pairs_mixed(menu.n_items, n_pairs, rng, weight=policy)
    res = await elicitor.elicit_pairs(
        pairs, Condition(TemplateName.GROUP_B_CONTEXTUAL, distribution=policy)
    )
    model = fit_bradley_terry_weighted(
        res.pairs["item_a"].to_numpy(),
        res.pairs["item_b"].to_numpy(),
        res.pairs["p_prefer_a"].to_numpy(),
        menu.n_items,
        weight=res.pairs["n_votes"].to_numpy(),
    )
    u_hat = model.score_all()
    return {
        "realized": float(policy @ u_hat),
        "u_hat": u_hat,
        "n_pairs": int(len(pairs)),
        "parse_failure_rate": float(res.pairs["n_parse_fail"].sum())
        / max(1, int(res.pairs["n_parse_fail"].sum() + res.pairs["n_votes"].sum())),
    }


def _policy_observables(policy: np.ndarray, pi_ref: np.ndarray) -> dict:
    """Construct-independent observables: reference the policy only, never the judge."""
    p = np.asarray(policy, dtype=np.float64)
    p = p / p.sum()
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return {
        "entropy": entropy,
        "effective_support": float(np.exp(entropy)),
        "js_to_ref": js_divergence(p, pi_ref),
    }


async def run_phase3(
    judges: dict[str, Judge],
    menu: Menu,
    v_hat: dict[str, np.ndarray],
    surrogates: dict[str, SurrogateUtility],
    config: Phase3Config | None = None,
    *,
    cache: ResponseCache | None = None,
) -> Phase3Result:
    config = config or Phase3Config()
    methods = [m for m in METHODS if config.include_deployment_conditioned or m != "deployment_conditioned"]

    rows: list[dict] = []
    policies: dict[tuple[str, int, str], np.ndarray] = {}
    optima: dict[str, np.ndarray] = {}

    progress = tqdm(total=len(judges) * len(config.seeds), desc="phase3 (family x seed)")
    for fam_idx, (family, judge) in enumerate(judges.items()):
        v = standardize_utility(v_hat[family])
        pi_ref = _softmax(v)
        surrogate = surrogates[family]
        pi_star = solve_deployment_consistent(
            surrogate, menu, within_cluster=config.optimum_within_cluster
        )
        optima[family] = pi_star

        elicitor = PairwiseElicitor(
            judge,
            menu,
            n_repeats=config.n_repeats,
            k_exposure=config.k_exposure,
            max_concurrency=config.max_concurrency,
            cache=cache,
            seed=1000 + fam_idx,
        )

        for seed in config.seeds:
            rng = np.random.default_rng([seed, fam_idx])

            # --- Mixed 50/50 A/B training preferences under D_ref ---------------
            pairs_a = sample_pairs(menu.n_items, config.n_train_pairs_per_group, rng)
            res_a = await elicitor.elicit_pairs(pairs_a, Condition(TemplateName.GROUP_A))
            pairs_b = sample_pairs(menu.n_items, config.n_train_pairs_per_group, rng)
            res_b = await elicitor.elicit_pairs(
                pairs_b, Condition(TemplateName.GROUP_B_CONTEXTUAL, distribution=pi_ref)
            )

            w_a, l_a, pair_idx_a = _expand_repeats(res_a.repeats)
            w_b, l_b, pair_idx_b = _expand_repeats(res_b.repeats)
            winners = np.concatenate([w_a, w_b])
            losers = np.concatenate([l_a, l_b])
            group_labels = np.concatenate(
                [np.zeros(len(w_a), dtype=np.int8), np.ones(len(w_b), dtype=np.int8)]
            )
            # Pseudo-rater IDs: pairs partitioned round-robin within each persona.
            rater_ids = np.concatenate(
                [
                    pair_idx_a % config.n_raters_per_group,
                    config.n_raters_per_group + (pair_idx_b % config.n_raters_per_group),
                ]
            )

            # --- Alignment conditions -------------------------------------------
            # Reward vectors are standardized before KL-regularized optimization:
            # BT score scale tracks judge decisiveness, and the reward/KL trade-off
            # should not (retention and argmax are scale-invariant regardless).
            rm_pooled = fit_bradley_terry(winners, losers, menu.n_items)
            trained: dict[str, np.ndarray] = {
                "rlhf": optimize_policy(
                    standardize_utility(rm_pooled.score_all()), pi_ref, config.kl_coef
                )
            }

            vpl_oracle = fit_vpl_oracle(winners, losers, group_labels, menu.n_items, 0.5)
            trained["vpl_oracle"] = optimize_policy(
                standardize_utility(vpl_oracle.population_reward()), pi_ref, config.kl_coef
            )

            vpl_inf, _ = fit_vpl_inferred_raters(
                winners, losers, rater_ids, menu.n_items, 0.5, n_em_iters=config.em_iters
            )
            trained["vpl_inferred"] = optimize_policy(
                standardize_utility(vpl_inf.population_reward()), pi_ref, config.kl_coef
            )

            if config.include_deployment_conditioned:
                trained["deployment_conditioned"] = deployment_conditioned_policy(
                    vpl_oracle.rm_a.score_all(),
                    surrogate,
                    pi_ref,
                    config.kl_coef,
                    n_iters=config.fp_iters,
                    damping=config.fp_damping,
                )

            # --- Fresh-judge evaluation (numerator and denominator alike) --------
            eval_rng = np.random.default_rng([seed, fam_idx, 999])
            star_eval = await evaluate_policy_fresh(
                elicitor, menu, pi_star, config.n_eval_pairs, eval_rng
            )
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "method": "optimum",
                    "realized_utility": star_eval["realized"],
                    "utility_retention": 1.0,
                    "utility_gap": 0.0,
                    "surrogate_expected": surrogate.expected_utility(pi_star),
                    "fresh_surrogate_corr": float(
                        np.corrcoef(star_eval["u_hat"], surrogate.utility(pi_star))[0, 1]
                    ),
                    **_policy_observables(pi_star, pi_ref),
                }
            )
            policies[(family, seed, "optimum")] = pi_star

            for method in methods:
                pi_hat = trained[method]
                policies[(family, seed, method)] = pi_hat
                m_rng = np.random.default_rng([seed, fam_idx, methods.index(method)])
                ev = await evaluate_policy_fresh(
                    elicitor, menu, pi_hat, config.n_eval_pairs, m_rng
                )
                retention = (
                    ev["realized"] / star_eval["realized"]
                    if abs(star_eval["realized"]) > 1e-12
                    else float("nan")
                )
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "method": method,
                        "realized_utility": ev["realized"],
                        "utility_retention": float(retention),
                        # Shift-invariant deficit vs the optimum in judge logit
                        # units; the retention ratio is only meaningful when the
                        # optimum's anchored realized utility is solidly nonzero.
                        "utility_gap": float(ev["realized"] - star_eval["realized"]),
                        "surrogate_expected": surrogate.expected_utility(pi_hat),
                        "fresh_surrogate_corr": float(
                            np.corrcoef(ev["u_hat"], surrogate.utility(pi_hat))[0, 1]
                        ),
                        **_policy_observables(pi_hat, pi_ref),
                    }
                )
            progress.update(1)
    progress.close()

    results = pd.DataFrame(rows)
    method_rows = results[results["method"] != "optimum"]
    summary = (
        method_rows.groupby(["family", "method"], sort=False)
        .agg(
            retention_mean=("utility_retention", "mean"),
            retention_std=("utility_retention", lambda s: float(s.std(ddof=1))),
            gap_mean=("utility_gap", "mean"),
            gap_std=("utility_gap", lambda s: float(s.std(ddof=1))),
            realized_mean=("realized_utility", "mean"),
            entropy_mean=("entropy", "mean"),
            effective_support_mean=("effective_support", "mean"),
            js_to_ref_mean=("js_to_ref", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return Phase3Result(
        config=config,
        results=results,
        summary=summary,
        policies=policies,
        optima=optima,
    )
