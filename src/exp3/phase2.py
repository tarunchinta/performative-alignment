"""Phase 2: Experiment 1 analogue with judge-derived quantities (design section 5).

Mirrors Section 4.2 of the paper: a static BT reward model r_hat is fit from
preferences elicited under the reference deployment D_ref = softmax(v_hat), then
evaluated against *fresh* judgments as the deployment interpolates from pi_ref
toward the concentrated pi_target = softmax(v_hat / tau).

Personas are carried by the prompt template, not the judge backend:

- ``group_b`` uses the feed-curator persona with contextual exposure (2.2) --
  the deployment-dependent group whose local accuracy is predicted to decline.
- ``group_a`` uses the intrinsic-quality persona *with the exposure block
  present* (template 2.5), so both personas see identical deployment context
  and the contrast isolates the persona, not the information channel. Its
  prediction is no accuracy decline at any t (negative control).

The claim tested is local-not-global degradation: global Spearman rho of r_hat
stays high while pairwise accuracy on pairs drawn from the top-mass region of
pi_t declines with t (for group B only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
from src.exp3.templates import TemplateName
from src.metrics import js_divergence
from src.reward_model import fit_bradley_terry_weighted

PERSONA_TEMPLATES: dict[str, TemplateName] = {
    "group_a": TemplateName.GROUP_A_CONTAMINATION,
    "group_b": TemplateName.GROUP_B_CONTEXTUAL,
}


@dataclass
class Phase2Config:
    t_grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    # pi_target = softmax(transform(v_hat) / tau). Note: because clusters are
    # quality-balanced by design, mild tau barely moves cluster prevalences (the
    # balance suppresses the manipulation); tau must concentrate hard enough that
    # top clusters gain real mass. tau=0.2 gives a several-logit penalty spread
    # at typical lambda_hat while keeping ~10 items in the 80%-mass local set.
    target_temperature: float = 0.2
    # "standardize": transform(v) = (v - mean)/sd. "quantile": rank-percentile in
    # [0, 1] -- use when the judge's v_hat has extreme outliers (a decisive judge
    # can put its top item several sd out, collapsing softmax targets to a single
    # item at any temperature; ranks degrade gracefully). Quantile taus live on a
    # different scale: ~0.05-0.1 rather than ~0.2-0.35.
    target_transform: str = "standardize"
    n_train_pairs: int = 800  # r_hat training elicitation under D_ref
    n_pairs_per_t: int = 400  # fresh judgments per t (half concentrated, half global)
    local_mass: float = 0.8  # top-mass region of pi_t defining "local" pairs
    # Headline accuracy counts only pairs the fresh judge is confident about
    # (|p_prefer_a - 0.5| >= margin); near-tie pairs are a vote-noise floor that
    # dilutes the miscalibration signal identically at every t.
    confident_margin: float = 0.3
    n_repeats: int = 5
    k_exposure: int = 20
    max_concurrency: int = 32
    seed: int = 202


@dataclass
class Phase2Result:
    config: Phase2Config
    results: pd.DataFrame  # per (family, persona, t)
    rm_scores: dict[tuple[str, str], np.ndarray]  # (family, persona) -> r_hat
    policies: dict[str, dict[str, np.ndarray]]  # family -> {pi_ref, pi_target}


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def standardize_utility(v: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-sd rescale of v_hat before building D_ref = softmax(v).

    BT utilities are only scale-calibrated up to the judge's decisiveness: a
    near-deterministic judge yields hugely inflated scores whose raw softmax is
    a delta function. Standardizing makes the reference deployment's shape
    comparable across judge families.
    """
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-12)


def top_mass_mask(distribution: np.ndarray, mass: float) -> np.ndarray:
    """Boolean mask of the smallest item set covering ``mass`` of the distribution."""
    distribution = np.asarray(distribution, dtype=np.float64)
    order = np.argsort(distribution)[::-1]
    cum = np.cumsum(distribution[order])
    k = int(np.searchsorted(cum, mass) + 1)
    mask = np.zeros(len(distribution), dtype=bool)
    mask[order[:k]] = True
    return mask


def rm_fresh_accuracy(
    rm_scores: np.ndarray,
    pairs: pd.DataFrame,
    item_mask: np.ndarray | None = None,
    min_margin: float = 0.0,
) -> tuple[float, int]:
    """Pairwise accuracy of r_hat against fresh judge majority preferences.

    Ties (p_prefer_a == 0.5), pairs below the confidence ``min_margin``, and
    parse-failed pairs are excluded; with ``item_mask``, only pairs with both
    items inside the mask count (the local split).
    """
    df = pairs.dropna(subset=["p_prefer_a"])
    df = df[np.abs(df["p_prefer_a"] - 0.5) >= max(min_margin, 1e-9)]
    if item_mask is not None:
        keep = item_mask[df["item_a"].to_numpy()] & item_mask[df["item_b"].to_numpy()]
        df = df[keep]
    if len(df) == 0:
        return float("nan"), 0
    pred = rm_scores[df["item_a"].to_numpy()] > rm_scores[df["item_b"].to_numpy()]
    truth = df["p_prefer_a"].to_numpy() > 0.5
    return float((pred == truth).mean()), int(len(df))


def _fit_weighted_bt(pairs: pd.DataFrame, n_items: int) -> np.ndarray:
    model = fit_bradley_terry_weighted(
        pairs["item_a"].to_numpy(),
        pairs["item_b"].to_numpy(),
        pairs["p_prefer_a"].to_numpy(),
        n_items,
        weight=pairs["n_votes"].to_numpy(),
    )
    return model.score_all()


async def run_phase2(
    judges: dict[str, Judge],
    menu: Menu,
    v_hat: dict[str, np.ndarray],
    config: Phase2Config | None = None,
    *,
    cache: ResponseCache | None = None,
) -> Phase2Result:
    config = config or Phase2Config()

    rows: list[dict] = []
    rm_scores: dict[tuple[str, str], np.ndarray] = {}
    policies: dict[str, dict[str, np.ndarray]] = {}

    n_steps = len(judges) * len(PERSONA_TEMPLATES) * (1 + len(config.t_grid))
    progress = tqdm(total=n_steps, desc="phase2 steps")
    for fam_idx, (family, judge) in enumerate(judges.items()):
        v = standardize_utility(v_hat[family])
        pi_ref = _softmax(v)
        if config.target_transform == "quantile":
            n = len(v)
            target_v = np.argsort(np.argsort(v)) / (n - 1)
        elif config.target_transform == "standardize":
            target_v = v
        else:
            raise ValueError(f"Unknown target_transform {config.target_transform!r}")
        pi_target = _softmax(target_v, temperature=config.target_temperature)
        policies[family] = {"pi_ref": pi_ref, "pi_target": pi_target}

        elicitor = PairwiseElicitor(
            judge,
            menu,
            n_repeats=config.n_repeats,
            k_exposure=config.k_exposure,
            max_concurrency=config.max_concurrency,
            cache=cache,
            seed=config.seed + fam_idx,
        )

        for persona_idx, (persona, template) in enumerate(PERSONA_TEMPLATES.items()):
            rng = np.random.default_rng([config.seed, fam_idx, persona_idx])

            # Static reward model: elicited under the reference deployment.
            train_pairs = sample_pairs(menu.n_items, config.n_train_pairs, rng)
            train_res = await elicitor.elicit_pairs(
                train_pairs, Condition(template, distribution=pi_ref)
            )
            r_hat = _fit_weighted_bt(train_res.pairs, menu.n_items)
            rm_scores[(family, persona)] = r_hat
            progress.update(1)

            for t in config.t_grid:
                pi_t = (1.0 - t) * pi_ref + t * pi_target
                pi_t = pi_t / pi_t.sum()

                eval_pairs = sample_pairs_mixed(
                    menu.n_items, config.n_pairs_per_t, rng, weight=pi_t
                )
                res = await elicitor.elicit_pairs(
                    eval_pairs, Condition(template, distribution=pi_t)
                )

                local = top_mass_mask(pi_t, config.local_mass)
                margin = config.confident_margin
                global_acc, n_global = rm_fresh_accuracy(r_hat, res.pairs, min_margin=margin)
                local_acc, n_local = rm_fresh_accuracy(
                    r_hat, res.pairs, item_mask=local, min_margin=margin
                )
                global_acc_all, _ = rm_fresh_accuracy(r_hat, res.pairs)
                local_acc_all, _ = rm_fresh_accuracy(r_hat, res.pairs, item_mask=local)

                u_t = _fit_weighted_bt(res.pairs, menu.n_items)
                rho, _ = spearmanr(r_hat, u_t)

                rows.append(
                    {
                        "family": family,
                        "persona": persona,
                        "t": float(t),
                        "js_to_ref": js_divergence(pi_t, pi_ref),
                        "global_accuracy": global_acc,
                        "n_global_pairs": n_global,
                        "local_accuracy": local_acc,
                        "n_local_pairs": n_local,
                        "global_accuracy_all": global_acc_all,
                        "local_accuracy_all": local_acc_all,
                        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                        "local_set_size": int(local.sum()),
                    }
                )
                progress.update(1)
    progress.close()

    return Phase2Result(
        config=config,
        results=pd.DataFrame(rows),
        rm_scores=rm_scores,
        policies=policies,
    )
