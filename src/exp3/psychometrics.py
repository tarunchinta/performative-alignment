"""Phase 1 judge psychometrics: empirical sensitivity, functional form, consistency.

From per-prevalence elicitation data (all under one condition), fit BT utilities
u_hat(y; p), estimate the empirical sensitivity lambda_hat via difference-in-
differences (BT utilities are identified only up to shift, so the manipulated
cluster is measured relative to the rest), attach a bootstrap CI over pair
resampling, compare linear/log/threshold forms of the prevalence penalty by AIC,
and report repeat-agreement / position-bias consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np
import pandas as pd

from src.reward_model import fit_bradley_terry_weighted


def fit_scores_by_p(
    pairs_by_p: dict[float, pd.DataFrame], n_items: int
) -> dict[float, np.ndarray]:
    """Fit a weighted BT utility vector u_hat(.; p) at each prevalence level."""
    scores: dict[float, np.ndarray] = {}
    for p, df in pairs_by_p.items():
        model = fit_bradley_terry_weighted(
            df["item_a"].to_numpy(),
            df["item_b"].to_numpy(),
            df["p_prefer_a"].to_numpy(),
            n_items,
            weight=df["n_votes"].to_numpy(),
        )
        scores[p] = model.score_all()
    return scores


def _delta(scores: np.ndarray, probe_items: np.ndarray, other_items: np.ndarray) -> float:
    """DiD contrast: mean u_hat over manipulated cluster minus mean over the rest."""
    return float(scores[probe_items].mean() - scores[other_items].mean())


def lambda_hat_point(
    scores_by_p: dict[float, np.ndarray],
    probe_items: np.ndarray,
    other_items: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (lambda_hat, slope, p_values, deltas).

    ``slope`` is the OLS slope of the DiD contrast on prevalence p (negative for
    prevalence-aversion). ``lambda_hat = -slope`` is reported so that positive
    lambda_hat means "utility for the manipulated cluster falls as it becomes
    more prevalent" -- matching the toy-model sign of lambda in U_B = v - lambda*D.
    """
    p_values = np.array(sorted(scores_by_p.keys()), dtype=np.float64)
    deltas = np.array([_delta(scores_by_p[p], probe_items, other_items) for p in p_values])
    slope = float(np.polyfit(p_values, deltas, 1)[0])
    return -slope, slope, p_values, deltas


def bootstrap_lambda_hat(
    pairs_by_p: dict[float, pd.DataFrame],
    n_items: int,
    probe_items: np.ndarray,
    other_items: np.ndarray,
    *,
    n_boot: int = 200,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI for lambda_hat, resampling pairs within each p."""
    rng = rng or np.random.default_rng(0)
    p_values = sorted(pairs_by_p.keys())
    p_arr = np.array(p_values, dtype=np.float64)
    # Pre-extract numpy arrays once (pandas indexing inside the loop dominates cost).
    cols = {
        p: (
            pairs_by_p[p]["item_a"].to_numpy(),
            pairs_by_p[p]["item_b"].to_numpy(),
            pairs_by_p[p]["p_prefer_a"].to_numpy(),
            pairs_by_p[p]["n_votes"].to_numpy(),
        )
        for p in p_values
    }
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        deltas = np.empty(len(p_values), dtype=np.float64)
        for i, p in enumerate(p_values):
            a, bb, pa, w = cols[p]
            idx = rng.integers(0, len(a), size=len(a))
            model = fit_bradley_terry_weighted(
                a[idx], bb[idx], pa[idx], n_items, weight=w[idx]
            )
            deltas[i] = _delta(model.score_all(), probe_items, other_items)
        samples[b] = -float(np.polyfit(p_arr, deltas, 1)[0])
    lo = float(np.percentile(samples, 100 * (1 - ci) / 2))
    hi = float(np.percentile(samples, 100 * (1 + ci) / 2))
    return lo, hi, samples


def _aic(sse: float, n: int, k: int) -> float:
    sse = max(sse, 1e-12)
    return n * np.log(sse / n) + 2 * k


def fit_functional_forms(p_values: np.ndarray, deltas: np.ndarray) -> dict:
    """Fit linear / log / threshold forms of the DiD contrast vs p; compare by AIC.

    The best form is the measured g_hat that will feed the Phase 2/3 surrogate
    (this instantiates Section 3.5's f-form check with a *measured* f).
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(p_values)
    forms: dict[str, dict] = {}

    # Linear: a + b p
    b, a = np.polyfit(p_values, deltas, 1)
    pred = a + b * p_values
    sse = float(np.sum((deltas - pred) ** 2))
    forms["linear"] = {"params": {"a": float(a), "b": float(b)}, "sse": sse, "aic": _aic(sse, n, 2)}

    # Log: a + b log(p)  (p > 0 by construction; min p is 0.02)
    lp = np.log(p_values)
    bl, al = np.polyfit(lp, deltas, 1)
    pred = al + bl * lp
    sse = float(np.sum((deltas - pred) ** 2))
    forms["log"] = {"params": {"a": float(al), "b": float(bl)}, "sse": sse, "aic": _aic(sse, n, 2)}

    # Threshold: a + b max(p - c, 0); grid-search c over interior p values.
    best = None
    candidate_c = p_values[(p_values > p_values.min()) & (p_values < p_values.max())]
    if candidate_c.size == 0:
        candidate_c = np.array([np.median(p_values)])
    for c in candidate_c:
        hinge = np.maximum(p_values - c, 0.0)
        if np.allclose(hinge, hinge[0]):
            continue
        bt, at = np.polyfit(hinge, deltas, 1)
        pred = at + bt * hinge
        sse_c = float(np.sum((deltas - pred) ** 2))
        if best is None or sse_c < best["sse"]:
            best = {"params": {"a": float(at), "b": float(bt), "c": float(c)}, "sse": sse_c}
    if best is not None:
        best["aic"] = _aic(best["sse"], n, 3)
        forms["threshold"] = best

    best_form = min(forms, key=lambda f: forms[f]["aic"])
    return {"forms": forms, "best_form": best_form}


def repeat_agreement(repeats: pd.DataFrame) -> float:
    """Mean same-order repeat agreement (spec Phase 0 item 3).

    Groups repeats by (pair, order) and averages the probability that two random
    same-order samples agree; for a group with m valid votes and k 'prefer A',
    agreement = (C(k,2)+C(m-k,2)) / C(m,2).
    """
    valid = repeats.dropna(subset=["picked_a"])
    agreements: list[float] = []
    for _, grp in valid.groupby(["pair_idx", "order_ab"]):
        m = len(grp)
        if m < 2:
            continue
        k = int(grp["picked_a"].sum())
        agreements.append((comb(k, 2) + comb(m - k, 2)) / comb(m, 2))
    return float(np.mean(agreements)) if agreements else float("nan")


def position_bias(repeats: pd.DataFrame) -> float:
    """P(choose the sentence shown in position A), pooled over order-swapped repeats."""
    valid = repeats.dropna(subset=["position_a"])
    return float(valid["position_a"].mean()) if len(valid) else float("nan")


def parse_failure_rate(repeats: pd.DataFrame) -> float:
    return float(repeats["choice"].isna().mean()) if len(repeats) else 0.0


@dataclass
class PsychometricResult:
    lambda_hat: float
    lambda_hat_ci: tuple[float, float]
    slope: float
    p_values: np.ndarray
    deltas: np.ndarray
    functional_forms: dict
    consistency: dict = field(default_factory=dict)

    @property
    def ci_excludes_zero(self) -> bool:
        lo, hi = self.lambda_hat_ci
        return lo > 0 or hi < 0


def estimate_psychometrics(
    pairs_by_p: dict[float, pd.DataFrame],
    n_items: int,
    probe_items: np.ndarray,
    other_items: np.ndarray,
    *,
    repeats_by_p: dict[float, pd.DataFrame] | None = None,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> PsychometricResult:
    scores_by_p = fit_scores_by_p(pairs_by_p, n_items)
    lambda_hat, slope, p_values, deltas = lambda_hat_point(scores_by_p, probe_items, other_items)
    lo, hi, _ = bootstrap_lambda_hat(
        pairs_by_p, n_items, probe_items, other_items, n_boot=n_boot, rng=rng
    )
    forms = fit_functional_forms(p_values, deltas)

    consistency: dict = {}
    if repeats_by_p is not None:
        all_repeats = pd.concat(list(repeats_by_p.values()), ignore_index=True)
        consistency = {
            "repeat_agreement": repeat_agreement(all_repeats),
            "position_bias": position_bias(all_repeats),
            "parse_failure_rate": parse_failure_rate(all_repeats),
        }

    return PsychometricResult(
        lambda_hat=lambda_hat,
        lambda_hat_ci=(lo, hi),
        slope=slope,
        p_values=p_values,
        deltas=deltas,
        functional_forms=forms,
        consistency=consistency,
    )
