"""Bradley-Terry reward model for discrete item preferences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class BradleyTerryRewardModel:
    """Per-item Bradley-Terry scores fit via logistic regression on pairwise data."""

    scores: np.ndarray  # shape (n_items,)

    def predict(self, items: np.ndarray | int) -> np.ndarray:
        return self.scores[items]

    def score_all(self) -> np.ndarray:
        return self.scores.copy()


def fit_bradley_terry(
    winners: np.ndarray,
    losers: np.ndarray,
    n_items: int,
    *,
    max_iter: int = 1000,
) -> BradleyTerryRewardModel:
    """
    Fit Bradley-Terry model: P(y_w > y_l) = sigmoid(r(y_w) - r(y_l)).

    Optimizes negative log-likelihood via L-BFGS-B with mean-zero identifiability constraint.
    """
    winners = np.asarray(winners, dtype=np.int64)
    losers = np.asarray(losers, dtype=np.int64)

    def neg_log_likelihood(scores: np.ndarray) -> float:
        scores = scores - scores.mean()
        diffs = scores[winners] - scores[losers]
        # Stable log-sigmoid: log P(win) = -softplus(-diff)
        return float(-np.sum(-np.logaddexp(0.0, -diffs)))

    def grad(scores: np.ndarray) -> np.ndarray:
        scores = scores - scores.mean()
        diffs = scores[winners] - scores[losers]
        sig = expit(diffs)
        g = np.zeros(n_items, dtype=np.float64)
        np.add.at(g, winners, sig - 1.0)
        np.add.at(g, losers, sig)
        g -= g.mean()
        return g

    x0 = np.zeros(n_items, dtype=np.float64)
    result = minimize(
        neg_log_likelihood,
        x0,
        jac=grad,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    scores = result.x - result.x.mean()
    return BradleyTerryRewardModel(scores=scores)


def fit_bradley_terry_weighted(
    item_a: np.ndarray,
    item_b: np.ndarray,
    p_a: np.ndarray,
    n_items: int,
    *,
    weight: np.ndarray | None = None,
    l2: float = 1e-4,
    max_iter: int = 1000,
) -> BradleyTerryRewardModel:
    """Fit Bradley-Terry from soft (vote-fraction) pairwise labels.

    Each row gives P(item_a preferred over item_b) = ``p_a`` with an optional
    ``weight`` (e.g. number of non-abstaining votes). Minimizes the weighted
    cross-entropy NLL with a mean-zero constraint and a small ridge for
    stability on sparsely-covered items. This is the elicitation analogue of
    ``fit_bradley_terry`` for judge preferences, which are fractions not 0/1.
    """
    item_a = np.asarray(item_a, dtype=np.int64)
    item_b = np.asarray(item_b, dtype=np.int64)
    p_a = np.asarray(p_a, dtype=np.float64)

    valid = ~np.isnan(p_a)
    item_a, item_b, p_a = item_a[valid], item_b[valid], p_a[valid]
    if weight is None:
        w = np.ones_like(p_a)
    else:
        w = np.asarray(weight, dtype=np.float64)[valid]

    def neg_log_likelihood(scores: np.ndarray) -> float:
        scores = scores - scores.mean()
        diffs = scores[item_a] - scores[item_b]
        # log sigma(d) = -softplus(-d); log sigma(-d) = -softplus(d)
        ll = p_a * (-np.logaddexp(0.0, -diffs)) + (1.0 - p_a) * (-np.logaddexp(0.0, diffs))
        return float(-np.sum(w * ll) + 0.5 * l2 * np.sum(scores**2))

    def grad(scores: np.ndarray) -> np.ndarray:
        scores = scores - scores.mean()
        diffs = scores[item_a] - scores[item_b]
        sig = expit(diffs)
        coeff = w * (sig - p_a)
        g = np.zeros(n_items, dtype=np.float64)
        np.add.at(g, item_a, coeff)
        np.add.at(g, item_b, -coeff)
        g += l2 * scores
        g -= g.mean()
        return g

    x0 = np.zeros(n_items, dtype=np.float64)
    result = minimize(
        neg_log_likelihood,
        x0,
        jac=grad,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    scores = result.x - result.x.mean()
    return BradleyTerryRewardModel(scores=scores)


def pairwise_accuracy(
    model: BradleyTerryRewardModel,
    winners: np.ndarray,
    losers: np.ndarray,
) -> float:
    """Fraction of pairs where model agrees with labels."""
    preds = model.predict(winners) >= model.predict(losers)
    return float(preds.mean())


def held_out_log_likelihood(
    model: BradleyTerryRewardModel,
    item_a: np.ndarray,
    item_b: np.ndarray,
    p_a: np.ndarray,
    *,
    weight: np.ndarray | None = None,
) -> float:
    """Mean per-vote log-likelihood of soft labels under a fitted BT model."""
    item_a = np.asarray(item_a, dtype=np.int64)
    item_b = np.asarray(item_b, dtype=np.int64)
    p_a = np.asarray(p_a, dtype=np.float64)
    valid = ~np.isnan(p_a)
    item_a, item_b, p_a = item_a[valid], item_b[valid], p_a[valid]
    w = np.ones_like(p_a) if weight is None else np.asarray(weight, dtype=np.float64)[valid]
    scores = model.score_all()
    diffs = scores[item_a] - scores[item_b]
    ll = p_a * (-np.logaddexp(0.0, -diffs)) + (1.0 - p_a) * (-np.logaddexp(0.0, diffs))
    total_w = float(np.sum(w))
    return float(np.sum(w * ll) / total_w) if total_w > 0 else 0.0
