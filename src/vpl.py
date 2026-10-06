"""Simplified Variational Preference Learning with two latent groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.reward_model import BradleyTerryRewardModel, fit_bradley_terry


@dataclass
class VPLModel:
    """Group-conditioned Bradley-Terry reward models."""

    rm_a: BradleyTerryRewardModel
    rm_b: BradleyTerryRewardModel
    group_a_fraction: float

    def population_reward(self) -> np.ndarray:
        """Population reward for a single globally deployed policy."""
        alpha = self.group_a_fraction
        return alpha * self.rm_a.score_all() + (1.0 - alpha) * self.rm_b.score_all()


def fit_vpl_oracle(
    winners: np.ndarray,
    losers: np.ndarray,
    group_labels: np.ndarray,
    n_items: int,
    group_a_fraction: float,
) -> VPLModel:
    """Fit separate group RMs using oracle group identity."""
    mask_a = group_labels == 0
    mask_b = group_labels == 1
    rm_a = fit_bradley_terry(winners[mask_a], losers[mask_a], n_items)
    rm_b = fit_bradley_terry(winners[mask_b], losers[mask_b], n_items)
    return VPLModel(rm_a=rm_a, rm_b=rm_b, group_a_fraction=group_a_fraction)


def fit_vpl_inferred(
    winners: np.ndarray,
    losers: np.ndarray,
    n_items: int,
    group_a_fraction: float,
    *,
    n_em_iters: int = 15,
    prior_a: float | None = None,
) -> VPLModel:
    """
    Fit group-conditioned RMs via EM with latent group assignment.

    E-step: posterior over group labels from current RMs.
    M-step: refit each group RM on assigned pairs.
    """
    winners = np.asarray(winners, dtype=np.int64)
    losers = np.asarray(losers, dtype=np.int64)
    n_pairs = len(winners)
    prior_a = group_a_fraction if prior_a is None else prior_a
    prior_b = 1.0 - prior_a

    pooled = fit_bradley_terry(winners, losers, n_items)
    scores = pooled.score_all()
    diff = scores[winners] - scores[losers]
    # Initialize assignments from preference direction vs pooled RM.
    assignments = (diff < 0).astype(np.int8)

    rm_a = fit_bradley_terry(winners[assignments == 0], losers[assignments == 0], n_items)
    rm_b = fit_bradley_terry(winners[assignments == 1], losers[assignments == 1], n_items)

    for _ in range(n_em_iters):
        scores_a = rm_a.score_all()
        scores_b = rm_b.score_all()
        diff_a = scores_a[winners] - scores_a[losers]
        diff_b = scores_b[winners] - scores_b[losers]
        log_p_a = np.log(prior_a + 1e-12) + _log_sigmoid(diff_a)
        log_p_b = np.log(prior_b + 1e-12) + _log_sigmoid(diff_b)
        log_norm = np.logaddexp(log_p_a, log_p_b)
        resp_a = np.exp(log_p_a - log_norm)

        assignments = (resp_a < 0.5).astype(np.int8)
        if assignments.sum() == 0 or assignments.sum() == n_pairs:
            assignments = (diff < 0).astype(np.int8)

        mask_a = assignments == 0
        mask_b = assignments == 1
        if mask_a.any():
            rm_a = fit_bradley_terry(winners[mask_a], losers[mask_a], n_items)
        if mask_b.any():
            rm_b = fit_bradley_terry(winners[mask_b], losers[mask_b], n_items)

    return VPLModel(rm_a=rm_a, rm_b=rm_b, group_a_fraction=group_a_fraction)


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)
