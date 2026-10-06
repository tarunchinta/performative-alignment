"""KL-regularized policy optimization (discrete RLHF/PPO analog)."""

from __future__ import annotations

import numpy as np


def optimize_policy(
    reward_scores: np.ndarray,
    pi_ref: np.ndarray,
    kl_coef: float,
    *,
    n_iters: int = 200,
    lr: float = 0.1,
) -> np.ndarray:
    """
    Maximize E_pi[r(y)] - kl_coef * KL(pi || pi_ref) over the simplex.

    Uses projected gradient ascent on softmax logits. The closed-form optimum
    pi* propto pi_ref * exp(r / kl_coef) is used when kl_coef > 0.
    """
    reward_scores = np.asarray(reward_scores, dtype=np.float64)
    pi_ref = np.asarray(pi_ref, dtype=np.float64)
    pi_ref = pi_ref / pi_ref.sum()

    if kl_coef <= 0:
        pi = np.zeros_like(pi_ref)
        pi[int(np.argmax(reward_scores))] = 1.0
        return pi

    # Closed-form KL-regularized optimum (exact for this objective).
    logits = np.log(pi_ref + 1e-12) + reward_scores / kl_coef
    logits = logits - logits.max()
    pi = np.exp(logits)
    pi = pi / pi.sum()

    # Optional refinement via gradient ascent for numerical stability at extremes.
    log_pi_ref = np.log(pi_ref + 1e-12)
    logits = np.log(pi + 1e-12)
    for _ in range(n_iters):
        pi = _softmax_from_logits(logits)
        kl_grad = np.log(pi + 1e-12) - log_pi_ref + 1.0
        grad = reward_scores - kl_coef * kl_grad
        grad = grad - grad @ pi  # project to tangent space of simplex
        logits = logits + lr * grad
        logits = logits - logits.max()

    pi = _softmax_from_logits(logits)
    return pi / pi.sum()


def _softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()
