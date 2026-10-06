"""Metrics for calibration, distribution divergence, and Experiment 2 evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from src.environment import PreferenceEnvironment


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetrized Jensen-Shannon divergence (natural log)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (_kl(p, m) + _kl(q, m))


def calibration_metrics(
    predicted: np.ndarray,
    true_utility: np.ndarray,
    *,
    distribution: np.ndarray | None = None,
    n_pairs: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Compute calibration metrics between static RM scores and true Group B utilities.

    Primary calibration error uses pairwise ranking accuracy (what the RM must predict).
    Item-level Spearman rho is reported for triangulation.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    true_utility = np.asarray(true_utility, dtype=np.float64)

    rho, _ = spearmanr(predicted, true_utility)
    if np.isnan(rho):
        rho = 0.0

    pred_norm = (predicted - predicted.mean()) / (predicted.std() + 1e-12)
    true_norm = (true_utility - true_utility.mean()) / (true_utility.std() + 1e-12)
    mse = float(np.mean((pred_norm - true_norm) ** 2))

    rank_acc = 1.0
    if distribution is not None and rng is not None:
        rank_acc = pairwise_ranking_accuracy(
            predicted, true_utility, distribution, n_pairs, rng
        )

    return {
        "spearman_rho": float(rho),
        "calibration_error": float(1.0 - rank_acc),
        "normalized_mse": mse,
        "pairwise_ranking_accuracy": float(rank_acc),
    }


def pairwise_ranking_accuracy(
    predicted: np.ndarray,
    true_utility: np.ndarray,
    distribution: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> float:
    """Held-out pairwise ranking accuracy under true utilities."""
    n_items = len(predicted)
    y1 = rng.choice(n_items, size=n_pairs, p=distribution)
    y2 = rng.choice(n_items, size=n_pairs, p=distribution)
    same = y1 == y2
    while same.any():
        y2[same] = rng.choice(n_items, size=same.sum(), p=distribution)
        same = y1 == y2

    true_wins = true_utility[y1] >= true_utility[y2]
    pred_wins = predicted[y1] >= predicted[y2]
    return float((true_wins == pred_wins).mean())


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - z.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def expected_utility_b(
    policy: np.ndarray, env: "PreferenceEnvironment", lam: float
) -> float:
    """E_{y~pi}[U_B(y; pi)] = pi·v - lam * ||pi||^2."""
    policy = np.asarray(policy, dtype=np.float64)
    policy = policy / policy.sum()
    return float(policy @ env.v - lam * np.sum(policy**2))


def solve_deployment_consistent_policy(
    env: "PreferenceEnvironment",
    lam: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Find pi* in argmax_pi E_{y~pi}[U_B(y; pi)] via concave optimization on the simplex.

    Objective: pi·v - lam * sum(pi^2). Falls back to fixed-point iteration if needed.
    """
    n = env.n_items
    v = env.v

    if lam == 0.0:
        pi = np.zeros(n, dtype=np.float64)
        pi[int(np.argmax(v))] = 1.0
        return pi

    def neg_objective(p: np.ndarray) -> float:
        return float(-(p @ v - lam * np.sum(p**2)))

    def neg_grad(p: np.ndarray) -> np.ndarray:
        return -(v - 2.0 * lam * p)

    x0 = env.pi_ref.copy()
    constraints = {"type": "eq", "fun": lambda p: np.sum(p) - 1.0}
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        neg_objective,
        x0,
        jac=neg_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": max_iter},
    )
    if result.success:
        pi = np.clip(result.x, 0.0, 1.0)
        return pi / pi.sum()

    # Fixed-point fallback: u(y) = v(y) - lam pi(y), pi <- softmax(u / tau)
    pi = env.pi_ref.copy()
    for _ in range(max_iter):
        u = v - lam * pi
        pi_new = _softmax(u, temperature=0.05)
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            return pi_new
        pi = pi_new
    return pi


def utility_retention(
    pi_aligned: np.ndarray,
    env: "PreferenceEnvironment",
    lam: float,
    *,
    pi_star: np.ndarray | None = None,
) -> float:
    """Ratio of realized Group B utility to deployment-consistent optimum."""
    pi_star = pi_star if pi_star is not None else solve_deployment_consistent_policy(env, lam)
    realized = expected_utility_b(pi_aligned, env, lam)
    optimal = expected_utility_b(pi_star, env, lam)
    if abs(optimal) < 1e-12:
        return 1.0 if abs(realized) < 1e-12 else 0.0
    return float(realized / optimal)


def subgroup_output_divergence(
    env: "PreferenceEnvironment",
    distribution: np.ndarray,
    lam: float,
    *,
    temperature: float = 1.0,
) -> float:
    """
    JS divergence between group-preferred output distributions at a given deployment.

    p_A = softmax(v / T); p_B = softmax((v - lam * pi) / T).
    """
    distribution = np.asarray(distribution, dtype=np.float64)
    distribution = distribution / distribution.sum()
    p_a = _softmax(env.v, temperature=temperature)
    u_b = env.utility_b(distribution, lam)
    p_b = _softmax(u_b, temperature=temperature)
    return js_divergence(p_a, p_b)
