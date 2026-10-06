"""Surrogate utility surface bridging elicitation and optimization (design section 4).

From pooled Phase 1 pair-level data, fit

    U_B_hat(y; D_pi) = alpha * v_hat(y) - g_hat(D_pi(cluster(y)))

where v_hat is the no-context BT utility (measured, template 2.6), g_hat is a
prevalence penalty whose functional form (linear / log / threshold) is selected
by held-out pair log-likelihood, and alpha is a scale freeing the contextual
condition's effective temperature from the no-context fit's.

Circularity discipline: the surrogate is used only *inside* optimization
(policy training, deployment-consistent optimum). All reported utilities come
from fresh judge queries on final policies (phase3), so the optimization target
and the evaluation instrument are never the same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.exp3.menu import Menu

FORMS = ("linear", "log", "threshold")

# Floor for cluster prevalence inside log(); arbitrary policies can drive a
# cluster's mass to ~0, where the Phase 1 grid (min p = 0.02) has no support.
P_FLOOR = 1e-3


def _phi(p: np.ndarray, form: str, c: float = 0.0) -> np.ndarray:
    """Nonlinearity applied to cluster prevalence before the fitted slope b."""
    p = np.asarray(p, dtype=np.float64)
    if form == "linear":
        return p
    if form == "log":
        return np.log(np.clip(p, P_FLOOR, 1.0))
    if form == "threshold":
        return np.maximum(p - c, 0.0)
    raise ValueError(f"Unknown form {form!r}")


def _phi_prime(p: np.ndarray, form: str, c: float = 0.0) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if form == "linear":
        return np.ones_like(p)
    if form == "log":
        return np.where(p > P_FLOOR, 1.0 / np.clip(p, P_FLOOR, 1.0), 0.0)
    if form == "threshold":
        return (p > c).astype(np.float64)
    raise ValueError(f"Unknown form {form!r}")


@dataclass
class SurrogateUtility:
    """Fitted U_B_hat(y; D_pi); pluggable into policy optimization loops."""

    v_hat: np.ndarray  # no-context BT utility, shape (n_items,)
    cluster_of: np.ndarray  # cluster id per item
    form: str  # "linear" | "log" | "threshold"
    alpha: float  # scale on v_hat
    b: float  # penalty slope (positive = prevalence-averse)
    c: float = 0.0  # threshold location (threshold form only)

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_of.max()) + 1

    def penalty(self, cluster_prevalence: np.ndarray) -> np.ndarray:
        """g_hat per cluster, shape (n_clusters,)."""
        return self.b * _phi(cluster_prevalence, self.form, self.c)

    def utility(self, distribution: np.ndarray) -> np.ndarray:
        """U_B_hat(y; D_pi) per item under the deployment ``distribution``."""
        distribution = np.asarray(distribution, dtype=np.float64)
        distribution = distribution / distribution.sum()
        prevalence = np.zeros(self.n_clusters, dtype=np.float64)
        np.add.at(prevalence, self.cluster_of, distribution)
        return self.alpha * self.v_hat - self.penalty(prevalence)[self.cluster_of]

    def expected_utility(self, distribution: np.ndarray) -> float:
        """E_{y~pi}[U_B_hat(y; D_pi)] for policy == deployment."""
        distribution = np.asarray(distribution, dtype=np.float64)
        distribution = distribution / distribution.sum()
        return float(distribution @ self.utility(distribution))


@dataclass
class SurrogateFit:
    surrogate: SurrogateUtility  # best form, refit on all data
    best_form: str
    metrics: pd.DataFrame  # per form: params, train/held-out mean per-vote LL
    heldout_ll_best: float
    heldout_ll_baseline: float  # alpha-only model (b = 0): the prevalence term's value
    n_pairs: int = 0
    extras: dict = field(default_factory=dict)


def build_surrogate_dataset(
    raw: pd.DataFrame,
    menu: Menu,
    *,
    family: str | None = None,
    formats: tuple[str, ...] = ("group_b_contextual", "group_b_stated"),
) -> pd.DataFrame:
    """Attach nominal cluster prevalences to Phase 1 pair rows.

    ``raw`` is Phase1Result.raw. Prevalence is the *nominal* D_pi of the cell's
    deployment (probe cluster at mass p, remainder uniform over other clusters);
    for the contextual format the judge saw a 20-sample draw from it, so the
    linear-form slope is estimated through an unbiased noisy channel.
    """
    df = raw[raw["format"].isin(formats)].copy()
    if family is not None:
        df = df[df["family"] == family]
    df = df.dropna(subset=["p_prefer_a"]).reset_index(drop=True)

    n_clusters = menu.n_clusters
    cluster_of = menu.cluster_of
    probe = df["probe_cluster"].to_numpy(dtype=np.int64)
    p = df["p"].to_numpy(dtype=np.float64)
    # Nominal cluster prevalence: p on the probe cluster, (1-p)/(k-1) elsewhere.
    other_mass = (1.0 - p) / (n_clusters - 1)

    cl_a = cluster_of[df["item_a"].to_numpy(dtype=np.int64)]
    cl_b = cluster_of[df["item_b"].to_numpy(dtype=np.int64)]
    df["prev_a"] = np.where(cl_a == probe, p, other_mass)
    df["prev_b"] = np.where(cl_b == probe, p, other_mass)
    return df


def _fit_alpha_b(
    dv: np.ndarray,
    dphi: np.ndarray,
    p_a: np.ndarray,
    w: np.ndarray,
    *,
    fit_b: bool = True,
    l2: float = 1e-6,
) -> tuple[float, float, float]:
    """Weighted BT logistic fit of diff = alpha*dv - b*dphi; returns (alpha, b, mean LL)."""

    def unpack(theta: np.ndarray) -> tuple[float, float]:
        alpha = theta[0]
        b = theta[1] if fit_b else 0.0
        return alpha, b

    def nll(theta: np.ndarray) -> float:
        alpha, b = unpack(theta)
        diffs = alpha * dv - b * dphi
        ll = p_a * (-np.logaddexp(0.0, -diffs)) + (1.0 - p_a) * (-np.logaddexp(0.0, diffs))
        return float(-np.sum(w * ll) + 0.5 * l2 * np.sum(theta**2))

    def grad(theta: np.ndarray) -> np.ndarray:
        alpha, b = unpack(theta)
        diffs = alpha * dv - b * dphi
        sig = 1.0 / (1.0 + np.exp(-np.clip(diffs, -500, 500)))
        coeff = w * (sig - p_a)
        g = np.zeros_like(theta)
        g[0] = float(np.sum(coeff * dv)) + l2 * theta[0]
        if fit_b:
            g[1] = float(np.sum(coeff * -dphi)) + l2 * theta[1]
        return g

    x0 = np.array([1.0, 0.0]) if fit_b else np.array([1.0])
    if not fit_b:
        # Single-parameter problem: keep the same interface via a 1-vector.
        def nll1(t: np.ndarray) -> float:
            return nll(np.array([t[0], 0.0])[: 2])

        def grad1(t: np.ndarray) -> np.ndarray:
            return grad(np.array([t[0]]))[:1]

        res = minimize(nll1, np.array([1.0]), jac=grad1, method="L-BFGS-B")
        alpha, b = float(res.x[0]), 0.0
    else:
        res = minimize(nll, x0, jac=grad, method="L-BFGS-B")
        alpha, b = unpack(res.x)

    diffs = alpha * dv - b * dphi
    ll = p_a * (-np.logaddexp(0.0, -diffs)) + (1.0 - p_a) * (-np.logaddexp(0.0, diffs))
    mean_ll = float(np.sum(w * ll) / np.sum(w))
    return float(alpha), float(b), mean_ll


def _mean_ll(
    alpha: float, b: float, dv: np.ndarray, dphi: np.ndarray, p_a: np.ndarray, w: np.ndarray
) -> float:
    diffs = alpha * dv - b * dphi
    ll = p_a * (-np.logaddexp(0.0, -diffs)) + (1.0 - p_a) * (-np.logaddexp(0.0, diffs))
    return float(np.sum(w * ll) / np.sum(w))


def fit_surrogate(
    raw: pd.DataFrame,
    v_hat: np.ndarray,
    menu: Menu,
    *,
    family: str | None = None,
    formats: tuple[str, ...] = ("group_b_contextual", "group_b_stated"),
    test_frac: float = 0.25,
    threshold_grid: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6),
    rng: np.random.Generator | None = None,
) -> SurrogateFit:
    """Fit the response surface from pooled Phase 1 data; select form on held-out LL."""
    rng = rng or np.random.default_rng(0)
    v_hat = np.asarray(v_hat, dtype=np.float64)

    ds = build_surrogate_dataset(raw, menu, family=family, formats=formats)
    if len(ds) == 0:
        raise ValueError("No Phase 1 rows matched the surrogate dataset filters.")

    item_a = ds["item_a"].to_numpy(dtype=np.int64)
    item_b = ds["item_b"].to_numpy(dtype=np.int64)
    p_a = ds["p_prefer_a"].to_numpy(dtype=np.float64)
    w = ds["n_votes"].to_numpy(dtype=np.float64)
    prev_a = ds["prev_a"].to_numpy(dtype=np.float64)
    prev_b = ds["prev_b"].to_numpy(dtype=np.float64)
    dv = v_hat[item_a] - v_hat[item_b]

    n = len(ds)
    test_mask = rng.random(n) < test_frac
    train = ~test_mask

    rows: list[dict] = []
    fitted: dict[str, dict] = {}
    for form in FORMS:
        best: dict | None = None
        c_grid = threshold_grid if form == "threshold" else (0.0,)
        for c in c_grid:
            dphi = _phi(prev_a, form, c) - _phi(prev_b, form, c)
            alpha, b, train_ll = _fit_alpha_b(
                dv[train], dphi[train], p_a[train], w[train]
            )
            heldout_ll = _mean_ll(alpha, b, dv[test_mask], dphi[test_mask], p_a[test_mask], w[test_mask])
            cand = {
                "form": form,
                "alpha": alpha,
                "b": b,
                "c": float(c),
                "train_ll": train_ll,
                "heldout_ll": heldout_ll,
            }
            if best is None or cand["heldout_ll"] > best["heldout_ll"]:
                best = cand
        assert best is not None
        rows.append(best)
        fitted[form] = best

    # Baseline: prevalence-free (b = 0). Its held-out gap to the best form is the
    # measured value of the prevalence term.
    alpha0, _, _ = _fit_alpha_b(dv[train], np.zeros(train.sum()), p_a[train], w[train], fit_b=False)
    baseline_ll = _mean_ll(alpha0, 0.0, dv[test_mask], np.zeros(int(test_mask.sum())), p_a[test_mask], w[test_mask])
    rows.append(
        {"form": "baseline_no_prevalence", "alpha": alpha0, "b": 0.0, "c": 0.0,
         "train_ll": float("nan"), "heldout_ll": baseline_ll}
    )

    best_form = max(FORMS, key=lambda f: fitted[f]["heldout_ll"])

    # Refit the winning form on all data for the deployed surrogate.
    c_best = fitted[best_form]["c"]
    dphi_all = _phi(prev_a, best_form, c_best) - _phi(prev_b, best_form, c_best)
    alpha, b, _ = _fit_alpha_b(dv, dphi_all, p_a, w)

    surrogate = SurrogateUtility(
        v_hat=v_hat,
        cluster_of=menu.cluster_of,
        form=best_form,
        alpha=alpha,
        b=b,
        c=c_best,
    )
    return SurrogateFit(
        surrogate=surrogate,
        best_form=best_form,
        metrics=pd.DataFrame(rows),
        heldout_ll_best=fitted[best_form]["heldout_ll"],
        heldout_ll_baseline=baseline_ll,
        n_pairs=n,
    )


def solve_deployment_consistent(
    surrogate: SurrogateUtility,
    menu: Menu,
    *,
    tol: float = 1e-12,
    max_iter: int = 1000,
    within_cluster: str = "best",
) -> np.ndarray:
    """pi* in argmax_pi E_{y~pi}[U_B_hat(y; D_pi)] on the 100-item simplex.

    The penalty depends only on cluster mass, so the residual problem is a
    10-dim concave (for b >= 0) program over cluster masses P:
    max sum_c P_c*(u*_c - g(P_c)).

    ``within_cluster`` picks how a cluster's mass is allocated to its items:

    - ``"best"``: all on the argmax of alpha*v_hat -- the unconstrained optimum
      under the surrogate.
    - ``"uniform"``: spread evenly over the cluster's items. Use this when the
      surrogate must stay inside its measurement domain: Phase 1 contexts never
      put more than ~p/n_cluster_items mass on any single item, and real judges
      can penalize verbatim item repetition far beyond the cluster-level
      penalty, making the "best" optimum an out-of-domain extrapolation.
    """
    if within_cluster not in ("best", "uniform"):
        raise ValueError(f"Unknown within_cluster mode {within_cluster!r}")
    n_clusters = menu.n_clusters
    best_item = np.array(
        [menu.items_in_cluster(c)[np.argmax(surrogate.alpha * surrogate.v_hat[menu.items_in_cluster(c)])]
         for c in range(n_clusters)],
        dtype=np.int64,
    )
    if within_cluster == "best":
        u_star = surrogate.alpha * surrogate.v_hat[best_item]
    else:
        u_star = np.array(
            [surrogate.alpha * surrogate.v_hat[menu.items_in_cluster(c)].mean()
             for c in range(n_clusters)]
        )

    def neg_obj(P: np.ndarray) -> float:
        g = surrogate.b * _phi(P, surrogate.form, surrogate.c)
        return float(-(P @ (u_star - g)))

    def neg_grad(P: np.ndarray) -> np.ndarray:
        g = surrogate.b * _phi(P, surrogate.form, surrogate.c)
        gp = surrogate.b * _phi_prime(P, surrogate.form, surrogate.c)
        return -(u_star - g - P * gp)

    x0 = np.full(n_clusters, 1.0 / n_clusters)
    result = minimize(
        neg_obj,
        x0,
        jac=neg_grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_clusters,
        constraints={"type": "eq", "fun": lambda P: float(np.sum(P) - 1.0)},
        options={"ftol": tol, "maxiter": max_iter},
    )
    if result.success:
        P = np.clip(result.x, 0.0, 1.0)
    else:
        # Damped fixed-point fallback: P <- softmax((u* - g(P) - P g'(P)) / tau).
        P = x0.copy()
        for _ in range(max_iter):
            g = surrogate.b * _phi(P, surrogate.form, surrogate.c)
            gp = surrogate.b * _phi_prime(P, surrogate.form, surrogate.c)
            z = (u_star - g - P * gp) / 0.05
            z = z - z.max()
            P_new = np.exp(z)
            P_new = P_new / P_new.sum()
            P_next = 0.5 * P_new + 0.5 * P
            if np.abs(P_next - P).sum() < 1e-10:
                P = P_next
                break
            P = P_next
    P = P / P.sum()

    pi = np.zeros(menu.n_items, dtype=np.float64)
    if within_cluster == "best":
        pi[best_item] = P
    else:
        for c in range(n_clusters):
            items = menu.items_in_cluster(c)
            pi[items] = P[c] / len(items)
    return pi
