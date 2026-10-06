"""Synthetic two-group preference environment for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Group = Literal["A", "B"]


@dataclass(frozen=True)
class EnvironmentConfig:
    n_items: int = 100
    ref_temperature: float = 1.0
    target_temperature: float = 0.1
    group_a_fraction: float = 0.5
    seed: int = 42


@dataclass
class PreferenceEnvironment:
    """Two-group discrete output space with shared intrinsic utility v(y)."""

    config: EnvironmentConfig
    v: np.ndarray  # intrinsic utility, shape (n_items,)
    pi_ref: np.ndarray  # reference policy distribution
    pi_target: np.ndarray  # concentrated target policy

    @classmethod
    def create(cls, config: EnvironmentConfig | None = None) -> "PreferenceEnvironment":
        config = config or EnvironmentConfig()
        rng = np.random.default_rng(config.seed)
        v = _generate_intrinsic_utility(config.n_items, rng)
        pi_ref = _softmax(v / config.ref_temperature)
        pi_target = _softmax(v / config.target_temperature)
        return cls(config=config, v=v, pi_ref=pi_ref, pi_target=pi_target)

    @property
    def n_items(self) -> int:
        return self.config.n_items

    def utility_a(self, distribution: np.ndarray | None = None) -> np.ndarray:
        """Group A utility: u_A(y) = v(y), deployment-independent."""
        del distribution
        return self.v.copy()

    def utility_b(self, distribution: np.ndarray, lam: float) -> np.ndarray:
        """Group B utility: U_B(y; D_pi) = v(y) - lam * D_pi(y)."""
        return self.v - lam * distribution

    def utility_for_group(
        self, group: Group, distribution: np.ndarray, lam: float
    ) -> np.ndarray:
        if group == "A":
            return self.utility_a(distribution)
        return self.utility_b(distribution, lam)

    def policy_at_t(self, t: float) -> np.ndarray:
        """Interpolate between pi_ref and pi_target: pi_t = normalize((1-t)*pi_ref + t*pi_target)."""
        t = float(np.clip(t, 0.0, 1.0))
        mixed = (1.0 - t) * self.pi_ref + t * self.pi_target
        return mixed / mixed.sum()

    def sample_preference_pairs(
        self,
        n_pairs: int,
        distribution: np.ndarray,
        lam: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample pairwise preferences from mixed population under given distribution.

        Returns (winners, losers, group_labels) where group_labels is 0 for A, 1 for B.
        """
        rng = rng or np.random.default_rng(self.config.seed)
        n = self.n_items

        y1 = rng.choice(n, size=n_pairs, p=distribution)
        y2 = rng.choice(n, size=n_pairs, p=distribution)
        same = y1 == y2
        while same.any():
            y2[same] = rng.choice(n, size=same.sum(), p=distribution)
            same = y1 == y2

        is_group_b = rng.random(n_pairs) >= self.config.group_a_fraction
        utilities = np.where(
            is_group_b[:, None],
            self.utility_b(distribution, lam)[None, :],
            self.utility_a(distribution)[None, :],
        )

        u1 = utilities[np.arange(n_pairs), y1]
        u2 = utilities[np.arange(n_pairs), y2]
        b_wins = u1 >= u2

        winners = np.where(b_wins, y1, y2)
        losers = np.where(b_wins, y2, y1)
        return winners, losers, is_group_b.astype(np.int8)


def _generate_intrinsic_utility(n_items: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth structured intrinsic utility via mixture of Gaussians over item indices."""
    positions = np.arange(n_items, dtype=float)
    centers = rng.choice(n_items, size=3, replace=False)
    weights = rng.dirichlet([1.0, 1.0, 1.0])
    widths = rng.uniform(8.0, 20.0, size=3)
    v = np.zeros(n_items)
    for center, weight, width in zip(centers, weights, widths):
        v += weight * np.exp(-0.5 * ((positions - center) / width) ** 2)
    v = v / v.max()
    v += 0.05 * rng.normal(size=n_items)
    return v


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - z.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()
