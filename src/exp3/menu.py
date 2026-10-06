"""Frozen output menu for Experiment 3.

The menu is a fixed set of N=100 short texts across 10 designed style clusters
(10 items each), generated once and frozen before any judging. Prevalence is
defined at the cluster level: D_pi(c) = sum_{y in c} pi(y).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

DEFAULT_MENU_PATH = Path(__file__).resolve().parents[2] / "data" / "exp3_menu.json"


@dataclass(frozen=True)
class MenuItem:
    id: int
    cluster_id: int
    cluster_name: str
    text: str
    v: float  # designed intrinsic (no-context) utility


@dataclass
class Menu:
    """Frozen 100-item output space with cluster structure and intrinsic utility."""

    domain: str
    clusters: list[str]
    items: list[MenuItem]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Menu":
        path = Path(path) if path is not None else DEFAULT_MENU_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = [
            MenuItem(
                id=int(it["id"]),
                cluster_id=int(it["cluster_id"]),
                cluster_name=str(it["cluster_name"]),
                text=str(it["text"]),
                v=float(it["v"]),
            )
            for it in raw["items"]
        ]
        items.sort(key=lambda it: it.id)
        menu = cls(domain=str(raw["domain"]), clusters=list(raw["clusters"]), items=items)
        menu._validate()
        return menu

    def _validate(self) -> None:
        n = len(self.items)
        if [it.id for it in self.items] != list(range(n)):
            raise ValueError("Menu item ids must be contiguous 0..N-1.")
        counts = np.bincount(self.cluster_of, minlength=self.n_clusters)
        if not np.all(counts == counts[0]):
            raise ValueError(f"Clusters must be balanced; got sizes {counts.tolist()}.")

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @cached_property
    def v(self) -> np.ndarray:
        """Designed intrinsic utility vector, shape (n_items,)."""
        return np.array([it.v for it in self.items], dtype=np.float64)

    @cached_property
    def cluster_of(self) -> np.ndarray:
        """Cluster id per item, shape (n_items,), dtype int."""
        return np.array([it.cluster_id for it in self.items], dtype=np.int64)

    @cached_property
    def texts(self) -> list[str]:
        return [it.text for it in self.items]

    def items_in_cluster(self, cluster_id: int) -> np.ndarray:
        return np.where(self.cluster_of == cluster_id)[0]

    def cluster_prevalence(self, distribution: np.ndarray) -> np.ndarray:
        """D_pi(c) = sum_{y in c} pi(y), shape (n_clusters,)."""
        distribution = np.asarray(distribution, dtype=np.float64)
        prevalence = np.zeros(self.n_clusters, dtype=np.float64)
        np.add.at(prevalence, self.cluster_of, distribution)
        return prevalence

    def deployment_distribution(self, probe_cluster: int, p_c: float) -> np.ndarray:
        """Item-level distribution placing cluster mass p_c on ``probe_cluster``.

        Mass ``p_c`` is split uniformly over the probe cluster's items; the
        remaining ``1 - p_c`` is spread uniformly over all other clusters
        (uniformly within each), so every non-probe cluster receives equal mass.
        """
        if not 0.0 <= p_c <= 1.0:
            raise ValueError(f"p_c must be in [0, 1]; got {p_c}.")
        dist = np.zeros(self.n_items, dtype=np.float64)
        probe_items = self.items_in_cluster(probe_cluster)
        dist[probe_items] = p_c / len(probe_items)

        other_clusters = [c for c in range(self.n_clusters) if c != probe_cluster]
        remaining = 1.0 - p_c
        per_cluster = remaining / len(other_clusters) if other_clusters else 0.0
        for c in other_clusters:
            c_items = self.items_in_cluster(c)
            dist[c_items] = per_cluster / len(c_items)

        total = dist.sum()
        if total <= 0:
            raise ValueError("Deployment distribution has zero mass.")
        return dist / total

    def sample_exposure(
        self,
        distribution: np.ndarray,
        k: int = 20,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw k item indices i.i.d. from ``distribution`` (the exposure block)."""
        rng = rng or np.random.default_rng()
        distribution = np.asarray(distribution, dtype=np.float64)
        distribution = distribution / distribution.sum()
        return rng.choice(self.n_items, size=k, p=distribution)
