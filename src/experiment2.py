"""Experiment 2: pluralistic alignment methods under deployment-dependent utility."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.environment import EnvironmentConfig, PreferenceEnvironment
from src.metrics import (
    solve_deployment_consistent_policy,
    subgroup_output_divergence,
    utility_retention,
)
from src.policy import optimize_policy
from src.reward_model import fit_bradley_terry
from src.vpl import fit_vpl_inferred, fit_vpl_oracle

Method = Literal["rlhf", "vpl_inferred", "vpl_oracle"]
METHODS: tuple[Method, ...] = ("rlhf", "vpl_inferred", "vpl_oracle")


@dataclass
class Experiment2Config:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)
    n_train_pairs: int = 80_000
    n_val_pairs: int = 10_000
    kl_coef: float = 0.5
    em_iters: int = 15
    seed: int = 42


@dataclass
class Experiment2Result:
    config: Experiment2Config
    env: PreferenceEnvironment
    results: pd.DataFrame


def run_experiment2(config: Experiment2Config | None = None) -> Experiment2Result:
    """Run Experiment 2 sweep over lambda and alignment methods."""
    config = config or Experiment2Config()
    rng = np.random.default_rng(config.seed)
    env = PreferenceEnvironment.create(config.env)

    rows: list[dict] = []
    for lam in tqdm(config.lambdas, desc="lambda sweep"):
        pi_star = solve_deployment_consistent_policy(env, lam)
        pre_div = subgroup_output_divergence(env, env.pi_ref, lam)

        winners, losers, group_labels = env.sample_preference_pairs(
            config.n_train_pairs, env.pi_ref, lam, rng=rng
        )

        rm_pooled = fit_bradley_terry(winners, losers, env.n_items)
        pi_rlhf = optimize_policy(rm_pooled.score_all(), env.pi_ref, config.kl_coef)

        vpl_oracle = fit_vpl_oracle(
            winners,
            losers,
            group_labels,
            env.n_items,
            env.config.group_a_fraction,
        )
        pi_vpl_oracle = optimize_policy(
            vpl_oracle.population_reward(), env.pi_ref, config.kl_coef
        )

        vpl_inferred = fit_vpl_inferred(
            winners,
            losers,
            env.n_items,
            env.config.group_a_fraction,
            n_em_iters=config.em_iters,
        )
        pi_vpl_inferred = optimize_policy(
            vpl_inferred.population_reward(), env.pi_ref, config.kl_coef
        )

        aligned = {
            "rlhf": pi_rlhf,
            "vpl_oracle": pi_vpl_oracle,
            "vpl_inferred": pi_vpl_inferred,
        }

        for method in METHODS:
            pi_aligned = aligned[method]
            retention = utility_retention(
                pi_aligned, env, lam, pi_star=pi_star
            )
            post_div = subgroup_output_divergence(env, pi_aligned, lam)
            rows.append(
                {
                    "lambda": lam,
                    "method": method,
                    "utility_retention": retention,
                    "subgroup_divergence_pre": pre_div,
                    "subgroup_divergence_post": post_div,
                    "optimal_utility_b": float(
                        np.dot(pi_star, env.v) - lam * np.sum(pi_star**2)
                    ),
                }
            )

    return Experiment2Result(
        config=config,
        env=env,
        results=pd.DataFrame(rows),
    )


@dataclass
class Experiment2MultiSeedResult:
    base_config: Experiment2Config
    seeds: tuple[int, ...]
    per_seed: pd.DataFrame  # all sweep rows, tagged with a "seed" column
    summary: pd.DataFrame  # aggregated mean/std per (lambda, method)


def run_experiment2_multiseed(
    base_config: Experiment2Config | None = None,
    *,
    n_seeds: int = 10,
    base_seed: int = 42,
) -> Experiment2MultiSeedResult:
    """Run Experiment 2 across multiple seeds, varying both the environment
    landscape (``env.seed``) and preference sampling (``seed``) together.

    Returns per-seed rows plus a mean/std summary per (lambda, method).
    """
    base_config = base_config or Experiment2Config()
    seeds = tuple(range(base_seed, base_seed + n_seeds))

    frames: list[pd.DataFrame] = []
    for seed in tqdm(seeds, desc="seed sweep"):
        seed_config = replace(
            base_config,
            env=replace(base_config.env, seed=seed),
            seed=seed,
        )
        seed_result = run_experiment2(seed_config)
        frame = seed_result.results.copy()
        frame.insert(0, "seed", seed)
        frames.append(frame)

    per_seed = pd.concat(frames, ignore_index=True)
    summary = _aggregate_multiseed(per_seed)

    return Experiment2MultiSeedResult(
        base_config=base_config,
        seeds=seeds,
        per_seed=per_seed,
        summary=summary,
    )


def _aggregate_multiseed(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed rows into mean/std per (lambda, method)."""
    grouped = per_seed.groupby(["lambda", "method"], sort=False)
    summary = grouped.agg(
        retention_mean=("utility_retention", "mean"),
        retention_std=("utility_retention", lambda s: float(s.std(ddof=1))),
        retention_min=("utility_retention", "min"),
        retention_max=("utility_retention", "max"),
        div_post_mean=("subgroup_divergence_post", "mean"),
        div_post_std=("subgroup_divergence_post", lambda s: float(s.std(ddof=1))),
        div_pre_mean=("subgroup_divergence_pre", "mean"),
        optimal_utility_b_mean=("optimal_utility_b", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return summary
