"""Experiment 1: reward model miscalibration under deployment shift."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.environment import EnvironmentConfig, PreferenceEnvironment
from src.metrics import calibration_metrics, js_divergence
from src.reward_model import BradleyTerryRewardModel, fit_bradley_terry, pairwise_accuracy


@dataclass
class Experiment1Config:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    train_lam: float = 1.0  # lambda used when generating RM training preferences
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)
    t_values: tuple[float, ...] = tuple(i / 20 for i in range(21))
    n_train_pairs: int = 80_000
    n_val_pairs: int = 10_000
    n_eval_pairs: int = 20_000
    seed: int = 42


@dataclass
class Experiment1Result:
    config: Experiment1Config
    env: PreferenceEnvironment
    reward_model: BradleyTerryRewardModel
    train_accuracy: float
    val_accuracy: float
    results: pd.DataFrame


def run_experiment1(config: Experiment1Config | None = None) -> Experiment1Result:
    """Run full Experiment 1 sweep and return results DataFrame."""
    config = config or Experiment1Config()
    rng = np.random.default_rng(config.seed)

    env = PreferenceEnvironment.create(config.env)

    # Train static RM on preferences under pi_ref at train_lam
    winners, losers, _ = env.sample_preference_pairs(
        config.n_train_pairs,
        env.pi_ref,
        config.train_lam,
        rng=rng,
    )
    val_winners, val_losers, _ = env.sample_preference_pairs(
        config.n_val_pairs,
        env.pi_ref,
        config.train_lam,
        rng=rng,
    )

    rm = fit_bradley_terry(winners, losers, env.n_items)
    train_acc = pairwise_accuracy(rm, winners, losers)
    val_acc = pairwise_accuracy(rm, val_winners, val_losers)

    rm_scores = rm.score_all()
    rows: list[dict] = []

    for lam in tqdm(config.lambdas, desc="lambda sweep"):
        for t in config.t_values:
            pi_t = env.policy_at_t(t)
            true_u_b = env.utility_b(pi_t, lam)
            cal = calibration_metrics(
                rm_scores,
                true_u_b,
                distribution=pi_t,
                n_pairs=config.n_eval_pairs,
                rng=rng,
            )
            js = js_divergence(pi_t, env.pi_ref)
            rows.append(
                {
                    "lambda": lam,
                    "t": t,
                    "js_divergence": js,
                    "spearman_rho": cal["spearman_rho"],
                    "calibration_error": cal["calibration_error"],
                    "normalized_mse": cal["normalized_mse"],
                    "pairwise_ranking_accuracy": cal["pairwise_ranking_accuracy"],
                }
            )

    return Experiment1Result(
        config=config,
        env=env,
        reward_model=rm,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        results=pd.DataFrame(rows),
    )
