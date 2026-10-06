"""Run Experiment 1 validation and save outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment import EnvironmentConfig, PreferenceEnvironment
from src.experiment1 import Experiment1Config, run_experiment1

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    config = Experiment1Config(env=EnvironmentConfig(seed=42), seed=42)
    env = PreferenceEnvironment.create(config.env)

    items = np.arange(env.n_items)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(items, env.v)
    axes[0, 1].bar(items, env.pi_ref, width=1.0)
    for t in [0.0, 0.5, 1.0]:
        axes[1, 0].plot(items, env.utility_b(env.policy_at_t(t), 1.0), label=f"t={t}")
    axes[1, 0].legend()
    axes[1, 1].bar(items, env.policy_at_t(1.0), width=1.0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_environment.png", dpi=150)
    plt.close()

    result = run_experiment1(config)
    df = result.results
    print(f"Train acc: {result.train_accuracy:.3f}")
    print(f"Val acc:   {result.val_accuracy:.3f}")

    lambdas = sorted(df["lambda"].unique())
    lam0 = df[df["lambda"] == 0.0].sort_values("js_divergence")
    t0 = df[df["t"] == 0.0].sort_values("lambda")
    print(
        f"lam0 cal err range: {lam0['calibration_error'].min():.4f} -> "
        f"{lam0['calibration_error'].max():.4f}"
    )
    print("t0 by lambda:")
    print(t0[["lambda", "calibration_error", "spearman_rho"]].to_string(index=False))

    for lam in [v for v in lambdas if v > 0]:
        sub = df[df["lambda"] == lam].sort_values("js_divergence")
        diffs = sub["calibration_error"].diff().dropna()
        violations = int((diffs < -0.02).sum())
        print(
            f"lam={lam:g}: violations={violations}/{len(diffs)}, "
            f"max_err={sub['calibration_error'].max():.4f}"
        )

    max_js_rows = df.loc[df.groupby("lambda")["js_divergence"].idxmax()]
    print("cal error at max JS by lambda:")
    print(max_js_rows[["lambda", "js_divergence", "calibration_error"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lambdas)))
    for lam, color in zip(lambdas, colors):
        sub = df[df["lambda"] == lam].sort_values("js_divergence")
        ax.plot(
            sub["js_divergence"],
            sub["calibration_error"],
            marker="o",
            markersize=4,
            label=f"λ={lam:g}",
            color=color,
        )
    ax.set_xlabel("JS(D_{π_t}, D_{π_ref})")
    ax.set_ylabel("Calibration error (1 − pairwise accuracy, Group B)")
    ax.set_title("Reward model miscalibration under deployment shift")
    ax.legend(title="Prevalence sensitivity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_calibration_curve.png", dpi=150)
    plt.close()

    df.to_csv(RESULTS_DIR / "exp1_results.csv", index=False)
    print(f"Saved outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
