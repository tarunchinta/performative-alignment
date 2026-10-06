"""Run Experiment 2 validation and save outputs."""

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
from src.experiment2 import Experiment2Config, METHODS, run_experiment2

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

METHOD_LABELS = {
    "rlhf": "Standard RLHF",
    "vpl_inferred": "VPL (inferred)",
    "vpl_oracle": "VPL (oracle)",
}
METHOD_COLORS = {
    "rlhf": "tab:blue",
    "vpl_inferred": "tab:orange",
    "vpl_oracle": "tab:green",
}


def main() -> None:
    config = Experiment2Config(env=EnvironmentConfig(seed=42), seed=42)
    env = PreferenceEnvironment.create(config.env)

    items = np.arange(env.n_items)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(items, env.v)
    axes[0, 0].set_title("Intrinsic utility v(y)")
    axes[0, 1].bar(items, env.pi_ref, width=1.0)
    axes[0, 1].set_title("Reference policy D_{pi_ref}")
    for t in [0.0, 0.5, 1.0]:
        axes[1, 0].plot(items, env.utility_b(env.policy_at_t(t), 1.0), label=f"t={t}")
    axes[1, 0].legend()
    axes[1, 0].set_title("Group B utility U_B(y; D_{pi_t}), lambda=1")
    axes[1, 1].bar(items, env.policy_at_t(1.0), width=1.0)
    axes[1, 1].set_title("Target policy D_{pi_target} (t=1)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_environment.png", dpi=150)
    plt.close()

    result = run_experiment2(config)
    df = result.results
    lambdas = sorted(df["lambda"].unique())

    print("Utility retention by lambda and method:")
    pivot = df.pivot(index="lambda", columns="method", values="utility_retention")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    lam0 = df[df["lambda"] == 0.0]
    print("\nlambda=0 retention (expect ~1.0):")
    print(lam0[["method", "utility_retention"]].to_string(index=False))

    pos = df[df["lambda"] > 0]
    mean_ret = pos.groupby("method")["utility_retention"].mean().sort_values(ascending=False)
    print("\nMean retention at lambda>0:")
    print(mean_ret.to_string(float_format=lambda x: f"{x:.4f}"))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    for method in METHODS:
        sub = df[df["method"] == method].sort_values("lambda")
        ax1.plot(
            sub["lambda"],
            sub["utility_retention"],
            marker="o",
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax2.plot(
            sub["lambda"],
            sub["subgroup_divergence_post"],
            marker="s",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            color=METHOD_COLORS[method],
        )

    pre_ref = df.groupby("lambda")["subgroup_divergence_pre"].first()
    ax2.plot(
        pre_ref.index,
        pre_ref.values,
        color="gray",
        linestyle=":",
        linewidth=2,
        label="Pre-alignment (pi_ref)",
    )

    ax1.set_xlabel("Prevalence sensitivity lambda")
    ax1.set_ylabel("Utility retention (Group B)")
    ax2.set_ylabel("Subgroup output divergence (JS)")
    ax1.set_title("Pluralistic methods under deployment-dependent utility")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_headline.png", dpi=150)
    plt.close()

    df.to_csv(RESULTS_DIR / "exp2_results.csv", index=False)
    print(f"\nSaved outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
