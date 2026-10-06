"""Run the full Experiment 3 pipeline (Phases 0-3) against the SyntheticJudge.

This is the free, end-to-end pipeline verification (spec section 1.2): the
synthetic judge answers pairwise questions from the known U_B of Experiments
1-2, so every stage has a ground truth to recover:

- Phase 0/1: lambda_hat with correct sign and CI; placebo/contamination ~ 0.
- Surrogate: best form = linear (the true g is linear), slope b ~= beta * lam.
- Phase 2: Group B local-not-global degradation; Group A flat (negative control).
- Phase 3: retention < 1 for all methods, and measured (fresh-query BT) retention
  tracking the exact retention computed from the known utility.

If all asserts pass, elicitation, BT fitting, the surrogate, policy optimization,
and the fresh-query evaluation are verified before any API dollar is spent. Real
judges (Anthropic/Azure/Local) drop into ``judges`` unchanged -- see
scripts/run_experiment3_live.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exp3.elicitation import Condition, PairwiseElicitor, elicit_v_hat, sample_pairs
from src.exp3.judge import SyntheticJudge
from src.exp3.menu import Menu
from src.exp3.phase0 import (
    checklist_from_result,
    quality_balance_report,
    separability_report,
)
from src.exp3.phase1 import (
    CONTAMINATION,
    CONTEXTUAL,
    PLACEBO,
    STATED,
    Phase1Config,
    format_agreement,
    run_phase1,
)
from src.exp3.phase2 import Phase2Config, run_phase2
from src.exp3.phase3 import Phase3Config, run_phase3
from src.exp3.surrogate import SurrogateUtility, fit_surrogate, solve_deployment_consistent
from src.exp3.templates import TemplateName

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FORMAT_LABELS = {
    CONTEXTUAL: "Contextual exposure",
    STATED: "Stated prevalence",
    PLACEBO: "Placebo (control)",
    CONTAMINATION: "Group A contam. (control)",
}
FORMAT_COLORS = {
    CONTEXTUAL: "tab:blue",
    STATED: "tab:green",
    PLACEBO: "tab:gray",
    CONTAMINATION: "tab:red",
}

# beta is the synthetic judge's decisiveness. The menu's designed v-spread is
# small (quality balance), so beta=8 keeps pairwise signals above vote noise --
# the validation tests harness recovery, which needs a recoverable judge; the
# low-SNR stress case is covered by Phase 1's own controls.
FAMILIES = {"synth_lam1.0": 1.0, "synth_lam2.0": 2.0}
BETA = 8.0
HIGH_FAMILY = "synth_lam2.0"


async def run_prompt_checklist(judge: SyntheticJudge, menu: Menu, seed: int):
    elicitor = PairwiseElicitor(judge, menu, seed=seed)
    pairs = sample_pairs(menu.n_items, 50, np.random.default_rng(seed))
    cond = Condition(TemplateName.GROUP_B_CONTEXTUAL, distribution=menu.deployment_distribution(5, 0.5))
    res = await elicitor.elicit_pairs(pairs, cond)
    return checklist_from_result(res)


def plot_functional_form(curves, summary, family: str) -> None:
    ctx = curves[(curves["family"] == family) & (curves["format"] == CONTEXTUAL)]
    clusters = sorted(ctx["probe_cluster"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(clusters)))
    for cluster, color in zip(clusters, colors):
        sub = ctx[ctx["probe_cluster"] == cluster].sort_values("p")
        row = summary[
            (summary["family"] == family)
            & (summary["format"] == CONTEXTUAL)
            & (summary["probe_cluster"] == cluster)
        ].iloc[0]
        ax.plot(
            sub["p"], sub["delta"], marker="o", color=color,
            label=f"cluster {cluster} (best: {row['best_form']})",
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Cluster prevalence p")
    ax.set_ylabel("DiD contrast  mean u_hat(probe) - mean u_hat(other)")
    ax.set_title(f"Measured prevalence response g_hat (contextual, {family})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_lambda_functional_form.png", dpi=150)
    plt.close()


def plot_psychometrics(summary, family: str) -> None:
    fam = summary[summary["family"] == family]
    formats = [CONTEXTUAL, STATED, PLACEBO, CONTAMINATION]
    clusters = sorted(fam["probe_cluster"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(formats)
    x = np.arange(len(clusters))
    for i, fmt in enumerate(formats):
        vals, los, his = [], [], []
        for cluster in clusters:
            row = fam[(fam["format"] == fmt) & (fam["probe_cluster"] == cluster)].iloc[0]
            vals.append(row["lambda_hat"])
            los.append(row["lambda_hat"] - row["ci_lo"])
            his.append(row["ci_hi"] - row["lambda_hat"])
        ax.bar(
            x + i * width, vals, width, yerr=[los, his], capsize=3,
            color=FORMAT_COLORS[fmt], label=FORMAT_LABELS[fmt], alpha=0.85,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(formats) - 1) / 2)
    ax.set_xticklabels([f"cluster {c}" for c in clusters])
    ax.set_ylabel("Empirical sensitivity lambda_hat (95% bootstrap CI)")
    ax.set_title(f"Phase 1 judge psychometrics ({family})")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_psychometrics.png", dpi=150)
    plt.close()


def plot_phase2(results, family: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, metric, title in zip(
        axes,
        ("local_accuracy", "spearman_rho"),
        ("Local pairwise accuracy of static r_hat", "Global Spearman rho of static r_hat"),
    ):
        for persona, color in (("group_a", "tab:red"), ("group_b", "tab:blue")):
            sub = results[(results["family"] == family) & (results["persona"] == persona)]
            sub = sub.sort_values("t")
            ax.plot(sub["t"], sub[metric], marker="o", color=color, label=persona)
        ax.set_xlabel("Interpolation t (pi_ref -> pi_target)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("accuracy / rho")
    fig.suptitle(f"Phase 2: local-not-global degradation ({family})", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_phase2_local_global.png", dpi=150)
    plt.close()


def plot_phase3(summary) -> None:
    families = list(summary["family"].unique())
    methods = list(summary["method"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(families)
    x = np.arange(len(methods))
    for i, family in enumerate(families):
        sub = summary[summary["family"] == family].set_index("method")
        vals = [sub.loc[m, "retention_mean"] for m in methods]
        errs = [sub.loc[m, "retention_std"] for m in methods]
        ax.bar(x + i * width, vals, width, yerr=errs, capsize=3, label=family, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x + width * (len(families) - 1) / 2)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Utility retention (fresh-judge measured, mean +- sd)")
    ax.set_title("Phase 3: retention under judge-defined utility")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_phase3_retention.png", dpi=150)
    plt.close()


def true_centered_realized(pi: np.ndarray, menu: Menu, lam: float) -> float:
    """Exact E_pi[U_B - mean_menu(U_B)] from the known synthetic utility.

    Mean-centered over the menu to match the BT mean-zero anchor of the measured
    utilities (the overall scale beta cancels in retention ratios).
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()
    prev = menu.cluster_prevalence(pi)
    u = menu.v - lam * prev[menu.cluster_of]
    u = u - u.mean()
    return float(pi @ u)


async def main() -> None:
    menu = Menu.load()
    print(f"Menu: {menu.n_items} items, {menu.n_clusters} clusters, domain='{menu.domain}'")

    # --- Phase 0 -------------------------------------------------------------
    print("\n=== Phase 0: menu validation ===")
    sep = separability_report(menu)
    print(
        f"Separability: cluster-label recoverability acc={sep.recoverability_accuracy:.3f} "
        f"(chance={sep.chance:.2f})  silhouette={sep.silhouette:.3f}  passed={sep.passed}"
    )

    judges = {
        name: SyntheticJudge(menu.v, menu.cluster_of, lam=lam, beta=BETA, model_id=name, seed=i + 1)
        for i, (name, lam) in enumerate(FAMILIES.items())
    }

    # Elicited intrinsic utility v_hat per family (template 2.6); defines D_ref.
    v_hat = {
        name: await elicit_v_hat(judge, menu, n_pairs=800, seed=7)
        for name, judge in judges.items()
    }

    qb = quality_balance_report(v_hat[HIGH_FAMILY], menu)
    print(
        f"Quality balance: max |cluster mean - grand| / item SD = {qb.max_abs_z:.3f}  "
        f"passed={qb.passed}"
    )
    checklist = await run_prompt_checklist(judges[HIGH_FAMILY], menu, seed=11)
    print(
        f"Prompt checklist: parse_fail={checklist.parse_failure_rate:.3f}, "
        f"position_bias={checklist.position_bias:.3f}, "
        f"repeat_agreement={checklist.repeat_agreement:.3f}"
    )
    print(
        "  (parse/bias are the harness gates; modest synthetic repeat-agreement just "
        "reflects near-tie pairs in the menu, not a harness fault.)"
    )

    # --- Phase 1 -------------------------------------------------------------
    print("\n=== Phase 1: judge psychometrics (2 synthetic families) ===")
    config = Phase1Config(
        probe_clusters=(1, 5, 8),
        n_pairs_contextual=250,
        n_pairs_stated=150,
        n_pairs_control=150,
        n_boot=60,
        keep_raw=True,
    )
    result = await run_phase1(judges, menu, config)
    summary = result.summary
    summary.to_csv(RESULTS_DIR / "exp3_phase1_results.csv", index=False)
    result.curves.to_csv(RESULTS_DIR / "exp3_phase1_curves.csv", index=False)
    assert result.raw is not None
    result.raw.to_csv(RESULTS_DIR / "exp3_phase1_raw.csv", index=False)

    print("\nContextual-exposure lambda_hat by family and probe cluster:")
    ctx = summary[summary["format"] == CONTEXTUAL]
    for _, r in ctx.iterrows():
        flag = "*" if r["ci_excludes_zero"] else " "
        print(
            f"  {r['family']:>12} cluster {int(r['probe_cluster'])}: "
            f"lambda_hat={r['lambda_hat']:+.3f}  CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] {flag}  "
            f"form={r['best_form']}"
        )

    print("\nControl lambda_hat (expect CI to include 0):")
    for fmt in (PLACEBO, CONTAMINATION):
        rows = summary[summary["format"] == fmt]
        n_excl = int(rows["ci_excludes_zero"].sum())
        print(
            f"  {FORMAT_LABELS[fmt]}: mean lambda_hat={rows['lambda_hat'].mean():+.3f}, "
            f"{n_excl}/{len(rows)} cells exclude 0"
        )

    fa = format_agreement(summary)
    print("\nStated-vs-contextual lambda_hat agreement (Spearman):")
    for _, r in fa.iterrows():
        print(f"  {r['family']}: rho={r['stated_contextual_spearman']:.3f}")

    print(f"\nGo/No-Go: {result.go_no_go['decision'].upper()} -- {result.go_no_go['reason']}")

    plot_functional_form(result.curves, summary, HIGH_FAMILY)
    plot_psychometrics(summary, HIGH_FAMILY)

    # --- Phase 1 validation asserts (synthetic recovery) ----------------------
    print("\n=== Validation: Phase 1 ===")
    assert result.go_no_go["decision"] == "go", "Expected go decision under synthetic Group B."
    ctx_probe = ctx.groupby("probe_cluster")["ci_excludes_zero"].any()
    assert (ctx["lambda_hat"] > 0).mean() > 0.8, "Contextual lambda_hat should be positive (aversion)."
    assert int(ctx_probe.sum()) >= 2, "Need >=2 probe clusters with CI excluding 0."
    for fmt in (PLACEBO, CONTAMINATION):
        rows = summary[summary["format"] == fmt]
        assert int(rows["ci_excludes_zero"].sum()) <= 1, f"{fmt} controls should read ~0."
    mean_by_family = ctx.groupby("family")["lambda_hat"].mean()
    assert mean_by_family["synth_lam2.0"] > mean_by_family["synth_lam1.0"], (
        "Higher true lambda should yield larger lambda_hat (monotonicity)."
    )
    print("Phase 1 synthetic-recovery checks passed.")

    # --- Surrogate (design section 4) -----------------------------------------
    print("\n=== Surrogate utility surface ===")
    surrogates: dict[str, SurrogateUtility] = {}
    surrogate_rows = []
    for name, lam in FAMILIES.items():
        fit = fit_surrogate(
            result.raw, v_hat[name], menu, family=name, rng=np.random.default_rng(3)
        )
        surrogates[name] = fit.surrogate
        frame = fit.metrics.copy()
        frame.insert(0, "family", name)
        surrogate_rows.append(frame)
        true_b = BETA * lam
        print(
            f"  {name}: best form={fit.best_form}  alpha={fit.surrogate.alpha:.3f}  "
            f"b={fit.surrogate.b:.3f} (true beta*lam={true_b:.1f})  "
            f"held-out LL={fit.heldout_ll_best:.4f} vs no-prevalence {fit.heldout_ll_baseline:.4f}"
        )
        assert fit.best_form == "linear", f"True g is linear; got {fit.best_form} for {name}."
        assert fit.heldout_ll_best > fit.heldout_ll_baseline, (
            "Prevalence term must improve held-out likelihood."
        )
        assert abs(fit.surrogate.b - true_b) / true_b < 0.5, (
            f"Recovered slope b={fit.surrogate.b:.2f} too far from true {true_b:.2f}."
        )
    pd.concat(surrogate_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "exp3_surrogate_fit.csv", index=False
    )
    print("Surrogate recovery checks passed.")

    # --- Phase 2 (Experiment 1 analogue) ---------------------------------------
    print("\n=== Phase 2: miscalibration under deployment shift ===")
    phase2_config = Phase2Config(n_train_pairs=800, n_pairs_per_t=600)
    phase2 = await run_phase2(judges, menu, v_hat, phase2_config)
    phase2.results.to_csv(RESULTS_DIR / "exp3_phase2_results.csv", index=False)
    plot_phase2(phase2.results, HIGH_FAMILY)

    fam2 = phase2.results[phase2.results["family"] == HIGH_FAMILY]
    for metric in ("local_accuracy", "global_accuracy", "spearman_rho"):
        print(f"\n{metric} by persona and t (family = {HIGH_FAMILY}):")
        for persona in ("group_a", "group_b"):
            sub = fam2[fam2["persona"] == persona].sort_values("t")
            accs = "  ".join(f"t={r['t']:.1f}:{r[metric]:.3f}" for _, r in sub.iterrows())
            print(f"  {persona}: {accs}")

    print("\n=== Validation: Phase 2 ===")

    # Pool early (t <= 0.2) vs late (t >= 0.8) windows: the local pair count
    # shrinks with t, so endpoint contrasts are noisy while pooled global
    # metrics (~250+ confident pairs per cell) are stable. With cluster-level
    # prevalence the penalty reorders whole clusters, so Group B degradation is
    # visible globally as well as locally; the persona DiD is the clean test.
    def early_late(persona: str, metric: str) -> tuple[float, float]:
        sub = fam2[fam2["persona"] == persona]
        early = float(sub[sub["t"] <= 0.2][metric].mean())
        late = float(sub[sub["t"] >= 0.8][metric].mean())
        return early, late

    b_g_early, b_g_late = early_late("group_b", "global_accuracy")
    a_g_early, a_g_late = early_late("group_a", "global_accuracy")
    b_rho_early, b_rho_late = early_late("group_b", "spearman_rho")
    a_rho_early, a_rho_late = early_late("group_a", "spearman_rho")
    b_l_early, b_l_late = early_late("group_b", "local_accuracy")

    b_decline = b_g_early - b_g_late
    a_shift = a_g_early - a_g_late
    print(f"Group B global accuracy decline (early -> late): {b_decline:+.3f}")
    print(f"Group A global accuracy shift  (early -> late): {a_shift:+.3f}")
    print(f"Group B local accuracy decline  (early -> late): {b_l_early - b_l_late:+.3f}")
    print(f"Spearman decline  B: {b_rho_early - b_rho_late:+.3f}  A: {a_rho_early - a_rho_late:+.3f}")
    assert b_decline > 0.04, "Group B accuracy should decline under deployment shift."
    assert abs(a_shift) < 0.08, "Group A (negative control) should stay flat."
    assert (b_rho_early - b_rho_late) > (a_rho_early - a_rho_late) + 0.04, (
        "Group B rank correlation should degrade relative to Group A."
    )
    print("Phase 2 checks passed: Group-B-specific degradation reproduced.")

    # --- Phase 3 (Experiment 2 analogue) ---------------------------------------
    print("\n=== Phase 3: retention under judge-defined utility ===")
    phase3_config = Phase3Config(seeds=(0, 1, 2), n_train_pairs_per_group=800, n_eval_pairs=600)
    phase3 = await run_phase3(judges, menu, v_hat, surrogates, phase3_config)
    phase3.results.to_csv(RESULTS_DIR / "exp3_phase3_results.csv", index=False)
    phase3.summary.to_csv(RESULTS_DIR / "exp3_phase3_summary.csv", index=False)
    plot_phase3(phase3.summary)

    print("\nRetention (fresh-judge measured, mean +- sd over seeds):")
    for _, r in phase3.summary.iterrows():
        print(
            f"  {r['family']:>12} {r['method']:>24}: {r['retention_mean']:.3f} "
            f"+- {r['retention_std']:.3f}  (eff. support {r['effective_support_mean']:.1f}, "
            f"JS to ref {r['js_to_ref_mean']:.3f})"
        )

    # Exact retention from the known utility, for the measured-vs-true check.
    print("\n=== Validation: Phase 3 ===")
    diffs = []
    for name, lam in FAMILIES.items():
        truth = SurrogateUtility(
            v_hat=menu.v, cluster_of=menu.cluster_of, form="linear", alpha=1.0, b=lam
        )
        pi_star_true = solve_deployment_consistent(truth, menu)
        star_true = true_centered_realized(pi_star_true, menu, lam)
        fam_rows = phase3.results[
            (phase3.results["family"] == name) & (phase3.results["method"] != "optimum")
        ]
        for _, r in fam_rows.iterrows():
            pi = phase3.policies[(name, int(r["seed"]), r["method"])]
            true_ret = true_centered_realized(pi, menu, lam) / star_true
            diffs.append(abs(float(r["utility_retention"]) - true_ret))
            if name == HIGH_FAMILY:
                print(
                    f"  seed {int(r['seed'])} {r['method']:>24}: measured={r['utility_retention']:.3f} "
                    f"true={true_ret:.3f}"
                )
    mean_diff = float(np.mean(diffs))
    print(f"Mean |measured - true| retention gap: {mean_diff:.3f}")

    high = phase3.summary[phase3.summary["family"] == HIGH_FAMILY]
    assert (high["retention_mean"] < 0.95).all(), (
        "All methods (oracle included) should retain meaningfully below 1."
    )
    assert mean_diff < 0.25, "Measured retention should track exact retention."
    corr_ok = phase3.results["fresh_surrogate_corr"].mean()
    print(f"Mean corr(fresh u_hat, surrogate utility): {corr_ok:.3f}")
    assert corr_ok > 0.6, "Fresh-query utilities should agree with the surrogate surface."
    print("Phase 3 checks passed: shared collapse + oracle failure reproduced.")

    print(f"\nAll synthetic end-to-end checks passed. Outputs in {RESULTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
