"""Run Experiment 3 against real judge APIs, phase by phase.

Usage (one model family per invocation; artifacts land under
results/exp3_live/<model>/ and every raw response is cached to
data/exp3_cache/<model>.jsonl so reruns replay for free):

    python scripts/run_experiment3_live.py phase0 --backend anthropic --model claude-3-5-haiku-latest
    python scripts/run_experiment3_live.py phase1 --backend anthropic --model claude-3-5-haiku-latest
    python scripts/run_experiment3_live.py phase1 --backend local --model llama-3.1-8b --subset crossfamily
    python scripts/run_experiment3_live.py gate            # combined go/no-go over all families
    python scripts/run_experiment3_live.py phase2 --backend anthropic --model claude-3-5-haiku-latest
    python scripts/run_experiment3_live.py phase3 --backend anthropic --model claude-3-5-haiku-latest

Discipline encoded here rather than left to memory:

- Every elicitor gets the JSONL response cache (replayable runs; the log ships
  as supplementary material).
- Phases 2 and 3 refuse to run unless the pre-registered Phase 1 go/no-go gate
  passed (combined across >= 2 families via the ``gate`` command); --force
  overrides with a loud warning.
- Each phase prints its expected judge-call count and requires --yes (or an
  interactive confirmation) before spending API money.
- Parse-failure rates are reported after every phase; > 2% means tighten the
  prompt, not the parser (spec section 1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.exp3.cache import ResponseCache
from src.exp3.elicitation import Condition, PairwiseElicitor, elicit_v_hat, sample_pairs
from src.exp3.judge import AnthropicJudge, AzureOpenAIJudge, Judge, LocalJudge
from src.exp3.menu import Menu
from src.exp3.phase0 import checklist_from_result, quality_balance_report, separability_report
from src.exp3.phase1 import Phase1Config, evaluate_go_no_go, format_agreement, run_phase1
from src.exp3.phase2 import Phase2Config, run_phase2
from src.exp3.phase3 import Phase3Config, run_phase3
from src.exp3.surrogate import SurrogateUtility, fit_surrogate
from src.exp3.templates import TemplateName

LIVE_DIR = ROOT / "results" / "exp3_live"
CACHE_DIR = ROOT / "data" / "exp3_cache"


def rel(p: Path) -> Path:
    """Repo-relative form of ``p``, so logs never embed an absolute home path."""
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p

# Rough per-call token budget for cost previews (spec section 7: ~600 in / 5 out).
TOKENS_IN_PER_CALL = 600
TOKENS_OUT_PER_CALL = 5


def sanitize(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def build_judge(args: argparse.Namespace) -> Judge:
    if args.backend == "anthropic":
        return AnthropicJudge(args.model)
    if args.backend == "azure":
        return AzureOpenAIJudge(args.model)
    if args.backend == "local":
        return LocalJudge(args.model, base_url=args.local_base_url)
    raise ValueError(f"Unknown backend {args.backend!r}")


def model_dir(args: argparse.Namespace) -> Path:
    d = LIVE_DIR / sanitize(args.model)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cache(args: argparse.Namespace) -> ResponseCache:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ResponseCache(CACHE_DIR / f"{sanitize(args.model)}.jsonl")


def confirm_spend(n_calls: int, args: argparse.Namespace) -> None:
    mtok_in = n_calls * TOKENS_IN_PER_CALL / 1e6
    mtok_out = n_calls * TOKENS_OUT_PER_CALL / 1e6
    print(
        f"\nExpected judge calls (before cache replay): ~{n_calls:,}"
        f"  (~{mtok_in:.1f}M input tokens, ~{mtok_out:.2f}M output tokens)"
    )
    if args.yes:
        return
    reply = input("Proceed with live API spend? [y/N] ").strip().lower()
    if reply not in ("y", "yes"):
        print("Aborted before any API call.")
        sys.exit(1)


def check_gate(args: argparse.Namespace) -> None:
    """Phases 2-3 run only behind the pre-registered Phase 1 gate."""
    combined = LIVE_DIR / "go_no_go.json"
    per_model = model_dir(args) / "go_no_go.json"
    gate_file = combined if combined.exists() else per_model
    if not gate_file.exists():
        if args.force:
            print("WARNING: no go/no-go record found; proceeding under --force.")
            return
        sys.exit(
            "No go/no-go decision found. Run phase1 (and `gate` for the combined "
            "cross-family decision) first, or pass --force."
        )
    decision = json.loads(gate_file.read_text())
    if decision.get("decision") != "go":
        if args.force:
            print(f"WARNING: gate says NO-GO ({decision.get('reason')}); proceeding under --force.")
            return
        sys.exit(
            f"Gate says NO-GO: {decision.get('reason')}\n"
            "Per the pre-registered protocol the insensitivity result is the finding; "
            "phases 2-3 do not run. Pass --force to override."
        )
    src = "combined" if gate_file == combined else "single-family (run `gate` for the combined decision)"
    print(f"Gate: GO ({src}) -- {decision.get('reason')}")


def report_elicitor_stats(elicitor: PairwiseElicitor | None, label: str) -> None:
    if elicitor is None:
        return
    print(f"[{label}] live calls: {elicitor.n_calls:,}  cache replays: {elicitor.n_cache_hits:,}")


def _menu_fingerprint(menu: Menu) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256("\x00".join(menu.texts).encode("utf-8")).hexdigest()[:16]


V_HAT_SEEDS = (7, 8)


async def ensure_v_hat(judge: Judge, menu: Menu, args: argparse.Namespace, cache: ResponseCache) -> np.ndarray:
    """Elicit (or reload) v_hat, invalidating the artifact if the menu changed.

    Pools two independent pair samples (seeds 7 and 8, ``--v-hat-pairs`` each)
    into one weighted BT fit: single-seed cluster-mean z estimates carry ~0.15-0.2
    of sampling noise, enough to flip the 0.5 quality-balance gate on its own.
    """
    out = model_dir(args)
    path = out / "v_hat.npy"
    meta_path = out / "v_hat_meta.json"
    fp = _menu_fingerprint(menu)
    meta_want = {
        "menu_fingerprint": fp,
        "n_pairs": args.v_hat_pairs,
        "seeds": list(V_HAT_SEEDS),
    }
    if path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == meta_want:
            return np.load(path)
        print("Menu or config changed since v_hat was elicited; re-eliciting.")
    print("Eliciting no-context v_hat (template 2.6, pooled over 2 pair samples)...")
    from src.exp3.elicitation import Condition, sample_pairs  # noqa: PLC0415
    from src.exp3.templates import TemplateName  # noqa: PLC0415
    from src.reward_model import fit_bradley_terry_weighted  # noqa: PLC0415

    frames = []
    for seed in V_HAT_SEEDS:
        elicitor = PairwiseElicitor(judge, menu, cache=cache, seed=seed)
        pairs = sample_pairs(menu.n_items, args.v_hat_pairs, np.random.default_rng(seed))
        res = await elicitor.elicit_pairs(pairs, Condition(TemplateName.NO_CONTEXT))
        frames.append(res.pairs)
    df = pd.concat(frames, ignore_index=True)
    model = fit_bradley_terry_weighted(
        df["item_a"].to_numpy(),
        df["item_b"].to_numpy(),
        df["p_prefer_a"].to_numpy(),
        menu.n_items,
        weight=df["n_votes"].to_numpy(),
    )
    v_hat = model.score_all()
    np.save(path, v_hat)
    meta_path.write_text(json.dumps(meta_want))
    return v_hat


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


async def cmd_phase0(args: argparse.Namespace) -> None:
    menu = Menu.load()
    judge = build_judge(args)
    cache = get_cache(args)
    out = model_dir(args)

    n_calls = args.v_hat_pairs * 5 * len(V_HAT_SEEDS) + 50 * 5 * 2
    confirm_spend(n_calls, args)

    sep = separability_report(menu)
    print(
        f"Separability: recoverability acc={sep.recoverability_accuracy:.3f} "
        f"(chance={sep.chance:.2f}) silhouette={sep.silhouette:.3f} passed={sep.passed}"
    )

    v_hat = await ensure_v_hat(judge, menu, args, cache)
    qb = quality_balance_report(v_hat, menu)
    print(f"Quality balance: max_abs_z={qb.max_abs_z:.3f} passed={qb.passed}")
    if not qb.passed:
        print(
            "WARNING: a cluster dominates on intrinsic quality; per the design, "
            "regenerate items for that cluster before Phase 1."
        )

    # Prompt checklist on the two load-bearing templates (spec section 3).
    rng = np.random.default_rng(11)
    elicitor = PairwiseElicitor(judge, menu, cache=cache, seed=11)
    pairs = sample_pairs(menu.n_items, 50, rng)
    reports = {}
    for name, cond in (
        ("no_context", Condition(TemplateName.NO_CONTEXT)),
        (
            "group_b_contextual",
            Condition(TemplateName.GROUP_B_CONTEXTUAL, distribution=menu.deployment_distribution(5, 0.5)),
        ),
    ):
        res = await elicitor.elicit_pairs(pairs, cond)
        chk = checklist_from_result(res)
        reports[name] = {
            "parse_failure_rate": chk.parse_failure_rate,
            "position_bias": chk.position_bias,
            "repeat_agreement": chk.repeat_agreement,
            "passed": chk.passed,
        }
        print(
            f"Checklist [{name}]: parse_fail={chk.parse_failure_rate:.3f} "
            f"bias={chk.position_bias:.3f} agree={chk.repeat_agreement:.3f} passed={chk.passed}"
        )

    (out / "phase0_report.json").write_text(
        json.dumps(
            {
                "separability": {
                    "recoverability_accuracy": sep.recoverability_accuracy,
                    "silhouette": sep.silhouette,
                    "passed": sep.passed,
                },
                "quality_balance": {"max_abs_z": qb.max_abs_z, "passed": qb.passed},
                "checklist": reports,
            },
            indent=2,
        )
    )
    report_elicitor_stats(elicitor, "phase0")
    print(f"Phase 0 artifacts in {rel(out)}")


def phase1_config(args: argparse.Namespace) -> Phase1Config:
    if args.subset == "crossfamily":
        # 200-pair cross-family subset (design section 2): enough to check the
        # sign and rough magnitude of lambda_hat on non-workhorse families.
        return Phase1Config(
            n_pairs_contextual=200, n_pairs_stated=100, n_pairs_control=100, keep_raw=True
        )
    return Phase1Config(
        n_pairs_contextual=800, n_pairs_stated=300, n_pairs_control=150, keep_raw=True
    )


def estimate_phase1_calls(config: Phase1Config) -> int:
    per_cluster = (
        config.n_pairs_contextual  # contextual
        + config.n_pairs_stated  # stated
        + 2 * config.n_pairs_control  # placebo + contamination
    )
    return len(config.probe_clusters) * len(config.p_grid) * per_cluster * config.n_repeats


async def cmd_phase1(args: argparse.Namespace) -> None:
    menu = Menu.load()
    judge = build_judge(args)
    cache = get_cache(args)
    out = model_dir(args)
    config = phase1_config(args)
    confirm_spend(estimate_phase1_calls(config), args)

    result = await run_phase1({sanitize(args.model): judge}, menu, config, cache=cache)
    result.summary.to_csv(out / "phase1_summary.csv", index=False)
    result.curves.to_csv(out / "phase1_curves.csv", index=False)
    assert result.raw is not None
    result.raw.to_csv(out / "phase1_raw.csv", index=False)
    # Per-model file records whether THIS family shows the effect (min_families=1);
    # the pre-registered >= 2-family decision is the `gate` command's combined file.
    single = evaluate_go_no_go(result.summary, min_families=1)
    (out / "go_no_go.json").write_text(json.dumps(single, indent=2))

    print("\nlambda_hat by format and probe cluster:")
    for _, r in result.summary.iterrows():
        flag = "*" if r["ci_excludes_zero"] else " "
        print(
            f"  {r['format']:>22} cluster {int(r['probe_cluster'])}: "
            f"{r['lambda_hat']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{flag} "
            f"form={r['best_form']} parse_fail={r['parse_failure_rate']:.3f}"
        )
    fa = format_agreement(result.summary)
    print(f"\nStated-vs-contextual agreement: {fa.to_dict('records')}")
    print(f"\nSingle-family go/no-go: {single['decision'].upper()} -- {single['reason']}")
    print("Run the `gate` command once >= 2 families have phase1 results.")

    worst = float(result.summary["parse_failure_rate"].max())
    if worst > 0.02:
        print(f"WARNING: parse-failure rate {worst:.1%} exceeds the 2% gate; tighten the prompt.")


def cmd_gate(_args: argparse.Namespace) -> None:
    frames = []
    for f in sorted(LIVE_DIR.glob("*/phase1_summary.csv")):
        frames.append(pd.read_csv(f))
        print(f"loaded {f}")
    if not frames:
        sys.exit("No phase1_summary.csv files found under results/exp3_live/*/")
    combined = pd.concat(frames, ignore_index=True)
    decision = evaluate_go_no_go(combined)
    (LIVE_DIR / "go_no_go.json").write_text(json.dumps(decision, indent=2))
    print(f"\nCombined go/no-go: {decision['decision'].upper()} -- {decision['reason']}")
    print(f"Families passing: {decision['families_passing']}")


async def cmd_phase2(args: argparse.Namespace) -> None:
    menu = Menu.load()
    judge = build_judge(args)
    cache = get_cache(args)
    out = model_dir(args)
    check_gate(args)

    config = Phase2Config(
        n_pairs_per_t=args.pairs_per_t,
        target_temperature=args.target_temperature,
        target_transform=args.target_transform,
    )
    n_calls = 2 * (config.n_train_pairs + len(config.t_grid) * config.n_pairs_per_t) * config.n_repeats
    confirm_spend(n_calls, args)

    family = sanitize(args.model)
    v_hat = await ensure_v_hat(judge, menu, args, cache)
    result = await run_phase2({family: judge}, menu, {family: v_hat}, config, cache=cache)
    result.results.to_csv(out / "phase2_results.csv", index=False)
    print(result.results.to_string(index=False))
    print(f"Phase 2 artifacts in {rel(out)}")


async def cmd_phase3(args: argparse.Namespace) -> None:
    menu = Menu.load()
    judge = build_judge(args)
    cache = get_cache(args)
    out = model_dir(args)
    check_gate(args)

    raw_path = out / "phase1_raw.csv"
    if not raw_path.exists():
        sys.exit("phase1_raw.csv not found; Phase 3's surrogate needs Phase 1 raw data.")
    raw = pd.read_csv(raw_path)
    family = sanitize(args.model)
    v_hat = await ensure_v_hat(judge, menu, args, cache)

    fit = fit_surrogate(raw, v_hat, menu, family=family, rng=np.random.default_rng(3))
    (out / "surrogate.json").write_text(
        json.dumps(
            {
                "form": fit.best_form,
                "alpha": fit.surrogate.alpha,
                "b": fit.surrogate.b,
                "c": fit.surrogate.c,
                "heldout_ll_best": fit.heldout_ll_best,
                "heldout_ll_baseline": fit.heldout_ll_baseline,
                "n_pairs": fit.n_pairs,
            },
            indent=2,
        )
    )
    fit.metrics.to_csv(out / "surrogate_metrics.csv", index=False)
    print(
        f"Surrogate: form={fit.best_form} alpha={fit.surrogate.alpha:.3f} b={fit.surrogate.b:.3f} "
        f"held-out LL {fit.heldout_ll_best:.4f} (no-prevalence baseline {fit.heldout_ll_baseline:.4f})"
    )
    if fit.heldout_ll_best <= fit.heldout_ll_baseline:
        print(
            "WARNING: surrogate does not beat the prevalence-free baseline; per the "
            "design's failure-mode plan, fall back to elicitation-in-the-loop on the "
            "10-dim cluster simplex."
        )

    seeds = tuple(range(args.n_seeds))
    config = Phase3Config(
        seeds=seeds,
        n_eval_pairs=args.eval_pairs,
        optimum_within_cluster=args.optimum_within_cluster,
    )
    n_policies = 4 + 1 if config.include_deployment_conditioned else 3 + 1
    n_calls = len(seeds) * (
        2 * config.n_train_pairs_per_group + n_policies * config.n_eval_pairs
    ) * config.n_repeats
    confirm_spend(n_calls, args)

    result = await run_phase3(
        {family: judge}, menu, {family: v_hat}, {family: fit.surrogate}, config, cache=cache
    )
    result.results.to_csv(out / "phase3_results.csv", index=False)
    result.summary.to_csv(out / "phase3_summary.csv", index=False)
    print("\nRetention (fresh-judge measured):")
    print(result.summary.to_string(index=False))
    print(f"Phase 3 artifacts in {rel(out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=("anthropic", "azure", "local"), required=True)
        p.add_argument("--model", required=True, help="model id / deployment name")
        p.add_argument("--local-base-url", default="http://localhost:8000/v1")
        p.add_argument("--v-hat-pairs", type=int, default=800)
        p.add_argument("--yes", action="store_true", help="skip the spend confirmation")
        p.add_argument("--force", action="store_true", help="bypass the go/no-go gate")

    add_common(sub.add_parser("phase0", help="menu validation + prompt checklist"))
    p1 = sub.add_parser("phase1", help="judge psychometrics + single-family go/no-go")
    add_common(p1)
    p1.add_argument("--subset", choices=("full", "crossfamily"), default="full")
    sub.add_parser("gate", help="combined cross-family go/no-go from all phase1 results")
    p2 = sub.add_parser("phase2", help="Experiment 1 analogue")
    add_common(p2)
    p2.add_argument("--pairs-per-t", type=int, default=400)
    p2.add_argument(
        "--target-temperature", type=float, default=0.2,
        help="tau for pi_target = softmax(transform(v_hat) / tau); quantile "
        "taus live on a ~0.05-0.1 scale",
    )
    p2.add_argument(
        "--target-transform", choices=("standardize", "quantile"), default="standardize",
        help="use 'quantile' when the judge's v_hat has extreme outliers that "
        "collapse the softmax target to a single item at any temperature",
    )
    p3 = sub.add_parser("phase3", help="surrogate + Experiment 2 analogue")
    add_common(p3)
    p3.add_argument("--n-seeds", type=int, default=3)
    p3.add_argument("--eval-pairs", type=int, default=400)
    p3.add_argument(
        "--optimum-within-cluster", choices=("best", "uniform"), default="uniform",
        help="'uniform' keeps the deployment-consistent optimum inside the "
        "surrogate's measured domain (low per-item mass); 'best' concentrates "
        "each cluster's mass on its top item, which real judges may punish as "
        "verbatim repetition beyond what the cluster-level surrogate models",
    )

    args = parser.parse_args()
    if args.command == "gate":
        cmd_gate(args)
        return
    coro = {
        "phase0": cmd_phase0,
        "phase1": cmd_phase1,
        "phase2": cmd_phase2,
        "phase3": cmd_phase3,
    }[args.command](args)
    asyncio.run(coro)


if __name__ == "__main__":
    main()
