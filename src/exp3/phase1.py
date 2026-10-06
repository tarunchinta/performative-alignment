"""Phase 1 orchestration: probe-cluster x prevalence sweep, controls, go/no-go gate.

For each judge "family" and each probe cluster, elicit preferences at a grid of
cluster prevalences under two conditioning formats (contextual exposure = the
load-bearing condition; stated prevalence = high-power secondary), plus two
controls (placebo prevalence, Group A contamination). Estimate the empirical
sensitivity lambda_hat with a bootstrap CI per cell, then apply the pre-registered
go/no-go rule (spec section 3).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from src.exp3.cache import ResponseCache
from src.exp3.elicitation import Condition, PairwiseElicitor, sample_pairs
from src.exp3.judge import Judge
from src.exp3.menu import Menu
from src.exp3.psychometrics import PsychometricResult, estimate_psychometrics
from src.exp3.templates import TemplateName

CONTEXTUAL = TemplateName.GROUP_B_CONTEXTUAL.value
STATED = TemplateName.GROUP_B_STATED.value
PLACEBO = TemplateName.PLACEBO.value
CONTAMINATION = TemplateName.GROUP_A_CONTAMINATION.value


@dataclass
class Phase1Config:
    probe_clusters: tuple[int, ...] = (1, 5, 8)
    p_grid: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.80)
    n_pairs_contextual: int = 800
    n_pairs_stated: int = 300
    n_pairs_control: int = 300
    n_repeats: int = 5
    k_exposure: int = 20
    n_boot: int = 200
    max_concurrency: int = 32
    seed: int = 42
    keep_raw: bool = False  # retain pair-level data (needed to fit the surrogate)


@dataclass
class Phase1Result:
    config: Phase1Config
    summary: pd.DataFrame  # one row per (family, probe_cluster, format)
    curves: pd.DataFrame  # per (family, probe_cluster, format, p): DiD contrast delta
    go_no_go: dict = field(default_factory=dict)
    # Tidy pair-level data across all cells (family, format, probe_cluster, p,
    # item_a, item_b, p_prefer_a, n_votes); None unless config.keep_raw.
    raw: pd.DataFrame | None = None


def _condition_for(fmt: str, menu: Menu, cluster: int, p: float) -> Condition:
    if fmt == CONTEXTUAL:
        return Condition(TemplateName.GROUP_B_CONTEXTUAL, distribution=menu.deployment_distribution(cluster, p))
    if fmt == STATED:
        return Condition(TemplateName.GROUP_B_STATED, probe_cluster=cluster, p_c=p)
    if fmt == PLACEBO:
        return Condition(TemplateName.PLACEBO, p_placebo=p)
    if fmt == CONTAMINATION:
        return Condition(TemplateName.GROUP_A_CONTAMINATION, distribution=menu.deployment_distribution(cluster, p))
    raise ValueError(f"Unknown format {fmt!r}")


def _n_pairs_for(fmt: str, config: Phase1Config) -> int:
    if fmt == CONTEXTUAL:
        return config.n_pairs_contextual
    if fmt == STATED:
        return config.n_pairs_stated
    return config.n_pairs_control


async def _run_cell(
    elicitor: PairwiseElicitor,
    menu: Menu,
    fmt: str,
    cluster: int,
    config: Phase1Config,
    rng: np.random.Generator,
) -> tuple[PsychometricResult, dict[float, pd.DataFrame]]:
    """Sweep the prevalence grid for one (family, cluster, format) cell.

    The same pair set is reused across prevalence levels so the DiD contrast is
    paired across p (variance reduction); only the deployment context changes.
    """
    probe_items = menu.items_in_cluster(cluster)
    other_items = np.array([i for i in range(menu.n_items) if menu.cluster_of[i] != cluster])
    pairs = sample_pairs(menu.n_items, _n_pairs_for(fmt, config), rng)

    pairs_by_p: dict[float, pd.DataFrame] = {}
    repeats_by_p: dict[float, pd.DataFrame] = {}
    for p in config.p_grid:
        cond = _condition_for(fmt, menu, cluster, p)
        res = await elicitor.elicit_pairs(pairs, cond)
        pairs_by_p[p] = res.pairs
        repeats_by_p[p] = res.repeats

    psy = estimate_psychometrics(
        pairs_by_p,
        menu.n_items,
        probe_items,
        other_items,
        repeats_by_p=repeats_by_p,
        n_boot=config.n_boot,
        rng=rng,
    )
    return psy, pairs_by_p


async def run_phase1(
    judges: dict[str, Judge],
    menu: Menu,
    config: Phase1Config | None = None,
    *,
    cache: ResponseCache | None = None,
) -> Phase1Result:
    config = config or Phase1Config()
    formats = (CONTEXTUAL, STATED, PLACEBO, CONTAMINATION)

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    raw_frames: list[pd.DataFrame] = []

    n_cells = len(judges) * len(formats) * len(config.probe_clusters)
    progress = tqdm(total=n_cells, desc="phase1 cells")
    for fam_idx, (family, judge) in enumerate(judges.items()):
        elicitor = PairwiseElicitor(
            judge,
            menu,
            n_repeats=config.n_repeats,
            k_exposure=config.k_exposure,
            max_concurrency=config.max_concurrency,
            cache=cache,
            seed=config.seed + fam_idx,
        )
        for fmt_idx, fmt in enumerate(formats):
            for cluster in config.probe_clusters:
                cell_rng = np.random.default_rng([config.seed, fam_idx, fmt_idx, cluster])
                psy, pairs_by_p = await _run_cell(elicitor, menu, fmt, cluster, config, cell_rng)
                if config.keep_raw:
                    for p, df in pairs_by_p.items():
                        frame = df.copy()
                        frame.insert(0, "family", family)
                        frame.insert(1, "format", fmt)
                        frame.insert(2, "probe_cluster", cluster)
                        frame.insert(3, "p", float(p))
                        raw_frames.append(frame)
                lo, hi = psy.lambda_hat_ci
                summary_rows.append(
                    {
                        "family": family,
                        "probe_cluster": cluster,
                        "format": fmt,
                        "lambda_hat": psy.lambda_hat,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "ci_excludes_zero": psy.ci_excludes_zero,
                        "best_form": psy.functional_forms["best_form"],
                        "repeat_agreement": psy.consistency.get("repeat_agreement", float("nan")),
                        "position_bias": psy.consistency.get("position_bias", float("nan")),
                        "parse_failure_rate": psy.consistency.get("parse_failure_rate", float("nan")),
                    }
                )
                for p, delta in zip(psy.p_values, psy.deltas):
                    curve_rows.append(
                        {
                            "family": family,
                            "probe_cluster": cluster,
                            "format": fmt,
                            "p": float(p),
                            "delta": float(delta),
                        }
                    )
                progress.update(1)
    progress.close()

    summary = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_rows)
    go_no_go = evaluate_go_no_go(summary)
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else None
    return Phase1Result(config=config, summary=summary, curves=curves, go_no_go=go_no_go, raw=raw)


def evaluate_go_no_go(summary: pd.DataFrame, *, min_clusters: int = 2, min_families: int = 2) -> dict:
    """Apply the gate: contextual lambda_hat CI excludes 0 with consistent sign
    across >= ``min_clusters`` probe clusters and >= ``min_families`` families.

    If satisfied -> proceed (go). Otherwise the insensitivity result is itself the
    finding (no-go); both branches are publishable.
    """
    ctx = summary[summary["format"] == CONTEXTUAL]
    sig = ctx[ctx["ci_excludes_zero"]]

    # Dominant sign among significant contextual cells (prevalence-aversion => +).
    if len(sig) == 0:
        return {
            "decision": "no-go",
            "reason": "No contextual lambda_hat CI excludes 0.",
            "n_significant_cells": 0,
            "families_passing": [],
            "consistent_sign": None,
        }
    dominant_sign = float(np.sign(np.median(sig["lambda_hat"])))

    families_passing: list[str] = []
    for family, grp in sig.groupby("family"):
        consistent = grp[np.sign(grp["lambda_hat"]) == dominant_sign]
        if consistent["probe_cluster"].nunique() >= min_clusters:
            families_passing.append(str(family))

    go = len(families_passing) >= min_families
    return {
        "decision": "go" if go else "no-go",
        "reason": (
            f"{len(families_passing)} families with >= {min_clusters} consistent-sign "
            f"clusters (need >= {min_families})."
        ),
        "n_significant_cells": int(len(sig)),
        "families_passing": families_passing,
        "consistent_sign": dominant_sign,
    }


def format_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-family Spearman correlation of stated vs contextual lambda_hat across clusters."""
    rows: list[dict] = []
    for family, grp in summary.groupby("family"):
        ctx = grp[grp["format"] == CONTEXTUAL].set_index("probe_cluster")["lambda_hat"]
        sta = grp[grp["format"] == STATED].set_index("probe_cluster")["lambda_hat"]
        common = ctx.index.intersection(sta.index)
        if len(common) >= 2:
            rho, _ = spearmanr(ctx.loc[common], sta.loc[common])
        else:
            rho = float("nan")
        rows.append({"family": str(family), "stated_contextual_spearman": float(rho)})
    return pd.DataFrame(rows)
