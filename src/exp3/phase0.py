"""Phase 0 validation: menu separability, intrinsic-quality balance, prompt checklist.

Phase 0 confirms the frozen menu is usable before any Phase 1 judging: designed
clusters are separable in text space, intrinsic quality is matched across
clusters (so preference variation is stylistic, not quality-driven), and the
prompt templates elicit clean A/B answers (spec section 3 checklist).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from src.exp3.elicitation import ElicitationResult
from src.exp3.menu import Menu
from src.exp3.psychometrics import parse_failure_rate, position_bias, repeat_agreement


@dataclass
class SeparabilityReport:
    recoverability_accuracy: float  # cross-validated cluster-label accuracy
    chance: float  # 1 / n_clusters
    silhouette: float  # secondary diagnostic (near 0 is expected for fine styles)
    passed: bool


def separability_report(menu: Menu, *, min_ratio: float = 2.0) -> SeparabilityReport:
    """Confirm designed clusters are recoverable from text (Phase 0).

    Because all items share one topic, silhouette over TF-IDF is near zero even
    for well-designed style clusters (topic vocabulary swamps style). The
    design's operative criterion is that "cluster labels are recoverable from
    text", so the primary signal here is a cross-validated classification
    accuracy (char n-gram TF-IDF + logistic regression) measured against the
    1/n_clusters chance baseline; silhouette is retained as a secondary
    diagnostic. This is an offline stand-in for the embedding/LLM label-recovery
    pass; a real sentence embedding would only sharpen it.
    """
    labels = menu.cluster_of
    chance = 1.0 / menu.n_clusters

    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1),
        LogisticRegression(max_iter=2000, C=10.0),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc = float(cross_val_score(clf, menu.texts, labels, cv=cv).mean())

    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1).fit_transform(
        menu.texts
    )
    sil = float(silhouette_score(vectors, labels, metric="cosine"))

    return SeparabilityReport(
        recoverability_accuracy=acc,
        chance=chance,
        silhouette=sil,
        passed=acc >= min_ratio * chance,
    )


@dataclass
class QualityBalanceReport:
    cluster_means: np.ndarray
    grand_mean: float
    item_sd: float
    max_abs_z: float  # max |cluster_mean - grand_mean| / item_sd
    passed: bool


def quality_balance_report(
    v_hat: np.ndarray, menu: Menu, *, tol_sd: float = 0.5
) -> QualityBalanceReport:
    """Check no cluster's mean no-context BT utility exceeds ~tol_sd of the grand mean."""
    v_hat = np.asarray(v_hat, dtype=np.float64)
    grand = float(v_hat.mean())
    item_sd = float(v_hat.std())
    means = np.array(
        [v_hat[menu.items_in_cluster(c)].mean() for c in range(menu.n_clusters)]
    )
    max_abs_z = float(np.max(np.abs(means - grand)) / (item_sd + 1e-12))
    return QualityBalanceReport(
        cluster_means=means,
        grand_mean=grand,
        item_sd=item_sd,
        max_abs_z=max_abs_z,
        passed=max_abs_z <= tol_sd,
    )


@dataclass
class ChecklistReport:
    parse_failure_rate: float
    position_bias: float
    repeat_agreement: float
    passed: bool


def checklist_from_result(
    result: ElicitationResult,
    *,
    max_parse_fail: float = 0.02,
    bias_tol: float = 0.05,
    min_agreement: float = 0.7,
) -> ChecklistReport:
    """Prompt-iteration checklist (spec section 3, items 1-3) from elicited repeats."""
    pf = parse_failure_rate(result.repeats)
    bias = position_bias(result.repeats)
    agree = repeat_agreement(result.repeats)
    passed = (
        pf < max_parse_fail
        and abs(bias - 0.5) <= bias_tol
        and (agree >= min_agreement or np.isnan(agree))
    )
    return ChecklistReport(
        parse_failure_rate=pf,
        position_bias=bias,
        repeat_agreement=agree,
        passed=bool(passed),
    )
