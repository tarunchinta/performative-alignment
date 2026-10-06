"""PairwiseElicitor: the backend-agnostic elicitation harness (spec section 1).

Everything that must be identical across judge backends lives here -- repeats,
order-swap debiasing, strict A/B parsing, per-(pair, repeat) seeded exposure
sampling, caching, and rate limiting -- so adapters stay ~15 lines and cannot
drift apart. Preference probability comes from vote fractions across repeats,
never logprobs (which are not uniformly available across families).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exp3.cache import ResponseCache, cache_key
from src.exp3.judge import Judge
from src.exp3.menu import Menu
from src.exp3.templates import (
    TEMPLATES,
    TemplateName,
    format_exposure_block,
    render,
)
from src.reward_model import fit_bradley_terry_weighted

# Formats that prepend a 20-item exposure block sampled i.i.d. from D_pi.
_EXPOSURE_FORMATS = {
    TemplateName.GROUP_B_CONTEXTUAL,
    TemplateName.GROUP_A_CONTAMINATION,
}


def parse_ab(raw: str) -> str | None:
    """Strict parse: accept only 'A' or 'B' after strip; else None (logged)."""
    s = raw.strip()
    return s if s in ("A", "B") else None


@dataclass
class Condition:
    """How to render a prompt and build judge meta for a template.

    - ``distribution``: item-level D_pi for exposure sampling (contextual/contamination).
    - ``probe_cluster`` / ``p_c``: stated-prevalence manipulation (fraction in [0, 1]).
    - ``p_placebo``: placebo prevalence fraction in [0, 1].
    """

    template_name: TemplateName
    distribution: np.ndarray | None = None
    probe_cluster: int | None = None
    p_c: float | None = None
    p_placebo: float | None = None


@dataclass
class ElicitationResult:
    pairs: pd.DataFrame  # one row per pair: item_a, item_b, p_prefer_a, n_votes, n_parse_fail
    repeats: pd.DataFrame  # one row per (pair, repeat): choice, picked_a, position_a, ...


class PairwiseElicitor:
    def __init__(
        self,
        judge: Judge,
        menu: Menu,
        *,
        n_repeats: int = 5,
        k_exposure: int = 20,
        max_concurrency: int = 16,
        cache: ResponseCache | None = None,
        seed: int = 0,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.judge = judge
        self.menu = menu
        self.n_repeats = n_repeats
        self.k_exposure = k_exposure
        self.cache = cache
        self._seed = seed
        self._sem = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        # Call accounting for live-run cost tracking (judge calls vs cache replays).
        self.n_calls = 0
        self.n_cache_hits = 0

    def _exemplars(self, cluster_id: int) -> tuple[str, str]:
        idx = self.menu.items_in_cluster(cluster_id)
        return self.menu.texts[int(idx[0])], self.menu.texts[int(idx[1])]

    def _build_prompt_and_meta(
        self,
        cond: Condition,
        first_id: int,
        second_id: int,
        rng: np.random.Generator,
    ) -> tuple[str, dict]:
        template = TEMPLATES[cond.template_name]
        first_text = self.menu.texts[first_id]
        second_text = self.menu.texts[second_id]
        meta: dict = {
            "format": cond.template_name.value,
            "first": first_id,
            "second": second_id,
        }

        if cond.template_name in _EXPOSURE_FORMATS:
            if cond.distribution is None:
                raise ValueError(f"{cond.template_name} requires a distribution.")
            exposure = self.menu.sample_exposure(cond.distribution, self.k_exposure, rng)
            block = format_exposure_block([self.menu.texts[int(i)] for i in exposure])
            meta["exposure_items"] = exposure
            prompt = render(template, first=first_text, second=second_text, exposure_block=block)
        elif cond.template_name == TemplateName.GROUP_B_STATED:
            if cond.probe_cluster is None or cond.p_c is None:
                raise ValueError("Stated-prevalence requires probe_cluster and p_c.")
            ex1, ex2 = self._exemplars(cond.probe_cluster)
            meta["probe_cluster"] = int(cond.probe_cluster)
            meta["p_c"] = float(cond.p_c)
            prompt = render(
                template,
                first=first_text,
                second=second_text,
                p_c=round(cond.p_c * 100),
                cluster_exemplar_1=ex1,
                cluster_exemplar_2=ex2,
            )
        elif cond.template_name == TemplateName.PLACEBO:
            if cond.p_placebo is None:
                raise ValueError("Placebo requires p_placebo.")
            prompt = render(template, first=first_text, second=second_text, p=round(cond.p_placebo * 100))
        else:  # intrinsic (GROUP_A / NO_CONTEXT)
            prompt = render(template, first=first_text, second=second_text)

        return prompt, meta

    async def _cached_call(self, prompt: str, meta: dict, repeat_index: int) -> str:
        key = cache_key(self.judge.model_id, prompt, repeat_index)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.n_cache_hits += 1
                return hit
        # Bounded exponential backoff with jitter lives here -- not in adapters --
        # so every backend gets identical transient-error treatment.
        attempt = 0
        while True:
            try:
                async with self._sem:
                    raw = await self.judge.choose(prompt, meta=meta)
                break
            except Exception:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = self.retry_base_delay * (2 ** (attempt - 1)) * (1.0 + random.random())
                await asyncio.sleep(delay)
        self.n_calls += 1
        if self.cache is not None:
            self.cache.put(
                key, raw, model_id=self.judge.model_id, repeat_index=repeat_index, prompt=prompt
            )
        return raw

    async def preference(
        self, item_a: int, item_b: int, cond: Condition, pair_idx: int
    ) -> tuple[dict, list[dict]]:
        """Return (pair_summary, per_repeat_records). p_prefer_a = P(prefer item_a)."""
        # Deterministic per-(pair, repeat) exposure sampling for recoverability.
        base = np.random.default_rng([self._seed, pair_idx, int(item_a), int(item_b)])
        votes: list[bool] = []
        repeat_records: list[dict] = []
        for rep in range(self.n_repeats):
            order_ab = rep % 2 == 0
            first_id, second_id = (item_a, item_b) if order_ab else (item_b, item_a)
            rep_rng = np.random.default_rng(base.integers(0, 2**63 - 1))
            prompt, meta = self._build_prompt_and_meta(cond, first_id, second_id, rep_rng)
            raw = await self._cached_call(prompt, meta, rep)
            choice = parse_ab(raw)
            picked_a: bool | None
            position_a: bool | None
            if choice is None:
                picked_a = None
                position_a = None
            else:
                picked_a = (choice == "A") == order_ab
                position_a = choice == "A"
                votes.append(picked_a)
            repeat_records.append(
                {
                    "pair_idx": pair_idx,
                    "item_a": item_a,
                    "item_b": item_b,
                    "repeat": rep,
                    "order_ab": order_ab,
                    "choice": choice,
                    "picked_a": picked_a,
                    "position_a": position_a,
                }
            )

        n_votes = len(votes)
        n_parse_fail = self.n_repeats - n_votes
        p_prefer_a = float(np.mean(votes)) if n_votes else float("nan")
        summary = {
            "pair_idx": pair_idx,
            "item_a": item_a,
            "item_b": item_b,
            "p_prefer_a": p_prefer_a,
            "n_votes": n_votes,
            "n_parse_fail": n_parse_fail,
        }
        return summary, repeat_records

    async def elicit_pairs(
        self, pairs: np.ndarray, cond: Condition
    ) -> ElicitationResult:
        """Elicit a batch of pairs under one condition. ``pairs`` is (n_pairs, 2)."""
        pairs = np.asarray(pairs, dtype=np.int64)
        tasks = [
            self.preference(int(a), int(b), cond, idx)
            for idx, (a, b) in enumerate(pairs)
        ]
        results = await asyncio.gather(*tasks)
        summaries = [r[0] for r in results]
        repeats = [rec for r in results for rec in r[1]]
        return ElicitationResult(
            pairs=pd.DataFrame(summaries),
            repeats=pd.DataFrame(repeats),
        )


def sample_pairs(
    n_items: int,
    n_pairs: int,
    rng: np.random.Generator,
    *,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Sample distinct unordered item pairs without replacement.

    If ``weight`` is given, one endpoint is drawn proportional to it (used for the
    Phase 2 'concentrated' half); otherwise pairs are uniform over the 4,950
    possible. Returns an (n_pairs, 2) int array.
    """
    max_pairs = n_items * (n_items - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    if weight is not None:
        weight = np.asarray(weight, dtype=np.float64)
        weight = weight / weight.sum()
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    # Attempt cap: under a concentrated ``weight`` (e.g. a near-deterministic
    # policy) the reachable pair set can be much smaller than n_pairs; return
    # what was found rather than spinning forever.
    max_attempts = 200 * max(n_pairs, 1)
    attempts = 0
    while len(out) < n_pairs and attempts < max_attempts:
        attempts += 1
        if weight is not None:
            a = int(rng.choice(n_items, p=weight))
            b = int(rng.integers(n_items))
        else:
            a = int(rng.integers(n_items))
            b = int(rng.integers(n_items))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return np.array(out, dtype=np.int64)


def sample_pairs_mixed(
    n_items: int,
    n_pairs: int,
    rng: np.random.Generator,
    weight: np.ndarray,
) -> np.ndarray:
    """Half concentrated (*both* endpoints ~ ``weight``), half uniform; deduplicated.

    This is the design's Phase 2/3 evaluation sampling: half the pairs probe the
    high-mass region of the current policy (both items policy-sampled, so they
    land inside the local set), half keep global coverage. Under a concentrated
    policy the reachable distinct concentrated pairs can run out; the uniform
    fill then tops the batch up to ``n_pairs``.
    """
    weight = np.asarray(weight, dtype=np.float64)
    weight = weight / weight.sum()
    n_half = n_pairs // 2
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    max_attempts = 200 * max(n_pairs, 1)
    attempts = 0
    while len(out) < n_half and attempts < max_attempts:
        attempts += 1
        a = int(rng.choice(n_items, p=weight))
        b = int(rng.choice(n_items, p=weight))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    attempts = 0
    while len(out) < n_pairs and attempts < max_attempts:
        attempts += 1
        a = int(rng.integers(n_items))
        b = int(rng.integers(n_items))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return np.array(out, dtype=np.int64)


async def elicit_v_hat(
    judge: Judge,
    menu: Menu,
    *,
    n_pairs: int = 800,
    seed: int = 7,
    cache: ResponseCache | None = None,
    n_repeats: int = 5,
    max_concurrency: int = 16,
) -> np.ndarray:
    """No-context BT fit v_hat (spec template 2.6): the measured intrinsic utility.

    This vector is the elicited stand-in for the toy model's shared v(y); it
    defines D_ref = softmax(v_hat) for Phases 2-3 and doubles as the Phase 0
    cluster-quality-balance input.
    """
    elicitor = PairwiseElicitor(
        judge,
        menu,
        n_repeats=n_repeats,
        max_concurrency=max_concurrency,
        cache=cache,
        seed=seed,
    )
    pairs = sample_pairs(menu.n_items, n_pairs, np.random.default_rng(seed))
    res = await elicitor.elicit_pairs(pairs, Condition(TemplateName.NO_CONTEXT))
    model = fit_bradley_terry_weighted(
        res.pairs["item_a"].to_numpy(),
        res.pairs["item_b"].to_numpy(),
        res.pairs["p_prefer_a"].to_numpy(),
        menu.n_items,
        weight=res.pairs["n_votes"].to_numpy(),
    )
    return model.score_all()
