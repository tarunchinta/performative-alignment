"""Pluggable judge interface and backends for Experiment 3.

The contract is minimal and stateless per call (spec section 1):

    class Judge(Protocol):
        async def choose(self, prompt: str, *, meta: dict | None = None) -> str: ...

``choose`` returns the raw completion text; the harness (PairwiseElicitor) owns
repeats, order-swap, parsing, caching, and rate limiting so backends cannot
drift apart. The optional ``meta`` payload is ignored by real API adapters and
used only by ``SyntheticJudge`` to compute a structural answer (so the whole
Phase 0->1 pipeline can be verified for free before spending an API dollar).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import expit

from src.exp3.templates import TemplateName

# Templates whose utility is intrinsic (deployment-independent): the synthetic
# judge ignores any exposure/prevalence channel for these, which is exactly what
# makes Group A the lambda=0 control and the contamination/placebo checks pass.
_INTRINSIC_FORMATS = {
    TemplateName.GROUP_A.value,
    TemplateName.NO_CONTEXT.value,
    TemplateName.GROUP_A_CONTAMINATION.value,
    TemplateName.PLACEBO.value,
}


@runtime_checkable
class Judge(Protocol):
    model_id: str

    async def choose(self, prompt: str, *, meta: dict | None = None) -> str:
        """Return the raw completion text for a single pairwise prompt."""
        ...


class SyntheticJudge:
    """Wraps the Experiment 1/2 U_B as an oracle pairwise judge.

    Utility in BT/logit space is ``u(y) = beta * (v[y] - lam * D_perceived(c_y))``.
    For contextual formats, ``D_perceived`` is the empirical cluster frequency of
    the sampled 20-item exposure block (so the harness must recover lambda from a
    noisy channel); for stated-prevalence it is the stated cluster mass. Intrinsic
    formats drop the prevalence term entirely.

    ``choose`` returns "A" if it prefers the item shown first, else "B"; the
    elicitor handles the order-swap bookkeeping.
    """

    def __init__(
        self,
        v: np.ndarray,
        cluster_of: np.ndarray,
        *,
        lam: float,
        beta: float = 3.0,
        model_id: str = "synthetic",
        seed: int = 0,
    ) -> None:
        self.v = np.asarray(v, dtype=np.float64)
        self.cluster_of = np.asarray(cluster_of, dtype=np.int64)
        self.n_clusters = int(self.cluster_of.max()) + 1
        self.lam = float(lam)
        self.beta = float(beta)
        self.model_id = model_id
        self._rng = np.random.default_rng(seed)

    @property
    def effective_lambda(self) -> float:
        """Prevalence-sensitivity in the BT/logit space the psychometrics fit in."""
        return self.beta * self.lam

    def _perceived_prevalence(self, meta: dict) -> np.ndarray | None:
        fmt = meta.get("format")
        if fmt in _INTRINSIC_FORMATS:
            return None
        if fmt == TemplateName.GROUP_B_CONTEXTUAL.value:
            exposure = np.asarray(meta["exposure_items"], dtype=np.int64)
            clusters = self.cluster_of[exposure]
            counts = np.bincount(clusters, minlength=self.n_clusters)
            return counts / counts.sum()
        if fmt == TemplateName.GROUP_B_STATED.value:
            probe = int(meta["probe_cluster"])
            p_c = float(meta["p_c"])
            prevalence = np.full(self.n_clusters, (1.0 - p_c) / (self.n_clusters - 1))
            prevalence[probe] = p_c
            return prevalence
        raise ValueError(f"SyntheticJudge received unknown format: {fmt!r}")

    def _utility(self, meta: dict) -> np.ndarray:
        prevalence = self._perceived_prevalence(meta)
        if prevalence is None:
            return self.beta * self.v
        penalty = self.lam * prevalence[self.cluster_of]
        return self.beta * (self.v - penalty)

    async def choose(self, prompt: str, *, meta: dict | None = None) -> str:
        del prompt  # synthetic answers structurally, not from rendered text
        if meta is None:
            raise ValueError("SyntheticJudge requires structured meta.")
        u = self._utility(meta)
        first = int(meta["first"])
        second = int(meta["second"])
        p_first = float(expit(u[first] - u[second]))
        return "A" if self._rng.random() < p_first else "B"


class AnthropicJudge:
    """Anthropic Messages API adapter (lazy import; requires ``anthropic``)."""

    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        *,
        temperature: float = 0.7,
        max_tokens: int = 4,
        api_key: str | None = None,
    ) -> None:
        from anthropic import AsyncAnthropic  # noqa: PLC0415

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def choose(self, prompt: str, *, meta: dict | None = None) -> str:
        del meta
        resp = await self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class AzureOpenAIJudge:
    """Azure AI Foundry chat-completions adapter (requires ``openai``).

    Uses the Foundry OpenAI-compatible **v1** surface: base_url is derived from
    ``AZURE_OPENAI_ENDPOINT`` (any route suffix like ``/responses`` is stripped
    down to ``.../openai/v1``) and the key from ``AZURE_OPENAI_API_KEY``. One
    endpoint serves every deployed family (GPT, Claude, Llama, ...) with the
    deployment name passed as ``model_id``.
    """

    _REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4", "codex")

    def __init__(
        self,
        model_id: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4,
        client: object | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        reasoning_model: bool | None = None,
    ) -> None:
        if client is None:
            import os  # noqa: PLC0415

            from openai import AsyncOpenAI  # noqa: PLC0415

            endpoint = endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]
            api_key = api_key or os.environ["AZURE_OPENAI_API_KEY"]
            if "/openai" in endpoint:
                base_url = endpoint.split("/openai")[0] + "/openai/v1"
            else:
                base_url = endpoint.rstrip("/") + "/openai/v1"
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        if reasoning_model is None:
            reasoning_model = model_id.lower().startswith(self._REASONING_PREFIXES)
        # Reasoning-family models (gpt-5.x, o-series) reject ``max_tokens`` and any
        # non-default temperature; they take ``max_completion_tokens`` and sample at
        # temperature 1 (a recorded deviation from the spec's 0.7 -- vote-fraction
        # repeats still provide the preference probability).
        self.reasoning_model = reasoning_model

    async def choose(self, prompt: str, *, meta: dict | None = None) -> str:
        del meta
        if self.reasoning_model:
            resp = await self._client.chat.completions.create(
                model=self.model_id,
                max_completion_tokens=max(self.max_tokens, 16),
                reasoning_effort="minimal",
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            resp = await self._client.chat.completions.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        return resp.choices[0].message.content or ""


class LocalJudge:
    """Open-weights adapter via an OpenAI-compatible endpoint (e.g. vLLM)."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.7,
        max_tokens: int = 4,
        api_key: str = "EMPTY",
    ) -> None:
        from openai import AsyncOpenAI  # noqa: PLC0415

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def choose(self, prompt: str, *, meta: dict | None = None) -> str:
        del meta
        resp = await self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
