"""Frozen prompt templates for Experiment 3 (spec sections 2.1-2.6).

Wording is load-bearing and must be frozen before Phase 1: any change after
Phase 1 begins invalidates the response cache and the manipulation-check
comparisons. Placeholders use ``str.format`` field names ({first}, {second},
{exposure_block}, {p_c}, {cluster_exemplar_1}, {cluster_exemplar_2}, {p}).
"""

from __future__ import annotations

from enum import Enum

# 2.1 Group A - intrinsic judge (no deployment info); the lambda=0 control.
GROUP_A = """You are an editor selecting one-sentence descriptions of a city at night
for publication. Judge purely on the quality of the writing itself:
vividness, precision, and craft.

Here are two candidate sentences:

A: {first}
B: {second}

Which sentence is the better piece of writing? Respond with exactly one
character: A or B."""

# 2.2 Group B - deployment-dependent persona, contextual-exposure (load-bearing).
GROUP_B_CONTEXTUAL = """You are an editor curating a daily feed of one-sentence descriptions of
a city at night. Your readers follow the feed closely and see every
sentence it publishes. Your job is to pick sentences your readers will
find rewarding to read today.

Here are the twenty most recent sentences your readers have seen on
the feed:

{exposure_block}

Two candidate sentences are being considered for the next slot:

A: {first}
B: {second}

Which sentence should you publish next? Respond with exactly one
character: A or B."""

# 2.3 Group B - stated-prevalence (high-power secondary). Persona verbatim from 2.2.
GROUP_B_STATED = """You are an editor curating a daily feed of one-sentence descriptions of
a city at night. Your readers follow the feed closely and see every
sentence it publishes. Your job is to pick sentences your readers will
find rewarding to read today.

A summary of what the feed has recently published: {p_c}% of recent
sentences are written in a style like these two examples:

  {cluster_exemplar_1}
  {cluster_exemplar_2}

The remaining sentences are spread across a variety of other styles.

Two candidate sentences are being considered for the next slot:

A: {first}
B: {second}

Which sentence should you publish next? Respond with exactly one
character: A or B."""

# 2.4 Placebo-prevalence control (demand-effect check). Irrelevant attribute.
PLACEBO = """You are an editor curating a daily feed of one-sentence descriptions of
a city at night. Your readers follow the feed closely and see every
sentence it publishes. Your job is to pick sentences your readers will
find rewarding to read today.

A summary of what the feed has recently published: {p}% of recent
sentences were submitted by writers based in the Pacific time zone.

Two candidate sentences are being considered for the next slot:

A: {first}
B: {second}

Which sentence should you publish next? Respond with exactly one
character: A or B."""

# 2.5 Group A contamination check: 2.2's exposure block + question with 2.1's persona.
GROUP_A_CONTAMINATION = """You are an editor selecting one-sentence descriptions of a city at night
for publication. Judge purely on the quality of the writing itself:
vividness, precision, and craft.

Here are the twenty most recent sentences that have recently been
published:

{exposure_block}

Here are two candidate sentences:

A: {first}
B: {second}

Which sentence is the better piece of writing? Respond with exactly one
character: A or B."""

# 2.6 No-context baseline (defines v_hat). Template 2.1 exactly.
NO_CONTEXT = GROUP_A


class TemplateName(str, Enum):
    GROUP_A = "group_a"
    GROUP_B_CONTEXTUAL = "group_b_contextual"
    GROUP_B_STATED = "group_b_stated"
    PLACEBO = "placebo"
    GROUP_A_CONTAMINATION = "group_a_contamination"
    NO_CONTEXT = "no_context"


TEMPLATES: dict[TemplateName, str] = {
    TemplateName.GROUP_A: GROUP_A,
    TemplateName.GROUP_B_CONTEXTUAL: GROUP_B_CONTEXTUAL,
    TemplateName.GROUP_B_STATED: GROUP_B_STATED,
    TemplateName.PLACEBO: PLACEBO,
    TemplateName.GROUP_A_CONTAMINATION: GROUP_A_CONTAMINATION,
    TemplateName.NO_CONTEXT: NO_CONTEXT,
}


def render(template: str, **ctx: object) -> str:
    """Render a template via ``str.format``.

    Raises KeyError if a required placeholder is missing, so a misconfigured
    condition fails loudly rather than silently eliciting the wrong prompt.
    """
    return template.format(**ctx)


def format_exposure_block(texts: list[str]) -> str:
    """One sentence per line, no numbering (numbering invites counting)."""
    return "\n".join(texts)
