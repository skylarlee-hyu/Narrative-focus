"""Minimal headline-based leader-reference detection.

Replace the demonstration dictionary with a validated, firm- and date-specific
set of executive names/titles. A missing mention is NOT proof that a headline
is firm-centered.
"""
from collections.abc import Iterable


def has_leader_reference(headline: str, terms: Iterable[str]) -> bool:
    if not isinstance(headline, str):
        return False
    return any(
        term in headline
        for raw in terms
        if (term := str(raw).strip())
    )


def classify_reference(headline: str, terms: Iterable[str]) -> str:
    return "leader_mention" if has_leader_reference(headline, terms) else "no_leader_mention"
