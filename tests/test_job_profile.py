"""Unit tests for the offline-encoded Job_Profile (task 5.2).

Asserts the committed ``job_profile.yaml`` at the repo root loads into a
``JobProfile`` whose positive/negative signals and location/notice preferences
match the Senior AI Engineer role.

Requirements: 2.2, 2.3, 2.4.
"""

from __future__ import annotations

import os

import pytest

from ranking.job_profile import (
    JobProfile,
    LocationPref,
    NegativeSignals,
    NoticePref,
    PositiveSignals,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROFILE_PATH = os.path.join(_REPO_ROOT, "job_profile.yaml")


@pytest.fixture
def profile() -> JobProfile:
    """Load the committed repo job_profile.yaml."""
    return JobProfile.load(_PROFILE_PATH)


def _joined(*tuples: tuple[str, ...]) -> str:
    """Lowercase concatenation of several string tuples for substring checks."""
    return " | ".join(term.lower() for group in tuples for term in group)


def test_load_returns_job_profile(profile: JobProfile) -> None:
    assert isinstance(profile, JobProfile)
    assert isinstance(profile.positive_signals, PositiveSignals)
    assert isinstance(profile.negative_signals, NegativeSignals)
    assert isinstance(profile.location_pref, LocationPref)
    assert isinstance(profile.notice_pref, NoticePref)
    assert isinstance(profile.profile_text, str)
    assert profile.profile_text.strip(), "profile_text must be a non-empty paragraph"


def test_positive_signals_present(profile: JobProfile) -> None:
    pos = profile.positive_signals
    blob = _joined(
        pos.phrases,
        pos.skill_terms,
        pos.title_terms,
        pos.eval_metrics,
        pos.product_companies,
        pos.nice_to_have,
    )

    # Production retrieval / ranking / embeddings (Requirement 2.2).
    assert "retrieval" in blob
    assert "ranking" in blob
    assert "embeddings" in blob
    assert "production" in blob

    # Strong Python (Requirement 2.2).
    assert "python" in blob

    # Evaluation frameworks: NDCG / MRR / MAP (Requirement 2.2).
    metrics = {m.lower() for m in pos.eval_metrics}
    assert {"ndcg", "mrr", "map"}.issubset(metrics)


def test_negative_signals_present(profile: JobProfile) -> None:
    neg = profile.negative_signals
    blob = _joined(
        neg.keyword_stuffer_titles,
        neg.consulting_firms,
        neg.flags,
        neg.off_domain_terms,
    )

    # Keyword-stuffer titles unrelated to engineering (Requirement 2.3).
    stuffer = {t.lower() for t in neg.keyword_stuffer_titles}
    assert any("marketing" in t or "hr" in t or "sales" in t for t in stuffer)

    # Consulting / services firms (Requirement 2.3).
    firms = {f.lower() for f in neg.consulting_firms}
    for firm in ("tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"):
        assert firm in firms, f"expected consulting firm {firm!r} in negative signals"

    # Pure research without production (Requirement 2.3).
    assert "research" in blob

    # CV / speech / robotics-only without NLP/IR (Requirement 2.3).
    assert "computer vision" in blob or "vision" in blob
    assert "speech" in blob
    assert "robotics" in blob

    # Title-chasing job-hopping (Requirement 2.3).
    assert "title chasing" in blob or "job hopping" in blob or "1.5" in blob


def test_location_preference_present(profile: JobProfile) -> None:
    loc = profile.location_pref
    cities = {c.lower() for c in loc.preferred_cities}
    assert "noida" in cities
    assert "pune" in cities
    assert loc.relocation_ok is True


def test_notice_preference_present(profile: JobProfile) -> None:
    notice = profile.notice_pref
    # Under 30 days preferred (Requirement 2.4).
    assert notice.preferred_max_days == 30
