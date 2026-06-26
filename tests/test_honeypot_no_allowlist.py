"""Tests asserting the ranking core uses NO candidate-id allowlist.

Both :class:`ranking.honeypot.HoneypotDetector` and
:class:`ranking.features.FeatureExtractor` must classify candidates using
profile / career evidence ONLY: there must be no hardcoded candidate ids and
no allowlist parameter. These tests verify that contractually by:

* inspecting the ``HoneypotDetector.__init__`` signature (only
  ``tolerance_months``; no id / allowlist parameter);
* confirming results are invariant to ``candidate_id`` for both the detector
  and the feature extractor (changing only the id changes nothing); and
* a light source-text check that neither module hardcodes a ``CAND_`` id or an
  allowlist constant for branching.

Validates: Requirements 6.6, 4.4.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import re

from ranking import features as features_module
from ranking import honeypot as honeypot_module
from ranking.config import ScoringConfig
from ranking.features import FeatureExtractor
from ranking.honeypot import HoneypotDetector
from ranking.job_profile import JobProfile
from ranking.models import (
    CandidateRecord,
    CareerEntry,
    EducationEntry,
    Profile,
    RedrobSignals,
    Skill,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Substrings that would indicate an id-based allowlist / denylist parameter.
_ALLOWLIST_HINTS = ("allow", "deny", "whitelist", "blacklist", "candidate_id", "ids")


# ---------------------------------------------------------------------------
# Builders (minimal, valid CandidateRecord)
# ---------------------------------------------------------------------------


def _profile(yoe: float = 6.0) -> Profile:
    return Profile(
        anonymized_name="Test Person",
        headline="Senior ML engineer building search and ranking systems",
        summary="Shipped recommendation and retrieval systems with NDCG eval.",
        location="Pune",
        country="India",
        years_of_experience=yoe,
        current_title="Senior Machine Learning Engineer",
        current_company="Acme",
        current_company_size="201-500",
        current_industry="Software",
    )


def _signals() -> RedrobSignals:
    return RedrobSignals(
        profile_completeness_score=0.8,
        signup_date="2020-01-01",
        last_active_date="2024-01-01",
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=2,
        recruiter_response_rate=0.5,
        avg_response_time_hours=12.0,
        connection_count=300,
        endorsements_received=40,
        notice_period_days=30,
        preferred_work_mode="hybrid",
        willing_to_relocate=True,
        search_appearance_30d=5,
        saved_by_recruiters_30d=1,
        interview_completion_rate=0.9,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )


def _record(candidate_id: str) -> CandidateRecord:
    """A consistent, non-trivial record parameterized only by candidate_id."""
    return CandidateRecord(
        candidate_id=candidate_id,
        profile=_profile(),
        career_history=[
            CareerEntry(
                company="Old Co",
                title="Machine Learning Engineer",
                start_date="2018-01-01",
                end_date="2021-01-01",
                duration_months=36,
                is_current=False,
                industry="Software",
                company_size="201-500",
                description="Built ranking and retrieval pipelines.",
            ),
            CareerEntry(
                company="Cur Co",
                title="Senior Machine Learning Engineer",
                start_date="2021-01-01",
                end_date=None,
                duration_months=36,
                is_current=True,
                industry="Software",
                company_size="201-500",
                description="Led recommendation and search ranking work.",
            ),
        ],
        education=[
            EducationEntry(
                institution="IIT",
                degree="B.Tech",
                field_of_study="CS",
                start_year=2012,
                end_year=2016,
                tier="tier_1",
            )
        ],
        skills=[
            Skill(name="Python", proficiency="expert", endorsements=20, duration_months=60),
            Skill(name="Retrieval", proficiency="advanced", endorsements=10, duration_months=36),
            Skill(name="Ranking", proficiency="intermediate", endorsements=8, duration_months=24),
        ],
        redrob_signals=_signals(),
    )


def _load_job() -> JobProfile:
    return JobProfile.load(os.path.join(_REPO_ROOT, "job_profile.yaml"))


# ---------------------------------------------------------------------------
# HoneypotDetector signature: no id / allowlist parameter
# ---------------------------------------------------------------------------


def test_honeypot_init_signature_has_no_allowlist_parameter():
    params = inspect.signature(HoneypotDetector.__init__).parameters
    names = [n for n in params if n != "self"]
    # Constructor knobs are numeric tolerances only; no id allowlist/denylist.
    assert names == ["tolerance_months", "duration_margin_months"], (
        f"HoneypotDetector.__init__ should accept only numeric tolerances, got {names}"
    )
    # And none of the parameters look like an id allowlist/denylist.
    for name in names:
        lowered = name.lower()
        assert not any(hint in lowered for hint in _ALLOWLIST_HINTS), (
            f"HoneypotDetector.__init__ parameter '{name}' looks like an id allowlist"
        )


def test_feature_extractor_init_signature_has_no_allowlist_parameter():
    params = inspect.signature(FeatureExtractor.__init__).parameters
    names = [n for n in params if n != "self"]
    # Only the job profile and scoring config; no id allowlist knob.
    assert names == ["job", "config"], (
        f"FeatureExtractor.__init__ should accept only 'job' and 'config', got {names}"
    )
    for name in names:
        lowered = name.lower()
        assert not any(hint in lowered for hint in _ALLOWLIST_HINTS), (
            f"FeatureExtractor.__init__ parameter '{name}' looks like an id allowlist"
        )


# ---------------------------------------------------------------------------
# Result invariance to candidate_id
# ---------------------------------------------------------------------------


def test_honeypot_result_is_independent_of_candidate_id():
    detector = HoneypotDetector()
    rec_a = _record("CAND_0000001")
    # Change ONLY the candidate_id (frozen dataclass -> use replace).
    rec_b = dataclasses.replace(rec_a, candidate_id="CAND_9999999")

    result_a = detector.check(rec_a)
    result_b = detector.check(rec_b)

    assert result_a.is_honeypot == result_b.is_honeypot
    assert result_a.reasons == result_b.reasons


def test_honeypot_result_is_independent_of_candidate_id_when_flagged():
    # An impossible profile (a role claiming far more months than its own dates
    # span) should be flagged the same way regardless of which id it carries.
    detector = HoneypotDetector()
    base = _record("CAND_0000002")
    impossible = dataclasses.replace(
        base,
        career_history=[
            CareerEntry(
                company="Impossible Co",
                title="Engineer",
                start_date="2021-01-01",
                end_date="2024-01-01",  # ~36 months of dates...
                duration_months=120,    # ...but claims 120 months.
                is_current=False,
                industry="Software",
                company_size="201-500",
                description="",
            )
        ],
    )

    flagged_a = detector.check(impossible)
    flagged_b = detector.check(
        dataclasses.replace(impossible, candidate_id="CAND_1234567")
    )

    assert flagged_a.is_honeypot is True
    assert flagged_a.is_honeypot == flagged_b.is_honeypot
    assert flagged_a.reasons == flagged_b.reasons


def test_feature_scores_are_independent_of_candidate_id():
    job = _load_job()
    config = ScoringConfig.load(None)
    extractor = FeatureExtractor(job, config)

    rec_a = _record("CAND_0000001")
    rec_b = dataclasses.replace(rec_a, candidate_id="CAND_7654321")

    semantic_similarity = 0.42  # fixed, so any difference would come from the id
    dims_a = extractor.extract(rec_a, semantic_similarity)
    dims_b = extractor.extract(rec_b, semantic_similarity)

    assert dims_a == dims_b


# ---------------------------------------------------------------------------
# Source-text check: no hardcoded CAND_ id / allowlist constant for branching
# ---------------------------------------------------------------------------


def test_source_modules_have_no_hardcoded_candidate_id():
    # A code-level allowlist would appear as an assignment, annotation, or
    # membership test (e.g. ``allowlist = {...}`` or ``id in whitelist``). We
    # match those constructs rather than prose, so docstrings that merely state
    # "no allowlist" do not trip the check.
    allowlist_code = re.compile(
        r"((allow|white|deny|black)list\s*[:=])"
        r"|(\bin\s+\w*(allow|white|deny|black)list\b)",
        re.IGNORECASE,
    )
    for module in (honeypot_module, features_module):
        source = inspect.getsource(module)
        # No literal candidate id (CAND_ followed by digits) anywhere.
        assert not re.search(r"CAND_\d", source), (
            f"{module.__name__} appears to hardcode a candidate id (CAND_<digits>)"
        )
        match = allowlist_code.search(source)
        assert match is None, (
            f"{module.__name__} appears to define/use an id allowlist: {match!r}"
        )
