"""Tests for :class:`ranking.honeypot.HoneypotDetector`.

Covers each of the five consistency checks in isolation (so a failure points at
exactly one rule), confirms a consistent profile is NOT flagged, exercises
graceful handling of malformed/empty dates, and runs the detector across the
bundled ``sample_candidates.json`` records without crashing.

Validates: Requirements 6.1, 6.2, 6.3, 6.6.
"""

from __future__ import annotations

import datetime
import glob
import json
import os

import pytest

from ranking.honeypot import HoneypotDetector
from ranking.models import CandidateRecord, CareerEntry, Profile, RedrobSignals, Skill


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_THIS_YEAR = datetime.date.today().year


def _profile(yoe: float) -> Profile:
    return Profile(
        anonymized_name="Test Person",
        headline="Engineer",
        summary="",
        location="Pune",
        country="India",
        years_of_experience=yoe,
        current_title="Engineer",
        current_company="Acme",
        current_company_size="201-500",
        current_industry="Software",
    )


def _career(
    company: str,
    start_date: str,
    end_date: str | None,
    duration_months: int,
    is_current: bool = False,
) -> CareerEntry:
    return CareerEntry(
        company=company,
        title="Engineer",
        start_date=start_date,
        end_date=end_date,
        duration_months=duration_months,
        is_current=is_current,
        industry="Software",
        company_size="201-500",
        description="",
    )


def _skill(name: str, proficiency: str, duration_months: int, endorsements: int = 5) -> Skill:
    return Skill(
        name=name,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=duration_months,
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


def _record(
    yoe: float,
    career: list[CareerEntry],
    skills: list[Skill],
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id="CAND_0000001",
        profile=_profile(yoe),
        career_history=career,
        education=[],
        skills=skills,
        redrob_signals=_signals(),
    )


# A consistent, non-honeypot baseline: ~6 years experience, two sequential
# roles whose durations sum to ~6 years, and modestly-tenured skills.
def _consistent_record() -> CandidateRecord:
    return _record(
        yoe=6.0,
        career=[
            _career("Old Co", "2018-01-01", "2021-01-01", 36),
            _career("Cur Co", "2021-01-01", None, 36, is_current=True),
        ],
        skills=[
            _skill("Python", "expert", 60),
            _skill("Retrieval", "advanced", 36),
            _skill("NDCG", "intermediate", 24),
        ],
    )


# ---------------------------------------------------------------------------
# Consistent record is NOT flagged
# ---------------------------------------------------------------------------


def test_consistent_record_is_not_flagged():
    result = HoneypotDetector().check(_consistent_record())
    assert result.is_honeypot is False
    assert result.reasons == []


# ---------------------------------------------------------------------------
# Check 1 (recalibrated): experience exceeding total career span is NORMAL
# (people omit old jobs) and must NOT be flagged.
# ---------------------------------------------------------------------------


def test_experience_exceeding_span_is_not_flagged():
    # 3-year listed career but 20 years experience: an incomplete profile, not
    # an impossible one. The high-precision detector must not flag it.
    rec = _record(
        yoe=20.0,
        career=[_career("Cur Co", "2021-01-01", "2024-01-01", 36)],
        skills=[_skill("Python", "expert", 36)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is False


# ---------------------------------------------------------------------------
# Check 2 (recalibrated): duration sum exceeding experience (overlapping roles)
# is NORMAL and must NOT be flagged.
# ---------------------------------------------------------------------------


def test_duration_sum_exceeding_experience_is_not_flagged():
    # Two overlapping roles whose months sum high vs a small yoe — common in
    # real data (concurrent roles, rounding). Not impossible -> not flagged.
    rec = _record(
        yoe=5.0,
        career=[
            _career("A", "2009-01-01", "2014-01-01", 60),
            _career("B", "2009-01-01", "2014-01-01", 60),
        ],
        skills=[_skill("Python", "expert", 36)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is False


# ---------------------------------------------------------------------------
# Check: expert/advanced-with-zero-duration cluster (the documented honeypot)
# ---------------------------------------------------------------------------


def test_expert_zero_duration_cluster_is_flagged():
    rec = _record(
        yoe=6.0,
        career=[_career("Cur Co", "2018-01-01", None, 72, is_current=True)],
        skills=[
            _skill("Retrieval", "expert", 0),
            _skill("Ranking", "advanced", 0),
            _skill("Python", "intermediate", 40),
        ],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is True
    assert any("0 months duration" in r for r in result.reasons)


def test_single_zero_duration_expert_not_flagged():
    # Only ONE expert/zero-duration skill -> below the >=2 cluster threshold.
    rec = _record(
        yoe=6.0,
        career=[_career("Cur Co", "2018-01-01", None, 72, is_current=True)],
        skills=[
            _skill("Retrieval", "expert", 0),
            _skill("Python", "expert", 60),
        ],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is False


# ---------------------------------------------------------------------------
# Check (recalibrated): a single skill duration exceeding experience is NOISE
# in this dataset and must NOT be flagged on its own.
# ---------------------------------------------------------------------------


def test_skill_duration_exceeding_experience_is_not_flagged():
    rec = _record(
        yoe=2.0,
        career=[_career("Cur Co", "2014-01-01", None, 24, is_current=True)],
        skills=[_skill("Python", "expert", 120)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is False


# ---------------------------------------------------------------------------
# Check: role duration vastly exceeds the span its own dates allow
# ("8 years at a company the dates say is 3 years").
# ---------------------------------------------------------------------------


def test_role_duration_exceeds_date_span_is_flagged():
    rec = _record(
        yoe=8.0,
        career=[
            # Dates span ~36 months but the role claims 96 months.
            _career("Impossible Co", "2021-01-01", "2024-01-01", 96),
        ],
        skills=[_skill("Python", "advanced", 36)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is True
    assert any("span only" in r for r in result.reasons)


def test_role_duration_matching_date_span_is_not_flagged():
    rec = _record(
        yoe=6.0,
        career=[_career("Real Co", "2018-01-01", "2021-01-01", 36)],
        skills=[_skill("Python", "advanced", 36)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is False


# ---------------------------------------------------------------------------
# Check: impossible date ordering
# ---------------------------------------------------------------------------


def test_check_end_before_start():
    rec = _record(
        yoe=6.0,
        career=[
            _career("Cur Co", "2018-01-01", None, 36, is_current=True),
            _career("Bad Co", "2020-06-01", "2019-06-01", 12),
        ],
        skills=[_skill("Python", "expert", 60)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is True
    assert any("before start_date" in r for r in result.reasons)


def test_check_out_of_range_year():
    rec = _record(
        yoe=6.0,
        career=[
            _career("Cur Co", "2018-01-01", None, 36, is_current=True),
            _career("Future Co", "2099-01-01", "2099-06-01", 5),
        ],
        skills=[_skill("Python", "expert", 60)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is True
    assert any("1900-2035" in r for r in result.reasons)


def test_near_future_synthetic_dates_not_flagged():
    # The dataset's own timeline runs into 2025-2026; such dates are valid and
    # must not be treated as "future" honeypots.
    rec = _record(
        yoe=6.0,
        career=[_career("Cur Co", "2023-01-01", "2026-01-01", 36)],
        skills=[_skill("Python", "advanced", 36)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is False


# ---------------------------------------------------------------------------
# Graceful handling of malformed / empty dates
# ---------------------------------------------------------------------------


def test_malformed_and_empty_dates_do_not_crash():
    rec = _record(
        yoe=6.0,
        career=[
            _career("A", "", None, 36, is_current=True),
            _career("B", "not-a-date", "also-bad", 36),
        ],
        skills=[_skill("Python", "expert", 60)],
    )
    # Should not raise; with no parseable dates, span-based checks are skipped.
    result = HoneypotDetector().check(rec)
    assert isinstance(result.is_honeypot, bool)


def test_tolerance_absorbs_small_rounding():
    # yoe slightly above span but within the 3-month tolerance -> not flagged.
    rec = _record(
        yoe=3.1,
        career=[_career("Cur Co", "2021-01-01", "2024-01-01", 36)],
        skills=[_skill("Python", "advanced", 36)],
    )
    result = HoneypotDetector().check(rec)
    assert result.is_honeypot is False


# ---------------------------------------------------------------------------
# Sample data smoke test
# ---------------------------------------------------------------------------


def _find_sample_candidates() -> str | None:
    repo_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    matches = glob.glob(
        os.path.join(repo_parent, "**", "sample_candidates.json"), recursive=True
    )
    return matches[0] if matches else None


def test_runs_against_sample_candidates_without_crashing():
    path = _find_sample_candidates()
    if path is None:
        pytest.skip("sample_candidates.json not found in workspace")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    detector = HoneypotDetector()
    flagged = 0
    for obj in data:
        rec = CandidateRecord.from_json(obj)
        result = detector.check(rec)
        assert isinstance(result.is_honeypot, bool)
        # Every flag must carry at least one reason; clean records carry none.
        assert (len(result.reasons) > 0) == result.is_honeypot
        flagged += int(result.is_honeypot)
    # Determinism: a second pass yields identical results.
    again = [detector.check(CandidateRecord.from_json(o)).is_honeypot for o in data]
    first = [detector.check(CandidateRecord.from_json(o)).is_honeypot for o in data]
    assert again == first
