"""Unit tests for the core data models (``ranking.models``).

Builds a ``CandidateRecord`` from a representative record mirrored inline from
``sample_candidates.json`` and asserts:
- nested sub-structures map onto the right dataclasses,
- optional fields default sensibly when absent (empty education /
  certifications / languages),
- the ``-1`` sentinels for ``github_activity_score`` and
  ``offer_acceptance_rate`` map to ``None`` ("unknown").

Requirements: 3.1.
"""

from __future__ import annotations

import copy

from ranking.models import (
    CandidateRecord,
    CareerEntry,
    EducationEntry,
    Profile,
    RedrobSignals,
    Skill,
)


# A representative record mirrored from sample_candidates.json (CAND_0000001),
# trimmed to a couple of career/skill entries for readability.
SAMPLE_RECORD = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Ira Vora",
        "headline": "Backend Engineer | SQL, Spark, Cloud",
        "summary": "Software / data professional with 6.9 years of experience.",
        "location": "Toronto",
        "country": "Canada",
        "years_of_experience": 6.9,
        "current_title": "Backend Engineer",
        "current_company": "Mindtree",
        "current_company_size": "10001+",
        "current_industry": "IT Services",
    },
    "career_history": [
        {
            "company": "Mindtree",
            "title": "Backend Engineer",
            "start_date": "2024-03-08",
            "end_date": None,
            "duration_months": 27,
            "is_current": True,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": "Streaming data pipelines on Kafka and Spark.",
        },
        {
            "company": "Dunder Mifflin",
            "title": "Analytics Engineer",
            "start_date": "2019-07-03",
            "end_date": "2024-01-08",
            "duration_months": 55,
            "is_current": False,
            "industry": "Paper Products",
            "company_size": "201-500",
            "description": "Built and maintained Airflow pipelines.",
        },
    ],
    "education": [
        {
            "institution": "Lovely Professional University",
            "degree": "B.E.",
            "field_of_study": "Computer Science",
            "start_year": 2017,
            "end_year": 2020,
            "grade": "8.24 CGPA",
            "tier": "tier_3",
        }
    ],
    "skills": [
        {
            "name": "NLP",
            "proficiency": "advanced",
            "endorsements": 37,
            "duration_months": 26,
        },
        {
            # No duration_months -> should default to 0.
            "name": "AWS",
            "proficiency": "beginner",
            "endorsements": 5,
        },
    ],
    "certifications": [],
    "languages": [
        {"language": "English", "proficiency": "professional"},
        {"language": "Hindi", "proficiency": "conversational"},
    ],
    "redrob_signals": {
        "profile_completeness_score": 86.9,
        "signup_date": "2025-10-16",
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "profile_views_received_30d": 23,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.34,
        "avg_response_time_hours": 177.8,
        "skill_assessment_scores": {"NLP": 38.8, "Speech Recognition": 53.7},
        "connection_count": 356,
        "endorsements_received": 35,
        "notice_period_days": 60,
        "expected_salary_range_inr_lpa": {"min": 18.7, "max": 36.1},
        "preferred_work_mode": "onsite",
        "willing_to_relocate": False,
        "github_activity_score": 9.2,
        "search_appearance_30d": 249,
        "saved_by_recruiters_30d": 4,
        "interview_completion_rate": 0.71,
        "offer_acceptance_rate": 0.58,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": False,
    },
}


def test_from_json_maps_top_level_structure():
    rec = CandidateRecord.from_json(SAMPLE_RECORD)

    assert isinstance(rec, CandidateRecord)
    assert rec.candidate_id == "CAND_0000001"
    assert isinstance(rec.profile, Profile)
    assert isinstance(rec.redrob_signals, RedrobSignals)
    assert all(isinstance(c, CareerEntry) for c in rec.career_history)
    assert all(isinstance(e, EducationEntry) for e in rec.education)
    assert all(isinstance(s, Skill) for s in rec.skills)


def test_from_json_maps_profile_fields():
    rec = CandidateRecord.from_json(SAMPLE_RECORD)
    assert rec.profile.anonymized_name == "Ira Vora"
    assert rec.profile.years_of_experience == 6.9
    assert rec.profile.current_company == "Mindtree"


def test_from_json_maps_career_history():
    rec = CandidateRecord.from_json(SAMPLE_RECORD)
    assert len(rec.career_history) == 2
    current = rec.career_history[0]
    assert current.is_current is True
    assert current.end_date is None
    assert current.duration_months == 27


def test_skill_duration_defaults_to_zero_when_absent():
    rec = CandidateRecord.from_json(SAMPLE_RECORD)
    aws = next(s for s in rec.skills if s.name == "AWS")
    assert aws.duration_months == 0
    assert aws.endorsements == 5


def test_records_are_frozen():
    import dataclasses

    import pytest

    rec = CandidateRecord.from_json(SAMPLE_RECORD)
    # Mutating a frozen dataclass raises FrozenInstanceError.
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.candidate_id = "CAND_9999999"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Optional-field defaults
# ---------------------------------------------------------------------------


def test_missing_optional_collections_default_empty():
    obj = copy.deepcopy(SAMPLE_RECORD)
    del obj["education"]
    del obj["certifications"]
    del obj["languages"]

    rec = CandidateRecord.from_json(obj)
    assert rec.education == []
    assert rec.certifications == []
    assert rec.languages == []


def test_empty_certifications_preserved():
    rec = CandidateRecord.from_json(SAMPLE_RECORD)
    assert rec.certifications == []
    assert len(rec.languages) == 2


# ---------------------------------------------------------------------------
# -1 sentinel handling -> "unknown" (None)
# ---------------------------------------------------------------------------


def test_present_github_and_offer_values_are_kept():
    rec = CandidateRecord.from_json(SAMPLE_RECORD)
    assert rec.redrob_signals.github_activity_score == 9.2
    assert rec.redrob_signals.offer_acceptance_rate == 0.58


def test_minus_one_sentinels_map_to_none():
    obj = copy.deepcopy(SAMPLE_RECORD)
    obj["redrob_signals"]["github_activity_score"] = -1
    obj["redrob_signals"]["offer_acceptance_rate"] = -1

    rec = CandidateRecord.from_json(obj)
    assert rec.redrob_signals.github_activity_score is None
    assert rec.redrob_signals.offer_acceptance_rate is None


def test_missing_sentinel_fields_map_to_none():
    obj = copy.deepcopy(SAMPLE_RECORD)
    del obj["redrob_signals"]["github_activity_score"]
    del obj["redrob_signals"]["offer_acceptance_rate"]

    rec = CandidateRecord.from_json(obj)
    assert rec.redrob_signals.github_activity_score is None
    assert rec.redrob_signals.offer_acceptance_rate is None


def test_zero_is_not_treated_as_unknown():
    # 0.0 is a legitimate score and must be preserved (only -1 is the sentinel).
    obj = copy.deepcopy(SAMPLE_RECORD)
    obj["redrob_signals"]["github_activity_score"] = 0
    obj["redrob_signals"]["offer_acceptance_rate"] = 0.0

    rec = CandidateRecord.from_json(obj)
    assert rec.redrob_signals.github_activity_score == 0.0
    assert rec.redrob_signals.offer_acceptance_rate == 0.0
