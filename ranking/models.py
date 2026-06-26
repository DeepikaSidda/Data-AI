"""Core domain data models (frozen dataclasses).

Defines the immutable value types used across the ranking pipeline. Field names
mirror the real ``candidate_schema.json`` so a parsed JSON object maps cleanly
onto :class:`CandidateRecord` via :meth:`CandidateRecord.from_json`.

The ``from_json`` constructor is tolerant of missing *optional* fields
(``education``, ``certifications``, ``languages`` may be empty/absent and a few
behavioral signals default sensibly) and treats the ``-1`` sentinels for
``github_activity_score`` and ``offer_acceptance_rate`` as "unknown", storing
them as ``None``.

Requirements: 3.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Candidate sub-structures (mirror candidate_schema.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CareerEntry:
    """One role in a candidate's ``career_history``."""

    company: str
    title: str
    start_date: str            # ISO date string
    end_date: Optional[str]    # None if the role is current
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "CareerEntry":
        return cls(
            company=str(obj.get("company", "")),
            title=str(obj.get("title", "")),
            start_date=str(obj.get("start_date", "")),
            end_date=obj.get("end_date"),
            duration_months=int(obj.get("duration_months", 0) or 0),
            is_current=bool(obj.get("is_current", False)),
            industry=str(obj.get("industry", "")),
            company_size=str(obj.get("company_size", "")),
            description=str(obj.get("description", "")),
        )


@dataclass(frozen=True)
class EducationEntry:
    """One entry in a candidate's ``education`` list."""

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"      # tier_1..tier_4 | unknown

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "EducationEntry":
        return cls(
            institution=str(obj.get("institution", "")),
            degree=str(obj.get("degree", "")),
            field_of_study=str(obj.get("field_of_study", "")),
            start_year=int(obj.get("start_year", 0) or 0),
            end_year=int(obj.get("end_year", 0) or 0),
            grade=obj.get("grade"),
            tier=str(obj.get("tier", "unknown")),
        )


@dataclass(frozen=True)
class Skill:
    """One entry in a candidate's ``skills`` list."""

    name: str
    proficiency: str           # beginner|intermediate|advanced|expert
    endorsements: int
    duration_months: int = 0

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Skill":
        return cls(
            name=str(obj.get("name", "")),
            proficiency=str(obj.get("proficiency", "")),
            endorsements=int(obj.get("endorsements", 0) or 0),
            duration_months=int(obj.get("duration_months", 0) or 0),
        )


@dataclass(frozen=True)
class Profile:
    """The ``profile`` object on a candidate record."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Profile":
        return cls(
            anonymized_name=str(obj.get("anonymized_name", "")),
            headline=str(obj.get("headline", "")),
            summary=str(obj.get("summary", "")),
            location=str(obj.get("location", "")),
            country=str(obj.get("country", "")),
            years_of_experience=float(obj.get("years_of_experience", 0.0) or 0.0),
            current_title=str(obj.get("current_title", "")),
            current_company=str(obj.get("current_company", "")),
            current_company_size=str(obj.get("current_company_size", "")),
            current_industry=str(obj.get("current_industry", "")),
        )


@dataclass(frozen=True)
class RedrobSignals:
    """The 23 behavioral / engagement fields in ``redrob_signals``.

    ``github_activity_score`` and ``offer_acceptance_rate`` use a ``-1`` sentinel
    in the source schema to mean "no data". :meth:`from_json` converts that
    sentinel to ``None`` so downstream code can treat the value as unknown
    rather than as a real (and misleadingly low) score.
    """

    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float                  # 0..1
    avg_response_time_hours: float
    connection_count: int
    endorsements_received: int
    notice_period_days: int                         # 0..180
    preferred_work_mode: str
    willing_to_relocate: bool
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
    # None when the source value was the -1 "unknown" sentinel.
    github_activity_score: Optional[float] = None   # else 0..100
    offer_acceptance_rate: Optional[float] = None   # else 0..1
    skill_assessment_scores: dict[str, float] = field(default_factory=dict)
    expected_salary_range_inr_lpa: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _unknown_if_sentinel(value: Any) -> Optional[float]:
        """Map a ``-1`` sentinel (or missing value) to ``None``; else float."""
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        return None if num == -1 else num

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "RedrobSignals":
        return cls(
            profile_completeness_score=float(
                obj.get("profile_completeness_score", 0.0) or 0.0
            ),
            signup_date=str(obj.get("signup_date", "")),
            last_active_date=str(obj.get("last_active_date", "")),
            open_to_work_flag=bool(obj.get("open_to_work_flag", False)),
            profile_views_received_30d=int(
                obj.get("profile_views_received_30d", 0) or 0
            ),
            applications_submitted_30d=int(
                obj.get("applications_submitted_30d", 0) or 0
            ),
            recruiter_response_rate=float(
                obj.get("recruiter_response_rate", 0.0) or 0.0
            ),
            avg_response_time_hours=float(
                obj.get("avg_response_time_hours", 0.0) or 0.0
            ),
            connection_count=int(obj.get("connection_count", 0) or 0),
            endorsements_received=int(obj.get("endorsements_received", 0) or 0),
            notice_period_days=int(obj.get("notice_period_days", 0) or 0),
            preferred_work_mode=str(obj.get("preferred_work_mode", "")),
            willing_to_relocate=bool(obj.get("willing_to_relocate", False)),
            search_appearance_30d=int(obj.get("search_appearance_30d", 0) or 0),
            saved_by_recruiters_30d=int(obj.get("saved_by_recruiters_30d", 0) or 0),
            interview_completion_rate=float(
                obj.get("interview_completion_rate", 0.0) or 0.0
            ),
            verified_email=bool(obj.get("verified_email", False)),
            verified_phone=bool(obj.get("verified_phone", False)),
            linkedin_connected=bool(obj.get("linkedin_connected", False)),
            github_activity_score=cls._unknown_if_sentinel(
                obj.get("github_activity_score")
            ),
            offer_acceptance_rate=cls._unknown_if_sentinel(
                obj.get("offer_acceptance_rate")
            ),
            skill_assessment_scores=dict(obj.get("skill_assessment_scores") or {}),
            expected_salary_range_inr_lpa=dict(
                obj.get("expected_salary_range_inr_lpa") or {}
            ),
        )


@dataclass(frozen=True)
class CandidateRecord:
    """A single candidate profile conforming to the Redrob schema."""

    candidate_id: str                                # ^CAND_[0-9]{7}$
    profile: Profile
    career_history: list[CareerEntry]
    education: list[EducationEntry]
    skills: list[Skill]
    redrob_signals: RedrobSignals
    certifications: list[dict] = field(default_factory=list)
    languages: list[dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "CandidateRecord":
        """Map a parsed JSON object onto a :class:`CandidateRecord`.

        Tolerates missing optional fields: ``education``, ``certifications`` and
        ``languages`` may be absent or empty. The ``-1`` sentinels for
        ``github_activity_score`` and ``offer_acceptance_rate`` are mapped to
        ``None`` ("unknown") by :meth:`RedrobSignals.from_json`.
        """
        return cls(
            candidate_id=str(obj.get("candidate_id", "")),
            profile=Profile.from_json(obj.get("profile") or {}),
            career_history=[
                CareerEntry.from_json(e) for e in (obj.get("career_history") or [])
            ],
            education=[
                EducationEntry.from_json(e) for e in (obj.get("education") or [])
            ],
            skills=[Skill.from_json(s) for s in (obj.get("skills") or [])],
            redrob_signals=RedrobSignals.from_json(obj.get("redrob_signals") or {}),
            certifications=list(obj.get("certifications") or []),
            languages=list(obj.get("languages") or []),
        )


# ---------------------------------------------------------------------------
# Scoring / ranking value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionScores:
    """Normalized [0,1] per-dimension scores feeding the hybrid scorer."""

    semantic: float            # cosine-derived
    skills_title: float        # title-aware + trust multiplier
    experience: float          # 5-9yr band, soft edges
    trajectory: float          # product-vs-services, anti-hopping
    education: float           # light, tier-based


@dataclass(frozen=True)
class HoneypotResult:
    """Outcome of the consistency checks for a candidate."""

    is_honeypot: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate with all intermediate and final scores attached."""

    candidate_id: str
    record: CandidateRecord
    dims: DimensionScores
    fit_score: float           # [0,1]
    behavioral_modifier: float  # bounded
    final_score: float         # [0,1]
    honeypot: HoneypotResult


@dataclass(frozen=True)
class RankedCandidate:
    """One row of the final shortlist / submission CSV."""

    rank: int                  # 1..100 unique
    candidate_id: str
    score: float               # final_score, [0,1]
    reasoning: str             # 1-2 sentences


# ---------------------------------------------------------------------------
# Ingestion / IO value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkipWarning:
    """A skipped input line with its 1-based line number and reason."""

    line_number: int
    reason: str


@dataclass(frozen=True)
class LoadResult:
    """The outcome of loading a candidate dataset."""

    records: list[CandidateRecord]
    valid_count: int
    skipped: list[SkipWarning] = field(default_factory=list)
    total_lines: int = 0


@dataclass(frozen=True)
class ArtifactPaths:
    """Local filesystem paths to the precomputed embedding artifacts."""

    embeddings: str
    id_order: str
    job_embedding: str
