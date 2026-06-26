"""Property-based tests for :class:`ranking.features.FeatureExtractor`.

Covers design Correctness Properties 6, 7, 8 and 9 (tasks 7.2-7.5). Each test
isolates the structured dimensions by holding the semantic similarity fixed at a
neutral 0.5, loading the real committed :class:`JobProfile` and a default
:class:`ScoringConfig`. Where a property is naturally a *dimension* property it
is asserted on that dimension (e.g. ``skills_title``, ``trajectory``) and, when
safe, also on the blended ``Fit_Score`` via :class:`HybridScorer`.

Validates: Requirements 4.1, 4.2, 4.3, 3.3, 3.4.
"""

from __future__ import annotations

import os

from hypothesis import given, settings
from hypothesis import strategies as st

from ranking.config import ScoringConfig
from ranking.features import FeatureExtractor
from ranking.job_profile import JobProfile
from ranking.models import (
    CandidateRecord,
    CareerEntry,
    Profile,
    RedrobSignals,
    Skill,
)
from ranking.scorer import HybridScorer

# ---------------------------------------------------------------------------
# Shared, immutable test context (loaded once).
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JOB_PATH = os.path.join(_REPO_ROOT, "job_profile.yaml")

JOB = JobProfile.load(_JOB_PATH)
CONFIG = ScoringConfig.load(None)
EXTRACTOR = FeatureExtractor(JOB, CONFIG)
SCORER = HybridScorer(CONFIG)

# Neutral semantic similarity so the structured dimensions are isolated.
NEUTRAL_SEM = 0.5
TOL = 1e-9

# Job-relevant skill terms (every one is "relevant" per the extractor).
SKILL_TERMS = [t for t in JOB.positive_signals.skill_terms if t]

# Build-evidence fragments that would leak into trajectory via a title; we strip
# these out of the "genuine title" pool so Property 6 isolates skills_title.
_BUILD_FRAGMENTS = ("search", "ranking", "recommend", "retrieval", "embedding")
GENUINE_TITLES = [
    t
    for t in JOB.positive_signals.title_terms
    if t and not any(f in t.lower() for f in _BUILD_FRAGMENTS)
]
STUFFER_TITLES = [t for t in JOB.negative_signals.keyword_stuffer_titles if t]
CONSULTING_FIRMS = [t for t in JOB.negative_signals.consulting_firms if t]
PRODUCT_COMPANIES = [t for t in JOB.positive_signals.product_companies if t]

# Neutral companies: not consulting firms, not product companies, so they add no
# trajectory bonus or penalty on their own.
NEUTRAL_COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Labs"]
# Titles with no positive-title and no build fragments -> title_relevance 0.
UNRELATED_TITLES = ["Project Lead", "Team Lead", "Coordinator", "Specialist", "Associate"]

PROFICIENCIES = ["beginner", "intermediate", "advanced", "expert"]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _signals() -> RedrobSignals:
    """A minimal RedrobSignals; FeatureExtractor does not read these fields."""
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


def _profile(current_title: str, yoe: float = 7.0) -> Profile:
    return Profile(
        anonymized_name="Test Person",
        headline="",
        summary="",
        location="Pune",
        country="India",
        years_of_experience=yoe,
        current_title=current_title,
        current_company="Acme Corp",
        current_company_size="201-500",
        current_industry="Software",
    )


def _career(
    company: str,
    title: str,
    duration_months: int,
    description: str = "",
) -> CareerEntry:
    return CareerEntry(
        company=company,
        title=title,
        start_date="2019-01-01",
        end_date="2021-01-01",
        duration_months=duration_months,
        is_current=False,
        industry="Software",
        company_size="201-500",
        description=description,
    )


def _record(
    current_title: str,
    career: list[CareerEntry],
    skills: list[Skill],
    yoe: float = 7.0,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id="CAND_0000001",
        profile=_profile(current_title, yoe),
        career_history=career,
        education=[],
        skills=skills,
        redrob_signals=_signals(),
    )


def _fit(rec: CandidateRecord) -> float:
    return SCORER.fit_score(EXTRACTOR.extract(rec, NEUTRAL_SEM))


def _skills_title(rec: CandidateRecord) -> float:
    return EXTRACTOR.extract(rec, NEUTRAL_SEM).skills_title


def _trajectory(rec: CandidateRecord) -> float:
    return EXTRACTOR.extract(rec, NEUTRAL_SEM).trajectory


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def skill_lists(draw, min_size: int = 1, max_size: int = 5):
    """A bounded list of job-relevant skills with varied proficiency/tenure."""
    return draw(
        st.lists(
            st.builds(
                lambda name, prof, dur, end: Skill(
                    name=name, proficiency=prof, endorsements=end, duration_months=dur
                ),
                st.sampled_from(SKILL_TERMS),
                st.sampled_from(PROFICIENCIES),
                st.integers(min_value=0, max_value=60),
                st.integers(min_value=0, max_value=20),
            ),
            min_size=min_size,
            max_size=max_size,
        )
    )


# ---------------------------------------------------------------------------
# Property 6 (task 7.2)
# ---------------------------------------------------------------------------
# Feature: ai-candidate-ranking, Property 6: A keyword-stuffer ranks below an otherwise-genuine equivalent
@settings(max_examples=200, deadline=None)
@given(
    skills=skill_lists(),
    genuine_current=st.sampled_from(GENUINE_TITLES),
    genuine_hist=st.sampled_from(GENUINE_TITLES),
    stuffer_current=st.sampled_from(STUFFER_TITLES),
    stuffer_hist=st.sampled_from(STUFFER_TITLES),
    company=st.sampled_from(NEUTRAL_COMPANIES),
    duration=st.integers(min_value=19, max_value=60),
)
def test_property6_keyword_stuffer_ranks_below_genuine(
    skills, genuine_current, genuine_hist, stuffer_current, stuffer_hist, company, duration
):
    """Genuine title + matching history beats the same skills under an unrelated title.

    Validates: Requirements 4.1, 4.3
    """
    # A and B share the exact same skills, company, duration and (empty)
    # descriptions; they differ ONLY in their titles. A single, long career
    # entry avoids any title-chasing trajectory effect, so trajectory is equal
    # for both and the difference is attributable to skills_title.
    genuine = _record(
        current_title=genuine_current,
        career=[_career(company, genuine_hist, duration)],
        skills=skills,
    )
    stuffer = _record(
        current_title=stuffer_current,
        career=[_career(company, stuffer_hist, duration)],
        skills=skills,
    )

    # The title-aware skills dimension strictly favours the genuine candidate.
    assert _skills_title(genuine) > _skills_title(stuffer) + TOL
    # And so does the blended Fit_Score (all other dimensions are equal).
    assert _fit(genuine) > _fit(stuffer) + TOL


# ---------------------------------------------------------------------------
# Property 7 (task 7.3)
# ---------------------------------------------------------------------------
# Feature: ai-candidate-ranking, Property 7: Expert/advanced skills with zero duration are discounted
@settings(max_examples=200, deadline=None)
@given(
    target_name=st.sampled_from(SKILL_TERMS),
    proficiency=st.sampled_from(["advanced", "expert"]),
    pos_duration=st.integers(min_value=1, max_value=60),
    endorsements=st.integers(min_value=0, max_value=20),
    others=skill_lists(min_size=0, max_size=3),
)
def test_property7_zero_duration_skill_is_discounted(
    target_name, proficiency, pos_duration, endorsements, others
):
    """Zeroing duration on an expert/advanced relevant skill lowers skills_title.

    Validates: Requirements 4.2
    """
    # Two candidates that differ ONLY in the target skill's duration_months.
    # Using an empty title keeps title_align at 0, so skills_title == the trust
    # term alone (no clamping at 1.0 can mask the difference).
    target_zero = Skill(
        name=target_name,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=0,
    )
    target_pos = Skill(
        name=target_name,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=pos_duration,
    )

    rec_zero = _record("", [], [*others, target_zero])
    rec_pos = _record("", [], [*others, target_pos])

    st_zero = _skills_title(rec_zero)
    st_pos = _skills_title(rec_pos)

    # Zero-duration never increases the dimension...
    assert st_zero <= st_pos + TOL
    # ...and for a genuinely relevant skill it strictly decreases it.
    assert st_pos > st_zero + TOL


# ---------------------------------------------------------------------------
# Property 8 (task 7.4)
# ---------------------------------------------------------------------------
# Feature: ai-candidate-ranking, Property 8: Negative signals reduce Fit_Score
@settings(max_examples=200, deadline=None)
@given(
    skills=skill_lists(),
    title=st.sampled_from(GENUINE_TITLES),
    companies=st.lists(st.sampled_from(NEUTRAL_COMPANIES), min_size=2, max_size=2),
    durations=st.lists(st.integers(min_value=19, max_value=60), min_size=2, max_size=2),
    consulting_firm=st.sampled_from(CONSULTING_FIRMS),
)
def test_property8_negative_signal_reduces_fit(
    skills, title, companies, durations, consulting_firm
):
    """Turning a clean career into a consulting-only one cannot raise Fit_Score.

    Validates: Requirements 3.4
    """
    # Baseline: two long stints at neutral (non-consulting, non-product)
    # companies -> no trajectory bonus or penalty.
    baseline_career = [
        _career(companies[0], title, durations[0]),
        _career(companies[1], title, durations[1]),
    ]
    baseline = _record(title, baseline_career, skills)

    # Variant: identical except every company is the same consulting firm,
    # introducing the consulting-only negative signal.
    variant_career = [
        _career(consulting_firm, title, durations[0]),
        _career(consulting_firm, title, durations[1]),
    ]
    variant = _record(title, variant_career, skills)

    # The negative signal cannot raise the trajectory dimension...
    assert _trajectory(variant) <= _trajectory(baseline) + TOL
    # ...nor the blended Fit_Score.
    assert _fit(variant) <= _fit(baseline) + TOL


# ---------------------------------------------------------------------------
# Property 9 (task 7.5)
# ---------------------------------------------------------------------------
# Feature: ai-candidate-ranking, Property 9: Product ranking/search experience does not lower fit
@settings(max_examples=200, deadline=None)
@given(
    skills=skill_lists(),
    base_titles=st.lists(st.sampled_from(UNRELATED_TITLES), min_size=1, max_size=2),
    base_companies=st.lists(st.sampled_from(NEUTRAL_COMPANIES), min_size=1, max_size=2),
    durations=st.lists(st.integers(min_value=19, max_value=60), min_size=1, max_size=2),
    product_company=st.sampled_from(PRODUCT_COMPANIES),
    injected_title=st.sampled_from(UNRELATED_TITLES),
    injected_duration=st.integers(min_value=24, max_value=60),
)
def test_property9_product_experience_does_not_lower_fit(
    skills,
    base_titles,
    base_companies,
    durations,
    product_company,
    injected_title,
    injected_duration,
):
    """Injecting "built a recsys/search/ranking system at a product co." never lowers fit.

    Validates: Requirements 3.3
    """
    # Build a baseline with unrelated titles, neutral companies, long stints and
    # NO build/AI buzzwords anywhere (skills/headline/descriptions are neutral).
    n = min(len(base_titles), len(base_companies), len(durations))
    base_career = [
        _career(base_companies[i], base_titles[i], durations[i]) for i in range(n)
    ]
    # The injected skills/headline stay buzzword-free; the only AI evidence is in
    # the injected career entry's description (a shipped product system).
    baseline = _record("", base_career, skills)

    injected = _career(
        product_company,
        injected_title,
        injected_duration,
        description=(
            "Built and shipped a recommendation and search ranking system "
            "serving real users in production."
        ),
    )
    variant = _record("", [*base_career, injected], skills)

    # Adding product ranking/search build-evidence can only raise trajectory...
    assert _trajectory(variant) >= _trajectory(baseline) - TOL
    # ...and therefore cannot lower the blended Fit_Score (titles kept unrelated
    # for both, so skills_title is unchanged).
    assert _fit(variant) >= _fit(baseline) - TOL
