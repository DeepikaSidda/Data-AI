"""Property-based tests for the offline grounded ReasoningGenerator.

# Feature: ai-candidate-ranking, Property 18: Reasoning references only facts present in the record
# Feature: ai-candidate-ranking, Property 19: Reasoning is 1-2 sentences and acknowledges present concerns
# Feature: ai-candidate-ranking, Property 20: Reasoning is varied, not templated

Validates Requirements: 8.1, 8.2, 8.4, 8.5, 8.6.

These tests exercise ``ranking.reasoning.ReasoningGenerator`` against the real,
committed ``job_profile.yaml`` (loaded once, fully offline). Strategies build
varied ``CandidateRecord`` instances whose skills and employers are drawn from a
"legit" pool; a disjoint "decoy" pool of skill / employer names that are *never*
placed on the record is used to assert the generator never invents a skill or
employer that the candidate does not actually have (the robust grounding check
suggested by Property 18).

The concern phrasing, sentence-joining, and clause wording asserted below were
read directly from ``reasoning.py`` so the assertions track the generator's
actual output rather than an idealized form.
"""

from __future__ import annotations

import datetime
import os
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from ranking.job_profile import JobProfile
from ranking.models import (
    CandidateRecord,
    CareerEntry,
    DimensionScores,
    HoneypotResult,
    Profile,
    RedrobSignals,
    Skill,
)
from ranking.reasoning import ReasoningGenerator

# ---------------------------------------------------------------------------
# Fixtures / shared constants
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Real, committed Job_Profile loaded once (pure offline, local file I/O only).
JOB = JobProfile.load(os.path.join(_REPO_ROOT, "job_profile.yaml"))

# Reference "most recent activity across the pool" for inactivity detection.
POOL_LATEST_ACTIVE = datetime.date(2026, 6, 1)

# Mirror reasoning.py's sentence terminator regex so sentence counting matches.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")

# Real skill names actually placed on records. A mix of role-relevant terms
# (classified by the generator) and ordinary-but-real skills.
LEGIT_SKILLS = (
    "Python",
    "PyTorch",
    "TensorFlow",
    "Elasticsearch",
    "FAISS",
    "Pinecone",
    "Semantic Search",
    "Information Retrieval",
    "Recommendation Systems",
    "NDCG",
    "Docker",
    "Kubernetes",
    "PostgreSQL",
    "React",
)

# Disjoint decoy skills: real-sounding names that are NEVER put on any record.
# None of these tokens appear in any static phrase emitted by reasoning.py, so
# if one shows up in the reasoning the generator hallucinated it.
DECOY_SKILLS = (
    "Rust",
    "Haskell",
    "COBOL",
    "Fortran",
    "Erlang",
    "Solidity",
    "Verilog",
    "Clojure",
)

# Product companies recognised by the Job_Profile (a "plus" signal).
PRODUCT_EMPLOYERS = ("Google", "Amazon", "Flipkart", "Swiggy", "Razorpay")
# Consulting / services firms recognised as a negative signal.
CONSULTING_EMPLOYERS = ("TCS", "Infosys", "Wipro", "Accenture", "Cognizant")
# Real product-ish employers NOT in any Job_Profile list (neutral).
NEUTRAL_EMPLOYERS = ("Freshworks", "Zoho", "Postman", "BrowserStack")

# Disjoint decoy employers never placed on a record.
DECOY_EMPLOYERS = ("Wibblesoft", "Florbcorp", "Zentar", "Quizzly", "Blorptech")

# Verbatim concern markers emitted by ReasoningGenerator._concern.
CONCERN_MARKERS = (
    "data inconsistencies",            # honeypot
    "consulting-services-only",        # consulting-only background
    "raises the bar",                  # notice period >= 30 days
    "inactive for an extended period",  # profile inactivity >= ~6 months
    "fit dimensions are weaker",       # weak fit dimensions
)

# Candidate titles (some match Job_Profile title_terms, some do not).
TITLES = (
    "Senior AI Engineer",
    "Machine Learning Engineer",
    "Backend Engineer",
    "Data Scientist",
    "Product Manager",
    "Senior Developer",
)


def _sentence_count(text: str) -> int:
    """Count sentences the same way reasoning.py does (decimal-safe)."""
    return len(_SENTENCE_END.findall(text))


def _iso(d: datetime.date) -> str:
    return d.isoformat()


# ---------------------------------------------------------------------------
# Record construction helpers (plain builders fed by drawn values)
# ---------------------------------------------------------------------------


def _make_signals(notice: int, last_active: datetime.date) -> RedrobSignals:
    """A valid RedrobSignals; only the fields reasoning reads are varied."""
    return RedrobSignals(
        profile_completeness_score=0.9,
        signup_date=_iso(POOL_LATEST_ACTIVE - datetime.timedelta(days=900)),
        last_active_date=_iso(last_active),
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=2,
        recruiter_response_rate=0.5,
        avg_response_time_hours=24.0,
        connection_count=300,
        endorsements_received=20,
        notice_period_days=notice,
        preferred_work_mode="hybrid",
        willing_to_relocate=True,
        search_appearance_30d=15,
        saved_by_recruiters_30d=3,
        interview_completion_rate=0.8,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )


def _make_record(
    cid: str,
    years: float,
    title: str,
    skill_names: list[str],
    employer_names: list[str],
    notice: int,
    last_active: datetime.date,
) -> CandidateRecord:
    """Assemble a CandidateRecord from drawn primitive values."""
    profile = Profile(
        anonymized_name="Candidate",
        headline="",
        summary="",
        location="Pune",
        country="India",
        years_of_experience=years,
        current_title=title,
        current_company=employer_names[0] if employer_names else "",
        current_company_size="201-500",
        current_industry="Technology",
    )
    skills = [
        Skill(name=name, proficiency="advanced", endorsements=5)
        for name in skill_names
    ]
    career = [
        CareerEntry(
            company=company,
            title=title,
            start_date="2018-01-01",
            end_date=None,
            duration_months=24,
            is_current=(idx == 0),
            industry="Technology",
            company_size="201-500",
            description="",
        )
        for idx, company in enumerate(employer_names)
    ]
    return CandidateRecord(
        candidate_id=cid,
        profile=profile,
        career_history=career,
        education=[],
        skills=skills,
        redrob_signals=_make_signals(notice, last_active),
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_candidate_id = st.integers(min_value=0, max_value=9_999_999).map(
    lambda i: f"CAND_{i:07d}"
)
_years = st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False)
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_skill_names = st.lists(
    st.sampled_from(LEGIT_SKILLS), min_size=1, max_size=5, unique=True
)
_employer_names = st.lists(
    st.sampled_from(PRODUCT_EMPLOYERS + NEUTRAL_EMPLOYERS),
    min_size=1,
    max_size=3,
    unique=True,
)


@st.composite
def dimension_scores(draw: st.DrawFn) -> DimensionScores:
    """Arbitrary per-dimension scores in [0, 1]."""
    return DimensionScores(
        semantic=draw(_unit),
        skills_title=draw(_unit),
        experience=draw(_unit),
        trajectory=draw(_unit),
        education=draw(_unit),
    )


@st.composite
def candidate_record(draw: st.DrawFn) -> CandidateRecord:
    """A varied, fully-grounded CandidateRecord built from the legit pools."""
    notice = draw(st.integers(min_value=0, max_value=180))
    stale_days = draw(st.integers(min_value=0, max_value=800))
    last_active = POOL_LATEST_ACTIVE - datetime.timedelta(days=stale_days)
    return _make_record(
        cid=draw(_candidate_id),
        years=draw(_years),
        title=draw(st.sampled_from(TITLES)),
        skill_names=draw(_skill_names),
        employer_names=draw(_employer_names),
        notice=notice,
        last_active=last_active,
    )


def _references_concrete_fact(text: str, rec: CandidateRecord) -> bool:
    """True if reasoning surfaces at least one concrete record fact (8.2)."""
    facts = []
    years = rec.profile.years_of_experience
    if years and years > 0:
        facts.append(f"{years:g} years of experience" in text)
    facts.append(any(s.name and s.name in text for s in rec.skills))
    facts.append(any(c.company and c.company in text for c in rec.career_history))
    if rec.profile.current_title:
        facts.append(rec.profile.current_title in text)
    notice = rec.redrob_signals.notice_period_days
    facts.append(
        f"notice period of {notice} days" in text
        or f"{notice}-day notice period" in text
    )
    return any(facts)


# ---------------------------------------------------------------------------
# Property 18: Reasoning references only facts present in the record
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    rec=candidate_record(),
    dims=dimension_scores(),
    is_honeypot=st.booleans(),
    rank=st.integers(min_value=1, max_value=100),
)
def test_reasoning_only_references_present_facts(rec, dims, is_honeypot, rank):
    """Property 18: no invented skills/employers; >=1 concrete grounded fact.

    Validates Requirements 8.2, 8.5.
    """
    gen = ReasoningGenerator(JOB, pool_latest_active=POOL_LATEST_ACTIVE)
    text = gen.generate(rec, dims, HoneypotResult(is_honeypot=is_honeypot), rank)
    lower = text.lower()

    # No decoy skill (one absent from the record) may appear in the reasoning.
    for decoy in DECOY_SKILLS:
        assert not re.search(rf"\b{re.escape(decoy.lower())}\b", lower), (
            f"hallucinated skill {decoy!r} in reasoning: {text!r}"
        )

    # No decoy employer may appear either.
    for decoy in DECOY_EMPLOYERS:
        assert not re.search(rf"\b{re.escape(decoy.lower())}\b", lower), (
            f"hallucinated employer {decoy!r} in reasoning: {text!r}"
        )

    # The reasoning must surface at least one concrete fact from the record.
    assert _references_concrete_fact(text, rec), (
        f"reasoning references no concrete record fact: {text!r}"
    )


# ---------------------------------------------------------------------------
# Property 19: 1-2 sentences and acknowledges present concerns
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    rec=candidate_record(),
    dims=dimension_scores(),
    is_honeypot=st.booleans(),
    rank=st.integers(min_value=1, max_value=100),
)
def test_reasoning_is_one_or_two_sentences(rec, dims, is_honeypot, rank):
    """Property 19 (bounds): reasoning is always 1-2 sentences.

    Validates Requirement 8.1.
    """
    gen = ReasoningGenerator(JOB, pool_latest_active=POOL_LATEST_ACTIVE)
    text = gen.generate(rec, dims, HoneypotResult(is_honeypot=is_honeypot), rank)
    count = _sentence_count(text)
    assert 1 <= count <= 2, f"expected 1-2 sentences, got {count}: {text!r}"


@st.composite
def concern_scenario(draw: st.DrawFn):
    """A record guaranteed to carry at least one notable concern.

    Returns ``(generator, rec, dims, honeypot, rank)`` where, by construction,
    the generator's concern detector must fire (some concern condition is true
    and no setup masks every condition).
    """
    kind = draw(
        st.sampled_from(["honeypot", "notice", "consulting", "inactivity", "weak"])
    )
    cid = draw(_candidate_id)
    years = draw(_years)
    title = draw(st.sampled_from(TITLES))
    skills = draw(_skill_names)
    rank = draw(st.integers(min_value=1, max_value=100))

    # Clean defaults: no concern unless the chosen kind introduces one.
    notice = draw(st.integers(min_value=0, max_value=29))
    employers = draw(_employer_names)  # product/neutral -> not consulting-only
    last_active = POOL_LATEST_ACTIVE - datetime.timedelta(
        days=draw(st.integers(min_value=0, max_value=60))
    )
    honeypot = HoneypotResult(is_honeypot=False)
    pool = None
    # Strong dims so the weak-dimension concern does not fire incidentally.
    dims = DimensionScores(
        semantic=draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
        skills_title=draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
        experience=draw(_unit),
        trajectory=draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
        education=draw(_unit),
    )

    if kind == "honeypot":
        honeypot = HoneypotResult(is_honeypot=True, reasons=["inconsistent"])
    elif kind == "notice":
        notice = draw(st.integers(min_value=30, max_value=180))
    elif kind == "consulting":
        # Every employer a services firm -> consulting-only background.
        employers = draw(
            st.lists(
                st.sampled_from(CONSULTING_EMPLOYERS),
                min_size=1,
                max_size=3,
                unique=True,
            )
        )
    elif kind == "inactivity":
        # Provide the pool reference and a stale activity date (>= ~6 months).
        pool = POOL_LATEST_ACTIVE
        last_active = POOL_LATEST_ACTIVE - datetime.timedelta(
            days=draw(st.integers(min_value=183, max_value=800))
        )
    else:  # weak dimensions; mask all higher-priority concerns
        dims = DimensionScores(
            semantic=draw(st.floats(min_value=0.0, max_value=0.4, allow_nan=False)),
            skills_title=draw(_unit),
            experience=draw(_unit),
            trajectory=draw(_unit),
            education=draw(_unit),
        )

    rec = _make_record(cid, years, title, skills, employers, notice, last_active)
    gen = ReasoningGenerator(JOB, pool_latest_active=pool)
    return gen, rec, dims, honeypot, rank


@settings(max_examples=100, deadline=None)
@given(scenario=concern_scenario())
def test_reasoning_acknowledges_present_concern(scenario):
    """Property 19 (concern): a present concern is acknowledged in 1-2 sentences.

    Validates Requirement 8.4.
    """
    gen, rec, dims, honeypot, rank = scenario
    text = gen.generate(rec, dims, honeypot, rank)

    assert any(marker in text for marker in CONCERN_MARKERS), (
        f"expected a concern clause, got: {text!r}"
    )
    # Acknowledging a concern must not break the sentence bound.
    assert 1 <= _sentence_count(text) <= 2, f"sentence bound broken: {text!r}"


# ---------------------------------------------------------------------------
# Property 20: Reasoning is varied, not templated
# ---------------------------------------------------------------------------


@st.composite
def distinct_scenarios(draw: st.DrawFn):
    """A list of >=10 scenarios with distinct candidate_ids and varied features."""
    raw_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=9_999_999),
            min_size=10,
            max_size=20,
            unique=True,
        )
    )
    scenarios = []
    for i in raw_ids:
        cid = f"CAND_{i:07d}"
        notice = draw(st.integers(min_value=0, max_value=180))
        last_active = POOL_LATEST_ACTIVE - datetime.timedelta(
            days=draw(st.integers(min_value=0, max_value=800))
        )
        rec = _make_record(
            cid=cid,
            years=draw(_years),
            title=draw(st.sampled_from(TITLES)),
            skill_names=draw(_skill_names),
            employer_names=draw(_employer_names),
            notice=notice,
            last_active=last_active,
        )
        dims = draw(dimension_scores())
        honeypot = HoneypotResult(is_honeypot=draw(st.booleans()))
        rank = draw(st.integers(min_value=1, max_value=100))
        scenarios.append((rec, dims, honeypot, rank))
    return scenarios


@settings(max_examples=100, deadline=None)
@given(scenarios=distinct_scenarios())
def test_reasoning_is_varied(scenarios):
    """Property 20: distinct candidates yield a high proportion of distinct text.

    Validates Requirement 8.6.
    """
    gen = ReasoningGenerator(JOB, pool_latest_active=POOL_LATEST_ACTIVE)
    texts = [gen.generate(rec, dims, hp, rank) for rec, dims, hp, rank in scenarios]

    unique_ratio = len(set(texts)) / len(texts)
    assert unique_ratio >= 0.7, (
        f"reasoning looks templated: only {unique_ratio:.0%} distinct "
        f"across {len(texts)} candidates"
    )
