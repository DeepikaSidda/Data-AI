"""Property-based tests for deterministic top-N ranking and tie-breaking.

# Feature: ai-candidate-ranking, Property 14: The shortlist has exactly 100 rows with unique ranks 1..100
# Feature: ai-candidate-ranking, Property 15: Final_Score is monotonically non-increasing with rank
# Feature: ai-candidate-ranking, Property 16: Ties break by candidate_id ascending
# Feature: ai-candidate-ranking, Property 17: Ranking is deterministic

Exercises :meth:`ranking.ranker.Ranker.rank`, which sorts scored candidates by
the composite key ``(-final_score, candidate_id)`` and assigns unique ranks
``1..n`` to the top ``top_n``. Scores are drawn from a small pool so equal
scores (ties) occur frequently, stressing the tie-break and determinism
guarantees described in the design's "Tie-break and ranking" subsection.

Validates Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.3, 9.5, 9.6.
"""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from ranking.models import (
    CandidateRecord,
    DimensionScores,
    HoneypotResult,
    Profile,
    RedrobSignals,
    ScoredCandidate,
)
from ranking.ranker import Ranker


# ---------------------------------------------------------------------------
# Minimal builders / generators
# ---------------------------------------------------------------------------

# A small pool of repeated score values so ties are common across a list.
_SCORE_POOL = [0.0, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0]

_DIGITS = st.text(alphabet="0123456789", min_size=7, max_size=7)


def _minimal_record(candidate_id: str) -> CandidateRecord:
    """A minimally-constructed, schema-shaped CandidateRecord."""
    profile = Profile(
        anonymized_name="",
        headline="",
        summary="",
        location="",
        country="",
        years_of_experience=0.0,
        current_title="",
        current_company="",
        current_company_size="",
        current_industry="",
    )
    signals = RedrobSignals(
        profile_completeness_score=0.0,
        signup_date="",
        last_active_date="",
        open_to_work_flag=False,
        profile_views_received_30d=0,
        applications_submitted_30d=0,
        recruiter_response_rate=0.0,
        avg_response_time_hours=0.0,
        connection_count=0,
        endorsements_received=0,
        notice_period_days=0,
        preferred_work_mode="",
        willing_to_relocate=False,
        search_appearance_30d=0,
        saved_by_recruiters_30d=0,
        interview_completion_rate=0.0,
        verified_email=False,
        verified_phone=False,
        linkedin_connected=False,
    )
    return CandidateRecord(
        candidate_id=candidate_id,
        profile=profile,
        career_history=[],
        education=[],
        skills=[],
        redrob_signals=signals,
    )


def _minimal_scored(candidate_id: str, final_score: float) -> ScoredCandidate:
    """Build a minimal ScoredCandidate carrying only id and final_score."""
    return ScoredCandidate(
        candidate_id=candidate_id,
        record=_minimal_record(candidate_id),
        dims=DimensionScores(
            semantic=0.0,
            skills_title=0.0,
            experience=0.0,
            trajectory=0.0,
            education=0.0,
        ),
        fit_score=0.0,
        behavioral_modifier=0.0,
        final_score=final_score,
        honeypot=HoneypotResult(False, []),
    )


@st.composite
def _scored_candidates(draw, min_size: int = 1, max_size: int = 150):
    """A list of ScoredCandidate with unique CAND_ ids and frequent score ties."""
    ids = draw(
        st.lists(_DIGITS, min_size=min_size, max_size=max_size, unique=True)
    )
    return [
        _minimal_scored(f"CAND_{seven}", draw(st.sampled_from(_SCORE_POOL)))
        for seven in ids
    ]


# ---------------------------------------------------------------------------
# Property 14 (task 12.2): shortlist size and unique ranks 1..n
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(
    scored=_scored_candidates(min_size=100, max_size=160),
    top_n=st.integers(min_value=1, max_value=200),
)
def test_shortlist_size_and_unique_ranks(scored, top_n):
    """rank() returns exactly min(top_n, N) rows with ranks == {1..len}.

    Validates Requirements: 7.1, 7.2, 9.3, 9.5.
    """
    result = Ranker().rank(scored, top_n=top_n)

    expected_len = min(top_n, len(scored))
    assert len(result) == expected_len

    ranks = [row.rank for row in result]
    assert sorted(ranks) == list(range(1, expected_len + 1))
    # Each rank appears exactly once.
    assert len(set(ranks)) == len(ranks)


# ---------------------------------------------------------------------------
# Property 15 (task 12.3): score monotonically non-increasing with rank
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(scored=_scored_candidates())
def test_score_monotonic_non_increasing_by_rank(scored):
    """result[i].score >= result[i+1].score and every score in [0,1].

    Validates Requirements: 7.3, 9.6.
    """
    result = Ranker().rank(scored, top_n=len(scored))

    for row in result:
        assert 0.0 <= row.score <= 1.0

    for earlier, later in zip(result, result[1:]):
        assert earlier.score >= later.score


# ---------------------------------------------------------------------------
# Property 16 (task 12.4): ties break by candidate_id ascending
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(scored=_scored_candidates())
def test_ties_break_by_candidate_id_ascending(scored):
    """Consecutive equal-score rows are ordered by ascending candidate_id.

    Validates Requirements: 7.4, 9.6.
    """
    result = Ranker().rank(scored, top_n=len(scored))

    for earlier, later in zip(result, result[1:]):
        if earlier.score == later.score:
            assert earlier.candidate_id < later.candidate_id


# ---------------------------------------------------------------------------
# Property 17 (task 12.5): ranking is deterministic
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(scored=_scored_candidates(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_ranking_is_deterministic(scored, seed):
    """Same input (and a shuffled copy) yields identical (rank, id, score) tuples.

    Validates Requirements: 7.5.
    """
    ranker = Ranker()

    def tuples(rows):
        return [(r.rank, r.candidate_id, r.score) for r in rows]

    first = ranker.rank(scored, top_n=100)
    second = ranker.rank(scored, top_n=100)

    shuffled = list(scored)
    random.Random(seed).shuffle(shuffled)
    third = ranker.rank(shuffled, top_n=100)

    assert tuples(first) == tuples(second)
    assert tuples(first) == tuples(third)
