"""Property-based tests for the HybridScorer Fit_Score.

# Feature: ai-candidate-ranking, Property 4: Fit_Score is bounded in [0,1] and matches the weighted formula
# Feature: ai-candidate-ranking, Property 5: Scoring is total and deterministic

Validates Requirements: 3.1, 3.4, 3.5, 3.6, 12.3.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ranking.config import ScoringConfig
from ranking.models import DimensionScores
from ranking.scorer import HybridScorer

# A single normalized dimension score lives in [0, 1].
_DIM = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
# A single non-negative dimension weight; drawn from [0, 5] per the task.
_WEIGHT = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)


@st.composite
def scoring_configs(draw: st.DrawFn) -> ScoringConfig:
    """Draw a valid ScoringConfig: five non-negative weights, not all zero."""
    w_semantic = draw(_WEIGHT)
    w_skills_title = draw(_WEIGHT)
    w_experience = draw(_WEIGHT)
    w_trajectory = draw(_WEIGHT)
    w_education = draw(_WEIGHT)
    # Reject the all-zero configuration (forbidden by ScoringConfig.validate).
    assume(
        (w_semantic + w_skills_title + w_experience + w_trajectory + w_education) > 0.0
    )
    return ScoringConfig(
        w_semantic=w_semantic,
        w_skills_title=w_skills_title,
        w_experience=w_experience,
        w_trajectory=w_trajectory,
        w_education=w_education,
    )


@st.composite
def dimension_scores(draw: st.DrawFn) -> DimensionScores:
    """Draw a DimensionScores with each field in [0, 1]."""
    return DimensionScores(
        semantic=draw(_DIM),
        skills_title=draw(_DIM),
        experience=draw(_DIM),
        trajectory=draw(_DIM),
        education=draw(_DIM),
    )


# Feature: ai-candidate-ranking, Property 4: Fit_Score is bounded in [0,1] and matches the weighted formula
@settings(max_examples=100, deadline=None)
@given(dims=dimension_scores(), config=scoring_configs())
def test_fit_score_bounded_and_matches_formula(dims: DimensionScores, config: ScoringConfig) -> None:
    """Fit_Score lies in [0,1] and equals the normalized weighted sum.

    Validates: Requirements 3.1, 3.4, 12.3.
    """
    scorer = HybridScorer(config)
    fit = scorer.fit_score(dims)

    # Bounded in [0, 1].
    assert 0.0 <= fit <= 1.0

    # Matches the normalized weighted-sum formula sum(w_i*d_i)/sum(w_i).
    weights = config.dimension_weights
    weight_sum = (
        weights["w_semantic"]
        + weights["w_skills_title"]
        + weights["w_experience"]
        + weights["w_trajectory"]
        + weights["w_education"]
    )
    raw = (
        weights["w_semantic"] * dims.semantic
        + weights["w_skills_title"] * dims.skills_title
        + weights["w_experience"] * dims.experience
        + weights["w_trajectory"] * dims.trajectory
        + weights["w_education"] * dims.education
    )
    expected = raw / weight_sum
    assert abs(fit - expected) <= 1e-9


# Feature: ai-candidate-ranking, Property 5: Scoring is total and deterministic
@settings(max_examples=100, deadline=None)
@given(dims=dimension_scores(), config=scoring_configs())
def test_fit_score_total_and_deterministic(dims: DimensionScores, config: ScoringConfig) -> None:
    """fit_score returns a value for every input and is deterministic.

    Validates: Requirements 3.5, 3.6.
    """
    scorer = HybridScorer(config)

    # Total: returns a float for any valid input (no exception raised).
    first = scorer.fit_score(dims)
    assert isinstance(first, float)

    # Deterministic: calling twice yields identical results.
    second = scorer.fit_score(dims)
    assert first == second
