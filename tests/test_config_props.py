"""Property-based tests for ScoringConfig weight validation.

# Feature: ai-candidate-ranking, Property 22: Weight validation rejects negative and all-zero configurations

Validates Requirements: 12.4, 12.5.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ranking.config import ScoringConfig
from ranking.errors import WeightValidationError

# Five dimension weights + behavioral_strength + honeypot_penalty.
_NON_NEG = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_ANY = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


def _make(weights: tuple[float, float, float, float, float], strength: float, penalty: float) -> ScoringConfig:
    w_semantic, w_skills_title, w_experience, w_trajectory, w_education = weights
    return ScoringConfig(
        w_semantic=w_semantic,
        w_skills_title=w_skills_title,
        w_experience=w_experience,
        w_trajectory=w_trajectory,
        w_education=w_education,
        behavioral_strength=strength,
        honeypot_penalty=penalty,
    )


@settings(max_examples=100)
@given(
    weights=st.tuples(_ANY, _ANY, _ANY, _ANY, _ANY),
    strength=_ANY,
    penalty=_ANY,
)
def test_negative_weight_is_rejected(weights, strength, penalty):
    """Any config carrying at least one negative weight must raise (Req 12.4)."""
    fields = list(weights) + [strength, penalty]
    config = _make(weights, strength, penalty)
    if any(value < 0 for value in fields):
        with pytest.raises(WeightValidationError):
            config.validate()


@settings(max_examples=100)
@given(strength=_NON_NEG, penalty=_NON_NEG)
def test_all_zero_dimension_weights_rejected(strength, penalty):
    """A config whose five dimension weights are all zero must raise (Req 12.5)."""
    config = _make((0.0, 0.0, 0.0, 0.0, 0.0), strength, penalty)
    with pytest.raises(WeightValidationError):
        config.validate()


@settings(max_examples=100)
@given(
    weights=st.tuples(_NON_NEG, _NON_NEG, _NON_NEG, _NON_NEG, _NON_NEG),
    strength=_NON_NEG,
    penalty=_NON_NEG,
)
def test_valid_configs_do_not_raise(weights, strength, penalty):
    """Non-negative, not-all-zero configs must validate cleanly."""
    # Constrain to the valid input space: at least one dimension weight positive.
    if all(value == 0 for value in weights):
        return
    config = _make(weights, strength, penalty)
    config.validate()  # must not raise
