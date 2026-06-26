"""Unit tests for ScoringConfig defaults.

Requirements: 12.1, 12.2.
"""

from __future__ import annotations

from ranking.config import ScoringConfig


def test_load_none_applies_documented_defaults():
    """load(None) returns the documented default weights (Req 12.2)."""
    config = ScoringConfig.load(None)

    assert config.w_semantic == 0.35
    assert config.w_skills_title == 0.25
    assert config.w_experience == 0.15
    assert config.w_trajectory == 0.20
    assert config.w_education == 0.05
    assert config.behavioral_strength == 0.15
    assert config.honeypot_penalty == 0.0


def test_all_weight_fields_exposed():
    """All weight/behavioral fields are exposed on the config (Req 12.1)."""
    config = ScoringConfig()

    for field_name in (
        "w_semantic",
        "w_skills_title",
        "w_experience",
        "w_trajectory",
        "w_education",
        "behavioral_strength",
        "honeypot_penalty",
    ):
        assert hasattr(config, field_name)
        assert isinstance(getattr(config, field_name), float)


def test_default_constructor_matches_load_none():
    """The bare constructor and load(None) agree on defaults (Req 12.2)."""
    assert ScoringConfig() == ScoringConfig.load(None)
