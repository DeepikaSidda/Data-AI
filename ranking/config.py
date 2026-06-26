"""Scoring configuration (weights, bounds, validation).

Defines the frozen :class:`ScoringConfig` dataclass that controls how the
scoring dimensions combine into the ``Fit_Score`` and how strongly the
behavioral modifier adjusts the ``Final_Score``. A ``load(path)`` classmethod
returns documented defaults when no path is supplied, otherwise merges
YAML/JSON overrides over those defaults and validates the result. ``validate()``
enforces the weight constraints from Requirements 12.4 and 12.5.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

from ranking.errors import WeightValidationError

# The five scoring dimensions that combine into the Fit_Score. These are the
# weights subject to the "not all zero" rule of Requirement 12.5;
# ``behavioral_strength`` and ``honeypot_penalty`` are deliberately excluded
# from that all-zero check (they tune the modifier/penalty, not the Fit_Score
# blend).
_DIMENSION_WEIGHT_NAMES: tuple[str, ...] = (
    "w_semantic",
    "w_skills_title",
    "w_experience",
    "w_trajectory",
    "w_education",
)

# Every numeric field that must be non-negative (Requirement 12.4).
_NON_NEGATIVE_FIELD_NAMES: tuple[str, ...] = _DIMENSION_WEIGHT_NAMES + (
    "behavioral_strength",
    "honeypot_penalty",
)


@dataclass(frozen=True)
class ScoringConfig:
    """Weights and bounds controlling Fit_Score and the behavioral modifier.

    The five ``w_*`` fields are the dimension weights blended into the
    ``Fit_Score`` (a convex combination once normalized, so the result stays in
    ``[0, 1]``). They default to a documented baseline that puts the most weight
    on semantic similarity and title-aware skills alignment.

    Attributes:
        w_semantic: Weight of the semantic (embedding cosine) dimension. Default ``0.35``.
        w_skills_title: Weight of the title-aware skills-alignment dimension. Default ``0.25``.
        w_experience: Weight of the years-of-experience fit dimension. Default ``0.15``.
        w_trajectory: Weight of the career-trajectory (product-vs-services) dimension. Default ``0.20``.
        w_education: Weight of the education-tier dimension. Default ``0.05``.
        behavioral_strength: Strength ``s`` of the behavioral modifier, which spans
            ``[1 - s, 1 + s]``. Intended to lie in ``[0, 1)`` so the modifier adjusts
            but never solely determines the Final_Score (Requirement 5.5). Default ``0.15``.
        honeypot_penalty: Multiplier applied to a flagged honeypot's Final_Score.
            The default ``0.0`` zeroes flagged records so they cannot reach the top 100.
    """

    w_semantic: float = 0.35
    w_skills_title: float = 0.25
    w_experience: float = 0.15
    w_trajectory: float = 0.20
    w_education: float = 0.05
    # Behavioral modifier strength: the modifier spans [1 - s, 1 + s]. Intended
    # to be in [0, 1) so the modifier adjusts but never dominates the Fit_Score.
    behavioral_strength: float = 0.15
    # Multiplier applied to flagged honeypot records (default 0.0 -> excluded).
    honeypot_penalty: float = 0.0

    @classmethod
    def load(cls, path: str | None) -> "ScoringConfig":
        """Build a validated config from defaults, optionally merging overrides.

        When ``path`` is ``None`` the documented defaults are returned
        (Requirement 12.2). Otherwise the file is parsed as YAML (``.yaml`` /
        ``.yml``) or JSON (``.json``) by extension, and any recognized keys are
        merged over the defaults (Requirement 12.3). Unknown keys are ignored so
        a config file may carry extra annotations. The resulting config is
        validated before it is returned (Requirements 12.4, 12.5).

        Args:
            path: Path to a YAML/JSON override file, or ``None`` for defaults.

        Returns:
            A validated :class:`ScoringConfig`.

        Raises:
            WeightValidationError: If the merged config has a negative weight or
                all five dimension weights are zero.
            ValueError: If ``path`` has an unrecognized extension.
        """
        if path is None:
            config = cls()
            config.validate()
            return config

        overrides = cls._read_overrides(path)
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in overrides.items() if k in known}
        config = replace(cls(), **filtered)
        config.validate()
        return config

    @staticmethod
    def _read_overrides(path: str) -> Mapping[str, Any]:
        """Parse a YAML or JSON override file into a mapping, detected by extension."""
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as handle:
            if ext in (".yaml", ".yml"):
                import yaml

                data = yaml.safe_load(handle)
            elif ext == ".json":
                data = json.load(handle)
            else:
                raise ValueError(
                    f"Unsupported scoring config extension '{ext}' for path '{path}'; "
                    "use a .yaml, .yml, or .json file."
                )
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            raise ValueError(
                f"Scoring config at '{path}' must be a mapping of weight names to values."
            )
        return data

    def validate(self) -> None:
        """Validate the weight constraints, raising on violations.

        Raises:
            WeightValidationError: If any weight (the five dimension weights,
                ``behavioral_strength``, or ``honeypot_penalty``) is negative
                (Requirement 12.4), or if all five dimension weights are zero
                (Requirement 12.5). ``behavioral_strength`` and
                ``honeypot_penalty`` are excluded from the all-zero check.
        """
        for name in _NON_NEGATIVE_FIELD_NAMES:
            value = getattr(self, name)
            if value < 0:
                raise WeightValidationError(
                    f"Scoring weight '{name}' must be non-negative, got {value}."
                )

        dimension_values = [getattr(self, name) for name in _DIMENSION_WEIGHT_NAMES]
        if all(value == 0 for value in dimension_values):
            raise WeightValidationError(
                "At least one scoring dimension weight must be greater than zero; "
                f"all of {', '.join(_DIMENSION_WEIGHT_NAMES)} are zero."
            )

    @property
    def dimension_weights(self) -> dict[str, float]:
        """The five Fit_Score dimension weights as an ordered name→weight mapping.

        This is the helper the scorer uses to blend the dimensions: it returns
        only the five ``w_*`` weights (excluding ``behavioral_strength`` and
        ``honeypot_penalty``) in the canonical dimension order. Insertion order
        follows :data:`_DIMENSION_WEIGHT_NAMES` so callers can rely on a stable
        iteration order (Requirement 12.1).

        Returns:
            A new ``dict`` mapping each dimension weight name to its value.
        """
        return {name: getattr(self, name) for name in _DIMENSION_WEIGHT_NAMES}
