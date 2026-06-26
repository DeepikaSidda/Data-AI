"""Hybrid weighted Fit_Score.

Defines :class:`HybridScorer`, whose :meth:`HybridScorer.fit_score` computes the
normalized weighted sum of the five dimension scores divided by the sum of the
weights. Because every dimension is in ``[0, 1]``, the weights are non-negative,
and at least one weight is positive (enforced by :meth:`ScoringConfig.validate`),
the result is a convex combination guaranteed to lie in ``[0, 1]``. The
computation is pure and deterministic: identical inputs always yield identical
output.

Requirements: 3.1, 3.4, 3.5, 3.6.
"""

from __future__ import annotations

from ranking.config import ScoringConfig
from ranking.models import DimensionScores


class HybridScorer:
    """Combine normalized dimension scores into a single ``Fit_Score``.

    The scorer holds an immutable :class:`ScoringConfig` supplying the five
    dimension weights. :meth:`fit_score` blends the dimensions into a value in
    ``[0, 1]`` using the normalized weighted-sum formula from the design.
    """

    def __init__(self, config: ScoringConfig) -> None:
        """Store the scoring configuration.

        Args:
            config: The validated weights controlling the dimension blend.
        """
        self._config = config

    @property
    def config(self) -> ScoringConfig:
        """The :class:`ScoringConfig` backing this scorer."""
        return self._config

    def fit_score(self, dims: DimensionScores) -> float:
        """Compute the normalized weighted-sum ``Fit_Score`` in ``[0, 1]``.

        The raw score is the weighted sum of the five dimension scores; it is
        divided by the sum of the weights to produce a convex combination::

            raw = w_semantic*semantic + w_skills_title*skills_title
                + w_experience*experience + w_trajectory*trajectory
                + w_education*education
            fit = raw / (w_semantic + w_skills_title + w_experience
                         + w_trajectory + w_education)

        Args:
            dims: The normalized ``[0, 1]`` per-dimension scores.

        Returns:
            The ``Fit_Score`` in ``[0, 1]``. The convex-combination result is
            clamped to ``[0, 1]`` to absorb floating-point rounding. Returns
            ``0.0`` defensively if the weight sum is zero
            (``ScoringConfig.validate`` already prevents an all-zero
            dimension-weight configuration).
        """
        cfg = self._config
        weights = cfg.dimension_weights
        weight_sum = (
            weights["w_semantic"]
            + weights["w_skills_title"]
            + weights["w_experience"]
            + weights["w_trajectory"]
            + weights["w_education"]
        )
        # Defensive guard: validate() forbids all-zero dimension weights, but
        # never divide by zero regardless of how the scorer was constructed.
        if weight_sum == 0:
            return 0.0

        raw = (
            weights["w_semantic"] * dims.semantic
            + weights["w_skills_title"] * dims.skills_title
            + weights["w_experience"] * dims.experience
            + weights["w_trajectory"] * dims.trajectory
            + weights["w_education"] * dims.education
        )
        fit = raw / weight_sum
        # Clamp defensively to absorb floating-point error; mathematically the
        # convex combination of [0,1] dimensions already lies in [0,1].
        if fit < 0.0:
            return 0.0
        if fit > 1.0:
            return 1.0
        return fit
