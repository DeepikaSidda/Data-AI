"""Typed exception hierarchy for the ranking pipeline.

All errors raised by the pipeline derive from :class:`RankingError` so callers
(notably ``rank.py``) can catch the whole family and surface a single,
actionable message. Each error carries a human-readable message; some carry
extra structured context (for example :class:`MissingArtifactError` carries the
artifact name and a build hint).

Requirements traceability: 13.1, 13.3 (deliverables surface clear,
actionable errors), and the per-component error requirements
(1.6, 1.7, 9.7, 11.5, 12.4, 12.5).
"""

from __future__ import annotations

from typing import Optional


class RankingError(Exception):
    """Base class for all errors raised by the ranking pipeline.

    Catching ``RankingError`` catches every pipeline-specific failure while
    letting unexpected programming errors propagate.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DatasetAccessError(RankingError):
    """The candidate dataset file could not be read.

    Raised when the input path is missing, permission-denied, or otherwise
    unreadable (Requirement 1.6).
    """


class NoValidCandidatesError(RankingError):
    """Zero valid candidate records remained after ingestion.

    Raised when every record was skipped as malformed or the dataset was empty
    (Requirement 1.7).
    """


class MissingArtifactError(RankingError):
    """An expected precomputed artifact was absent at ranking time.

    Carries the missing ``artifact`` name and a ``build_hint`` describing how to
    regenerate it (for example the exact ``python precompute_embeddings.py ...``
    command) so the message is actionable (Requirement 11.5).
    """

    def __init__(
        self,
        message: str,
        artifact: Optional[str] = None,
        build_hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.artifact = artifact
        self.build_hint = build_hint

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        parts = [self.message]
        if self.artifact:
            parts.append(f"missing artifact: {self.artifact}")
        if self.build_hint:
            parts.append(f"build it with: {self.build_hint}")
        return " | ".join(parts)


class WeightValidationError(RankingError):
    """A ScoringConfig contained invalid weights.

    Raised when any weight is negative (Requirement 12.4) or when all dimension
    weights are zero (Requirement 12.5).
    """


class OutputWriteError(RankingError):
    """The submission CSV could not be written.

    Raised on permission denial, insufficient disk space, or other write
    failures (Requirement 9.7).
    """
