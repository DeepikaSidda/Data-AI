"""ranking — fully local, CPU-only, offline candidate ranking pipeline.

This package implements the AI Candidate Ranking system for the Redrob
Intelligent Candidate Discovery & Ranking Challenge. All intelligence runs
locally on CPU with no network access during the ranking step.

Module overview (most are stubs filled in by later tasks):
    models      — frozen dataclasses for candidate / scoring domain objects
    config      — ScoringConfig (weights, bounds, validation)
    loader      — streaming candidate ingestion (.jsonl / .gz / .json array)
    job_profile — offline-encoded Job_Profile
    embedding   — local sentence-transformers wrapper (CPU, offline)
    store       — precomputed embedding store + cosine -> [0,1]
    features    — structured feature dimension extraction
    honeypot    — internal-consistency / honeypot detection
    scorer      — hybrid weighted Fit_Score
    behavioral  — bounded behavioral modifier + Final_Score
    ranker      — deterministic top-100 ranking + tie-break
    reasoning   — offline grounded reasoning generation
    writer      — submission CSV writer
    pipeline    — end-to-end orchestration
    errors      — typed exception hierarchy (implemented)
"""

from ranking.errors import (
    RankingError,
    DatasetAccessError,
    NoValidCandidatesError,
    MissingArtifactError,
    WeightValidationError,
    OutputWriteError,
)

__all__ = [
    "RankingError",
    "DatasetAccessError",
    "NoValidCandidatesError",
    "MissingArtifactError",
    "WeightValidationError",
    "OutputWriteError",
]
