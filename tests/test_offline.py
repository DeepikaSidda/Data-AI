"""Offline-assurance integration test (task 15.5).

Proves the ranking pipeline performs **no network access** during ranking. The
test installs a hard network guard for the duration of the run by monkeypatching
``socket.socket`` and ``socket.create_connection`` to raise immediately, then
drives the full :class:`ranking.pipeline.RankingPipeline` over a small in-memory
candidate pool on the on-the-fly embedding path (``artifacts=None``).

Two layers of protection make the guarantee meaningful:

* The real :class:`ranking.embedding.EmbeddingModel` depends on
  ``sentence-transformers`` (which can attempt downloads). Its ``embed`` /
  ``embed_batch`` are patched with deterministic, content-derived fixed vectors
  (sha256-derived, L2-normalized, mirroring ``tests/test_pipeline_props.py``),
  and ``_ensure_model`` is patched to raise — so any accidental real model load
  fails loudly rather than silently reaching the network.
* Every socket creation raises ``RuntimeError("network access attempted")`` for
  the duration of the run, so any networked dependency the pipeline reached
  would surface as a failure rather than silently succeeding.

The test asserts the run completes with sockets blocked and writes a CSV with
``min(top_n, N)`` rows, i.e. no network dependency exists during ranking.

Validates Requirements: 10.5, 10.6, 11.1.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket

import numpy as np
import pytest

import ranking.embedding as _emb
from ranking.config import ScoringConfig
from ranking.pipeline import RankingPipeline

# ---------------------------------------------------------------------------
# Paths: committed job profile
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JOB_PROFILE_PATH = os.path.join(_REPO_ROOT, "job_profile.yaml")


# ---------------------------------------------------------------------------
# Deterministic, offline embedding stand-in (no sentence-transformers)
# ---------------------------------------------------------------------------
_EMB_DIM = 16


def _fake_embed(self, text: str) -> np.ndarray:
    """A deterministic, L2-normalized vector derived from a hash of ``text``."""
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    vec = np.frombuffer(digest[:_EMB_DIM], dtype=np.uint8).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec = np.ones(_EMB_DIM, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32)


def _fake_embed_batch(self, texts, batch_size: int = 64) -> np.ndarray:
    """Batch variant matching :meth:`EmbeddingModel.embed_batch` shape."""
    if not texts:
        return np.zeros((0, _EMB_DIM), dtype=np.float32)
    return np.vstack([_fake_embed(self, t) for t in texts]).astype(np.float32)


def _no_model(self):
    """Guard: the real model loader must never run in this offline test."""
    raise AssertionError(
        "EmbeddingModel._ensure_model was called; sentence-transformers must "
        "never load in the offline-assurance test."
    )


# ---------------------------------------------------------------------------
# Candidate factory (schema-complete enough for from_json + the pipeline)
# ---------------------------------------------------------------------------
def _redrob_signals() -> dict:
    """A complete, plausible redrob_signals block."""
    return {
        "profile_completeness_score": 0.9,
        "signup_date": "2019-01-01",
        "last_active_date": "2023-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 40,
        "applications_submitted_30d": 5,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 6.0,
        "connection_count": 500,
        "endorsements_received": 60,
        "notice_period_days": 15,
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "search_appearance_30d": 30,
        "saved_by_recruiters_30d": 8,
        "interview_completion_rate": 0.9,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "github_activity_score": 70.0,
        "offer_acceptance_rate": 0.7,
        "skill_assessment_scores": {"python": 0.9},
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
    }


def _make_candidate(candidate_id: str, yoe: int = 5) -> dict:
    """Build an internally-consistent (non-honeypot) candidate dict."""
    start_year = 2023 - yoe
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Candidate",
            "headline": "ML Engineer shipping production ranking systems",
            "summary": "Built embeddings-based retrieval and ranking deployed to real users.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": float(yoe),
            "current_title": "Machine Learning Engineer",
            "current_company": "Flipkart",
            "current_company_size": "1000-5000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "Flipkart",
                "title": "Machine Learning Engineer",
                "start_date": f"{start_year}-01-01",
                "end_date": "2023-01-01",
                "duration_months": yoe * 12,
                "is_current": False,
                "industry": "Technology",
                "company_size": "1000-5000",
                "description": "Built embeddings-based retrieval and ranking.",
            }
        ],
        "education": [
            {
                "institution": "IIT",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": start_year - 4,
                "end_year": start_year,
                "grade": "8.5",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {
                "name": "python",
                "proficiency": "advanced",
                "endorsements": 9,
                "duration_months": 48,
            },
            {
                "name": "ranking",
                "proficiency": "intermediate",
                "endorsements": 6,
                "duration_months": 30,
            },
        ],
        "redrob_signals": _redrob_signals(),
        "certifications": [],
        "languages": [],
    }


def _write_jsonl(records: list[dict], path: str) -> None:
    """Write ``records`` as one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")


def _count_csv_rows(path: str) -> int:
    """Return the number of data rows (excluding the header) in the CSV."""
    import csv

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return sum(1 for row in reader if any(cell.strip() for cell in row))


# ---------------------------------------------------------------------------
# Network guard
# ---------------------------------------------------------------------------
def _blocked_socket(*args, **kwargs):
    """Stand-in that refuses any socket creation."""
    raise RuntimeError("network access attempted")


@pytest.fixture
def _block_network(monkeypatch):
    """Make any attempt to open a network socket raise immediately."""
    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_socket)


@pytest.fixture
def _patch_embedding(monkeypatch):
    """Patch the embedding model so no real (networked) model can load."""
    monkeypatch.setattr(_emb.EmbeddingModel, "embed", _fake_embed)
    monkeypatch.setattr(_emb.EmbeddingModel, "embed_batch", _fake_embed_batch)
    monkeypatch.setattr(_emb.EmbeddingModel, "_ensure_model", _no_model)


# ---------------------------------------------------------------------------
# Task 15.5: offline-assurance integration test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_candidates,top_n", [(5, 100), (20, 100), (20, 10)])
def test_pipeline_runs_with_network_blocked(
    n_candidates, top_n, tmp_path, _block_network, _patch_embedding
):
    """The full ranking pipeline completes and writes the expected CSV while
    every network socket is blocked, proving ranking has no network dependency.

    Validates Requirements: 10.5, 10.6, 11.1.
    """
    # Sanity: the guard is actually armed for the duration of this test.
    with pytest.raises(RuntimeError, match="network access attempted"):
        socket.socket()

    records = [_make_candidate(f"CAND_{i:07d}") for i in range(n_candidates)]
    candidates_path = str(tmp_path / "candidates.jsonl")
    out_path = str(tmp_path / "submission.csv")
    _write_jsonl(records, candidates_path)

    pipeline = RankingPipeline(ScoringConfig.load(None))

    # If anything in the pipeline reached the network, this would raise
    # RuntimeError("network access attempted") and fail the test.
    report = pipeline.run(
        candidates_path,
        out_path,
        artifacts=None,
        top_n=top_n,
        job_profile_path=_JOB_PROFILE_PATH,
    )

    expected_rows = min(top_n, n_candidates)
    assert report.written_rows == expected_rows
    assert report.valid_count == n_candidates
    assert os.path.exists(out_path)
    assert _count_csv_rows(out_path) == expected_rows
