"""Performance / smoke test for the ranking pipeline at scale.

This test exercises the end-to-end :class:`ranking.pipeline.RankingPipeline`
over a *synthetic* candidate pool to prove the ranking step scales and stays
within the compute budget described in the design.

Two sizes are supported:

* **Default (CI) path** — a reduced ``N = 2000`` synthetic pool. The test must
  run quickly and fully offline, so the real embedding model is patched with
  deterministic fixed vectors (the same pattern used in
  ``tests/test_pipeline_props.py``). This isolates *our* compute: the design's
  note that performance tests mock Bedrock/embeddings means we are measuring the
  pipeline's own work (ingestion, feature extraction, honeypot checks, scoring,
  ranking, reasoning, CSV writing), not model inference. Target: a few seconds.
* **Full 100K path** — gated behind the ``RUN_FULL_PERF=1`` environment variable
  so it never slows down ordinary CI runs. When enabled, ``N = 100000`` and the
  wall-clock bound is the design's 5-minute budget; peak RSS is sampled when a
  memory probe is available.

Both paths assert the pipeline writes exactly 100 ranked rows and completes
within a generous time bound, and print the elapsed time for visibility.

Validates Requirements: 1.4, 10.1, 10.2, 10.3, 10.4.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pytest

import ranking.embedding as _emb
from ranking.config import ScoringConfig
from ranking.pipeline import RankingPipeline

# ---------------------------------------------------------------------------
# Paths and run-size configuration
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JOB_PROFILE_PATH = os.path.join(_REPO_ROOT, "job_profile.yaml")

# Full 100K run is opt-in; default to a fast reduced pool for CI.
_RUN_FULL = os.environ.get("RUN_FULL_PERF") == "1"
_N = 100_000 if _RUN_FULL else 2_000

# Generous wall-clock bounds. The default reduced pool should finish in a few
# seconds; the bound is loose so the assertion is about "scales reasonably",
# not micro-benchmarking. The full run uses the design's 5-minute budget.
_TIME_BUDGET_S = 300.0 if _RUN_FULL else 60.0

# Design compute budget: peak RSS < 16 GB on CPU (only checked on the full run,
# and only when a memory probe is available).
_RSS_BUDGET_BYTES = 16 * 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Deterministic, offline embedding stand-in (no sentence-transformers).
# Mirrors tests/test_pipeline_props.py so embedding is instant and offline.
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
        "never load in the offline performance test."
    )


@pytest.fixture(autouse=True)
def _patch_embedding():
    """Patch the embedding model's methods so embedding is instant and offline."""
    original = (
        _emb.EmbeddingModel.embed,
        _emb.EmbeddingModel.embed_batch,
        _emb.EmbeddingModel._ensure_model,
    )
    _emb.EmbeddingModel.embed = _fake_embed
    _emb.EmbeddingModel.embed_batch = _fake_embed_batch
    _emb.EmbeddingModel._ensure_model = _no_model
    try:
        yield
    finally:
        (
            _emb.EmbeddingModel.embed,
            _emb.EmbeddingModel.embed_batch,
            _emb.EmbeddingModel._ensure_model,
        ) = original


# ---------------------------------------------------------------------------
# Synthetic candidate factory (minimal-but-schema-complete)
# ---------------------------------------------------------------------------
# A few genuine archetypes so the synthetic pool has varied, non-degenerate
# scores while staying cheap to generate.
_ARCHETYPES = [
    {
        "yoe": 5,
        "title": "Machine Learning Engineer",
        "headline": "ML Engineer shipping production ranking systems",
        "summary": "Built embeddings-based retrieval and ranking deployed to users.",
        "company": "Flipkart",
        "tier": "tier_1",
        "skills": [
            ("python", "advanced", 9, 48),
            ("ranking", "intermediate", 6, 30),
        ],
    },
    {
        "yoe": 6,
        "title": "Search Engineer",
        "headline": "Search engineer with semantic search in production",
        "summary": "Operated vector search infrastructure and evaluation.",
        "company": "Swiggy",
        "tier": "tier_2",
        "skills": [
            ("python", "expert", 12, 60),
            ("semantic search", "advanced", 8, 36),
        ],
    },
    {
        "yoe": 7,
        "title": "Applied Scientist",
        "headline": "Applied scientist for recommendation systems",
        "summary": "Shipped recommendation and learning-to-rank features.",
        "company": "Myntra",
        "tier": "tier_1",
        "skills": [
            ("python", "advanced", 10, 54),
            ("recommendation", "advanced", 7, 42),
        ],
    },
    {
        "yoe": 8,
        "title": "AI Engineer",
        "headline": "AI engineer with hybrid search experience",
        "summary": "Built hybrid search and a/b testing for ranking quality.",
        "company": "Razorpay",
        "tier": "tier_3",
        "skills": [
            ("python", "expert", 14, 72),
            ("hybrid search", "advanced", 9, 48),
        ],
    },
]


def _redrob_signals() -> dict:
    """A complete, plausible redrob_signals block (same for every record)."""
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


def _make_candidate(index: int) -> dict:
    """Build one minimal-but-schema-complete candidate dict with a unique id."""
    a = _ARCHETYPES[index % len(_ARCHETYPES)]
    yoe = a["yoe"]
    start_year = 2023 - yoe
    skills = [
        {
            "name": name,
            "proficiency": prof,
            "endorsements": endo,
            "duration_months": dur,
        }
        for (name, prof, endo, dur) in a["skills"]
    ]
    return {
        "candidate_id": f"CAND_{index:07d}",
        "profile": {
            "anonymized_name": "Candidate",
            "headline": a["headline"],
            "summary": a["summary"],
            "location": "Pune",
            "country": "India",
            "years_of_experience": float(yoe),
            "current_title": a["title"],
            "current_company": a["company"],
            "current_company_size": "1000-5000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": a["company"],
                "title": a["title"],
                "start_date": f"{start_year}-01-01",
                "end_date": "2023-01-01",
                "duration_months": yoe * 12,
                "is_current": False,
                "industry": "Technology",
                "company_size": "1000-5000",
                "description": a["summary"],
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
                "tier": a["tier"],
            }
        ],
        "skills": skills,
        "redrob_signals": _redrob_signals(),
        "certifications": [],
        "languages": [],
    }


def _write_synthetic_jsonl(path: str, n: int) -> None:
    """Stream ``n`` synthetic candidate records to ``path`` as JSON Lines."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps(_make_candidate(i)))
            f.write("\n")


def _peak_rss_bytes() -> int | None:
    """Best-effort current process peak RSS in bytes, or ``None`` if unknown.

    Tries ``resource`` (Unix) then ``psutil`` (cross-platform). On platforms
    where neither is available (e.g. Windows without psutil) this returns
    ``None`` and the caller skips the memory assertion.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is kilobytes on Linux, bytes on macOS.
        import sys

        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The performance / smoke test
# ---------------------------------------------------------------------------
def test_ranking_scales_and_writes_top_100(tmp_path):
    """The ranking step scales to a large pool and writes exactly 100 rows.

    Default (CI): ``N = 2000``, must finish well under a minute.
    Full run (``RUN_FULL_PERF=1``): ``N = 100000``, must finish under 5 minutes
    and (when a memory probe is available) keep peak RSS under 16 GB.

    Validates Requirements: 1.4, 10.1, 10.2, 10.3, 10.4.
    """
    candidates_path = os.path.join(str(tmp_path), "candidates.jsonl")
    out_path = os.path.join(str(tmp_path), "submission.csv")

    _write_synthetic_jsonl(candidates_path, _N)

    pipeline = RankingPipeline(ScoringConfig.load(None))

    start = time.perf_counter()
    report = pipeline.run(
        candidates_path,
        out_path,
        artifacts=None,
        top_n=100,
        job_profile_path=_JOB_PROFILE_PATH,
    )
    elapsed = time.perf_counter() - start

    mode = "FULL 100K" if _RUN_FULL else "CI reduced"
    print(
        f"\n[perf] mode={mode} N={_N} elapsed={elapsed:.2f}s "
        f"(budget {_TIME_BUDGET_S:.0f}s) written_rows={report.written_rows}"
    )

    # Requirement 1.4 / 10.x: a 100K-class pool produces exactly the top 100.
    assert report.written_rows == 100, report
    assert report.valid_count == _N

    with open(out_path, "r", encoding="utf-8", newline="") as f:
        data_rows = [line for line in f.read().splitlines()[1:] if line.strip()]
    assert len(data_rows) == 100

    # Requirement 10.1: completes within the wall-clock compute budget.
    assert elapsed < _TIME_BUDGET_S, (
        f"ranking took {elapsed:.1f}s, exceeding the {_TIME_BUDGET_S:.0f}s budget"
    )

    # Requirement 10.3/10.4: peak memory stays within the 16 GB CPU budget.
    # Only meaningful on the full run; skip cleanly when no probe is available.
    if _RUN_FULL:
        peak = _peak_rss_bytes()
        if peak is not None:
            print(f"[perf] peak RSS ~= {peak / (1024 ** 3):.2f} GB")
            assert peak < _RSS_BUDGET_BYTES, (
                f"peak RSS {peak / (1024 ** 3):.2f} GB exceeded the 16 GB budget"
            )
        # else: no memory probe on this platform; memory bound not asserted.
