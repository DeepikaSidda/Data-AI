"""End-to-end pipeline property-based tests.

# Feature: ai-candidate-ranking, Property 11: No honeypot reaches the top 100
# Feature: ai-candidate-ranking, Property 21: The generated CSV passes the challenge validator

These tests drive the full :class:`ranking.pipeline.RankingPipeline` over the
on-the-fly embedding path (``artifacts=None``). The real
:class:`ranking.embedding.EmbeddingModel` depends on ``sentence-transformers``
(not installed in CI), so its ``embed`` / ``embed_batch`` are monkeypatched with
deterministic, content-derived fixed vectors and its lazy model loader is
patched to assert it is never reached. This keeps the tests fully offline and
fast while still exercising the genuine pipeline orchestration, ranking,
honeypot penalty, reasoning, and CSV writing.

Property 21 additionally imports the *committed challenge validator*
(``validate_submission.py``) and asserts the generated CSV passes it with zero
errors, guaranteeing conformance against the actual grader.

Validates Requirements: 6.4, 6.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ranking.embedding as _emb
from ranking.config import ScoringConfig
from ranking.pipeline import RankingPipeline

# ---------------------------------------------------------------------------
# Paths: committed job profile and the challenge validator
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JOB_PROFILE_PATH = os.path.join(_REPO_ROOT, "job_profile.yaml")

# The committed challenge validator lives alongside the repo under Downloads.
_DOWNLOADS = os.path.dirname(_REPO_ROOT)
_VALIDATOR_DIR = os.path.join(
    _DOWNLOADS,
    "[PUB] India_runs_data_and_ai_challenge",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
)
_VALIDATOR_PATH = os.path.join(_VALIDATOR_DIR, "validate_submission.py")


def _load_validator():
    """Import the committed ``validate_submission`` module from its file."""
    spec = importlib.util.spec_from_file_location(
        "challenge_validate_submission", _VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_submission


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
    """Guard: the real model loader must never run in these offline tests."""
    raise AssertionError(
        "EmbeddingModel._ensure_model was called; sentence-transformers must "
        "never load in offline pipeline tests."
    )


@pytest.fixture(scope="module", autouse=True)
def _patch_embedding():
    """Patch the embedding model's class methods for the whole module.

    Module-scoped (not function-scoped) so Hypothesis is happy and the patch is
    installed once before any example runs. Originals are restored afterwards.
    """
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
# Candidate factories (schema-complete enough for from_json + the pipeline)
# ---------------------------------------------------------------------------
# A handful of genuine "archetypes". Within an archetype the scoring-relevant
# content is byte-identical, so every replicated candidate gets an identical
# Final_Score and the only difference is candidate_id. That makes ties exact
# (the ranker breaks them by candidate_id ascending, exactly as the validator
# requires) and keeps distinct archetype scores well separated, so the 6-dp CSV
# rounding can never invert the tie-break ordering.

_GENUINE_ARCHETYPES = [
    {
        "yoe": 5,
        "title": "Machine Learning Engineer",
        "headline": "ML Engineer shipping production ranking systems",
        "summary": "Built embeddings-based retrieval and ranking deployed to real users.",
        "company": "Flipkart",
        "tier": "tier_1",
        "skills": [
            ("python", "advanced", 9, 48),
            ("ranking", "intermediate", 6, 30),
            ("retrieval", "intermediate", 5, 24),
        ],
    },
    {
        "yoe": 6,
        "title": "Search Engineer",
        "headline": "Search engineer with semantic search in production",
        "summary": "Operated vector search infrastructure and evaluation frameworks.",
        "company": "Swiggy",
        "tier": "tier_2",
        "skills": [
            ("python", "expert", 12, 60),
            ("semantic search", "advanced", 8, 36),
            ("ndcg", "intermediate", 4, 24),
        ],
    },
    {
        "yoe": 7,
        "title": "Applied Scientist",
        "headline": "Applied scientist for recommendation systems",
        "summary": "Shipped recommendation and learning-to-rank features end to end.",
        "company": "Myntra",
        "tier": "tier_1",
        "skills": [
            ("python", "advanced", 10, 54),
            ("recommendation", "advanced", 7, 42),
            ("pytorch", "intermediate", 5, 30),
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
            ("a/b testing", "intermediate", 6, 36),
        ],
    },
]

# Two distinct honeypot shapes (Property 10's contradictions): impossible
# experience-vs-span, and a >=2 expert/advanced-with-zero-duration cluster.
_HONEYPOT_KINDS = 2


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


def _make_genuine(candidate_id: str, archetype_index: int) -> dict:
    """Build an internally-consistent (non-honeypot) candidate dict."""
    a = _GENUINE_ARCHETYPES[archetype_index]
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
        "candidate_id": candidate_id,
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


def _make_honeypot(candidate_id: str, kind: int) -> dict:
    """Build an internally-inconsistent candidate dict (a honeypot).

    ``kind == 0``: years_of_experience hugely exceeds the career span.
    ``kind == 1``: two expert/advanced skills claimed with zero duration.
    """
    base = _make_genuine(candidate_id, 0)
    if kind == 0:
        # 3-year career span but 60 years of claimed experience.
        base["profile"]["years_of_experience"] = 60.0
        base["career_history"] = [
            {
                "company": "Flipkart",
                "title": "Machine Learning Engineer",
                "start_date": "2020-01-01",
                "end_date": "2023-01-01",
                "duration_months": 36,
                "is_current": False,
                "industry": "Technology",
                "company_size": "1000-5000",
                "description": "Short stint.",
            }
        ]
    else:
        # >=2 expert/advanced skills with duration_months == 0.
        base["skills"] = [
            {
                "name": "embeddings",
                "proficiency": "expert",
                "endorsements": 20,
                "duration_months": 0,
            },
            {
                "name": "ranking",
                "proficiency": "advanced",
                "endorsements": 15,
                "duration_months": 0,
            },
        ]
    return base


def _write_jsonl(records: list[dict], path: str) -> None:
    """Write ``records`` as one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")


def _read_csv_ids(path: str) -> list[str]:
    """Return the candidate_id column (data rows only) from the submission CSV."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [row[0] for row in reader if any(cell.strip() for cell in row)]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------
_ID_INT = st.integers(min_value=0, max_value=9_999_999)


@st.composite
def _genuine_pool(draw, min_n: int = 100, max_n: int = 120):
    """A pool of >=100 unique-id, non-honeypot candidates across archetypes."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    nums = draw(st.lists(_ID_INT, min_size=n, max_size=n, unique=True))
    records = []
    for num in nums:
        arch = draw(st.integers(min_value=0, max_value=len(_GENUINE_ARCHETYPES) - 1))
        records.append(_make_genuine(f"CAND_{num:07d}", arch))
    return records


@st.composite
def _mixed_pool(draw):
    """>=100 genuine candidates plus a handful of injected honeypots.

    Returns ``(records, honeypot_ids)`` where ids are all unique.
    """
    gn = draw(st.integers(min_value=100, max_value=112))
    hn = draw(st.integers(min_value=3, max_value=12))
    nums = draw(
        st.lists(_ID_INT, min_size=gn + hn, max_size=gn + hn, unique=True)
    )
    genuine_nums = nums[:gn]
    honeypot_nums = nums[gn:]

    records = []
    for num in genuine_nums:
        arch = draw(st.integers(min_value=0, max_value=len(_GENUINE_ARCHETYPES) - 1))
        records.append(_make_genuine(f"CAND_{num:07d}", arch))

    honeypot_ids = set()
    for num in honeypot_nums:
        hid = f"CAND_{num:07d}"
        honeypot_ids.add(hid)
        kind = draw(st.integers(min_value=0, max_value=_HONEYPOT_KINDS - 1))
        records.append(_make_honeypot(hid, kind))

    return records, honeypot_ids


# ---------------------------------------------------------------------------
# Property 21 (task 15.4): generated CSV passes the challenge validator
# ---------------------------------------------------------------------------
@settings(max_examples=30, deadline=None)
@given(records=_genuine_pool())
def test_generated_csv_passes_challenge_validator(records):
    """Running the pipeline over >=100 valid candidates yields a CSV the
    committed ``validate_submission.py`` accepts with zero errors.

    Validates Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6.
    """
    validate_submission = _load_validator()
    pipeline = RankingPipeline(ScoringConfig.load(None))

    with tempfile.TemporaryDirectory() as tmp:
        candidates_path = os.path.join(tmp, "candidates.jsonl")
        out_path = os.path.join(tmp, "submission.csv")
        _write_jsonl(records, candidates_path)

        report = pipeline.run(
            candidates_path,
            out_path,
            artifacts=None,
            top_n=100,
            job_profile_path=_JOB_PROFILE_PATH,
        )

        assert report.written_rows == 100
        ids = _read_csv_ids(out_path)
        assert len(ids) == 100

        errors = validate_submission(out_path)
        assert errors == [], f"validator reported issues: {errors}"


# ---------------------------------------------------------------------------
# Property 11 (task 15.3): no honeypot reaches the top 100
# ---------------------------------------------------------------------------
@settings(max_examples=30, deadline=None)
@given(pool=_mixed_pool())
def test_no_honeypot_reaches_top_100(pool):
    """With the default honeypot penalty (0.0), no injected honeypot id appears
    in the top-100 shortlist.

    Validates Requirements: 6.4, 6.5.
    """
    records, honeypot_ids = pool
    pipeline = RankingPipeline(ScoringConfig.load(None))

    with tempfile.TemporaryDirectory() as tmp:
        candidates_path = os.path.join(tmp, "candidates.jsonl")
        out_path = os.path.join(tmp, "submission.csv")
        _write_jsonl(records, candidates_path)

        report = pipeline.run(
            candidates_path,
            out_path,
            artifacts=None,
            top_n=100,
            job_profile_path=_JOB_PROFILE_PATH,
        )

        assert report.written_rows == 100
        top_ids = set(_read_csv_ids(out_path))
        leaked = top_ids & honeypot_ids
        assert not leaked, f"honeypot(s) reached the top 100: {sorted(leaked)}"
