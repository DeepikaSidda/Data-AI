"""Reproduce-command integration test (task 15.6).

Drives the committed ``rank.py`` CLI end to end via :func:`rank.main` over the
real challenge ``sample_candidates.json`` (50 records) and asserts a structurally
valid submission CSV is produced. This exercises the genuine argument parsing,
pipeline orchestration, on-the-fly embedding fallback, ranking, reasoning, and
CSV writing — the same path a grader runs with the documented Reproduce_Command:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

The real :class:`ranking.embedding.EmbeddingModel` depends on
``sentence-transformers`` (not installed in CI), so its ``embed`` /
``embed_batch`` are monkeypatched with deterministic, content-derived fixed
vectors and its lazy loader (``_ensure_model``) is patched to raise — guaranteeing
the test stays fully offline. This mirrors the pattern in
``tests/test_pipeline_props.py``.

An empty ``--artifacts-dir`` is passed so the pipeline falls back to on-the-fly
embedding (no precomputed artifacts are read).

Note: the committed ``validate_submission.py`` requires EXACTLY 100 rows, so it
will NOT pass on a 50-row sample. Full 100-row validator conformance is covered
by ``tests/test_pipeline_props.py`` (Property 21). Here we instead assert the
structural properties of the produced CSV.

Validates Requirements: 13.2, 13.7, 13.8.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re

import numpy as np
import pytest

import ranking.embedding as _emb

# ---------------------------------------------------------------------------
# Paths: repo root, committed job profile, and the real sample candidates file
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JOB_PROFILE_PATH = os.path.join(_REPO_ROOT, "job_profile.yaml")

# The sample dataset ships with the challenge bundle, alongside the repo under
# Downloads (sibling of the repo root).
_DOWNLOADS = os.path.dirname(_REPO_ROOT)
_SAMPLE_PATH = os.path.join(
    _DOWNLOADS,
    "[PUB] India_runs_data_and_ai_challenge",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
    "sample_candidates.json",
)

_CAND_ID_RE = re.compile(r"^CAND_[0-9]{7}$")
_EXPECTED_HEADER = ["candidate_id", "rank", "score", "reasoning"]

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
        "never load in the offline reproduce-command test."
    )


@pytest.fixture(autouse=True)
def _patch_embedding(monkeypatch):
    """Patch the embedding model's class methods so the CLI runs offline."""
    monkeypatch.setattr(_emb.EmbeddingModel, "embed", _fake_embed)
    monkeypatch.setattr(_emb.EmbeddingModel, "embed_batch", _fake_embed_batch)
    monkeypatch.setattr(_emb.EmbeddingModel, "_ensure_model", _no_model)


def _read_csv_rows(path: str) -> tuple[list[str], list[list[str]]]:
    """Return ``(header, data_rows)`` from the submission CSV (skips blank rows)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    return header, rows


def _load_sample_ids() -> set[str]:
    """Return the set of candidate ids present in the sample dataset."""
    with open(_SAMPLE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {rec["candidate_id"] for rec in records}


def test_reproduce_command_produces_valid_submission(tmp_path):
    """``rank.main`` runs end to end on the 50-record sample and writes a
    structurally valid submission CSV.

    Validates Requirements: 13.2, 13.7, 13.8.
    """
    if not os.path.isfile(_SAMPLE_PATH):
        pytest.skip(f"sample candidates file not present: {_SAMPLE_PATH}")

    import rank  # imported lazily so the embedding patch is already installed

    sample_ids = _load_sample_ids()
    expected_rows = min(100, len(sample_ids))  # sample has 50 candidates

    out_csv = tmp_path / "submission.csv"
    artifacts_dir = tmp_path / "artifacts"  # empty -> on-the-fly embedding fallback
    artifacts_dir.mkdir()

    exit_code = rank.main([
        "--candidates", _SAMPLE_PATH,
        "--out", str(out_csv),
        "--artifacts-dir", str(artifacts_dir),
        "--job-profile", _JOB_PROFILE_PATH,
    ])

    # The CLI must succeed end to end.
    assert exit_code == 0

    # The output CSV must exist with the exact mandated header.
    assert out_csv.is_file()
    header, rows = _read_csv_rows(str(out_csv))
    assert header == _EXPECTED_HEADER

    # min(100, 50) == 50 data rows for the sample.
    assert len(rows) == expected_rows

    # Structural well-formedness of the submission:
    #  - ranks are exactly 1..N, unique, in order
    #  - scores are non-increasing
    #  - candidate_ids match the schema pattern and exist in the sample
    ranks = [int(r[1]) for r in rows]
    assert ranks == list(range(1, expected_rows + 1))

    scores = [float(r[2]) for r in rows]
    assert all(a >= b for a, b in zip(scores, scores[1:])), "scores not non-increasing"

    ids = [r[0] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate candidate_ids in submission"
    for cid in ids:
        assert _CAND_ID_RE.match(cid), f"candidate_id {cid!r} violates ^CAND_[0-9]{{7}}$"
        assert cid in sample_ids, f"candidate_id {cid!r} not present in the sample"
