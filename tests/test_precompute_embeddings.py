"""Unit tests for the offline precompute script (task 6.5).

The heavy local model is replaced with a deterministic stub that returns fixed
vectors, so these tests run without ``sentence-transformers`` installed and
without any network access. They confirm:

* importing ``precompute_embeddings`` works and ``--help`` parses;
* ``precompute`` writes ``embeddings.npy``, ``id_order.json`` (matching the
  candidate order from ``sample_candidates.json``), and ``job_embedding.npy``;
* the written matrix rows align with the id order and can be read back.

Requirements: 11.1, 11.2, 11.4.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import precompute_embeddings as pe
from ranking.job_profile import JobProfile
from ranking.loader import CandidateLoader

SAMPLE = (
    r"c:\Users\sidda\Downloads\[PUB] India_runs_data_and_ai_challenge"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\India_runs_data_and_ai_challenge\sample_candidates.json"
)


class _StubModel:
    """Deterministic stand-in for EmbeddingModel (no real model needed).

    Produces a fixed-dim vector per text derived from its length so rows are
    distinguishable and reproducible. Mirrors the real API surface used by the
    precompute script: ``candidate_text``, ``embed_batch``, ``embed``.
    """

    DIM = 4

    @staticmethod
    def candidate_text(rec) -> str:
        return EmbeddingModelText(rec)

    def embed_batch(self, texts, batch_size: int = 64) -> np.ndarray:
        rows = [self._vec(t) for t in texts]
        return np.asarray(rows, dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        n = float(len(text) % 17 + 1)
        v = np.array([n, n / 2.0, n / 3.0, 1.0], dtype=np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)


def EmbeddingModelText(rec) -> str:
    """Reuse the real (pure) candidate_text builder for fidelity."""
    from ranking.embedding import EmbeddingModel

    return EmbeddingModel.candidate_text(rec)


@pytest.fixture
def sample_ids() -> list[str]:
    loader = CandidateLoader()
    return [rec.candidate_id for rec in loader.iter_records(SAMPLE)]


def test_import_and_help_parse():
    parser = pe.build_parser()
    # --help should exit cleanly (SystemExit 0); ensure parser is well-formed.
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_present():
    args = pe.build_parser().parse_args([])
    assert args.model == "BAAI/bge-small-en-v1.5"
    assert args.cache_dir == "./models"
    assert args.batch_size == 64
    assert args.out_emb.endswith("embeddings.npy")
    assert args.out_ids.endswith("id_order.json")
    assert args.out_job.endswith("job_embedding.npy")


def test_precompute_writes_aligned_artifacts(tmp_path, sample_ids):
    out_emb = os.path.join(str(tmp_path), "nested", "embeddings.npy")
    out_ids = os.path.join(str(tmp_path), "nested", "id_order.json")
    out_job = os.path.join(str(tmp_path), "nested", "job_embedding.npy")

    job = JobProfile.load("job_profile.yaml")
    model = _StubModel()

    summary = pe.precompute(
        candidates_path=SAMPLE,
        model=model,
        out_emb=out_emb,
        out_ids=out_ids,
        out_job=out_job,
        job=job,
        batch_size=3,  # force multiple batches
        progress_every=1,
    )

    # Artifacts exist.
    assert os.path.exists(out_emb)
    assert os.path.exists(out_ids)
    assert os.path.exists(out_job)

    embeddings = np.load(out_emb)
    with open(out_ids, "r", encoding="utf-8") as fh:
        ids = json.load(fh)
    job_vec = np.load(out_job)

    # id order matches candidate order from sample_candidates.json.
    assert ids == sample_ids
    # Row count aligns with ids; dims consistent.
    assert embeddings.shape[0] == len(ids)
    assert embeddings.shape[1] == _StubModel.DIM
    assert embeddings.dtype == np.float32
    # Job embedding is a 1-D float32 vector of the same dim.
    assert job_vec.ndim == 1
    assert job_vec.shape[0] == _StubModel.DIM
    # Summary is accurate.
    assert summary["count"] == len(ids)
    assert summary["dims"] == _StubModel.DIM

    # Rows align with the stub's deterministic mapping (read-back integrity).
    loader = CandidateLoader()
    for i, rec in enumerate(loader.iter_records(SAMPLE)):
        expected = model._vec(model.candidate_text(rec))
        np.testing.assert_allclose(embeddings[i], expected, rtol=1e-6, atol=1e-6)
