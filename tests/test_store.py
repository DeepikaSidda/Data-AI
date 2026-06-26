"""Unit tests for the precomputed embedding store (``ranking.store``).

Covers construction from synthetic numpy arrays, the cosine -> [0,1] mapping
(identical -> 1.0, opposite -> 0.0, orthogonal -> 0.5), the neutral fallback for
unknown candidate ids, the on-the-fly ``similarity_for`` path, and the
:meth:`EmbeddingStore.load` round-trip plus its ``MissingArtifactError`` with
build hint when an artifact is absent.

Requirements: 3.2, 11.1, 11.5.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ranking.errors import MissingArtifactError
from ranking.store import BUILD_HINT, NEUTRAL_SIMILARITY, EmbeddingStore


def _store() -> EmbeddingStore:
    """A small synthetic store with a 2-D job embedding along the x-axis.

    Candidate rows: identical, orthogonal, and opposite to the job vector.
    """
    job = np.array([1.0, 0.0], dtype=np.float32)
    embeddings = np.array(
        [
            [1.0, 0.0],   # identical -> cos 1 -> 1.0
            [0.0, 1.0],   # orthogonal -> cos 0 -> 0.5
            [-1.0, 0.0],  # opposite -> cos -1 -> 0.0
        ],
        dtype=np.float32,
    )
    id_order = ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
    return EmbeddingStore(embeddings, id_order, job)


# ---------------------------------------------------------------------------
# Construction + similarity mapping
# ---------------------------------------------------------------------------


def test_similarity_values_in_unit_interval():
    store = _store()
    for cid in store.id_order:
        sim = store.similarity(cid)
        assert 0.0 <= sim <= 1.0


def test_identical_vector_maps_to_one():
    store = _store()
    assert store.similarity("CAND_0000001") == pytest.approx(1.0)


def test_orthogonal_vector_maps_to_half():
    store = _store()
    assert store.similarity("CAND_0000002") == pytest.approx(0.5)


def test_opposite_vector_maps_to_zero():
    store = _store()
    assert store.similarity("CAND_0000003") == pytest.approx(0.0)


def test_unknown_candidate_returns_neutral():
    store = _store()
    assert store.similarity("CAND_9999999") == pytest.approx(NEUTRAL_SIMILARITY)


def test_similarity_for_arbitrary_embedding():
    store = _store()
    assert store.similarity_for(np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert store.similarity_for(np.array([0.0, 1.0])) == pytest.approx(0.5)
    assert store.similarity_for(np.array([-1.0, 0.0])) == pytest.approx(0.0)


def test_similarity_handles_non_normalized_input():
    # A non-normalized but co-directional vector is still cosine 1 -> 1.0.
    store = _store()
    assert store.similarity_for(np.array([5.0, 0.0])) == pytest.approx(1.0)


def test_zero_vector_is_neutral():
    store = _store()
    assert store.similarity_for(np.array([0.0, 0.0])) == pytest.approx(0.5)


def test_index_lookup_built():
    store = _store()
    assert store._index["CAND_0000002"] == 1


# ---------------------------------------------------------------------------
# load() round-trip + missing-artifact errors
# ---------------------------------------------------------------------------


def _write_artifacts(tmp_path):
    emb_path = os.path.join(tmp_path, "embeddings.npy")
    id_path = os.path.join(tmp_path, "id_order.json")
    job_path = os.path.join(tmp_path, "job_embedding.npy")

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    id_order = ["CAND_0000001", "CAND_0000002"]
    job = np.array([1.0, 0.0], dtype=np.float32)

    np.save(emb_path, embeddings)
    with open(id_path, "w", encoding="utf-8") as fh:
        json.dump(id_order, fh)
    np.save(job_path, job)
    return emb_path, id_path, job_path


def test_load_round_trip(tmp_path):
    emb_path, id_path, job_path = _write_artifacts(str(tmp_path))
    store = EmbeddingStore.load(emb_path, id_path, job_path)

    assert store.id_order == ["CAND_0000001", "CAND_0000002"]
    assert store.similarity("CAND_0000001") == pytest.approx(1.0)
    assert store.similarity("CAND_0000002") == pytest.approx(0.5)


@pytest.mark.parametrize("missing", ["emb", "id", "job"])
def test_load_missing_artifact_raises_with_build_hint(tmp_path, missing):
    emb_path, id_path, job_path = _write_artifacts(str(tmp_path))
    paths = {"emb": emb_path, "id": id_path, "job": job_path}
    os.remove(paths[missing])

    with pytest.raises(MissingArtifactError) as exc:
        EmbeddingStore.load(emb_path, id_path, job_path)

    err = exc.value
    assert err.artifact is not None
    assert err.build_hint == BUILD_HINT
    assert "precompute_embeddings.py" in str(err)
