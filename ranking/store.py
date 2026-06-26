"""Precomputed embedding store and cosine -> [0,1] mapping.

Defines :class:`EmbeddingStore`, which loads the committed precompute artifacts
(``embeddings.npy``, ``id_order.json``, ``job_embedding.npy``) from local disk
and exposes the semantic-similarity dimension used by the hybrid scorer.

Loading raises :class:`MissingArtifactError` — naming the absent artifact and
the exact ``python precompute_embeddings.py ...`` build command — when any
artifact is missing (Requirement 11.5). Similarity is the cosine of two
(assumed L2-normalized) vectors mapped from ``[-1, 1]`` into ``[0, 1]`` via
``(cos + 1) / 2`` (Requirements 3.2, 11.1). The cosine is computed defensively
(dividing by vector norms) so non-normalized inputs are still mapped correctly.

Requirements: 3.2, 11.1, 11.5.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from ranking.errors import MissingArtifactError

# Neutral similarity returned for an unknown candidate id. 0.5 is the midpoint
# of the [0,1] range (it corresponds to cosine == 0, i.e. orthogonal vectors),
# so an unknown candidate neither helps nor hurts the semantic dimension and
# the small-sample on-the-fly path can override it via ``similarity_for``.
NEUTRAL_SIMILARITY = 0.5

# Exact command that regenerates the precomputed artifacts, surfaced in the
# MissingArtifactError build hint so the failure is actionable (Requirement 11.5).
BUILD_HINT = "python precompute_embeddings.py --candidates ./candidates.jsonl"


class EmbeddingStore:
    """In-memory view over the precomputed candidate / job embeddings.

    The constructor builds a ``candidate_id -> row index`` dictionary so a
    candidate's embedding can be retrieved in O(1) for the similarity lookup.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        id_order: List[str],
        job_embedding: np.ndarray,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.id_order = list(id_order)
        self.job_embedding = np.asarray(job_embedding, dtype=np.float32).reshape(-1)
        # O(1) candidate_id -> row index lookup.
        self._index: Dict[str, int] = {
            cid: i for i, cid in enumerate(self.id_order)
        }

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls, emb_path: str, id_path: str, job_path: str
    ) -> "EmbeddingStore":
        """Load the precomputed artifacts from local disk.

        Raises :class:`MissingArtifactError` (naming the artifact and including
        the build command) if any of the three files is absent (Requirement
        11.5). No network access occurs.
        """
        for label, path in (
            ("embeddings.npy", emb_path),
            ("id_order.json", id_path),
            ("job_embedding.npy", job_path),
        ):
            if not os.path.isfile(path):
                raise MissingArtifactError(
                    f"Required precomputed artifact '{label}' was not found at "
                    f"'{path}'.",
                    artifact=label,
                    build_hint=BUILD_HINT,
                )

        embeddings = np.load(emb_path)
        with open(id_path, "r", encoding="utf-8") as fh:
            id_order = json.load(fh)
        job_embedding = np.load(job_path)

        return cls(
            embeddings=embeddings,
            id_order=list(id_order),
            job_embedding=job_embedding,
        )

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------
    def similarity(self, candidate_id: str) -> float:
        """Return the job-similarity of a precomputed candidate in ``[0, 1]``.

        Cosine of the candidate's embedding against the job embedding, mapped
        via ``(cos + 1) / 2``. If ``candidate_id`` is not present in the store
        (e.g. the small-sample on-the-fly path), returns the neutral value
        ``0.5`` so the missing candidate neither helps nor hurts and the
        on-the-fly path can override via :meth:`similarity_for`.
        """
        idx = self._index.get(candidate_id)
        if idx is None:
            return NEUTRAL_SIMILARITY
        return self.similarity_for(self.embeddings[idx])

    def similarity_for(self, embedding: np.ndarray) -> float:
        """Return job-similarity for an arbitrary embedding in ``[0, 1]``.

        Used by the on-the-fly small-sample path where the candidate has no
        precomputed row. Maps cosine into ``[0, 1]`` via ``(cos + 1) / 2``.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        cos = self._cosine(vec, self.job_embedding)
        return (cos + 1.0) / 2.0

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in ``[-1, 1]``, defensive against non-normalized
        inputs and zero vectors.

        Inputs are assumed L2-normalized, but we divide by the norms anyway so
        a non-normalized vector still yields a correct cosine; a zero-norm
        vector yields ``0.0`` (orthogonal / neutral).
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cos = float(np.dot(a, b) / (norm_a * norm_b))
        # Guard against tiny floating-point excursions outside [-1, 1].
        return max(-1.0, min(1.0, cos))
