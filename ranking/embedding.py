"""Local sentence-transformers embedding wrapper (CPU, offline).

Defines :class:`EmbeddingModel`, a thin wrapper around a locally-cached
``sentence-transformers`` model (default ``BAAI/bge-small-en-v1.5``). The model
is loaded **lazily** on first use so that pure helpers (notably
:meth:`EmbeddingModel.candidate_text`) and the rest of the pipeline can be
imported and unit-tested without ``sentence-transformers`` (or torch) installed.

Offline guarantee: the constructor sets ``HF_HUB_OFFLINE=1`` and
``TRANSFORMERS_OFFLINE=1`` and the lazy loader passes ``local_files_only=True``,
so no model download can occur during ranking — all weights must already be in
the local cache (Requirements 2.5, 10.2, 10.5, 10.6).

Requirements: 2.5, 10.2, 10.5, 10.6.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List

import numpy as np

from ranking.models import CandidateRecord

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    pass


class EmbeddingModel:
    """Local, CPU-only, offline sentence-transformers embedding wrapper.

    The heavy ``sentence-transformers`` import and model instantiation are
    deferred until the first :meth:`embed` / :meth:`embed_batch` call so the
    class can be constructed (and :meth:`candidate_text` used) without the model
    dependency installed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str = "./models",
        device: str = "cpu",
        max_seq_length: int = 128,
    ) -> None:
        """Store configuration and enforce the offline environment.

        No model is loaded here; loading is deferred to first embed call.
        ``max_seq_length`` caps the tokens the model processes per text; the
        default 256 keeps CPU throughput high while still covering the
        role-relevant signal (headline, summary, titles, skills).
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.max_seq_length = max_seq_length
        self._model: Any = None

        # Enforce fully-offline behavior for any downstream HF / transformers
        # machinery so that no network download can ever be attempted.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        """Lazily import and instantiate the sentence-transformers model.

        The import is wrapped here so the module imports cleanly without the
        dependency. The offline guarantee is enforced by the ``HF_HUB_OFFLINE``
        / ``TRANSFORMERS_OFFLINE`` environment variables set in ``__init__``
        (the ``SentenceTransformer`` constructor across supported versions does
        not accept a ``local_files_only`` keyword), so no download can occur
        once the weights are cached locally.
        """
        if self._model is None:
            # Imported lazily on purpose; keeps the module importable without
            # sentence-transformers installed (tests rely on this).
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device,
            )
            # Cap the sequence length for CPU throughput. The role-relevant
            # signal (headline, summary, titles, skills) fits comfortably; long
            # career descriptions (often deliberately misleading in this
            # dataset) are truncated rather than dominating embedding cost.
            try:
                self._model.max_seq_length = self.max_seq_length
            except Exception:  # pragma: no cover - defensive, model-version safe
                pass
        return self._model

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalized float32 embedding for a single ``text``."""
        model = self._ensure_model()
        vec = model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        return self._l2_normalize(arr)

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Return an L2-normalized float32 matrix (one row per input text)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._ensure_model()
        mat = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(mat, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Re-normalize defensively (guards against any backend that skips it).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a 1-D vector, guarding divide-by-zero."""
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)

    # ------------------------------------------------------------------
    # Pure text construction (no model required)
    # ------------------------------------------------------------------
    @staticmethod
    def candidate_text(rec: CandidateRecord) -> str:
        """Build the canonical embedding text for a candidate.

        The high-signal fields are **front-loaded** — headline, current title,
        a capped summary snippet, past role titles, then skill names — so that
        when the model truncates to ``max_seq_length`` tokens the role-relevant
        content is preserved. The long, often-misleading ``career_history``
        descriptions are intentionally omitted from the embedding text (the
        structured :class:`~ranking.features.FeatureExtractor` still reads them
        for trajectory evidence). This is a **pure** function — it never touches
        the model (Requirement 2.5).
        """
        parts: List[str] = []

        profile = rec.profile
        if profile.headline:
            parts.append(profile.headline)
        if profile.current_title:
            parts.append(profile.current_title)

        # Decisive fields (past titles, skills) come BEFORE the summary so they
        # always fall inside the truncation window.
        for entry in rec.career_history:
            if entry.title:
                parts.append(entry.title)

        for skill in rec.skills:
            if skill.name:
                parts.append(skill.name)

        if profile.summary:
            # A capped summary snippet carries the role framing without crowding
            # out the title/skill signal above.
            parts.append(profile.summary[:200])

        # Collapse internal whitespace per part and join with single spaces.
        cleaned = [" ".join(part.split()) for part in parts if part and part.strip()]
        return " ".join(cleaned)
