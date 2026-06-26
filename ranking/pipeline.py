"""End-to-end ranking orchestration.

Defines :class:`RankingPipeline`, which wires the streaming pipeline together:
load -> job profile + embedding store -> feature extraction -> honeypot ->
hybrid score -> behavioral modifier -> rank top N -> reasoning -> write CSV.

Two semantic-similarity paths are supported, matching the design's
precompute-vs-rank split:

* **Precomputed store path** (100K full run): when ``artifacts`` is supplied and
  the committed ``embeddings.npy`` / ``id_order.json`` / ``job_embedding.npy``
  load successfully, each candidate's semantic similarity is read from the
  :class:`~ranking.store.EmbeddingStore`.
* **On-the-fly embedding path** (small sample): when ``artifacts`` is ``None`` or
  the store cannot be loaded, the job profile text is embedded once with the
  lazy, offline :class:`~ranking.embedding.EmbeddingModel` and each candidate's
  text is embedded on demand, mapping cosine into ``[0, 1]`` via the store math.

The orchestration bounds in-memory state to one lightweight
:class:`~ranking.models.ScoredCandidate` per candidate (no per-candidate
embeddings are retained beyond what the store already holds), runs entirely on
CPU, and makes no network calls.

Requirements: 10.1, 10.3, 10.4, 10.6.
"""

from __future__ import annotations

import dataclasses
import datetime
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ranking.behavioral import BehavioralModifier
from ranking.config import ScoringConfig
from ranking.embedding import EmbeddingModel
from ranking.errors import MissingArtifactError
from ranking.features import FeatureExtractor
from ranking.honeypot import HoneypotDetector
from ranking.job_profile import JobProfile
from ranking.loader import CandidateLoader
from ranking.models import (
    ArtifactPaths,
    CandidateRecord,
    LoadResult,
    ScoredCandidate,
)
from ranking.ranker import Ranker
from ranking.reasoning import ReasoningGenerator
from ranking.scorer import HybridScorer
from ranking.store import EmbeddingStore
from ranking.writer import SubmissionWriter

# Number of records between progress callbacks (keeps callback overhead low on
# the 100K path while still reporting smoothly).
_PROGRESS_BATCH = 1000


@dataclass(frozen=True)
class RunReport:
    """Summary of a completed pipeline run.

    Attributes:
        total_candidates: Total input lines/records seen during ingestion
            (valid + skipped).
        valid_count: Number of records that parsed and validated successfully.
        skipped_count: Number of malformed/invalid records skipped during load.
        honeypot_count: Number of valid records flagged as honeypots across the
            whole pool.
        written_rows: Number of data rows written to the submission CSV
            (``min(top_n, valid_count)``).
        elapsed_s: Wall-clock seconds the run took end to end.
    """

    total_candidates: int
    valid_count: int
    skipped_count: int
    honeypot_count: int
    written_rows: int
    elapsed_s: float


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    """Parse an ISO date (or datetime) string safely; ``None`` on failure."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip().split("T", 1)[0]
    if not text:
        return None
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


class _SemanticResolver:
    """Resolve each candidate's semantic similarity in ``[0, 1]``.

    Encapsulates the two-path design: a precomputed :class:`EmbeddingStore` when
    available, falling back to lazy on-the-fly :class:`EmbeddingModel` embedding
    for candidates without a precomputed row (or when no store is loaded at
    all). The embedding model and the on-the-fly store are created lazily so the
    fully-precomputed 100K path never imports ``sentence-transformers``.
    """

    def __init__(self, store: Optional[EmbeddingStore], job: JobProfile) -> None:
        self._store = store
        self._job = job
        self._store_ids = set(store.id_order) if store is not None else set()
        self._model: Optional[EmbeddingModel] = None
        # On-the-fly store used only when no precomputed store is available; it
        # holds just the job embedding and reuses the store's cosine -> [0,1].
        self._otf_store: Optional[EmbeddingStore] = None

    @property
    def used_on_the_fly(self) -> bool:
        """Whether any on-the-fly embedding was actually performed."""
        return self._model is not None

    def _ensure_model(self) -> EmbeddingModel:
        """Lazily construct the offline embedding model on first use."""
        if self._model is None:
            self._model = EmbeddingModel()
        return self._model

    def _ensure_otf_store(self) -> EmbeddingStore:
        """Lazily build the job-only on-the-fly store (no precomputed store)."""
        if self._otf_store is None:
            model = self._ensure_model()
            job_emb = model.embed(self._job.profile_text)
            dim = int(job_emb.shape[0])
            self._otf_store = EmbeddingStore(
                embeddings=np.zeros((0, dim), dtype=np.float32),
                id_order=[],
                job_embedding=job_emb,
            )
        return self._otf_store

    def similarity(self, rec: CandidateRecord) -> float:
        """Return the candidate's job similarity mapped into ``[0, 1]``."""
        if self._store is not None:
            if rec.candidate_id in self._store_ids:
                return self._store.similarity(rec.candidate_id)
            # Precomputed store exists but this candidate is absent: embed on
            # the fly and reuse the real store's job embedding for the cosine.
            model = self._ensure_model()
            emb = model.embed(EmbeddingModel.candidate_text(rec))
            return self._store.similarity_for(emb)

        # No precomputed store: fully on-the-fly small-sample path.
        otf = self._ensure_otf_store()
        emb = self._ensure_model().embed(EmbeddingModel.candidate_text(rec))
        return otf.similarity_for(emb)


class RankingPipeline:
    """Streaming end-to-end ranking orchestration.

    Constructed with a validated :class:`ScoringConfig`; :meth:`run` performs a
    full pass over the candidate dataset and writes the submission CSV.
    """

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config

    def run(
        self,
        candidates_path: str,
        out_path: str,
        artifacts: Optional[ArtifactPaths] = None,
        top_n: int = 100,
        progress: Optional[Callable[[float], None]] = None,
        job_profile_path: str = "job_profile.yaml",
    ) -> RunReport:
        """Run the full ranking pipeline and write the submission CSV.

        Args:
            candidates_path: Path to the candidate dataset (``.jsonl``,
                ``.jsonl.gz`` or ``.json`` array).
            out_path: Destination path for the submission CSV.
            artifacts: Optional precomputed embedding artifact paths. When
                ``None`` (or when loading fails), the on-the-fly embedding path
                is used.
            top_n: Maximum number of ranked rows to write (default 100).
            progress: Optional callback invoked with a fraction in ``[0, 1]`` as
                scoring proceeds and once more (with ``1.0``) at completion.
            job_profile_path: Path to the committed offline job profile.

        Returns:
            A :class:`RunReport` summarizing the run.

        Raises:
            DatasetAccessError: If the candidate file cannot be read.
            NoValidCandidatesError: If no valid candidate remains after loading.
            OutputWriteError: If the submission CSV cannot be written.
        """
        start = time.perf_counter()

        # 1. Offline job profile (local file I/O only, no network).
        job = JobProfile.load(job_profile_path)

        # 2. Embedding similarity source: precomputed store if available,
        #    otherwise on-the-fly embedding (small-sample path).
        store = self._load_store(artifacts)
        resolver = _SemanticResolver(store, job)

        # 3. Ingest. Using ``load`` (not ``iter_records``) lets us report exact
        #    skip counts; 100K minimal records fit comfortably under the 16 GB
        #    budget. For a larger-than-memory pool, swap to ``iter_records``.
        load_result: LoadResult = CandidateLoader().load(candidates_path)
        records = load_result.records
        total = len(records)

        # Pool latest activity drives the behavioral recency signal.
        pool_latest_active = self._pool_latest_active(records)

        # 4. Per-component collaborators (pure, deterministic, no I/O).
        extractor = FeatureExtractor(job, self._config)
        detector = HoneypotDetector()
        scorer = HybridScorer(self._config)
        behavioral = BehavioralModifier(self._config, pool_latest_active)

        scored: list[ScoredCandidate] = []
        dims_by_id: dict[str, object] = {}
        honeypot_by_id: dict[str, object] = {}
        honeypot_count = 0

        for index, rec in enumerate(records, start=1):
            semantic = resolver.similarity(rec)
            dims = extractor.extract(rec, semantic)
            honeypot = detector.check(rec)
            if honeypot.is_honeypot:
                honeypot_count += 1
            fit = scorer.fit_score(dims)
            final = behavioral.final_score(fit, rec.redrob_signals, honeypot)

            scored.append(
                ScoredCandidate(
                    candidate_id=rec.candidate_id,
                    record=rec,
                    dims=dims,
                    fit_score=fit,
                    behavioral_modifier=behavioral.modifier(rec.redrob_signals),
                    final_score=final,
                    honeypot=honeypot,
                )
            )
            # Keep only the lightweight per-candidate data needed for reasoning
            # of the (at most top_n) shortlist; dims/honeypot are small.
            dims_by_id[rec.candidate_id] = dims
            honeypot_by_id[rec.candidate_id] = honeypot

            if progress is not None and index % _PROGRESS_BATCH == 0:
                progress(index / total)

        # 5. Rank top N (deterministic).
        ranked = Ranker().rank(scored, top_n=top_n)

        # 6. Reasoning for each shortlisted candidate, using the original record
        #    plus its precomputed dims/honeypot. RankedCandidate is frozen, so we
        #    rebuild each row with the reasoning filled in.
        record_by_id = {r.candidate_id: r for r in records}
        generator = ReasoningGenerator(
            job, pool_latest_active=pool_latest_active
        )
        filled = []
        for rc in ranked:
            rec = record_by_id[rc.candidate_id]
            reasoning = generator.generate(
                rec,
                dims_by_id[rc.candidate_id],
                honeypot_by_id[rc.candidate_id],
                rc.rank,
            )
            filled.append(dataclasses.replace(rc, reasoning=reasoning))

        # 7. Write the submission CSV.
        SubmissionWriter().write(filled, out_path)

        if progress is not None:
            progress(1.0)

        elapsed = time.perf_counter() - start
        return RunReport(
            total_candidates=load_result.total_lines,
            valid_count=load_result.valid_count,
            skipped_count=len(load_result.skipped),
            honeypot_count=honeypot_count,
            written_rows=len(filled),
            elapsed_s=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_store(
        artifacts: Optional[ArtifactPaths],
    ) -> Optional[EmbeddingStore]:
        """Load the precomputed store, or ``None`` to use on-the-fly embedding.

        Returns ``None`` when ``artifacts`` is not supplied or when the
        artifacts are missing (small-sample path); the design reserves a hard
        ``MissingArtifactError`` for the explicit 100K precomputed run, which is
        driven by the CLI rather than this fallback.
        """
        if artifacts is None:
            return None
        try:
            return EmbeddingStore.load(
                artifacts.embeddings,
                artifacts.id_order,
                artifacts.job_embedding,
            )
        except MissingArtifactError:
            return None

    @staticmethod
    def _pool_latest_active(
        records: list[CandidateRecord],
    ) -> datetime.date:
        """Most recent parseable ``last_active_date`` across the pool.

        Falls back to today's date when no record carries a parseable date, so
        the behavioral recency signal always has a reference point.
        """
        latest: Optional[datetime.date] = None
        for rec in records:
            d = _parse_date(rec.redrob_signals.last_active_date)
            if d is not None and (latest is None or d > latest):
                latest = d
        return latest if latest is not None else datetime.date.today()
