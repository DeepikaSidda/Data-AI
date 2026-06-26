"""precompute_embeddings.py — offline, run-once embedding precompute.

Streams the candidate dataset, builds the canonical embedding text for each
record, embeds the texts in batches with the local CPU sentence-transformers
model, embeds the offline-encoded job profile text, and writes three committed
artifacts to local disk:

* ``embeddings.npy``     — float32 matrix, one L2-normalized row per candidate,
  in the same order as ``id_order.json``.
* ``id_order.json``      — JSON list of ``candidate_id`` strings, parallel to the
  rows of the embedding matrix.
* ``job_embedding.npy``  — float32 L2-normalized vector for the job profile.

This script is meant to be run **once, offline**. It uses only the local
``EmbeddingModel`` (which enforces ``HF_HUB_OFFLINE=1`` /
``local_files_only=True``) — there are no hosted services and no network calls.

The heavy work (loading the model, streaming + embedding) lives entirely in
:func:`main` / :func:`precompute`, so this module can be imported (and its CLI
inspected) without ``sentence-transformers`` installed.

Requirements: 11.1, 11.2, 11.4.

Example:
    python precompute_embeddings.py \
        --candidates ./candidates.jsonl \
        --out-emb ./artifacts/embeddings.npy \
        --out-ids ./artifacts/id_order.json \
        --out-job ./artifacts/job_embedding.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np

from ranking.embedding import EmbeddingModel
from ranking.job_profile import JobProfile
from ranking.loader import CandidateLoader

# How often (in candidates) to print a streaming progress line.
_PROGRESS_EVERY = 1000


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the precompute command."""
    parser = argparse.ArgumentParser(
        prog="precompute_embeddings.py",
        description=(
            "Offline, run-once embedding precompute for the ranking pipeline "
            "(local CPU model, no network)."
        ),
    )
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidate dataset (.jsonl / .jsonl.gz / .json array).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="Local sentence-transformers model name.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./models",
        help="Local model cache directory (no download occurs).",
    )
    parser.add_argument(
        "--out-emb",
        default="./artifacts/embeddings.npy",
        help="Output path for the candidate embedding matrix (.npy).",
    )
    parser.add_argument(
        "--out-ids",
        default="./artifacts/id_order.json",
        help="Output path for the candidate-id order file (.json).",
    )
    parser.add_argument(
        "--out-job",
        default="./artifacts/job_embedding.npy",
        help="Output path for the job profile embedding (.npy).",
    )
    parser.add_argument(
        "--job-profile",
        default="./job_profile.yaml",
        help="Path to the offline-encoded job_profile.yaml.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (CPU).",
    )
    return parser


def precompute(
    candidates_path: str,
    model: EmbeddingModel,
    out_emb: str,
    out_ids: str,
    out_job: str,
    job: JobProfile,
    batch_size: int = 64,
    progress_every: int = _PROGRESS_EVERY,
) -> dict:
    """Embed all candidates + the job profile and write the artifacts.

    Streams candidate records via :meth:`CandidateLoader.iter_records`, builds
    :meth:`EmbeddingModel.candidate_text` for each, and embeds them in batches
    of ``batch_size``. Per-batch embedding matrices are accumulated in a list
    and concatenated once at the end (bounded, predictable memory for the 100K
    path), then written with ``np.save``. The parallel ``candidate_id`` list is
    written to ``out_ids`` in row order, and the job embedding to ``out_job``.

    Returns a small summary dict (``count``, ``dims``, ``elapsed_s``).
    """
    start = time.perf_counter()
    loader = CandidateLoader()

    candidate_ids: List[str] = []
    batch_blocks: List[np.ndarray] = []
    text_buffer: List[str] = []
    id_buffer: List[str] = []
    embedded = 0
    dims = 0

    def _flush() -> None:
        nonlocal embedded, dims
        if not text_buffer:
            return
        mat = model.embed_batch(text_buffer, batch_size=batch_size)
        mat = np.asarray(mat, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        batch_blocks.append(mat)
        candidate_ids.extend(id_buffer)
        embedded += len(id_buffer)
        if dims == 0 and mat.shape[1] > 0:
            dims = int(mat.shape[1])
        text_buffer.clear()
        id_buffer.clear()
        print(f"  embedded {embedded} candidate(s)...", flush=True)

    print(f"Streaming candidates from {candidates_path!r}...", flush=True)
    for rec in loader.iter_records(candidates_path):
        text_buffer.append(model.candidate_text(rec))
        id_buffer.append(rec.candidate_id)
        if len(text_buffer) >= batch_size:
            _flush()
        elif embedded and progress_every and embedded % progress_every == 0:
            # progress already printed by _flush; nothing extra needed here
            pass
    _flush()

    if batch_blocks:
        embeddings = np.vstack(batch_blocks).astype(np.float32)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)
    if embeddings.size:
        dims = int(embeddings.shape[1])

    print("Embedding job profile text...", flush=True)
    job_embedding = np.asarray(model.embed(job.profile_text), dtype=np.float32)

    _ensure_parent_dir(out_emb)
    _ensure_parent_dir(out_ids)
    _ensure_parent_dir(out_job)

    np.save(out_emb, embeddings)
    with open(out_ids, "w", encoding="utf-8") as handle:
        json.dump(candidate_ids, handle)
    np.save(out_job, job_embedding)

    elapsed = time.perf_counter() - start
    summary = {
        "count": len(candidate_ids),
        "dims": dims,
        "elapsed_s": elapsed,
    }
    print(
        "Done. Wrote {count} embedding(s) of dim {dims} in {elapsed:.2f}s.".format(
            count=summary["count"], dims=summary["dims"], elapsed=elapsed
        ),
        flush=True,
    )
    print(f"  embeddings -> {out_emb}", flush=True)
    print(f"  id order   -> {out_ids}", flush=True)
    print(f"  job vector -> {out_job}", flush=True)
    return summary


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not yet exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the local model + job profile, and precompute."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Limit CPU thread oversubscription: sentence-transformers' encode plus
    # torch intra-op parallelism can contend on many-core machines, which
    # *lowers* throughput. Capping intra-op threads keeps embedding fast.
    try:
        import torch

        torch.set_num_threads(min(8, (os.cpu_count() or 8)))
    except Exception:  # pragma: no cover - torch always present at runtime
        pass

    model = EmbeddingModel(model_name=args.model, cache_dir=args.cache_dir)
    job = JobProfile.load(args.job_profile)

    precompute(
        candidates_path=args.candidates,
        model=model,
        out_emb=args.out_emb,
        out_ids=args.out_ids,
        out_job=args.out_job,
        job=job,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
