"""rank.py — CLI entry point for the AI Candidate Ranking pipeline.

Runs the fully local, CPU-only, offline ranking pipeline end to end and writes
the top-100 submission CSV. Supports the single Reproduce_Command:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

The heavy work (embedding, scoring, ranking) happens inside
:meth:`ranking.pipeline.RankingPipeline.run`; importing this module does not
import ``sentence-transformers`` or touch the network.

Requirements: 13.2.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from ranking.config import ScoringConfig
from ranking.errors import RankingError
from ranking.models import ArtifactPaths
from ranking.pipeline import RankingPipeline

# Default artifact filenames inside ``--artifacts-dir`` (see precompute step).
_EMBEDDINGS_NAME = "embeddings.npy"
_ID_ORDER_NAME = "id_order.json"
_JOB_EMBEDDING_NAME = "job_embedding.npy"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the ranking command."""
    parser = argparse.ArgumentParser(
        prog="rank.py",
        description=(
            "Rank candidates for the Senior AI Engineer role and write a "
            "top-100 submission CSV (fully local, CPU-only, offline)."
        ),
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to the candidate dataset (.jsonl, .jsonl.gz, or .json array).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the submission CSV.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a scoring config YAML/JSON file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="./artifacts",
        help=(
            "Directory containing precomputed embeddings.npy, id_order.json, "
            "and job_embedding.npy. If the files are absent, the pipeline "
            "falls back to on-the-fly embedding (default: ./artifacts)."
        ),
    )
    parser.add_argument(
        "--job-profile",
        default="./job_profile.yaml",
        help="Path to the committed offline job profile (default: ./job_profile.yaml).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Maximum number of ranked rows to write (default: 100).",
    )
    return parser


def _progress_to_stderr(fraction: float) -> None:
    """Render scoring progress as a percentage on a single stderr line."""
    pct = max(0.0, min(1.0, fraction)) * 100.0
    end = "\n" if fraction >= 1.0 else ""
    print(f"\rranking... {pct:5.1f}%", end=end, file=sys.stderr, flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments, run the pipeline, and surface typed errors.

    Returns ``0`` on success; a non-zero exit code when a typed
    :class:`~ranking.errors.RankingError` is raised, after printing a clear,
    actionable message to stderr.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = ScoringConfig.load(args.config)
    except RankingError as err:
        print(f"rank.py: invalid scoring config: {err}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as err:
        print(f"rank.py: could not load scoring config: {err}", file=sys.stderr)
        return 2

    artifacts = ArtifactPaths(
        embeddings=os.path.join(args.artifacts_dir, _EMBEDDINGS_NAME),
        id_order=os.path.join(args.artifacts_dir, _ID_ORDER_NAME),
        job_embedding=os.path.join(args.artifacts_dir, _JOB_EMBEDDING_NAME),
    )

    try:
        report = RankingPipeline(config).run(
            args.candidates,
            args.out,
            artifacts=artifacts,
            top_n=args.top_n,
            progress=_progress_to_stderr,
            job_profile_path=args.job_profile,
        )
    except RankingError as err:
        print(f"rank.py: {err}", file=sys.stderr)
        return 1

    print(
        f"rank.py: wrote {report.written_rows} rows to {args.out} "
        f"({report.valid_count} valid / {report.skipped_count} skipped, "
        f"{report.honeypot_count} honeypots) in {report.elapsed_s:.2f}s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
