"""Submission CSV writer.

Defines :class:`SubmissionWriter`, the only output-side I/O component. It writes
a UTF-8 ``.csv`` with the exact header ``candidate_id,rank,score,reasoning`` and
one data row per :class:`~ranking.models.RankedCandidate`, in the order given
(the pipeline guarantees 100 rows in rank order with unique ranks 1..100 and
non-increasing scores). On permission/disk failure it raises
:class:`~ranking.errors.OutputWriteError` describing the reason.

The CSV is produced with the stdlib :mod:`csv` module using
``csv.QUOTE_MINIMAL`` so reasoning strings containing commas or quotes are
correctly escaped, and the file is opened with ``newline=""`` and
``encoding="utf-8"`` to avoid platform-specific line-ending corruption.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7.
"""

from __future__ import annotations

import csv

from ranking.errors import OutputWriteError
from ranking.models import RankedCandidate

# Exact header mandated by the challenge validator (column order matters).
HEADER = ["candidate_id", "rank", "score", "reasoning"]

# Single source of truth for score precision. The submission CSV prints scores
# with this many decimal places; the Ranker imports this same constant to round
# its sort key, guaranteeing the order the validator sees (based on the printed
# values) matches the candidate_id tie-break. Changing this here changes both.
SCORE_DECIMALS = 6

# Fixed score precision. Six decimals keep adjacent scores distinguishable
# while staying compact; float(".6f") round-trips for the validator's
# non-increasing check, and it never inverts the input ordering. Derived from
# SCORE_DECIMALS so the printed precision and the ranker's rounding stay in sync.
_SCORE_FORMAT = f"{{:.{SCORE_DECIMALS}f}}"


class SubmissionWriter:
    """Writes the final ranked shortlist to the submission CSV."""

    def write(self, ranked: list[RankedCandidate], out_path: str) -> None:
        """Write ``ranked`` to ``out_path`` as a UTF-8 submission CSV.

        The header row is exactly ``candidate_id,rank,score,reasoning`` followed
        by one row per candidate in the supplied order. Columns are written as
        candidate_id (str), rank (int), score (float, fixed 6-dp), reasoning
        (str, minimally quoted).

        Raises:
            OutputWriteError: if the file cannot be opened or written (for
                example permission denied or insufficient disk space).
        """
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(HEADER)
                for rc in ranked:
                    writer.writerow(
                        [
                            rc.candidate_id,
                            int(rc.rank),
                            _SCORE_FORMAT.format(float(rc.score)),
                            rc.reasoning,
                        ]
                    )
        except OSError as exc:
            raise OutputWriteError(
                f"Could not write submission CSV to {out_path!r}: {exc}"
            ) from exc
