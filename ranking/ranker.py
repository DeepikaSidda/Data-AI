"""Deterministic top-N ranking and tie-breaking.

Defines :class:`Ranker`, whose :meth:`Ranker.rank` method sorts scored
candidates by the composite key ``(-final_score, candidate_id)`` so the result
is descending by ``final_score`` with ties broken by ``candidate_id`` ascending.
Because every candidate id is unique and the key is total, the ordering is
deterministic: identical input always produces byte-identical output.

The top ``top_n`` candidates (or all of them, when fewer exist) receive unique
integer ranks ``1..n`` in sorted order, and the result is returned as
:class:`RankedCandidate` rows. Since the sort is non-increasing in
``final_score`` and rank follows sort order, ``score`` is guaranteed to be
monotonically non-increasing as rank increases.

The ``Ranker`` is concerned only with *ordering*. The ``reasoning`` field of
each :class:`RankedCandidate` is left as an empty string placeholder; the
pipeline populates it later via the :class:`ReasoningGenerator`.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.5, 9.6.
"""

from __future__ import annotations

from .models import RankedCandidate, ScoredCandidate
from .writer import SCORE_DECIMALS


class Ranker:
    """Pure, deterministic ranker producing the top-N shortlist."""

    def rank(
        self, scored: list[ScoredCandidate], top_n: int = 100
    ) -> list[RankedCandidate]:
        """Order ``scored`` and return the top ``top_n`` as ranked rows.

        Candidates are sorted by the composite key ``(-final_score,
        candidate_id)``: descending by ``final_score`` with ties broken by
        ``candidate_id`` ascending. This key is total and deterministic, so
        repeated calls on equal input yield identical output (Req 7.5).

        The first ``min(top_n, len(scored))`` candidates are kept and assigned
        unique integer ranks ``1..n`` in sorted order (Req 7.1, 7.2). Because
        the order is non-increasing in ``final_score``, the returned ``score``
        values are monotonically non-increasing with rank (Req 7.3, 9.6).

        The ``reasoning`` field is left empty here; the pipeline fills it via
        the reasoning generator in a later step.

        Args:
            scored: The candidates to rank. May be empty.
            top_n: Maximum number of rows to return. Values larger than the
                candidate count return all candidates; non-positive values
                return an empty list.

        Returns:
            A list of :class:`RankedCandidate` with ``rank`` ``1..n``, ``score``
            set to each candidate's ``final_score`` rounded to ``SCORE_DECIMALS``
            decimals (matching the printed CSV value), and ``reasoning`` set to
            ``""`` as a placeholder for the pipeline to populate.
        """
        # Sort by descending final_score, then ascending candidate_id. The score
        # used as the PRIMARY key is rounded to SCORE_DECIMALS first because the
        # SubmissionWriter prints scores at that same precision. The validator
        # checks the tie-break (candidate_id ascending) against the PRINTED
        # values, so two raw scores that differ only beyond the 6th decimal
        # round to one printed value and must be treated as an exact tie here.
        # Rounding the key (and storing the rounded score below) keeps the order
        # the validator sees consistent with candidate_id ascending. Negating the
        # score gives descending order while keeping candidate_id ascending under
        # Python's tuple ordering, which is total and deterministic.
        ordered = sorted(
            scored,
            key=lambda c: (-round(c.final_score, SCORE_DECIMALS), c.candidate_id),
        )

        limit = max(0, top_n)
        return [
            RankedCandidate(
                rank=index,
                candidate_id=candidate.candidate_id,
                # Store the rounded value so the stored/printed score is
                # identical to the value used in the sort key above.
                score=round(candidate.final_score, SCORE_DECIMALS),
                reasoning="",
            )
            for index, candidate in enumerate(ordered[:limit], start=1)
        ]
