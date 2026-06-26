"""Unit tests for :class:`ranking.writer.SubmissionWriter`.

Covers the submission CSV contract:
- exact header row ``candidate_id,rank,score,reasoning``;
- rows written in the given order with correct columns and CSV-quoting of
  reasoning containing commas / double-quotes (verified by round-trip);
- fixed-precision, float-parseable score that keeps a descending input
  non-increasing in the file;
- UTF-8 encoding (non-ASCII reasoning round-trips);
- :class:`OutputWriteError` on unwritable paths.

Validates: Requirements 9.1, 9.2, 9.7.
"""

from __future__ import annotations

import csv

import pytest

from ranking.errors import OutputWriteError
from ranking.models import RankedCandidate
from ranking.writer import HEADER, SubmissionWriter, _SCORE_FORMAT


def _read_rows(path) -> list[list[str]]:
    """Read every CSV row back as a list of string fields (UTF-8)."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_header_is_exact_first_row(tmp_path):
    """First line must be exactly ``candidate_id,rank,score,reasoning``."""
    out = tmp_path / "submission.csv"
    SubmissionWriter().write(
        [RankedCandidate(rank=1, candidate_id="CAND_0000001", score=0.9, reasoning="ok")],
        str(out),
    )

    # Raw first physical line, exact bytes (no quoting, trailing CRLF stripped).
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "candidate_id,rank,score,reasoning"

    rows = _read_rows(out)
    assert rows[0] == HEADER


def test_rows_written_in_given_order_with_correct_columns(tmp_path):
    """Each data row preserves order and maps fields to the right columns."""
    out = tmp_path / "submission.csv"
    ranked = [
        RankedCandidate(rank=1, candidate_id="CAND_0000003", score=0.80, reasoning="best"),
        RankedCandidate(rank=2, candidate_id="CAND_0000001", score=0.50, reasoning="mid"),
        RankedCandidate(rank=3, candidate_id="CAND_0000002", score=0.10, reasoning="last"),
    ]
    SubmissionWriter().write(ranked, str(out))

    rows = _read_rows(out)
    data = rows[1:]
    assert len(data) == len(ranked)
    for written, rc in zip(data, ranked):
        candidate_id, rank, score, reasoning = written
        assert candidate_id == rc.candidate_id
        assert int(rank) == rc.rank
        assert float(score) == pytest.approx(rc.score)
        assert reasoning == rc.reasoning

    # Order is exactly the input order (by candidate_id).
    assert [r[0] for r in data] == [rc.candidate_id for rc in ranked]


def test_reasoning_with_commas_and_quotes_round_trips(tmp_path):
    """Commas and double-quotes in reasoning must be CSV-quoted and recovered."""
    out = tmp_path / "submission.csv"
    tricky = 'Strong fit: Python, ML, and "leadership"; 7 yrs, product-led.'
    ranked = [
        RankedCandidate(rank=1, candidate_id="CAND_0000001", score=0.42, reasoning=tricky),
    ]
    SubmissionWriter().write(ranked, str(out))

    rows = _read_rows(out)
    assert rows[1][0] == "CAND_0000001"
    # The reasoning field must round-trip byte-for-byte through csv.reader.
    assert rows[1][3] == tricky


def test_score_fixed_precision_parseable_and_non_increasing(tmp_path):
    """Score uses fixed precision, parses as float, and stays non-increasing."""
    out = tmp_path / "submission.csv"
    scores = [0.987654321, 0.5, 0.5, 0.250000004, 0.0]
    ranked = [
        RankedCandidate(
            rank=i + 1,
            candidate_id=f"CAND_{i + 1:07d}",
            score=s,
            reasoning=f"r{i}",
        )
        for i, s in enumerate(scores)
    ]
    SubmissionWriter().write(ranked, str(out))

    rows = _read_rows(out)
    score_strings = [r[2] for r in rows[1:]]

    # Fixed precision: matches the writer's format spec exactly.
    expected = [_SCORE_FORMAT.format(s) for s in scores]
    assert score_strings == expected
    # All have the same number of decimal places (fixed precision).
    decimals = {len(s.split(".")[1]) for s in score_strings}
    assert decimals == {6}

    # Every score parses as a float.
    parsed = [float(s) for s in score_strings]
    # Descending input remains non-increasing in the file.
    assert all(parsed[i] >= parsed[i + 1] for i in range(len(parsed) - 1))


def test_file_is_utf8_non_ascii_round_trips(tmp_path):
    """Non-ASCII reasoning must be stored as UTF-8 and round-trip exactly."""
    out = tmp_path / "submission.csv"
    reasoning = "Fluent in हिन्दी; café-grade résumé, 日本語 ok ✓"
    ranked = [
        RankedCandidate(rank=1, candidate_id="CAND_0000001", score=0.7, reasoning=reasoning),
    ]
    SubmissionWriter().write(ranked, str(out))

    # Decodes cleanly as UTF-8 (would raise on a wrong codec).
    out.read_bytes().decode("utf-8")

    rows = _read_rows(out)
    assert rows[1][3] == reasoning


def test_write_to_nonexistent_directory_raises_output_write_error(tmp_path):
    """A path inside a missing directory cannot be opened -> OutputWriteError."""
    bad_path = tmp_path / "no_such_dir" / "submission.csv"
    ranked = [
        RankedCandidate(rank=1, candidate_id="CAND_0000001", score=0.5, reasoning="x"),
    ]
    with pytest.raises(OutputWriteError):
        SubmissionWriter().write(ranked, str(bad_path))


def test_write_to_directory_path_raises_output_write_error(tmp_path):
    """Opening a directory as a file fails -> OutputWriteError."""
    ranked = [
        RankedCandidate(rank=1, candidate_id="CAND_0000001", score=0.5, reasoning="x"),
    ]
    with pytest.raises(OutputWriteError):
        # tmp_path itself is a directory; cannot be opened for writing.
        SubmissionWriter().write(ranked, str(tmp_path))
