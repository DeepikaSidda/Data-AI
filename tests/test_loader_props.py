"""Property-based tests for streaming candidate ingestion.

# Feature: ai-candidate-ranking, Property 1: Ingestion partitions every line into valid or skipped
# Feature: ai-candidate-ranking, Property 2: Ingestion is format-equivalent across jsonl, gzip, and json array

Validates Requirements: 1.1, 1.2, 1.3, 1.5.
"""

from __future__ import annotations

import gzip
import json
import os

from hypothesis import given, settings
from hypothesis import strategies as st

import pytest

from ranking.errors import NoValidCandidatesError
from ranking.loader import CandidateLoader


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
_DIGITS = st.text(alphabet="0123456789", min_size=7, max_size=7)


@st.composite
def _valid_candidate(draw) -> dict:
    """A minimal but valid candidate dict with a well-formed CAND_ id."""
    seven = draw(_DIGITS)
    return {
        "candidate_id": f"CAND_{seven}",
        "profile": {"years_of_experience": draw(st.floats(0, 40, allow_nan=False))},
    }


# A line is either a valid candidate dict (rendered as JSON) or a malformed
# line: bad JSON, or a JSON object missing/with-invalid candidate_id.
def _valid_line(draw) -> tuple[str, bool]:
    return json.dumps(draw(_valid_candidate())), True


_MALFORMED_LINES = st.sampled_from(
    [
        "{not valid json",
        "][",
        "null-ish nonsense",
        json.dumps({"profile": {}}),                       # missing candidate_id
        json.dumps({"candidate_id": "BADID_001"}),          # invalid pattern
        json.dumps({"candidate_id": "CAND_123"}),           # too few digits
        json.dumps({"candidate_id": 12345}),                # non-string id
        json.dumps([1, 2, 3]),                              # not an object
        json.dumps("a string"),                             # not an object
    ]
)


@st.composite
def _mixed_lines(draw) -> tuple[list[str], int, int]:
    """Generate a list of lines mixing valid and malformed; report expected counts."""
    n = draw(st.integers(min_value=1, max_value=40))
    lines: list[str] = []
    valid = 0
    skipped = 0
    for _ in range(n):
        if draw(st.booleans()):
            lines.append(json.dumps(draw(_valid_candidate())))
            valid += 1
        else:
            lines.append(draw(_MALFORMED_LINES))
            skipped += 1
    return lines, valid, skipped


# ---------------------------------------------------------------------------
# Property 1: Ingestion partitions every line into valid or skipped
# ---------------------------------------------------------------------------
@settings(max_examples=150, deadline=None)
@given(data=_mixed_lines())
def test_ingestion_partitions_every_line(data, tmp_path_factory):
    """valid_count + len(skipped) == total_lines, with structured skip warnings.

    Validates Requirements: 1.1, 1.5.
    """
    lines, expected_valid, expected_skipped = data
    path = os.path.join(str(tmp_path_factory.mktemp("part")), "candidates.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    # When no valid records exist, the loader correctly raises per Requirement
    # 1.7 rather than returning an (empty) partitioned result.
    if expected_valid == 0:
        with pytest.raises(NoValidCandidatesError):
            CandidateLoader().load(path)
        return

    result = CandidateLoader().load(path)

    # Every line is accounted for as exactly valid or skipped.
    assert result.valid_count + len(result.skipped) == result.total_lines
    assert result.total_lines == len(lines)
    assert result.valid_count == expected_valid
    assert len(result.skipped) == expected_skipped

    # Every skip warning carries a line number and a non-empty reason.
    for warning in result.skipped:
        assert isinstance(warning.line_number, int)
        assert 1 <= warning.line_number <= len(lines)
        assert isinstance(warning.reason, str)
        assert warning.reason.strip() != ""


# ---------------------------------------------------------------------------
# Property 2: Ingestion is format-equivalent across jsonl, gzip, and json array
# ---------------------------------------------------------------------------
@st.composite
def _unique_valid_records(draw) -> list[dict]:
    """A set of valid records with distinct candidate_ids (size <= 100)."""
    ids = draw(
        st.lists(_DIGITS, min_size=1, max_size=100, unique=True)
    )
    return [{"candidate_id": f"CAND_{seven}", "profile": {}} for seven in ids]


@settings(max_examples=100, deadline=None)
@given(records=_unique_valid_records())
def test_ingestion_format_equivalence(records, tmp_path_factory):
    """Loading the same records from .jsonl, .jsonl.gz, and .json yields the same ids.

    Validates Requirements: 1.1, 1.2, 1.3.
    """
    base = str(tmp_path_factory.mktemp("fmt"))
    jsonl_path = os.path.join(base, "candidates.jsonl")
    gz_path = os.path.join(base, "candidates.jsonl.gz")
    json_path = os.path.join(base, "candidates.json")

    lines = [json.dumps(rec) for rec in records]
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle)

    loader = CandidateLoader()
    ids_jsonl = {r.candidate_id for r in loader.load(jsonl_path).records}
    ids_gz = {r.candidate_id for r in loader.load(gz_path).records}
    ids_json = {r.candidate_id for r in loader.load(json_path).records}

    expected = {rec["candidate_id"] for rec in records}
    assert ids_jsonl == expected
    assert ids_gz == expected
    assert ids_json == expected
