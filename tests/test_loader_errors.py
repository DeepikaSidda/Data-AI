"""Unit tests for candidate ingestion error paths.

Covers the failure behavior of :meth:`CandidateLoader.load`:

* unreadable / missing files raise :class:`DatasetAccessError` (Requirement 1.6),
* datasets with no valid record raise :class:`NoValidCandidatesError`
  (Requirement 1.7),
* a ``.json`` file that is not a JSON array raises :class:`DatasetAccessError`
  (per the loader's documented behavior).

A happy-path mixed-file case is included to anchor the success contract the
error cases deviate from (correct ``valid_count``, skip count, and 1-based skip
line numbers).

Validates Requirements: 1.6, 1.7.
"""

from __future__ import annotations

import json
import os

import pytest

from ranking.errors import DatasetAccessError, NoValidCandidatesError
from ranking.loader import CandidateLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write(path, text: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return str(path)


# ---------------------------------------------------------------------------
# DatasetAccessError (Requirement 1.6)
# ---------------------------------------------------------------------------
def test_missing_file_raises_dataset_access_error(tmp_path):
    """A non-existent path is surfaced as DatasetAccessError (1.6)."""
    missing = os.path.join(str(tmp_path), "does_not_exist.jsonl")
    assert not os.path.exists(missing)

    with pytest.raises(DatasetAccessError):
        CandidateLoader().load(missing)


def test_missing_gz_file_raises_dataset_access_error(tmp_path):
    """A non-existent .gz path is also surfaced as DatasetAccessError (1.6)."""
    missing = os.path.join(str(tmp_path), "does_not_exist.jsonl.gz")

    with pytest.raises(DatasetAccessError):
        CandidateLoader().load(missing)


def test_unreadable_directory_path_raises_dataset_access_error(tmp_path):
    """Pointing the loader at a directory is unreadable -> DatasetAccessError (1.6).

    Opening a directory as a text file raises an OSError subclass
    (IsADirectoryError on POSIX, PermissionError on Windows), both of which the
    loader wraps in DatasetAccessError. This is a cross-platform stand-in for a
    permission-denied / unreadable file.
    """
    directory = tmp_path / "a_directory.jsonl"
    directory.mkdir()

    with pytest.raises(DatasetAccessError):
        CandidateLoader().load(str(directory))


def test_json_file_that_is_not_an_array_raises_dataset_access_error(tmp_path):
    """A .json file holding an object (not an array) raises DatasetAccessError.

    Per loader behavior, the .json path requires a top-level JSON array; a
    JSON object is rejected with DatasetAccessError rather than being treated
    as a single record.
    """
    path = _write(
        tmp_path / "candidates.json",
        json.dumps({"candidate_id": "CAND_0000001", "profile": {}}),
    )

    with pytest.raises(DatasetAccessError):
        CandidateLoader().load(path)


def test_json_file_with_invalid_json_raises_dataset_access_error(tmp_path):
    """A .json file that is not valid JSON at all raises DatasetAccessError."""
    path = _write(tmp_path / "candidates.json", "{not valid json")

    with pytest.raises(DatasetAccessError):
        CandidateLoader().load(path)


# ---------------------------------------------------------------------------
# NoValidCandidatesError (Requirement 1.7)
# ---------------------------------------------------------------------------
def test_all_malformed_jsonl_raises_no_valid_candidates(tmp_path):
    """A file of only invalid JSON / bad ids yields no valid records (1.7)."""
    lines = [
        "{not valid json",                                   # unparseable
        json.dumps({"profile": {}}),                         # missing id
        json.dumps({"candidate_id": "BADID_001"}),           # wrong pattern
        json.dumps({"candidate_id": "CAND_123"}),            # too few digits
        json.dumps({"candidate_id": 1234567}),               # non-string id
    ]
    path = _write(tmp_path / "candidates.jsonl", "\n".join(lines) + "\n")

    with pytest.raises(NoValidCandidatesError):
        CandidateLoader().load(path)


def test_empty_file_raises_no_valid_candidates(tmp_path):
    """A zero-line file has no valid records and raises (1.7)."""
    path = _write(tmp_path / "candidates.jsonl", "")

    with pytest.raises(NoValidCandidatesError):
        CandidateLoader().load(path)


def test_blank_lines_only_raises_no_valid_candidates(tmp_path):
    """A file of only blank lines is treated as empty and raises (1.7)."""
    path = _write(tmp_path / "candidates.jsonl", "\n\n   \n\t\n")

    with pytest.raises(NoValidCandidatesError):
        CandidateLoader().load(path)


# ---------------------------------------------------------------------------
# Happy-path anchor: mixed valid + malformed file
# ---------------------------------------------------------------------------
def test_mixed_file_reports_counts_and_one_based_skip_lines(tmp_path):
    """A mixed file returns correct valid/skip counts and 1-based skip lines.

    Layout (1-based line numbers):
      1: valid
      2: malformed JSON
      3: valid
      4: missing candidate_id
      5: valid
    """
    lines = [
        json.dumps({"candidate_id": "CAND_0000001", "profile": {}}),  # 1 valid
        "{not valid json",                                            # 2 skip
        json.dumps({"candidate_id": "CAND_0000002", "profile": {}}),  # 3 valid
        json.dumps({"profile": {}}),                                  # 4 skip
        json.dumps({"candidate_id": "CAND_0000003", "profile": {}}),  # 5 valid
    ]
    path = _write(tmp_path / "candidates.jsonl", "\n".join(lines) + "\n")

    result = CandidateLoader().load(path)

    assert result.valid_count == 3
    assert len(result.skipped) == 2
    assert result.total_lines == 5
    assert result.valid_count + len(result.skipped) == result.total_lines

    # Skip warnings carry 1-based line numbers matching the malformed lines.
    skip_lines = sorted(w.line_number for w in result.skipped)
    assert skip_lines == [2, 4]
    for warning in result.skipped:
        assert isinstance(warning.reason, str) and warning.reason.strip()

    # Records preserve the valid ids in file order.
    assert [r.candidate_id for r in result.records] == [
        "CAND_0000001",
        "CAND_0000002",
        "CAND_0000003",
    ]
