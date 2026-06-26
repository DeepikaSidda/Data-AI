"""Streaming candidate ingestion.

``CandidateLoader`` streams candidate records from ``.jsonl``, ``.jsonl.gz``,
or a ``.json`` array. Format is detected by file extension:

* ``.gz``   -> opened with :mod:`gzip` in text mode and parsed as JSON Lines.
* ``.json`` -> the whole file is loaded as a single JSON array.
* anything else -> treated as JSON Lines (one JSON object per line).

Each record is validated for parseability and a ``candidate_id`` matching
``^CAND_[0-9]{7}$``. Malformed rows (invalid JSON, or missing/invalid id) are
skipped: :meth:`iter_records` skips them silently for the bounded-memory 100K
path, while :meth:`load` captures each skip as a :class:`SkipWarning` with a
1-based line number and a human-readable reason.

:meth:`load` raises :class:`DatasetAccessError` if the file cannot be opened or
read, and :class:`NoValidCandidatesError` if zero valid records remain.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7.
"""

from __future__ import annotations

import gzip
import json
import re
from typing import Any, Iterator

from ranking.errors import DatasetAccessError, NoValidCandidatesError
from ranking.models import CandidateRecord, LoadResult, SkipWarning


class CandidateLoader:
    """Stream-parse candidate datasets in ``.jsonl``, ``.jsonl.gz``, or ``.json``."""

    def __init__(self, id_pattern: str = r"^CAND_[0-9]{7}$") -> None:
        self._id_pattern = re.compile(id_pattern)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_obj(self, obj: Any) -> str | None:
        """Return ``None`` if ``obj`` is a valid candidate, else a reason string."""
        if not isinstance(obj, dict):
            return "record is not a JSON object"
        candidate_id = obj.get("candidate_id")
        if candidate_id is None:
            return "missing candidate_id"
        if not isinstance(candidate_id, str) or not self._id_pattern.match(candidate_id):
            return f"candidate_id does not match pattern: {candidate_id!r}"
        return None

    # ------------------------------------------------------------------
    # Lazy streaming iterator (bounded memory, used for the 100K path)
    # ------------------------------------------------------------------
    def iter_records(self, path: str) -> Iterator[CandidateRecord]:
        """Lazily yield valid :class:`CandidateRecord` objects from ``path``.

        Malformed lines/records are skipped silently. Memory stays bounded for
        the JSON Lines / gzip paths; the ``.json`` array path materializes the
        parsed array (an array file is inherently non-streaming).
        """
        for obj in self._iter_raw_objects(path):
            if obj is _MALFORMED:
                continue
            if self._validate_obj(obj) is not None:
                continue
            yield CandidateRecord.from_json(obj)

    # ------------------------------------------------------------------
    # Eager load with skip warnings + error handling
    # ------------------------------------------------------------------
    def load(self, path: str) -> LoadResult:
        """Load all candidates from ``path`` into a :class:`LoadResult`.

        Collects valid records and a :class:`SkipWarning` for every malformed
        row. Raises :class:`DatasetAccessError` when the file is unreadable and
        :class:`NoValidCandidatesError` when no valid record remains.
        """
        records: list[CandidateRecord] = []
        skipped: list[SkipWarning] = []
        total_lines = 0

        for line_number, obj in enumerate(self._iter_raw_objects(path), start=1):
            total_lines += 1
            if obj is _MALFORMED:
                skipped.append(SkipWarning(line_number, "malformed JSON"))
                continue
            reason = self._validate_obj(obj)
            if reason is not None:
                skipped.append(SkipWarning(line_number, reason))
                continue
            records.append(CandidateRecord.from_json(obj))

        if not records:
            raise NoValidCandidatesError(
                f"no valid candidate records found in {path!r} "
                f"({total_lines} line(s), {len(skipped)} skipped)"
            )

        return LoadResult(
            records=records,
            valid_count=len(records),
            skipped=skipped,
            total_lines=total_lines,
        )

    # ------------------------------------------------------------------
    # Raw object streaming (format detection + access-error wrapping)
    # ------------------------------------------------------------------
    def _iter_raw_objects(self, path: str) -> Iterator[Any]:
        """Yield parsed JSON objects (or ``_MALFORMED``) one per logical line.

        Wraps file-access failures in :class:`DatasetAccessError`. The actual
        parsing of each line is tolerant: a line that is not valid JSON yields
        the ``_MALFORMED`` sentinel rather than raising.
        """
        lower = path.lower()
        if lower.endswith(".gz"):
            yield from self._iter_jsonl_gz(path)
        elif lower.endswith(".json"):
            yield from self._iter_json_array(path)
        else:
            yield from self._iter_jsonl(path)

    def _iter_jsonl(self, path: str) -> Iterator[Any]:
        try:
            handle = open(path, "r", encoding="utf-8")
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise DatasetAccessError(f"cannot read candidate file {path!r}: {exc}") from exc
        with handle:
            yield from self._parse_lines(handle, path)

    def _iter_jsonl_gz(self, path: str) -> Iterator[Any]:
        try:
            handle = gzip.open(path, "rt", encoding="utf-8")
        except (FileNotFoundError, PermissionError, OSError, gzip.BadGzipFile) as exc:
            raise DatasetAccessError(f"cannot read candidate file {path!r}: {exc}") from exc
        try:
            with handle:
                yield from self._parse_lines(handle, path)
        except (OSError, gzip.BadGzipFile, EOFError) as exc:
            raise DatasetAccessError(f"cannot read candidate file {path!r}: {exc}") from exc

    @staticmethod
    def _parse_lines(handle: Any, path: str) -> Iterator[Any]:
        try:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    # Blank lines are not data; treat as malformed so the
                    # partition (valid + skipped == total) stays exact only for
                    # non-empty lines. We simply skip truly empty lines.
                    continue
                try:
                    yield json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    yield _MALFORMED
        except (OSError, gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
            raise DatasetAccessError(f"cannot read candidate file {path!r}: {exc}") from exc

    def _iter_json_array(self, path: str) -> Iterator[Any]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise DatasetAccessError(f"cannot read candidate file {path!r}: {exc}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise DatasetAccessError(
                f"candidate file {path!r} is not a valid JSON array: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise DatasetAccessError(
                f"candidate file {path!r} must contain a JSON array of records"
            )
        for element in data:
            yield element


# Sentinel marking a line that could not be parsed as JSON.
_MALFORMED: Any = object()
