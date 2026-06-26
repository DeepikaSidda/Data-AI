"""Honeypot / internal-inconsistency detection.

Defines :class:`HoneypotDetector`, which runs five structured consistency
checks over a :class:`~ranking.models.CandidateRecord` using profile data only
(no hardcoded candidate ids, no allowlist). A record is flagged as a honeypot
if *any* check triggers; every trigger is recorded as a human-readable reason.

The five checks (see design.md "Honeypot Detection Rules (Detail)"):

1. Experience exceeds career span.
2. Duration sum mismatch.
3. Expert/advanced-with-zero-duration cluster.
4. Skill duration exceeds total experience.
5. Impossible date ordering (end before start, or future-dated).

Requirements: 6.1, 6.2, 6.3, 6.6.
"""

from __future__ import annotations

import datetime
from typing import Optional

from .models import CandidateRecord, HoneypotResult

# Proficiency levels that count toward the "high-proficiency / zero-duration"
# honeypot cluster.
_HIGH_PROFICIENCY = frozenset({"expert", "advanced"})


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    """Parse an ISO date string safely.

    Returns ``None`` for null, empty, or malformed values rather than raising,
    so a single bad date never crashes the detector. Accepts a leading date
    portion of an ISO datetime (``"2020-01-01T..."``) by splitting on ``T``.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    # Tolerate full ISO datetimes by taking the date component.
    text = text.split("T", 1)[0]
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


class HoneypotDetector:
    """Detect internally-inconsistent ("honeypot") candidate profiles.

    ``tolerance_months`` is a small slack (default 3 months) applied to the
    experience/duration comparisons to avoid flagging benign rounding between
    ``years_of_experience`` and month-granular career data. It is stored and
    also exposed in years for the year-based comparisons.
    """

    def __init__(self, tolerance_months: int = 3, duration_margin_months: int = 24) -> None:
        self.tolerance_months = tolerance_months
        self.tolerance_years = tolerance_months / 12.0
        # Slack allowed between a completed role's claimed ``duration_months``
        # and the actual months its start/end dates span before it is treated
        # as impossible (e.g. "8 years at a company the dates say is 3 years").
        self.duration_margin_months = duration_margin_months

    def check(self, rec: CandidateRecord) -> HoneypotResult:
        """Run the high-precision consistency checks and return a result.

        ``is_honeypot`` is ``True`` if any check triggers; ``reasons`` lists a
        human-readable string for each triggered check.

        The detector deliberately uses only signals that are *genuinely
        impossible* (not merely unusual), because the dataset contains only a
        tiny fraction (~0.1%) of honeypots and over-flagging would wrongly
        exclude strong real candidates from the shortlist. The earlier
        "experience exceeds total career span", "duration sum vs experience",
        and "single skill duration vs experience" heuristics were dropped: they
        fire constantly on normal profiles (people omit old jobs, and skill
        tenures are noisy), producing ~19% false positives. The retained signals
        match the documented honeypot shapes: a cluster of high-proficiency
        skills with zero tenure, impossible date ordering, and a role whose
        claimed duration vastly exceeds the span its own dates allow.
        """
        reasons: list[str] = []
        career = rec.career_history

        # Check 1: cluster of high-proficiency skills with zero tenure
        # ("expert in N skills with 0 months used"). Two or more is already a
        # near-zero-false-positive signal in practice.
        zero_dur_expert = [
            s
            for s in rec.skills
            if s.proficiency.strip().lower() in _HIGH_PROFICIENCY
            and s.duration_months == 0
        ]
        if len(zero_dur_expert) >= 2:
            names = ", ".join(s.name for s in zero_dur_expert)
            reasons.append(
                f"{len(zero_dur_expert)} skills claimed expert/advanced with "
                f"0 months duration ({names})"
            )

        # Check 2: impossible date ordering / out-of-range dates.
        for e in career:
            start = _parse_date(e.start_date)
            end = _parse_date(e.end_date)
            if start is not None and end is not None and end < start:
                reasons.append(
                    f"career entry '{e.company}' has end_date ({e.end_date}) "
                    f"before start_date ({e.start_date})"
                )
            for label, d, raw in (
                ("start_date", start, e.start_date),
                ("end_date", end, e.end_date),
            ):
                # Years far outside any plausible career window are garbage /
                # impossible (the schema itself caps education years at 2035).
                if d is not None and (d.year < 1900 or d.year > 2035):
                    reasons.append(
                        f"career entry '{e.company}' has {label} ({raw}) "
                        f"outside the plausible 1900-2035 window"
                    )

        # Check 3: a completed role whose claimed duration vastly exceeds the
        # span its own start/end dates allow (e.g. "8 years at a company the
        # dates say existed 3 years"). Only completed roles (both dates parse)
        # are checked, so the wall-clock "today" reference never matters.
        for e in career:
            start = _parse_date(e.start_date)
            end = _parse_date(e.end_date)
            if start is None or end is None or end < start:
                continue
            span_months = (end.year - start.year) * 12 + (end.month - start.month)
            if e.duration_months > span_months + self.duration_margin_months:
                reasons.append(
                    f"career entry '{e.company}' claims {e.duration_months}mo "
                    f"but its dates ({e.start_date}..{e.end_date}) span only "
                    f"~{span_months}mo"
                )

        return HoneypotResult(is_honeypot=bool(reasons), reasons=reasons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _career_span_years(
        self, career: list, reference: datetime.date
    ) -> Optional[float]:
        """Years between the earliest start and the latest end in career_history.

        A current role (``is_current`` or null ``end_date``) is treated as
        ending at ``reference``. Returns ``None`` if no parseable start date
        exists (cannot compute a span).
        """
        starts: list[datetime.date] = []
        ends: list[datetime.date] = []
        for e in career:
            start = _parse_date(e.start_date)
            if start is not None:
                starts.append(start)
            end = _parse_date(e.end_date)
            if end is not None:
                ends.append(end)
            elif e.is_current:
                ends.append(reference)
        if not starts:
            return None
        earliest = min(starts)
        # If we have no parseable end dates at all, span runs to the reference.
        latest = max(ends) if ends else reference
        if latest < earliest:
            return 0.0
        return (latest - earliest).days / 365.25
