"""Bounded behavioral modifier and Final_Score.

Defines :class:`BehavioralModifier`, which derives a bounded multiplier in
``[1 - s, 1 + s]`` (where ``s = config.behavioral_strength``) from a candidate's
:class:`RedrobSignals`. The signals are grouped into:

* **Availability** (dominant, weight 0.65) — the signals most directly tied to
  "can this person actually be hired right now": ``last_active_date`` recency,
  ``recruiter_response_rate``, ``notice_period_days``, and ``open_to_work_flag``.
* **Reliability / demand / trust** (weight 0.35) — ``interview_completion_rate``,
  ``offer_acceptance_rate`` (neutral when unknown), recruiter demand via
  ``saved_by_recruiters_30d``, and contactability via ``verified_email`` /
  ``verified_phone`` / ``linkedin_connected``.

It then computes ``final_score = clamp01(fit * modifier)`` and applies the
honeypot penalty so flagged records sink (default penalty ``0.0`` excludes them
entirely).

Because the modifier spans ``[1 - s, 1 + s]`` it adjusts but never solely
determines the Final_Score, preserving the relative influence of the
Fit_Score (Requirement 5.5).

See design.md "Behavioral modifier (bounded)" and "Honeypot penalty and
Final_Score". This implements the guidance in ``redrob_signals_doc.md`` to use
the behavioral signals as a modifier on top of skill-match scoring.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.4.
"""

from __future__ import annotations

import datetime
from typing import Optional

from .config import ScoringConfig
from .models import HoneypotResult, RedrobSignals

# Approximate "6 months" as 183 days for the recency boundary (design.md).
_RECENCY_WINDOW_DAYS = 183

# Neutral contribution used when a signal is missing/unknown: neither rewarded
# nor penalized.
_NEUTRAL = 0.5

# Relative weight of the availability group vs the reliability/demand/trust
# group. Availability stays dominant per redrob_signals_doc.md.
_AVAILABILITY_WEIGHT = 0.65
_ENGAGEMENT_WEIGHT = 0.35

# Normalizers for count-style demand signals (saturate at these values).
_SAVED_BY_RECRUITERS_FULL = 10.0


def _clamp01(value: float) -> float:
    """Clamp a value into the closed interval ``[0, 1]``."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    """Parse an ISO date string safely, returning ``None`` on failure.

    Tolerates a full ISO datetime by taking the leading date component, and
    never raises: null, empty, or malformed values yield ``None`` so a single
    bad date is treated as neutral rather than crashing the modifier.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = text.split("T", 1)[0]
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


class BehavioralModifier:
    """Compute a bounded behavioral multiplier and the resulting Final_Score.

    ``pool_latest_active`` is the most recent ``last_active_date`` across the
    Candidate_Pool; it is the reference against which each candidate's recency
    is measured, so "stale" is defined relative to the freshest activity in the
    pool rather than to wall-clock today (keeping results deterministic for a
    given pool).
    """

    def __init__(
        self, config: ScoringConfig, pool_latest_active: datetime.date
    ) -> None:
        self.config = config
        self.pool_latest_active = pool_latest_active

    def _availability_mean(self, signals: RedrobSignals) -> float:
        """Mean of the four core availability sub-signals, each in ``[0, 1]``.

        These are the signals the doc calls out as most predictive of whether a
        candidate can actually be hired right now.
        """
        # b_recency: 1.0 if active within 6 months of pool-latest, else 0.0;
        # neutral when the date cannot be parsed.
        last_active = _parse_date(signals.last_active_date)
        if last_active is None:
            b_recency = _NEUTRAL
        else:
            staleness_days = (self.pool_latest_active - last_active).days
            b_recency = 1.0 if staleness_days < _RECENCY_WINDOW_DAYS else 0.0

        # b_response: recruiter_response_rate is already in [0, 1]; clamp for
        # safety against out-of-range source data.
        b_response = _clamp01(signals.recruiter_response_rate)

        # b_notice: shorter notice is better.
        notice = signals.notice_period_days
        if notice < 30:
            b_notice = 1.0
        elif notice < 60:
            b_notice = 0.5
        else:
            b_notice = 0.0

        # b_open: open-to-work is rewarded; not-open is neutral (0.5), not zero.
        b_open = 1.0 if signals.open_to_work_flag else 0.5

        return (b_recency + b_response + b_notice + b_open) / 4.0

    def _engagement_mean(self, signals: RedrobSignals) -> float:
        """Mean of the reliability / demand / trust sub-signals, each in ``[0, 1]``.

        Adds breadth from the remaining Redrob signals without letting them
        dominate: interview reliability, historical offer acceptance (neutral
        when unknown), recruiter demand, and contactability/verification.
        """
        # Reliability: fraction of scheduled interviews actually attended.
        b_interview = _clamp01(signals.interview_completion_rate)

        # Historical offer acceptance. The schema uses -1 ("no history"), which
        # models.py maps to None -> treat as neutral so first-time candidates
        # are neither rewarded nor penalized.
        if signals.offer_acceptance_rate is None:
            b_offer = _NEUTRAL
        else:
            b_offer = _clamp01(signals.offer_acceptance_rate)

        # Recruiter demand: how many recruiters bookmarked the profile recently
        # (saturating). A market signal that others find the profile worth
        # saving.
        b_saved = _clamp01(signals.saved_by_recruiters_30d / _SAVED_BY_RECRUITERS_FULL)

        # Contactability / trust: verified email + phone + connected LinkedIn.
        verified = (
            int(bool(signals.verified_email))
            + int(bool(signals.verified_phone))
            + int(bool(signals.linkedin_connected))
        )
        b_verified = verified / 3.0

        return (b_interview + b_offer + b_saved + b_verified) / 4.0

    def modifier(self, signals: RedrobSignals) -> float:
        """Return a bounded multiplier in ``[1 - s, 1 + s]`` for ``signals``.

        Combines the dominant availability group (weight 0.65) with the
        reliability/demand/trust group (weight 0.35) into a single
        ``signal_mean`` in ``[0, 1]``, then maps it via
        ``(1 - s) + 2*s*signal_mean`` so a fully-available, highly-engaged
        candidate approaches ``1 + s`` and a fully-unavailable one approaches
        ``1 - s`` (Requirements 5.1-5.5).
        """
        s = self.config.behavioral_strength
        availability = self._availability_mean(signals)
        engagement = self._engagement_mean(signals)
        signal_mean = (
            _AVAILABILITY_WEIGHT * availability + _ENGAGEMENT_WEIGHT * engagement
        )
        return (1.0 - s) + 2.0 * s * signal_mean

    def final_score(
        self, fit: float, signals: RedrobSignals, honeypot: HoneypotResult
    ) -> float:
        """Combine ``fit`` with the behavioral modifier and honeypot penalty.

        ``Final_Score = clamp01(fit * modifier(signals))``; if the record is a
        honeypot the result is multiplied by ``config.honeypot_penalty``
        (default ``0.0``, which zeroes flagged records so they cannot reach the
        top 100). The returned value is always in ``[0, 1]``
        (Requirements 5.1, 6.4).
        """
        final = _clamp01(fit * self.modifier(signals))
        if honeypot.is_honeypot:
            final = final * self.config.honeypot_penalty
        return _clamp01(final)
