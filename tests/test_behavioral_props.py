"""Property-based tests for the bounded BehavioralModifier.

# Feature: ai-candidate-ranking, Property 12: The behavioral modifier is bounded and never dominates
# Feature: ai-candidate-ranking, Property 13: The modifier is monotonic in availability signals

Validates Requirements: 5.1, 5.2, 5.3, 5.4, 5.5.

Both properties exercise ``BehavioralModifier`` against a fixed
``pool_latest_active`` reference date so recency is measured deterministically.
The modifier formula is ``(1 - s) + 2*s*signal_mean`` where the four sub-signals
each map into ``[0, 1]``; the realized minimum may sit above ``1 - s`` because
the open-to-work and notice/recency contributions have floors, but ``[1 - s,
1 + s]`` is still a valid envelope, so the bound assertions use a small
tolerance.
"""

from __future__ import annotations

import dataclasses
import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from ranking.behavioral import BehavioralModifier
from ranking.config import ScoringConfig
from ranking.models import HoneypotResult, RedrobSignals

# Fixed reference: the most recent activity across the pool.
POOL_LATEST_ACTIVE = datetime.date(2026, 6, 1)

# Numeric tolerance for floating-point comparisons on the bound envelope.
_TOL = 1e-9


def _iso(d: datetime.date) -> str:
    """Render a date as an ISO ``YYYY-MM-DD`` string."""
    return d.isoformat()


# Offsets (in days) before POOL_LATEST_ACTIVE, spanning both fresh (<183 days)
# and stale (>=183 days) regions so b_recency exercises both branches.
_recent_offset = st.integers(min_value=0, max_value=400)


@st.composite
def redrob_signals(draw: st.DrawFn) -> RedrobSignals:
    """Generate a ``RedrobSignals`` with valid ranges for the modifier inputs.

    Only the four fields the modifier reads are varied broadly
    (``last_active_date``, ``recruiter_response_rate``, ``notice_period_days``,
    ``open_to_work_flag``); the remaining required fields are filled with valid
    but fixed-shaped values since the modifier ignores them.
    """
    offset = draw(_recent_offset)
    last_active = POOL_LATEST_ACTIVE - datetime.timedelta(days=offset)
    response_rate = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    notice = draw(st.integers(min_value=0, max_value=180))
    open_to_work = draw(st.booleans())

    return RedrobSignals(
        profile_completeness_score=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        ),
        signup_date=_iso(POOL_LATEST_ACTIVE - datetime.timedelta(days=730)),
        last_active_date=_iso(last_active),
        open_to_work_flag=open_to_work,
        profile_views_received_30d=draw(st.integers(min_value=0, max_value=1000)),
        applications_submitted_30d=draw(st.integers(min_value=0, max_value=200)),
        recruiter_response_rate=response_rate,
        avg_response_time_hours=draw(
            st.floats(min_value=0.0, max_value=240.0, allow_nan=False)
        ),
        connection_count=draw(st.integers(min_value=0, max_value=5000)),
        endorsements_received=draw(st.integers(min_value=0, max_value=500)),
        notice_period_days=notice,
        preferred_work_mode=draw(st.sampled_from(["remote", "hybrid", "onsite"])),
        willing_to_relocate=draw(st.booleans()),
        search_appearance_30d=draw(st.integers(min_value=0, max_value=1000)),
        saved_by_recruiters_30d=draw(st.integers(min_value=0, max_value=200)),
        interview_completion_rate=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        ),
        verified_email=draw(st.booleans()),
        verified_phone=draw(st.booleans()),
        linkedin_connected=draw(st.booleans()),
    )


# behavioral_strength in [0, 0.99] so it stays inside the intended [0, 1) range.
_strength = st.floats(
    min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False
)


@settings(max_examples=100, deadline=None)
@given(
    signals=redrob_signals(),
    s=_strength,
    f1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    f2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_modifier_bounded_and_non_dominating(signals, s, f1, f2):
    """Property 12: modifier within [1-s, 1+s]; final_score monotonic in fit.

    Validates Requirements 5.1, 5.5.
    """
    config = ScoringConfig(behavioral_strength=s)
    bm = BehavioralModifier(config, POOL_LATEST_ACTIVE)

    m = bm.modifier(signals)
    # Bounded by the [1-s, 1+s] envelope (small tolerance for float error).
    assert m >= (1.0 - s) - _TOL
    assert m <= (1.0 + s) + _TOL

    # With identical signals the modifier is constant, so final_score ordering
    # must follow fit ordering: a smaller fit cannot yield a larger final_score.
    lo, hi = (f1, f2) if f1 <= f2 else (f2, f1)
    honeypot = HoneypotResult(is_honeypot=False)
    final_lo = bm.final_score(lo, signals, honeypot)
    final_hi = bm.final_score(hi, signals, honeypot)
    assert final_lo <= final_hi + _TOL


# Improvement axes for Property 13. Each holds all other signals fixed.
_IMPROVEMENTS = ("response", "recency", "notice", "open")


@settings(max_examples=100, deadline=None)
@given(
    signals=redrob_signals(),
    s=_strength,
    axis=st.sampled_from(_IMPROVEMENTS),
    delta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    days_improve=st.integers(min_value=0, max_value=400),
    notice_improve=st.integers(min_value=0, max_value=180),
)
def test_modifier_monotonic_in_availability(
    signals, s, axis, delta, days_improve, notice_improve
):
    """Property 13: improving one availability signal never lowers the modifier.

    Validates Requirements 5.2, 5.3, 5.4. The improved variant changes exactly
    one field via ``dataclasses.replace`` and is constructed so the improved
    value stays on the "better" side of the signal (higher response rate, more
    recent activity, shorter notice, or open-to-work true).
    """
    config = ScoringConfig(behavioral_strength=s)
    bm = BehavioralModifier(config, POOL_LATEST_ACTIVE)

    baseline_mod = bm.modifier(signals)

    if axis == "response":
        # Increase recruiter_response_rate, clamped to [current, 1.0].
        improved_rate = min(1.0, signals.recruiter_response_rate + delta)
        improved = dataclasses.replace(
            signals, recruiter_response_rate=improved_rate
        )
    elif axis == "recency":
        # Make last_active_date more recent: move it forward in time toward (or
        # up to) the pool-latest date. A later date can only lower staleness, so
        # b_recency is non-decreasing across the 183-day threshold.
        current = datetime.date.fromisoformat(signals.last_active_date)
        improved_date = min(
            POOL_LATEST_ACTIVE, current + datetime.timedelta(days=days_improve)
        )
        improved = dataclasses.replace(
            signals, last_active_date=_iso(improved_date)
        )
    elif axis == "notice":
        # Shorten notice_period_days: clamp the improved value to [0, current].
        improved_notice = min(signals.notice_period_days, notice_improve)
        improved = dataclasses.replace(
            signals, notice_period_days=improved_notice
        )
    else:  # axis == "open"
        # Set open_to_work_flag True (an improvement when previously False,
        # otherwise unchanged).
        improved = dataclasses.replace(signals, open_to_work_flag=True)

    improved_mod = bm.modifier(improved)
    assert improved_mod >= baseline_mod - _TOL
