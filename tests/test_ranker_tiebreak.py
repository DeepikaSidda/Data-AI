"""Regression test for the score-precision / tie-break consistency bug.

The SubmissionWriter prints ``score`` with ``SCORE_DECIMALS`` (6) decimal
places, while the Ranker previously sorted by the RAW float. Two distinct raw
scores that differ only beyond the 6th decimal round to the SAME printed value;
if the higher raw score belonged to the LARGER candidate_id, the printed CSV
showed equal scores with DESCENDING candidate_id at those ranks, which the
challenge validator rejects.

This test constructs exactly that adversarial case (raw scores differing only
in the 7th decimal, with candidate_ids in descending order relative to the raw
score), pads to exactly 100 rows with clearly-separated higher scores, writes a
real CSV via :class:`SubmissionWriter`, and asserts:

1. The committed ``validate_submission.py`` reports zero errors.
2. For any two adjacent ranked rows with equal printed (6-dp) score, the
   candidate_id is ascending.

Validates Requirements: 7.4, 9.6.
"""

from __future__ import annotations

import importlib.util
import os

from ranking.models import (
    CandidateRecord,
    DimensionScores,
    HoneypotResult,
    Profile,
    RedrobSignals,
    ScoredCandidate,
)
from ranking.ranker import Ranker
from ranking.writer import SCORE_DECIMALS, SubmissionWriter

# Absolute path to the committed challenge validator.
_VALIDATOR_PATH = (
    r"c:\Users\sidda\Downloads"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\India_runs_data_and_ai_challenge\validate_submission.py"
)


def _load_validator():
    """Import the committed validate_submission.py from its absolute path."""
    spec = importlib.util.spec_from_file_location(
        "challenge_validate_submission", _VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_record(candidate_id: str) -> CandidateRecord:
    profile = Profile(
        anonymized_name="",
        headline="",
        summary="",
        location="",
        country="",
        years_of_experience=0.0,
        current_title="",
        current_company="",
        current_company_size="",
        current_industry="",
    )
    signals = RedrobSignals(
        profile_completeness_score=0.0,
        signup_date="",
        last_active_date="",
        open_to_work_flag=False,
        profile_views_received_30d=0,
        applications_submitted_30d=0,
        recruiter_response_rate=0.0,
        avg_response_time_hours=0.0,
        connection_count=0,
        endorsements_received=0,
        notice_period_days=0,
        preferred_work_mode="",
        willing_to_relocate=False,
        search_appearance_30d=0,
        saved_by_recruiters_30d=0,
        interview_completion_rate=0.0,
        verified_email=False,
        verified_phone=False,
        linkedin_connected=False,
    )
    return CandidateRecord(
        candidate_id=candidate_id,
        profile=profile,
        career_history=[],
        education=[],
        skills=[],
        redrob_signals=signals,
    )


def _scored(candidate_id: str, final_score: float) -> ScoredCandidate:
    return ScoredCandidate(
        candidate_id=candidate_id,
        record=_minimal_record(candidate_id),
        dims=DimensionScores(
            semantic=0.0,
            skills_title=0.0,
            experience=0.0,
            trajectory=0.0,
            education=0.0,
        ),
        fit_score=0.0,
        behavioral_modifier=0.0,
        final_score=final_score,
        honeypot=HoneypotResult(False, []),
    )


def _build_adversarial_scored() -> list[ScoredCandidate]:
    """Exactly 100 candidates exercising the rounding tie-break.

    The first two share a printed score (raw scores differ only in the 7th
    decimal). The candidate with the HIGHER raw score is given the LARGER
    candidate_id, so a raw-float sort would place the larger id first and the
    printed CSV would show equal scores in DESCENDING id order (the bug). The
    remaining 98 rows have clearly-separated, strictly higher scores so they
    sort above the colliding pair and keep all 100 ids/ranks unique.
    """
    scored: list[ScoredCandidate] = []

    # Colliding pair: 0.1234561 and 0.1234564 both print as "0.123456".
    # Higher raw score -> larger candidate_id (descending id vs raw score).
    scored.append(_scored("CAND_0000002", 0.1234564))  # higher raw, larger id
    scored.append(_scored("CAND_0000001", 0.1234561))  # lower raw, smaller id

    # 98 padding rows, all strictly higher and well-separated so they outrank
    # the colliding pair and never collide with each other at 6 dp.
    for i in range(98):
        cid = f"CAND_{i + 100:07d}"
        score = 0.5 + i * 0.001  # 0.500 .. 0.597, all distinct at 6 dp
        scored.append(_scored(cid, score))

    assert len(scored) == 100
    return scored


def test_tiebreak_consistent_with_printed_score(tmp_path):
    """Ranked output written to CSV passes the committed validator.

    Validates Requirements: 7.4, 9.6.
    """
    scored = _build_adversarial_scored()
    ranked = Ranker().rank(scored, top_n=100)

    out_path = os.path.join(str(tmp_path), "team_test.csv")
    SubmissionWriter().write(ranked, out_path)

    validator = _load_validator()
    errors = validator.validate_submission(out_path)
    assert errors == [], f"validator reported errors: {errors}"


def test_adjacent_equal_printed_scores_ascending_id():
    """Adjacent rows with equal printed (6-dp) score have ascending id.

    Validates Requirements: 7.4, 9.6.
    """
    scored = _build_adversarial_scored()
    ranked = Ranker().rank(scored, top_n=100)

    fmt = f"{{:.{SCORE_DECIMALS}f}}"
    for earlier, later in zip(ranked, ranked[1:]):
        if fmt.format(earlier.score) == fmt.format(later.score):
            assert earlier.candidate_id < later.candidate_id, (
                "equal printed scores must be ordered by ascending candidate_id: "
                f"{earlier.candidate_id} !< {later.candidate_id}"
            )
