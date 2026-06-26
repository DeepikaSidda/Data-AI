"""Structured feature dimension extraction and anti-keyword-stuffing logic.

Defines :class:`FeatureExtractor`, which turns a :class:`CandidateRecord` plus a
precomputed semantic-similarity scalar into a :class:`DimensionScores` value with
five normalized ``[0, 1]`` dimensions:

* ``semantic``     - the passed-in cosine-derived similarity (already in ``[0, 1]``).
* ``skills_title`` - title-aware skills alignment, ``0.7*title_align +
  0.3*skill_align*trust``. Titles are weighted above the raw skill count
  (Requirement 4.3) and the per-skill *trust* multiplier resists keyword
  stuffing (Requirements 4.1, 4.2).
* ``experience``   - fit to the 5-9 year band with soft edges (Requirement 3.1).
* ``trajectory``   - product-vs-services and anti-hopping evidence
  (Requirements 3.3, 3.4).
* ``education``    - light tier-based score.

All matching is driven by the offline-encoded :class:`JobProfile` signal lists
(``positive_signals`` / ``negative_signals``), is case-insensitive, and uses no
hardcoded candidate identifiers (Requirement 4.4). The extractor is pure and
deterministic: identical inputs always yield identical outputs.

Requirements: 3.1, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4.
"""

from __future__ import annotations

from typing import Iterable

from ranking.config import ScoringConfig
from ranking.job_profile import JobProfile
from ranking.models import CandidateRecord, DimensionScores, Skill

# Proficiency -> base weight for the per-skill trust multiplier.
_PROFICIENCY_WEIGHT: dict[str, float] = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}

# Skill-alignment is normalized against this many distinct matched job skill
# terms: a candidate matching this many (or more) genuinely relevant skills gets
# a full skill_align of 1.0. Kept small so a focused, genuine engineer reaches
# the ceiling while the trust multiplier (not raw count) does the stuffing
# resistance.
_SKILL_ALIGN_REFERENCE = 5.0

# Build-evidence keywords that signal someone shipped recommendation / search /
# ranking / retrieval systems. These are the role's core "build" signals; the
# job's positive skill terms and phrases are also scanned in addition to these.
_BUILD_KEYWORDS: tuple[str, ...] = (
    "recommend",
    "search",
    "ranking",
    "recsys",
    "retrieval",
    "embedding",
    "learning to rank",
    "personalization",
    "personalisation",
)

# A career stint at or below this many months counts as "short" for the
# title-chasing / job-hopping penalty.
_SHORT_STINT_MONTHS = 18

# Education tier -> score (best/highest tier wins; unknown/missing -> 0.2).
_EDUCATION_TIER_SCORE: dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.7,
    "tier_3": 0.4,
    "tier_4": 0.2,
    "unknown": 0.2,
}
_EDUCATION_DEFAULT_SCORE = 0.2


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the closed ``[0, 1]`` interval."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    """Return True if any non-empty term is a case-insensitive substring of text."""
    lowered = text.lower()
    return any(term and term.lower() in lowered for term in terms)


class FeatureExtractor:
    """Computes normalized structured dimension scores for a candidate.

    The extractor is constructed once per run with the offline-encoded
    :class:`JobProfile` and the :class:`ScoringConfig`, then ``extract`` is
    called per candidate. It holds no mutable state and performs no I/O, so it
    is pure and deterministic.
    """

    def __init__(self, job: JobProfile, config: ScoringConfig) -> None:
        """Bind the extractor to a job profile and scoring config.

        Args:
            job: The offline-encoded role profile whose signal lists drive all
                title / skill / trajectory matching.
            config: The scoring configuration (held for parity with the other
                pure core components; dimension blending happens in the scorer).
        """
        self._job = job
        self._config = config

        pos = job.positive_signals
        neg = job.negative_signals
        # Lower-cased signal lists, prepared once for repeated matching.
        self._pos_title_terms = tuple(t.lower() for t in pos.title_terms if t)
        self._pos_skill_terms = tuple(t.lower() for t in pos.skill_terms if t)
        self._stuffer_titles = tuple(t.lower() for t in neg.keyword_stuffer_titles if t)
        self._consulting_firms = tuple(t.lower() for t in neg.consulting_firms if t)
        self._product_companies = tuple(t.lower() for t in pos.product_companies if t)
        # Build-evidence terms = curated build keywords plus the role's own
        # positive phrases (so trajectory tracks the job profile, not hardcoding).
        self._build_terms = _BUILD_KEYWORDS + tuple(p.lower() for p in pos.phrases if p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, rec: CandidateRecord, semantic_similarity: float
    ) -> DimensionScores:
        """Compute the five normalized ``[0, 1]`` dimension scores for ``rec``.

        Args:
            rec: The candidate to score.
            semantic_similarity: The candidate's cosine-derived similarity to the
                job profile, already mapped into ``[0, 1]``.

        Returns:
            A :class:`DimensionScores` with every field in ``[0, 1]``.
        """
        semantic = _clamp01(float(semantic_similarity))
        skills_title = self._skills_title(rec)
        experience = self._experience(rec.profile.years_of_experience)
        trajectory = self._trajectory(rec)
        education = self._education(rec)
        return DimensionScores(
            semantic=semantic,
            skills_title=skills_title,
            experience=experience,
            trajectory=trajectory,
            education=education,
        )

    # ------------------------------------------------------------------
    # skills_title: title-aware alignment + anti-stuffing trust multiplier
    # ------------------------------------------------------------------

    def _skills_title(self, rec: CandidateRecord) -> float:
        """``clamp01(0.7*title_align + 0.3*skill_align*trust)`` (Req 4.1-4.3)."""
        title_align = self._title_align(rec)
        skill_align = self._skill_align(rec.skills)
        trust = self._skill_trust_mean(rec.skills)
        return _clamp01(0.7 * title_align + 0.3 * skill_align * trust)

    def _title_align(self, rec: CandidateRecord) -> float:
        """Overlap of the candidate's titles with the role's positive titles.

        The current title is weighted above past titles. A title that matches a
        keyword-stuffer negative title but no positive title contributes 0, so an
        unrelated title (e.g. "Marketing Manager") drives ``title_align`` toward
        zero regardless of how many skills are listed.
        """
        current_rel = self._title_relevance(rec.profile.current_title)
        history_titles = [e.title for e in rec.career_history]
        if history_titles:
            history_rel = sum(
                self._title_relevance(t) for t in history_titles
            ) / len(history_titles)
        else:
            history_rel = current_rel
        return _clamp01(0.6 * current_rel + 0.4 * history_rel)

    def _title_relevance(self, title: str) -> float:
        """Relevance of a single title: 1.0 if positive, 0.0 otherwise.

        A title matching a negative keyword-stuffer term but no positive term is
        explicitly 0.0 (a defensive no-op since a non-matching title is already
        0.0, but it documents intent).
        """
        if not title:
            return 0.0
        is_positive = _contains_any(title, self._pos_title_terms)
        if is_positive:
            return 1.0
        # Unrelated / keyword-stuffer title.
        return 0.0

    def _skill_align(self, skills: list[Skill]) -> float:
        """Presence-based coverage of job-relevant skill terms in ``[0, 1]``.

        Counts the distinct job skill terms the candidate lists at all and
        normalizes against :data:`_SKILL_ALIGN_REFERENCE`. This measures *what*
        relevant skills are present; the trust multiplier separately discounts
        unsupported (zero-duration / unendorsed) claims.
        """
        if not skills or not self._pos_skill_terms:
            return 0.0
        names = [s.name.lower() for s in skills if s.name]
        matched_terms = sum(
            1 for term in self._pos_skill_terms if any(term in name for name in names)
        )
        return _clamp01(matched_terms / _SKILL_ALIGN_REFERENCE)

    def _skill_trust_mean(self, skills: list[Skill]) -> float:
        """Mean per-skill trust over the candidate's job-relevant skills.

        Per-skill trust is
        ``proficiency_weight * duration_factor * (0.5 + 0.5*endorsement_factor)``
        where ``duration_factor = min(1, duration_months/24)`` and
        ``endorsement_factor = min(1, endorsements/10)``. A skill claimed at
        ``expert`` with ``duration_months == 0`` therefore has trust 0 and
        contributes nothing (Requirement 4.2). Returns 0.0 when the candidate
        lists no job-relevant skills.
        """
        relevant = [s for s in skills if self._is_relevant_skill(s)]
        if not relevant:
            return 0.0
        return sum(self._skill_trust(s) for s in relevant) / len(relevant)

    def _is_relevant_skill(self, skill: Skill) -> bool:
        """True if the skill name matches any positive job skill term."""
        if not skill.name:
            return False
        name = skill.name.lower()
        return any(term in name for term in self._pos_skill_terms)

    @staticmethod
    def _skill_trust(skill: Skill) -> float:
        """Per-skill trust multiplier in ``[0, 1]`` (anti-keyword-stuffing)."""
        proficiency_weight = _PROFICIENCY_WEIGHT.get(skill.proficiency.lower(), 0.0)
        duration_factor = min(1.0, max(0, skill.duration_months) / 24.0)
        endorsement_factor = min(1.0, max(0, skill.endorsements) / 10.0)
        return proficiency_weight * duration_factor * (0.5 + 0.5 * endorsement_factor)

    # ------------------------------------------------------------------
    # experience: 5-9 year soft band
    # ------------------------------------------------------------------

    @staticmethod
    def _experience(years: float) -> float:
        """Fit to the 5-9 year band with soft lower/upper edges (Requirement 3.1)."""
        y = float(years)
        if 5.0 <= y <= 9.0:
            return 1.0
        if y < 5.0:
            return max(0.0, 1.0 - (5.0 - y) / 3.0)
        return max(0.0, 1.0 - (y - 9.0) / 6.0)

    # ------------------------------------------------------------------
    # trajectory: product-vs-services and anti-hopping
    # ------------------------------------------------------------------

    def _trajectory(self, rec: CandidateRecord) -> float:
        """Career trajectory score, starting at 0.5 (Requirements 3.3, 3.4).

        Rewards shipping recommendation/search/ranking systems and product-company
        experience; penalizes consulting-only careers and title-chasing hopping.
        """
        score = 0.5
        career = rec.career_history

        # Build-evidence: did they ship recsys / search / ranking / retrieval?
        career_text = " ".join(
            f"{e.title} {e.description}" for e in career
        )
        # Also fold in the headline / summary as supporting evidence.
        career_text = f"{career_text} {rec.profile.headline} {rec.profile.summary}"
        if _contains_any(career_text, self._build_terms):
            score += 0.25

        # Product-company experience.
        if self._has_product_company(career):
            score += 0.15

        # Consulting-only career penalty.
        if self._is_consulting_only(career):
            score -= 0.35

        # Title-chasing job-hopping penalty.
        if self._is_title_chasing(career):
            score -= 0.25

        return _clamp01(score)

    def _has_product_company(self, career: list) -> bool:
        """True if any career company matches a known product company."""
        if not self._product_companies:
            return False
        return any(
            _contains_any(e.company, self._product_companies) for e in career if e.company
        )

    def _is_consulting_only(self, career: list) -> bool:
        """True if the candidate has career entries and *all* are consulting firms."""
        named = [e for e in career if e.company]
        if not named or not self._consulting_firms:
            return False
        return all(
            _contains_any(e.company, self._consulting_firms) for e in named
        )

    def _is_title_chasing(self, career: list) -> bool:
        """True for many short stints with frequent title changes (Req 3.4)."""
        if len(career) < 3:
            return False
        short_stints = sum(
            1 for e in career if 0 < e.duration_months <= _SHORT_STINT_MONTHS
        )
        if short_stints < 3:
            return False
        distinct_titles = {e.title.strip().lower() for e in career if e.title}
        return len(distinct_titles) >= 3

    # ------------------------------------------------------------------
    # education: tier-based, best tier wins
    # ------------------------------------------------------------------

    @staticmethod
    def _education(rec: CandidateRecord) -> float:
        """Best (highest) education tier score; 0.2 when no education present."""
        if not rec.education:
            return _EDUCATION_DEFAULT_SCORE
        return max(
            _EDUCATION_TIER_SCORE.get(e.tier.strip().lower(), _EDUCATION_DEFAULT_SCORE)
            for e in rec.education
        )
