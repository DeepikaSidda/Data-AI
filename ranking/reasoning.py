"""Offline grounded reasoning generation.

Defines :class:`ReasoningGenerator`, a fully offline, deterministic,
fact-grounded sentence assembler (no hosted LLM, no network — Requirement 8.1).
For each ranked candidate it produces a 1-2 sentence justification that:

* references at least one concrete fact drawn from the candidate's own
  :class:`~ranking.models.CandidateRecord` — ``years_of_experience``,
  ``current_title``, a named skill actually in ``rec.skills``, an employer in
  ``rec.career_history``, or a specific ``redrob_signals`` value such as
  ``notice_period_days`` / ``recruiter_response_rate`` / ``last_active_date``
  (Requirement 8.2);
* connects that fact to a specific Job_Profile requirement — production
  retrieval / ranking experience, strong Python, evaluation frameworks
  (NDCG/MRR/MAP), or product-company experience (Requirement 8.3);
* never emits a skill, employer, or attribute not present in the record — skill
  and employer strings are pulled verbatim from ``rec.skills`` and
  ``rec.career_history`` (Requirement 8.5);
* acknowledges a notable concern when present — a notice period of 30+ days, a
  consulting-services-only background, profile inactivity, or weak fit
  dimensions (Requirement 8.4);
* varies phrasing across candidates by selecting among multiple sentence
  structures and fact orderings via a deterministic hash seeded by
  ``candidate_id`` (default ``seed_field``), so different candidates read
  differently while the same candidate is always identical
  (Requirements 8.6, 7.5); the tone is bucketed by rank band (Requirement 8.6);
* stays within 1-2 sentences (Requirement 8.1).

See design.md "Reasoning Generation Strategy".

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6.
"""

from __future__ import annotations

import datetime
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from .job_profile import JobProfile
from .models import CandidateRecord, DimensionScores, HoneypotResult


# Profile inactivity threshold (~6 months), matching the behavioral modifier.
_INACTIVITY_DAYS = 183

# A dimension at or below this value is treated as a notable weak signal.
_WEAK_DIMENSION = 0.4

# Sentence terminators followed by whitespace or end-of-string. Used to count
# sentences without being fooled by decimals like "6.9" or "0.42".
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    """Parse an ISO date (or datetime) string safely; ``None`` on failure."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip().split("T", 1)[0]
    if not text:
        return None
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


def _sentence_count(text: str) -> int:
    """Count sentences in ``text`` by terminators, ignoring decimal points."""
    return len(_SENTENCE_END.findall(text))


@dataclass(frozen=True)
class GroundedFacts:
    """Facts extracted *only* from a candidate record (no external data).

    Every string field here originates verbatim from the record so the
    generator can never reference a skill, employer, or attribute the candidate
    does not actually have (Requirement 8.5).
    """

    years: float
    current_title: str
    relevant_skill: Optional[str] = None      # job-relevant skill name (verbatim)
    relevant_skill_kind: str = ""             # connective category for that skill
    other_skill: Optional[str] = None         # any named skill (verbatim) fallback
    product_employer: Optional[str] = None    # employer that is a product company
    employer: Optional[str] = None            # any employer name (verbatim)
    notice_days: int = 0
    recruiter_response_rate: float = 0.0
    last_active: Optional[datetime.date] = None
    consulting_only: bool = False
    employer_count: int = 0


def _classify_skill(name_lower: str, job: JobProfile) -> str:
    """Return a connective category for a job-relevant skill name.

    Categories map a candidate skill to a specific Job_Profile requirement
    (Requirement 8.3). Returns ``""`` if the skill is not job-relevant.
    """
    metrics = {m.lower() for m in job.positive_signals.eval_metrics}
    if name_lower in metrics or name_lower in {"ndcg", "mrr", "map"}:
        return "eval"
    if "python" in name_lower:
        return "python"
    retrieval_markers = (
        "retriev", "rank", "recommend", "recsys", "search", "embedding",
        "vector", "information retrieval", "nlp", "semantic",
    )
    if any(marker in name_lower for marker in retrieval_markers):
        return "retrieval"
    # Otherwise relevant only if it matches a positive skill term.
    for term in job.positive_signals.skill_terms:
        term_l = term.lower()
        if term_l and (term_l in name_lower or name_lower in term_l):
            return "relevant"
    return ""


def extract_facts(rec: CandidateRecord, job: JobProfile) -> GroundedFacts:
    """Extract grounded facts from ``rec`` connected to ``job`` requirements.

    Pulls skill and employer names verbatim from the record. Selects the most
    role-relevant skill (retrieval/ranking > python > eval metric > other
    relevant) and the most role-relevant employer (a product company if one is
    present) so the generated reasoning can ground itself in the strongest
    available evidence. No data outside ``rec`` is introduced (Requirement 8.5).
    """
    profile = rec.profile

    # --- skills: choose the most role-relevant, keep verbatim name ---
    relevant_skill: Optional[str] = None
    relevant_kind = ""
    # Priority of connective categories when several relevant skills exist.
    kind_priority = {"retrieval": 0, "python": 1, "eval": 2, "relevant": 3}
    best_priority = 99
    other_skill: Optional[str] = None
    for skill in rec.skills:
        name = (skill.name or "").strip()
        if not name:
            continue
        if other_skill is None:
            other_skill = name
        kind = _classify_skill(name.lower(), job)
        if kind:
            pri = kind_priority.get(kind, 50)
            if pri < best_priority:
                best_priority = pri
                relevant_skill = name
                relevant_kind = kind

    # --- employers: any employer, and a product-company employer if present ---
    product_companies = {c.lower() for c in job.positive_signals.product_companies}
    consulting_firms = [c.lower() for c in job.negative_signals.consulting_firms]
    employer: Optional[str] = None
    product_employer: Optional[str] = None
    company_names: list[str] = []
    for entry in rec.career_history:
        company = (entry.company or "").strip()
        if not company:
            continue
        company_names.append(company)
        if employer is None:
            employer = company
        if product_employer is None and company.lower() in product_companies:
            product_employer = company

    # --- consulting-only background (every employer is a services firm) ---
    consulting_only = False
    if company_names:
        consulting_only = all(
            any(firm in name.lower() for firm in consulting_firms)
            for name in company_names
        )

    signals = rec.redrob_signals
    return GroundedFacts(
        years=profile.years_of_experience,
        current_title=(profile.current_title or "").strip(),
        relevant_skill=relevant_skill,
        relevant_skill_kind=relevant_kind,
        other_skill=other_skill,
        product_employer=product_employer,
        employer=employer,
        notice_days=signals.notice_period_days,
        recruiter_response_rate=signals.recruiter_response_rate,
        last_active=_parse_date(signals.last_active_date),
        consulting_only=consulting_only,
        employer_count=len(company_names),
    )


# Tone leads keyed by rank band; selected deterministically per candidate.
_LEADS = {
    "top": (
        "A standout fit", "A top-tier match", "An exceptional candidate",
        "A clear front-runner",
    ),
    "strong": (
        "A strong fit", "A well-aligned candidate", "A solid, strong match",
        "A compelling candidate",
    ),
    "solid": (
        "A solid candidate", "A reasonable fit", "A credible option",
        "A workable fit",
    ),
}

# Connective phrases tying a skill category to a Job_Profile requirement.
_SKILL_CONNECT = {
    "retrieval": (
        "aligned with the role's production retrieval and ranking focus",
        "matching the role's emphasis on search and ranking systems",
    ),
    "python": (
        "matching the role's strong-Python requirement",
        "covering the role's core Python expectation",
    ),
    "eval": (
        "speaking to the role's NDCG/MRR/MAP evaluation rigor",
        "fitting the role's ranking-evaluation requirements",
    ),
    "relevant": (
        "relevant to the role's engineering needs",
        "useful for the role",
    ),
}

# Caveat lead-ins for a standalone concern sentence.
_CAVEAT_LEAD = ("One caveat:", "Worth flagging:", "Note:", "Main watch-point:")


def _band(rank: int) -> str:
    """Bucket a rank into a tone band (Requirement 8.6)."""
    if rank <= 10:
        return "top"
    if rank <= 50:
        return "strong"
    return "solid"


class ReasoningGenerator:
    """Assemble grounded, varied, offline reasoning for a ranked candidate.

    The generator is pure and deterministic: given the same record, dimension
    scores, honeypot result, and rank it always returns the same string, and
    different candidates receive substantively different phrasing because every
    template choice is keyed off a hash of ``getattr(rec, seed_field)``
    (Requirements 8.6, 7.5).

    ``pool_latest_active`` is optional; when supplied it is the most recent
    activity date across the pool and lets the generator recognize profile
    inactivity (>= ~6 months stale) as a concern (Requirement 8.4). When it is
    not supplied, inactivity is simply not raised (the other concern checks
    still apply).
    """

    def __init__(
        self,
        job: JobProfile,
        seed_field: str = "candidate_id",
        pool_latest_active: Optional[datetime.date] = None,
    ) -> None:
        self.job = job
        self.seed_field = seed_field
        self.pool_latest_active = pool_latest_active

    # -- deterministic hashing ------------------------------------------------

    def _digest(self, rec: CandidateRecord) -> bytes:
        """Stable hash bytes seeded by the configured seed field."""
        seed = str(getattr(rec, self.seed_field, "") or "")
        return hashlib.sha256(seed.encode("utf-8")).digest()

    @staticmethod
    def _pick(options: tuple, byte: int):
        """Deterministically choose one option using a digest byte."""
        return options[byte % len(options)]

    # -- fact clause assembly -------------------------------------------------

    def _fact_clauses(self, facts: GroundedFacts, digest: bytes) -> list[str]:
        """Build grounded fact clauses, ordered by a deterministic hash.

        Always returns at least one clause referencing a concrete record fact
        (Requirement 8.2). Each clause connects the fact to a specific
        Job_Profile requirement (Requirement 8.3) and uses only verbatim record
        strings for skills/employers (Requirement 8.5).
        """
        clauses: list[str] = []

        # Years of experience, connected to the 5-9 year band when it fits.
        years = facts.years
        if years and years > 0:
            years_text = f"{years:g}"
            if 5.0 <= years <= 9.0:
                clauses.append(
                    f"{years_text} years of experience squarely in the role's "
                    f"5-9 year band"
                )
            else:
                clauses.append(f"{years_text} years of experience")

        # Role-relevant skill (verbatim name) tied to a specific requirement.
        if facts.relevant_skill:
            connect = self._pick(
                _SKILL_CONNECT.get(facts.relevant_skill_kind, _SKILL_CONNECT["relevant"]),
                digest[4],
            )
            clauses.append(f"hands-on {facts.relevant_skill} {connect}")

        # Product-company experience (employer verbatim) -> product-company req.
        if facts.product_employer:
            clauses.append(
                f"product-company experience at {facts.product_employer}"
            )

        # Current title as a grounded engineering-background signal.
        title = facts.current_title
        if title:
            title_terms = [t.lower() for t in self.job.positive_signals.title_terms]
            if any(t in title.lower() for t in title_terms):
                clauses.append(f"a genuine {title} background")
            else:
                clauses.append(f"a {title} background")

        # Fallbacks so a concrete fact is always present (Requirement 8.2).
        if not clauses and facts.other_skill:
            clauses.append(f"listed {facts.other_skill} among their skills")
        if not clauses and facts.employer:
            clauses.append(f"experience at {facts.employer}")
        if not clauses:
            # Last-resort grounded numeric facts that always exist.
            clauses.append(
                f"a notice period of {facts.notice_days} days"
            )

        # Deterministically rotate clause order for variation, then keep <=2.
        if len(clauses) > 1:
            rotate = digest[1] % len(clauses)
            clauses = clauses[rotate:] + clauses[:rotate]
        return clauses[:2]

    # -- concern detection ----------------------------------------------------

    def _concern(
        self,
        facts: GroundedFacts,
        dims: DimensionScores,
        honeypot: HoneypotResult,
    ) -> Optional[str]:
        """Return a single grounded concern clause, or ``None``.

        Priority: honeypot-adjacent inconsistency, consulting-only background,
        long notice period, profile inactivity, then weak fit dimensions
        (Requirement 8.4). The clause is phrased as a concession.
        """
        if honeypot.is_honeypot:
            return "the profile shows internal data inconsistencies"

        if facts.consulting_only:
            return (
                "a consulting-services-only background is a watch-point for "
                "production depth"
            )

        if facts.notice_days >= 30:
            return f"the {facts.notice_days}-day notice period raises the bar"

        if (
            self.pool_latest_active is not None
            and facts.last_active is not None
        ):
            stale_days = (self.pool_latest_active - facts.last_active).days
            if stale_days >= _INACTIVITY_DAYS:
                return "the profile has been inactive for an extended period"

        weak = min(dims.semantic, dims.skills_title, dims.trajectory)
        if weak <= _WEAK_DIMENSION:
            return "some fit dimensions are weaker than ideal"

        return None

    # -- public API -----------------------------------------------------------

    def generate(
        self,
        rec: CandidateRecord,
        dims: DimensionScores,
        honeypot: HoneypotResult,
        rank: int,
    ) -> str:
        """Generate a 1-2 sentence grounded, varied justification for ``rec``.

        Deterministic in all inputs and fully offline (Requirements 8.1, 7.5).
        """
        digest = self._digest(rec)
        facts = extract_facts(rec, self.job)
        band = _band(rank)
        lead = self._pick(_LEADS[band], digest[0])
        clauses = self._fact_clauses(facts, digest)
        concern = self._concern(facts, dims, honeypot)

        # Join up to two grounded fact clauses.
        if len(clauses) == 2:
            fact_text = f"{clauses[0]} and {clauses[1]}"
        else:
            fact_text = clauses[0]

        # Choose one of several sentence structures for variation (8.6).
        structure = digest[2] % 3
        if structure == 0:
            sentence = f"{lead} for this Senior AI Engineer role, with {fact_text}"
        elif structure == 1:
            sentence = f"{lead} for the role: {fact_text}"
        else:
            sentence = f"{fact_text} make this {lead.lower()} for the role"

        if concern is None:
            text = sentence + "."
        else:
            # Either fold the concern into sentence 1 ("though ...") or emit it
            # as a short second sentence — chosen deterministically.
            if digest[3] % 2 == 0:
                text = f"{sentence}, though {concern}."
            else:
                caveat_lead = self._pick(_CAVEAT_LEAD, digest[5])
                text = f"{sentence}. {caveat_lead} {concern}."

        # Safety net: never exceed two sentences (Requirement 8.1).
        if _sentence_count(text) > 2:
            text = sentence + "."
        return text
