"""Offline-encoded Job_Profile.

Defines the immutable structured representation of the Senior AI Engineer role
(Redrob challenge) and a :meth:`JobProfile.load` classmethod that reads the
committed, hand-encoded ``job_profile.yaml`` derived offline from
``job_description.md``.

All loading is local file I/O only; there is **no network access** anywhere in
this module (Requirement 2.1, 2.5).

Requirements: 2.1, 2.2, 2.3, 2.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class PositiveSignals:
    """True signals of genuine fit for the role (Requirement 2.2)."""

    phrases: tuple[str, ...] = ()
    skill_terms: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    eval_metrics: tuple[str, ...] = ()
    product_companies: tuple[str, ...] = ()
    nice_to_have: tuple[str, ...] = ()


@dataclass(frozen=True)
class NegativeSignals:
    """Disqualifying / down-weighting signals (Requirement 2.3)."""

    keyword_stuffer_titles: tuple[str, ...] = ()
    consulting_firms: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    off_domain_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocationPref:
    """Location preference for the role (Requirement 2.4)."""

    preferred_cities: tuple[str, ...] = ()
    welcome_cities: tuple[str, ...] = ()
    relocation_ok: bool = True
    india_required: bool = False
    outside_india_case_by_case: bool = True


@dataclass(frozen=True)
class NoticePref:
    """Notice-period preference for the role (Requirement 2.4)."""

    preferred_max_days: int = 30
    raises_bar_at_days: int = 30


@dataclass(frozen=True)
class JobProfile:
    """Structured, offline-derived representation of the Job_Description.

    Hand-encoded once from ``job_description.md`` and committed as
    ``job_profile.yaml``; loaded with no network access (Requirement 2.1).
    """

    positive_signals: PositiveSignals
    negative_signals: NegativeSignals
    location_pref: LocationPref
    notice_pref: NoticePref
    profile_text: str

    @classmethod
    def load(cls, path: str = "job_profile.yaml") -> "JobProfile":
        """Load the hand-encoded profile from a local YAML file.

        Args:
            path: Path to the committed ``job_profile.yaml``.

        Returns:
            A fully-populated, immutable :class:`JobProfile`.

        No network access occurs; this is local file I/O only
        (Requirements 2.1, 2.5).
        """
        with open(path, "r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle) or {}

        if not isinstance(raw, Mapping):
            raise ValueError(
                f"job profile at {path!r} must be a YAML mapping, "
                f"got {type(raw).__name__}"
            )

        pos = _as_mapping(raw.get("positive_signals"))
        neg = _as_mapping(raw.get("negative_signals"))
        loc = _as_mapping(raw.get("location_pref"))
        notice = _as_mapping(raw.get("notice_pref"))

        positive_signals = PositiveSignals(
            phrases=_as_str_tuple(pos.get("phrases")),
            skill_terms=_as_str_tuple(pos.get("skill_terms")),
            title_terms=_as_str_tuple(pos.get("title_terms")),
            eval_metrics=_as_str_tuple(pos.get("eval_metrics")),
            product_companies=_as_str_tuple(pos.get("product_companies")),
            nice_to_have=_as_str_tuple(pos.get("nice_to_have")),
        )
        negative_signals = NegativeSignals(
            keyword_stuffer_titles=_as_str_tuple(neg.get("keyword_stuffer_titles")),
            consulting_firms=_as_str_tuple(neg.get("consulting_firms")),
            flags=_as_str_tuple(neg.get("flags")),
            off_domain_terms=_as_str_tuple(neg.get("off_domain_terms")),
        )
        location_pref = LocationPref(
            preferred_cities=_as_str_tuple(loc.get("preferred_cities")),
            welcome_cities=_as_str_tuple(loc.get("welcome_cities")),
            relocation_ok=bool(loc.get("relocation_ok", True)),
            india_required=bool(loc.get("india_required", False)),
            outside_india_case_by_case=bool(
                loc.get("outside_india_case_by_case", True)
            ),
        )
        notice_pref = NoticePref(
            preferred_max_days=int(notice.get("preferred_max_days", 30)),
            raises_bar_at_days=int(notice.get("raises_bar_at_days", 30)),
        )

        profile_text = str(raw.get("profile_text", "")).strip()

        return cls(
            positive_signals=positive_signals,
            negative_signals=negative_signals,
            location_pref=location_pref,
            notice_pref=notice_pref,
            profile_text=profile_text,
        )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, or an empty mapping if absent/invalid."""
    if isinstance(value, Mapping):
        return value
    return {}


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a YAML list (or scalar) into a tuple of trimmed strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),)
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)
