<!-- Convert to PDF with Marp: `marp methodology_deck.md --pdf`  •  or Pandoc: `pandoc methodology_deck.md -o methodology_deck.pdf`  •  or print-to-PDF from any Markdown viewer. Slides are separated by `---`. -->

---
marp: true
paginate: true
title: Intelligent Candidate Discovery & Ranking — Redrob Challenge
---

# Intelligent Candidate Discovery & Ranking
## Redrob Challenge — Senior AI Engineer, Founding Team

A **fully local, CPU-only, offline** ranker that scores *genuine* fit — semantic similarity fused with title-aware feature logic, behavioral availability, and honeypot detection — to produce a deterministic, validator-clean top-100.

<!-- One-line approach summary: hybrid scoring beats keyword stuffing and traps, runs in <=5 min for 100K on CPU with no network. -->

---

# The Problem

- **Keyword filters miss genuine fit.** Real engineers who shipped ranking/search may not list the buzzwords; stuffers list them all.
- **The challenge punishes keyword stuffing.** The provided `sample_submission.csv` is a deliberate anti-pattern — it ranks an "HR Manager with 9 AI core skills" at the top.
- **Honeypots are planted.** Internally inconsistent profiles (impossible date math, "expert / 0 months") exist to trap naive rankers.
- **Behavioral availability matters.** Notice period, recency, responsiveness, and openness separate otherwise-equal candidates.

**Scoring metric (hidden ground truth):**
`NDCG@10 = 0.50 · NDCG@50 = 0.30 · MAP = 0.15 · Precision@10 = 0.05`
→ Two thirds of the weight sits on the top 10 / top 50 ordering.

**Disqualifiers:**
- ≥ **10% honeypots** in the top 100.
- Breaching compute: GPU use, network during ranking, **> 5 min / 100K**, or > 16 GB RAM / > 5 GB disk.

---

# Key Insight: Says vs. Means

The job description's literal words and the role's true needs are different things.

- A **"Marketing Manager"** padded with `embeddings, retrieval, ranking, NDCG, RAG` is **NOT a fit** — aspiration, not experience.
- An **unbuzzwordy engineer** who shipped a **search / ranking / recommendation system at a product company** **IS a fit** — even with zero AI buzzwords in skills or headline.

We rank for what the role actually needs: **production retrieval / ranking / embeddings at product companies, strong Python, and evaluation-framework experience (NDCG / MRR / MAP)** — while down-weighting stuffers, negative-signal profiles, and traps.

---

# Architecture

**Offline precompute (run once, committed artifacts):** local BGE embeddings → `embeddings.npy` + `id_order.json` + `job_profile_embedding.npy`.
**Online ranking (`rank.py`, CPU-only, offline, ≤ 5 min / 100K):** loads artifacts, never re-embeds, never touches the network.

```
                OFFLINE (once)                         ONLINE (rank.py)
   job_description.md                          candidates.jsonl(.gz) / .json
          │                                              │
   hand-encoded JobProfile                          [1] ingest (stream, skip bad rows)
          │                                              │
   precompute_embeddings.py  ── local BGE ──┐       [2] job profile + embedding
          │                                 │           │
   embeddings.npy / id_order.json ──────────┼──────►[3] semantic similarity  (cosine→[0,1])
   job_profile_embedding.npy                │           │
                                            └──────►[4] feature extraction (5 dims)
                                                        │
                                                    [5] honeypot detection
                                                        │
                                                    [6] hybrid Fit_Score
                                                        │
                                                    [7] behavioral modifier → Final_Score (+honeypot penalty)
                                                        │
                                                    [8] rank top 100 (tie-break id asc)
                                                        │
                                                    [9] grounded reasoning
                                                        │
                                                   [10] submission.csv
```

**Data flow:** ingest → features → semantic similarity → hybrid score → behavioral modifier → honeypot penalty → rank top 100 → reasoning → CSV.

---

# Why Hybrid Beats Embedding-Only

Pure embedding-similarity ranking fails this challenge in three ways — each corrected by the hybrid:

1. **Rewards stuffers.** A padded profile reads *semantically close* to the JD even with no real experience. Cosine can't tell aspiration from experience.
   → **Title-aware feature layer + anti-stuffing trust multiplier** fixes it.
2. **Blind to inconsistency.** Honeypots embed just like real profiles.
   → **Structured honeypot consistency checks** flag and exclude them.
3. **Ignores availability & trajectory.** Two identical-reading profiles differ in notice period, recency, product-vs-services background, job-hopping.
   → **Structured feature dimensions + bounded behavioral modifier** separate them — exactly the fine-grained ordering NDCG@10 rewards.

Recall of semantic similarity **+** precision of explicit feature logic = better NDCG and MAP on a genuine-fit ground truth.

---

# Hybrid Scoring — Five Dimensions

| Dimension | How it's scored (normalized to [0,1]) |
|---|---|
| **Semantic** | `(cosine(job, cand) + 1) / 2` on L2-normalized vectors |
| **Skills + title** | `clamp01(0.7·title_align + 0.3·skill_align·trust)` — titles weighted above raw skill count |
| **Experience** | `1.0` in the 5–9 yr band; soft edges below (`/3`) and above (`/6`) |
| **Trajectory** | product-vs-services; reward shipping ranking/search at product cos; penalize consulting-only & ~1.5 yr title-hops |
| **Education** | light, tier-based: tier_1→1.0, tier_2→0.7, tier_3→0.4, tier_4/unknown→0.2 |

**Anti-stuffing trust multiplier** (per skill):
`skill_trust = proficiency_weight · min(1, duration/24) · (0.5 + 0.5·min(1, endorsements/10))`
→ **expert + 0 months ⇒ duration_factor = 0 ⇒ trust = 0** (contributes nothing).

**Fit_Score** = convex combination (weights default `0.35 / 0.25 / 0.15 / 0.20 / 0.05`), normalized → guaranteed in **[0,1]**.

**Behavioral modifier** ∈ `[1−s, 1+s]` (default `s=0.15`): `(1−s) + 2s·signal_mean`.
**Final_Score** = `clamp01(Fit_Score · modifier)`, then honeypot penalty.

---

# Honeypot Detection

Five **consistency checks** on profile data only (no hardcoded candidate ids). Any trigger flags the record:

1. **Experience exceeds career span** — `years_of_experience` > span from earliest start to latest end (+ tolerance).
2. **Duration sum mismatch** — `sum(role durations)/12` > `years_of_experience` (+ tolerance).
3. **Expert-with-zero-duration cluster** — ≥ 2 skills at expert/advanced with `duration_months == 0`.
4. **Skill predates career** — any skill `duration_months/12` > `years_of_experience` (+ tolerance).
5. **Impossible date ordering** — `end_date < start_date`, or dates in the future vs latest pool activity.

With the default **honeypot penalty = 0.0**, flagged records score 0 and **cannot enter the top 100** — targeting **zero honeypots**, far under the 10% disqualification threshold.

---

# Reasoning Generation

A fully **offline, deterministic, fact-grounded sentence assembler** — no hosted LLM.

- **Grounded:** uses only facts present in the record — years of experience, current title, named skills, employers, specific `redrob_signals` (notice period, last active, response rate). Never emits a skill, employer, or attribute not in the record.
- **Connected to the role:** maps facts to specific Job_Profile requirements (e.g. "ranking systems at a product company" → production-retrieval positive signal).
- **Concern-acknowledging:** when a record carries a notable concern (notice ≥ 30 days, consulting-only, inactivity ≥ 6 months, weak honeypot-adjacent inconsistency), the reasoning adds a concession clause.
- **Varied, not templated:** selects from a fact pool + sentence structures keyed off a deterministic hash seeded by `candidate_id`; tone bucketed by rank band (1–10 / 11–50 / 51–100).
- **Bounded:** 1–2 sentences. Deterministic — same candidate always gets the same text.

---

# Compute & Reproducibility

- **CPU-only**, no GPU required.
- **Zero network** during ranking — model loaded with `local_files_only=True` / `HF_HUB_OFFLINE=1`, so any download attempt raises immediately.
- **≤ 5 minutes wall-clock for 100,000 candidates**; ≤ 16 GB RAM; ≤ 5 GB intermediate disk.
- **Streaming ingestion** keeps peak memory bounded for the 100K path.
- **Single reproduce command:**
  ```
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  ```
- **Committed artifacts** (or a documented build script): `embeddings.npy`, `id_order.json`, `job_profile_embedding.npy`, built once by `precompute_embeddings.py` using only local models.
- **Docker sandbox** runs the pipeline on a ≤ 100-candidate sample end-to-end within budget.

---

# Validation

- **Passes `validate_submission.py`** with zero errors — verified by a property test that runs the full pipeline and calls the committed challenge validator directly.
- **Exact format:** UTF-8 CSV, header `candidate_id,rank,score,reasoning`, exactly 100 data rows.
- **Well-formed ranking:** ids match `^CAND_[0-9]{7}$`, no duplicate id or rank, ranks `1..100`, `score` monotonically non-increasing, ties ordered by `candidate_id` ascending.
- **Deterministic:** identical input + identical `ScoringConfig` ⇒ byte-identical output.
- **Honeypot-safe:** default penalty zeros flagged records ⇒ zero honeypots in the shortlist.

Backed by 22 property-based tests (Hypothesis) plus focused unit, integration (network-blocked), and smoke/perf tests.

---

# Trade-offs & Future Work

- **Hand-encoded Job_Profile** is transparent and fast, but tuned by hand. *Future:* a small **learning-to-rank** model trained on labeled fit data to learn dimension weights instead of setting them.
- **Single job-profile embedding** captures the role coarsely. *Future:* **richer job-profile encoding** — multi-aspect embeddings (skills, trajectory, seniority) for finer semantic matching.
- **Scores are relative, not calibrated.** *Future:* **score calibration** so the `score` column reflects a meaningful probability of fit, improving interpretability.
- **Honeypot tolerance** is a fixed heuristic; could be learned from flagged-vs-clean distributions.

These are upgrades on a deliberately simple, auditable, deterministic baseline that already satisfies every challenge constraint.

---

# Closing

A hybrid ranker that reasons about **genuine fit, not keywords** —

- Local BGE semantic similarity **+** title-aware feature logic **+** behavioral availability **+** honeypot defense.
- Fully **local, CPU-only, offline**; **≤ 5 min / 100K**; one reproduce command.
- **Deterministic**, **validator-clean**, **honeypot-safe** top-100.

Built to win on NDCG@10 — by ranking the engineers who actually shipped, and demoting the ones who only said they did.

**Thank you.**
