# Implementation Plan: AI Candidate Ranking (Redrob Challenge)

## Overview

This plan implements the fully local, CPU-only, offline candidate ranking pipeline described in the design. Work is built bottom-up: data models and config first, then the pure deterministic core (ingestion, job profile, features, honeypot, hybrid scoring, behavioral modifier, ranking, reasoning, CSV writing), then the embedding layer and offline precompute, then the orchestrating `RankingPipeline` and `rank.py` CLI wired together, and finally the deliverables (README, Docker recipe, `submission_metadata.yaml`, methodology deck) and the performance/validator conformance tests.

Each correctness property from the design is its own optional property-based test sub-task using `hypothesis`, tagged `# Feature: ai-candidate-ranking, Property N: ...`, placed next to the code it validates so regressions surface early. The embedding model is replaced with fixed deterministic vectors in pure tests.

Implementation language: **Python** (as specified in the design).

## Tasks

- [x] 1. Set up project scaffold and pinned dependencies
  - Create the package layout: `ranking/` package with module stubs (`models.py`, `config.py`, `loader.py`, `job_profile.py`, `embedding.py`, `store.py`, `features.py`, `honeypot.py`, `scorer.py`, `behavioral.py`, `ranker.py`, `reasoning.py`, `writer.py`, `pipeline.py`, `errors.py`), the `rank.py` CLI entry stub at repo root, the `precompute_embeddings.py` stub at repo root, and a `tests/` directory with `conftest.py`
  - Create a pinned `requirements.txt` (exact `==` versions for `sentence-transformers`, `numpy`, `faiss-cpu` (optional), `hypothesis`, `pytest`, `pyyaml`)
  - Define the typed exception hierarchy in `errors.py`: `DatasetAccessError`, `NoValidCandidatesError`, `MissingArtifactError`, `WeightValidationError`, `OutputWriteError`
  - _Requirements: 13.1, 13.3_

- [x] 2. Implement core data models
  - [x] 2.1 Implement frozen dataclasses in `models.py`
    - `CareerEntry`, `EducationEntry`, `Skill`, `Profile`, `RedrobSignals`, `CandidateRecord`, `DimensionScores`, `HoneypotResult`, `ScoredCandidate`, `RankedCandidate`, `SkipWarning`, `LoadResult`, `ArtifactPaths`
    - Add a `from_json(obj)` constructor on `CandidateRecord` that maps the real `candidate_schema.json` fields, tolerates missing optional fields, and treats `-1` sentinels (`github_activity_score`, `offer_acceptance_rate`) as "unknown"
    - _Requirements: 3.1_
  - [x]* 2.2 Write unit tests for model construction from schema
    - Build a `CandidateRecord` from a `sample_candidates.json` record; assert optional-field defaults and `-1` sentinel handling
    - _Requirements: 3.1_

- [x] 3. Implement scoring configuration
  - [x] 3.1 Implement `ScoringConfig` in `config.py`
    - Frozen dataclass with documented default weights (`w_semantic` 0.35, `w_skills_title` 0.25, `w_experience` 0.15, `w_trajectory` 0.20, `w_education` 0.05, `behavioral_strength` 0.15, `honeypot_penalty` 0.0)
    - `load(path)` merges YAML/JSON overrides over defaults; `validate()` raises `WeightValidationError` on any negative weight and on all-dimension-weights-zero
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  - [x]* 3.2 Write property test for weight validation
    - **Property 22: Weight validation rejects negative and all-zero configurations**
    - **Validates: Requirements 12.4, 12.5**
  - [x]* 3.3 Write unit tests for default config
    - Assert defaults applied when no config supplied and that all weight/behavioral fields are exposed
    - _Requirements: 12.1, 12.2_

- [x] 4. Implement streaming candidate ingestion
  - [x] 4.1 Implement `CandidateLoader` in `loader.py`
    - Detect format by extension (`.gz` gzip text stream, `.json` array, else JSON Lines); stream line-by-line via `iter_records` for bounded memory
    - Validate parseability and `candidate_id` against `^CAND_[0-9]{7}$`; skip malformed rows collecting `SkipWarning(line_number, reason)` and continue
    - Raise `DatasetAccessError` on unreadable file; raise `NoValidCandidatesError` when zero valid records remain; return `LoadResult`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  - [x]* 4.2 Write property test for ingestion partitioning
    - **Property 1: Ingestion partitions every line into valid or skipped**
    - **Validates: Requirements 1.1, 1.5**
  - [x]* 4.3 Write property test for format equivalence
    - **Property 2: Ingestion is format-equivalent across jsonl, gzip, and json array**
    - **Validates: Requirements 1.1, 1.2, 1.3**
  - [x]* 4.4 Write unit tests for ingestion error paths
    - Unreadable file raises `DatasetAccessError` (1.6); all-malformed file raises `NoValidCandidatesError` (1.7)
    - _Requirements: 1.6, 1.7_

- [x] 5. Implement offline-encoded job profile
  - [x] 5.1 Implement `JobProfile` and the committed profile artifact
    - Create `job_description.md` (text derived from the challenge `.docx`) and a hand-encoded `job_profile.yaml` capturing positive signals, negative signals, location preference, and notice preference
    - Implement `JobProfile.load(path)` and the `PositiveSignals`/`NegativeSignals`/`LocationPref`/`NoticePref` holders plus canonical `profile_text`; no network access
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x]* 5.2 Write unit tests for job profile content
    - Assert positive signals (production retrieval/ranking/embeddings, Python, NDCG/MRR/MAP), negative signals (keyword-stuffer titles, consulting-only firms, pure research, CV/speech/robotics-only, title-chasing hops), and location/notice preferences are present
    - _Requirements: 2.2, 2.3, 2.4_

- [x] 6. Implement embedding model, store, and offline precompute
  - [x] 6.1 Implement `EmbeddingModel` wrapper in `embedding.py`
    - Load sentence-transformers (`BAAI/bge-small-en-v1.5`) from local cache with `local_files_only=True` and `HF_HUB_OFFLINE=1` enforced; `embed`, `embed_batch`, and `candidate_text(rec)` producing L2-normalized CPU vectors
    - _Requirements: 2.5, 10.2, 10.5, 10.6_
  - [x] 6.2 Implement `EmbeddingStore` in `store.py`
    - `load(emb_path, id_path, job_path)` from local disk; raise `MissingArtifactError` naming the artifact and exact `python precompute_embeddings.py ...` build command when absent
    - `similarity(candidate_id)` and `similarity_for(embedding)` mapping cosine of L2-normalized vectors via `(cos + 1) / 2` into [0,1]
    - _Requirements: 3.2, 11.1, 11.5_
  - [x]* 6.3 Write property test for similarity bounds
    - **Property 3: Cosine similarity is mapped into [0,1]**
    - **Validates: Requirements 3.2**
  - [x]* 6.4 Write unit test for missing-artifact error
    - Missing precomputed artifact raises `MissingArtifactError` with the build hint
    - _Requirements: 11.5_
  - [x] 6.5 Implement `precompute_embeddings.py` (offline, run once)
    - Stream candidates, embed each candidate text and the job profile with the local model on CPU, write committed `embeddings.npy`, `id_order.json`, and `job_embedding.npy`; uses only local models, no hosted services
    - _Requirements: 11.1, 11.2, 11.4_

- [x] 7. Implement feature extraction and anti-keyword-stuffing logic
  - [x] 7.1 Implement `FeatureExtractor` in `features.py`
    - Compute normalized [0,1] dimensions: `semantic` (from store), `skills_title` (title-aware `0.7*title_align + 0.3*skill_align*trust`), `experience` (5–9yr soft band), `trajectory` (product-vs-services, anti-hopping), `education` (tier-based)
    - Implement the per-skill trust multiplier (`proficiency_weight * duration_factor * (0.5 + 0.5*endorsement_factor)`) so expert/zero-duration skills contribute nothing; classify keyword stuffers from profile/career evidence only (no hardcoded ids)
    - _Requirements: 3.1, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4_
  - [x]* 7.2 Write property test for keyword-stuffer ordering
    - **Property 6: A keyword-stuffer ranks below an otherwise-genuine equivalent**
    - **Validates: Requirements 4.1, 4.3**
  - [x]* 7.3 Write property test for zero-duration skill discounting
    - **Property 7: Expert/advanced skills with zero duration are discounted**
    - **Validates: Requirements 4.2**
  - [x]* 7.4 Write property test for negative-signal reduction
    - **Property 8: Negative signals reduce Fit_Score**
    - **Validates: Requirements 3.4**
  - [x]* 7.5 Write property test for product-experience reward
    - **Property 9: Product ranking/search experience does not lower fit**
    - **Validates: Requirements 3.3**

- [x] 8. Implement honeypot / inconsistency detection
  - [x] 8.1 Implement `HoneypotDetector` in `honeypot.py`
    - Run the five consistency checks (experience exceeds career span, duration sum mismatch, expert/advanced-with-zero-duration cluster, skill duration exceeds total experience, impossible date ordering); return `HoneypotResult(is_honeypot, reasons)` using profile data only, no hardcoded ids
    - _Requirements: 6.1, 6.2, 6.3, 6.6_
  - [x]* 8.2 Write property test for honeypot flagging
    - **Property 10: Impossible profiles are flagged as honeypots**
    - **Validates: Requirements 6.1, 6.2, 6.3**
  - [x]* 8.3 Write unit test for no candidate-id allowlist
    - Assert the detector exposes no candidate-id allowlist parameter
    - _Requirements: 6.6_

- [x] 9. Implement hybrid scorer
  - [x] 9.1 Implement `HybridScorer.fit_score` in `scorer.py`
    - Normalized weighted sum of dimensions over the sum of weights, guaranteeing [0,1]; pure and deterministic
    - _Requirements: 3.1, 3.4, 3.5, 3.6_
  - [x]* 9.2 Write property test for Fit_Score bounds and formula
    - **Property 4: Fit_Score is bounded in [0,1] and matches the weighted formula**
    - **Validates: Requirements 3.1, 3.4, 12.3**
  - [x]* 9.3 Write property test for total deterministic scoring
    - **Property 5: Scoring is total and deterministic**
    - **Validates: Requirements 3.5, 3.6**

- [x] 10. Implement behavioral modifier
  - [x] 10.1 Implement `BehavioralModifier` in `behavioral.py`
    - Bounded `modifier` in `[1-s, 1+s]` from recency vs pool-latest-active, `recruiter_response_rate`, `notice_period_days`, and `open_to_work_flag`; `final_score = clamp01(fit * modifier)` then honeypot penalty (default 0.0 excludes flagged records)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.4_
  - [x]* 10.2 Write property test for modifier bounds and non-dominance
    - **Property 12: The behavioral modifier is bounded and never dominates**
    - **Validates: Requirements 5.1, 5.5**
  - [x]* 10.3 Write property test for modifier monotonicity
    - **Property 13: The modifier is monotonic in availability signals**
    - **Validates: Requirements 5.2, 5.3, 5.4**

- [x] 11. Checkpoint - core scoring complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement deterministic ranking
  - [x] 12.1 Implement `Ranker.rank` in `ranker.py`
    - Sort by `(-final_score, candidate_id)`, take top 100, assign unique ranks 1..n; deterministic and total
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [x]* 12.2 Write property test for shortlist size and unique ranks
    - **Property 14: The shortlist has exactly 100 rows with unique ranks 1..100**
    - **Validates: Requirements 7.1, 7.2, 9.3, 9.5**
  - [x]* 12.3 Write property test for score monotonicity by rank
    - **Property 15: Final_Score is monotonically non-increasing with rank**
    - **Validates: Requirements 7.3, 9.6**
  - [x]* 12.4 Write property test for tie-breaking
    - **Property 16: Ties break by candidate_id ascending**
    - **Validates: Requirements 7.4, 9.6**
  - [x]* 12.5 Write property test for ranking determinism
    - **Property 17: Ranking is deterministic**
    - **Validates: Requirements 7.5**

- [x] 13. Implement offline reasoning generation
  - [x] 13.1 Implement `ReasoningGenerator.generate` in `reasoning.py`
    - Fact-grounded assembler using only `CandidateRecord` facts connected to Job_Profile requirements; concern-acknowledgement clause for notice >=30d / consulting-only / inactivity >=6mo; varied via deterministic hash seeded by `candidate_id`; tone bucketed by rank band; bounded to 1–2 sentences; fully offline
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  - [x]* 13.2 Write property test for reasoning grounding
    - **Property 18: Reasoning references only facts present in the record**
    - **Validates: Requirements 8.2, 8.5**
  - [x]* 13.3 Write property test for sentence bounds and concern acknowledgement
    - **Property 19: Reasoning is 1–2 sentences and acknowledges present concerns**
    - **Validates: Requirements 8.1, 8.4**
  - [x]* 13.4 Write property test for reasoning variation
    - **Property 20: Reasoning is varied, not templated**
    - **Validates: Requirements 8.6**

- [x] 14. Implement submission CSV writer
  - [x] 14.1 Implement `SubmissionWriter.write` in `writer.py`
    - UTF-8 `.csv` with exact header `candidate_id,rank,score,reasoning` and exactly 100 data rows in column order; raise `OutputWriteError` on permission/disk failure
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_
  - [x]* 14.2 Write unit tests for CSV format and write errors
    - Assert exact header, UTF-8 encoding, `.csv` extension; unwritable path raises `OutputWriteError`
    - _Requirements: 9.1, 9.2, 9.7_

- [x] 15. Wire pipeline and CLI together
  - [x] 15.1 Implement `RankingPipeline.run` in `pipeline.py`
    - Streaming end-to-end orchestration: load -> job profile + embedding store -> feature extraction -> honeypot -> hybrid score -> behavioral modifier -> rank top 100 -> reasoning -> write CSV; embed on the fly for the small sample path; bound in-memory state; return `RunReport`
    - _Requirements: 10.1, 10.3, 10.4, 10.6_
  - [x] 15.2 Implement `rank.py` CLI entry
    - Parse `--candidates`, `--out`, optional `--config`; construct `ScoringConfig` and `ArtifactPaths`; invoke `RankingPipeline.run`; surface typed errors with actionable messages; supports the Reproduce_Command
    - _Requirements: 13.2_
  - [x]* 15.3 Write property test for honeypot exclusion from top 100
    - **Property 11: No honeypot reaches the top 100**
    - **Validates: Requirements 6.4, 6.5**
  - [x]* 15.4 Write validator-conformance property test
    - **Property 21: The generated CSV passes the challenge validator**
    - Generate a pool of >=100 valid candidates, run the full pipeline to a temp `submission.csv`, then import and call the committed `validate_submission.py` and assert zero errors
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6**
  - [x]* 15.5 Write offline-assurance integration test
    - Run the pipeline with network sockets blocked (monkeypatch `socket.socket`) over a small sample and assert completion, proving no network dependency during ranking
    - _Requirements: 10.5, 10.6, 11.1_
  - [x]* 15.6 Write reproduce-command integration test
    - Run `rank.py` end-to-end on `sample_candidates.json` and assert a valid `submission.csv` is produced
    - _Requirements: 13.2, 13.7, 13.8_

- [x] 16. Checkpoint - end-to-end pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Produce deliverable artifacts
  - [x] 17.1 Write README with single reproduce command, precompute docs, and Docker recipe
    - Document `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`, the offline precompute process and approximate runtime, and a self-contained `docker run` recipe that runs the pipeline on a sample of at most 100 candidates within the Compute_Budget
    - _Requirements: 11.2, 11.3, 13.1, 13.2, 13.3, 13.8_
  - [x] 17.2 Create `submission_metadata.yaml` at repo root
    - Declare team identity/contact, repo URL, sandbox link, reproduce command, compute environment with `uses_gpu_for_inference: false` and `has_network_during_ranking: false`, precompute use + runtime, AI tools usage summary, methodology summary, and required declarations
    - _Requirements: 13.4_
  - [x] 17.3 Create `Dockerfile` for the sandbox/demo recipe
    - Pin the base image, install pinned `requirements.txt`, copy source + committed artifacts, and default to the sample reproduce command within the Compute_Budget
    - _Requirements: 13.7, 13.8_
  - [x] 17.4 Author the methodology presentation deck content (PDF source)
    - Write the deck content (markdown/source) explaining the genuine-fit approach, hybrid scoring, honeypot detection, and offline/compute conformance, to be converted to PDF
    - _Requirements: 13.6_

- [x] 18. Write performance and deliverable-presence tests
  - [x]* 18.1 Write 100K performance/smoke test
    - Generate a synthetic 100,000-record `candidates.jsonl` plus matching precomputed embeddings, run `rank.py`, and assert wall-clock < 5 min, peak RSS < 16 GB on CPU, and artifact/temp disk usage < 5 GB
    - _Requirements: 1.4, 10.1, 10.2, 10.3, 10.4_
  - [x]* 18.2 Write deliverable-presence test
    - Assert `requirements.txt` is pinned, `submission_metadata.yaml` exists with the required offline/GPU flags, the precompute script (or committed artifacts) exists, and the README documents the reproduce command and `docker run` recipe
    - _Requirements: 11.2, 11.3, 13.1, 13.4_

- [x] 19. Final checkpoint - full suite and deliverables verified
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each property-based test maps to exactly one of the 22 design properties, is tagged `# Feature: ai-candidate-ranking, Property N: ...`, runs `>= 100` generated examples via `hypothesis`, and replaces the embedding model with fixed deterministic vectors.
- Property tests carry input-coverage weight (scoring, ranking, honeypot, ingestion, reasoning); unit tests cover specific error types and static content; integration/smoke tests cover offline and compute-budget guarantees.
- Each task references the specific requirements clauses it implements for traceability.
- Checkpoints ensure incremental validation at natural breaks.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1", "4.1", "5.1", "6.1", "8.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "4.2", "4.3", "4.4", "5.2", "6.2", "6.5", "7.1", "8.2", "8.3"] },
    { "id": 4, "tasks": ["6.3", "6.4", "7.2", "7.3", "7.4", "7.5", "9.1"] },
    { "id": 5, "tasks": ["9.2", "9.3", "10.1", "12.1"] },
    { "id": 6, "tasks": ["10.2", "10.3", "12.2", "12.3", "12.4", "12.5", "13.1", "14.1"] },
    { "id": 7, "tasks": ["13.2", "13.3", "13.4", "14.2", "15.1"] },
    { "id": 8, "tasks": ["15.2"] },
    { "id": 9, "tasks": ["15.3", "15.4", "15.5", "15.6", "17.1", "17.2", "17.3", "17.4"] },
    { "id": 10, "tasks": ["18.1", "18.2"] }
  ]
}
```
