# Redrob AI Candidate Ranker

A fully local, CPU-only, offline candidate ranker for the Redrob "India Runs"
Data & AI challenge. Given a dataset of candidate profiles, it produces a
deterministic top-100 ranking for a Senior AI Engineer role. The approach is
deliberately **anti-keyword-matching**: instead of rewarding profiles that
simply stuff the right buzzwords, it combines local semantic embeddings
(`BAAI/bge-small-en-v1.5`) with a hybrid feature score that reads skills in the
context of job titles, applies an anti-keyword-stuffing trust multiplier,
rewards a genuine 5–9 year experience band, weighs a product-vs-services career
trajectory and education, layers a bounded behavioral modifier derived from
`redrob_signals`, and runs honeypot/consistency detection to discount fabricated
or inconsistent profiles. Ranking is deterministic (ties broken by
`candidate_id`) and every row carries a short, offline, grounded explanation.

## Setup

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Because ranking runs **offline**, the embedding model weights must already be
cached on local disk before you run. Cache the
`BAAI/bge-small-en-v1.5` weights once into a local directory (the precompute and
ranking steps default to `./models`) while you have network access. After that,
no network access is needed or attempted during precompute or ranking.

## Pre-computation (run once, may exceed the 5-minute budget)

Embedding 100K candidate texts on CPU is the slowest part of the system, so it
is done once, up front, and cached as artifacts:

```bash
python precompute_embeddings.py --candidates ./candidates.jsonl
```

This writes three artifacts under `./artifacts/`:

- `embeddings.npy` — float32 matrix, one L2-normalized row per candidate.
- `id_order.json` — JSON list of `candidate_id` values, parallel to the matrix rows.
- `job_embedding.npy` — float32 L2-normalized vector for the job profile.

Useful flags (all have sensible defaults):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--candidates` | `./candidates.jsonl` | Input dataset (`.jsonl`, `.jsonl.gz`, or `.json` array). |
| `--model` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers model name. |
| `--cache-dir` | `./models` | Local model cache (no download occurs). |
| `--out-emb` | `./artifacts/embeddings.npy` | Candidate embedding matrix output. |
| `--out-ids` | `./artifacts/id_order.json` | Candidate-id order output. |
| `--out-job` | `./artifacts/job_embedding.npy` | Job profile embedding output. |
| `--job-profile` | `./job_profile.yaml` | Offline-encoded job profile. |
| `--batch-size` | `64` | CPU embedding batch size. |

The ranking step then loads these artifacts instead of re-embedding, which is
what lets it stay within the compute budget. Approximate precompute runtime is
dominated by embedding 100K texts on CPU.

## Run / Reproduce

The single reproduce command:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This loads the precomputed artifacts, scores and ranks all candidates, and
writes the top-100 submission CSV with columns
`candidate_id,rank,score,reasoning`.

Optional flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--config` | _(none)_ | Path to a scoring config YAML/JSON to override weights. |
| `--artifacts-dir` | `./artifacts` | Directory holding the precomputed artifacts. |
| `--job-profile` | `./job_profile.yaml` | Committed offline job profile. |
| `--top-n` | `100` | Maximum number of ranked rows to write. |

If no precomputed artifacts are present in `--artifacts-dir`, the pipeline falls
back to embedding candidates on the fly. That is fine for small samples but will
exceed the budget on the full 100K dataset, so precompute first for the real run.

## Validate

Check that the produced CSV conforms to the submission spec:

```bash
python validate_submission.py submission.csv
```

## Docker recipe (self-contained sandbox)

For a self-contained, reproducible sandbox, build the image from the
[`Dockerfile`](./Dockerfile) and run the ranker on a small sample within budget:

```bash
docker build -t redrob-ranker .
docker run --rm -v ${PWD}:/work redrob-ranker \
  --candidates /work/sample_candidates.json \
  --out /work/submission.csv
```

The `-v ${PWD}:/work` mount makes the working directory available inside the
container so the input sample is read and the resulting `submission.csv` is
written back to your host. The container runs fully offline.

## Architecture overview

The pipeline is a straight, deterministic flow:

1. **Load** candidate records (streaming, schema-tolerant).
2. **Embed** candidate text + the job profile with the local CPU model (or load
   precomputed artifacts).
3. **Store** vectors and compute cosine similarity (FAISS when available, with a
   pure-numpy fallback).
4. **Feature scoring** — semantic similarity, title-aware skill matching with an
   anti-keyword-stuffing trust multiplier, a 5–9 year experience band,
   product-vs-services trajectory, and education.
5. **Honeypot / consistency detection** — discount fabricated or internally
   inconsistent profiles.
6. **Behavioral modifier** — a bounded adjustment from `redrob_signals`.
7. **Rank** deterministically, breaking ties by `candidate_id`.
8. **Reasoning** — generate a short, offline, grounded explanation per candidate.
9. **Write** the top-N submission CSV.

### Module map

| Module | Responsibility |
| --- | --- |
| `ranking/loader.py` | Stream and parse candidate records. |
| `ranking/job_profile.py` | Load the offline-encoded job profile. |
| `ranking/embedding.py` | Local CPU sentence-transformers embedding model. |
| `ranking/store.py` | Vector store + cosine similarity (FAISS / numpy fallback). |
| `ranking/features.py` | Hybrid feature scoring (skills, experience, trajectory, education). |
| `ranking/honeypot.py` | Honeypot / consistency detection. |
| `ranking/scorer.py` | Combine features into the final score. |
| `ranking/behavioral.py` | Bounded behavioral modifier from `redrob_signals`. |
| `ranking/ranker.py` | Deterministic ranking and tie-breaking. |
| `ranking/reasoning.py` | Offline grounded reasoning strings. |
| `ranking/writer.py` | Submission CSV writer. |
| `ranking/pipeline.py` | End-to-end orchestration. |
| `ranking/config.py` | Scoring configuration loading and validation. |

## Testing

```bash
python -m pytest
```

The suite includes both unit tests and property-based tests (via
[Hypothesis](https://hypothesis.readthedocs.io/)) covering config, loading, the
job profile, and core models.

## Compute constraints & offline guarantee

- **CPU only** — no GPU is used or required at any stage.
- **Offline** — no network access is performed during precompute or ranking. The
  embedding model enforces offline mode (`HF_HUB_OFFLINE=1` /
  `local_files_only=True`) and reads weights from the local cache.
- **Budget** — the ranking step (`rank.py`) completes in **≤ 5 minutes for 100K
  candidates** within **≤ 16 GB RAM** when the precomputed artifacts are present.
- **Deterministic** — the same inputs always produce the same ranking, with ties
  broken by `candidate_id`.
