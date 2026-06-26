# =============================================================================
# Dockerfile — AI Candidate Ranking (self-contained sandbox image)
#
# Fully local, CPU-only, OFFLINE candidate ranker. The image bakes in the pinned
# dependencies, the project source, the local embedding-model cache (./models),
# and a 50-candidate sample, so `docker run` produces a ranked CSV end-to-end
# with NO network access at run time.
#
# -----------------------------------------------------------------------------
# Quick start (after `docker pull <your-image>`)
# -----------------------------------------------------------------------------
#   # A) Zero-arg run on the baked-in 50-candidate sample, copy result out:
#   docker run --rm -v "${PWD}:/work" <your-image> \
#       --candidates sample_candidates.json --out /work/submission.csv
#
#   # B) Rank YOUR own small sample (<=100 candidates) mounted from the host:
#   docker run --rm -v "${PWD}:/work" <your-image> \
#       --candidates /work/my_sample.jsonl --out /work/submission.csv
#
#   # C) See all CLI flags:
#   docker run --rm <your-image> --help
#
# (PowerShell uses the same `-v "${PWD}:/work"` form; bash uses `-v "$PWD":/work`.)
#
# -----------------------------------------------------------------------------
# Offline guarantee
# -----------------------------------------------------------------------------
# The container sets HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and the embedding
# model weights are baked into ./models, so NO download happens at run time.
# For a <=100-candidate sample the pipeline embeds on the fly from that cache
# (the large 100K precomputed `artifacts/` are intentionally excluded from this
# sandbox image to keep it small — they live in the GitHub repo for the full
# Stage-3 reproduction).
# =============================================================================

# Pinned slim CPU base image (no GPU components).
FROM python:3.11-slim

# Keep Python output unbuffered and skip .pyc files for a lean image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Enforce fully-offline behaviour for HuggingFace / transformers at run time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Install pinned runtime dependencies first for better layer caching.
# Uses the slim Docker requirements (CPU-only torch, no gradio/test deps) with
# generous timeout/retries so a slow mirror doesn't fail the build.
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir --timeout 1200 --retries 10 -r requirements-docker.txt

# Copy the project source, the offline-encoded job profile, the baked-in model
# cache (./models), and the bundled sample. The .dockerignore excludes the large
# 100K artifacts/, the Gradio app, tests, and VCS/IDE noise.
#   - ranking/                 (pipeline package)
#   - rank.py                  (CLI entry point)
#   - precompute_embeddings.py (offline run-once artifact builder)
#   - job_profile.yaml         (offline-encoded job profile)
#   - job_description.md       (source job description)
#   - models/                  (local sentence-transformers cache -> offline)
#   - sample_candidates.json   (50-candidate demo sample)
COPY . .

# Fail the build early if the embedding model cache was not included, since the
# offline runtime depends on it.
RUN test -d models || (echo "ERROR: ./models cache missing from build context; run precompute or cache the model first." && exit 1)

# Run the ranker directly: `docker run IMAGE [--candidates ... --out ...]`.
ENTRYPOINT ["python", "rank.py"]

# Default (no args): rank the baked-in sample to /app/submission.csv so a bare
# `docker run <image>` works out of the box. Override with --candidates/--out.
CMD ["--candidates", "sample_candidates.json", "--out", "submission.csv"]
