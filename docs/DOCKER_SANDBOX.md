# Publishing the Docker Sandbox Image

This gives you the **`docker pull` + `docker run` link to a public registry
image** that the challenge accepts as a sandbox (submission_spec §10.5).

The image is **self-contained**: it bakes in the code, the offline embedding
model cache (`./models`), and a 50-candidate sample, so anyone can run it with
no network and get a ranked CSV.

You need: Docker Desktop running + a free **Docker Hub** account (or GitHub
Container Registry). ~10 minutes (the image is ~2–3 GB due to torch + model).

> I couldn't build/push this for you — pushing requires *your* registry login,
> and the resulting `pull` link lives under your namespace. The commands below
> are everything you need.

---

## Prerequisites (one time)

1. Install / start **Docker Desktop** (the daemon must be running —
   `docker info` should succeed).
2. Make a free account at https://hub.docker.com .
3. Make sure the model cache exists locally (it does if you ran precompute):
   the `models/` folder must be present in the repo root. If missing, run:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='./models')"
   ```

---

## Build, push, and get the link

Run these from the repo root (`c:\Users\sidda\Downloads\IndiaRuns`). Replace
`YOUR_DOCKERHUB_USERNAME`:

```bash
# 1. Log in to Docker Hub
docker login

# 2. Build the image (tagged for your registry namespace)
docker build -t YOUR_DOCKERHUB_USERNAME/redrob-ranker:latest .

# 3. Smoke-test locally before pushing (must print a CSV path, no network)
docker run --rm -v "${PWD}:/work" YOUR_DOCKERHUB_USERNAME/redrob-ranker:latest --candidates sample_candidates.json --out /work/_test_submission.csv

# 4. Push to the public registry
docker push YOUR_DOCKERHUB_USERNAME/redrob-ranker:latest
```

After step 4, your **public image link** is:

```
docker pull YOUR_DOCKERHUB_USERNAME/redrob-ranker:latest
```

and the page URL `https://hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/redrob-ranker`
is what you paste into the submission portal's **Sandbox / demo link** field.

---

## The run command to put in your submission notes / README

```bash
# Rank the baked-in 50-candidate sample (writes submission.csv to your cwd):
docker run --rm -v "${PWD}:/work" YOUR_DOCKERHUB_USERNAME/redrob-ranker:latest \
    --candidates sample_candidates.json --out /work/submission.csv

# Or rank your own small sample (<=100 candidates) mounted from the host:
docker run --rm -v "${PWD}:/work" YOUR_DOCKERHUB_USERNAME/redrob-ranker:latest \
    --candidates /work/my_sample.jsonl --out /work/submission.csv
```

This runs CPU-only, offline, and completes well within the 5-minute budget for
≤100 candidates.

---

## GitHub Container Registry (GHCR) alternative

If you prefer GHCR over Docker Hub:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GH_USERNAME --password-stdin
docker build -t ghcr.io/YOUR_GH_USERNAME/redrob-ranker:latest .
docker push ghcr.io/YOUR_GH_USERNAME/redrob-ranker:latest
# Make the package "public" in your GitHub package settings.
```

Pull link: `docker pull ghcr.io/YOUR_GH_USERNAME/redrob-ranker:latest`

---

## Notes

- The image excludes the 146 MB `artifacts/` (the full 100K precomputed
  embeddings) — they aren't needed for the small-sample sandbox, which embeds on
  the fly. They remain in your GitHub repo for the Stage-3 full reproduction.
- The build will `RUN test -d models` and fail fast if the model cache wasn't in
  the build context, so you can't accidentally ship a non-offline image.
- Image size is dominated by PyTorch (CPU) + the BGE model (~130 MB). That's
  normal and fine for a public sandbox image.
