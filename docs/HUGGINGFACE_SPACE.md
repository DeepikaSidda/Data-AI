# Deploying the Sandbox to Hugging Face Spaces

This repo ships a ready-to-run Gradio sandbox (`app.py`) that satisfies the
challenge's mandatory **sandbox / demo link** requirement (submission_spec
§10.5): it accepts a small candidate sample (≤100 records), runs the real
ranking pipeline end-to-end on CPU, and returns a ranked CSV.

You only need a free Hugging Face account. ~5 minutes.

---

## What's already done for you

- `app.py` — the Gradio app (pre-loaded with the bundled 50-candidate sample).
- `sample_candidates.json` — the bundled sample so the Space works with zero upload.
- `requirements.txt` — includes `gradio` (pinned) plus the pipeline deps.
- The app downloads the embedding model **once at startup** (network is allowed
  while the Space boots), then the pipeline runs offline from the local cache —
  faithful to the "no network during ranking" rule.

---

## Steps

### 1. Create the Space
1. Go to https://huggingface.co/ and sign in (create a free account if needed).
2. Click your avatar → **New Space**.
3. Settings:
   - **Owner**: your username
   - **Space name**: e.g. `redrob-ranker`
   - **License**: your choice (e.g. `mit`)
   - **SDK**: **Gradio**
   - **Hardware**: **CPU basic (free)** — this is required; the ranker is CPU-only.
   - Visibility: **Public** (so organizers can open it).
4. Click **Create Space**.

### 2. Add the Space front-matter to README
Hugging Face reads a small YAML header at the top of `README.md`. In the Space's
`README.md` (you can edit it in the Space's **Files** tab), put this block at the
very top:

```yaml
---
title: Redrob Candidate Ranker
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
```

(Everything below the `---` block can stay as your normal README.)

### 3. Push the code
From the repo root, add the Space as a git remote and push. Replace
`YOUR_USERNAME` / `redrob-ranker`:

```bash
git init                 # if the repo isn't already a git repo
git add app.py sample_candidates.json requirements.txt ranking job_profile.yaml job_description.md README.md
git commit -m "Add Hugging Face Space sandbox"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker
git push space main      # or: git push space HEAD:main
```

> You do **not** need to push `artifacts/` (the 146 MB precomputed embeddings)
> or `models/` to the Space — the small-sample demo embeds on the fly and the
> model auto-downloads on first boot. Keeping them out makes the Space build
> faster. (They *are* needed in your GitHub repo for the full 100K Stage-3
> reproduction.)

### 4. Wait for the build, then grab the link
- The Space will install dependencies and boot (first boot is slower because it
  downloads the embedding model). 
- When it shows **Running**, open it, click **Rank**, and confirm a ranked table
  appears with a downloadable `submission.csv`.
- The Space URL — `https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker` —
  is your **sandbox link** for the submission portal.

---

## Test it locally first (optional)

```bash
pip install -r requirements.txt
python app.py
# open the local URL Gradio prints, click "Rank"
```

---

## If you'd rather not host anything

The spec also accepts a **self-contained `docker run` recipe** in the README as
an alternative to a hosted sandbox. That recipe is already in the main
`README.md` (and `Dockerfile`), so you can submit that instead — but a live
Hugging Face Space is the smoother, higher-confidence option for reviewers.
