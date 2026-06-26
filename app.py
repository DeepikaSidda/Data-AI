"""Hugging Face Space (Gradio) — Redrob candidate ranking sandbox.

A small, hosted demo that satisfies the challenge's mandatory sandbox
requirement (submission_spec section 10.5): it accepts a small candidate sample
(<= 100 records), runs the *real* ranking pipeline end-to-end on CPU, and
returns a ranked CSV — all within the compute budget.

How it stays faithful to the offline ranking contract:
  * The embedding model weights are downloaded ONCE at app startup (network is
    allowed while the Space boots, not during the graded ranking step). After
    caching, the pipeline runs fully offline via ``ranking.embedding`` (which
    enforces ``HF_HUB_OFFLINE=1``).
  * For a small sample the pipeline embeds on the fly (no precomputed artifacts
    needed), so the Space is self-contained.

Run locally:  python app.py
Deploy:       see docs/HUGGINGFACE_SPACE.md
"""

from __future__ import annotations

import json
import os
import tempfile

# ---------------------------------------------------------------------------
# Ensure the embedding model is cached BEFORE anything enables offline mode.
# ranking.embedding sets HF_HUB_OFFLINE=1 when an EmbeddingModel is constructed,
# which would block a download. So we pre-fetch the weights here, at import
# time, while the network is still available (Space boot).
# ---------------------------------------------------------------------------
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CACHE_DIR = "./models"


def _ensure_model_cached() -> None:
    """Download the embedding model into the local cache if not already present."""
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(_MODEL_NAME, cache_folder=_CACHE_DIR)
    except Exception as exc:  # pragma: no cover - surfaced in the UI instead
        print(f"[app] model pre-cache warning: {exc}")


_ensure_model_cached()

import gradio as gr  # noqa: E402  (after model pre-cache on purpose)

from ranking.config import ScoringConfig  # noqa: E402
from ranking.pipeline import RankingPipeline  # noqa: E402

_JOB_PROFILE = "job_profile.yaml"
_SAMPLE = "sample_candidates.json"
_MAX_CANDIDATES = 100  # sandbox cap per submission_spec 10.5


def _normalize_to_jsonl(raw_path: str, out_path: str) -> int:
    """Write the uploaded data as JSON Lines and return the record count.

    Accepts either a JSON array (``.json``) or JSON Lines (``.jsonl``). Caps the
    number of records at ``_MAX_CANDIDATES`` to honor the sandbox limit.
    """
    text = open(raw_path, "r", encoding="utf-8").read().strip()
    records: list
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = records[:_MAX_CANDIDATES]
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec))
            fh.write("\n")
    return len(records)


def rank_candidates(file_obj):
    """Gradio handler: rank an uploaded (or the bundled) candidate sample."""
    src = file_obj.name if file_obj is not None else _SAMPLE
    if not os.path.exists(src):
        return None, "No input file found.", None

    tmpdir = tempfile.mkdtemp()
    candidates_path = os.path.join(tmpdir, "candidates.jsonl")
    out_path = os.path.join(tmpdir, "submission.csv")

    try:
        n = _normalize_to_jsonl(src, candidates_path)
    except Exception as exc:
        return None, f"Could not parse input: {exc}", None

    top_n = min(100, n)
    try:
        report = RankingPipeline(ScoringConfig.load(None)).run(
            candidates_path,
            out_path,
            artifacts=None,           # small-sample path: embed on the fly
            top_n=top_n,
            job_profile_path=_JOB_PROFILE,
        )
    except Exception as exc:
        return None, f"Ranking failed: {exc}", None

    # Build a preview table.
    import csv

    rows = list(csv.DictReader(open(out_path, "r", encoding="utf-8")))
    preview = [[r["candidate_id"], r["rank"], r["score"], r["reasoning"]] for r in rows]

    status = (
        f"Ranked {report.valid_count} candidate(s) "
        f"({report.skipped_count} skipped, {report.honeypot_count} honeypots flagged) "
        f"in {report.elapsed_s:.2f}s. Showing top {len(rows)}."
    )
    return preview, status, out_path


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown(
        "# Redrob Intelligent Candidate Ranker\n"
        "Upload a small candidate sample (a JSON array or JSON Lines, "
        f"up to {_MAX_CANDIDATES} records) for the **Senior AI Engineer** role, "
        "or just click **Rank** to use the bundled 50-candidate sample.\n\n"
        "The ranker runs fully on CPU: local BGE embeddings + a hybrid feature "
        "score (title-aware skills, experience band, product-vs-services "
        "trajectory) + a behavioral-availability modifier + honeypot detection, "
        "with grounded reasoning per candidate."
    )
    with gr.Row():
        file_in = gr.File(
            label="Candidate sample (.json array or .jsonl) — optional",
            file_types=[".json", ".jsonl"],
        )
    run_btn = gr.Button("Rank", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    table = gr.Dataframe(
        headers=["candidate_id", "rank", "score", "reasoning"],
        label="Ranked shortlist",
        wrap=True,
    )
    download = gr.File(label="Download submission.csv")

    run_btn.click(rank_candidates, inputs=file_in, outputs=[table, status, download])


if __name__ == "__main__":
    demo.launch()
