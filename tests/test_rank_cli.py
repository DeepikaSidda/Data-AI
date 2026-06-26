"""CLI wiring tests for rank.py.

Verify main() wires arguments to RankingPipeline.run and reports a summary on
success, and that a typed DatasetAccessError surfaces as a non-zero exit with a
clear stderr message. The pipeline itself is exercised elsewhere; here we
monkeypatch run so the test stays offline and fast.

Requirements: 13.2.
"""

from __future__ import annotations

import rank
from ranking.errors import DatasetAccessError
from ranking.models import ArtifactPaths
from ranking.pipeline import RankingPipeline, RunReport


def test_main_success_invokes_pipeline_and_prints_summary(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_run(self, candidates_path, out_path, artifacts=None, top_n=100,
                 progress=None, job_profile_path="job_profile.yaml"):
        captured["candidates_path"] = candidates_path
        captured["out_path"] = out_path
        captured["artifacts"] = artifacts
        captured["top_n"] = top_n
        captured["job_profile_path"] = job_profile_path
        if progress is not None:
            progress(0.5)
            progress(1.0)
        return RunReport(
            total_candidates=120,
            valid_count=118,
            skipped_count=2,
            honeypot_count=3,
            written_rows=100,
            elapsed_s=1.23,
        )

    monkeypatch.setattr(RankingPipeline, "run", fake_run)

    out_csv = tmp_path / "submission.csv"
    exit_code = rank.main([
        "--candidates", "./candidates.jsonl",
        "--out", str(out_csv),
        "--artifacts-dir", str(tmp_path / "artifacts"),
    ])

    assert exit_code == 0
    assert captured["candidates_path"] == "./candidates.jsonl"
    assert captured["out_path"] == str(out_csv)
    assert captured["top_n"] == 100
    assert isinstance(captured["artifacts"], ArtifactPaths)
    assert captured["artifacts"].embeddings.endswith("embeddings.npy")
    assert captured["artifacts"].id_order.endswith("id_order.json")
    assert captured["artifacts"].job_embedding.endswith("job_embedding.npy")

    out = capsys.readouterr()
    assert "wrote 100 rows" in out.out
    assert "118 valid / 2 skipped" in out.out
    assert "3 honeypots" in out.out


def test_main_dataset_access_error_returns_nonzero(monkeypatch, capsys, tmp_path):
    def fake_run(self, *args, **kwargs):
        raise DatasetAccessError("could not read candidate file './missing.jsonl'")

    monkeypatch.setattr(RankingPipeline, "run", fake_run)

    exit_code = rank.main([
        "--candidates", "./missing.jsonl",
        "--out", str(tmp_path / "out.csv"),
    ])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "rank.py:" in err
    assert "could not read candidate file" in err
