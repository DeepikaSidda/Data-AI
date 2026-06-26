"""Deliverable-presence tests (task 18.2).

Assert that the repository deliverables exist and carry the required content:
pinned dependencies, an offline/CPU-safe ``submission_metadata.yaml``, the
precompute artifact builder, a Docker recipe, a README documenting the single
reproduce command, and the methodology deck.

Validates: Requirements 11.2, 11.3, 13.1, 13.4
"""

from pathlib import Path

import pytest
import yaml

# Repo root is two levels up from this test file: <repo>/tests/test_deliverables.py
REPO_ROOT = Path(__file__).resolve().parents[1]

# The single canonical reproduce command, reused across several assertions.
REPRODUCE_COMMAND = "python rank.py --candidates ./candidates.jsonl --out ./submission.csv"


# --------------------------------------------------------------------------- #
# requirements.txt — pinned dependencies (Requirement 11.2 / 13.1)
# --------------------------------------------------------------------------- #
def _requirement_lines(text: str):
    """Return non-empty, non-comment dependency lines from requirements.txt."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def test_requirements_txt_exists():
    assert (REPO_ROOT / "requirements.txt").is_file()


def test_requirements_txt_all_versions_pinned():
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    deps = _requirement_lines(text)
    assert deps, "requirements.txt has no dependency lines"
    unpinned = [line for line in deps if "==" not in line]
    assert not unpinned, f"unpinned dependencies found: {unpinned}"


@pytest.mark.parametrize(
    "package",
    ["sentence-transformers", "numpy", "hypothesis", "pytest", "pyyaml"],
)
def test_requirements_txt_key_packages_present(package):
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    deps = _requirement_lines(text)
    # Match the pinned package name at the start of a dependency line.
    assert any(line.split("==")[0].strip() == package for line in deps), (
        f"expected pinned package '{package}' in requirements.txt"
    )


# --------------------------------------------------------------------------- #
# README.md — reproduce command, precompute docs, Docker recipe (Req 13.1/13.4)
# --------------------------------------------------------------------------- #
def test_readme_exists():
    assert (REPO_ROOT / "README.md").is_file()


def test_readme_contains_reproduce_command():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert REPRODUCE_COMMAND in text, "README must document the single reproduce command"


def test_readme_mentions_precompute():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "precompute_embeddings.py" in text, "README must document the precompute step"


def test_readme_contains_docker_recipe():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docker build" in text, "README must include a 'docker build' step"
    assert "docker run" in text, "README must include a 'docker run' recipe"


# --------------------------------------------------------------------------- #
# submission_metadata.yaml — offline/GPU flags + declarations (Req 11.2/11.3)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def submission_metadata():
    path = REPO_ROOT / "submission_metadata.yaml"
    assert path.is_file(), "submission_metadata.yaml must exist at repo root"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "submission_metadata.yaml must parse to a mapping"
    return data


def test_submission_metadata_compute_flags(submission_metadata):
    compute = submission_metadata.get("compute")
    assert isinstance(compute, dict), "submission_metadata.yaml must have a 'compute' mapping"
    assert compute.get("uses_gpu_for_inference") is False
    assert compute.get("has_network_during_ranking") is False
    assert compute.get("pre_computation_required") is True


def test_submission_metadata_reproduce_command(submission_metadata):
    assert submission_metadata.get("reproduce_command") == REPRODUCE_COMMAND


def test_submission_metadata_declarations(submission_metadata):
    declarations = submission_metadata.get("declarations")
    assert isinstance(declarations, dict), "submission_metadata.yaml must have a 'declarations' mapping"
    expected_keys = {
        "read_submission_spec",
        "code_is_original_work",
        "no_collusion",
        "honeypot_check_done",
        "reproduction_tested",
    }
    assert expected_keys.issubset(declarations.keys()), (
        f"missing declaration keys: {expected_keys - set(declarations.keys())}"
    )


# --------------------------------------------------------------------------- #
# Precompute artifact builder (Requirement 13.1)
# --------------------------------------------------------------------------- #
def test_precompute_script_or_artifacts_exist():
    script = REPO_ROOT / "precompute_embeddings.py"
    artifacts_dir = REPO_ROOT / "artifacts"
    committed_artifacts = artifacts_dir.is_dir() and any(artifacts_dir.iterdir())
    # At minimum the artifact builder script must exist; committed artifacts are
    # an acceptable additional form of the deliverable.
    assert script.is_file(), "precompute_embeddings.py (artifact builder) must exist at repo root"
    assert script.is_file() or committed_artifacts


# --------------------------------------------------------------------------- #
# Dockerfile — base image + entrypoint (Requirement 13.4)
# --------------------------------------------------------------------------- #
def test_dockerfile_exists():
    assert (REPO_ROOT / "Dockerfile").is_file()


def test_dockerfile_references_python_base_and_entrypoint():
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python" in text, "Dockerfile must use a python base image"
    assert "ENTRYPOINT" in text, "Dockerfile must define an ENTRYPOINT"
    assert "rank.py" in text, "Dockerfile entrypoint must invoke rank.py"


# --------------------------------------------------------------------------- #
# Methodology deck (Requirement 13.x — deliverable bundle)
# --------------------------------------------------------------------------- #
def test_methodology_deck_exists_and_is_non_trivial():
    path = REPO_ROOT / "docs" / "methodology_deck.md"
    assert path.is_file(), "docs/methodology_deck.md must exist"
    text = path.read_text(encoding="utf-8")
    assert len(text) > 1000, "methodology deck should be non-trivial (> 1000 chars)"
    lowered = text.lower()
    assert "honeypot" in lowered, "methodology deck must mention honeypot detection"
    assert "ndcg" in lowered, "methodology deck must mention the NDCG scoring metric"
