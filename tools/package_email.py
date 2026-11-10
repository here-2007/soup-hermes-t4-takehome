#!/usr/bin/env python3
"""Build a curated email attachment with complete logs and declared exclusions."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from package_submission import build
from review import read_json, review, sha256


ROOT = Path(__file__).resolve().parents[1]
FULL_ARCHIVE = "https://github.com/here-2007/soup-hermes-t4-takehome/releases/latest/download/soup-ai-engineer-submission.zip"
MODEL = "https://www.kaggle.com/models/pernavjain/soup-experiment/PyTorch/experiment-artifacts/1"
PREPARATION = {
    "manifest.json", "audit/data-audit.json", "audit/policy.json", "audit/sources.json",
    "audit/integrity-checks.json", "audit/split-manifest.json", "audit/retention.json", "audit/dataset-card.md",
}
OLD_DRAFTS = {"REPORT.md", "report.pdf", "REFLECTION_AND_AI_USE.md"}


def populate(destination):
    destination = Path(destination)
    for name in ("report.pdf", "REPORT.md", "requirements-review.txt", "THIRD_PARTY.md", "SUBMISSION.json",
                 "docs/reflection-and-ai-use.md", "docs/analysis.md", "docs/evidence-map.md", "tools/review.py",
                 "licenses/Apache-2.0-Qwen.txt"):
        target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copytree(ROOT / "notebooks", destination / "notebooks")
    for path in (ROOT / "run").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT / "run")
        if relative.as_posix() in OLD_DRAFTS or (relative.parts[:2] == ("evidence", "probes") and path.suffix == ".pt"):
            continue
        target = destination / "run" / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    for name in PREPARATION:
        target = destination / "data-preparation" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "data-preparation" / name, target)
    original_run = read_json(ROOT / "run/evidence-index.json")["files"]
    original_preparation = read_json(ROOT / "data-preparation/manifest.json")
    omitted_run = {name: digest for name, digest in original_run.items() if not (destination / "run" / name).is_file()}
    omitted_preparation = {name: entry for name, entry in original_preparation.items()
                           if not (destination / "data-preparation" / name).is_file()}
    text = f"""# Email submission: Pernav Jain / Soup AI Engineer

Open **report.pdf** first. It is the final two-page report. The recorded verdict is **DON'T SHIP**.

## Included deliverables

- Final PDF and Markdown report, with personal reflection and AI-use disclosure.
- Both executed notebooks, in `notebooks/`.
- Original model verification script: `run/verify_artifacts.py`.
- Exact train/validation/test files, Soup configurations, settings and frozen policy in `run/`.
- Every original command log, complete raw nvidia-smi trace and host-memory trace in `run/evidence/`.
- All saved scores, B/M/Q predictions, training/update reports, control results and Soup ship decisions.
- Data-source, split and structural audit records under `data-preparation/`.

## Check this attachment on CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-review.txt
python tools/review.py --email-bundle
```

The command verifies the included files against their original hashes, rescores all 450 saved B/M/Q test
predictions and repeats all three paired-bootstrap calculations. Audit PASS means the records agree;
the model verdict remains DON'T SHIP. This review does not rerun model inference.

## What was left out

- Seven large binary probe tensors, which are available in the complete archive below. Their numerical
  results and failed merge check remain in this attachment.
- Three older automated report/reflection drafts; the final documents are included at the top level.
- The remaining preparation cache, tokenizer, full mask arrays and duplicate preparation exports.
- Model weights, hosted separately on Kaggle.

`EMAIL_MANIFEST.json` lists every included file and each omitted original run/preparation file.
No failed command log, result, score or test prediction was removed or shortened.

Complete code and records: https://github.com/here-2007/soup-hermes-t4-takehome

Complete archive: {FULL_ARCHIVE}

Exact B/M/Q weights and training checkpoints: {MODEL}
"""
    (destination / "README.md").write_text(text)
    manifest = {
        "purpose": "Curated email attachment; not a repository export",
        "files": {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()},
        "omitted_run_files": omitted_run,
        "omitted_preparation_files": omitted_preparation,
        "full_archive": FULL_ARCHIVE, "model_artifacts": MODEL,
        "original_run_index_sha256": sha256(ROOT / "run/evidence-index.json"),
        "original_preparation_manifest_sha256": sha256(ROOT / "data-preparation/manifest.json"),
    }
    (destination / "EMAIL_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = review(destination / "run", destination / "data-preparation", destination / "EMAIL_MANIFEST.json")
    if result["test_predictions_rescored"] != 450 or result["release_verdict"] != "DON'T SHIP":
        raise ValueError("Email replay does not reproduce the completed run")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="soup-email-") as directory:
        replay = populate(directory)
        result = build(args.output, directory, archive_root="Pernav-Jain-Soup-AI-Engineer")
    if result["bytes"] >= 10 * 2**20:
        raise ValueError("Email attachment exceeds the 10 MiB target")
    print(f"Email attachment verified: {result['bytes']/2**20:.2f} MiB, {result['files']} files")
    print(f"Original run files checked: {replay['original_run_files_hash_verified']}; omissions explicitly listed: {replay['original_run_files_omitted']}")
    print(f"Saved test generations rescored: {replay['test_predictions_rescored']}; verdict: {replay['release_verdict']}")
    print(f"SHA-256: {result['sha256']}")


if __name__ == "__main__":
    main()
