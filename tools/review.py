#!/usr/bin/env python3
"""Replay the saved experiment on CPU. This does not run model inference."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root, manifest, nested=False):
    root = Path(root).resolve()
    for name, entry in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe manifest entry: {name}")
        expected = entry["sha256"] if nested else entry
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        if nested and path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
    return len(manifest)


def same(actual, expected, label):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Different keys: {label}")
        for key, value in expected.items():
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different lengths: {label}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError(f"Different measurement: {label}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Different value: {label}: {actual!r} != {expected!r}")


def selected_manifest(original, omitted):
    for name, expected in omitted.items():
        if name not in original or original[name] != expected:
            raise ValueError("Email omission differs from the original manifest: " + name)
    return {name: entry for name, entry in original.items() if name not in omitted}


def review(run, preparation=None, email_manifest=None):
    run = Path(run).resolve()
    bundle = read_json(email_manifest) if email_manifest else {}
    if email_manifest:
        verify_manifest(Path(email_manifest).parent, bundle["files"])
    omitted_run = bundle.get("omitted_run_files", {})
    omitted_preparation = bundle.get("omitted_preparation_files", {})
    indexed = verify_manifest(run, selected_manifest(read_json(run / "evidence-index.json")["files"], omitted_run))
    spec = importlib.util.spec_from_file_location("recorded_verifier", run / "verify_artifacts.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    environment = read_json(run / "evidence/environment.json")
    if sha256(run / "verify_artifacts.py") != environment["runtime_sha256"]:
        raise ValueError("Verifier source differs from the source recorded during execution")
    verdict = read_json(run / "verdict.json")
    policy = read_json(run / "run_policy.json")
    frozen = read_json(run / "evidence/policy-frozen-before-test.json")
    if sha256(run / "run_policy.json") != frozen["policy_sha256"]:
        raise ValueError("Frozen policy changed")
    for name, expected in policy["dataset_hashes"].items():
        path = run / ("data/" + name if name.endswith(".jsonl") else name)
        if sha256(path) != expected:
            raise ValueError("Dataset changed: " + name)
    for name, expected in policy["config_hashes"].items():
        if sha256(run / name) != expected:
            raise ValueError("Training configuration changed: " + name)
    refs = engine.read_jsonl(run / "evidence/references.jsonl")
    test = [row for row in refs if row["split"] == "test"]
    expected_ids = [row["id"] for row in test]
    predictions, metrics = {}, {}
    score_keys = ("parse_ok", "known_tools", "tool_selection", "schema_ok", "arguments_ok",
                  "complete", "truncated", "extra_calls", "unknown_calls")
    for variant in ("B", "M", "Q"):
        rows = engine.read_jsonl(run / f"evidence/predictions/test-{variant}.jsonl")
        if [row["id"] for row in rows] != expected_ids:
            raise ValueError("Missing, duplicated or reordered test predictions: " + variant)
        for row, reference in zip(rows, test):
            if row["group"] != reference["group"]:
                raise ValueError("Changed test grouping: " + variant)
            scored = engine.score_turn(row["text"], reference["expected"], reference["functions"],
                                       truncated=row["truncated"])
            for key in score_keys:
                same(scored[key], row[key], f"{variant}.{row['id']}.{key}")
        predictions[variant] = rows
        metrics[variant] = engine.summarize_predictions(rows)
        same(metrics[variant], verdict["metrics"][variant], "metrics." + variant)
    pairs = {
        "Q_minus_B": engine.grouped_pairs(predictions["B"], predictions["Q"]),
        "Q_minus_M": engine.grouped_pairs(predictions["M"], predictions["Q"]),
        "M_minus_B": engine.grouped_pairs(predictions["B"], predictions["M"]),
    }
    same(pairs, verdict["paired_bootstrap"], "paired_bootstrap")
    general = {v: read_json(run / f"evidence/general/{v}.json")["scores"] for v in ("B", "M", "Q")}
    same(general, verdict["general_suites"], "general_suites")
    integrity = {name: verdict["gates"][name] for name in engine.REQUIRED_INTEGRITY}
    decision = engine.decision_gates(pairs["Q_minus_B"], pairs["Q_minus_M"], general["B"], general["Q"], integrity)
    for key in ("verdict", "gates", "blockers"):
        same(decision[key], verdict[key], key)
    training = {str(seed): read_json(run / f"evidence/training-{seed}.json") for seed in policy["seeds"]}
    for seed, result in training.items():
        observation = result["observations"]
        if not (observation["optimizer_membership_exact"] and observation["first_materialized_update"]
                and result["changed_adapter_tensors"] == 392
                and result["frozen_projection_hashes_before"] == result["frozen_projection_hashes_after"]
                and result["checkpoint_selection"]["export_matches_checkpoint_tensors"]):
            raise ValueError("Incomplete recorded training proof: " + seed)
    stage_reports = [read_json(p) for p in (run / "evidence/stages").glob("*.json") if ".previous-" not in p.name]
    gpu = [r for r in stage_reports if r.get("hardware")]
    memory = {
        "usable_device_bytes": min(r["hardware"]["total_bytes"] for r in gpu),
        "max_allocated_bytes": max(r["resources"]["peak_allocated_bytes"] for r in gpu),
        "max_reserved_bytes": max(r["resources"]["peak_reserved_bytes"] for r in gpu),
        "max_device_used_bytes_sampled": max(r["resources"]["peak_device_used_mib_sampled"] for r in gpu) * 2**20,
    }
    memory["headroom_bytes"] = memory["usable_device_bytes"] - max(memory["max_reserved_bytes"], memory["max_device_used_bytes_sampled"])
    for name, value in memory.items():
        same(value, verdict["memory"][name], "memory." + name)
    raw_samples, raw_peak = 0, 0
    with (run / "evidence/nvidia-smi.raw.log").open() as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if len(row) == 10 and row[1] == "0":
                raw_samples += 1
                raw_peak = max(raw_peak, int(row[6].split()[0]))
    same(raw_peak * 2**20, memory["max_device_used_bytes_sampled"], "raw_nvidia_smi_peak")
    preparation_count = None
    data_summary = None
    if preparation:
        preparation = Path(preparation)
        preparation_count = verify_manifest(preparation, selected_manifest(
            read_json(preparation / "manifest.json"), omitted_preparation), nested=True)
        data_summary = read_json(preparation / "audit/data-audit.json")["counts_and_distribution"]
    selection = read_json(run / "evidence/selection.json")
    accuracy = [row["complete"] for row in selection["all_seeds"]]
    transitions = {}
    for label, before, after in (("B_to_M", "B", "M"), ("M_to_Q", "M", "Q")):
        transitions[label] = {
            "both_correct": sum(b["complete"] and a["complete"] for b, a in zip(predictions[before], predictions[after])),
            "lost_correct": sum(b["complete"] and not a["complete"] for b, a in zip(predictions[before], predictions[after])),
            "gained_correct": sum(not b["complete"] and a["complete"] for b, a in zip(predictions[before], predictions[after])),
            "both_incorrect": sum(not b["complete"] and not a["complete"] for b, a in zip(predictions[before], predictions[after])),
        }
    examples = []
    for reference, merged, quantized in zip(test, predictions["M"], predictions["Q"]):
        if merged["complete"] and not quantized["complete"]:
            examples.append({"id": reference["id"], "request": reference["messages"][1]["content"],
                             "expected": reference["expected"], "merged": merged["text"], "nf4": quantized["text"],
                             "nf4_checks": {key: quantized[key] for key in score_keys}})
    return {
        "review_status": "PASS", "release_verdict": verdict["verdict"], "run_id": read_json(run / "settings.json")["run_id"],
        "original_run_files_hash_verified": indexed, "preparation_files_hash_verified": preparation_count,
        "original_run_files_omitted": len(omitted_run), "preparation_files_omitted": len(omitted_preparation),
        "test_predictions_rescored": sum(len(rows) for rows in predictions.values()),
        "bootstrap_comparisons_reproduced": len(pairs), "raw_gpu0_samples": raw_samples,
        "metrics": metrics, "paired_bootstrap": pairs, "general_suites": general, "memory": memory,
        "seed_validation": {"results": selection["all_seeds"], "mean": statistics.mean(accuracy),
                            "sample_std": statistics.stdev(accuracy), "test_seed": selection["selected"]["seed"]},
        "transitions": transitions, "quantization_loss_examples": examples[:3], "data_summary": data_summary,
        "blockers": verdict["blockers"],
        "limits": ["CPU replay uses saved predictions and checksums; no model inference was rerun.",
                   "Training, identity and general-suite observations are read from the original hashed reports.",
                   "Checksums detect changes relative to the supplied manifest; they are not an external attestation.",
                   "Reference semantics and alternative valid calls were not adjudicated by this replay."]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "run")
    parser.add_argument("--preparation", type=Path, default=ROOT / "data-preparation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--email-bundle", action="store_true",
                        help="Verify the curated attachment using its explicit EMAIL_MANIFEST.json omissions")
    args = parser.parse_args()
    try:
        result = review(args.run, args.preparation, ROOT / "EMAIL_MANIFEST.json" if args.email_bundle else None)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"Evidence audit FAIL: {exc}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("review_status", "release_verdict", "original_run_files_hash_verified",
          "preparation_files_hash_verified", "original_run_files_omitted", "preparation_files_omitted",
          "test_predictions_rescored", "bootstrap_comparisons_reproduced",
          "raw_gpu0_samples", "blockers")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
