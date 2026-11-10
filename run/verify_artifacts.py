"""Standalone Soup experiment runner and verifier; embedded in the Kaggle notebook.

Only the standard library is imported at module load so pure checks need no GPU.
Run --help for stage commands. Reports describe observed results, never intentions.
"""

import argparse
import collections
import contextlib
import copy
import csv
import datetime as dt
from decimal import Decimal
import faulthandler
import functools
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import unicodedata

SOUP_REV = "4cff43bf967e96a484381205e5125e2fc39227f1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
T4_ALLOCATOR_CONFIG = "backend:native,expandable_segments:True,garbage_collection_threshold:0.80"
EXPECTED_DATA = {
    "train.jsonl": "3e1c19821c79e777c6e1524e3b94880c6ecda03240f630f25b8c945b31e418bf",
    "validation.jsonl": "d46eee968314585345c6fd50f570ddfdde242f226f828d22b0ad1d32239063da",
    "test.jsonl": "b7f4c4942fc77e9bf16ab0812be382de458a2077b0eed846824e81b055a7b19f",
    "soup-data.yaml": "1daab66b6b161b65fc9f2dfe6b63cfd0288750132b8627b5508ac4a67f1fd9f8",
}
ROOT = None
SETTINGS = None
TORCH = None
CAPTURE = None
TRAINER_POLICY = {
    "eval_strategy": "epoch", "save_strategy": "epoch", "do_eval": True,
    "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
    "save_total_limit": 2, "per_device_eval_batch_size": 1, "prediction_loss_only": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False}, "dataloader_num_workers": 0,
    "disable_tqdm": True,
}
REQUIRED_INTEGRITY = (
    "required_stage_reports_pass", "policy_and_test_unchanged", "artifact_hashes_bound",
    "updated_and_active_adapter", "selected_frozen_weights_unchanged", "disabled_base_agreement",
    "merge_fidelity", "scorer_data_controls", "base_for_merge_control", "q0_for_q_control",
    "Soup_noop_adapter_control", "portable_M_Q", "real_saved_nf4",
    "no_new_critical_challenge_failures", "one_gib_tested_headroom", "three_seed_development_complete",
    "Soup_environment_bounds_ok", "Soup_fixture_coverage_resolved",
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def benchmark_headroom(report, allocator_config):
    """Use the more conservative of device usage and allocator reservation."""
    resources = report.get("resources", {})
    hardware = report.get("hardware", {})
    capacity = hardware.get("total_bytes", 0)
    sampled = resources.get("peak_device_used_mib_sampled", 0) * 2**20
    allocated = resources.get("peak_allocated_bytes", 0)
    reserved = resources.get("peak_reserved_bytes", 0)
    remaining = capacity - max(sampled, reserved)
    measured = (
        report.get("status") == "passed" and hardware.get("visible_count") == 1
        and "T4" in hardware.get("name", "") and capacity > 0
        and resources.get("device_samples", 0) > 0 and sampled > 0 and 0 < allocated <= reserved
        and not any("cuda_measurement_error" in segment for segment in resources.get("segments", []))
        and hardware.get("allocator_backend") == "native" and hardware.get("allocator_config") == allocator_config
    )
    return {
        "passed": measured and remaining >= 2**30,
        "measurement_complete": measured,
        "usable_device_gib": round(capacity / 2**30, 3),
        "sampled_device_peak_gib": round(sampled / 2**30, 3),
        "allocated_peak_gib": round(allocated / 2**30, 3),
        "reserved_peak_gib": round(reserved / 2**30, 3),
        "headroom_gib": round(remaining / 2**30, 3) if measured else None,
        "required_headroom_gib": 1,
        "allocator_config": hardware.get("allocator_config"),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path):
    return [strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key: " + key)
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result) or Decimal(value) != Decimal(str(result)):
            raise ValueError("nonfinite or precision-losing number")
        return result

    def invalid(value):
        raise ValueError("nonstandard JSON constant: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)


def parse_calls(text):
    if not isinstance(text, str):
        raise ValueError("response is not text")
    calls, position = [], 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        match = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S).match(text, position)
        if match is None:
            raise ValueError(f"unexpected text/incomplete tag at {position}")
        call = strict_json(match.group(1))
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("call keys must be name and arguments")
        if not isinstance(call["name"], str) or not call["name"] or not isinstance(call["arguments"], dict):
            raise ValueError("invalid call name/arguments")
        calls.append(call)
        position = match.end()
    if not calls:
        raise ValueError("no complete calls")
    return calls


def tagged(calls):
    return "\n".join("<tool_call>\n" + canonical(call) + "\n</tool_call>" for call in calls)


def json_equal(a, b):
    pending = [(a,b)]
    while pending:
        x,y = pending.pop()
        if isinstance(x,bool) or isinstance(y,bool):
            if type(x) is not type(y) or x != y:return False
        elif isinstance(x,(int,float)) and isinstance(y,(int,float)):
            if x != y:return False
        elif type(x) is not type(y):
            return False
        elif isinstance(x,dict):
            if x.keys() != y.keys():return False
            pending.extend((x[k],y[k]) for k in x)
        elif isinstance(x,list):
            if len(x) != len(y):return False
            pending.extend(zip(x,y))
        elif x != y:
            return False
    return True


@functools.lru_cache(maxsize=10000)
def schema_validator(schema_text):
    from jsonschema import Draft7Validator, FormatChecker, validators
    schema = strict_json(schema_text)
    checker = FormatChecker()

    def guard(value):
        if isinstance(value, dict):
            if any(key in value for key in ("$dynamicRef", "$recursiveRef", "nullable")):
                raise ValueError("unsupported schema keyword")
            if "$ref" in value and not str(value["$ref"]).startswith("#"):
                raise ValueError("external schema references forbidden")
            if "format" in value and value["format"] not in checker.checkers:
                raise ValueError("unknown schema format")
            for item in value.values():
                guard(item)
        elif isinstance(value, list):
            for item in value:
                guard(item)
    guard(schema)
    if schema.get("type") != "object":
        raise ValueError("parameters must declare object type")
    known = {"http://json-schema.org/draft-04/schema#", "http://json-schema.org/draft-06/schema#",
             "http://json-schema.org/draft-07/schema#", "https://json-schema.org/draft/2019-09/schema",
             "https://json-schema.org/draft/2020-12/schema"}
    if "$schema" in schema and schema["$schema"] not in known:
        raise ValueError("unsupported schema dialect")
    cls = validators.validator_for(schema, default=Draft7Validator)
    cls.check_schema(schema)
    return cls(schema, format_checker=checker)


def functions_from_messages(messages):
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected explicit system/user/assistant messages")
    blocks = [x.strip() for x in re.findall(r"<tools>(.*?)</tools>", messages[0]["content"], flags=re.S) if x.strip()]
    if len(blocks) != 1:
        raise ValueError("expected one nonempty tools block")
    tools = strict_json(blocks[0])
    if not isinstance(tools, list) or not tools:
        raise ValueError("empty tools")
    result = {}
    for tool in tools:
        if tool.get("type") != "function":
            raise ValueError("unsupported tool type")
        function = tool["function"]
        name = function["name"]
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("missing/duplicate tool name")
        schema_validator(canonical(function["parameters"]))
        result[name] = function
    return result


def score_turn(text, expected, functions, truncated=False, matcher=None):
    result = {key: False for key in ("parse_ok", "known_tools", "tool_selection", "schema_ok", "arguments_ok", "complete")}
    result.update(truncated=bool(truncated), extra_calls=False, unknown_calls=False)
    try:
        actual = parse_calls(text)
    except (ValueError, TypeError, RecursionError) as exc:
        return {**result, "error": str(exc)}
    result["parse_ok"] = True
    result["extra_calls"] = len(actual) > len(expected)
    result["known_tools"] = all(call["name"] in functions for call in actual)
    result["unknown_calls"] = not result["known_tools"]
    result["tool_selection"] = [x["name"] for x in actual] == [x["name"] for x in expected]
    schema_ok = result["known_tools"]
    if schema_ok:
        for call in actual:
            try:
                schema_validator(canonical(functions[call["name"]]["parameters"])).validate(call["arguments"])
            except Exception as exc:
                schema_ok = False
                result["schema_error"] = str(exc)[:500]
    result["schema_ok"] = schema_ok
    result["arguments_ok"] = len(actual) == len(expected) and all(json_equal(x["arguments"], y["arguments"]) for x, y in zip(actual, expected))
    matches = []
    if len(actual) == len(expected):
        for a, b in zip(actual, expected):
            matches.append(bool(matcher(canonical({"function": a}), canonical({"function": b})))
                           if matcher is not None else json_equal(a, b))
    result["complete"] = bool(not truncated and schema_ok and len(actual) == len(expected) and matches and all(matches))
    return result


def grouped_pairs(before, after, replicates=10000, seed=42):
    if len(before) != len(after) or not before:
        raise ValueError("paired denominators differ or are empty")
    groups = collections.defaultdict(lambda: [0.0, 0])
    for b, a in zip(before, after):
        if b["id"] != a["id"] or b["group"] != a["group"]:
            raise ValueError("paired IDs/groups do not match")
        groups[b["group"]][0] += float(a["complete"]) - float(b["complete"])
        groups[b["group"]][1] += 1
    values = list(groups.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        chosen = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(x[0] for x in chosen) / sum(x[1] for x in chosen))
    draws.sort()
    return {"delta": sum(x[0] for x in values) / len(before), "ci95": [draws[int(.025 * (replicates - 1))], draws[int(.975 * (replicates - 1))]],
            "turns": len(before), "groups": len(values), "replicates": replicates, "method": "paired cluster bootstrap, all turns retained"}


def summarize_predictions(rows):
    if not rows:
        raise ValueError("no predictions")
    keys = ("parse_ok", "known_tools", "tool_selection", "schema_ok", "arguments_ok", "complete", "truncated", "extra_calls", "unknown_calls")
    result = {key: sum(bool(x[key]) for x in rows) / len(rows) for key in keys}
    result["n"] = len(rows)
    result["mean_assistant_nll"] = sum(x["nll_sum"] for x in rows) / sum(x["nll_tokens"] for x in rows)
    result["slices"] = {}
    for label, members in (("single_call", [x for x in rows if x["call_count"] == 1]),
                           ("multiple_call", [x for x in rows if x["call_count"] > 1]),
                           ("length_le_512", [x for x in rows if x["reference_tokens"] <= 512]),
                           ("length_gt_512", [x for x in rows if x["reference_tokens"] > 512])):
        result["slices"][label] = {"n": len(members), "complete": sum(x["complete"] for x in members) / len(members) if members else None}
    result["latency_seconds"] = {"median": statistics.median(x["generation_seconds"] for x in rows),
                                 "p95": sorted(x["generation_seconds"] for x in rows)[math.ceil(.95*len(rows))-1]}
    return result


def decision_gates(pair_bq, pair_mq, general_b, general_q, integrity):
    gates = {name:False for name in REQUIRED_INTEGRITY}
    gates.update(integrity)
    gates.update({
        "task_gain_ge_5pp": round(pair_bq["delta"],12) >= .05,
        "task_gain_ci_positive": pair_bq["ci95"][0] > 0,
        "quant_point_loss_le_2pp": round(pair_mq["delta"],12) >= -.02,
        "quant_ci_lower_ge_minus5pp": round(pair_mq["ci95"][0],12) >= -.05,
        "general_suite_regression_le_5pp": bool(general_b) and general_b.keys() == general_q.keys()
            and all(round(general_q[k] - general_b[k],12) >= -.05 for k in general_b),
    })
    return {"verdict": "SHIP" if all(value is True for value in gates.values()) else "DON'T SHIP",
            "gates": gates, "blockers": [name for name, passed in gates.items() if passed is not True]}


def event(kind, **values):
    row = {"utc": now(), "event": kind, **values}
    path = ROOT / "evidence/events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(canonical(row) + "\n")
    if kind in {"stage_start", "stage_end", "segment", "progress"}:
        print(canonical(row), file=sys.stderr, flush=True)


class ResourceCapture:
    def __init__(self, label):
        import psutil
        self.psutil = psutil
        self.label = label
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.segments = []
        self.peak_rss = 0
        self.device_peak_mib = 0
        self.device_samples = 0
        self.phase_name = "startup"
        self.thread = None

    def sample(self):
        process = self.psutil.Process()
        raw_path = ROOT / "evidence/nvidia-smi.raw.log"
        while not self.stop.is_set():
            record = {"utc": now(), "stage": self.label, "phase": self.phase_name,
                      "rss_bytes": process.memory_info().rss, "available_ram_bytes": self.psutil.virtual_memory().available,
                      "disk_free_bytes": shutil.disk_usage(ROOT).free}
            self.peak_rss = max(self.peak_rss, record["rss_bytes"])
            try:
                command = ["nvidia-smi", "--query-gpu=timestamp,index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,clocks.sm,power.draw", "--format=csv"]
                observed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
                with raw_path.open("a") as handle:
                    handle.write(f"# {canonical(record)} exit={observed.returncode}\n{observed.stdout}{observed.stderr}")
                if observed.returncode == 0:
                    for row in csv.reader(observed.stdout.splitlines()[1:], skipinitialspace=True):
                        if len(row) == 10 and row[1] == "0":
                            self.device_peak_mib = max(self.device_peak_mib, int(row[6].split()[0]))
                            self.device_samples += 1
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                record["sample_error"] = f"{type(exc).__name__}: {exc}"
            with (ROOT / "evidence/host-samples.jsonl").open("a") as handle:
                handle.write(canonical(record) + "\n")
            self.stop.wait(.2)

    def mark(self, name, reset=True):
        global TORCH
        record = {"name": name, "utc": now(), "elapsed_seconds": time.monotonic() - self.started}
        if TORCH is not None and TORCH.cuda.is_initialized():
            try:
                TORCH.cuda.synchronize()
                record.update(allocated_bytes=TORCH.cuda.memory_allocated(), reserved_bytes=TORCH.cuda.memory_reserved(),
                              peak_allocated_bytes=TORCH.cuda.max_memory_allocated(), peak_reserved_bytes=TORCH.cuda.max_memory_reserved())
                if reset:
                    TORCH.cuda.reset_peak_memory_stats()
            except RuntimeError as exc:
                record["cuda_measurement_error"] = str(exc)
        self.segments.append(record)
        self.phase_name = name
        event("segment", stage=self.label, **record)
        return record

    @contextlib.contextmanager
    def phase(self, name):
        self.mark("before_" + name)
        try:
            yield
        finally:
            self.mark("after_" + name)

    def start(self):
        event("stage_start", stage=self.label)
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def finish(self):
        self.mark("stage_end")
        self.stop.set()
        self.thread.join(timeout=15)
        return {"elapsed_seconds": time.monotonic() - self.started, "peak_rss_bytes_sampled": self.peak_rss,
                "peak_device_used_mib_sampled": self.device_peak_mib, "device_samples": self.device_samples,
                "segments": self.segments, "peak_allocated_bytes": max((x.get("peak_allocated_bytes", 0) for x in self.segments), default=0),
                "peak_reserved_bytes": max((x.get("peak_reserved_bytes", 0) for x in self.segments), default=0),
                "scope": "Child-local allocator segments; 0.2 s target device/RSS sampling may miss transients. Bench CLI resets its own allocator after setup."}


def cuda():
    global TORCH
    import torch
    TORCH = torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible CUDA GPU is required"
    torch.cuda.init()
    assert "T4" in torch.cuda.get_device_name(0), "Use the Kaggle T4 pool"
    CAPTURE.mark("empty_cuda_context")
    return torch


def cfg(seed=42):
    from soup_cli.config.loader import load_config
    return load_config(str(ROOT / f"configs/seed-{seed}.yaml"))


def rows(split):
    return read_jsonl(ROOT / f"data/{split}.jsonl")


def tokenizer(directory=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory or ROOT / "artifacts/base", local_files_only=True, trust_remote_code=False)


def formatter(tok):
    from soup_cli.data.sft_format import build_format_row
    config = cfg(SETTINGS["seeds"][0])
    return build_format_row(tok, config.data, training_cfg=config.training)


def audit_mask(messages, tok, format_fn):
    full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    if len(full) > SETTINGS["max_length"]:
        raise ValueError("complete reference exceeds frozen max_length")
    if full[:len(prompt)] != prompt:
        raise ValueError("inference prefix changed")
    encoded = format_fn({"messages": messages})
    eot = full.index(tok.eos_token_id, len(prompt))
    labels = [-100] * len(prompt) + full[len(prompt):eot+1] + [-100] * (len(full)-eot-1)
    if encoded["input_ids"] != full or encoded["labels"] != labels:
        raise ValueError("exact response/EOT label policy violated")
    preferred = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True)
    if preferred.get("assistant_masks") != [int(x != -100) for x in labels]:
        raise ValueError("generation-mask path unavailable; heuristic fallback forbidden")
    if encoded["attention_mask"] != [1] * len(full) or not any(x != -100 for x in labels[1:]):
        raise ValueError("invalid attention/shifted labels")
    if tok.decode([x for x in labels if x != -100], skip_special_tokens=False, clean_up_tokenization_spaces=False) != messages[-1]["content"] + tok.eos_token:
        raise ValueError("target/EOT text changed")
    return encoded


def model_manifest(path):
    path = Path(path)
    files = {p.name: {"sha256": file_sha(p), "bytes": p.stat().st_size} for p in sorted(path.iterdir())
             if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja", ".txt", ".model"} and p.name != "provenance.json"}
    return {"path": str(path.relative_to(ROOT)), "files": files, "sha256": digest(files)}


def check_bound_artifact(binding, directory):
    actual = model_manifest(directory)
    return actual["sha256"] == binding["sha256"]


def run_soup(arguments, allowed=(0,)):
    from soup_cli.cli import app
    code = 0
    try:
        result = app(args=arguments, standalone_mode=False)
        if type(result) is int:
            code = result
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception as exc:
        import typer
        if isinstance(exc, typer.Exit):
            code = exc.exit_code
        else:
            raise
    if code not in allowed:
        raise RuntimeError(f"Soup {arguments[:3]} exited {code}; allowed={allowed}")
    return code


def base_config_mapping(fragment, seed):
    return {"base": "./artifacts/base", "task": "sft", "backend": "transformers",
            "data": {**fragment["data"], "train": "./data/train.jsonl", "val_split": 0},
            "training": {"epochs": 2, "lr": .0001, "batch_size": 1, "gradient_accumulation_steps": 8,
                "optimizer": "adamw_torch", "scheduler": "cosine", "warmup_ratio": .05, "weight_decay": .01,
                "max_grad_norm": 1., "gradient_checkpointing": True, "quantization": "none", "packing": False,
                "train_on_eot": True, "seed": seed, "data_seed": seed, "logging_steps": 1, "save_steps": 200,
                "lora": {"r": 8, "alpha": 16, "dropout": .05,
                         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}},
            "output": f"./artifacts/training-{seed}"}


def configure_training(trainer):
    from transformers.trainer_callback import ProgressCallback
    from transformers.trainer_utils import IntervalStrategy, SaveStrategy
    for name, value in TRAINER_POLICY.items():
        setattr(trainer.args, name, copy.deepcopy(value))
    trainer.args.eval_strategy = IntervalStrategy(TRAINER_POLICY["eval_strategy"])
    trainer.args.save_strategy = SaveStrategy(TRAINER_POLICY["save_strategy"])
    trainer.model.config.use_cache = False
    # The progress callback is installed during construction, before these overrides.
    trainer.remove_callback(ProgressCallback)


def checkpoint_evidence(trainer, export_dir):
    import numpy as np
    from safetensors import safe_open
    evaluations = [{"epoch": float(row["epoch"]), "step": int(row["step"]), "eval_loss": float(row["eval_loss"])}
                   for row in trainer.state.log_history if "eval_loss" in row]
    epochs = int(trainer.args.num_train_epochs)
    if ({row["epoch"] for row in evaluations} != set(range(1, epochs + 1))
        or not all(math.isfinite(row["eval_loss"]) for row in evaluations)
        or not math.isclose(trainer.state.epoch, epochs)):
        raise ValueError("Missing complete, finite validation history for each training epoch")
    best = min(evaluations, key=lambda row: (row["eval_loss"], row["step"]))
    if (trainer.state.best_global_step != best["step"]
        or trainer.state.best_metric is None
        or not math.isclose(trainer.state.best_metric, best["eval_loss"], rel_tol=1e-7)):
        raise ValueError("Trainer best checkpoint disagrees with validation-loss selection")
    checkpoint = Path(trainer.state.best_model_checkpoint or "").resolve()
    expected = checkpoint / "adapter_model.safetensors"
    exported = Path(export_dir) / "adapter_model.safetensors"
    # Compare tensors: serialization metadata/order can change without changing weights.
    with safe_open(expected, framework="np") as before, safe_open(exported, framework="np") as after:
        if set(before.keys()) != set(after.keys()):
            raise ValueError("Exported adapter keys differ from the best validation checkpoint")
        values = 0
        for name in before.keys():
            left, right = before.get_tensor(name), after.get_tensor(name)
            if left.dtype != right.dtype or not np.array_equal(left, right) or not np.isfinite(right).all():
                raise ValueError("Exported adapter differs from the best validation checkpoint: " + name)
            values += right.size
        tensors = len(before.keys())
    if tensors != 392 or values != 9232384:
        raise ValueError("Unexpected exported LoRA tensor/weight count")
    return {"metric": "eval_loss", "selection_uses_test": False, "epoch_validation": evaluations,
            "epochs_completed": trainer.state.epoch, "optimizer_steps_completed": trainer.state.global_step,
            "selected_epoch": best["epoch"], "selected_step": best["step"], "best_validation_loss": best["eval_loss"],
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_adapter_sha256": file_sha(expected),
            "export_matches_checkpoint_tensors": True, "trainable_tensor_count": tensors, "trainable_parameter_count": values}


def group_metadata(all_rows):
    parent = list(range(len(all_rows)))
    owners = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(all_rows):
        keys = [("request", " ".join(unicodedata.normalize("NFKC", row["messages"][1]["content"]).casefold().split()))]
        keys += [("tool", name.casefold()) for name in row["functions"]]
        for key in keys:
            if key in owners:
                a, b = find(i), find(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = i
    components = collections.defaultdict(list)
    for i, row in enumerate(all_rows):
        components[find(i)].append(row)
    for members in components.values():
        if len({row["split"] for row in members}) != 1:
            raise ValueError("cross-split tool/request family overlap")
        group = digest(sorted(row["id"] for row in members))
        for row in members:
            row["group"] = group
    return all_rows


def challenge_rows():
    tools = [
        {"type": "function", "function": {"name": "check_city_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}}, "required": ["city", "unit"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "set_alert_enabled", "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "multiply_values", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"], "additionalProperties": False}}},
    ]
    cases = [("Get the weather for Paris in celsius.", [{"name": "check_city_weather", "arguments": {"city": "Paris", "unit": "celsius"}}]),
             ("Get the weather for Tokyo in fahrenheit.", [{"name": "check_city_weather", "arguments": {"city": "Tokyo", "unit": "fahrenheit"}}]),
             ("Enable the alert.", [{"name": "set_alert_enabled", "arguments": {"enabled": True}}]),
             ("Disable the alert.", [{"name": "set_alert_enabled", "arguments": {"enabled": False}}]),
             ("Multiply 0 by 7.", [{"name": "multiply_values", "arguments": {"a": 0, "b": 7}}]),
             ("First get Paris weather in celsius, then disable the alert.", [
                 {"name": "check_city_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
                 {"name": "set_alert_enabled", "arguments": {"enabled": False}}])]
    system = "Return only the requested function calls, in order, within <tool_call> tags. Available tools: <tools>" + canonical(tools) + "</tools>"
    return [{"messages": [{"role": "system", "content": system}, {"role": "user", "content": query},
                          {"role": "assistant", "content": tagged(expected)}]} for query, expected in cases]


def prepare_stage():
    import yaml
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from soup_cli.config.loader import load_config_from_string
    direct_url = json.loads(metadata.distribution("soup-cli").read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info",{}).get("commit_id") != SOUP_REV:
        raise ValueError("Installed Soup source commit does not match")
    environment = {"python":sys.version,"versions":{name:metadata.version(name) for name in
        ("soup-cli","torch","transformers","peft","trl","bitsandbytes","accelerate","jsonschema")},
        "soup_install":direct_url,"runtime_sha256":file_sha(__file__)}
    write_json(ROOT/"evidence/environment.json",environment)
    source = Path(SETTINGS["dataset_dir"])
    for name, sha in EXPECTED_DATA.items():
        if file_sha(source / name) != sha:
            raise ValueError(f"Dataset identity changed: {name}; freeze/review a new dataset version explicitly")
        target = ROOT / ("data/" + name if name.endswith("jsonl") else "soup-data.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    fragment = yaml.safe_load((ROOT / "soup-data.yaml").read_text())
    if fragment["data"]["val_split"] != 0 or not fragment["data"]["train_on_responses_only"]:
        raise ValueError("Prepared split/mask settings changed")
    SETTINGS["max_length"] = fragment["data"]["max_length"]
    write_json(ROOT / "settings.json", SETTINGS)
    base = ROOT / "artifacts/base"
    snapshot_download(MODEL_ID, revision=MODEL_REV, local_dir=str(base),
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE"])
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    native = tok.chat_template
    (ROOT / "evidence/native-template.jinja").write_text(native)
    tok.chat_template = fragment["data"]["chat_template"]
    tok.padding_side = "right"
    tok.save_pretrained(base)
    if not (ROOT / "configs").exists():
        (ROOT / "configs").mkdir()
    for seed in SETTINGS["seeds"]:
        mapping = base_config_mapping(fragment, seed)
        text = yaml.safe_dump(mapping, sort_keys=False)
        load_config_from_string(text)
        (ROOT / f"configs/seed-{seed}.yaml").write_text(text)
    config_hashes = {str(p.relative_to(ROOT)):file_sha(p) for p in (ROOT/"configs").glob("seed-*.yaml")}
    format_fn = formatter(tok)
    all_rows = []
    counts = {}
    for split in ("train", "validation", "test"):
        raw = rows(split)
        counts[split] = len(raw)
        for item in raw:
            messages = item["messages"]
            if any(token in message["content"] for token in tok.all_special_tokens for message in messages):
                raise ValueError("reserved tokenizer token in source content")
            functions = functions_from_messages(messages)
            expected = parse_calls(messages[-1]["content"])
            score = score_turn(messages[-1]["content"], expected, functions)
            if not score["complete"]:
                raise ValueError("invalid prepared reference")
            for prefix, generation in ((messages, False), (messages[:-1], True)):
                if tok.apply_chat_template(prefix, chat_template=native, tokenize=False, add_generation_prompt=generation) != tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=generation):
                    raise ValueError("marked template changes native Qwen rendering")
            encoded = audit_mask(messages, tok, format_fn)
            all_rows.append({"id": digest(messages), "split": split, "messages": messages, "functions": functions,
                             "expected": expected, "call_count": len(expected), "reference_tokens": len(encoded["input_ids"]),
                             "supervised_tokens": sum(x != -100 for x in encoded["labels"][1:])})
    if not 1000 <= counts["train"] <= 2000 or counts["validation"] < 100 or counts["test"] < 100:
        raise ValueError("Prepared corpus does not meet experiment size requirements")
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("duplicate example IDs")
    group_metadata(all_rows)
    write_jsonl(ROOT / "evidence/references.jsonl", all_rows)
    write_jsonl(ROOT / "data/challenge.jsonl", challenge_rows())
    validation = [row for row in all_rows if row["split"] == "validation"]
    review_sample = sorted(validation,key=lambda row:digest([42,"reference-review",row["id"]]))[:12]
    write_jsonl(ROOT/"evidence/validation-reference-review.jsonl",
                [{"id":row["id"],"messages":row["messages"]} for row in review_sample])
    longest_target = max(row["supervised_tokens"] for row in validation)
    generation_cap = int(math.ceil((longest_target + 32) / 32) * 32)
    if generation_cap > 1024:
        raise ValueError("Validation requires >1024 new tokens; revise policy before training")
    policy = {"created_utc": now(), "dataset": "pernavjain/tool-calling", "dataset_hashes": EXPECTED_DATA,
        "model": MODEL_ID, "model_revision": MODEL_REV, "soup_commit": SOUP_REV,
        "seeds": SETTINGS["seeds"], "allocator_config": SETTINGS["allocator_config"],
        "selection": "within seed: lowest epoch validation loss; across seeds: validation complete-turn accuracy, then token-weighted NLL, then seed",
        "trainer_policy": TRAINER_POLICY,
        "generation_max_new_tokens": generation_cap, "generation_cap_source": "validation target lengths only +32, rounded to32",
        "max_length": SETTINGS["max_length"], "bootstrap_replicates": 10000, "bootstrap_seed": 42,
        "gain_min": .05, "quant_loss_max": .02, "quant_ci_lower_min": -.05, "general_regression_max": .05,
        "headroom_bytes_min": 2**30, "merge_logprob_ceiling": .10, "merge_logprob_floor_budget": .02,
        "merge_tolerance_formula": "max(0.02,10*measured_repeat_and_noop_floor), capped at0.10; frozen before test",
        "merge_max_kl":.02,"weight_relative_residual_max":.25,"weight_absolute_residual_max":.001,
        "base_disabled_logprob_floor":.002,"minimum_adapter_or_q_effect":1e-5,"effect_floor_multiplier":10,
        "general_max_new_tokens":256,
        "probe_rows": 4, "probe_positions": 4, "controls_required": True,
        "config_hashes":config_hashes,
        "source_labels": "not included in the four-file export; no invented per-source result slices",
        "scope": "call-only single-turn function calling; structural reference audit, not semantic adjudication"}
    write_json(ROOT / "run_policy.json", policy)
    base_identity = model_manifest(base)
    write_json(ROOT / "evidence/base-identity.json", base_identity)
    initial_estimate = {"created_before_training": now(), "unique_parameters": 1543714304,
        "lora_parameters_expected": 9232384, "base_fp16_gib": 1543714304 * 2 / 2**30,
        "lora_fp32_parameter_gradient_adam_gib": 9232384 * 16 / 2**30,
        "one_fp16_full_logits_gib": SETTINGS["max_length"] * 151936 * 2 / 2**30,
        "one_fp32_full_logits_gib": SETTINGS["max_length"] * 151936 * 4 / 2**30,
        "kv_cache_bytes_per_token_batch1": 28 * 2 * 2 * 128 * 2,
        "nf4_storage_floor_gib": (1310195712 * .5 + 233373696 * 2) / 2**30,
        "not_in_floor": ["checkpointed/recomputed activations", "attention/logit/loss workspaces", "allocator caches/context", "NF4 metadata", "host staging"],
        "training_memory_preflight_budget_gib": 12.0,
        "budget_label": "conservative planning envelope, not a prediction of actual peak",
        "phase_model": "baseline/base+LoRA residency; forward/backward; lazy Adam state; eval/save; CPU merge; GPU NF4; cold load+KV inference"}
    write_json(ROOT / "evidence/memory-estimate-before.json", initial_estimate)
    training = base_config_mapping(fragment, SETTINGS["seeds"][0])["training"]
    steps_per_epoch = math.ceil(counts["train"] / (training["batch_size"] * training["gradient_accumulation_steps"]))
    write_json(ROOT / "evidence/training-plan.json", {
        "training": training, "trainer_policy": TRAINER_POLICY, "seeds": SETTINGS["seeds"],
        "examples_per_epoch": counts["train"], "optimizer_steps_per_epoch": steps_per_epoch,
        "optimizer_steps_per_seed": steps_per_epoch * training["epochs"],
        "rationale": "Two passes over the curated 1600-example set, lower 1e-4 learning rate and 5% warmup; retain the better validation-loss epoch. This recipe is not claimed optimal.",
        "training_quantization": "none: FP16 base with LoRA; NF4 is a separate post-merge deployment experiment",
    })
    short = min(validation, key=lambda x: x["reference_tokens"])
    fixtures = {"malformed_call": copy.deepcopy(short["messages"]), "missing_tools": copy.deepcopy(short["messages"]),
                "double_template": copy.deepcopy(short["messages"]), "length_boundary": copy.deepcopy(max(validation, key=lambda x: x["reference_tokens"])["messages"])}
    fixtures["malformed_call"][-1]["content"] = "<tool_call>{bad json}</tool_call>"
    fixtures["missing_tools"][0]["content"] = "You are a helpful assistant."
    fixtures["double_template"][1]["content"] = "<|im_start|>user\n" + fixtures["double_template"][1]["content"] + "<|im_end|>"
    for name, messages in fixtures.items():
        write_jsonl(ROOT / f"controls/{name}.jsonl", [{"messages": messages}])
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    train = [row for row in all_rows if row["split"] == "train"]
    encoded = [format_fn({"messages": row["messages"]}) for row in (min(train,key=lambda x:x["reference_tokens"]),max(train,key=lambda x:x["reference_tokens"]))]
    assert len(encoded[0]["input_ids"]) < len(encoded[1]["input_ids"])
    for pad in (tok.pad_token_id, tok.eos_token_id):
        batch = DataCollatorForLanguageModeling(pad_token_id=pad, padding_free=False)(encoded)
        for i, row in enumerate(encoded):
            n = len(row["input_ids"])
            assert batch["labels"][i,:n].tolist() == row["labels"]
            assert (batch["labels"][i,n:] == -100).all().item()
    summary = {"counts": counts, "max_length": SETTINGS["max_length"], "generation_cap": generation_cap,
        "all_reference_masks_exact": True, "reserved_token_check": True, "split_overlap": 0, "collator_padding_checks": True,
        "native_template_sha256": digest(native), "training_template_sha256": digest(tok.chat_template),
        "group_counts": {s:len({x["group"] for x in all_rows if x["split"] == s}) for s in counts}}
    write_json(ROOT / "evidence/data-ready.json", summary)
    return {"outputs": ["evidence/data-ready.json", "evidence/base-identity.json", "run_policy.json",
                        "evidence/references.jsonl", "data/challenge.jsonl", "evidence/memory-estimate-before.json",
                        "evidence/training-plan.json"], **summary}


def clean_observation(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_observation": repr(value)}
    if isinstance(value, dict):
        return {key: clean_observation(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_observation(item) for item in value]
    return value


def reference_rows(split,tok=None):
    if split != "challenge":
        return [x for x in read_jsonl(ROOT / "evidence/references.jsonl") if x["split"] == split]
    tok = tok or tokenizer()
    fmt = formatter(tok)
    result = []
    for item in rows("challenge"):
        messages = item["messages"]
        encoded = audit_mask(messages, tok, fmt)
        result.append({"id": digest(messages), "group": digest(messages), "split": split, "messages": messages,
                       "functions": functions_from_messages(messages), "expected": parse_calls(messages[-1]["content"]),
                       "call_count": len(parse_calls(messages[-1]["content"])), "reference_tokens": len(encoded["input_ids"])})
    return result


def probe_rows():
    candidates = reference_rows("validation")
    ordered = sorted(candidates, key=lambda x: (x["reference_tokens"], x["id"]))
    selected = [ordered[0], ordered[-1], max(candidates, key=lambda x:x["call_count"])]
    selected += sorted(candidates, key=lambda x:digest([42,x["id"]]))
    unique = {row["id"]: row for row in selected}
    return list(unique.values())[:read_json(ROOT / "run_policy.json")["probe_rows"]]


def encoded_probe(model, tok, row):
    torch = TORCH
    encoded = audit_mask(row["messages"], tok, formatter(tok))
    batch = {key: torch.tensor([value], device="cuda") for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode():
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        positions = torch.nonzero(batch["labels"][0,1:] != -100).flatten()
        targets = batch["labels"][0,positions + 1]
        log_probs = output.logits[0,positions,:].float().log_softmax(-1)
        reference = log_probs.gather(-1,targets[:,None]).flatten()
        n = min(read_json(ROOT / "run_policy.json")["probe_positions"],len(positions))
        choice = torch.linspace(0,len(positions)-1,n,device="cuda").long().unique()
        return {"id": row["id"], "nll": float(-reference.mean()), "reference_log_probs": reference.cpu().tolist(),
                "positions": positions[choice].cpu().tolist(), "distribution": log_probs[choice].cpu()}


def probe_model(model, tok):
    return [encoded_probe(model, tok, row) for row in probe_rows()]


def compare_probes(before, after):
    torch = TORCH
    if not before or len(before) != len(after):
        raise ValueError("probe counts differ")
    errors, nll_errors, kls = [], [], []
    for a,b in zip(before,after):
        if a["id"] != b["id"] or a["positions"] != b["positions"]:
            raise ValueError("probe identities differ")
        if len(a["reference_log_probs"]) != len(b["reference_log_probs"]) or a["distribution"].shape != b["distribution"].shape:
            raise ValueError("probe dimensions differ")
        errors.extend(abs(x-y) for x,y in zip(a["reference_log_probs"],b["reference_log_probs"]))
        nll_errors.append(abs(a["nll"]-b["nll"]))
        kls.extend((a["distribution"].exp() * (a["distribution"]-b["distribution"])).sum(-1).tolist())
    if not all(math.isfinite(v) for v in errors+nll_errors+kls):
        raise ValueError("nonfinite probe comparison")
    return {"max_reference_logprob_error": max(errors), "mean_reference_logprob_error": statistics.mean(errors),
            "max_row_nll_difference": max(nll_errors), "mean_kl_before_after": statistics.mean(kls),
            "max_kl_before_after": max(kls), "rows": len(before), "reference_tokens": len(errors), "distribution_positions":len(kls)}


def frozen_fingerprints(model):
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and re.search(r"layers\.(0|27)\.self_attn\.(q_proj|v_proj)\.base_layer\.weight$", name):
            result[name] = hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if len(result) != 4:
        raise ValueError("unexpected architecture: selected frozen projections not found")
    return result


def train_stage(seed):
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.bench.collector import BenchCollector
    from transformers import TrainerCallback, set_seed
    from dataclasses import asdict
    benchmark_path = ROOT / "evidence/stages/bench-train.json"
    benchmark = read_json(benchmark_path) if benchmark_path.is_file() else {}
    if (benchmark.get("settings_sha256") != digest(SETTINGS)
        or benchmark.get("runtime_sha256") != file_sha(__file__)
        or not benchmark_headroom(benchmark, SETTINGS["allocator_config"])["passed"]):
        raise RuntimeError("Primary training requires a passing T4 scratch memory gate under the same allocator policy.")
    torch = cuda()
    set_seed(seed)
    if not check_bound_artifact(read_json(ROOT/"evidence/base-identity.json"),artifact_path("B")):
        raise ValueError("Pristine base bytes changed before training")
    config = cfg(seed)
    out = ROOT / f"artifacts/adapter-{seed}"
    if out.exists():
        raise ValueError("Adapter output already exists; choose a new run after an interrupted training stage")
    train_data = load_dataset(config.data)["train"]
    val_data = load_dataset(config.data.model_copy(update={"train":"./data/validation.jsonl","val_split":0}))["train"]
    for split,data in (("train",train_data),("validation",val_data)):
        if [digest(x["messages"]) for x in data] != [digest(x["messages"]) for x in rows(split)]:
            raise ValueError("Soup loader changed prepared split membership/order")
    with CAPTURE.phase("base_load_and_lora_setup"):
        wrapper = SFTTrainerWrapper(config, device="cuda", report_to="none")
        wrapper.setup({"train":train_data,"val":val_data})
        trainer = wrapper.trainer
        configure_training(trainer)
        model = trainer.model
        assert trainer.args.fp16 and not trainer.args.bf16 and trainer.args.per_device_train_batch_size == 1
        trainables = {name:p for name,p in model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in trainables.values()) == 9232384
        assert sum(p.numel() for p in model.parameters()) - 9232384 == 1543714304
        assert all("lora_" in name for name in trainables)
        assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
        assert all(not p.is_meta and p.device.type == "cuda" for p in model.parameters())
        before = {name:p.detach().cpu().clone() for name,p in trainables.items()}
        frozen_before = frozen_fingerprints(model)
        initial_dir = ROOT / f"artifacts/initial-{seed}"
        model.save_pretrained(initial_dir, safe_serialization=True)
        wrapper.tokenizer.save_pretrained(initial_dir)
    with CAPTURE.phase("initial_noop_and_numerical_floor"):
        first = seed == SETTINGS["seeds"][0]
        with model.disable_adapter():
            base_probe = probe_model(model, wrapper.tokenizer)
            repeated = [probe_model(model, wrapper.tokenizer) for _ in range(2)] if first else []
        zero_probe = probe_model(model, wrapper.tokenizer)
        zero_error = compare_probes(base_probe,zero_probe)["max_reference_logprob_error"]
        repeat_error = max((compare_probes(base_probe,x)["max_reference_logprob_error"] for x in repeated),default=0.)
        if zero_error > .02:
            raise ValueError("Initialized adapter is not a no-op within the predeclared floor ceiling")
        floor = max(zero_error,repeat_error)
        if first:
            calibration = {"utc_before_training":now(),"repeat_base_runs":3,"repeat_error":repeat_error,
                "initialized_adapter_error":zero_error,"floor":floor,"merge_reference_logprob_tolerance":max(.02,10*floor),
                "ceiling":.10,"scope":"four fixed validation rows; deterministic baseline under actual kernels"}
            if calibration["merge_reference_logprob_tolerance"] > calibration["ceiling"]:
                raise ValueError("Numerical instrument exceeds frozen tolerance ceiling")
            write_json(ROOT / "evidence/numerical-calibration.json",calibration)
    collector = BenchCollector(warmup_steps=3)
    collector._sync = torch.cuda.synchronize
    trainer.add_callback(collector)
    observed = {"microbatches":0,"shifted_supervised_tokens":0,"nonpadding_tokens":0,"padded_tokens":0,
                "optimizer_callbacks":0,"successful_nonzero_lr_steps":0,"amp_skipped_steps":0,"first_materialized_update":None}
    probe_parameter = next(p for name,p in trainables.items() if "lora_B" in name)
    first_forward_hook = None
    original_step = trainer.training_step

    def training_step(current_model, inputs, *args, **kwargs):
        collector.observe_batch(inputs)
        observed["microbatches"] += 1
        observed["shifted_supervised_tokens"] += int((inputs["labels"][:,1:] != -100).sum().item())
        observed["nonpadding_tokens"] += int(inputs["attention_mask"].sum().item())
        observed["padded_tokens"] += inputs["input_ids"].numel()
        value = original_step(current_model,inputs,*args,**kwargs)
        if observed["microbatches"] == 1:
            CAPTURE.mark("first_backward_complete")
        return value
    trainer.training_step = training_step

    def first_forward(_model,_inputs,_output):
        CAPTURE.mark("first_forward_complete")
        first_forward_hook.remove()
    first_forward_hook = model.register_forward_hook(first_forward)

    class UpdateEvidence(TrainerCallback):
        def __init__(self):
            self.sample = None
            self.lr = None
            self.first_optimizer = True

        def on_train_begin(self,args,state,control,**kwargs):
            optimizer = kwargs["optimizer"]
            wanted = {id(p) for p in model.parameters() if p.requires_grad}
            actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
            assert wanted == actual, "Optimizer membership does not match trainable parameters"
            observed["optimizer_membership_exact"] = True

        def on_pre_optimizer_step(self,args,state,control,**kwargs):
            self.lr = max(group["lr"] for group in kwargs["optimizer"].param_groups)
            if observed["first_materialized_update"] is None and self.lr > 0:
                self.sample = probe_parameter.detach().clone()
            if self.first_optimizer:
                CAPTURE.mark("before_first_optimizer_step")

        def on_optimizer_step(self,args,state,control,**kwargs):
            observed["optimizer_callbacks"] += 1
            skipped = bool(trainer.accelerator.optimizer_step_was_skipped)
            observed["amp_skipped_steps"] += int(skipped)
            if not skipped and self.lr > 0:
                observed["successful_nonzero_lr_steps"] += 1
                if self.sample is not None:
                    delta = float((probe_parameter.detach()-self.sample).abs().max())
                    if delta > 0 and observed["first_materialized_update"] is None:
                        observed["first_materialized_update"] = {"global_step_before_increment":state.global_step,"lr":self.lr,"max_abs_delta":delta}
            self.sample = None
            if self.first_optimizer:
                state_bytes = collections.Counter()
                for state_values in kwargs["optimizer"].state.values():
                    for value in state_values.values():
                        if isinstance(value,torch.Tensor):
                            state_bytes[str(value.dtype)+"@"+str(value.device)] += value.numel()*value.element_size()
                observed["optimizer_state_bytes_after_first_attempt"] = dict(state_bytes)
                CAPTURE.mark("after_first_optimizer_step")
                self.first_optimizer = False

        def on_log(self,args,state,control,logs=None,**kwargs):
            event("training_log",seed=seed,step=state.global_step,logs=clean_observation(logs or {}))
            if state.global_step and state.global_step % 20 == 0:
                event("progress",stage=f"train-{seed}",optimizer_step=state.global_step)

        def on_evaluate(self,args,state,control,**kwargs):
            CAPTURE.mark(f"evaluation_complete_step_{state.global_step}")

        def on_save(self,args,state,control,**kwargs):
            CAPTURE.mark(f"checkpoint_saved_step_{state.global_step}")

    trainer.add_callback(UpdateEvidence())
    original_save = trainer.save_model
    def observed_save(*args,**kwargs):
        with CAPTURE.phase("model_save"):
            return original_save(*args,**kwargs)
    trainer.save_model = observed_save
    with CAPTURE.phase("primary_training"):
        summary = wrapper.train(run_id=f"seed-{seed}")
    # Keep Trainer optimizer/checkpoint files outside the clean deployable adapter.
    with CAPTURE.phase("clean_adapter_export"):
        model.save_pretrained(out,safe_serialization=True)
        wrapper.tokenizer.save_pretrained(out)
    checkpoint = checkpoint_evidence(trainer, out)
    assert collector.steps and observed["first_materialized_update"] is not None
    assert observed["successful_nonzero_lr_steps"] > 0 and observed["optimizer_membership_exact"]
    assert all(torch.isfinite(p).all().item() for p in trainables.values())
    deltas = {name:float((p.detach().cpu().float()-before[name].float()).abs().max()) for name,p in trainables.items()}
    assert max(deltas.values()) > 0
    frozen_after = frozen_fingerprints(model)
    assert frozen_after == frozen_before
    with CAPTURE.phase("trained_adapter_on_off"):
        active = probe_model(model,wrapper.tokenizer)
        with model.disable_adapter():
            disabled = probe_model(model,wrapper.tokenizer)
        effect = compare_probes(disabled,active)
        disabled_error = compare_probes(base_probe,disabled)
    initial_manifest = model_manifest(initial_dir)
    final_manifest = model_manifest(out)
    assert initial_manifest["sha256"] != final_manifest["sha256"]
    counted = collector.steps[3:]
    result = {"seed":seed,"training_summary":summary,"checkpoint_selection":checkpoint,
        "trainer_log_history":clean_observation(trainer.state.log_history),
        "trainable_shapes_dtypes":{name:{"shape":list(p.shape),"dtype":str(p.dtype),"device":str(p.device)} for name,p in trainables.items()},
        "base_unique_parameters":1543714304,"lora_parameters":9232384,"embedding_and_head_tied":True,
        "resolved_training_args":trainer.args.to_dict(),"observations":observed,"changed_adapter_tensors":sum(x>0 for x in deltas.values()),
        "max_parameter_delta":max(deltas.values()),"frozen_projection_hashes_before":frozen_before,"frozen_projection_hashes_after":frozen_after,
        "adapter_effect":effect,"disabled_vs_pristine":disabled_error,"initial":initial_manifest,"final":final_manifest,
        "bench_collector":{"steps":[asdict(x) for x in collector.steps],"param_snapshot":asdict(collector.param_snapshot()),
            "postwarmup_seconds":sum(x.wall_seconds for x in counted),"postwarmup_useful_tokens_unshifted":sum(x.useful_tokens for x in counted),
            "timing_scope":"instrumented synchronized optimizer intervals, including between-step checkpoint/data overhead; full wall time separate"}}
    result = clean_observation(result)
    destination = ROOT / f"evidence/training-{seed}.json"
    write_json(destination,result)
    write_json(out / "provenance.json",{"seed":seed,"model_revision":MODEL_REV,"base_identity":read_json(ROOT/"evidence/base-identity.json")["sha256"],
        "training_data_sha256":EXPECTED_DATA["train.jsonl"],"soup_commit":SOUP_REV,"runtime_sha256":file_sha(__file__),"adapter_sha256":final_manifest["sha256"]})
    return {"outputs":[str(destination.relative_to(ROOT)),str((out/"provenance.json").relative_to(ROOT)),
                       str((out/"adapter_model.safetensors").relative_to(ROOT)),str((initial_dir/"adapter_model.safetensors").relative_to(ROOT))],
            "global_steps":trainer.state.global_step,"selected_epoch":checkpoint["selected_epoch"],"updated":True,"seed":seed}


def artifact_path(variant,seed=None):
    if variant == "B":
        return ROOT / "artifacts/base"
    if variant in {"A","A0"}:
        return ROOT / f"artifacts/{'adapter' if variant=='A' else 'initial'}-{seed}"
    return ROOT / {"M":"artifacts/merged-fp16","Q":"artifacts/merged-nf4","Q0":"artifacts/control-q0"}[variant]


def load_variant(variant, seed=None, portable=False):
    from transformers import AutoModelForCausalLM
    path = artifact_path(variant,seed)
    if variant == "B":
        bound = read_json(ROOT/"evidence/base-identity.json")
    elif variant in {"A","A0"}:
        training = read_json(ROOT/f"evidence/training-{seed}.json")
        bound = training["final" if variant == "A" else "initial"]
        if not check_bound_artifact(read_json(ROOT/"evidence/base-identity.json"),artifact_path("B")):
            raise ValueError("Adapter base bytes changed")
    else:
        bound = read_json(ROOT/f"evidence/artifact-{variant}.json")["artifact"]
    if not check_bound_artifact(bound,path):
        raise ValueError("Artifact bytes drifted from their recorded identity")
    if variant in {"A","A0"}:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(artifact_path("B"),dtype=TORCH.float16,device_map={"":"cuda:0"},local_files_only=True)
        model = PeftModel.from_pretrained(base,path,is_trainable=False,local_files_only=True)
    else:
        if portable:
            isolated = ROOT / "artifacts/portable" / variant
            isolated.mkdir(parents=True,exist_ok=True)
            for file in path.iterdir():
                if file.is_file() and file.name != "provenance.json":
                    target = isolated / file.name
                    if not target.exists():
                        os.link(file,target)
            forbidden = [str((ROOT/"artifacts/base").resolve())] + [str(p.resolve()) for p in (ROOT/"artifacts").glob("adapter-*")]
            def deny_base_reads(event_name,args):
                if event_name == "open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
                    candidate = os.fsdecode(args[0])
                    if any(candidate == prefix or candidate.startswith(prefix+os.sep) for prefix in forbidden):
                        raise RuntimeError("Portable load attempted to read a base/adapter path")
            sys.addaudithook(deny_base_reads)
            path = isolated
        model = AutoModelForCausalLM.from_pretrained(path,dtype=TORCH.float16,device_map={"":"cuda:0"},local_files_only=True)
    model.eval()
    if not all(not p.is_meta and p.device.type == "cuda" for p in model.parameters()):
        raise ValueError("CPU/disk/meta fallback detected")
    from bitsandbytes.nn import Linear4bit
    quantized = sum(isinstance(module,Linear4bit) for module in model.modules())
    if variant in {"Q","Q0"}:
        quant = model.config.quantization_config
        quant = quant.to_dict() if hasattr(quant,"to_dict") else quant
        assert quantized == 196 and quant["load_in_4bit"] and quant["bnb_4bit_quant_type"] == "nf4" and quant["bnb_4bit_use_double_quant"]
        assert all(module.weight.quant_state is not None for module in model.modules() if isinstance(module,Linear4bit))
        assert all(module.compute_dtype == TORCH.float16 and getattr(module.weight.quant_state,"state2",None) is not None
                   for module in model.modules() if isinstance(module,Linear4bit))
    else:
        quant = None
        assert quantized == 0
    info = {"variant":variant,"resolved_path":str(path),"class":type(model).__name__,"quantized_linear_count":quantized,
            "quantization_config":quant,"devices":sorted({str(p.device) for p in model.parameters()}),
            "parameter_dtypes":dict(collections.Counter(str(p.dtype) for p in model.parameters())),"portable_base_adapter_reads_blocked":portable}
    return model, info


def generate(model,tok,messages,max_new_tokens,general=False):
    template = (ROOT/"evidence/native-template.jinja").read_text() if general else tok.chat_template
    encoded = tok.apply_chat_template(messages,chat_template=template,tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True).to("cuda")
    TORCH.cuda.synchronize()
    start = time.monotonic()
    with TORCH.inference_mode():
        output = model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
                                eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
    TORCH.cuda.synchronize()
    generated = output[0,encoded["input_ids"].shape[-1]:].cpu().tolist()
    ended = bool(generated and generated[-1] == tok.eos_token_id)
    text_ids = generated[:-1] if ended else generated
    return {"text":tok.decode(text_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            "raw_text":tok.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            "generated_ids":generated,"generated_tokens":len(generated),"prompt_tokens":encoded["input_ids"].shape[-1],
            "truncated":not ended,"generation_seconds":time.monotonic()-start,"stop":"eot" if ended else "length_or_other_stop"}


def teacher_nll(model,tok,messages):
    batch = audit_mask(messages,tok,formatter(tok))
    tensors = {key:TORCH.tensor([value],device="cuda") for key,value in batch.items()}
    with TORCH.inference_mode():
        output = model(input_ids=tensors["input_ids"],attention_mask=tensors["attention_mask"],use_cache=False)
        positions = TORCH.nonzero(tensors["labels"][0,1:] != -100).flatten()
        targets = tensors["labels"][0,positions+1]
        probabilities = output.logits[0,positions,:].float().log_softmax(-1)
        loss_sum = float(-probabilities.gather(-1,targets[:,None]).sum())
    if not math.isfinite(loss_sum):
        raise ValueError("nonfinite assistant NLL")
    return loss_sum,len(positions)


def evaluate_stage(variant,split,seed=None,general=False,probes_only=False):
    from soup_cli.eval.custom import tool_call_match
    cuda()
    if split == "test" and not (ROOT/"evidence/policy-frozen-before-test.json").exists():
        raise ValueError("Freeze selection, calibration and policy before test evaluation")
    if split == "test":
        frozen = read_json(ROOT/"evidence/policy-frozen-before-test.json")
        if file_sha(ROOT/"run_policy.json") != frozen["policy_sha256"] or file_sha(ROOT/"data/test.jsonl") != frozen["test_hash"]:
            raise ValueError("Frozen test/policy bytes changed")
        if file_sha(ROOT/"evidence/references.jsonl") != frozen["references_hash"]:
            raise ValueError("Frozen evaluation reference bytes changed")
    tok = tokenizer(artifact_path(variant)) if variant in {"M","Q","Q0"} else tokenizer()
    items = reference_rows(split,tok) if not probes_only else []
    with CAPTURE.phase("cold_load"):
        model, identity = load_variant(variant,seed,portable=variant in {"M","Q"})
    label = variant + (f"-{seed}" if seed is not None else "")
    outputs = []
    if split == "validation":
        with CAPTURE.phase("fixed_validation_distribution_probes"):
            probes = probe_model(model,tok)
            probe_path = ROOT / f"evidence/probes/{split}-{label}.pt"
            probe_path.parent.mkdir(parents=True,exist_ok=True)
            TORCH.save(probes,probe_path)
        outputs.append(str(probe_path.relative_to(ROOT)))
    with CAPTURE.phase("generation_warmup_excluded_from_latency"):
        generate(model,tok,probe_rows()[0]["messages"][:-1],8)
    if not probes_only:
        predictions = []
        policy = read_json(ROOT/"run_policy.json")
        path = ROOT / f"evidence/predictions/{split}-{label}.jsonl"
        path.parent.mkdir(parents=True,exist_ok=True)
        with CAPTURE.phase("teacher_forcing_and_greedy_generation"),path.open("w") as handle:
            for i,row in enumerate(items):
                loss_sum,count = teacher_nll(model,tok,row["messages"])
                generated = generate(model,tok,row["messages"][:-1],policy["generation_max_new_tokens"])
                scored = score_turn(generated["text"],row["expected"],row["functions"],generated["truncated"],tool_call_match)
                record = {"id":row["id"],"group":row["group"],"variant":variant,"seed":seed,
                          "call_count":row["call_count"],"reference_tokens":row["reference_tokens"],
                          "nll_sum":loss_sum,"nll_tokens":count,**generated,**scored}
                handle.write(canonical(record)+"\n");handle.flush()
                predictions.append(record)
                if (i+1)%10 == 0:
                    event("progress",stage=f"eval-{split}-{label}",completed=i+1,total=len(items))
        summary = {"identity":identity,"split":split,"variant":variant,"seed":seed,**summarize_predictions(predictions)}
        destination = ROOT / f"evidence/scores/{split}-{label}.json"
        write_json(destination,summary)
        outputs += [str(path.relative_to(ROOT)),str(destination.relative_to(ROOT))]
    if general:
        from soup_cli.eval.gate_suites import DEFAULT_GENERAL_SUITE, BUNDLED_SCORER_REVISION, BUNDLED_SCORER_FINGERPRINT, score_bundled_suite
        suites = {}
        raw = []
        generation_errors = []
        with CAPTURE.phase("bundled_general_suites"):
            for suite in DEFAULT_GENERAL_SUITE:
                def gen(prompt):
                    try:
                        result = generate(model,tok,[{"role":"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                                     {"role":"user","content":prompt}],256,general=True)
                    except Exception as exc:
                        generation_errors.append({"suite":suite,"prompt":prompt,"error":f"{type(exc).__name__}: {exc}"})
                        return None
                    raw.append({"suite":suite,"prompt":prompt,**result})
                    return None if result["truncated"] else result["text"]
                suites[suite] = score_bundled_suite(suite,gen)
                event("progress",stage=f"general-{label}",suite=suite,score=suites[suite])
        destination = ROOT / f"evidence/general/{label}.json"
        write_jsonl(destination.with_suffix(".jsonl"),[*raw,*generation_errors])
        if generation_errors:
            raise RuntimeError(f"{len(generation_errors)} bundled-suite generation errors; see {destination.with_suffix('.jsonl')}")
        write_json(destination,{"scores":suites,"scorer_revision":BUNDLED_SCORER_REVISION,"scorer_fingerprint":BUNDLED_SCORER_FINGERPRINT,
                                "counts":dict(collections.Counter(x["suite"] for x in raw)),"generation_budget":256,"native_template":True})
        outputs += [str(destination.relative_to(ROOT)),str(destination.with_suffix(".jsonl").relative_to(ROOT))]
    write_json(ROOT/f"evidence/identity-{label}.json",identity)
    return {"outputs":[*outputs,f"evidence/identity-{label}.json"],"identity":identity}


def select_stage():
    candidates = []
    for seed in SETTINGS["seeds"]:
        score = read_json(ROOT/f"evidence/scores/validation-A-{seed}.json")
        training = read_json(ROOT/f"evidence/training-{seed}.json")
        assert training["observations"]["first_materialized_update"] is not None
        candidates.append({"seed":seed,"complete":score["complete"],"nll":score["mean_assistant_nll"],
                           "selected_epoch":training["checkpoint_selection"]["selected_epoch"],
                           "best_validation_loss":training["checkpoint_selection"]["best_validation_loss"],
                           "adapter_sha256":training["final"]["sha256"]})
    selected = sorted(candidates,key=lambda x:(-x["complete"],x["nll"],x["seed"]))[0]
    result = {"selected":selected,"all_seeds":candidates,"selection_uses_test":False,"utc":now()}
    write_json(ROOT/"evidence/selection.json",result)
    return {"outputs":["evidence/selection.json"],**result}


def merge_stage(variant):
    cuda()
    selected = read_json(ROOT/"evidence/selection.json")["selected"]["seed"]
    seed = SETTINGS["seeds"][0] if variant == "Q0" else selected
    adapter = artifact_path("A0" if variant == "Q0" else "A",seed)
    if not check_bound_artifact(read_json(ROOT/"evidence/base-identity.json"),artifact_path("B")):
        raise ValueError("Wrong/stale base before merge")
    expected_adapter = read_json(ROOT/f"evidence/training-{seed}.json")["initial" if variant == "Q0" else "final"]
    if not check_bound_artifact(expected_adapter,adapter):
        raise ValueError("Wrong/stale adapter before merge")
    output = artifact_path(variant)
    if output.exists():
        raise ValueError("Export destination exists after a partial stage; use a fresh run or inspect/archive that destination explicitly")
    args = ["merge","--adapter",str(adapter),"--base",str(artifact_path("B")),"--output",str(output),"--dtype","float16"]
    if variant in {"Q","Q0"}:
        args += ["--save-format","4bit"]
    with CAPTURE.phase("cpu_dense_merge" if variant == "M" else "dense_staging_nf4_save"):
        run_soup(args)
    if not list(output.glob("*.safetensors")) or (output/"adapter_config.json").exists():
        raise ValueError("Export is not a standalone saved model")
    run_soup(["adapters","check-safetensors",str(output),"--strict"])
    identity = model_manifest(output)
    provenance = {"variant":variant,"utc":now(),"base":read_json(ROOT/"evidence/base-identity.json"),
                  "adapter":model_manifest(adapter),"artifact":identity,"runtime_sha256":file_sha(__file__),"soup_commit":SOUP_REV}
    write_json(output/"provenance.json",provenance)
    write_json(ROOT/f"evidence/artifact-{variant}.json",provenance)
    return {"outputs":[f"evidence/artifact-{variant}.json",str((output/"provenance.json").relative_to(ROOT)),
                       *[str(p.relative_to(ROOT)) for p in output.glob("*.safetensors")]],"variant":variant,"artifact_sha256":identity["sha256"]}


def tensor_from(directory,key=None,suffix=None):
    from safetensors import safe_open
    for filename in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(filename,framework="pt",device="cpu") as handle:
            for name in handle.keys():
                if (key is not None and name == key) or (suffix is not None and name.endswith(suffix)):
                    return handle.get_tensor(name)
    raise KeyError(key or suffix)


def weight_comparison(candidate_dir,adapter_dir):
    config = read_json(adapter_dir/"adapter_config.json")
    scale = config["lora_alpha"] / config["r"]
    comparisons = []
    for layer in (0,27):
        for projection in ("q_proj","v_proj"):
            prefix = f"model.layers.{layer}.self_attn.{projection}"
            # The Hub checkpoint can be BF16; the measured merge starts from FP16.
            base = tensor_from(artifact_path("B"),key=prefix+".weight").half()
            a = tensor_from(adapter_dir,suffix=prefix+".lora_A.weight")
            b = tensor_from(adapter_dir,suffix=prefix+".lora_B.weight")
            expected = (base.float()+(b.float()@a.float())*scale).to(base.dtype).float()
            candidate = tensor_from(candidate_dir,key=prefix+".weight").float()
            effect = float((expected-base.float()).norm())
            residual = float((candidate-expected).norm())
            comparisons.append({"projection":prefix,"expected_effect_norm":effect,"residual_norm":residual,
                                "relative_residual":residual/effect if effect else None,
                                "max_abs_residual":float((candidate-expected).abs().max())})
    policy = read_json(ROOT/"run_policy.json")
    passed = all(x["relative_residual"] is not None
        and x["relative_residual"] <= policy["weight_relative_residual_max"]
        and x["max_abs_residual"] <= policy["weight_absolute_residual_max"] for x in comparisons)
    return {"passed":passed,"selected_projections":comparisons,"scope":"four projections; float32 BA then FP16 rounding; 25% relative and0.001 absolute ceilings"}


def load_probes(name):
    return TORCH.load(ROOT/f"evidence/probes/{name}.pt",map_location="cpu",weights_only=True)


def fidelity_stage():
    cuda()
    seed = read_json(ROOT/"evidence/selection.json")["selected"]["seed"]
    calibration = read_json(ROOT/"evidence/numerical-calibration.json")
    policy = read_json(ROOT/"run_policy.json")
    a = load_probes(f"validation-A-{seed}")
    b = load_probes("validation-B")
    m = load_probes("validation-M")
    q = load_probes("validation-Q")
    q0 = load_probes("validation-Q0")
    am = compare_probes(a,m)
    mq = compare_probes(m,q)
    bq0 = compare_probes(b,q0)
    q0q = compare_probes(q0,q)
    weights = weight_comparison(artifact_path("M"),artifact_path("A",seed))
    wrong_base_export = weight_comparison(artifact_path("B"),artifact_path("A",seed))
    q_identity = read_json(ROOT/"evidence/artifact-Q.json")["artifact"]
    wrong_q0_rejected = not check_bound_artifact(q_identity,artifact_path("Q0"))
    result = {"calibration":calibration,"A_to_M":am,"M_to_Q":mq,"B_to_Q0":bq0,"Q0_to_Q":q0q,"weights":weights,
        "merge_faithful":weights["passed"] and am["max_reference_logprob_error"] <= calibration["merge_reference_logprob_tolerance"]
                         and am["max_row_nll_difference"] <= calibration["merge_reference_logprob_tolerance"]
                         and am["max_kl_before_after"] <= policy["merge_max_kl"],
        "controls":{"base_substituted_for_merged_rejected":not wrong_base_export["passed"],"wrong_base_weight_details":wrong_base_export,
                    "q0_substituted_for_q_identity_rejected":wrong_q0_rejected,
                    "q_distinguishable_from_q0":q0q["max_reference_logprob_error"] > max(
                        policy["minimum_adapter_or_q_effect"],policy["effect_floor_multiplier"]*calibration["floor"]),
                    "q0_is_not_a_quantization_correction":"B/Q0 and M/Q are reported independently; errors are nonlinear"}}
    write_json(ROOT/"evidence/fidelity-and-export-controls.json",result)
    return {"outputs":["evidence/fidelity-and-export-controls.json"],"merge_faithful":result["merge_faithful"]}


def selftest_stage():
    from soup_cli.eval.custom import tool_call_match
    functions = {"f":{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"integer"},"enabled":{"type":"boolean"}},
                                                   "required":["x","enabled"],"additionalProperties":False}}}
    reference = [{"name":"f","arguments":{"x":1,"enabled":True}}]
    correct = tagged(reference)
    variants = {
        "correct":(correct,True), "numeric_json_semantics":(tagged([{"name":"f","arguments":{"x":1.0,"enabled":True}}]),True),
        "wrong_argument":(tagged([{"name":"f","arguments":{"x":2,"enabled":True}}]),False),
        "boolean_to_number":(tagged([{"name":"f","arguments":{"x":1,"enabled":1}}]),False),
        "number_to_string":(tagged([{"name":"f","arguments":{"x":"1","enabled":True}}]),False),
        "extra_call":(correct+"\n"+correct,False), "prose":("Here: "+correct,False),
        "unknown_tool":(tagged([{"name":"missing","arguments":{"x":1,"enabled":True}}]),False),
        "duplicate_key":('<tool_call>{"name":"f","name":"f","arguments":{}}</tool_call>',False),
        "nonfinite":('<tool_call>{"name":"f","arguments":{"x":NaN}}</tool_call>',False),
        "escaped_newline":('<tool_call>\\n{"name":"f","arguments":{}}\\n</tool_call>',False),
        "incomplete":("<tool_call>{}",False),
    }
    results = {}
    for name,(text,wanted) in variants.items():
        scored = score_turn(text,reference,functions,matcher=tool_call_match)
        assert scored["complete"] is wanted,(name,scored)
        assert scored["complete"] == score_turn(text,reference,functions)["complete"],"Soup matcher and outer policy disagree"
        results[name] = scored
    assert not score_turn(correct,reference,functions,truncated=True,matcher=tool_call_match)["complete"]
    multi = reference + [{"name":"f","arguments":{"x":2,"enabled":False}}]
    assert score_turn(tagged(multi),multi,functions,matcher=tool_call_match)["complete"]
    assert not score_turn(tagged(list(reversed(multi))),multi,functions,matcher=tool_call_match)["complete"]
    tok = tokenizer()
    fmt = formatter(tok)
    independent = {}
    for name in ("malformed_call","missing_tools","double_template","length_boundary"):
        messages = read_jsonl(ROOT/f"controls/{name}.jsonl")[0]["messages"]
        rejected = False
        try:
            functions_from_messages(messages)
            parse_calls(messages[-1]["content"])
            if any(token in m["content"] for token in tok.all_special_tokens for m in messages):
                raise ValueError("reserved token")
            if name == "length_boundary":
                if len(tok.apply_chat_template(messages,tokenize=True,return_dict=False)) > 16:
                    raise ValueError("would truncate at16")
            audit_mask(messages,tok,fmt)
        except ValueError:
            rejected = True
        assert rejected,name
        independent[name] = "observed_caught"
    write_json(ROOT/"evidence/scorer-and-data-controls.json",{"all_passed":True,"scorer_fixtures":results,"independent_data_controls":independent,
        "multiple_call_order_control":True,"stop_failure_control":True,"Soup_scorer_reused":True})
    return {"outputs":["evidence/scorer-and-data-controls.json"],"fixtures_passed":len(results)+2}


def freeze_stage():
    policy = read_json(ROOT/"run_policy.json")
    selection = read_json(ROOT/"evidence/selection.json")
    required = ["evidence/numerical-calibration.json","evidence/fidelity-and-export-controls.json","evidence/scorer-and-data-controls.json"]
    if any(not (ROOT/p).exists() for p in required):
        raise ValueError("Development checks missing before test freeze")
    frozen = {"frozen_utc":now(),"policy_sha256":file_sha(ROOT/"run_policy.json"),"policy":policy,"selection":selection,
              "calibration":read_json(ROOT/required[0]),"artifacts":{v:read_json(ROOT/f"evidence/artifact-{v}.json")["artifact"] for v in ("M","Q","Q0")},
              "test_hash":file_sha(ROOT/"data/test.jsonl"),"challenge_hash":file_sha(ROOT/"data/challenge.jsonl"),
              "references_hash":file_sha(ROOT/"evidence/references.jsonl"),"rule_changes_after_test_allowed":False}
    write_json(ROOT/"evidence/policy-frozen-before-test.json",frozen)
    return {"outputs":["evidence/policy-frozen-before-test.json"]}


def soup_coverage():
    names = {"malformed_call":"fixture-malformed-validate","missing_tools":"fixture-missing-doctor",
             "double_template":"fixture-double-validate","length_boundary":"fixture-length-doctor","no_op_adapter":"initial-adapter-check"}
    result = []
    for failure,label in names.items():
        path = ROOT/f"evidence/stages/{label}.json"
        observed = read_json(path) if path.exists() else {}
        code = observed.get("result",{}).get("exit_code")
        status = "NOT_RUN" if code is None else "observed_missed" if code == 0 else "observed_caught" if code == 2 else "UNRESOLVED_RUNTIME_OR_USAGE_FAILURE"
        result.append({"failure":failure,"Soup_status":status,
                       "Soup_exit_code":code,"stage_report":str(path.relative_to(ROOT)),
                       "scope_note":"Format validation does not validate embedded function-call semantics; narrow checks are not automatically Soup bugs."})
    return result


def finalize_stage():
    required = ["evidence/policy-frozen-before-test.json","evidence/fidelity-and-export-controls.json","evidence/scorer-and-data-controls.json",
                *[f"evidence/predictions/test-{v}.jsonl" for v in ("B","M","Q")],
                *[f"evidence/general/{v}.json" for v in ("B","M","Q")],
                *[f"evidence/predictions/challenge-{v}.jsonl" for v in ("B","M","Q")]]
    missing = [p for p in required if not (ROOT/p).exists()]
    coverage = soup_coverage()
    write_json(ROOT/"evidence/silent-failure-coverage.json",coverage)
    if missing:
        verdict = {"verdict":"DON'T SHIP","gates":{"required_evidence_complete":False},"blockers":["Missing: "+p for p in missing],"metrics":None}
        write_json(ROOT/"verdict.json",verdict)
        return {"outputs":["verdict.json","evidence/silent-failure-coverage.json"],"verdict":"DON'T SHIP","missing":missing}
    frozen = read_json(ROOT/required[0])
    fidelity = read_json(ROOT/required[1])
    selected = frozen["selection"]["selected"]["seed"]
    training = read_json(ROOT/f"evidence/training-{selected}.json")
    predictions = {v:read_jsonl(ROOT/f"evidence/predictions/test-{v}.jsonl") for v in ("B","M","Q")}
    refs = reference_rows("test")
    expected_ids = [row["id"] for row in refs]
    if len(expected_ids) != len(rows("test")) or any([row["id"] for row in result] != expected_ids for result in predictions.values()):
        raise ValueError("Predictions must cover every frozen test turn in order")
    pairs = {"Q_minus_B":grouped_pairs(predictions["B"],predictions["Q"]),"Q_minus_M":grouped_pairs(predictions["M"],predictions["Q"]),
             "M_minus_B":grouped_pairs(predictions["B"],predictions["M"])}
    general = {v:read_json(ROOT/f"evidence/general/{v}.json") for v in ("B","M","Q")}
    challenges = {v:read_jsonl(ROOT/f"evidence/predictions/challenge-{v}.jsonl") for v in ("B","M","Q")}
    challenge_ids = [digest(row["messages"]) for row in rows("challenge")]
    if any([x["id"] for x in results] != challenge_ids for results in challenges.values()):
        raise ValueError("Challenge IDs differ")
    new_critical = [q["id"] for b,q in zip(challenges["B"],challenges["Q"])
                    if b["complete"] and (not q["tool_selection"] or not q["schema_ok"] or not q["arguments_ok"])]
    stage_paths = sorted((ROOT/"evidence/stages").glob("*.json"))
    stage_reports = [read_json(p) for p in stage_paths if not re.search(r"\.previous-",p.name)]
    resource = [r.get("resources",{}) for r in stage_reports]
    gpu_reports = [r for r in stage_reports if r.get("hardware")]
    available = min((r["hardware"]["total_bytes"] for r in stage_reports if r.get("hardware")),default=0)
    allocated = max((r.get("peak_allocated_bytes",0) for r in resource),default=0)
    reserved = max((r.get("peak_reserved_bytes",0) for r in resource),default=0)
    device_sampled = max((r.get("peak_device_used_mib_sampled",0)*2**20 for r in resource),default=0)
    memory = {"usable_device_bytes":available,"max_allocated_bytes":allocated,"max_reserved_bytes":reserved,
              "max_device_used_bytes_sampled":device_sampled,"headroom_bytes":available-max(reserved,device_sampled),
              "per_gpu_stage":{r["stage"]:{k:r["resources"][k] for k in
                  ("elapsed_seconds","peak_allocated_bytes","peak_reserved_bytes","peak_device_used_mib_sampled",
                   "device_samples","peak_rss_bytes_sampled")} for r in gpu_reports},
              "sampling_limit":"Device samples can miss short spikes; allocator reservation and sampled driver memory are distinct instruments."}
    estimate = read_json(ROOT/"evidence/memory-estimate-before.json")
    memory["preflight_budget_bytes"] = estimate["training_memory_preflight_budget_gib"]*2**30
    memory["peak_allocated_minus_parameter_state_floor_bytes"] = allocated-(estimate["base_fp16_gib"]+estimate["lora_fp32_parameter_gradient_adam_gib"])*2**30
    artifacts_ok = all(check_bound_artifact(frozen["artifacts"][v],artifact_path(v)) for v in ("M","Q","Q0"))
    freeze_ok = file_sha(ROOT/"run_policy.json") == frozen["policy_sha256"] and file_sha(ROOT/"data/test.jsonl") == frozen["test_hash"]
    freeze_ok = freeze_ok and all(file_sha(ROOT/name) == sha for name,sha in frozen["policy"]["config_hashes"].items())
    freeze_ok = freeze_ok and file_sha(ROOT/"data/challenge.jsonl") == frozen["challenge_hash"]
    freeze_ok = freeze_ok and file_sha(ROOT/"evidence/references.jsonl") == frozen["references_hash"]
    required_stage_labels = [f"train-{s}" for s in SETTINGS["seeds"]]
    required_stage_labels += [f"evaluate-validation-A-{s}" for s in SETTINGS["seeds"]]
    required_stage_labels += [f"evaluate-{split}-{variant}" for split in ("test","challenge") for variant in ("B","M","Q")]
    required_stage_labels += ["prepare","selftest","freeze","select","fidelity","merge-M","merge-Q","merge-Q0",
                              "preflight-dry-run","bench-train","selected-adapter-check","selected-safe-check","selected-adapter-diff"]
    stage_ok = all((ROOT/f"evidence/stages/{label}.json").exists() and read_json(ROOT/f"evidence/stages/{label}.json")["status"] == "passed" for label in required_stage_labels)
    if stage_ok:
        stage_ok = benchmark_headroom(read_json(ROOT/"evidence/stages/bench-train.json"), SETTINGS["allocator_config"])["passed"]
    resource_ok = bool(gpu_reports) and all(r["resources"]["device_samples"] > 0 and
        not any("cuda_measurement_error" in segment for segment in r["resources"]["segments"]) for r in gpu_reports)
    identities = {v:read_json(ROOT/f"evidence/identity-{v}.json") for v in ("M","Q")}
    no_op_caught = next((x["Soup_status"] == "observed_caught" for x in coverage if x["failure"] == "no_op_adapter"),False)
    integrity = {"required_stage_reports_pass":stage_ok,"policy_and_test_unchanged":freeze_ok,"artifact_hashes_bound":artifacts_ok,
        "updated_and_active_adapter":training["observations"]["first_materialized_update"] is not None
            and training["adapter_effect"]["max_reference_logprob_error"] > max(1e-5,10*frozen["calibration"]["floor"]),
        "selected_frozen_weights_unchanged":training["frozen_projection_hashes_before"] == training["frozen_projection_hashes_after"],
        "disabled_base_agreement":training["disabled_vs_pristine"]["max_reference_logprob_error"] <= max(.002,10*frozen["calibration"]["floor"]),
        "merge_fidelity":fidelity["merge_faithful"],"scorer_data_controls":read_json(ROOT/required[2])["all_passed"],
        "base_for_merge_control":fidelity["controls"]["base_substituted_for_merged_rejected"],
        "q0_for_q_control":fidelity["controls"]["q0_substituted_for_q_identity_rejected"] and fidelity["controls"]["q_distinguishable_from_q0"],
        "Soup_noop_adapter_control":no_op_caught,"portable_M_Q":all(identities[v]["portable_base_adapter_reads_blocked"] for v in identities),
        "real_saved_nf4":identities["Q"]["quantized_linear_count"] == 196,
        "no_new_critical_challenge_failures":not new_critical,
        "one_gib_tested_headroom":resource_ok and available > 0 and memory["headroom_bytes"] >= 2**30,
        "three_seed_development_complete":len(SETTINGS["seeds"]) >= 3,
        "Soup_environment_bounds_ok":(ROOT/"evidence/stages/env-check.json").exists()
            and read_json(ROOT/"evidence/stages/env-check.json").get("result",{}).get("exit_code") == 0,
        "Soup_fixture_coverage_resolved":all(x["Soup_status"] in {"observed_caught","observed_missed"} for x in coverage)}
    verdict = decision_gates(pairs["Q_minus_B"],pairs["Q_minus_M"],general["B"]["scores"],general["Q"]["scores"],integrity)
    verdict.update(utc=now(),selected_seed=selected,metrics={v:summarize_predictions(x) for v,x in predictions.items()},
        paired_bootstrap=pairs,general_suites={v:g["scores"] for v,g in general.items()},memory=memory,
        new_critical_challenges=new_critical,selection=frozen["selection"],policy_sha256=frozen["policy_sha256"],
        scope_limits=["Four validation distribution probes and four weight projections do not prove full-distribution equivalence.",
                      "Prepared holdouts are source-shifted toward primary Hermes; source labels are absent from the four-file export.",
                      "References are structurally audited; semantic correctness and alternative valid calls are not automatically adjudicated.",
                      "Bundled general suites and six challenge turns have finite coverage; this is not an arbitrary-tool safety claim."],
        remediation="Inspect exact failed gates: export/identity failures need re-export or verifier fixes; task/regression failures need development work before a fresh test policy.")
    write_json(ROOT/"verdict.json",verdict)
    write_json(ROOT/"evidence/memory-reconciliation.json",memory)
    for variant in ("M","Q"):
        payload = {"task":{"mode":"metric","base":verdict["metrics"]["B"]["complete"],"tuned":verdict["metrics"][variant]["complete"]},
                   "benchmarks":{name:{"base":general["B"]["scores"][name],"tuned":general[variant]["scores"][name]} for name in general["B"]["scores"]},
                   "provenance":{"source":"offline replay of measured predictions","runtime_sha256":file_sha(__file__),
                                 "predictions_sha256":file_sha(ROOT/f"evidence/predictions/test-{variant}.jsonl"),"artifact_sha256":frozen["artifacts"][variant]["sha256"]}}
        destination = ROOT/f"evidence/soup-ship-input-{variant}.json"
        write_json(destination,payload)
        run_soup(["ship","--evidence",str(destination),"--forgetting-threshold","0.05","--output",str(ROOT/f"evidence/soup-ship-{variant}.json")],allowed=(0,2))
    # Reuse quant-check on cached, already generated single-call wrappers; never pretend this loaded models.
    from soup_cli.eval.custom import EvalTask
    from soup_cli.eval.quant_check import run_quant_check
    tasks = [EvalTask(prompt=x["id"],expected=canonical({"function":x["expected"][0]}),scoring="tool_call_match") for x in refs if x["call_count"] == 1]
    indexes = {v:{x["id"]:x for x in predictions[v]} for v in ("M","Q")}
    def cached(variant,prompt):
        row = indexes[variant][prompt]
        try:
            calls = parse_calls(row["text"])
            return canonical({"function":calls[0]}) if len(calls)==1 and not row["truncated"] else "INVALID"
        except ValueError:
            return "INVALID"
    quant_result = run_quant_check(before_gen=lambda p:cached("M",p),after_gen=lambda p:cached("Q",p),tasks=tasks,task_name="single-call subset, cached measured generations")
    write_json(ROOT/"evidence/soup-quant-check.json",{"source":"cached-generation decision replay; no additional model load","single_call_turns":len(tasks),"result":json.loads(quant_result.to_json())})
    return {"outputs":["verdict.json","evidence/memory-reconciliation.json","evidence/silent-failure-coverage.json"],"verdict":verdict["verdict"],"blockers":verdict["blockers"]}


def report_pages(verdict):
    metrics = verdict.get("metrics") or {}
    page1 = ["SOUP / HERMES / QWEN2.5-1.5B", "Measured experiment report - " + verdict["verdict"], "",
             f"Base: {MODEL_ID}@{MODEL_REV[:12]}",f"Soup: {SOUP_REV[:12]}; prepared Kaggle dataset: pernavjain/tool-calling",
             "Data: 1600 training /150 validation /150 test; exact file hashes in run_policy.json.",
             "Training: frozen FP16 base, rank8 all seven projections, batch1, accumulation8, two epochs.",
             "Recipe: LR1e-4, cosine decay, 5% warmup; export the lowest validation-loss epoch.",
             "Seed selection uses validation complete-turn correctness, then NLL, then seed.", "",
             "TASK RESULTS - complete ordered turns; every held-out turn remains in denominator"]
    for variant in ("B","M","Q"):
        value = metrics.get(variant)
        page1.append(f"{variant}: " + (f"{value['complete']:.3%} complete; {value['parse_ok']:.3%} parse; assistant NLL={value['mean_assistant_nll']:.5f}; n={value['n']}" if value else "NOT MEASURED"))
    page1 += ["", "PAIRED GROUP BOOTSTRAP - 10000 seeded resamples"]
    for name,pair in verdict.get("paired_bootstrap",{}).items():
        page1.append(f"{name}: {pair['delta']:+.3%};95% CI [{pair['ci95'][0]:+.3%},{pair['ci95'][1]:+.3%}];groups={pair['groups']}")
    memory = verdict.get("memory",{})
    page1 += ["", "RESOURCE RECONCILIATION"]
    for name in ("usable_device_bytes","max_allocated_bytes","max_reserved_bytes","max_device_used_bytes_sampled","headroom_bytes"):
        page1.append(f"{name}: " + (f"{memory[name]/2**30:.3f} GiB" if name in memory else "NOT MEASURED"))
    page1 += ["Allocator counters belong to the tensor-owning child. Device/RSS samples can miss spikes.",
              "The original estimate precedes optimization. Activations, workspaces and context add to the floor.", ""]
    if metrics:
        page1 += ["ARTIFACTS: adapter, merged-fp16, merged-nf4 retained under artifacts/.",
                  "Cold M/Q loads use local files, block base/adapter reads and reject CPU/meta fallback."]
    else:
        page1.append("ARTIFACTS: verification incomplete. See stage reports for observed outputs.")
    page2 = ["GATES, CONTROLS AND LIMITATIONS", "", "RELEASE GATES"]
    page2 += [f"{'PASS' if passed is True else 'BLOCK'}: {name}" for name,passed in verdict.get("gates",{}).items()]
    page2 += ["", "GENERAL REGRESSION - absolute suite scores B /M /Q"]
    general = verdict.get("general_suites",{})
    for name in general.get("B",{}):
        page2.append(name + ": " + " / ".join(f"{general[v][name]:.3f}" for v in ("B","M","Q")))
    page2 += ["", "LIMITS AND REFLECTION"]
    page2 += verdict.get("scope_limits",["Required experiment results missing; inspect stage errors and verdict blockers."])
    reflection = SETTINGS.get("reflection","").strip()
    page2.append("Author reflection: " + (reflection or "PENDING - add your own observed surprise, concern and next step before submission."))
    page2.append("AI disclosure: AI assisted notebook/code design; this report uses saved run outputs. Author should describe their checks and edits.")
    page2.append("This is an automated report draft. Inspect the final two-page PDF and complete reflection before submission.")
    return page1,page2


def package_stage():
    import textwrap
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from pypdf import PdfReader
    verdict = read_json(ROOT/"verdict.json") if (ROOT/"verdict.json").exists() else {"verdict":"DON'T SHIP","metrics":None,"gates":{"required_verdict_present":False}}
    pages = report_pages(verdict)
    markdown = "\n\n---\n\n".join("\n".join(page) for page in pages)
    (ROOT/"REPORT.md").write_text(markdown+"\n")
    pdf_path = ROOT/"report.pdf"
    pdf = canvas.Canvas(str(pdf_path),pagesize=A4)
    width,height = A4
    for number,paragraphs in enumerate(pages,1):
        pdf.setFont("Helvetica-Bold",13)
        pdf.drawString(42,height-40,"Soup AI Engineer experiment - " + ("results" if number == 1 else "verification"))
        lines = []
        for paragraph in paragraphs:
            ascii_text = paragraph.encode("ascii","replace").decode()
            lines.extend(textwrap.wrap(ascii_text,width=96) or [""])
        if len(lines) > 61:
            raise ValueError("Report would overflow two pages; shorten the author reflection/report text")
        text = pdf.beginText(42,height-65)
        text.setFont("Courier",8)
        text.setLeading(11)
        for line in lines:
            text.textLine(line)
        pdf.drawText(text)
        pdf.setFont("Helvetica",8)
        pdf.drawString(42,30,f"{number}/2 - measured evidence: evidence/; verdict: verdict.json")
        pdf.showPage()
    pdf.save()
    assert len(PdfReader(pdf_path).pages) == 2
    (ROOT/"REFLECTION_AND_AI_USE.md").write_text(
        "# Author reflection and AI-use disclosure\n\n" + (SETTINGS.get("reflection","").strip() or
        "Add your first-person reflection based on this run: what surprised you, what concerns you, and what you would test next.") +
        "\n\nAI assisted the notebook and verifier implementation. Add which suggestions you accepted, changed or rejected and how you independently checked them.\n")
    outputs = ["REPORT.md","report.pdf","REFLECTION_AND_AI_USE.md"]
    return {"outputs":outputs,"verdict":verdict["verdict"],"pdf_pages":2,
            "report_status":"automated draft; human reflection/visual inspection pending"}


def main():
    global ROOT,SETTINGS,CAPTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,help="Kaggle run directory with settings.json, data/ and artifacts/")
    sub = parser.add_subparsers(dest="stage",required=True)
    for name in ("prepare","select","fidelity","selftest","freeze","finalize","package"):
        sub.add_parser(name)
    p = sub.add_parser("train");p.add_argument("--seed",type=int,required=True)
    p = sub.add_parser("merge");p.add_argument("--variant",choices=("M","Q","Q0"),required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--variant",choices=("B","A","A0","M","Q","Q0"),required=True)
    p.add_argument("--split",choices=("validation","test","challenge"),required=True)
    p.add_argument("--seed",type=int)
    p.add_argument("--general",action="store_true")
    p.add_argument("--probes-only",action="store_true")
    p = sub.add_parser("cli")
    p.add_argument("--label",required=True)
    p.add_argument("--allowed",default="0")
    p.add_argument("--cuda-owner",action="store_true")
    p.add_argument("arguments",nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ROOT = Path(args.root).resolve()
    SETTINGS = read_json(ROOT/"settings.json")
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = SETTINGS["gpu_uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    allocator_config = SETTINGS.setdefault("allocator_config", T4_ALLOCATOR_CONFIG)
    # Set the alias to the same value for older PyTorch versions.
    os.environ["PYTORCH_ALLOC_CONF"] = allocator_config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_config
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    if args.stage != "prepare":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.stage in {"train","evaluate","merge","freeze","selftest"}:
        for name,sha in EXPECTED_DATA.items():
            local = ROOT/("data/"+name if name.endswith(".jsonl") else name)
            if file_sha(local) != sha:
                raise ValueError("Prepared input bytes changed: " + name)
        policy = read_json(ROOT/"run_policy.json")
        if policy.get("allocator_config") != SETTINGS["allocator_config"]:
            raise ValueError("Frozen allocator policy changed")
        if policy.get("trainer_policy") != TRAINER_POLICY:
            raise ValueError("Frozen trainer policy changed; use a new run for this training recipe")
        if any(file_sha(ROOT/name) != sha for name,sha in policy["config_hashes"].items()):
            raise ValueError("Frozen training configuration changed")
    if args.stage in {"train","evaluate"} and getattr(args,"variant",None) in {"A","A0"} and args.seed is None:
        parser.error("Adapter evaluation needs --seed")
    if args.stage == "evaluate" and args.probes_only and args.split != "validation":
        parser.error("Probe-only runs use the fixed validation probes")
    label = args.stage
    if args.stage == "cli":label = args.label
    elif args.stage == "train":label = f"train-{args.seed}"
    elif args.stage == "merge":label = f"merge-{args.variant}"
    elif args.stage == "evaluate":label = f"evaluate-{args.split}-{args.variant}" + (f"-{args.seed}" if args.seed is not None else "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",label):
        parser.error("Invalid stage label")
    stage_path = ROOT/f"evidence/stages/{label}.json"
    stage_path.parent.mkdir(parents=True,exist_ok=True)
    request = {key:value for key,value in vars(args).items() if key != "root"}
    runtime_sha = file_sha(__file__)
    if stage_path.exists():
        old = read_json(stage_path)
        if old.get("status") == "passed" and args.stage not in {"finalize","package"}:
            if old.get("runtime_sha256") != runtime_sha or old.get("settings_sha256") != digest(SETTINGS) or old.get("request") != request:
                raise ValueError("Cached stage settings/code/request drifted; create a new run")
            for name,sha in old.get("output_bindings",{}).items():
                if not (ROOT/name).is_file() or file_sha(ROOT/name) != sha:
                    raise ValueError("Cached stage output changed: " + name)
            print("Verified completed stage; reusing: " + label,flush=True)
            return
        stage_path.rename(stage_path.with_name(label+".previous-"+str(time.time_ns())+".json"))
    report = {"stage":label,"request":request,"started_utc":now(),"status":"running","runtime_sha256":runtime_sha}
    write_json(stage_path,report)
    faulthandler.enable()
    CAPTURE = ResourceCapture(label)
    CAPTURE.start()
    try:
        if args.stage == "prepare":result = prepare_stage()
        elif args.stage == "train":result = train_stage(args.seed)
        elif args.stage == "evaluate":result = evaluate_stage(args.variant,args.split,args.seed,args.general,args.probes_only)
        elif args.stage == "merge":result = merge_stage(args.variant)
        elif args.stage == "select":result = select_stage()
        elif args.stage == "fidelity":result = fidelity_stage()
        elif args.stage == "selftest":result = selftest_stage()
        elif args.stage == "freeze":result = freeze_stage()
        elif args.stage == "finalize":result = finalize_stage()
        elif args.stage == "package":result = package_stage()
        else:
            if args.cuda_owner:cuda()
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            code = run_soup(arguments,allowed=tuple(int(x) for x in args.allowed.split(",")))
            outputs = []
            for flag in ("--output","--state"):
                if flag in arguments:
                    target = (ROOT/arguments[arguments.index(flag)+1]).resolve()
                    if target.is_file():
                        outputs.append(str(target.relative_to(ROOT)))
            result = {"exit_code":code,"arguments":arguments,"outputs":outputs}
        report.update(status="passed",result=result)
        report["output_bindings"] = {name:file_sha(ROOT/name) for name in result.get("outputs",[]) if (ROOT/name).is_file()}
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        report["resources"] = CAPTURE.finish()
        if TORCH is not None and TORCH.cuda.is_initialized():
            props = TORCH.cuda.get_device_properties(0)
            report["hardware"] = {"name":props.name,"total_bytes":props.total_memory,"visible_count":TORCH.cuda.device_count(),
                                  "selected_physical_uuid":SETTINGS["gpu_uuid"],"capability":list(TORCH.cuda.get_device_capability(0)),
                                  "torch_version":TORCH.__version__,"torch_cuda_build":TORCH.version.cuda,
                                  "allocator_backend":TORCH.cuda.get_allocator_backend(),
                                  "allocator_config":os.environ["PYTORCH_ALLOC_CONF"]}
        if label == "bench-train":
            report["memory_gate"] = benchmark_headroom(report, SETTINGS["allocator_config"])
        report.update(finished_utc=now(),settings_sha256=digest(SETTINGS))
        write_json(stage_path,clean_observation(report))
        if args.stage == "evaluate" and args.variant in {"M","Q"}:
            shutil.rmtree(ROOT/f"artifacts/portable/{args.variant}",ignore_errors=True)
        event("stage_end",stage=label,status=report["status"])


if __name__ == "__main__":
    main()
