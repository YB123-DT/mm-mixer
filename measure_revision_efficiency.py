#!/usr/bin/env python3
"""Measure one saved downstream model per process on real test utterances.

Example (run only on an idle GPU):
  python measure_revision_efficiency.py --artifact-dir runs/meld/full/seed2025/best_peak \
    --dataset meld --variant full --seed 2025 --server biggpu --gpu GPU-UUID \
    --output efficiency/meld_full_2025.json

The formal protocol is 1000 distinct test targets, batch size 32 (last batch 8),
float32 eval/no_grad forward, with inputs already resident on the GPU. Dataset
loading, feature extraction, host-to-device transfer, parameter auditing and
metric computation are outside the timed region. The default model output,
including any auxiliary heads it computes, is preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EFFICIENCY_VARIANTS = ("full", "amm_mlp", "amm_attention", "amm_cubemlp")
BUNDLE_ARTIFACTS = (
    "best_peak_test_state_dict.pt", "peak_test_predictions.pt",
    "peak_test_metrics.json", "classification_report.txt", "history.json", "config.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summarize_times(seconds, samples):
    if not seconds or samples <= 0 or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("timings must be finite positive durations for a positive sample count")
    ordered = sorted(seconds)
    position = .9 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    p90 = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    median = statistics.median(ordered)
    return {"seconds_each_repeat": list(seconds), "median_seconds": median,
            "min_seconds": min(ordered), "p90_seconds": p90,
            "all_repeats_total_seconds": sum(seconds),
            "median_utterances_per_second": samples / median,
            "median_ms_per_utterance_amortized": median * 1000 / samples}


def sample_indices(available, smoke_samples=None):
    count = 1000 if smoke_samples is None else smoke_samples
    if count < 1 or (smoke_samples is not None and count > 1000):
        raise ValueError("smoke samples must be between 1 and 1000")
    if available < count:
        raise ValueError(f"need {count} distinct test utterances; dataset has {available}")
    return list(range(count))


def remap_path(path, mappings):
    value = str(path)
    for old, new in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
        if value == old or value.startswith(old.rstrip("/") + "/"):
            return Path(new + value[len(old):])
    return Path(value)


def select_gpu(requested, server):
    """Resolve a physical device before importing torch or initializing CUDA."""
    if server == "biggpu" and requested == "4":
        raise ValueError("biggpu physical GPU 4 is prohibited")
    command = ["nvidia-smi", "-i", requested,
               "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    rows = list(csv.reader(io.StringIO(subprocess.check_output(command, text=True))))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("--gpu must resolve to exactly one physical GPU")
    index, uuid, name, memory, driver = [part.strip() for part in rows[0]]
    if server == "biggpu" and int(index) == 4:
        raise ValueError(f"prohibited biggpu GPU 4 resolved from {requested}: {uuid}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return {"physical_index": int(index), "uuid": uuid, "name": name,
            "memory_total_mib": float(memory), "driver_version": driver,
            "logical_device": "cuda:0"}


def assert_idle_gpu(uuid):
    raw = subprocess.check_output([
        "nvidia-smi", "-i", uuid, "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True)
    others = [row for row in csv.reader(io.StringIO(raw))
              if row and row[0].strip() != str(os.getpid())]
    if others:
        raise RuntimeError(f"GPU has other compute processes; speed comparison requires an idle GPU: {others}")
    return {"checked_at_unix": time.time(), "other_compute_processes": []}


def validate_config_identity(config, dataset, variant, seed):
    """Check variant identity beyond a potentially mislabeled manifest."""
    if variant not in EFFICIENCY_VARIANTS:
        raise ValueError(f"efficiency comparison supports {EFFICIENCY_VARIANTS}, got {variant!r}")
    control = config.get("runtime_audit", {}).get("revision_control", {})
    if variant == "full":
        if control:
            raise ValueError("Full config unexpectedly contains a revision replacement")
    elif control.get("variant") != variant:
        raise ValueError("config revision_control.variant differs from requested replacement")
    if dataset == "iemocap":
        for key, expected in (("dataset", dataset), ("variant", variant), ("seed", seed)):
            if config.get(key) != expected:
                raise ValueError(f"config {key} differs from requested {expected!r}")
    else:
        # Historical MELD configs encode identity in nested trainer settings.
        for key, expected in (("dataset", dataset), ("variant", variant), ("seed", seed)):
            if key in config and config[key] != expected:
                raise ValueError(f"config {key} differs from requested {expected!r}")
        if "meld" not in config.get("classes", {}) or "meld" not in config.get("feature_paths", {}):
            raise ValueError("config is not a MELD trainer configuration")
        fixed = config.get("fixed_params", {})
        if fixed.get("seed") != seed or fixed.get("capacity_variant") != "M4_PAIR":
            raise ValueError("MELD config seed/capacity differs from the requested comparison")
        if fixed.get("aux_loss_weights") != {"t": 1.0, "a": 1.0, "v": 1.0}:
            raise ValueError("efficiency comparison requires the default three auxiliary objectives")
        enabled = config.get("runtime_audit", {}).get("active_modalities", ["t", "a", "v"])
        if set(enabled) != {"t", "a", "v"}:
            raise ValueError("efficiency comparison requires all three modalities")


def validate_artifact(directory, dataset, variant, seed):
    config_path, manifest_path = directory / "config.json", directory / "manifest.json"
    status_path = directory / "status.json"
    checkpoint = directory / "best_peak_test_state_dict.pt"
    paths = [directory / name for name in (*BUNDLE_ARTIFACTS, "manifest.json", "status.json")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    config, manifest = json.loads(config_path.read_text()), json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    if status.get("state") != "complete":
        raise ValueError("checkpoint bundle status is not complete")
    if status.get("fresh_strict_replay_exact") is not True or manifest.get("fresh_strict_replay_exact") is not True:
        raise ValueError("status and manifest must both confirm historical strict replay")
    for key, expected in (("dataset", dataset), ("variant", variant), ("seed", seed)):
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key}={manifest.get(key)!r}, requested {expected!r}")
    validate_config_identity(config, dataset, variant, seed)
    hashes = {p.name: sha256(p) for p in paths}
    recorded_hashes = manifest.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict) or not set(BUNDLE_ARTIFACTS) <= recorded_hashes.keys():
        raise ValueError("manifest must contain hashes for all six completed bundle artifacts")
    for name in BUNDLE_ARTIFACTS:
        if recorded_hashes[name] != hashes[name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    recorded_config = manifest.get("config_sha256")
    if recorded_config != json_hash(config):
        raise ValueError("manifest config payload hash missing or mismatched")
    for key, name in (("checkpoint_sha256", checkpoint.name), ("predictions_sha256", "peak_test_predictions.pt")):
        if status.get(key) != hashes[name]:
            raise ValueError(f"status {key} missing or mismatched")
    return config, manifest, checkpoint, hashes


def build_test_dataset(dataset, runner, config, mappings, meld_csv_dir=None):
    """Use the exact dataset/collate classes used by the formal runners."""
    if dataset == "iemocap":
        from utterance_history.runner import _dataset_class
        paths = config["feature_paths"]
        sources = {key: (str(paths[key]), remap_path(paths[key], mappings))
                   for key in ("metadata", "packed")}
        value = _dataset_class()(sources["metadata"][1], sources["packed"][1], "test", False)
        ids = []
        for dialogue, turn in value.index:
            raw = value.ids[dialogue][turn]
            ids.append({"dialogue": str(dialogue), "turn": int(turn), "utterance_id": str(raw)})
        return value, value.collate_fn, ids, sources

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    paths = config["feature_paths"]["meld"]["test"]
    names = {"v": "visual", "a": "audio", "t": "text"}
    sources = {mod: (str(paths[name]), remap_path(paths[name], mappings)) for mod, name in names.items()}
    csv_original = str(Path(meld_csv_dir or os.environ.get("MELD_CSV_DIR", runner.CSV_DIR)) / "test_sent_emo.csv")
    sources["metadata"] = (csv_original, remap_path(csv_original, mappings))
    metadata = pd.read_csv(sources["metadata"][1])
    features = {mod: json.loads(sources[mod][1].read_text()) for mod in names}
    labels = LabelEncoder()
    labels.classes_ = np.asarray(config["classes"]["meld"])
    value = runner.model_module.MELDDataset(
        metadata, features["v"], features["a"], features["t"], labels,
        ["v", "a", "t"], config["embed_dims_full"], is_training=False)
    ids = [{"dialogue": int(row.Dialogue_ID), "turn": int(row.Utterance_ID),
            "utterance_id": f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"}
           for row in metadata.itertuples()]
    return value, runner.model_module.custom_collate, ids, sources


def input_hash(batches):
    """Include ordered tensor names, shapes, dtypes, bytes and true labels."""
    digest = hashlib.sha256()
    for features, labels in batches:
        for name, tensor in [*sorted(features.items()), ("__labels__", labels)]:
            value = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main_logits(output):
    # Preserve the default forward tuple, extracting only for correctness checks.
    return output[0] if isinstance(output, tuple) else output


def parameter_audit(model, features, torch):
    """Gradient connectivity is a structural audit, not an optimization step."""
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    with torch.enable_grad():
        output = model(features)
        logits = main_logits(output)
        gradients = torch.autograd.grad(logits.sum(), [p for _, p in named], allow_unused=True) if named else []
    disconnected = [name for (name, _), grad in zip(named, gradients) if grad is None]
    connected = sum(p.numel() for (_, p), grad in zip(named, gradients) if grad is not None)
    return {"registered_total": sum(p.numel() for p in model.parameters()),
            "registered_trainable": sum(p.numel() for _, p in named),
            "main_logit_gradient_connected_trainable": connected,
            "main_logit_disconnected_trainable_names": disconnected,
            "connected_count_definition": "Requires-grad parameters connected to main logits on one eval batch; zero gradients still count. This excludes aux-only dependencies, not necessarily all forward compute. No optimizer/update is used; default forward is not pruned."}


def source_provenance():
    hashes = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT) and path.is_file():
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True)
        return result.stdout if result.returncode == 0 else None
    return {"git_commit": (git("rev-parse", "HEAD") or "").strip(),
            "git_status": git("status", "--short"), "tracked_diff": git("diff", "HEAD", "--"),
            "loaded_source_sha256": hashes}


def compare_recorded_sources(recorded, current):
    """Map relocated source paths by unambiguous repository-relative suffix."""
    entries = []
    for old_path, expected in recorded.items():
        matches = [relative for relative in current if old_path.endswith("/" + relative) or old_path == relative]
        if len(matches) == 1:
            relative = matches[0]
            entries.append({"recorded_path": old_path, "current_relative_path": relative,
                            "recorded_sha256": expected, "current_sha256": current[relative],
                            "status": "match" if expected == current[relative] else "changed"})
        else:
            entries.append({"recorded_path": old_path, "recorded_sha256": expected,
                            "status": "not_loaded_or_ambiguous"})
    return {"entries": entries,
            "all_recorded_sources_match": bool(entries) and all(row["status"] == "match" for row in entries),
            "interpretation": "Current-code timing of a strict-loaded checkpoint. Source differences are recorded; completed timing does not establish historical prediction/accuracy reproduction."}


def benchmark(args):
    if args.repeats < 3 or args.warmup_batches < 1:
        raise ValueError("need at least 3 timed repeats and 1 warmup batch")
    if args.cpu_threads < 1:
        raise ValueError("CPU thread count must be positive")
    sample_indices(1000, args.smoke_samples)
    artifact = Path(args.artifact_dir).resolve()
    config, manifest, checkpoint, hashes = validate_artifact(artifact, args.dataset, args.variant, args.seed)
    gpu = select_gpu(args.gpu, args.server)
    idle_before = assert_idle_gpu(gpu["uuid"])
    import torch
    from torch.utils.data import DataLoader, Subset
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one explicitly selected CUDA device is required")
    properties = torch.cuda.get_device_properties(0)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is not None and str(device_uuid).lower() != gpu["uuid"].lower():
        raise RuntimeError(f"CUDA device UUID mismatch: {device_uuid} versus {gpu['uuid']}")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Import exactly one runner: their vendored modules have colliding names.
    runner = importlib.import_module(f"dataset_runners.{args.dataset}")
    if args.dataset == "meld":
        dropout = float(config["fixed_params"].get("fusion_dropout", .2))
    else:
        # Dropout is disabled in eval; the frozen builder fixes structural sizes.
        dropout = .2
    model = runner.build_variant_model(args.variant, dropout)
    if args.dataset == "meld":
        model.disable_alignment()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    from mm_mixer_final.revision_controls import revision_control_metadata
    control_metadata = revision_control_metadata(model)
    mappings = []
    for entry in args.path_map:
        old, separator, new = entry.partition("=")
        if not separator or not old or not new:
            raise ValueError("--path-map needs OLD_PREFIX=NEW_PREFIX")
        mappings.append((old.rstrip("/"), new.rstrip("/")))
    dataset, collate, ids, sources = build_test_dataset(args.dataset, runner, config, mappings, args.meld_csv_dir)
    selected = sample_indices(len(dataset), args.smoke_samples)
    selected_ids = [ids[index] for index in selected]
    if len({json.dumps(x, sort_keys=True) for x in selected_ids}) != len(selected):
        raise ValueError("test target IDs are not unique")
    features_provenance = {}
    for key, (original, actual) in sources.items():
        actual_hash = sha256(actual)
        expected = manifest.get("feature_hashes", {}).get(original)
        if expected is not None and expected != actual_hash:
            raise ValueError(f"feature hash differs from checkpoint manifest: {original}")
        features_provenance[key] = {"recorded_path": original, "actual_path": str(actual),
                                   "sha256": actual_hash, "manifest_hash_verified": expected is not None}
    cpu_batches = list(DataLoader(Subset(dataset, selected), batch_size=32, shuffle=False,
                                  num_workers=0, drop_last=False, collate_fn=collate))
    if sum(len(labels) for _, labels in cpu_batches) != len(selected):
        raise RuntimeError("loader did not produce the exact requested target count")
    input_sha = input_hash(cpu_batches)
    batches = [{key: value.to(device) for key, value in features.items()} for features, _ in cpu_batches]
    parameters = parameter_audit(model, batches[0], torch)
    torch.cuda.synchronize()
    with torch.no_grad():
        # Full untimed pass verifies all inputs, including the partial final batch.
        for batch in batches:
            logits = main_logits(model(batch))
            if logits.shape[0] != len(next(iter(batch.values()))) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("invalid model output on actual test features")
        for index in range(args.warmup_batches):
            model(batches[index % len(batches)])
        torch.cuda.synchronize()
        idle_pre_timing = assert_idle_gpu(gpu["uuid"])
        torch.cuda.reset_peak_memory_stats()
        base_allocated = torch.cuda.memory_allocated()
        times = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for batch in batches:
                model(batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        memory = {"baseline_allocated_bytes": base_allocated,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - base_allocated,
                  "resident_input_tensor_bytes": sum(value.numel() * value.element_size()
                      for batch in batches for value in batch.values()),
                  "definition": "PyTorch process allocation; includes model, all selected GPU inputs and retained model state. Incremental peak is relative to post-warmup baseline, not full activation memory. Not nvidia-smi device memory."}
    idle_after = assert_idle_gpu(gpu["uuid"])
    sources_now = source_provenance()
    result = {"status": "completed", "protocol": "downstream_feature_forward_v1",
              "smoke_only": args.smoke_samples is not None,
              "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
              "server": args.server, "hostname": socket.gethostname(), "pid": os.getpid(),
              "gpu": gpu, "idle_checks": [idle_before, idle_pre_timing, idle_after],
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "cudnn_version": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads(),
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "dtype": "float32", "mode": "eval/no_grad", "batch_size": 32,
              "samples": len(selected), "batches": len(batches),
              "batch_sizes": [len(labels) for _, labels in cpu_batches],
              "warmup_batches": args.warmup_batches, "timed_repeats": args.repeats,
              "timing_boundary": "Default downstream forward, GPU-resident pre-extracted features; synchronized before/after each entire target sweep. Excludes feature extraction, loading, H2D, parameter audit, predictions/metrics. Includes auxiliary outputs computed by default forward.",
              "timing": summarize_times(times, len(selected)), "memory": memory,
              "parameters": parameters, "revision_control": control_metadata,
              "sample_selection": "first N test targets in dataset order; no resampling or duplication",
              "sample_indices": selected, "sample_ids": selected_ids,
              "input_tensor_sha256": input_sha, "sample_ids_sha256": json_hash(selected_ids),
              "artifact_dir": str(artifact), "checkpoint": str(checkpoint),
              "artifact_sha256": hashes, "checkpoint_load": "strict=True",
              "artifact_validation": "Completed bundle, stored strict-replay flags and all six artifact checksums verified; no fresh predictive replay or accuracy measurement is claimed.",
              "feature_sources": features_provenance, "artifact_config": config,
              "artifact_manifest": manifest, "source_provenance": sources_now,
              "checkpoint_source_comparison": compare_recorded_sources(
                  manifest.get("source_hashes", {}), sources_now["loaded_source_sha256"]),
              "finished_at_unix": time.time()}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=("iemocap", "meld"))
    parser.add_argument("--variant", required=True, choices=EFFICIENCY_VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", required=True, help="physical host index or complete GPU UUID")
    parser.add_argument("--server", choices=("biggpu", "local"), default="biggpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--warmup-batches", default=20, type=int)
    parser.add_argument("--cpu-threads", default=1, type=int)
    parser.add_argument("--smoke-samples", type=int, help="marks output nonformal; 1..1000 real targets")
    parser.add_argument("--path-map", action="append", default=[], help="OLD_PREFIX=NEW_PREFIX for relocated feature files")
    parser.add_argument("--meld-csv-dir", help="directory containing test_sent_emo.csv")
    parser.add_argument("--skip-missing", action="store_true", help="write explicit skipped status if required artifacts/features are absent")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {args.output}")
    try:
        result = benchmark(args)
    except Exception as error:
        skipped = args.skip_missing and isinstance(error, FileNotFoundError)
        result = {"status": "skipped" if skipped else "failed", "dataset": args.dataset,
                  "variant": args.variant, "seed": args.seed, "error": str(error),
                  "error_type": type(error).__name__, "artifact_dir": args.artifact_dir}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not skipped:
            raise
        print(json.dumps(result))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "dataset", "variant", "seed", "smoke_only", "samples", "timing")}))


if __name__ == "__main__":
    main()
