#!/usr/bin/env python3
"""Verify saved baseline counts and aggregate explicit matrix/conv FLOPs."""
import argparse
import csv
import json
import math
from pathlib import Path


NAMES = {
    "dialoguernn": "DialogueRNN", "mmgcn": "MMGCN", "mmdfn": "MM-DFN",
    "m3net": "M3Net", "ada2i": "Ada2I", "sdt": "SDT", "css": "CSS",
}


def summarize(root):
    rows = {}
    for group in ("rnn", "graph", "transformers"):
        for path in sorted((root / group).glob("*.json")):
            data = json.loads(path.read_text())
            if "model" not in data or "dataset" not in data:
                continue  # Counter self-tests are separate artifacts.
            model, dataset = data["model"].lower(), data["dataset"]
            assert model in NAMES and dataset in ("iemocap", "meld"), path
            assert data["status"] == "completed", path
            assert (model, dataset) not in rows, f"Duplicate result: {path}"
            records = data["records"]
            total = sum(sum(r["counted_operators"].values()) for r in records)
            # The comparison excludes elementwise/scatter message aggregation.
            assert total == data["total_flops"], f"Counting scope differs: {path}"
            n = sum(r["valid_utterances"] for r in records)
            assert n == data["valid_utterances"] == (1623 if dataset == "iemocap" else 2610), path
            assert math.isclose(total / n, data["flops_per_utterance"], rel_tol=1e-12), path
            diffs = [r["max_abs_logit_difference"] for r in records]
            assert all(math.isfinite(v) and v >= 0 for v in diffs), path
            rows[(model, dataset)] = {
                "model": NAMES[model], "dataset": dataset,
                "matrix_convolution_flops_per_utterance": total / n,
                "MFLOPs_per_utterance": total / n / 1e6,
                "total_test_flops": total, "test_utterances": n,
                "registered_parameters": data["registered_parameters"],
                "batch_unit": "32 dialogues, final batch may be smaller",
                "weight_source": data.get("weight_source", "trained_checkpoint"),
                "max_abs_output_difference": max(diffs),
                "source": str(path.relative_to(root)),
            }
    assert len(rows) == 14, f"Expected 14 baseline results; have {len(rows)}"
    ordered = [rows[(model, dataset)] for model in NAMES for dataset in ("iemocap", "meld")]
    for dataset in ("iemocap", "meld"):
        path = root / f"{dataset}_full.json"
        data = json.loads(path.read_text())
        values = []
        for record in data["records"]:
            total = sum(record["counted_operators"].values())
            assert total == record["matrix_convolution_flops"]
            values.append(total / record["batch_size"])
        assert len(set(values)) == 1
        ordered.append({
            "model": "MM-Mixer", "dataset": dataset,
            "matrix_convolution_flops_per_utterance": values[0],
            "MFLOPs_per_utterance": values[0] / 1e6,
            "total_test_flops": "", "test_utterances": "",
            "registered_parameters": data["registered_parameters"],
            "batch_unit": "1 and 32 utterances; fixed-shape per-utterance count",
            "weight_source": "trained_checkpoint",
            "max_abs_output_difference": max(r["max_abs_logit_difference"] for r in data["records"]),
            "source": str(path.relative_to(root)),
        })
    with (root / "baseline_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    print("Verified 14 full-test baseline results and 2 fixed-shape MM-Mixer results.")
    for row in ordered:
        print(f"{row['model']:12s} {row['dataset']:7s} {row['MFLOPs_per_utterance']:12.6f} MFLOPs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results/flops_20261004")
    summarize(parser.parse_args().results)
