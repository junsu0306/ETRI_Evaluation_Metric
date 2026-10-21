#!/usr/bin/env python3
"""Validate the published evidence without datasets, weights or a GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(directory: Path):
    directory = directory.resolve()
    provenance = json.loads((directory / "provenance.json").read_text())
    for relative, expected in provenance["export_sha256"].items():
        path = directory / Path(relative).relative_to("results")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Checksum mismatch: {relative}")

    rows = read_csv(directory / "final_comparison.csv")
    indexed = {(r["model"], r["form"]): r for r in rows}
    require(len(indexed) == len(rows), "Duplicate model/form rows")
    for model, metadata in provenance["models"].items():
        metrics = read_csv(directory / "epochs" / f"{model}.csv")
        require([int(r["epoch"]) for r in metrics] == list(range(1, metadata["completed_epoch"] + 1)), f"Incomplete epoch history: {model}")
        rank = metadata["selection_metric"].rsplit("@", 1)[1]
        best = max(metrics, key=lambda r: float(r[f"recall_at_{rank}"]))
        require(int(best["epoch"]) == metadata["selected_epoch"], f"Selection mismatch: {model}")
        for form, checkpoint in metadata["checkpoints"].items():
            row = indexed[(model, form)]
            require(int(row["epoch"]) == metadata["selected_epoch"], f"Wrong epoch: {model}/{form}")
            for field in ("sha256", "bytes"):
                require(str(row[field]) == str(checkpoint[field]), f"Wrong {field}: {model}/{form}")
            require(row["checkpoint"] == checkpoint["path"], f"Wrong checkpoint: {model}/{form}")
            require(int(row["total"]) >= int(row["trainable"]) > 0, f"Invalid parameter counts: {model}/{form}")
            for split in ("val", "test"):
                recalls = [float(row[f"{split}_r{k}"]) for k in (1, 5, 10, 20)]
                require(all(0 <= r <= 1 for r in recalls) and recalls == sorted(recalls), f"Invalid recalls: {model}/{form}/{split}")
            if form in ("unpruned", "zero_pruned"):
                for k in (1, 5, 10, 20):
                    require(float(row[f"val_r{k}"]) == float(best[f"recall_at_{k}"]), f"Validation/epoch mismatch: {model}/{form}")
            for split, evidence in checkpoint["evaluations"].items():
                evaluation = json.loads((directory / Path(evidence["export"]).relative_to("results")).read_text())
                require(evaluation["split"] == split and evaluation["checkpoint_epoch"] == int(row["epoch"]), f"Evaluation split/epoch mismatch: {model}/{form}")
                require(evaluation["checkpoint"] == row["checkpoint"], f"Evaluation checkpoint mismatch: {model}/{form}")
                for k in (1, 5, 10, 20):
                    require(evaluation["recalls"][f"recall@{k}"] == float(row[f"{split}_r{k}"]), f"Evaluation recall mismatch: {model}/{form}")
                for key in ("total", "trainable"):
                    require(evaluation["parameter_report"][key] == int(row[key]), f"Evaluation parameter mismatch: {model}/{form}")

    for row in read_csv(directory / "kd_ablation.csv"):
        model = "v3_c80_kdon_teacher14" if row["condition"] == "on" else "v3_c80_kdoff"
        reference = indexed[(model, row["form"])]
        require(all(reference[key] == value for key, value in row.items() if key != "condition"), "Ablation/comparison mismatch")
    for model in provenance["models"]:
        if model.startswith("v3_"):
            before, after = indexed[(model, "zero_pruned")], indexed[(model, "reduced")]
            require(all(before[f"{split}_r{k}"] == after[f"{split}_r{k}"] for split in ("val", "test") for k in (1, 5, 10, 20)), f"V3 reduction changed measured recall: {model}")
    on = indexed[("v3_c80_kdon_teacher14", "reduced")]
    off = indexed[("v3_c80_kdoff", "reduced")]
    delta = 100 * (float(on["test_r1"]) - float(off["test_r1"]))
    print(f"Verified {len(provenance['export_sha256'])} file checksums, {len(rows)} comparison rows, {len(provenance['models'])} epoch histories.")
    print(f"Reduced c80 KD-on minus KD-off Test R@1: {delta:+.4f} percentage points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    verify(parser.parse_args().results)
