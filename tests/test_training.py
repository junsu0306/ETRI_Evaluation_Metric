import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from robot_compression import cli
from robot_compression.config import Paths
from robot_compression.training.engine import EpochLosses
from robot_compression.models import build_model
from robot_compression.compression import prune_model, apply_structural_reduction


@pytest.mark.parametrize("scores,patience,expected", [
    ([0.8] + [0.7] * 15, 10, 11),
    ([0.8] + [0.7] * 9 + [0.9] + [0.9] * 12, 10, 21),
    ([0.8] + [0.7] * 11, 0, 12),
])
def test_cli_r1_stopping_saves_checkpoint_at_exact_boundary(tmp_path, monkeypatch, scores, patience, expected):
    paths = Paths(project_root=tmp_path, pittsburgh_root=tmp_path / "data",
                  runs_root=tmp_path / "runs", cache_root=tmp_path / "cache",
                  centroids_root=tmp_path / "centroids")
    centroids = tmp_path / "dummy.hdf5"
    centroids.touch()
    monkeypatch.setattr(cli, "Paths", lambda: paths)
    monkeypatch.setattr(cli, "require_dataset", lambda _: None)
    monkeypatch.setattr(cli, "get_whole_training_set", lambda *_: [])
    monkeypatch.setattr(cli, "get_training_query_set", lambda *_: SimpleNamespace(dataset="pitts30k"))
    monkeypatch.setattr(cli, "get_whole_set", lambda *_: [])
    monkeypatch.setattr(cli, "centroid_path", lambda *_: centroids)
    monkeypatch.setattr(cli, "load_centroids", lambda *_: None)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    def build(*args, **kwargs):
        model = torch.nn.Linear(1, 1)
        model.arch = "mobilenetv2"
        return model
    monkeypatch.setattr(cli, "build_model", build)
    def train(model, teacher, dataset, whole, optimizer, *args, **kwargs):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        return EpochLosses(0.1, 0.1, 0.0)
    monkeypatch.setattr(cli, "train_one_epoch", train)
    calls = []
    def evaluate(*_):
        i = len(calls)
        calls.append(i)
        # R@5 improves every epoch: only R@1 can cause this stop.
        return {1: scores[i], 5: 0.9 + i / 1000, 10: 0.97, 20: 0.99}
    monkeypatch.setattr(cli, "evaluate", evaluate)
    run = tmp_path / "chosen_run"
    assert cli.main(["train", "--cpu", "--kd-weight", "0", "--run-dir", str(run),
                     "--epochs", str(len(scores)), "--patience", str(patience),
                     "--early-stop-rank", "1"]) == 0
    assert len(calls) == expected
    payload = torch.load(run / "checkpoints/latest.pt", weights_only=True)
    assert payload["epoch"] == expected
    assert payload["not_improved"] == 0
    assert payload["not_improved_r1"] == (10 if patience else 11)
    completion = json.loads((run / "training_complete.json").read_text())
    assert completion["stop_reason"] == ("early_stopping" if patience else "max_epochs")
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    assert len(rows) == expected
    best = torch.load(run / "checkpoints/best_r1.pt", weights_only=True)
    assert best["epoch"] == (11 if expected == 21 else 1)


@pytest.mark.parametrize("ratio", [266 / 960, 369 / 960, 472 / 960])
def test_v3_reduction_preserves_zero_pruned_descriptor(ratio):
    model = build_model("mobilenetv3-large", num_clusters=4, pretrained=False, freeze_backbone=True)
    prune_model(model, ratio)
    model.eval()
    reduced, _ = apply_structural_reduction(model)
    reduced.eval()
    with torch.no_grad():
        images = torch.randn(2, 3, 64, 64)
        torch.testing.assert_close(model(images), reduced(images), atol=1e-5, rtol=1e-4)
