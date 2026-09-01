from __future__ import annotations

# Load .env before importing torch so CUDA_VISIBLE_DEVICES is applied correctly.
from robot_compression.config import Paths, centroid_path

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from robot_compression.compression import (
    apply_mobilenet_v2_reduction,
    parameter_report,
    prune_mobilenet_v2_last_output,
)
from robot_compression.data import (
    get_training_query_set,
    get_whole_set,
    get_whole_training_set,
    validate_layout,
)
from robot_compression.models import build_model, load_centroids
from robot_compression.training import build_clusters, evaluate, train_one_epoch
from robot_compression.training.engine import distillation_loss


ARCHITECTURES = ["alexnet", "vgg16", "resnet18", "mobilenetv2", "mobilenetv3-small"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ETRI NetVLAD compression evaluation")
    parser.add_argument("mode", choices=["check", "smoke", "cluster", "train", "test", "reduce"])
    parser.add_argument("--arch", choices=ARCHITECTURES, default="mobilenetv2")
    parser.add_argument("--teacher-arch", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test", "val250k", "test250k"], default="val")
    parser.add_argument("--pooling", choices=["netvlad", "max", "avg"], default="netvlad")
    parser.add_argument("--num-clusters", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--pruning-ratio", type=float, default=0.15)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-descriptors", type=int, default=50_000)
    parser.add_argument("--descriptors-per-image", type=int, default=100)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --cpu only for smoke checks.")
    # CUDA_VISIBLE_DEVICES=2 maps physical GPU 2 to logical cuda:0.
    return torch.device("cuda:0")


def checkpoint_payload(model, args, epoch: int = 0, best_recall: float = 0.0, **extra):
    return {
        "format_version": 2,
        "epoch": epoch,
        "best_recall": best_recall,
        "arch": model.arch,
        "pooling": args.pooling,
        "num_clusters": args.num_clusters,
        "freeze_backbone": not args.from_scratch,
        "pruning": bool(args.pruning),
        "pruning_ratio": float(args.pruning_ratio),
        "kd_weight": float(args.kd_weight),
        "state_dict": model.state_dict(),
        **extra,
    }


def _git_metadata(project_root: Path) -> dict[str, object]:
    def run_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
        "status": status.splitlines(),
    }


def _write_run_metadata(run_dir: Path, args, paths: Paths, resumed: bool) -> None:
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "resumed": resumed,
        "git": _git_metadata(paths.project_root),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "<unset>"),
            "pittsburgh_root": str(paths.pittsburgh_root),
            "centroids_root": str(paths.centroids_root),
            "seed": args.seed,
        },
        "arguments": vars(args),
    }
    name = (
        f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if resumed
        else "run_metadata.json"
    )
    (run_dir / name).write_text(json.dumps(metadata, default=str, indent=2))


def _append_epoch_metrics(path: Path, row: dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_model_checkpoint(path: Path, device: torch.device, pretrained: bool = False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(
        payload["arch"],
        num_clusters=int(payload.get("num_clusters", 64)),
        pooling=payload.get("pooling", "netvlad"),
        pretrained=pretrained,
        freeze_backbone=bool(payload.get("freeze_backbone", True)),
    )
    kept = payload.get("reduction_indices")
    if kept is not None:
        model, _ = apply_mobilenet_v2_reduction(model, kept)
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def run_check(paths: Paths) -> int:
    print(f"project_root={paths.project_root}")
    print(f"pittsburgh_root={paths.pittsburgh_root}")
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()} visible_devices={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"visible_gpu_0={torch.cuda.get_device_name(0)}")
    missing = validate_layout(paths.pittsburgh_root)
    if missing:
        print("dataset_status=missing")
        for item in missing:
            print(f"  missing: {item}")
        return 2
    dataset = get_whole_set("val", paths.pittsburgh_root)
    print(f"dataset_status=ready db={dataset.dbStruct.numDb} query={dataset.dbStruct.numQ}")
    return 0


def run_smoke(args, device: torch.device) -> None:
    model = build_model(
        args.arch,
        num_clusters=args.num_clusters,
        pooling=args.pooling,
        pretrained=False,
    ).to(device)
    sample = torch.randn(1, 3, 224, 224, device=device)
    output = model(sample)
    print(f"forward_shape={tuple(output.shape)}")
    before = parameter_report(model)
    print(f"parameters_before={before.to_dict()}")

    if model.arch == "mobilenetv2" and args.pooling == "netvlad":
        prune_mobilenet_v2_last_output(model, args.pruning_ratio)
        pruned = parameter_report(model)
        reduced, kept = apply_mobilenet_v2_reduction(model)
        reduced_output = reduced(sample)
        after = parameter_report(reduced)
        print(f"parameters_pruned={pruned.to_dict()}")
        print(f"parameters_reduced={after.to_dict()} kept_channels={len(kept)}")
        print(f"reduced_forward_shape={tuple(reduced_output.shape)}")

        model.zero_grad(set_to_none=True)
        student_features = model.encoder(sample)
        student_logits = model.pool.get_soft_assign_logit(student_features)
        with torch.no_grad():
            teacher_logits = student_logits.detach() + 0.01 * torch.randn_like(student_logits)
        kd = distillation_loss(student_logits, teacher_logits, args.temperature)
        kd.backward()
        gradient = model.pool.conv.weight.grad
        if gradient is None or gradient.norm() == 0:
            raise RuntimeError("KD smoke check failed: student received no gradient")
        print(f"kd_gradient_norm={gradient.norm().item():.8f}")


def require_dataset(paths: Paths) -> None:
    missing = validate_layout(paths.pittsburgh_root)
    if missing:
        joined = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            f"Pittsburgh dataset is incomplete:\n{joined}\nRun scripts/download_pittsburgh.sh"
        )


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    paths = Paths()
    paths.create_runtime_dirs()
    set_seed(args.seed)

    if args.mode == "check":
        return run_check(paths)

    device = get_device(args.cpu)
    print(f"device={device} physical_gpu={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if args.mode == "smoke":
        run_smoke(args, device)
        return 0

    require_dataset(paths)
    if args.mode == "cluster":
        dataset = get_whole_training_set(paths.pittsburgh_root, only_db=True)
        model = build_model(
            args.arch,
            num_clusters=args.num_clusters,
            pretrained=not args.from_scratch,
            cluster_mode=True,
        ).to(device)
        output = centroid_path(paths, model.arch, dataset.dataset, args.num_clusters)
        build_clusters(
            model,
            dataset,
            output,
            device,
            args.num_clusters,
            args.num_descriptors,
            args.descriptors_per_image,
            args.cache_batch_size,
            args.workers,
        )
        print(f"centroids={output}")
        return 0

    if args.mode == "test":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for test mode")
        model, _ = load_model_checkpoint(args.checkpoint, device)
        dataset = get_whole_set(args.split, paths.pittsburgh_root)
        recalls = evaluate(model, dataset, device, args.cache_batch_size, args.workers)
        print(json.dumps({f"recall@{rank}": value for rank, value in recalls.items()}, indent=2))
        return 0

    if args.mode == "reduce":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for reduce mode")
        model, payload = load_model_checkpoint(args.checkpoint, device)
        if args.pruning_ratio > 0:
            prune_mobilenet_v2_last_output(model, args.pruning_ratio)
        reduced, kept = apply_mobilenet_v2_reduction(model)
        output = args.output or args.checkpoint.with_name(
            f"{args.checkpoint.stem}_reduced.pt"
        )
        # A structurally reduced checkpoint cannot resume the original optimizer:
        # its parameter shapes no longer match.  Keep deployment metadata but do
        # not carry large, incompatible training-only states into the artifact.
        deployment_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"optimizer", "scheduler"}
        }
        reduced_payload = {
            **deployment_payload,
            "state_dict": reduced.state_dict(),
            "reduction_indices": kept,
            "parameter_report": parameter_report(reduced).to_dict(),
            "source_checkpoint": str(args.checkpoint),
        }
        torch.save(reduced_payload, output)
        print(f"reduced_checkpoint={output}")
        print(json.dumps(reduced_payload["parameter_report"], indent=2))
        return 0

    if args.kd_weight > 0 and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required when --kd-weight is positive")
    whole_train = get_whole_training_set(paths.pittsburgh_root)
    train_dataset = get_training_query_set(paths.pittsburgh_root, args.margin)
    validation = get_whole_set("val", paths.pittsburgh_root)
    if args.kd_weight > 0 and args.pooling != "netvlad":
        raise ValueError("Knowledge distillation currently requires --pooling netvlad")
    if args.pruning and args.arch != "mobilenetv2":
        raise ValueError("Structural pruning currently supports --arch mobilenetv2 only")

    resume_payload = None
    if args.checkpoint is not None:
        model, resume_payload = load_model_checkpoint(args.checkpoint, device)
        if resume_payload.get("reduction_indices") is not None:
            raise ValueError("Training cannot resume from a structurally reduced checkpoint")
        if model.arch != args.arch:
            raise ValueError(
                f"Resume architecture mismatch: checkpoint={model.arch}, args={args.arch}"
            )
        if resume_payload.get("pooling", "netvlad") != args.pooling:
            raise ValueError("Resume pooling does not match --pooling")
        if int(resume_payload.get("num_clusters", 64)) != args.num_clusters:
            raise ValueError("Resume cluster count does not match --num-clusters")
        if bool(resume_payload.get("pruning", False)) != args.pruning:
            raise ValueError("Resume pruning setting does not match --pruning")
        checkpoint_ratio = float(resume_payload.get("pruning_ratio", 0.0))
        if args.pruning and abs(checkpoint_ratio - args.pruning_ratio) > 1e-12:
            raise ValueError("Resume pruning ratio does not match --pruning-ratio")
        if abs(float(resume_payload.get("kd_weight", 0.0)) - args.kd_weight) > 1e-12:
            raise ValueError("Resume KD weight does not match --kd-weight")
    else:
        model = build_model(
            args.arch,
            args.num_clusters,
            args.pooling,
            pretrained=not args.from_scratch,
        )
        centroids = centroid_path(paths, model.arch, train_dataset.dataset, args.num_clusters)
        if not centroids.is_file():
            raise FileNotFoundError(f"Missing centroids: {centroids}. Run cluster mode first.")
        load_centroids(model, centroids)
        model = model.to(device)
    teacher = None
    if args.teacher_checkpoint is not None:
        teacher, _ = load_model_checkpoint(args.teacher_checkpoint, device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.TripletMarginLoss(margin=args.margin**0.5, p=2, reduction="sum")
    if resume_payload is not None:
        if "optimizer" not in resume_payload or "scheduler" not in resume_payload:
            raise ValueError("Resume checkpoint does not contain optimizer/scheduler state")
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch = int(resume_payload["epoch"])
        best = float(resume_payload.get("best_recall", 0.0))
        best_r1 = float(resume_payload.get("best_recall_at_1", 0.0))
        run_dir = args.checkpoint.resolve().parent.parent
    else:
        start_epoch = 0
        best = 0.0
        best_r1 = 0.0
        pruning_tag = (
            f"_prune{str(args.pruning_ratio).replace('.', 'p')}" if args.pruning else ""
        )
        run_dir = paths.runs_root / datetime.now().strftime(
            f"%Y%m%d_%H%M%S_{model.arch}_{args.pooling}{pruning_tag}"
        )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    if resume_payload is None:
        (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    _write_run_metadata(run_dir, args, paths, resumed=resume_payload is not None)
    metrics_path = run_dir / "metrics.csv"
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            loss = train_one_epoch(
                model,
                teacher,
                train_dataset,
                whole_train,
                optimizer,
                criterion,
                device,
                paths.cache_root / "pitts30k_train_features.hdf5",
                args.batch_size,
                args.cache_batch_size,
                args.workers,
                kd_weight=args.kd_weight,
                temperature=args.temperature,
                pruning=args.pruning,
                pruning_ratio=args.pruning_ratio,
            )
            recalls = evaluate(model, validation, device, args.cache_batch_size, args.workers)
            writer.add_scalar("train/loss", loss, epoch)
            for rank, value in recalls.items():
                writer.add_scalar(f"val/recall@{rank}", value, epoch)
            is_best = recalls[5] > best
            is_best_r1 = recalls[1] > best_r1
            best = max(best, recalls[5])
            best_r1 = max(best_r1, recalls[1])
            report = parameter_report(model)
            scheduler.step()
            payload = checkpoint_payload(
                model,
                args,
                epoch=epoch,
                best_recall=best,
                best_recall_at_1=best_r1,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                learning_rate=epoch_learning_rate,
                recalls=recalls,
                parameter_report=report.to_dict(),
            )
            torch.save(payload, checkpoint_dir / "latest.pt")
            if is_best:
                torch.save(payload, checkpoint_dir / "best.pt")
            if is_best_r1:
                torch.save(payload, checkpoint_dir / "best_r1.pt")
            row = {
                "epoch": epoch,
                "loss": loss,
                "learning_rate": epoch_learning_rate,
                "recall_at_1": recalls[1],
                "recall_at_5": recalls[5],
                "recall_at_10": recalls[10],
                "recall_at_20": recalls[20],
                "best_r1": best_r1,
                "best_r5": best,
                "is_best_r1": int(is_best_r1),
                "is_best_r5": int(is_best),
                "total_parameters": report.total,
                "trainable_parameters": report.trainable,
                "zero_parameters": report.zero,
            }
            _append_epoch_metrics(metrics_path, row)
            print(json.dumps(row), flush=True)
    finally:
        writer.close()
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
