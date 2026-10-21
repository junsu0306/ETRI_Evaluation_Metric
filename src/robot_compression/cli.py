from __future__ import annotations

# Load .env before importing torch so CUDA_VISIBLE_DEVICES is applied correctly.
from robot_compression.config import Paths, centroid_path

import argparse
import csv
import hashlib
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
    apply_structural_reduction,
    parameter_report,
    prune_model,
)
from robot_compression.data import (
    get_training_query_set,
    get_whole_set,
    get_whole_training_set,
    validate_layout,
)
from robot_compression.models import build_model, load_centroids
from robot_compression.training import (
    build_clusters,
    build_feature_cache,
    evaluate,
    relational_distillation_loss,
    train_one_epoch,
)


ARCHITECTURES = [
    "alexnet",
    "vgg16",
    "resnet18",
    "mobilenetv2",
    "mobilenetv3-small",
    "mobilenetv3-large",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ETRI NetVLAD compression evaluation")
    parser.add_argument("mode", choices=["check", "smoke", "cluster", "train", "test", "reduce"])
    parser.add_argument("--arch", choices=ARCHITECTURES, default="mobilenetv2")
    parser.add_argument("--teacher-arch", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path, help="New training run directory (must not exist)")
    parser.add_argument("--split", choices=["train", "val", "test", "val250k", "test250k"], default="val")
    parser.add_argument("--pooling", choices=["netvlad", "max", "avg"], default="netvlad")
    parser.add_argument("--num-clusters", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Stop after this many consecutive non-improving epochs; 0 disables early stopping",
    )
    parser.add_argument("--early-stop-rank", type=int, choices=[1, 5], default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=0.5,
        help="Lambda in (1-lambda)*triplet + lambda*query-ranking KD",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for query-ranking KD",
    )
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
        "kd_method": "relational-ranking-v1",
        "kd_objective": "convex-combination-v1",
        "temperature": float(args.temperature),
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


def _checkpoint_cache_key(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


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
        model, _ = apply_structural_reduction(model, kept)
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

    if model.arch in {"mobilenetv2", "mobilenetv3-large"} and args.pooling == "netvlad":
        prune_model(model, args.pruning_ratio)
        pruned = parameter_report(model)
        reduced, kept = apply_structural_reduction(model)
        reduced_output = reduced(sample)
        after = parameter_report(reduced)
        print(f"parameters_pruned={pruned.to_dict()}")
        print(f"parameters_reduced={after.to_dict()} kept_channels={len(kept)}")
        print(f"reduced_forward_shape={tuple(reduced_output.shape)}")

        model.zero_grad(set_to_none=True)
        kd_samples = torch.randn(4, 3, 224, 224, device=device)
        student_embedding = model(kd_samples)
        with torch.no_grad():
            teacher_embedding = torch.randn(4, 13, device=device)
        kd = relational_distillation_loss(
            student_embedding,
            teacher_embedding,
            query_count=1,
            negative_counts=torch.tensor([2]),
            temperature=args.temperature,
        )
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
        model, payload = load_model_checkpoint(args.checkpoint, device)
        dataset = get_whole_set(args.split, paths.pittsburgh_root)
        recalls = evaluate(model, dataset, device, args.cache_batch_size, args.workers)
        result = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", 0)),
            "split": args.split,
            "recalls": {f"recall@{rank}": value for rank, value in recalls.items()},
            "parameter_report": parameter_report(model).to_dict(),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(f"evaluation={args.output}")
        print(json.dumps(result, indent=2))
        return 0

    if args.mode == "reduce":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for reduce mode")
        model, payload = load_model_checkpoint(args.checkpoint, device)
        if args.pruning_ratio > 0:
            prune_model(model, args.pruning_ratio)
        reduced, kept = apply_structural_reduction(model)
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
    if not 0.0 <= args.kd_weight <= 1.0:
        raise ValueError("--kd-weight must satisfy 0 <= value <= 1")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    if args.run_dir is not None:
        if args.checkpoint is not None:
            raise ValueError("Use --checkpoint alone when resuming an existing run")
        if args.run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite run directory: {args.run_dir}")
    whole_train = get_whole_training_set(paths.pittsburgh_root)
    train_dataset = get_training_query_set(paths.pittsburgh_root, args.margin)
    validation = get_whole_set("val", paths.pittsburgh_root)
    if args.kd_weight > 0 and args.pooling != "netvlad":
        raise ValueError("Knowledge distillation currently requires --pooling netvlad")
    if args.pruning and args.arch not in {"mobilenetv2", "mobilenetv3-large"}:
        raise ValueError(
            "Structural pruning supports mobilenetv2 and mobilenetv3-large only"
        )

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
        if args.kd_weight > 0 and resume_payload.get("kd_method") != "relational-ranking-v1":
            raise ValueError("Resume checkpoint does not use relational-ranking-v1 KD")
        if (
            args.kd_weight > 0
            and resume_payload.get("kd_objective") != "convex-combination-v1"
        ):
            raise ValueError("Resume checkpoint does not use convex-combination-v1 KD")
        checkpoint_temperature = float(resume_payload.get("temperature", 0.1))
        if (
            args.kd_weight > 0
            and abs(checkpoint_temperature - args.temperature) > 1e-12
        ):
            raise ValueError("Resume KD temperature does not match --temperature")
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
        not_improved = int(resume_payload.get("not_improved", 0))
        if args.early_stop_rank == 1 and "not_improved_r1" not in resume_payload:
            raise ValueError("Legacy checkpoint has no R@1 patience counter; cannot resume R@1 stopping")
        not_improved_r1 = int(resume_payload.get("not_improved_r1", 0))
        run_dir = args.checkpoint.resolve().parent.parent
    else:
        start_epoch = 0
        best = 0.0
        best_r1 = 0.0
        not_improved = 0
        not_improved_r1 = 0
        pruning_tag = (
            f"_prune{str(args.pruning_ratio).replace('.', 'p')}" if args.pruning else ""
        )
        kd_tag = f"_kdrel{str(args.kd_weight).replace('.', 'p')}" if args.kd_weight > 0 else ""
        run_dir = paths.runs_root / datetime.now().strftime(
            f"%Y%m%d_%H%M%S_{model.arch}_{args.pooling}{pruning_tag}{kd_tag}"
        )
        if args.run_dir is not None:
            run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    if resume_payload is None:
        (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    _write_run_metadata(run_dir, args, paths, resumed=resume_payload is not None)
    metrics_path = run_dir / "metrics.csv"
    teacher_cache_path = None
    if args.kd_weight > 0:
        cache_key = _checkpoint_cache_key(args.teacher_checkpoint)
        teacher_cache_path = (
            paths.cache_root / f"teacher_{cache_key}_pitts30k_train.hdf5"
        )
        if not teacher_cache_path.is_file():
            print(f"building_teacher_cache={teacher_cache_path}")
            build_feature_cache(
                teacher,
                whole_train,
                teacher_cache_path,
                device,
                args.cache_batch_size,
                args.workers,
            )
        print(f"teacher_cache={teacher_cache_path}")
        teacher = teacher.cpu()
        torch.cuda.empty_cache()
    try:
        stop_reason = "max_epochs"
        completed_epoch = start_epoch
        for epoch in range(start_epoch + 1, args.epochs + 1):
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            losses = train_one_epoch(
                model,
                teacher,
                train_dataset,
                whole_train,
                optimizer,
                criterion,
                device,
                paths.cache_root / f"{run_dir.name}_pitts30k_train_features.hdf5",
                args.batch_size,
                args.cache_batch_size,
                args.workers,
                kd_weight=args.kd_weight,
                temperature=args.temperature,
                pruning=args.pruning,
                pruning_ratio=args.pruning_ratio,
                teacher_cache_path=teacher_cache_path,
            )
            recalls = evaluate(model, validation, device, args.cache_batch_size, args.workers)
            writer.add_scalar("train/loss", losses.total, epoch)
            writer.add_scalar("train/triplet_loss", losses.triplet, epoch)
            writer.add_scalar("train/kd_loss", losses.kd, epoch)
            for rank, value in recalls.items():
                writer.add_scalar(f"val/recall@{rank}", value, epoch)
            is_best = recalls[5] > best
            is_best_r1 = recalls[1] > best_r1
            not_improved = 0 if is_best else not_improved + 1
            not_improved_r1 = 0 if is_best_r1 else not_improved_r1 + 1
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
                not_improved=not_improved,
                not_improved_r1=not_improved_r1,
                early_stop_rank=args.early_stop_rank,
                patience=args.patience,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                learning_rate=epoch_learning_rate,
                losses={
                    "total": losses.total,
                    "triplet": losses.triplet,
                    "kd": losses.kd,
                },
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
                "loss": losses.total,
                "triplet_loss": losses.triplet,
                "kd_loss": losses.kd,
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
            completed_epoch = epoch
            print(json.dumps(row), flush=True)
            stalled = not_improved_r1 if args.early_stop_rank == 1 else not_improved
            if args.patience > 0 and stalled >= args.patience:
                stop_reason = "early_stopping"
                print(
                    f"early_stopping=R@{args.early_stop_rank} did not improve for {stalled} epochs",
                    flush=True,
                )
                break
    finally:
        writer.close()
    (run_dir / "training_complete.json").write_text(json.dumps({
        "completed_epoch": completed_epoch,
        "stop_reason": stop_reason,
        "early_stop_rank": args.early_stop_rank,
        "patience": args.patience,
        "not_improved_r1": not_improved_r1,
        "not_improved_r5": not_improved,
        "finished_at": datetime.now().astimezone().isoformat(),
    }, indent=2))
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
