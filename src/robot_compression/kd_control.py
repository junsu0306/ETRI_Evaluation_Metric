"""Reproduce the V3 80% control with 0.5 * triplet and no KD inference.

The process-local criterion wrapper preserves the objective and logging used
in the reported experiment. Paths and GPU selection come from the normal CLI.
"""
from __future__ import annotations

import inspect
import sys

OBJECTIVE = "fixed-triplet-half-kd-off-v1"


def scaled_epoch(original, *args, **kwargs):
    from robot_compression.training.engine import EpochLosses

    bound = inspect.signature(original).bind(*args, **kwargs)
    if bound.arguments.get("kd_weight", 0.5) != 0:
        raise ValueError("The KD-off wrapper only accepts kd_weight=0")
    criterion = bound.arguments["criterion"]
    bound.arguments["criterion"] = lambda *a, **kw: 0.5 * criterion(*a, **kw)
    teacher = bound.arguments.get("teacher")
    if teacher is not None:
        # Constructing the teacher matches the KD-on random-number consumption.
        # With kd_weight=0 the engine neither infers nor caches its descriptors.
        teacher.cpu()
    losses = original(*bound.args, **bound.kwargs)
    if losses.kd != 0:
        raise ValueError("Unexpected nonzero KD loss in the control")
    return EpochLosses(total=losses.total, triplet=losses.triplet / 0.5, kd=0.0)


def run_training(argv):
    from robot_compression import cli

    args = cli.create_parser().parse_args(argv)
    if (
        args.mode != "train"
        or args.arch != "mobilenetv3-large"
        or args.kd_weight != 0
        or args.num_clusters != 64
        or not args.pruning
        or abs(args.pruning_ratio - 472 / 960) > 1e-12
        or args.checkpoint is not None
        or args.teacher_checkpoint is None
    ):
        raise ValueError(
            "Use a fresh V3-Large/NetVLAD64 c80 run with --kd-weight 0, "
            "--pruning --pruning-ratio 0.49166666666666664 and "
            "the same --teacher-checkpoint as KD-on"
        )
    original_epoch = cli.train_one_epoch
    original_parser = cli.create_parser
    original_payload = cli.checkpoint_payload

    def parser():
        result = original_parser()
        result.set_defaults(triplet_weight=0.5, kd_objective=OBJECTIVE)
        return result

    def payload(*args, **kwargs):
        result = original_payload(*args, **kwargs)
        result.update(triplet_weight=0.5, kd_objective=OBJECTIVE, kd_weight=0.0)
        return result

    cli.create_parser = parser
    cli.train_one_epoch = lambda *a, **kw: scaled_epoch(original_epoch, *a, **kw)
    cli.checkpoint_payload = payload
    try:
        print("control_objective=0.5*triplet+0*KD; teacher inference disabled")
        return cli.main(argv)
    finally:
        cli.create_parser = original_parser
        cli.train_one_epoch = original_epoch
        cli.checkpoint_payload = original_payload


if __name__ == "__main__":
    raise SystemExit(run_training(sys.argv[1:]))
