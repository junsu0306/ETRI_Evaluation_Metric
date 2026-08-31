import torch

from robot_compression.compression import (
    apply_mobilenet_v2_reduction,
    parameter_report,
    prune_mobilenet_v2_last_output,
)
from robot_compression.models import NetVLAD, build_model
from robot_compression.training.engine import distillation_loss


def test_netvlad_output_is_normalized():
    layer = NetVLAD(num_clusters=4, dim=8)
    output = layer(torch.randn(2, 8, 5, 5))
    assert output.shape == (2, 32)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)


def test_kd_loss_reaches_student():
    student = torch.randn(2, 4, 6, requires_grad=True)
    teacher = torch.randn(2, 4, 9)
    loss = distillation_loss(student, teacher, temperature=1.5)
    loss.backward()
    assert student.grad is not None
    assert student.grad.norm() > 0


def test_mobilenet_pruning_and_reduction_forward():
    model = build_model("mobilenetv2", num_clusters=4, pretrained=False)
    before = parameter_report(model)
    prune_mobilenet_v2_last_output(model, 0.2)
    reduced, kept = apply_mobilenet_v2_reduction(model)
    after = parameter_report(reduced)
    output = reduced(torch.randn(1, 3, 64, 64))
    assert len(kept) == 256
    assert after.total < before.total
    assert output.shape == (1, 4 * len(kept))


def test_documented_parameter_counts_are_reproduced():
    baseline = build_model(
        "resnet18", num_clusters=64, pretrained=False, freeze_backbone=True
    )
    student = build_model(
        "mobilenetv2", num_clusters=64, pretrained=False, freeze_backbone=True
    )
    prune_mobilenet_v2_last_output(student, 0.15)
    reduced, kept = apply_mobilenet_v2_reduction(student)

    assert parameter_report(baseline).total == 2_815_552
    assert parameter_report(baseline).trainable == 2_132_480
    assert parameter_report(student).trainable == 514_880
    assert len(kept) == 272
    assert parameter_report(reduced).total == 1_800_352
    assert parameter_report(reduced).trainable == 462_560
