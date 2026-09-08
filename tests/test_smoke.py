import torch

from robot_compression.compression import (
    apply_mobilenet_v3_large_reduction,
    apply_mobilenet_v2_reduction,
    parameter_report,
    prune_mobilenet_v3_large_last_expansion,
    prune_mobilenet_v2_last_output,
)
from robot_compression.models import NetVLAD, build_model
from robot_compression.training.engine import (
    _teacher_cache_indices,
    distillation_loss,
    relational_distillation_loss,
    set_train_mode_with_frozen_backbone,
)


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


def test_relational_kd_supports_different_descriptor_dimensions():
    student = torch.randn(5, 11, requires_grad=True)
    teacher = torch.randn(5, 7)
    loss = relational_distillation_loss(
        student,
        teacher,
        query_count=1,
        negative_counts=torch.tensor([3]),
        temperature=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert student.grad.norm() > 0
    assert not teacher.requires_grad


def test_relational_kd_is_small_when_rankings_match():
    teacher = torch.tensor(
        [
            [1.0, 0.0],  # query
            [0.9, 0.1],  # positive
            [0.2, 0.8],  # negative 1
            [-0.8, 0.2],  # negative 2
        ]
    )
    matching = teacher.clone().requires_grad_()
    reversed_student = teacher[[0, 3, 2, 1]].clone().requires_grad_()
    counts = torch.tensor([2])
    matching_loss = relational_distillation_loss(matching, teacher, 1, counts, 0.1)
    reversed_loss = relational_distillation_loss(
        reversed_student, teacher, 1, counts, 0.1
    )
    assert matching_loss < 1e-6
    assert reversed_loss > matching_loss


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


def test_26_percent_pruning_exceeds_trainable_parameter_target():
    baseline = build_model(
        "resnet18", num_clusters=64, pretrained=False, freeze_backbone=True
    )
    student = build_model(
        "mobilenetv2", num_clusters=64, pretrained=False, freeze_backbone=True
    )
    prune_mobilenet_v2_last_output(student, 0.26)
    reduced, kept = apply_mobilenet_v2_reduction(student)

    compression = 1.0 - (
        parameter_report(reduced).trainable / parameter_report(baseline).trainable
    )
    assert len(kept) == 237
    assert parameter_report(reduced).trainable == 424_410
    assert compression >= 0.80


def test_discrete_channels_for_76_and_80_percent_targets():
    baseline = build_model(
        "resnet18", num_clusters=64, pretrained=False, freeze_backbone=True
    )
    expected = [
        (3 / 320, 317, 511_610, 0.76),
        (82 / 320, 238, 425_500, 0.80),
    ]
    for ratio, channels, trainable, target in expected:
        student = build_model(
            "mobilenetv2", num_clusters=64, pretrained=False, freeze_backbone=True
        )
        prune_mobilenet_v2_last_output(student, ratio)
        reduced, kept = apply_mobilenet_v2_reduction(student)
        compression = 1.0 - (
            parameter_report(reduced).trainable / parameter_report(baseline).trainable
        )
        assert len(kept) == channels
        assert parameter_report(reduced).trainable == trainable
        assert compression >= target


def test_mobilenet_v3_large_target_compressions_and_forward():
    baseline = build_model(
        "resnet18", num_clusters=4, pretrained=False, freeze_backbone=True
    )
    baseline_trainable_64_clusters = 2_132_480
    expected = [
        (266 / 960, 694, 597_060, 0.72),
        (369 / 960, 591, 511_570, 0.76),
        (472 / 960, 488, 426_080, 0.80),
    ]
    # The reference trainable count is defined with 64 NetVLAD clusters.  Four
    # clusters keep this structural forward test small, so account for the
    # missing 60 * 160 * 2 NetVLAD parameters when checking the target count.
    cluster_adjustment = (64 - 4) * 160 * 2
    for ratio, hidden_channels, expected_trainable, target in expected:
        student = build_model(
            "mobilenetv3-large",
            num_clusters=4,
            pretrained=False,
            freeze_backbone=True,
        )
        prune_mobilenet_v3_large_last_expansion(student, ratio)
        reduced, kept = apply_mobilenet_v3_large_reduction(student)
        adjusted_trainable = parameter_report(reduced).trainable + cluster_adjustment
        compression = 1.0 - adjusted_trainable / baseline_trainable_64_clusters
        output = reduced(torch.randn(1, 3, 64, 64))
        assert len(kept) == hidden_channels
        assert adjusted_trainable == expected_trainable
        assert compression >= target
        assert output.shape == (1, 4 * 160)


def test_frozen_backbone_batch_norm_stays_in_eval_mode():
    student = build_model(
        "mobilenetv3-large",
        num_clusters=4,
        pretrained=False,
        freeze_backbone=True,
    )
    set_train_mode_with_frozen_backbone(student)
    assert not student.encoder[0].training
    assert not student.encoder[-2].training
    assert student.encoder[-1].training
    assert student.pool.training


def test_teacher_cache_indices_match_concatenated_image_order():
    # collate_fn flattens [q, p, n...] for each query, while the network input
    # is concatenated as [all q, all p, all n].
    flattened = [4, 10, 20, 21, 5, 11, 30]
    ordered = _teacher_cache_indices(flattened, torch.tensor([2, 1]), 100)
    assert ordered == [104, 105, 10, 11, 20, 21, 30]
