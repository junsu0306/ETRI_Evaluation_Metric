from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn

from robot_compression.models.factory import PlaceRecognitionModel
from robot_compression.models.netvlad import NetVLAD


@dataclass(frozen=True)
class ParameterReport:
    total: int
    trainable: int
    nonzero: int
    zero: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def parameter_report(model: nn.Module) -> ParameterReport:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    nonzero = sum((parameter != 0).sum().item() for parameter in model.parameters())
    return ParameterReport(total, trainable, nonzero, total - nonzero)


def _last_projection(model: PlaceRecognitionModel):
    if model.arch != "mobilenetv2":
        raise ValueError("Structural pruning currently supports mobilenetv2 only")
    block = model.encoder[-1]
    projection = block.conv[-2]
    batch_norm = block.conv[-1]
    if not isinstance(projection, nn.Conv2d) or not isinstance(batch_norm, nn.BatchNorm2d):
        raise TypeError("Unexpected torchvision MobileNetV2 last-block layout")
    return projection, batch_norm


def prune_mobilenet_v2_last_output(
    model: PlaceRecognitionModel, ratio: float
) -> torch.Tensor:
    if not 0.0 <= ratio < 1.0:
        raise ValueError("Pruning ratio must satisfy 0 <= ratio < 1")
    projection, batch_norm = _last_projection(model)
    norms = projection.weight.detach().flatten(1).norm(p=2, dim=1)
    prune_count = int(norms.numel() * ratio)
    if prune_count == 0:
        return torch.empty(0, dtype=torch.long, device=norms.device)
    prune_indices = torch.topk(norms, prune_count, largest=False).indices
    with torch.no_grad():
        projection.weight[prune_indices] = 0
        if projection.bias is not None:
            projection.bias[prune_indices] = 0
        batch_norm.weight[prune_indices] = 0
        batch_norm.bias[prune_indices] = 0
        batch_norm.running_mean[prune_indices] = 0
        batch_norm.running_var[prune_indices] = 1
    return prune_indices


def _mobilenet_v3_large_last_expansion(model: PlaceRecognitionModel):
    if model.arch != "mobilenetv3-large":
        raise ValueError("Expected a mobilenetv3-large model")
    block = model.encoder[-1].block
    expansion_conv = block[0][0]
    expansion_bn = block[0][1]
    depthwise_conv = block[1][0]
    depthwise_bn = block[1][1]
    squeeze_excitation = block[2]
    projection_conv = block[3][0]
    if not (
        isinstance(expansion_conv, nn.Conv2d)
        and isinstance(expansion_bn, nn.BatchNorm2d)
        and isinstance(depthwise_conv, nn.Conv2d)
        and isinstance(depthwise_bn, nn.BatchNorm2d)
        and isinstance(squeeze_excitation.fc1, nn.Conv2d)
        and isinstance(squeeze_excitation.fc2, nn.Conv2d)
        and isinstance(projection_conv, nn.Conv2d)
    ):
        raise TypeError("Unexpected torchvision MobileNetV3-Large last-block layout")
    return (
        expansion_conv,
        expansion_bn,
        depthwise_conv,
        depthwise_bn,
        squeeze_excitation,
        projection_conv,
    )


def prune_mobilenet_v3_large_last_expansion(
    model: PlaceRecognitionModel, ratio: float
) -> torch.Tensor:
    """Zero a dependency-complete set of hidden channels in the last block."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("Pruning ratio must satisfy 0 <= ratio < 1")
    (
        expansion_conv,
        expansion_bn,
        depthwise_conv,
        depthwise_bn,
        squeeze_excitation,
        projection_conv,
    ) = _mobilenet_v3_large_last_expansion(model)
    norms = expansion_conv.weight.detach().flatten(1).norm(p=2, dim=1)
    prune_count = int(norms.numel() * ratio)
    if prune_count == 0:
        return torch.empty(0, dtype=torch.long, device=norms.device)
    prune_indices = torch.topk(norms, prune_count, largest=False).indices
    with torch.no_grad():
        expansion_conv.weight[prune_indices] = 0
        if expansion_conv.bias is not None:
            expansion_conv.bias[prune_indices] = 0
        expansion_bn.weight[prune_indices] = 0
        expansion_bn.bias[prune_indices] = 0
        expansion_bn.running_mean[prune_indices] = 0
        expansion_bn.running_var[prune_indices] = 1

        depthwise_conv.weight[prune_indices] = 0
        if depthwise_conv.bias is not None:
            depthwise_conv.bias[prune_indices] = 0
        depthwise_bn.weight[prune_indices] = 0
        depthwise_bn.bias[prune_indices] = 0
        depthwise_bn.running_mean[prune_indices] = 0
        depthwise_bn.running_var[prune_indices] = 1

        squeeze_excitation.fc1.weight[:, prune_indices] = 0
        squeeze_excitation.fc2.weight[prune_indices] = 0
        if squeeze_excitation.fc2.bias is not None:
            squeeze_excitation.fc2.bias[prune_indices] = 0
        projection_conv.weight[:, prune_indices] = 0
    return prune_indices


def prune_model(model: PlaceRecognitionModel, ratio: float) -> torch.Tensor:
    if model.arch == "mobilenetv2":
        return prune_mobilenet_v2_last_output(model, ratio)
    if model.arch == "mobilenetv3-large":
        return prune_mobilenet_v3_large_last_expansion(model, ratio)
    raise ValueError(f"Structural pruning is unsupported for {model.arch}")


def _new_conv_with_selected_outputs(
    source: nn.Conv2d, output_indices: torch.Tensor
) -> nn.Conv2d:
    target = nn.Conv2d(
        in_channels=source.in_channels,
        out_channels=len(output_indices),
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight[output_indices])
        if source.bias is not None:
            target.bias.copy_(source.bias[output_indices])
    target.weight.requires_grad = source.weight.requires_grad
    if target.bias is not None and source.bias is not None:
        target.bias.requires_grad = source.bias.requires_grad
    return target


def _new_batch_norm(source: nn.BatchNorm2d, indices: torch.Tensor) -> nn.BatchNorm2d:
    target = nn.BatchNorm2d(
        len(indices),
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        if source.affine:
            target.weight.copy_(source.weight[indices])
            target.bias.copy_(source.bias[indices])
        if source.track_running_stats:
            target.running_mean.copy_(source.running_mean[indices])
            target.running_var.copy_(source.running_var[indices])
            target.num_batches_tracked.copy_(source.num_batches_tracked)
    if source.affine:
        target.weight.requires_grad = source.weight.requires_grad
        target.bias.requires_grad = source.bias.requires_grad
    return target


def _new_pool_conv(source: nn.Conv2d, input_indices: torch.Tensor) -> nn.Conv2d:
    target = nn.Conv2d(
        in_channels=len(input_indices),
        out_channels=source.out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight[:, input_indices])
        if source.bias is not None:
            target.bias.copy_(source.bias)
    target.weight.requires_grad = source.weight.requires_grad
    if target.bias is not None and source.bias is not None:
        target.bias.requires_grad = source.bias.requires_grad
    return target


def _new_depthwise_conv(source: nn.Conv2d, indices: torch.Tensor) -> nn.Conv2d:
    target = nn.Conv2d(
        in_channels=len(indices),
        out_channels=len(indices),
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=len(indices),
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight[indices])
        if source.bias is not None:
            target.bias.copy_(source.bias[indices])
    target.weight.requires_grad = source.weight.requires_grad
    if target.bias is not None and source.bias is not None:
        target.bias.requires_grad = source.bias.requires_grad
    return target


def _new_conv_with_selected_inputs(
    source: nn.Conv2d, input_indices: torch.Tensor
) -> nn.Conv2d:
    target = nn.Conv2d(
        in_channels=len(input_indices),
        out_channels=source.out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight[:, input_indices])
        if source.bias is not None:
            target.bias.copy_(source.bias)
    target.weight.requires_grad = source.weight.requires_grad
    if target.bias is not None and source.bias is not None:
        target.bias.requires_grad = source.bias.requires_grad
    return target


def apply_mobilenet_v2_reduction(
    model: PlaceRecognitionModel, kept_indices: Iterable[int] | torch.Tensor | None = None
) -> tuple[PlaceRecognitionModel, list[int]]:
    projection, _ = _last_projection(model)
    if kept_indices is None:
        norms = projection.weight.detach().flatten(1).norm(p=2, dim=1)
        indices = torch.where(norms != 0)[0]
    else:
        indices = torch.as_tensor(
            list(kept_indices) if not isinstance(kept_indices, torch.Tensor) else kept_indices,
            dtype=torch.long,
            device=projection.weight.device,
        )
    if indices.numel() == 0:
        raise ValueError("Reduction cannot remove every output channel")

    reduced = copy.deepcopy(model)
    source_projection, source_bn = _last_projection(model)
    reduced_block = reduced.encoder[-1]
    reduced_block.conv[-2] = _new_conv_with_selected_outputs(source_projection, indices)
    reduced_block.conv[-1] = _new_batch_norm(source_bn, indices)

    if not isinstance(model.pool, NetVLAD) or not isinstance(reduced.pool, NetVLAD):
        raise ValueError("MobileNetV2 reduction requires NetVLAD pooling")
    reduced.pool.conv = _new_pool_conv(model.pool.conv, indices)
    reduced.pool.centroids = nn.Parameter(model.pool.centroids[:, indices].detach().clone())
    reduced.pool.dim = len(indices)
    reduced.encoder_dim = len(indices)
    return reduced, indices.detach().cpu().tolist()


def apply_mobilenet_v3_large_reduction(
    model: PlaceRecognitionModel,
    kept_indices: Iterable[int] | torch.Tensor | None = None,
) -> tuple[PlaceRecognitionModel, list[int]]:
    (
        expansion_conv,
        _,
        _,
        _,
        _,
        _,
    ) = _mobilenet_v3_large_last_expansion(model)
    if kept_indices is None:
        norms = expansion_conv.weight.detach().flatten(1).norm(p=2, dim=1)
        indices = torch.where(norms != 0)[0]
    else:
        indices = torch.as_tensor(
            list(kept_indices)
            if not isinstance(kept_indices, torch.Tensor)
            else kept_indices,
            dtype=torch.long,
            device=expansion_conv.weight.device,
        )
    if indices.numel() == 0:
        raise ValueError("Reduction cannot remove every expansion channel")

    reduced = copy.deepcopy(model)
    (
        source_expansion_conv,
        source_expansion_bn,
        source_depthwise_conv,
        source_depthwise_bn,
        source_se,
        source_projection_conv,
    ) = _mobilenet_v3_large_last_expansion(model)
    reduced_block = reduced.encoder[-1].block
    reduced_block[0][0] = _new_conv_with_selected_outputs(
        source_expansion_conv, indices
    )
    reduced_block[0][1] = _new_batch_norm(source_expansion_bn, indices)
    reduced_block[1][0] = _new_depthwise_conv(source_depthwise_conv, indices)
    reduced_block[1][1] = _new_batch_norm(source_depthwise_bn, indices)
    reduced_block[2].fc1 = _new_conv_with_selected_inputs(source_se.fc1, indices)
    reduced_block[2].fc2 = _new_conv_with_selected_outputs(source_se.fc2, indices)
    reduced_block[3][0] = _new_conv_with_selected_inputs(
        source_projection_conv, indices
    )
    return reduced, indices.detach().cpu().tolist()


def apply_structural_reduction(
    model: PlaceRecognitionModel,
    kept_indices: Iterable[int] | torch.Tensor | None = None,
) -> tuple[PlaceRecognitionModel, list[int]]:
    if model.arch == "mobilenetv2":
        return apply_mobilenet_v2_reduction(model, kept_indices)
    if model.arch == "mobilenetv3-large":
        return apply_mobilenet_v3_large_reduction(model, kept_indices)
    raise ValueError(f"Structural reduction is unsupported for {model.arch}")
