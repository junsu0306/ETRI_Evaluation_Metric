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
