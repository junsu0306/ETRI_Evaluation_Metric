
import torch
import torch.nn as nn
import copy
import numpy as np
import h5py
import torch.nn.functional as F


from math import ceil
import faiss
import h5py
from torch.utils.data import DataLoader, SubsetRandomSampler


def filter_pruning(layer, pruning_idx):
    with torch.no_grad():
        pruning_idx = pruning_idx[pruning_idx < layer.weight.shape[0]]
        if pruning_idx.numel() == 0:
            return
        layer.weight[pruning_idx, :, :, :] = 0.0


def bn_pruning(layer, pruning_idx):
    weight = layer.weight
    bias = layer.bias
    mean = layer.running_mean
    var = layer.running_var

    pruning_idx = pruning_idx[pruning_idx < weight.shape[0]]

    if pruning_idx.numel() == 0:
        return

    with torch.no_grad():
        weight[pruning_idx] = 0.0
        bias[pruning_idx] = 0.0
        mean[pruning_idx] = 0.0
        var[pruning_idx] = 1.0


def get_filter_norms(layer, inf=99999):
    with torch.no_grad():
        filter_norms = torch.norm(layer.weight.view(
            layer.weight.shape[0], -1), dim=1)

        max_idx = torch.argmax(filter_norms)
        filter_norms[max_idx] = inf

        return filter_norms


def get_pruning_indices(filter_norms, sparsity):

    all_norms = torch.cat(filter_norms)
    num_pruning_filters = int(all_norms.numel() * sparsity)

    _, global_pruning_idx = torch.topk(
        all_norms, num_pruning_filters, largest=False)

    pruning_indices = []
    current_position = 0

    for norms in filter_norms:
        layer_size = norms.numel()
        pruning_idx = global_pruning_idx[(global_pruning_idx >= current_position) & (
            global_pruning_idx < current_position + layer_size)]

        pruning_idx -= current_position

        pruning_indices.append(pruning_idx)
        current_position += layer_size

    return pruning_indices


def mobilenetv2_L_layer_pair_pruning(conv_layer, bn_layer, sparsity):
    assert conv_layer.groups == 1, "it using for only standard convolution"
    assert conv_layer.out_channels == bn_layer.num_features, "out_channels of conv and num_features of bn should be same"

    filter_norms = []
    norms = get_filter_norms(conv_layer)
    filter_norms.append(norms)
    pruning_indices = get_pruning_indices(filter_norms, sparsity)
    filter_pruning(conv_layer, pruning_indices[0])
    bn_pruning(bn_layer, pruning_indices[0])


def mobilenetv2_L_block_pruning(model, sparsity):

    # Conv2d 레이어들을 수집 (MobileNetV2의 구조를 고려)
    conv_layers = []
    bn_layers = []

    def collect_layers(module, name=""):
        for child_name, child_module in module.named_children():
            current_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child_module, nn.Conv2d):
                conv_layers.append((current_name, child_module))
            elif isinstance(child_module, nn.BatchNorm2d):
                bn_layers.append((current_name, child_module))
            else:
                collect_layers(child_module, current_name)

    collect_layers(model.encoder[-1])

    filter_norms = []
    conv_info = []

    for name, conv_layer in conv_layers:
        if conv_layer.groups == conv_layer.in_channels and conv_layer.groups > 1:
            # no pruning for depthwise convolution
            conv_info.append((name, conv_layer, 'depthwise'))
        else:
            norms = get_filter_norms(conv_layer)
            filter_norms.append(norms)
            conv_info.append((name, conv_layer, 'standard'))

    # 글로벌 프루닝 인덱스 계산
    pruning_indices = get_pruning_indices(filter_norms, sparsity)

    # 레이어별 프루닝 실행
    pruned_filters_total = 0

    for i, (pruning_idx, (conv_name, conv_layer, conv_type)) in enumerate(zip(pruning_indices, conv_info)):
        if len(pruning_idx) > 0:

            filter_pruning(conv_layer, pruning_idx)
            pruned_filters_total += len(pruning_idx)
            for bn_name, bn_layer in bn_layers:
                # Conv와 BatchNorm이 같은 부모 모듈에 있는지 확인
                conv_parent = '.'.join(conv_name.split('.')[:-1])
                bn_parent = '.'.join(bn_name.split('.')[:-1])

                if conv_parent == bn_parent:
                    # 같은 부모 모듈 내에서 Conv 다음에 오는 BatchNorm인지 확인
                    if 'bn' in bn_name or 'norm' in bn_name:
                        bn_pruning(bn_layer, pruning_idx)
                        break

    total_filters = sum(len(norms) for norms in filter_norms)
    pruning_ratio = pruned_filters_total / \
        total_filters if total_filters > 0 else 0  # for debug

    return model


def get_survived_filter_idx(layer):
    weight = layer.weight
    num_filters = weight.shape[0]
    filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
    survived_filter_idx = torch.where(filter_norms != 0)[0]
    return survived_filter_idx


def conv_reduce(
    layer, reduced_layer, survived_out_channels_idx, survived_in_channels_idx
):

    # set reduced_layer's in_channels and out_channels
    reduced_layer.in_channels = len(survived_in_channels_idx)
    reduced_layer.out_channels = len(survived_out_channels_idx)

    # in case of depthwise conv, the number of groups and out_channels have to be same
    if reduced_layer.groups != 1:
        reduced_layer.groups = reduced_layer.out_channels

    # re-defining the weight parameter of reduced_layer
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_channels,
            reduced_layer.in_channels,
            reduced_layer.kernel_size[0],
            reduced_layer.kernel_size[1],
        ),
        requires_grad=True,
    )

    weight = layer.weight
    reduced_weight = reduced_layer.weight

    assert (
        len(survived_out_channels_idx) == reduced_weight.shape[0]
    ), "reduced_num_filters != survived_idx {}, {}".format(
        len(survived_out_channels_idx), reduced_weight.shape[0]
    )

    assert (
        len(survived_in_channels_idx) == reduced_weight.shape[1]
    ), "reduced_num_channel != survived_idx {}, {}".format(
        len(survived_in_channels_idx), reduced_weight.shape[1]
    )

    # copy the survived weights to the reduced layer
    with torch.no_grad():
        reduced_weight.copy_(
            weight[survived_out_channels_idx, :, :, :][
                :, survived_in_channels_idx, :, :
            ]
        )


def conv_reduce_depthwise(layer, reduced_layer, survived_out_idx):

    assert layer.groups == layer.out_channels == layer.in_channels, "depthwise convolution should have in_channels == out_channels == groups"

    new_channel = len(survived_out_idx)

    reduced_layer.out_channels = new_channel
    reduced_layer.in_channels = new_channel
    reduced_layer.groups = new_channel

    kernel_size = layer.kernel_size

    device = layer.weight.device
    dtype = layer.weight.dtype

    new_w = torch.zeros(
        (new_channel, 1, kernel_size[0], kernel_size[1]), dtype=dtype, device=device)

    with torch.no_grad():
        new_w.copy_(layer.weight[survived_out_idx, :, :, :])
    reduced_layer.weight = nn.Parameter(new_w, requires_grad=True)

    if layer.bias is not None:
        new_b = torch.zeros(
            (new_channel,), dtype=layer.bias.dtype, device=layer.bias.device)
        with torch.no_grad():
            new_b.copy_(layer.bias[survived_out_idx])
        reduced_layer.bias = nn.Parameter(new_b, requires_grad=True)
    else:
        reduced_layer.bias = None


def bn_reduce(layer, reduced_layer, survived_features_idx):
    reduced_layer.num_features = len(survived_features_idx)

    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.num_features,
        ),
        requires_grad=True,
    )
    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.num_features,
        ),
        requires_grad=True,
    )
    reduced_layer.running_mean = torch.zeros(
        reduced_layer.num_features,
    )
    reduced_layer.running_var = torch.zeros(
        reduced_layer.num_features,
    )

    weight = layer.weight
    reduced_weight = reduced_layer.weight
    bias = layer.bias
    reduced_bias = reduced_layer.bias
    mean = layer.running_mean
    reduced_mean = reduced_layer.running_mean
    var = layer.running_var
    reduced_var = reduced_layer.running_var

    assert (
        len(survived_features_idx) == reduced_weight.shape[0]
    ), "reduced_bn_weight != survived_idx"
    assert (
        len(survived_features_idx) == reduced_bias.shape[0]
    ), "reduced_bn_bias != survived_idx"
    assert (
        len(survived_features_idx) == reduced_mean.shape[0]
    ), "reduced_bn_mean != survived_idx"
    assert (
        len(survived_features_idx) == reduced_var.shape[0]
    ), "reduced_bn_var != survived_idx"
    with torch.no_grad():
        reduced_weight.copy_(weight[survived_features_idx])
        reduced_bias.copy_(bias[survived_features_idx])
        reduced_mean.copy_(mean[survived_features_idx])
        reduced_var.copy_(var[survived_features_idx])


# def mobilenetv2_L_block_reduction(model):
#     """
#     encoder[-1] InvertedResidual 블록만 리듀스:
#       - 확장 1x1: out 채널 슬림 + BN 슬림
#       - DW 3x3: 확장과 동일 인덱스로 in/out/groups 슬림 + BN 슬림
#       - 투영 1x1: in 채널만 슬림 (out=320 유지), 뒤 BN은 유지
#     """
#     device = next(model.parameters()).device
#     reduced_model = copy.deepcopy(model).to(device)

#     last_block = model.encoder[-1]
#     reduced_last_block = reduced_model.encoder[-1]

#     # 예상 구조 이름(질문에 준함)
#     paths = {
#         "exp_conv": "conv.0.0",  # 160 -> 960
#         "exp_bn":   "conv.0.1",
#         "dw_conv":  "conv.1.0",  # 960 depthwise
#         "dw_bn":    "conv.1.1",
#         "proj_conv":"conv.2",    # 960 -> 320
#         "proj_bn":  "conv.3",
#     }

#     # 모듈 핸들
#     exp_conv      = last_block.get_submodule(paths["exp_conv"])
#     exp_bn        = last_block.get_submodule(paths["exp_bn"])
#     dw_conv       = last_block.get_submodule(paths["dw_conv"])
#     dw_bn         = last_block.get_submodule(paths["dw_bn"])
#     proj_conv     = last_block.get_submodule(paths["proj_conv"])
#     # proj_bn     = last_block.get_submodule(paths["proj_bn"])  # 그대로 유지

#     r_exp_conv   = reduced_last_block.get_submodule(paths["exp_conv"])
#     r_exp_bn     = reduced_last_block.get_submodule(paths["exp_bn"])
#     r_dw_conv    = reduced_last_block.get_submodule(paths["dw_conv"])
#     r_dw_bn      = reduced_last_block.get_submodule(paths["dw_bn"])
#     r_proj_conv  = reduced_last_block.get_submodule(paths["proj_conv"])
#     # r_proj_bn   = reduced_last_block.get_submodule(paths["proj_bn"])

#     # 1) 확장 1x1: 살아남은 out 인덱스 수집
#     survived_out_idx = get_survived_filter_idx(exp_conv)
#     survived_in_idx  = torch.arange(exp_conv.in_channels, device=device)

#     # 확장 1x1 슬림 + BN 슬림
#     conv_reduce(exp_conv, r_exp_conv, survived_out_idx, survived_in_idx)
#     bn_reduce(exp_bn, r_exp_bn, survived_out_idx)

#     # 2) DW 3x3: 확장과 동일 인덱스로 슬림 + BN 슬림
#     conv_reduce_depthwise(dw_conv, r_dw_conv, survived_out_idx)
#     bn_reduce(dw_bn, r_dw_bn, survived_out_idx)

#     survived_in_idx = survived_out_idx
#     survived_out_idx = get_survived_filter_idx(proj_conv)
#     conv_reduce(proj_conv, r_proj_conv, survived_out_idx, survived_in_idx)

#     return reduced_model


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def mobilenetv2_L_block_reduction(model):
    device = next(model.parameters()).device
    reduced_model = copy.deepcopy(model).to(device)

    last_block = model.encoder[-1]

    reduced_last_block = reduced_model.encoder[-1]
    layers = [(name, module) for name, module in last_block.named_modules(
    ) if name != "" and len(list(module.children())) == 0]
    layers.append(("pool_conv", model.pool.conv))

    prev_survived_idx = None
    survived_in_idx = None
    survived_out_idx = None
    survived_pool_idx = None
    for name, module in layers:

        if name == "pool_conv":
            survived_pool_idx = survived_out_idx
            reduced_module = reduced_model.pool.conv
        else:
            reduced_module = reduced_last_block.get_submodule(name)

        if isinstance(module, nn.Conv2d):
            survived_out_idx = get_survived_filter_idx(module)

            if prev_survived_idx is None:
                survived_in_idx = torch.arange(
                    module.in_channels, device=device)
            else:
                survived_in_idx = prev_survived_idx

            if module.groups == module.in_channels and module.groups > 1:
                survived_out_idx = survived_in_idx
                conv_reduce_depthwise(module, reduced_module, survived_out_idx)
            else:
                conv_reduce(module, reduced_module,
                            survived_out_idx, survived_in_idx)
            prev_survived_idx = survived_out_idx

        elif isinstance(module, nn.BatchNorm2d):
            bn_reduce(module, reduced_module, survived_out_idx)

    reduced_model.pool.dim = reduced_model.pool.conv.out_channels
    reduced_model.encoder_dim = reduced_model.pool.conv.in_channels

    cluster_modify(reduced_model, survived_pool_idx)

    return reduced_model


def cluster_modify(reduced_model, survived_out_idx, num_clusters=64, device="cuda:0"):

    centroids = reduced_model.pool.centroids
    centroids = centroids.to(device)
    reduced_centroids = centroids[:, survived_out_idx]

    reduced_model.pool.centroids = nn.Parameter(reduced_centroids)

    new_dim = len(survived_out_idx)
    reduced_model.pool.dim = new_dim
