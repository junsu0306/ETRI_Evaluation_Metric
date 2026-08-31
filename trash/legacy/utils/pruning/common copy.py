
import torch
import torch.nn as nn
import copy
import numpy as np


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
        filter_norms = torch.norm(layer.weight.view(layer.weight.shape[0], -1), dim=1)


        max_idx = torch.argmax(filter_norms)
        filter_norms[max_idx] = inf

        return filter_norms

def get_pruning_indices(filter_norms, sparsity):

    all_norms = torch.cat(filter_norms)
    num_pruning_filters = int(all_norms.numel() * sparsity) 


    _, global_pruning_idx = torch.topk(all_norms, num_pruning_filters, largest=False)

    pruning_indices = []
    current_position = 0 


    for norms in filter_norms:
        layer_size = norms.numel()
        pruning_idx = global_pruning_idx[(global_pruning_idx >= current_position) & (global_pruning_idx < current_position + layer_size)]

        pruning_idx -= current_position

        pruning_indices.append(pruning_idx)
        current_position += layer_size 

    return pruning_indices


def mobilenetv2_layerwise_pruning(model, sparsity):

    
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
    
    collect_layers(model)

    # Conv 레이어별로 필터 norm 계산
    filter_norms = []
    conv_info = []
    
    for name, conv_layer in conv_layers:
        if conv_layer.groups == conv_layer.in_channels and conv_layer.groups > 1:
            norms = get_filter_norms(conv_layer)
            filter_norms.append(norms)
            conv_info.append((name, conv_layer, 'depthwise'))

        else:
            # Standard convolution (pointwise 포함)
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
                        bn_found = True
                        break
            
   
    
    total_filters = sum(len(norms) for norms in filter_norms)
    pruning_ratio = pruned_filters_total / total_filters if total_filters > 0 else 0 #for debug

    return model




def mobilenetv2_L_layer_pruning(model, sparsity):

    
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


    
    # Conv 레이어별로 필터 norm 계산
    filter_norms = []
    conv_info = []
    
    for name, conv_layer in conv_layers:
        if conv_layer.groups == conv_layer.in_channels and conv_layer.groups > 1:
            conv_info.append((name, conv_layer, 'depthwise')) # no pruning for depthwise convolution
        else:
            # Standard convolution (pointwise 포함)
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
    pruning_ratio = pruned_filters_total / total_filters if total_filters > 0 else 0 #for debug
    #print(f"Total filters: {total_filters}, Pruned filters: {pruned_filters_total}, Pruning ratio: {pruning_ratio:.2%}")


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

    orig_w = layer.weight
    orig_b = layer.bias
    dev, dt = orig_w.device, orig_w.dtype
    k0, k1 = layer.kernel_size

    if layer.groups > 1:
        # --- Depthwise convolution 처리 ---
        # For depthwise conv, input and output channels must be the same
        # Use intersection of survived_in and survived_out indices
        if len(survived_in_channels_idx) != len(survived_out_channels_idx):
            # If sizes don't match, use the intersection
            survived_idx = survived_in_channels_idx if len(survived_in_channels_idx) < len(survived_out_channels_idx) else survived_out_channels_idx
        else:
            survived_idx = survived_out_channels_idx
            
        new_C = len(survived_idx)

        reduced_layer.in_channels  = new_C
        reduced_layer.out_channels = new_C
        reduced_layer.groups       = new_C

        new_w = torch.zeros(new_C, 1, k0, k1, device=dev, dtype=dt)
        with torch.no_grad():
            # Make sure we're indexing correctly
            new_w.copy_(orig_w[survived_idx, :, :, :])
        reduced_layer.weight = torch.nn.Parameter(new_w, requires_grad=True)

        # bias가 있으면 동일하게 복사
        if orig_b is not None:
            new_b = torch.zeros(new_C, device=dev, dtype=dt)
            with torch.no_grad():
                new_b.copy_(orig_b[survived_idx])
            reduced_layer.bias = torch.nn.Parameter(new_b, requires_grad=True)

    else:
        # standard convolution reducing
        new_out = len(survived_out_channels_idx)
        new_in  = len(survived_in_channels_idx)
        reduced_layer.in_channels  = new_in
        reduced_layer.out_channels = new_out
        reduced_layer.groups       = 1

        # [new_out, new_in, k, k] 형태로 new_w 생성
        new_w = torch.zeros(new_out, new_in, k0, k1, device=dev, dtype=dt)
        with torch.no_grad():
            tmp = orig_w[survived_out_channels_idx, :, :, :]
            new_w.copy_(tmp[:, survived_in_channels_idx, :, :])
        reduced_layer.weight = torch.nn.Parameter(new_w, requires_grad=True)

        if orig_b is not None:
            new_b = torch.zeros(new_out, device=dev, dtype=dt)
            with torch.no_grad():
                new_b.copy_(orig_b[survived_out_channels_idx])
            reduced_layer.bias = torch.nn.Parameter(new_b, requires_grad=True)


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







def mobilenetv2_L_layer_reduction(model):
    device = next(model.parameters()).device
    reduced_model = copy.deepcopy(model).to(device)

    last_layer = model.encoder[-1]
    reduced_last_layer = reduced_model.encoder[-1]

    def collect_conv_bn_pairs(module, prefix=""):
        pairs = []
        children = list(module.named_children())
        for i, (name, mod) in enumerate(children):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(mod, nn.Conv2d) and i + 1 < len(children):
                next_name, next_mod = children[i+1]
                full_next = f"{prefix}.{next_name}" if prefix else next_name
                if isinstance(next_mod, nn.BatchNorm2d):
                    pairs.append(((full_name, mod), (full_next, next_mod)))
            pairs.extend(collect_conv_bn_pairs(mod, full_name))
        return pairs

    # Conv-BN 쌍 수집
    target_pairs = collect_conv_bn_pairs(last_layer)
    reduced_pairs = collect_conv_bn_pairs(reduced_last_layer)

    prev_survived_idx = None

    for idx in range(len(target_pairs)):
        # Unpack original and reduced pairs by index
        (orig_conv_name, orig_conv), (orig_bn_name, orig_bn) = target_pairs[idx]
        (red_conv_name, red_conv), (red_bn_name, red_bn) = reduced_pairs[idx]
      
        # Check if this is a depthwise convolution
        is_depthwise = orig_conv.groups == orig_conv.in_channels and orig_conv.groups > 1
        
        if is_depthwise:
      
            if prev_survived_idx is not None:
                # Use previous layer's output as both input and output indices
                survived_out_idx = prev_survived_idx
                survived_in_idx = prev_survived_idx
            else:
                # First layer case - get survived filters
                survived_out_idx = get_survived_filter_idx(orig_conv)
                survived_in_idx = survived_out_idx
        else:
            # Standard convolution
            survived_out_idx = get_survived_filter_idx(orig_conv)
            
            if prev_survived_idx is None:
                # First layer - use all input channels
                survived_in_idx = torch.arange(orig_conv.in_channels, device=device)
            else:
                # Use previous layer's output as input
                survived_in_idx = prev_survived_idx

        # Apply reduction
        conv_reduce(orig_conv, red_conv, survived_out_idx, survived_in_idx)
        bn_reduce(orig_bn, red_bn, survived_out_idx)
        
        # Update prev_survived_idx for next iteration
        prev_survived_idx = survived_out_idx

    return reduced_model


# def mobilenetv2_L_layer_reduction(model):


#     device = next(model.parameters()).device
#     reduced_model = copy.deepcopy(model).to(device)

#     last_layer = model.encoder[-1]
#     reduced_last_layer = reduced_model.encoder[-1]

#     def collect_conv_bn_pairs(module, prefix=""):
#         pairs = []
#         children = list(module.named_children())
#         for i, (name, mod) in enumerate(children):
#             full_name = f"{prefix}.{name}" if prefix else name
#             if isinstance(mod, nn.Conv2d) and i + 1 < len(children):
#                 next_name, next_mod = children[i+1]
#                 full_next = f"{prefix}.{next_name}" if prefix else next_name
#                 if isinstance(next_mod, nn.BatchNorm2d):
#                     pairs.append(((full_name, mod), (full_next, next_mod)))
#             pairs.extend(collect_conv_bn_pairs(mod, full_name))
#         return pairs

#     target_pairs = collect_conv_bn_pairs(last_layer)
#     reduced_pairs = collect_conv_bn_pairs(reduced_last_layer)

#     prev_survived_idx = None

#     for idx in range(len(target_pairs)):
#         # Unpack original and reduced pairs by index
#         (orig_conv_name, orig_conv), (orig_bn_name, orig_bn) = target_pairs[idx]
#         (red_conv_name, red_conv), (red_bn_name, red_bn) = reduced_pairs[idx]


#         if orig_conv.groups == orig_conv.in_channels and orig_conv.groups > 1:
#             continue  # Skip depthwise convolutions, beacuse we already handle them in previous iteration

#         # Compute surviving output channels for the current conv
#         survived_out_idx = get_survived_filter_idx(orig_conv)

#         # Determine input indices
#         if prev_survived_idx is None:
#             survived_in_idx = torch.arange(orig_conv.in_channels, device=device)
#         else:
#             survived_in_idx = prev_survived_idx

#         # Handle depthwise branching: if next is depthwise, merge indices first
#         if idx + 1 < len(target_pairs):
#             (next_conv_name, next_conv), (next_bn_name, next_bn) = target_pairs[idx + 1]
#             if next_conv.groups > 1:
#                 # Find reduced counterparts for next conv
#                 (next_red_conv_name, next_red_conv), (_, next_red_bn) = reduced_pairs[idx + 1]
#                 next_survived_idx = get_survived_filter_idx(next_conv)


#                 intersect_idx = torch.from_numpy(
#                     np.intersect1d(survived_out_idx.cpu().numpy(), next_survived_idx.cpu().numpy())
#                 ).to(device)

#                 survived_out_idx = intersect_idx


                
#                 # Apply reduction on both layers
#                 conv_reduce(orig_conv, red_conv, survived_out_idx, survived_in_idx)
#                 bn_reduce(orig_bn, red_bn, survived_out_idx)

#                 survived_in_idx = survived_out_idx # for depthwise convolution 
     
#                 conv_reduce(next_conv, next_red_conv, survived_out_idx, survived_in_idx)
#                 bn_reduce(next_bn, next_red_bn, survived_out_idx)
#                 # Update prev_survived for following layers
#                 prev_survived_idx = survived_out_idx
#                 continue

#         # Standard reduction for this layer
#         conv_reduce(orig_conv, red_conv, survived_out_idx, survived_in_idx)
#         bn_reduce(orig_bn, red_bn, survived_out_idx)

#         # Update for next iteration
#         prev_survived_idx = survived_out_idx

#     return reduced_model

