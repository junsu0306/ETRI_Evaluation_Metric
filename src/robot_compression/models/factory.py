from __future__ import annotations

from pathlib import Path

import h5py
import torch.nn as nn
from torchvision import models

from .netvlad import L2Norm, NetVLAD


class PlaceRecognitionModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        pool: nn.Module | None,
        arch: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.arch = arch
        if pool is not None:
            self.pool = pool

    def forward(self, image):
        encoded = self.encoder(image)
        return self.pool(encoded)


def _weights(enabled: bool, enum):
    return enum.DEFAULT if enabled else None


def build_encoder(
    arch: str,
    pretrained: bool = True,
    cluster_mode: bool = False,
    freeze_backbone: bool | None = None,
):
    normalized = arch.lower().replace("_", "-")
    if normalized == "alexnet":
        source = models.alexnet(weights=_weights(pretrained, models.AlexNet_Weights))
        layers = list(source.features.children())[:-2]
        encoder_dim = 256
        frozen = layers[:-1]
    elif normalized == "vgg16":
        source = models.vgg16(weights=_weights(pretrained, models.VGG16_Weights))
        layers = list(source.features.children())[:-2]
        encoder_dim = 512
        frozen = layers[:-5]
    elif normalized in {"resnet18", "resnet-18"}:
        source = models.resnet18(weights=_weights(pretrained, models.ResNet18_Weights))
        layers = list(source.children())[:-3]
        encoder_dim = 256
        frozen = layers[:-1]
        normalized = "resnet18"
    elif normalized in {"mobilenetv2", "mobilenet-v2"}:
        # The legacy experiments used torchvision's old ``pretrained=True``
        # behavior, which maps to IMAGENET1K_V1.  New torchvision releases use
        # IMAGENET1K_V2 for DEFAULT, so selecting V1 explicitly is required for
        # a meaningful reproduction of the recorded Pittsburgh results.
        mobilenet_weights = (
            models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        source = models.mobilenet_v2(
            weights=mobilenet_weights
        )
        layers = list(source.features.children())[:-1]
        encoder_dim = 320
        frozen = layers[:-1]
        normalized = "mobilenetv2"
    elif normalized in {"mobilenetv3-large", "mobilenet-v3-large"}:
        # Match the old torchvision ``pretrained=True`` initialization used by
        # the imported experiments.  The final 160->960 convolution and the
        # classifier are omitted, leaving the last residual block's 160-channel
        # output for NetVLAD.
        source = models.mobilenet_v3_large(
            weights=(
                models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
        )
        layers = list(source.features.children())[:-1]
        encoder_dim = 160
        frozen = layers[:-1]
        normalized = "mobilenetv3-large"
    elif normalized in {"mobilenetv3", "mobilenetv3-small", "mobilenet-v3-small"}:
        source = models.mobilenet_v3_small(
            weights=_weights(pretrained, models.MobileNet_V3_Small_Weights)
        )
        layers = list(source.features.children())[:-1]
        encoder_dim = 96
        frozen = layers[:-1]
        normalized = "mobilenetv3-small"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if freeze_backbone is None:
        freeze_backbone = pretrained
    if freeze_backbone:
        for layer in frozen:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    if cluster_mode:
        layers.append(L2Norm())
    return nn.Sequential(*layers), encoder_dim, normalized


def build_model(
    arch: str,
    num_clusters: int = 64,
    pooling: str = "netvlad",
    pretrained: bool = True,
    cluster_mode: bool = False,
    vladv2: bool = False,
    freeze_backbone: bool | None = None,
) -> PlaceRecognitionModel:
    encoder, encoder_dim, normalized_arch = build_encoder(
        arch,
        pretrained=pretrained,
        cluster_mode=cluster_mode,
        freeze_backbone=freeze_backbone,
    )
    if cluster_mode:
        pool = None
    elif pooling == "netvlad":
        pool = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
    elif pooling == "max":
        pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(), L2Norm())
    elif pooling == "avg":
        pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), L2Norm())
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    return PlaceRecognitionModel(encoder, encoder_dim, pool, normalized_arch)


def load_centroids(model: PlaceRecognitionModel, path: Path) -> None:
    if not isinstance(model.pool, NetVLAD):
        return
    with h5py.File(path, "r") as handle:
        model.pool.init_params(handle["centroids"][...], handle["descriptors"][...])
