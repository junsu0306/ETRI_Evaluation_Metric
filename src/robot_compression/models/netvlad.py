from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class L2Norm(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value, p=2, dim=self.dim)


class NetVLAD(nn.Module):
    def __init__(
        self,
        num_clusters: int = 64,
        dim: int = 128,
        normalize_input: bool = True,
        vladv2: bool = False,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clusters: np.ndarray, descriptors: np.ndarray) -> None:
        clusters = clusters.astype(np.float32, copy=False)
        descriptors = descriptors.astype(np.float32, copy=False)
        if not self.vladv2:
            normalized = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
            dots = np.dot(normalized, descriptors.T)
            dots.sort(axis=0)
            dots = dots[::-1]
            self.alpha = float(-np.log(0.01) / np.mean(dots[0] - dots[1]))
            self.centroids = nn.Parameter(torch.from_numpy(clusters))
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * normalized).unsqueeze(2).unsqueeze(3)
            )
            self.conv.bias = None
            return

        knn = NearestNeighbors(n_jobs=-1).fit(descriptors)
        distances = np.square(knn.kneighbors(clusters, 2)[0])
        self.alpha = float(-np.log(0.01) / np.mean(distances[:, 1] - distances[:, 0]))
        self.centroids = nn.Parameter(torch.from_numpy(clusters))
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def get_soft_assign_logit(self, value: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            value = F.normalize(value, p=2, dim=1)
        return self.conv(value).view(value.shape[0], self.num_clusters, -1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels = value.shape[:2]
        if channels != self.dim:
            raise ValueError(f"NetVLAD expected {self.dim} channels, received {channels}")
        if self.normalize_input:
            value = F.normalize(value, p=2, dim=1)

        soft_assign = F.softmax(
            self.conv(value).view(batch, self.num_clusters, -1), dim=1
        )
        flattened = value.view(batch, channels, -1)
        vlad = torch.zeros(
            batch,
            self.num_clusters,
            channels,
            dtype=value.dtype,
            device=value.device,
        )
        for cluster_index in range(self.num_clusters):
            residual = flattened - self.centroids[cluster_index].view(1, -1, 1)
            residual = residual * soft_assign[:, cluster_index : cluster_index + 1]
            vlad[:, cluster_index] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = F.normalize(vlad.reshape(batch, -1), p=2, dim=1)
        return vlad
