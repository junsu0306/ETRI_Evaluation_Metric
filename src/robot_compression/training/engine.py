from __future__ import annotations

from math import ceil
from pathlib import Path

import faiss
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm.auto import tqdm

from robot_compression.compression import prune_mobilenet_v2_last_output
from robot_compression.data.pittsburgh import collate_fn
from robot_compression.models.factory import PlaceRecognitionModel
from robot_compression.models.netvlad import NetVLAD


def _pool_size(model: PlaceRecognitionModel) -> int:
    if isinstance(getattr(model, "pool", None), NetVLAD):
        return model.encoder_dim * model.pool.num_clusters
    return model.encoder_dim


def extract_features(
    model: PlaceRecognitionModel,
    dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    features = np.empty((len(dataset), _pool_size(model)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="features", unit="batch"):
            output = model(images.to(device, non_blocking=True))
            features[indices.numpy()] = output.cpu().numpy()
    return features


def evaluate(
    model: PlaceRecognitionModel,
    dataset,
    device: torch.device,
    batch_size: int = 16,
    workers: int = 4,
) -> dict[int, float]:
    features = extract_features(model, dataset, device, batch_size, workers)
    query_features = features[dataset.dbStruct.numDb :]
    database_features = features[: dataset.dbStruct.numDb]
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(database_features)
    ranks = [1, 5, 10, 20]
    _, predictions = index.search(query_features, max(ranks))
    positives = dataset.getPositives()
    correct = np.zeros(len(ranks))
    for query_index, prediction in enumerate(predictions):
        for rank_index, rank in enumerate(ranks):
            if np.any(np.isin(prediction[:rank], positives[query_index])):
                correct[rank_index:] += 1
                break
    return {
        rank: float(value)
        for rank, value in zip(ranks, correct / dataset.dbStruct.numQ)
    }


def build_clusters(
    model: PlaceRecognitionModel,
    dataset,
    output_path: Path,
    device: torch.device,
    num_clusters: int = 64,
    num_descriptors: int = 50_000,
    descriptors_per_image: int = 100,
    batch_size: int = 16,
    workers: int = 4,
) -> Path:
    image_count = min(len(dataset), ceil(num_descriptors / descriptors_per_image))
    actual_descriptors = image_count * descriptors_per_image
    sampler = SubsetRandomSampler(
        np.random.choice(len(dataset), image_count, replace=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )
    descriptors = np.empty((actual_descriptors, model.encoder_dim), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="cluster descriptors", unit="batch"):
            encoded = model.encoder(images.to(device, non_blocking=True))
            encoded = encoded.flatten(2).permute(0, 2, 1)
            for image_descriptors in encoded:
                count = image_descriptors.shape[0]
                sample = np.random.choice(
                    count, descriptors_per_image, replace=count < descriptors_per_image
                )
                descriptors[cursor : cursor + descriptors_per_image] = (
                    image_descriptors[sample].cpu().numpy()
                )
                cursor += descriptors_per_image

    kmeans = faiss.Kmeans(model.encoder_dim, num_clusters, niter=100, verbose=False)
    kmeans.train(descriptors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("descriptors", data=descriptors)
        handle.create_dataset("centroids", data=kmeans.centroids)
    return output_path


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    student = student_logits.mean(dim=2)
    teacher = teacher_logits.mean(dim=2)
    return F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2


def _build_feature_cache(
    model: PlaceRecognitionModel,
    whole_dataset,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = extract_features(model, whole_dataset, device, batch_size, workers)
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("features", data=features)


def train_one_epoch(
    model: PlaceRecognitionModel,
    teacher: PlaceRecognitionModel | None,
    train_dataset,
    whole_train_dataset,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    cache_path: Path,
    batch_size: int = 1,
    cache_batch_size: int = 16,
    workers: int = 4,
    cache_refresh_rate: int = 1000,
    kd_weight: float = 0.5,
    temperature: float = 1.5,
    pruning: bool = False,
    pruning_ratio: float = 0.15,
) -> float:
    if kd_weight > 0 and teacher is None:
        raise ValueError("A teacher model is required when kd_weight is positive")
    if teacher is not None:
        teacher.eval()
    subset_count = (
        ceil(len(train_dataset) / cache_refresh_rate) if cache_refresh_rate > 0 else 1
    )
    subset_indices = np.array_split(np.arange(len(train_dataset)), subset_count)
    total_loss = 0.0
    batches = 0

    for indices in subset_indices:
        _build_feature_cache(
            model, whole_train_dataset, cache_path, device, cache_batch_size, workers
        )
        train_dataset.cache = str(cache_path)
        loader = DataLoader(
            Subset(train_dataset, indices),
            batch_size=batch_size,
            num_workers=workers,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        model.train()
        for query, positives, negatives, negative_counts, _ in tqdm(
            loader, desc="train", unit="batch"
        ):
            if query is None:
                continue
            query_count = query.shape[0]
            negative_count = int(torch.sum(negative_counts).item())
            images = torch.cat([query, positives, negatives]).to(device)

            student_features = model.encoder(images)
            student_embedding = model.pool(student_features)

            query_embedding, positive_embedding, negative_embedding = torch.split(
                student_embedding, [query_count, query_count, negative_count]
            )
            triplet = torch.zeros((), device=device)
            for query_index, count in enumerate(negative_counts):
                start = int(torch.sum(negative_counts[:query_index]).item())
                for offset in range(int(count.item())):
                    triplet = triplet + criterion(
                        query_embedding[query_index : query_index + 1],
                        positive_embedding[query_index : query_index + 1],
                        negative_embedding[start + offset : start + offset + 1],
                    )
            triplet = triplet / max(negative_count, 1)
            if kd_weight > 0:
                student_logits = model.pool.get_soft_assign_logit(student_features)
                with torch.no_grad():
                    teacher_features = teacher.encoder(images)
                    teacher_logits = teacher.pool.get_soft_assign_logit(teacher_features)
                kd = distillation_loss(student_logits, teacher_logits, temperature)
            else:
                kd = torch.zeros((), device=device)
            loss = (1.0 - kd_weight) * triplet + kd_weight * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if pruning:
                prune_mobilenet_v2_last_output(model, pruning_ratio)
            total_loss += float(loss.item())
            batches += 1
        cache_path.unlink(missing_ok=True)
    return total_loss / max(batches, 1)
