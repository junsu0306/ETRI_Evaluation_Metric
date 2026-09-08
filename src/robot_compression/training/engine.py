from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import faiss
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm.auto import tqdm

from robot_compression.compression import prune_model
from robot_compression.data.pittsburgh import collate_fn
from robot_compression.models.factory import PlaceRecognitionModel
from robot_compression.models.netvlad import NetVLAD


@dataclass(frozen=True)
class EpochLosses:
    total: float
    triplet: float
    kd: float


def set_train_mode_with_frozen_backbone(model: PlaceRecognitionModel) -> None:
    """Enable training while keeping fully frozen encoder stages in eval mode.

    ``requires_grad=False`` does not stop BatchNorm running statistics from
    changing.  Keeping frozen stages in eval mode preserves the ImageNet
    feature distribution used to build the NetVLAD centroids.
    """
    model.train()
    for layer in model.encoder:
        parameters = list(layer.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            layer.eval()


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


def relational_distillation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    query_count: int,
    negative_counts: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Distill each teacher query-to-candidate ranking into the student.

    Teacher and student descriptors may have different dimensions.  The loss
    compares only cosine similarities within each descriptor space, so it does
    not assume that independently learned NetVLAD cluster indices are aligned.
    """
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    if student_embedding.ndim != 2 or teacher_embedding.ndim != 2:
        raise ValueError("Relational KD expects two-dimensional embeddings")
    if student_embedding.shape[0] != teacher_embedding.shape[0]:
        raise ValueError("Teacher and student KD batches must contain the same images")
    if len(negative_counts) != query_count:
        raise ValueError("negative_counts must contain one value per query")

    negative_total = int(torch.sum(negative_counts).item())
    expected_images = 2 * query_count + negative_total
    if student_embedding.shape[0] != expected_images:
        raise ValueError(
            f"Expected {expected_images} KD images, received {student_embedding.shape[0]}"
        )

    student = F.normalize(student_embedding, p=2, dim=1)
    teacher = F.normalize(teacher_embedding.detach(), p=2, dim=1)
    student_q, student_p, student_n = torch.split(
        student, [query_count, query_count, negative_total]
    )
    teacher_q, teacher_p, teacher_n = torch.split(
        teacher, [query_count, query_count, negative_total]
    )

    losses = []
    negative_start = 0
    for query_index, count_tensor in enumerate(negative_counts):
        count = int(count_tensor.item())
        negative_end = negative_start + count
        student_candidates = torch.cat(
            [
                student_p[query_index : query_index + 1],
                student_n[negative_start:negative_end],
            ],
            dim=0,
        )
        teacher_candidates = torch.cat(
            [
                teacher_p[query_index : query_index + 1],
                teacher_n[negative_start:negative_end],
            ],
            dim=0,
        )
        student_scores = (
            student_candidates @ student_q[query_index : query_index + 1].T
        ).squeeze(1) / temperature
        teacher_scores = (
            teacher_candidates @ teacher_q[query_index : query_index + 1].T
        ).squeeze(1) / temperature
        losses.append(
            F.kl_div(
                F.log_softmax(student_scores, dim=0),
                F.softmax(teacher_scores, dim=0),
                reduction="sum",
            )
        )
        negative_start = negative_end

    if not losses:
        return student_embedding.sum() * 0.0
    return torch.stack(losses).mean()


def build_feature_cache(
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


def _teacher_cache_indices(
    flattened_indices: list[int], negative_counts: torch.Tensor, num_database: int
) -> list[int]:
    queries: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    cursor = 0
    for count_tensor in negative_counts:
        count = int(count_tensor.item())
        queries.append(int(flattened_indices[cursor]) + num_database)
        positives.append(int(flattened_indices[cursor + 1]))
        negatives.extend(
            int(index) for index in flattened_indices[cursor + 2 : cursor + 2 + count]
        )
        cursor += 2 + count
    if cursor != len(flattened_indices):
        raise ValueError("Unexpected flattened triplet indices")
    return queries + positives + negatives


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
    teacher_cache_path: Path | None = None,
) -> EpochLosses:
    if kd_weight > 0 and teacher is None and teacher_cache_path is None:
        raise ValueError("A teacher model or feature cache is required for KD")
    if teacher is not None:
        teacher.eval()
    if teacher_cache_path is not None and not teacher_cache_path.is_file():
        raise FileNotFoundError(f"Missing teacher feature cache: {teacher_cache_path}")
    subset_count = (
        ceil(len(train_dataset) / cache_refresh_rate) if cache_refresh_rate > 0 else 1
    )
    subset_indices = np.array_split(np.arange(len(train_dataset)), subset_count)
    total_loss = 0.0
    total_triplet = 0.0
    total_kd = 0.0
    batches = 0

    teacher_cache = (
        h5py.File(teacher_cache_path, "r") if teacher_cache_path is not None else None
    )
    try:
        for indices in subset_indices:
            build_feature_cache(
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
            set_train_mode_with_frozen_backbone(model)
            for query, positives, negatives, negative_counts, sample_indices in tqdm(
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
                    if teacher_cache is not None:
                        cache_indices = _teacher_cache_indices(
                            sample_indices,
                            negative_counts,
                            train_dataset.dbStruct.numDb,
                        )
                        teacher_array = np.stack(
                            [teacher_cache["features"][index] for index in cache_indices]
                        )
                        teacher_embedding = torch.from_numpy(teacher_array).to(
                            device, non_blocking=True
                        )
                    else:
                        with torch.no_grad():
                            teacher_features = teacher.encoder(images)
                            teacher_embedding = teacher.pool(teacher_features)
                    kd = relational_distillation_loss(
                        student_embedding,
                        teacher_embedding,
                        query_count,
                        negative_counts,
                        temperature,
                    )
                else:
                    kd = torch.zeros((), device=device)
                # Match the legacy KD experiment's objective weighting while
                # replacing its detached cluster-logit KD with relational KD.
                loss = (1.0 - kd_weight) * triplet + kd_weight * kd
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if pruning:
                    prune_model(model, pruning_ratio)
                total_loss += float(loss.item())
                total_triplet += float(triplet.item())
                total_kd += float(kd.item())
                batches += 1
            cache_path.unlink(missing_ok=True)
    finally:
        if teacher_cache is not None:
            teacher_cache.close()
    denominator = max(batches, 1)
    return EpochLosses(
        total=total_loss / denominator,
        triplet=total_triplet / denominator,
        kd=total_kd / denominator,
    )
