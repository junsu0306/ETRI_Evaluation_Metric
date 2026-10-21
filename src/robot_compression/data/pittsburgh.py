from __future__ import annotations

import itertools
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import torch
from joblib import parallel_backend
from PIL import Image
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from torch.utils import data
from torchvision import transforms

from robot_compression.config import Paths


DbStruct = namedtuple(
    "DbStruct",
    [
        "whichSet",
        "dataset",
        "dbImage",
        "utmDb",
        "qImage",
        "utmQ",
        "numDb",
        "numQ",
        "posDistThr",
        "posDistSqThr",
        "nonTrivPosDistSqThr",
    ],
)


def dataset_paths(root: Path | None = None) -> dict[str, Path]:
    root = Path(root or Paths().pittsburgh_root).resolve()
    return {
        "root": root,
        "images": root / "images",
        "queries": root / "queries_real",
        "specs": root / "datasets",
    }


def validate_layout(root: Path | None = None) -> list[str]:
    paths = dataset_paths(root)
    missing: list[str] = []
    for key in ("images", "queries", "specs"):
        if not paths[key].is_dir():
            missing.append(str(paths[key]))
    for name in ("pitts30k_train.mat", "pitts30k_val.mat", "pitts30k_test.mat"):
        candidate = paths["specs"] / name
        if not candidate.is_file():
            missing.append(str(candidate))
    return missing


def input_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def parse_db_struct(path: Path) -> DbStruct:
    mat_struct = loadmat(path)["dbStruct"].item()
    dataset = "pitts250k" if "250k" in path.name else "pitts30k"
    return DbStruct(
        mat_struct[0].item(),
        dataset,
        [item[0].item() for item in mat_struct[1]],
        mat_struct[2].T,
        [item[0].item() for item in mat_struct[3]],
        mat_struct[4].T,
        int(mat_struct[5].item()),
        int(mat_struct[6].item()),
        float(mat_struct[7].item()),
        float(mat_struct[8].item()),
        float(mat_struct[9].item()),
    )


class WholeDatasetFromStruct(data.Dataset):
    def __init__(
        self,
        struct_file: Path,
        root: Path | None = None,
        transform=None,
        only_db: bool = False,
    ) -> None:
        super().__init__()
        self.paths = dataset_paths(root)
        self.transform = transform or input_transform()
        self.dbStruct = parse_db_struct(Path(struct_file))
        self.images = [self.paths["images"] / image for image in self.dbStruct.dbImage]
        if not only_db:
            self.images.extend(
                self.paths["queries"] / image for image in self.dbStruct.qImage
            )
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.positives = None
        self.distances = None

    def __getitem__(self, index: int):
        with Image.open(self.images[index]) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, index

    def __len__(self) -> int:
        return len(self.images)

    def getPositives(self):
        if self.positives is None:
            with parallel_backend("threading", n_jobs=-1):
                knn = NearestNeighbors(n_jobs=-1).fit(self.dbStruct.utmDb)
                self.distances, self.positives = knn.radius_neighbors(
                    self.dbStruct.utmQ, radius=self.dbStruct.posDistThr
                )
        return self.positives


class QueryDatasetFromStruct(data.Dataset):
    def __init__(
        self,
        struct_file: Path,
        root: Path | None = None,
        n_neg_sample: int = 1000,
        n_neg: int = 10,
        margin: float = 0.1,
        transform=None,
    ) -> None:
        super().__init__()
        self.paths = dataset_paths(root)
        self.transform = transform or input_transform()
        self.margin = margin
        self.dbStruct = parse_db_struct(Path(struct_file))
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = n_neg_sample
        self.nNeg = n_neg

        with parallel_backend("threading", n_jobs=-1):
            knn = NearestNeighbors(n_jobs=-1).fit(self.dbStruct.utmDb)
            self.nontrivial_positives = list(
                knn.radius_neighbors(
                    self.dbStruct.utmQ,
                    radius=self.dbStruct.nonTrivPosDistSqThr**0.5,
                    return_distance=False,
                )
            )
            potential_positives = knn.radius_neighbors(
                self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr,
                return_distance=False,
            )

        self.nontrivial_positives = [
            np.sort(values) for values in self.nontrivial_positives
        ]
        self.queries = np.where(
            np.asarray([len(values) for values in self.nontrivial_positives]) > 0
        )[0]
        all_db = np.arange(self.dbStruct.numDb)
        self.potential_negatives = [
            np.setdiff1d(all_db, positives, assume_unique=True)
            for positives in potential_positives
        ]
        self.cache: str | None = None
        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, item: int):
        if self.cache is None:
            raise RuntimeError("Feature cache must be built before triplet sampling")
        index = self.queries[item]
        with h5py.File(self.cache, mode="r") as handle:
            features = handle["features"]
            query_feature = features[index + self.dbStruct.numDb]
            positive_features = features[self.nontrivial_positives[index].tolist()]
            with parallel_backend("threading", n_jobs=-1):
                knn = NearestNeighbors(n_jobs=-1).fit(positive_features)
                positive_distance, positive_nn = knn.kneighbors(
                    query_feature.reshape(1, -1), 1
                )
                positive_distance = positive_distance.item()
                positive_index = self.nontrivial_positives[index][
                    positive_nn[0]
                ].item()
                sample_count = min(self.nNegSample, len(self.potential_negatives[index]))
                negative_sample = np.random.choice(
                    self.potential_negatives[index], sample_count, replace=False
                )
                negative_sample = np.unique(
                    np.concatenate([self.negCache[index], negative_sample])
                )
                negative_features = features[list(map(int, negative_sample))]
                knn.fit(negative_features)
                neighbor_count = min(self.nNeg * 10, len(negative_sample))
                negative_distance, negative_nn = knn.kneighbors(
                    query_feature.reshape(1, -1), neighbor_count
                )

        violating = negative_distance.reshape(-1) < positive_distance + self.margin**0.5
        if not np.any(violating):
            return None
        negative_nn = negative_nn.reshape(-1)[violating][: self.nNeg]
        negative_indices = negative_sample[negative_nn].astype(np.int32)
        self.negCache[index] = negative_indices

        query = self._load(self.paths["queries"] / self.dbStruct.qImage[index])
        positive = self._load(self.paths["images"] / self.dbStruct.dbImage[positive_index])
        negatives = torch.stack(
            [
                self._load(self.paths["images"] / self.dbStruct.dbImage[negative])
                for negative in negative_indices
            ],
            0,
        )
        return query, positive, negatives, [index, positive_index, *negative_indices.tolist()]

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image) if self.transform else image

    def __len__(self) -> int:
        return len(self.queries)


def collate_fn(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None, None, None, None, None
    query, positive, negatives, indices = zip(*valid)
    return (
        data.default_collate(query),
        data.default_collate(positive),
        torch.cat(negatives, 0),
        data.default_collate([item.shape[0] for item in negatives]),
        list(itertools.chain(*indices)),
    )


def get_whole_set(split: str, root: Path | None = None) -> WholeDatasetFromStruct:
    if split not in {"train", "val", "test", "val250k", "test250k"}:
        raise ValueError(f"Unsupported Pittsburgh split: {split}")
    stem = split.replace("val250k", "250k_val").replace("test250k", "250k_test")
    if not stem.startswith("250k"):
        stem = f"30k_{stem}"
    struct_file = dataset_paths(root)["specs"] / f"pitts{stem}.mat"
    return WholeDatasetFromStruct(struct_file, root=root)


def get_whole_training_set(
    root: Path | None = None, only_db: bool = False
) -> WholeDatasetFromStruct:
    struct_file = dataset_paths(root)["specs"] / "pitts30k_train.mat"
    return WholeDatasetFromStruct(struct_file, root=root, only_db=only_db)


def get_training_query_set(
    root: Path | None = None, margin: float = 0.1
) -> QueryDatasetFromStruct:
    struct_file = dataset_paths(root)["specs"] / "pitts30k_train.mat"
    return QueryDatasetFromStruct(struct_file, root=root, margin=margin)
