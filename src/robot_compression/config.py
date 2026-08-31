from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    pittsburgh_root: Path = Path(
        os.getenv("PITTSBURGH_ROOT", PROJECT_ROOT / "data/datasets/pittsburgh")
    )
    runs_root: Path = Path(os.getenv("RUNS_ROOT", PROJECT_ROOT / "artifacts/runs"))
    cache_root: Path = Path(os.getenv("CACHE_ROOT", PROJECT_ROOT / "artifacts/cache"))
    centroids_root: Path = Path(
        os.getenv("CENTROIDS_ROOT", PROJECT_ROOT / "artifacts/centroids")
    )

    def create_runtime_dirs(self) -> None:
        for path in (self.runs_root, self.cache_root, self.centroids_root):
            path.mkdir(parents=True, exist_ok=True)


def centroid_path(paths: Paths, arch: str, dataset: str, clusters: int) -> Path:
    safe_arch = arch.replace("-", "")
    return paths.centroids_root / f"{safe_arch}_{dataset}_{clusters}_desc_cen.hdf5"
