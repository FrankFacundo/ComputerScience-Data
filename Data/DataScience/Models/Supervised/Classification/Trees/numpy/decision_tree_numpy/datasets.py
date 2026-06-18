"""Built-in dataset metadata for the NumPy decision tree examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NUMPY_DIR = Path(__file__).resolve().parents[1]
TREES_DIR = NUMPY_DIR.parent
DEFAULT_DATASET = "iris"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    path: Path
    feature_names: list[str]
    target_name: str
    default_sample: list[float]
    default_model_path: Path
    delimiter: str | None = None
    skip_rows: int = 0


DATASETS: dict[str, DatasetSpec] = {
    "file1": DatasetSpec(
        name="file1",
        description="Small teaching dataset from the Trees folder.",
        path=TREES_DIR / "file1.txt",
        feature_names=["A", "B"],
        target_name="C",
        default_sample=[0, 1],
        default_model_path=NUMPY_DIR / "decision_tree_numpy_model.json",
    ),
    "iris": DatasetSpec(
        name="iris",
        description="Fisher's public Iris flower dataset.",
        path=NUMPY_DIR / "datasets" / "iris.csv",
        feature_names=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        target_name="species",
        default_sample=[5.1, 3.5, 1.4, 0.2],
        default_model_path=NUMPY_DIR / "decision_tree_numpy_iris_model.json",
        delimiter=",",
        skip_rows=1,
    ),
}


def dataset_choices() -> list[str]:
    return sorted(DATASETS)


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {name!r}. Available datasets: {dataset_choices()}"
        ) from exc
