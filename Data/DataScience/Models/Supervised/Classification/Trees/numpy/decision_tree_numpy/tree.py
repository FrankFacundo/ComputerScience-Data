"""A small DecisionTreeClassifier implemented with Python and NumPy only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DecisionTreeConfig


FEATURE_THRESHOLD = 1e-7
IMPURITY_EPSILON = np.finfo(float).eps


@dataclass
class TreeNode:
    depth: int
    n_samples: int
    impurity: float
    class_counts: list[int]
    prediction_index: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "n_samples": self.n_samples,
            "impurity": self.impurity,
            "class_counts": self.class_counts,
            "prediction_index": self.prediction_index,
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        node = cls(
            depth=int(data["depth"]),
            n_samples=int(data["n_samples"]),
            impurity=float(data["impurity"]),
            class_counts=[int(value) for value in data["class_counts"]],
            prediction_index=int(data["prediction_index"]),
            feature_index=(
                int(data["feature_index"]) if data["feature_index"] is not None else None
            ),
            threshold=(
                float(data["threshold"]) if data["threshold"] is not None else None
            ),
        )
        if data["left"] is not None:
            node.left = cls.from_dict(data["left"])
        if data["right"] is not None:
            node.right = cls.from_dict(data["right"])
        return node


class NumpyDecisionTreeClassifier:
    """Binary-threshold classification tree using only NumPy arrays.

    The split search follows the core sklearn ``DecisionTreeClassifier``
    dense-data ``splitter="best"`` idea for numeric features:

    1. Calculate the node impurity.
    2. Sort the samples by one candidate feature.
    3. Sweep possible split positions from left to right, updating class counts
       incrementally.
    4. Pick the split with the largest impurity decrease.
    5. Recurse until the node is pure or a stopping rule is reached.

    sklearn's production implementation is Cython and also supports features
    such as sample weights, sparse matrices, missing-value routing, random
    feature subsets, and monotonic constraints. Those are intentionally omitted
    here so the core best-split algorithm remains readable.
    """

    def __init__(self, config: DecisionTreeConfig | None = None):
        self.config = config or DecisionTreeConfig()
        self.root_: TreeNode | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_names_: list[str] | None = None
        self.target_name_: str = self.config.target_name

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        target_name: str | None = None,
    ) -> "NumpyDecisionTreeClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of rows")
        if len(x) == 0:
            raise ValueError("Cannot train on an empty dataset")

        self.classes_ = np.unique(y)
        y_indices = np.searchsorted(self.classes_, y)
        self.n_features_in_ = x.shape[1]
        self.feature_names_ = self._resolve_feature_names(feature_names)
        self.target_name_ = target_name or self.config.target_name
        self.root_ = self._build_tree(x, y_indices, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {x.shape[1]}"
            )

        prediction_indices = np.array([self._predict_index(row) for row in x], dtype=int)
        return self.classes_[prediction_indices]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {x.shape[1]}"
            )

        probabilities = []
        for row in x:
            leaf = self._predict_leaf(row)
            counts = np.array(leaf.class_counts, dtype=float)
            probabilities.append(counts / counts.sum())
        return np.array(probabilities)

    def export_text(self) -> str:
        self._check_is_fitted()
        return self._export_node(self.root_)

    def to_dict(self) -> dict[str, Any]:
        self._check_is_fitted()
        return {
            "config": self.config.to_dict(),
            "classes": self.classes_.tolist(),
            "n_features_in": self.n_features_in_,
            "feature_names": self.feature_names_,
            "target_name": self.target_name_,
            "tree": self.root_.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NumpyDecisionTreeClassifier":
        model = cls(config=DecisionTreeConfig.from_dict(data["config"]))
        model.classes_ = np.array(data["classes"])
        model.n_features_in_ = int(data["n_features_in"])
        model.feature_names_ = list(data["feature_names"])
        model.target_name_ = data["target_name"]
        model.root_ = TreeNode.from_dict(data["tree"])
        return model

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        counts = self._class_counts(y)
        impurity = self._impurity(counts)
        prediction_index = int(np.argmax(counts))
        node = TreeNode(
            depth=depth,
            n_samples=len(y),
            impurity=impurity,
            class_counts=counts.astype(int).tolist(),
            prediction_index=prediction_index,
        )

        if self._should_stop(y, depth, impurity):
            return node

        split = self._best_split(x, y, impurity)
        if split is None:
            return node

        feature_index, threshold = split
        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask

        node.feature_index = int(feature_index)
        node.threshold = float(threshold)
        node.left = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(x[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        parent_impurity: float,
    ) -> tuple[int, float] | None:
        best_feature: int | None = None
        best_threshold: float | None = None
        best_gain = -np.inf
        n_samples, n_features = x.shape
        total_counts = self._class_counts(y).astype(float)

        for feature_index in range(n_features):
            feature_values = x[:, feature_index]
            order = np.argsort(feature_values, kind="mergesort")
            sorted_values = feature_values[order]
            sorted_targets = y[order]

            # sklearn treats very small floating-point differences as equal
            # when searching thresholds. A feature that is constant at this
            # tolerance cannot create a useful threshold.
            if sorted_values[-1] <= sorted_values[0] + FEATURE_THRESHOLD:
                continue

            left_counts = np.zeros_like(total_counts)
            right_counts = total_counts.copy()
            p = 0

            while p < n_samples:
                group_start = p
                # Move over all samples whose feature values are effectively
                # equal. Splitting inside this group would send equal values to
                # different children, which threshold trees do not do.
                while (
                    p + 1 < n_samples
                    and sorted_values[p + 1] <= sorted_values[p] + FEATURE_THRESHOLD
                ):
                    p += 1

                group_targets = sorted_targets[group_start : p + 1]
                group_counts = self._class_counts(group_targets).astype(float)
                left_counts += group_counts
                right_counts -= group_counts

                split_pos = p + 1
                if split_pos >= n_samples:
                    break

                left_count = split_pos
                right_count = n_samples - split_pos
                if left_count < self.config.min_samples_leaf:
                    p = split_pos
                    continue
                if right_count < self.config.min_samples_leaf:
                    p = split_pos
                    continue

                left_impurity = self._impurity(left_counts)
                right_impurity = self._impurity(right_counts)
                weighted_impurity = (
                    (left_count / n_samples) * left_impurity
                    + (right_count / n_samples) * right_impurity
                )
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    threshold = (
                        sorted_values[p] / 2.0 + sorted_values[split_pos] / 2.0
                    )
                    if threshold == sorted_values[split_pos] or not np.isfinite(
                        threshold
                    ):
                        threshold = sorted_values[p]

                    best_gain = float(gain)
                    best_feature = int(feature_index)
                    best_threshold = float(threshold)

                p = split_pos

        if best_feature is None or best_threshold is None:
            return None
        return best_feature, best_threshold

    def _should_stop(self, y: np.ndarray, depth: int, impurity: float) -> bool:
        if impurity <= IMPURITY_EPSILON:
            return True
        if len(y) < self.config.min_samples_split:
            return True
        if self.config.max_depth is not None and depth >= self.config.max_depth:
            return True
        return False

    def _class_counts(self, y: np.ndarray) -> np.ndarray:
        return np.bincount(y, minlength=len(self.classes_))

    def _impurity(self, counts: np.ndarray) -> float:
        total = np.sum(counts)
        if total == 0:
            return 0.0
        probabilities = counts[counts > 0] / total

        if self.config.criterion == "gini":
            return float(1.0 - np.sum(probabilities**2))
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _predict_index(self, row: np.ndarray) -> int:
        return self._predict_leaf(row).prediction_index

    def _predict_leaf(self, row: np.ndarray) -> TreeNode:
        node = self.root_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def _export_node(self, node: TreeNode, prefix: str = "") -> str:
        if node.is_leaf:
            prediction = self.classes_[node.prediction_index]
            return f"{prefix}|--- class: {self._format_value(prediction)}\n"

        feature_name = self.feature_names_[node.feature_index]
        threshold = self._format_threshold(node.threshold)
        left = f"{prefix}|--- {feature_name} <= {threshold}\n"
        left += self._export_node(node.left, prefix + "|   ")
        right = f"{prefix}|--- {feature_name} >  {threshold}\n"
        right += self._export_node(node.right, prefix + "|   ")
        return left + right

    def _resolve_feature_names(self, feature_names: list[str] | None) -> list[str]:
        names = list(feature_names or self.config.feature_names)
        names.extend(f"X{i + 1}" for i in range(len(names), self.n_features_in_))
        return names[: self.n_features_in_]

    def _check_is_fitted(self) -> None:
        if self.root_ is None or self.classes_ is None or self.n_features_in_ is None:
            raise RuntimeError("Model is not fitted yet")

    @staticmethod
    def _format_threshold(value: float) -> str:
        return f"{value:.2f}"

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)
