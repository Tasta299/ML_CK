"""Incremental DBSCAN implemented from neighborhood and core-point rules.

NumPy performs array arithmetic; no clustering or metric library is used.
Insertion updates neighborhoods and core status, then rebuilds connected
components of core points. This handles cluster creation and merges exactly
for a fixed eps/min_samples under insertion-only updates.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def pairwise_distances(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Euclidean distances calculated with NumPy matrix arithmetic."""
    a = np.asarray(a, dtype=float)
    b = a if b is None else np.asarray(b, dtype=float)
    squared = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(squared, 0.0))


def prepare_features(frame, category_weight: float = 1.0) -> np.ndarray:
    """Six scaled measurements plus one-hot shelf-life group.

    Different groups have Euclidean distance ``category_weight`` from the
    category encoding alone. Audit/stream columns are intentionally excluded.
    """
    columns = [
        "Unitprice_scaled",
        "Outbound number_scaled",
        "Pal grossweight_scaled",
        "Pal height_scaled",
        "Units per pal_scaled",
        "Avg_units_per_order_scaled",
    ]
    continuous = frame[columns].to_numpy(dtype=float)
    groups = frame["Expire_group"].to_numpy(dtype=int)
    if np.any((groups < 0) | (groups > 3)):
        raise ValueError("Expire_group must contain integers from 0 to 3")
    categorical = np.eye(4, dtype=float)[groups] * (category_weight / np.sqrt(2.0))
    result = np.column_stack((continuous, categorical))
    if not np.isfinite(result).all():
        raise ValueError("Features must be finite")
    return result


class IncrementalDBSCAN:
    """Insertion-only DBSCAN with exact neighborhood updates.

    ``min_samples`` includes the point itself. Cluster IDs are retained when
    possible; when components merge, the smallest old ID survives. Border
    points adjacent to several components receive the smallest component ID.
    """

    def __init__(self, eps: float = 0.8, min_samples: int = 8):
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        if not isinstance(min_samples, (int, np.integer)) or min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self._reset()

    def _reset(self) -> None:
        self.X_ = np.empty((0, 0), dtype=float)
        self.neighbors_: list[np.ndarray] = []
        self.core_mask_ = np.empty(0, dtype=bool)
        self.labels_ = np.empty(0, dtype=int)
        self.next_cluster_id_ = 0
        self.history_: list[dict] = []

    def fit(self, X: np.ndarray) -> "IncrementalDBSCAN":
        self._reset()
        return self.partial_fit(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_.copy()

    def partial_fit(self, X_new: np.ndarray) -> "IncrementalDBSCAN":
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim != 2 or len(X_new) == 0:
            raise ValueError("X_new must be a non-empty 2D array")
        if not np.isfinite(X_new).all():
            raise ValueError("X_new must contain finite values")
        old_n = len(self.labels_)
        if old_n and X_new.shape[1] != self.X_.shape[1]:
            raise ValueError("Feature count changed between batches")

        old_labels = self.labels_.copy()
        old_core = self.core_mask_.copy()
        new_n = len(X_new)
        new_ids = np.arange(old_n, old_n + new_n)

        # Only distances touching new points are computed on an update.
        if old_n:
            cross = pairwise_distances(self.X_, X_new) <= self.eps + 1e-12
            for i in range(old_n):
                additions = new_ids[cross[i]]
                if len(additions):
                    self.neighbors_[i] = np.concatenate((self.neighbors_[i], additions))
        else:
            cross = np.empty((0, new_n), dtype=bool)
        within = pairwise_distances(X_new) <= self.eps + 1e-12
        for j in range(new_n):
            old_neighbors = np.flatnonzero(cross[:, j])
            new_neighbors = new_ids[np.flatnonzero(within[j])]
            self.neighbors_.append(np.concatenate((old_neighbors, new_neighbors)))

        self.X_ = X_new.copy() if old_n == 0 else np.vstack((self.X_, X_new))
        self.core_mask_ = np.array(
            [len(neighbors) >= self.min_samples for neighbors in self.neighbors_], dtype=bool
        )
        self._assign_components(old_labels, old_core)

        old_clusters = set(old_labels[old_core]) - {-1}
        new_clusters = set(self.labels_[self.core_mask_]) - {-1}
        self.history_.append({
            "n_seen": len(self.labels_),
            "n_added": new_n,
            "n_clusters": len(new_clusters),
            "n_noise": int(np.sum(self.labels_ == -1)),
            "new_core": int(np.sum(self.core_mask_[old_n:])),
            "promoted_core": int(np.sum(self.core_mask_[:old_n] & ~old_core)),
            "previous_noise_absorbed": int(np.sum((old_labels == -1) & (self.labels_[:old_n] != -1))),
            "old_cluster_ids_retired_by_merges": len(old_clusters - new_clusters),
        })
        return self

    def _assign_components(self, old_labels: np.ndarray, old_core: np.ndarray) -> None:
        n = len(self.X_)
        labels = np.full(n, -1, dtype=int)
        seen = np.zeros(n, dtype=bool)

        # Core-point components are the density-connected DBSCAN clusters.
        for start in np.flatnonzero(self.core_mask_):
            if seen[start]:
                continue
            queue = deque([int(start)])
            seen[start] = True
            component = []
            while queue:
                point = queue.popleft()
                component.append(point)
                for neighbor in self.neighbors_[point]:
                    if self.core_mask_[neighbor] and not seen[neighbor]:
                        seen[neighbor] = True
                        queue.append(int(neighbor))
            old_ids = {
                int(old_labels[i]) for i in component
                if i < len(old_labels) and old_core[i] and old_labels[i] >= 0
            }
            if old_ids:
                cluster_id = min(old_ids)
            else:
                cluster_id = self.next_cluster_id_
                self.next_cluster_id_ += 1
            labels[component] = cluster_id

        # A border point belongs to any neighboring core component.
        for point in np.flatnonzero(~self.core_mask_):
            adjacent = [int(labels[i]) for i in self.neighbors_[point] if self.core_mask_[i]]
            if adjacent:
                labels[point] = min(adjacent)
        self.labels_ = labels

    @property
    def core_sample_indices_(self) -> np.ndarray:
        return np.flatnonzero(self.core_mask_)

    @property
    def n_clusters_(self) -> int:
        return len(set(self.labels_) - {-1})

    @property
    def n_noise_(self) -> int:
        return int(np.sum(self.labels_ == -1))

    def get_params(self) -> dict:
        return {"eps": self.eps, "min_samples": self.min_samples}


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray, *, silhouette_sample: int | None = 800, seed: int = 42
) -> dict:
    """Manual internal DBSCAN diagnostics; noise is excluded from geometry metrics."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same row count")
    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    n_clusters = len(cluster_ids)
    n_noise = int(np.sum(labels == -1))
    covered = len(labels) - n_noise
    result = {
        "n_samples": len(labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels),
        "coverage": covered / len(labels),
        "largest_cluster_share": float(max(counts) / covered) if covered else np.nan,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if n_clusters < 2 or covered <= n_clusters:
        return result

    clustered_X = X[labels >= 0]
    clustered_labels = labels[labels >= 0]
    center = np.mean(clustered_X, axis=0)
    centroids = np.vstack([np.mean(clustered_X[clustered_labels == k], axis=0) for k in cluster_ids])
    scatters = np.array([
        np.mean(np.linalg.norm(clustered_X[clustered_labels == k] - centroids[j], axis=1))
        for j, k in enumerate(cluster_ids)
    ])
    between_centers = pairwise_distances(centroids)
    ratios = (scatters[:, None] + scatters[None, :]) / np.where(between_centers > 0, between_centers, np.nan)
    ratios = np.nan_to_num(ratios, nan=np.inf)
    np.fill_diagonal(ratios, -np.inf)
    result["davies_bouldin"] = float(np.mean(np.max(ratios, axis=1)))
    between = sum(counts[j] * np.sum((centroids[j] - center) ** 2) for j in range(n_clusters))
    within = sum(
        np.sum((clustered_X[clustered_labels == k] - centroids[j]) ** 2)
        for j, k in enumerate(cluster_ids)
    )
    result["calinski_harabasz"] = float(
        (between / (n_clusters - 1)) / (within / (covered - n_clusters))
    ) if within > 0 else np.inf

    index = np.arange(covered)
    if silhouette_sample is not None and covered > silhouette_sample:
        index = np.sort(np.random.default_rng(seed).choice(index, silhouette_sample, replace=False))
    sample_X = clustered_X[index]
    sample_labels = clustered_labels[index]
    sample_clusters, sample_counts = np.unique(sample_labels, return_counts=True)
    if len(sample_clusters) < 2:
        return result
    distances = pairwise_distances(sample_X)
    silhouettes = np.zeros(len(index))
    for j, cluster in enumerate(sample_clusters):
        own = np.flatnonzero(sample_labels == cluster)
        if sample_counts[j] < 2:
            continue
        a = distances[np.ix_(own, own)].sum(axis=1) / (len(own) - 1)
        other_means = [
            distances[np.ix_(own, np.flatnonzero(sample_labels == other))].mean(axis=1)
            for other in sample_clusters if other != cluster
        ]
        b = np.min(other_means, axis=0)
        denominator = np.maximum(a, b)
        silhouettes[own] = np.divide(b - a, denominator, out=np.zeros_like(a), where=denominator > 0)
    result["silhouette"] = float(np.mean(silhouettes))
    return result


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Partition agreement independent of numeric cluster IDs, implemented manually."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if len(labels_a) != len(labels_b):
        raise ValueError("Label vectors must have the same length")
    n = len(labels_a)
    if n < 2:
        return 1.0
    _, a = np.unique(labels_a, return_inverse=True)
    _, b = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)
    choose2 = lambda values: np.sum(values * (values - 1) // 2)
    observed = choose2(contingency)
    sum_a = choose2(contingency.sum(axis=1))
    sum_b = choose2(contingency.sum(axis=0))
    total = n * (n - 1) // 2
    expected = sum_a * sum_b / total
    maximum = (sum_a + sum_b) / 2
    denominator = maximum - expected
    return float((observed - expected) / denominator) if denominator else 1.0
