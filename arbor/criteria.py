"""Impurity and split-quality measures shared by ID3 and C4.5.

Every function works on *weighted* class counts. ID3 passes plain integer
counts; C4.5 passes fractional counts because it spreads cases with missing
values across branches (Quinlan, 1993, ch. 3).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPSILON = 1e-12


def entropy(counts: Sequence[float] | np.ndarray) -> float:
    """Shannon entropy, in bits, of a class-count vector.

    H(S) = -sum_i p_i * log2(p_i), with 0 * log2(0) taken as 0.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= EPSILON:
        return 0.0
    p = counts[counts > EPSILON] / total
    return float(-(p * np.log2(p)).sum())


def entropy_rows(counts: np.ndarray) -> np.ndarray:
    """Row-wise entropy for a 2-D array of class counts (one row per partition)."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > EPSILON, counts / totals, 0.0)
        log_p = np.where(p > EPSILON, np.log2(p), 0.0)
    return -(p * log_p).sum(axis=1)


def conditional_entropy(partitions: np.ndarray) -> float:
    """Expected entropy after a split: sum_v |S_v| / |S| * H(S_v).

    ``partitions`` has one row of class counts per branch.
    """
    partitions = np.asarray(partitions, dtype=float)
    sizes = partitions.sum(axis=1)
    total = sizes.sum()
    if total <= EPSILON:
        return 0.0
    return float((sizes / total * entropy_rows(partitions)).sum())


def information_gain(parent: np.ndarray, partitions: np.ndarray) -> float:
    """Gain(S, A) = H(S) - sum_v |S_v| / |S| * H(S_v)."""
    return entropy(parent) - conditional_entropy(partitions)


def split_information(partition_sizes: Sequence[float] | np.ndarray) -> float:
    """SplitInfo(S, A) = -sum_v |S_v| / |S| * log2(|S_v| / |S|).

    This is the entropy of the partition sizes. It grows with the number of
    branches, which is what lets gain ratio penalise many-valued attributes.
    """
    return entropy(partition_sizes)


def gain_ratio(gain: float, split_info: float) -> float:
    """GainRatio(S, A) = Gain(S, A) / SplitInfo(S, A); 0 when SplitInfo is ~0."""
    if split_info <= EPSILON:
        return 0.0
    return gain / split_info
