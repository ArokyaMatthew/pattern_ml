"""ID3 (Quinlan, 1986).

* Split criterion: information gain.
* Attributes: categorical only. Each split is multiway, one branch per value
  present at the node, and an attribute is used at most once per path.
  Continuous attributes must be discretised first (``TreePreprocessor``
  does this when the pipeline is built for ID3).
* Missing values: treated as their own category, ``"(missing)"``. In the
  workbench they are imputed before the tree ever sees them.
* No pruning. Growth stops when a node is pure, no attributes remain, or one
  of the optional pre-pruning limits below is reached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arbor.criteria import EPSILON, gain_ratio, information_gain, split_information
from arbor.tree import CATEGORICAL, BaseDecisionTree, SplitCandidate

MISSING = "(missing)"


class ID3Classifier(BaseDecisionTree):
    """ID3 decision tree classifier.

    Parameters
    ----------
    max_depth : int or None
        Grow at most this many levels below the root. ``None`` grows until
        the classic ID3 stopping rules apply.
    min_samples_split : int
        A node with fewer cases than this becomes a leaf.
    min_gain : float
        A split must gain more than this many bits to be made.
    """

    algorithm = "ID3"
    _distribute_unknown = False

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, min_gain: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if self.min_gain < 0:
            raise ValueError("min_gain cannot be negative.")

    def _feature_kind(self, column: pd.Series) -> str:
        return CATEGORICAL

    def _prepare_column(self, column: pd.Series, kind: str) -> pd.Series:
        if not column.isna().any():
            return column
        if isinstance(column.dtype, pd.CategoricalDtype):
            if MISSING not in column.cat.categories:
                column = column.cat.add_categories([MISSING])
            return column.fillna(MISSING)
        return column.astype(object).where(column.notna(), MISSING)

    def _should_stop(self, counts: np.ndarray, depth: int) -> bool:
        if np.count_nonzero(counts > EPSILON) <= 1:
            return True
        if counts.sum() < self.min_samples_split:
            return True
        return self.max_depth is not None and depth >= self.max_depth

    def _score(self, feature: str, idx: np.ndarray, w: np.ndarray, counts: np.ndarray) -> SplitCandidate:
        candidate = SplitCandidate(feature=feature, kind=CATEGORICAL)
        table, _ = self._contingency(feature, idx, w)
        if len(table) < 2:
            candidate.eligible = False
            candidate.note = "one value at this node"
            return candidate
        candidate.gain = information_gain(counts, table)
        candidate.split_info = split_information(table.sum(axis=1))
        candidate.gain_ratio = gain_ratio(candidate.gain, candidate.split_info)
        return candidate

    def _select(self, candidates: list[SplitCandidate]) -> SplitCandidate | None:
        eligible = [c for c in candidates if c.eligible]
        if not eligible:
            return None
        # max() keeps the first of equal gains, so ties go to column order.
        best = max(eligible, key=lambda c: c.gain)
        if best.gain <= max(self.min_gain, EPSILON):
            best.note = f"gain below threshold {self.min_gain:g}" if self.min_gain else "no information gain"
            return None
        return best
