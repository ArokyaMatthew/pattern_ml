"""C4.5 (Quinlan, 1993), following Release 8 where the two differ.

What C4.5 adds over ID3:

* **Gain ratio.** Splits are ranked by gain / split information, and only
  attributes whose gain is at least the average gain of all candidates are
  admitted. That stops gain ratio from favouring near-useless splits whose
  split information happens to be tiny.
* **Continuous attributes.** Binary tests ``A <= t``. Cases are sorted once
  per node and every boundary between distinct values is scored; the cut
  with the highest gain wins. Release 8 then charges log2(cuts) / N bits for
  having searched that many cuts.
* **Unknown values.** Gain is computed on cases with a known value and
  scaled by the fraction known; split information counts "unknown" as an
  extra outcome. When the node is split, each case with an unknown value is
  sent down every branch carrying a fraction of its weight. Prediction does
  the same and blends the leaf distributions.
* **Minimum cases.** A split must put at least ``min_samples_leaf`` cases on
  at least two of its branches.
* **Error-based pruning.** After growing, each subtree is compared bottom-up
  with a single leaf using the upper confidence limit of the binomial error
  rate (``confidence`` = CF, 0.25 by default). If the leaf is not worse the
  subtree is replaced. Subtree raising is not implemented.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from arbor.criteria import EPSILON, entropy, entropy_rows, gain_ratio, information_gain, split_information
from arbor.tree import CATEGORICAL, CONTINUOUS, BaseDecisionTree, Node, SplitCandidate


def added_errors(n: float, e: float, cf: float, z_squared: float) -> float:
    """Extra errors C4.5 adds to ``e`` observed errors among ``n`` cases.

    ``e + added_errors(n, e)`` is the upper limit of the confidence interval
    (level ``cf``) for the number of errors a leaf would make on unseen
    cases. This is a direct port of ``AddErrs`` from C4.5's ``stats.c``,
    which uses the normal approximation for e >= 1 and exact binomial
    expressions below that.
    """
    if n <= EPSILON:
        return 0.0
    if e < 1e-6:
        return n * (1 - math.exp(math.log(cf) / n))
    if e < 0.9999:
        at_zero = n * (1 - math.exp(math.log(cf) / n))
        return at_zero + e * (added_errors(n, 1.0, cf, z_squared) - at_zero)
    if e + 0.5 >= n:
        return 0.67 * (n - e)
    upper = (
        e + 0.5 + z_squared / 2
        + math.sqrt(z_squared * ((e + 0.5) * (1 - (e + 0.5) / n) + z_squared / 4))
    ) / (n + z_squared)
    return n * upper - e


class C45Classifier(BaseDecisionTree):
    """C4.5 decision tree classifier.

    Parameters
    ----------
    max_depth : int or None
        Optional cap on depth. C4.5 itself has none.
    min_samples_leaf : int
        C4.5's ``-m`` option: at least two branches of a split must receive
        this many (possibly fractional) cases. Default 2, as in C4.5.
    prune : bool
        Apply error-based pruning after growing.
    confidence : float
        Pruning confidence factor CF in (0, 0.5]. Smaller values prune more.
    """

    algorithm = "C4.5"
    _distribute_unknown = True

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        prune: bool = True,
        confidence: float = 0.25,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.prune = prune
        self.confidence = confidence

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if not 0 < self.confidence <= 0.5:
            raise ValueError("confidence must be in (0, 0.5].")

    def _feature_kind(self, column: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(column) or isinstance(column.dtype, pd.CategoricalDtype):
            return CATEGORICAL
        return CONTINUOUS if pd.api.types.is_numeric_dtype(column) else CATEGORICAL

    def _should_stop(self, counts: np.ndarray, depth: int) -> bool:
        if np.count_nonzero(counts > EPSILON) <= 1:
            return True
        if counts.sum() < 2 * self.min_samples_leaf:
            return True
        return self.max_depth is not None and depth >= self.max_depth

    # -- scoring ---------------------------------------------------------
    def _score(self, feature: str, idx: np.ndarray, w: np.ndarray, counts: np.ndarray) -> SplitCandidate:
        if self.feature_kinds_[feature] == CATEGORICAL:
            return self._score_categorical(feature, idx, w)
        return self._score_continuous(feature, idx, w)

    def _score_categorical(self, feature: str, idx: np.ndarray, w: np.ndarray) -> SplitCandidate:
        candidate = SplitCandidate(feature=feature, kind=CATEGORICAL)
        table, unknown_weight = self._contingency(feature, idx, w)
        known_weight = float(table.sum())
        if len(table) < 2:
            candidate.eligible = False
            candidate.note = "one value at this node"
            return candidate

        known_fraction = known_weight / (known_weight + unknown_weight)
        sizes = table.sum(axis=1)
        candidate.gain = known_fraction * information_gain(table.sum(axis=0), table)
        outcomes = list(sizes) + ([unknown_weight] if unknown_weight > EPSILON else [])
        candidate.split_info = split_information(outcomes)
        candidate.gain_ratio = gain_ratio(candidate.gain, candidate.split_info)
        if np.count_nonzero(sizes >= self.min_samples_leaf - EPSILON) < 2:
            candidate.eligible = False
            candidate.note = f"fewer than two branches with {self.min_samples_leaf}+ cases"
        return candidate

    def _score_continuous(self, feature: str, idx: np.ndarray, w: np.ndarray) -> SplitCandidate:
        candidate = SplitCandidate(feature=feature, kind=CONTINUOUS)
        values = self._X[feature][idx]
        known = ~np.isnan(values)
        unknown_weight = float(w[~known].sum())
        v, y, weights = values[known], self._y[idx][known], w[known]
        known_weight = float(weights.sum())
        m = self.min_samples_leaf
        if known_weight < 2 * m - EPSILON:
            candidate.eligible = False
            candidate.note = "too few known values"
            return candidate

        order = np.argsort(v, kind="mergesort")
        v, y, weights = v[order], y[order], weights[order]
        per_case = np.zeros((len(v), self.n_classes_))
        per_case[np.arange(len(v)), y] = weights
        # left[i] holds class counts of cases 0..i, i.e. a cut after case i.
        left = np.cumsum(per_case, axis=0)[:-1]
        known_counts = per_case.sum(axis=0)
        right = known_counts - left
        boundary = v[:-1] < v[1:]
        n_cuts = int(boundary.sum())
        if n_cuts == 0:
            candidate.eligible = False
            candidate.note = "one value at this node"
            return candidate

        left_weight, right_weight = left.sum(axis=1), right.sum(axis=1)
        valid = boundary & (left_weight >= m - EPSILON) & (right_weight >= m - EPSILON)
        if not valid.any():
            candidate.eligible = False
            candidate.note = f"no cut leaves {m}+ cases on both sides"
            return candidate

        expected = (left_weight * entropy_rows(left) + right_weight * entropy_rows(right)) / known_weight
        best = int(np.argmin(np.where(valid, expected, np.inf)))
        raw_gain = entropy(known_counts) - expected[best]
        penalty = math.log2(n_cuts) / known_weight if n_cuts > 1 else 0.0

        candidate.threshold = float((v[best] + v[best + 1]) / 2)
        candidate.gain = float(known_weight / (known_weight + unknown_weight) * (raw_gain - penalty))
        outcomes = [float(left_weight[best]), float(right_weight[best])]
        if unknown_weight > EPSILON:
            outcomes.append(unknown_weight)
        candidate.split_info = split_information(outcomes)
        candidate.gain_ratio = gain_ratio(candidate.gain, candidate.split_info)
        candidate.note = f"best of {n_cuts} cut point{'s' if n_cuts > 1 else ''}"
        return candidate

    def _select(self, candidates: list[SplitCandidate]) -> SplitCandidate | None:
        for candidate in candidates:
            if candidate.eligible and candidate.gain <= EPSILON:
                reason = "no gain after the cut-point penalty" if candidate.kind == CONTINUOUS else "no gain"
                candidate.note = f"{candidate.note}; {reason}" if candidate.note else reason
        admissible = [c for c in candidates if c.eligible and c.gain > EPSILON]
        if not admissible:
            return None
        average_gain = float(np.mean([c.gain for c in admissible]))
        best = None
        for candidate in admissible:
            if candidate.gain < average_gain - 1e-9:
                candidate.note = "below average gain"
                continue
            if candidate.split_info <= EPSILON:
                continue
            if best is None or candidate.gain_ratio > best.gain_ratio + EPSILON:
                best = candidate
        return best

    # -- pruning ---------------------------------------------------------
    def _post_grow(self) -> None:
        if not self.prune:
            return
        z = NormalDist().inv_cdf(1 - self.confidence)
        self._z_squared = z * z
        self._prune(self.root_)
        del self._z_squared

    def _estimated_errors(self, node: Node) -> float:
        n, e = node.n_samples, node.errors
        return e + added_errors(n, e, self.confidence, self._z_squared)

    def _prune(self, node: Node) -> float:
        """Prune bottom-up; return the estimated errors of what remains."""
        as_leaf = self._estimated_errors(node)
        if node.is_leaf:
            return as_leaf
        as_subtree = sum(self._prune(child) for child in node.children.values())
        node.leaf_error_estimate, node.subtree_error_estimate = as_leaf, as_subtree
        # C4.5 keeps a subtree only if it is clearly (0.1 errors) better.
        if as_leaf <= as_subtree + 0.1:
            node.pruned_feature = node.feature
            node.feature = node.kind = node.threshold = None
            node.children, node.branch_weights = {}, {}
            return as_leaf
        return as_subtree

    @property
    def n_pruned_(self) -> int:
        """Nodes removed by pruning."""
        return self.n_nodes_grown_ - self.n_nodes_
