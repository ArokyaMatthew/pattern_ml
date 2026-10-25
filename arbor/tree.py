"""Tree structure and the estimator machinery shared by ID3 and C4.5.

The two algorithms differ only in how they score and admit candidate splits,
when they stop growing, and what they do after growing (C4.5 prunes). All of
that lives in the subclasses; this module owns everything else: encoding the
input frame, recursive growth, routing cases at prediction time, and
read-only views of the fitted tree (text, rules, decision paths).

The estimators follow scikit-learn's conventions, so they work with
``Pipeline``, ``clone``, ``cross_validate`` and friends.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from arbor.criteria import EPSILON, entropy

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

# Branch keys for binary splits on a continuous attribute.
LEFT = "<="
RIGHT = ">"


def format_number(value: float, digits: int = 4) -> str:
    """Compact, readable rendering of thresholds, bin edges and case counts."""
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if value.is_integer() and abs(value) < 1e9:
        return str(int(value))
    if abs(value) >= 10 ** digits:
        return f"{value:,.0f}"
    return f"{value:.{digits}g}"


@dataclass
class SplitCandidate:
    """How one attribute scored when a node was considered for splitting.

    Every candidate is kept on its node so the UI can show *why* an attribute
    won, not just which one did.
    """

    feature: str
    kind: str
    gain: float = 0.0
    split_info: float = 0.0
    gain_ratio: float = 0.0
    threshold: float | None = None
    eligible: bool = True
    selected: bool = False
    note: str = ""


@dataclass
class Node:
    """A node of a fitted tree. Leaves are nodes without children."""

    counts: np.ndarray
    depth: int
    id: int = -1
    feature: str | None = None
    kind: str | None = None
    threshold: float | None = None
    children: dict[Any, Node] = field(default_factory=dict)
    # Share of the node's known-valued training weight that went down each
    # branch. C4.5 uses it to spread cases whose value is unknown.
    branch_weights: dict[Any, float] = field(default_factory=dict)
    gain: float = 0.0
    gain_ratio: float = 0.0
    candidates: list[SplitCandidate] = field(default_factory=list)
    # Set by C4.5 pruning when this node's subtree was replaced by a leaf.
    pruned_feature: str | None = None
    leaf_error_estimate: float | None = None
    subtree_error_estimate: float | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def n_samples(self) -> float:
        return float(self.counts.sum())

    @property
    def entropy(self) -> float:
        return entropy(self.counts)

    @property
    def distribution(self) -> np.ndarray:
        total = self.counts.sum()
        if total <= EPSILON:
            return np.full(len(self.counts), 1.0 / len(self.counts))
        return self.counts / total

    @property
    def majority(self) -> int:
        return int(np.argmax(self.counts))

    @property
    def errors(self) -> float:
        """Training weight at this node that the majority class gets wrong."""
        return float(self.counts.sum() - self.counts.max())

    def iter_nodes(self) -> Iterator[Node]:
        """Pre-order walk over this node and its descendants."""
        yield self
        for child in self.children.values():
            yield from child.iter_nodes()


@dataclass
class Rule:
    """One root-to-leaf path written as an IF-THEN rule."""

    conditions: list[str]
    prediction: Any
    support: float
    confidence: float
    leaf_id: int

    def as_text(self, target: str = "class") -> str:
        premise = " AND ".join(self.conditions) if self.conditions else "TRUE"
        return f"IF {premise} THEN {target} = {self.prediction}"


class BaseDecisionTree(ClassifierMixin, BaseEstimator):
    """Shared growth, prediction and inspection logic for ID3 and C4.5."""

    #: Display name of the algorithm.
    algorithm: str = ""
    #: When a case has an unknown (missing or unseen) value at a split, C4.5
    #: sends it down every branch weighted by branch size; ID3 stops at that
    #: node and answers with the node's class distribution.
    _distribute_unknown: bool = False

    # -- hooks implemented by subclasses ---------------------------------
    def _feature_kind(self, column: pd.Series) -> str:
        raise NotImplementedError

    def _prepare_column(self, column: pd.Series, kind: str) -> pd.Series:
        return column

    def _validate_params(self) -> None:
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be None or a positive integer.")

    def _should_stop(self, counts: np.ndarray, depth: int) -> bool:
        raise NotImplementedError

    def _score(self, feature: str, idx: np.ndarray, w: np.ndarray, counts: np.ndarray) -> SplitCandidate:
        raise NotImplementedError

    def _select(self, candidates: list[SplitCandidate]) -> SplitCandidate | None:
        raise NotImplementedError

    def _post_grow(self) -> None:
        """Called once the full tree is grown (C4.5 prunes here)."""

    # -- fitting ---------------------------------------------------------
    def fit(self, X, y):
        self._validate_params()
        X = self._as_frame(X)
        y = self._as_target(y, len(X))
        if X.shape[1] == 0:
            raise ValueError("At least one feature column is required.")
        if len(X) == 0:
            raise ValueError("Cannot fit a tree on an empty dataset.")

        started = time.perf_counter()
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        X.columns = list(self.feature_names_in_)
        self.n_features_in_ = X.shape[1]
        self.classes_, y_codes = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        self.feature_kinds_: dict[str, str] = {}
        self.categories_: dict[str, list] = {}
        self._code_of: dict[str, dict[Any, int]] = {}
        encoded: dict[str, np.ndarray] = {}
        for name in self.feature_names_in_:
            kind = self._feature_kind(X[name])
            column = self._prepare_column(X[name], kind)
            self.feature_kinds_[name] = kind
            if kind == CATEGORICAL:
                cat = pd.Categorical(column)
                self.categories_[name] = list(cat.categories)
                self._code_of[name] = {value: i for i, value in enumerate(cat.categories)}
                encoded[name] = cat.codes.astype(np.int64)
            else:
                encoded[name] = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)

        self._X, self._y = encoded, y_codes.astype(np.int64)
        try:
            n = len(X)
            self.root_ = self._grow(np.arange(n), np.ones(n), depth=0, used=frozenset())
            self.n_nodes_grown_ = sum(1 for _ in self.root_.iter_nodes())
            self._post_grow()
        finally:
            del self._X, self._y
        self._finalize()
        self.fit_time_ = time.perf_counter() - started
        return self

    def _grow(self, idx: np.ndarray, w: np.ndarray, depth: int, used: frozenset) -> Node:
        counts = np.bincount(self._y[idx], weights=w, minlength=self.n_classes_)
        node = Node(counts=counts, depth=depth)
        if self._should_stop(counts, depth):
            return node

        node.candidates = [
            self._score(feature, idx, w, counts)
            for feature in self.feature_names_in_
            if feature not in used
        ]
        best = self._select(node.candidates)
        if best is None:
            return node
        branches = self._partition(best, idx, w)
        if len(branches) < 2:
            return node

        best.selected = True
        node.feature, node.kind, node.threshold = best.feature, best.kind, best.threshold
        node.gain, node.gain_ratio = best.gain, best.gain_ratio
        # A categorical attribute is exhausted once split on: every case below
        # the split shares one value. Continuous attributes can be re-used
        # with a different threshold.
        child_used = used | {best.feature} if best.kind == CATEGORICAL else used
        for key, child_idx, child_w, share in branches:
            node.children[key] = self._grow(child_idx, child_w, depth + 1, child_used)
            node.branch_weights[key] = share
        return node

    def _contingency(self, feature: str, idx: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
        """Weighted value-by-class counts of a categorical attribute at a node.

        Returns the table (one row per value present at the node) and the
        total weight of cases whose value is unknown.
        """
        codes = self._X[feature][idx]
        known = codes >= 0
        k = self.n_classes_
        n_values = len(self.categories_[feature])
        flat = codes[known] * k + self._y[idx][known]
        table = np.bincount(flat, weights=w[known], minlength=n_values * k).reshape(n_values, k)
        return table[table.sum(axis=1) > EPSILON], float(w[~known].sum())

    def _partition(self, split: SplitCandidate, idx: np.ndarray, w: np.ndarray):
        """Split the cases at a node into ``(key, idx, weights, share)`` branches."""
        column = self._X[split.feature][idx]
        if split.kind == CONTINUOUS:
            unknown = np.isnan(column)
            masks = [
                (LEFT, ~unknown & (column <= split.threshold)),
                (RIGHT, ~unknown & (column > split.threshold)),
            ]
        else:
            unknown = column < 0
            categories = self.categories_[split.feature]
            masks = [(categories[code], column == code) for code in np.unique(column[~unknown])]

        known_weight = w[~unknown].sum()
        spread = self._distribute_unknown and unknown.any()
        branches = []
        for key, mask in masks:
            branch_weight = w[mask].sum()
            if branch_weight <= EPSILON:
                continue
            share = float(branch_weight / known_weight)
            child_idx, child_w = idx[mask], w[mask]
            if spread:
                child_idx = np.concatenate([child_idx, idx[unknown]])
                child_w = np.concatenate([child_w, w[unknown] * share])
            branches.append((key, child_idx, child_w, share))
        return branches

    def _finalize(self) -> None:
        # Breadth-first ids read top-to-bottom, left-to-right in the drawing.
        nodes: list[Node] = []
        queue = deque([self.root_])
        while queue:
            node = queue.popleft()
            node.id = len(nodes)
            nodes.append(node)
            queue.extend(node.children.values())
        self.nodes_ = nodes
        self.n_nodes_ = len(nodes)
        self.n_leaves_ = sum(node.is_leaf for node in nodes)
        self.depth_ = max(node.depth for node in nodes)

        # Importance = training-weight-weighted gain, normalised to sum to 1.
        position = {name: i for i, name in enumerate(self.feature_names_in_)}
        importance = np.zeros(self.n_features_in_)
        total = self.root_.n_samples
        for node in nodes:
            if not node.is_leaf:
                importance[position[node.feature]] += node.n_samples / total * max(node.gain, 0.0)
        if importance.sum() > EPSILON:
            importance /= importance.sum()
        self.feature_importances_ = importance

    # -- prediction ------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "root_")
        encoded, n = self._encode(X)
        out = np.zeros((n, self.n_classes_))
        self._route(self.root_, encoded, np.arange(n), np.ones(n), out)
        totals = out.sum(axis=1, keepdims=True)
        return out / np.where(totals > EPSILON, totals, 1.0)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _route(self, node: Node, encoded, rows: np.ndarray, weights: np.ndarray, out: np.ndarray) -> None:
        if rows.size == 0:
            return
        if node.is_leaf:
            out[rows] += weights[:, None] * node.distribution
            return

        values = encoded[node.feature][rows]
        if node.kind == CONTINUOUS:
            unknown = np.isnan(values)
            masks = {LEFT: ~unknown & (values <= node.threshold), RIGHT: ~unknown & (values > node.threshold)}
        else:
            code_of = self._code_of[node.feature]
            masks = {key: values == code_of[key] for key in node.children}
            # Missing values and categories this node never saw in training.
            unknown = ~np.logical_or.reduce(list(masks.values()))

        for key, child in node.children.items():
            self._route(child, encoded, rows[masks[key]], weights[masks[key]], out)

        if unknown.any():
            rows_u, weights_u = rows[unknown], weights[unknown]
            if self._distribute_unknown:
                for key, child in node.children.items():
                    self._route(child, encoded, rows_u, weights_u * node.branch_weights[key], out)
            else:
                out[rows_u] += weights_u[:, None] * node.distribution

    def _encode(self, X) -> tuple[dict[str, np.ndarray], int]:
        X = self._as_frame(X)
        absent = [name for name in self.feature_names_in_ if name not in X.columns]
        if absent:
            raise ValueError(f"Input is missing feature columns: {absent}")
        encoded = {}
        for name in self.feature_names_in_:
            kind = self.feature_kinds_[name]
            column = self._prepare_column(X[name], kind)
            if kind == CATEGORICAL:
                # -1 for missing values and for categories never seen in training.
                categories = pd.Index(self.categories_[name], dtype=object)
                encoded[name] = categories.get_indexer(column.astype(object)).astype(np.int64)
            else:
                encoded[name] = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
        return encoded, len(X)

    # -- inspection ------------------------------------------------------
    def get_depth(self) -> int:
        check_is_fitted(self, "root_")
        return self.depth_

    def get_n_leaves(self) -> int:
        check_is_fitted(self, "root_")
        return self.n_leaves_

    def condition(self, node: Node, key: Any, unicode: bool = True) -> str:
        """Human-readable test for the branch ``key`` of ``node``."""
        if node.kind == CONTINUOUS:
            op = ("≤" if unicode else "<=") if key == LEFT else ">"
            return f"{node.feature} {op} {format_number(node.threshold)}"
        return f"{node.feature} = {key}"

    def branch_label(self, node: Node, key: Any) -> str:
        """Edge label for the drawing: just the value or the comparison."""
        if node.kind == CONTINUOUS:
            return f"{'≤' if key == LEFT else '>'} {format_number(node.threshold)}"
        return str(key)

    def decision_path(self, X) -> list[tuple[Node, Any]]:
        """Follow the first case of ``X`` from the root.

        Returns ``(node, branch_key)`` pairs. The key is ``None`` at the leaf,
        or at the node where the case's value is unknown (C4.5 then blends all
        branches; ID3 answers with that node's distribution).
        """
        check_is_fitted(self, "root_")
        frame = self._as_frame(X).iloc[[0]]
        encoded, _ = self._encode(frame)
        path: list[tuple[Node, Any]] = []
        node = self.root_
        while not node.is_leaf:
            key = self._branch_for(node, encoded[node.feature][0])
            path.append((node, key))
            if key is None:
                return path
            node = node.children[key]
        path.append((node, None))
        return path

    def _branch_for(self, node: Node, value) -> Any:
        if node.kind == CONTINUOUS:
            if np.isnan(value):
                return None
            return LEFT if value <= node.threshold else RIGHT
        for key in node.children:
            if self._code_of[node.feature][key] == value:
                return key
        return None

    def rules(self) -> list[Rule]:
        """Every root-to-leaf path as an IF-THEN rule, in drawing order."""
        check_is_fitted(self, "root_")
        found: list[Rule] = []

        def walk(node: Node, conditions: list[str]) -> None:
            if node.is_leaf:
                found.append(
                    Rule(
                        conditions=conditions,
                        prediction=self.classes_[node.majority],
                        support=node.n_samples,
                        confidence=float(node.distribution.max()),
                        leaf_id=node.id,
                    )
                )
                return
            for key, child in node.children.items():
                walk(child, conditions + [self.condition(node, key)])

        walk(self.root_, [])
        return found

    def export_text(self) -> str:
        """Indented text rendering in the style of Weka's J48 output.

        Leaves read ``class (cases/errors)``; fractional counts come from
        C4.5's handling of unknown values.
        """
        check_is_fitted(self, "root_")

        def leaf(node: Node) -> str:
            label = self.classes_[node.majority]
            return f"{label} ({format_number(node.n_samples, 3)}/{format_number(node.errors, 3)})"

        if self.root_.is_leaf:
            return f": {leaf(self.root_)}"
        lines: list[str] = []

        def walk(node: Node, level: int) -> None:
            for key, child in node.children.items():
                text = "|   " * level + self.condition(node, key, unicode=False)
                if child.is_leaf:
                    lines.append(f"{text}: {leaf(child)}")
                else:
                    lines.append(text)
                    walk(child, level + 1)

        walk(self.root_, 0)
        return "\n".join(lines)

    # -- input handling --------------------------------------------------
    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        frame = pd.DataFrame(X)
        names = getattr(self, "feature_names_in_", None)
        if names is not None and frame.shape[1] == len(names):
            frame.columns = list(names)
        else:
            frame.columns = [f"x{i}" for i in range(frame.shape[1])]
        return frame

    @staticmethod
    def _as_target(y, n_rows: int) -> np.ndarray:
        y = pd.Series(np.asarray(y).ravel())
        if len(y) != n_rows:
            raise ValueError(f"X has {n_rows} rows but y has {len(y)}.")
        if y.isna().any():
            raise ValueError("The target contains missing values; drop those rows before fitting.")
        return y.to_numpy()
