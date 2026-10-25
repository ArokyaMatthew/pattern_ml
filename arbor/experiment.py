"""End-to-end experiments: build pipelines, train, evaluate, compare.

Each algorithm gets its own scikit-learn ``Pipeline`` of
``TreePreprocessor`` -> tree, because the two need different inputs: ID3
wants every attribute discretised and complete, C4.5 takes raw numbers and
can keep missing values. Wrapping both in pipelines means cross-validation
refits the preprocessing inside every fold, so no fold leaks into another.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from arbor import metrics
from arbor.c45 import C45Classifier
from arbor.id3 import ID3Classifier
from arbor.preprocessing import TreePreprocessor, to_text

ID3 = "ID3"
C45 = "C4.5"
CART = "CART (scikit-learn)"
ALGORITHMS = (ID3, C45)

MISSING_STRATEGIES = {
    "impute": "Impute (median / mode)",
    "drop": "Drop incomplete rows",
    "native": "C4.5 native, impute for ID3",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that determines an experiment's outcome.

    Frozen and hashable so the UI can cache results on it.
    """

    target: str
    features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    # preprocessing
    missing_strategy: str = "impute"
    numeric_impute: str = "median"
    n_bins: int = 4
    binning: str = "quantile"
    drop_duplicates: bool = False
    # validation
    test_size: float = 0.3
    stratify: bool = True
    seed: int = 42
    # ID3
    id3_max_depth: int | None = None
    id3_min_samples_split: int = 2
    id3_min_gain: float = 0.0
    # C4.5
    c45_max_depth: int | None = None
    c45_min_samples_leaf: int = 2
    c45_prune: bool = True
    c45_confidence: float = 0.25

    @property
    def categorical_features(self) -> tuple[str, ...]:
        numeric = set(self.numeric_features)
        return tuple(f for f in self.features if f not in numeric)

    @property
    def holdout(self) -> bool:
        return self.test_size > 0


def build_pipeline(algorithm: str, cfg: ExperimentConfig) -> Pipeline:
    numeric, categorical = cfg.numeric_features, cfg.categorical_features
    if algorithm == ID3:
        prep = TreePreprocessor(
            numeric, categorical, impute=True, numeric_strategy=cfg.numeric_impute,
            discretize=True, n_bins=cfg.n_bins, binning=cfg.binning,
        )
        model = ID3Classifier(
            max_depth=cfg.id3_max_depth,
            min_samples_split=cfg.id3_min_samples_split,
            min_gain=cfg.id3_min_gain,
        )
        return Pipeline([("prep", prep), ("tree", model)])
    if algorithm == C45:
        prep = TreePreprocessor(
            numeric, categorical, impute=cfg.missing_strategy != "native", numeric_strategy=cfg.numeric_impute,
        )
        model = C45Classifier(
            max_depth=cfg.c45_max_depth,
            min_samples_leaf=cfg.c45_min_samples_leaf,
            prune=cfg.c45_prune,
            confidence=cfg.c45_confidence,
        )
        return Pipeline([("prep", prep), ("tree", model)])
    if algorithm == CART:
        # Reference point only: scikit-learn's binary CART with the entropy
        # criterion, given the same depth and leaf-size limits as C4.5.
        prep = TreePreprocessor(numeric, categorical, impute=True, numeric_strategy=cfg.numeric_impute)
        encode = ColumnTransformer(
            [("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(categorical))],
            remainder="passthrough",
        )
        model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=cfg.c45_max_depth,
            min_samples_leaf=cfg.c45_min_samples_leaf,
            random_state=cfg.seed,
        )
        return Pipeline([("prep", prep), ("encode", encode), ("tree", model)])
    raise ValueError(f"Unknown algorithm {algorithm!r}.")


def prepare_dataset(frame: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Select columns and apply the row-level cleaning steps.

    Returns features, target (as text labels) and notes describing what was
    done, for the UI's preprocessing log.
    """
    notes: list[str] = []
    data = frame[list(cfg.features) + [cfg.target]].copy()

    no_target = data[cfg.target].isna()
    if no_target.any():
        data = data[~no_target]
        notes.append(f"Dropped {int(no_target.sum())} rows with no target value.")
    if cfg.drop_duplicates:
        before = len(data)
        data = data.drop_duplicates()
        if len(data) < before:
            notes.append(f"Dropped {before - len(data)} duplicate rows.")
    if cfg.missing_strategy == "drop":
        before = len(data)
        data = data.dropna(subset=list(cfg.features))
        if len(data) < before:
            notes.append(f"Dropped {before - len(data)} rows with a missing attribute value.")

    y = to_text(data[cfg.target]).astype(str).rename(cfg.target)
    return data[list(cfg.features)], y, notes


@dataclass
class ModelRun:
    algorithm: str
    pipeline: Pipeline
    y_pred: np.ndarray
    proba: np.ndarray
    train_accuracy: float
    fit_seconds: float
    predict_seconds: float

    @property
    def model(self):
        return self.pipeline.named_steps["tree"]

    @property
    def prep(self) -> TreePreprocessor:
        return self.pipeline.named_steps["prep"]


@dataclass
class Experiment:
    config: ExperimentConfig
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    labels: list[str]
    runs: dict[str, ModelRun]
    notes: list[str] = field(default_factory=list)

    @property
    def holdout(self) -> bool:
        return self.config.holdout

    def scores(self, average: str = "macro") -> pd.DataFrame:
        """Algorithms as rows, metrics as columns."""
        rows = {name: metrics.score(self.y_test, run.y_pred, average) for name, run in self.runs.items()}
        return pd.DataFrame.from_dict(rows, orient="index")[list(metrics.METRICS)]


def split_data(X: pd.DataFrame, y: pd.Series, cfg: ExperimentConfig):
    """Hold-out split, stratified when every class can be represented."""
    notes: list[str] = []
    if not cfg.holdout:
        notes.append("No test set: models are evaluated on the training data (resubstitution), which is optimistic.")
        return X, X, y, y, notes

    n_test = math.ceil(cfg.test_size * len(y))
    n_classes = y.nunique()
    stratify = None
    if cfg.stratify:
        if y.value_counts().min() >= 2 and n_classes <= n_test <= len(y) - n_classes:
            stratify = y
        else:
            notes.append("Stratified split was not possible (a class has too few rows); used a random split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify
    )
    kind = "Stratified" if stratify is not None else "Random"
    notes.append(f"{kind} hold-out split: {len(y_train)} training rows, {len(y_test)} test rows.")
    return X_train, X_test, y_train, y_test, notes


def run_experiment(frame: pd.DataFrame, cfg: ExperimentConfig, algorithms=ALGORITHMS) -> Experiment:
    X, y, notes = prepare_dataset(frame, cfg)
    if y.nunique() < 2:
        raise ValueError("After preprocessing the target has fewer than two classes.")
    X_train, X_test, y_train, y_test, split_notes = split_data(X, y, cfg)
    notes += split_notes

    runs: dict[str, ModelRun] = {}
    for algorithm in algorithms:
        pipeline = build_pipeline(algorithm, cfg)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        proba = pipeline.predict_proba(X_test)
        y_pred = pipeline.classes_[np.argmax(proba, axis=1)]
        predict_seconds = time.perf_counter() - started
        train_accuracy = float(np.mean(pipeline.predict(X_train) == y_train.to_numpy()))
        runs[algorithm] = ModelRun(algorithm, pipeline, y_pred, proba, train_accuracy, fit_seconds, predict_seconds)

    labels = sorted(set(y_train) | set(y_test))
    return Experiment(cfg, X_train, X_test, y_train, y_test, labels, runs, notes)


def cross_validate_models(
    frame: pd.DataFrame, cfg: ExperimentConfig, folds: int = 5, average: str = "macro", algorithms=ALGORITHMS
) -> pd.DataFrame:
    """k-fold scores per fold and algorithm (long format)."""
    X, y, _ = prepare_dataset(frame, cfg)
    if y.value_counts().min() >= folds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
    rows = []
    for algorithm in algorithms:
        result = cross_validate(build_pipeline(algorithm, cfg), X, y, cv=splitter, scoring=metrics.scorers(average))
        for fold in range(folds):
            row = {"algorithm": algorithm, "fold": fold + 1, "fit_seconds": result["fit_time"][fold]}
            row.update({name: result[f"test_{name}"][fold] for name in metrics.METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def depth_sweep(experiment: Experiment, max_depth: int, algorithms=ALGORITHMS) -> pd.DataFrame:
    """Train and test accuracy as the depth limit grows from 1 to ``max_depth``."""
    rows = []
    y_train, y_test = experiment.y_train.to_numpy(), experiment.y_test.to_numpy()
    for depth in range(1, max_depth + 1):
        cfg = replace(experiment.config, id3_max_depth=depth, c45_max_depth=depth)
        for algorithm in algorithms:
            pipeline = clone(build_pipeline(algorithm, cfg)).fit(experiment.X_train, experiment.y_train)
            rows.append(
                {
                    "algorithm": algorithm,
                    "depth": depth,
                    "train": float(np.mean(pipeline.predict(experiment.X_train) == y_train)),
                    "test": float(np.mean(pipeline.predict(experiment.X_test) == y_test)),
                    "leaves": int(pipeline.named_steps["tree"].get_n_leaves()),
                }
            )
    return pd.DataFrame(rows)
