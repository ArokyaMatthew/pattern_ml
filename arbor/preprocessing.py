"""Loading, profiling and preprocessing of tabular classification data.

Two layers live here:

* Dataset-level helpers (``read_csv``, ``profile_columns``,
  ``suggest_target``...) that run before any train/test split exists.
* ``TreePreprocessor``, a scikit-learn transformer fitted on the training
  split only, so imputation values and bin edges never see test data.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted

from arbor.tree import format_number

NUMERIC = "numeric"
CATEGORICAL = "categorical"

#: Cell values read as "missing". ``?`` is the UCI repository convention.
NA_TOKENS = ["?", "", "NA", "N/A", "n/a", "na", "null", "NULL", "None", "none", "nan", "NaN"]

BINNING_STRATEGIES = {
    "quantile": "Equal frequency",
    "uniform": "Equal width",
    "kmeans": "k-means",
}

_TARGET_HINTS = (
    "class", "target", "label", "y", "outcome", "play", "species",
    "diagnosis", "risk", "survived", "decision", "result",
)
_ID_NAMES = {"id", "index", "row", "rowid", "row_id", "no", "sno", "s.no", "serial", "unnamed: 0"}


# -- loading ---------------------------------------------------------------
def read_csv(source: str | Path | bytes | io.IOBase) -> pd.DataFrame:
    """Read a CSV with delimiter and encoding detection, then tidy it.

    Rows are numbered from 1 so the UI can refer to "row 12" the way a
    spreadsheet would.
    """
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read()

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text.strip():
        raise ValueError("The file is empty.")

    frame = pd.read_csv(
        io.StringIO(text),
        sep=_detect_delimiter(text),
        na_values=NA_TOKENS,
        keep_default_na=True,
        skipinitialspace=True,
    )
    frame = tidy_frame(frame)
    if frame.shape[1] < 2:
        raise ValueError("A classification dataset needs at least two columns (features and a target).")
    if frame.empty:
        raise ValueError("The file has a header but no data rows.")
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="row")
    return frame


def _detect_delimiter(text: str) -> str:
    """Pick the delimiter that splits the first lines most consistently."""
    lines = [line for line in text.splitlines()[:20] if line.strip()]
    best, best_score = ",", 0
    for delimiter in (",", ";", "\t", "|"):
        counts = [line.count(delimiter) for line in lines]
        if not counts or counts[0] == 0:
            continue
        consistent = sum(count == counts[0] for count in counts)
        score = consistent * 100 + counts[0]
        if score > best_score:
            best, best_score = delimiter, score
    return best


def tidy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip names and text cells, drop empty rows/columns, fix numeric text."""
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for name in frame.columns:
        column = frame[name]
        if is_text(column):
            stripped = column.astype(object).where(column.isna(), column.astype(str).str.strip())
            stripped = stripped.replace("", np.nan)
            as_number = pd.to_numeric(stripped, errors="coerce")
            if stripped.notna().sum() and as_number.notna().sum() == stripped.notna().sum():
                frame[name] = as_number
            else:
                frame[name] = stripped
    return frame


def is_text(column: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(column)
        or pd.api.types.is_bool_dtype(column)
        or pd.api.types.is_datetime64_any_dtype(column)
        or isinstance(column.dtype, pd.CategoricalDtype)
    )


def to_text(column: pd.Series) -> pd.Series:
    """Categorical view of a column: values as strings, missing kept missing.

    Whole-number floats (common when an integer column has gaps) lose the
    trailing ``.0`` so ``3.0`` and ``3`` are the same category.
    """
    if pd.api.types.is_float_dtype(column):
        known = column.dropna()
        if len(known) and np.all(np.mod(known.to_numpy(), 1) == 0):
            column = column.astype("Int64")
    values = column.astype(object)
    return values.where(values.isna(), values.astype(str)).astype(object)


# -- profiling -------------------------------------------------------------
def infer_kind(column: pd.Series) -> str:
    """Default role of a column: numeric unless text, boolean or binary."""
    if pd.api.types.is_bool_dtype(column) or is_text(column):
        return CATEGORICAL
    if column.nunique(dropna=True) <= 2:
        return CATEGORICAL
    return NUMERIC


def profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in frame.columns:
        column = frame[name]
        distinct = column.dropna().unique()
        examples = ", ".join(_short(v) for v in distinct[:3])
        rows.append(
            {
                "column": name,
                "dtype": str(column.dtype),
                "kind": infer_kind(column),
                "missing": int(column.isna().sum()),
                "missing_pct": float(column.isna().mean() * 100),
                "unique": int(column.nunique(dropna=True)),
                "examples": examples,
            }
        )
    return pd.DataFrame(rows)


def _short(value, width: int = 18) -> str:
    text = format_number(value) if isinstance(value, (int, float, np.number)) and not isinstance(value, bool) else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def suggest_target(frame: pd.DataFrame) -> str:
    """Guess the class column: a conventional name, else the last column."""
    lowered = {str(c).lower(): c for c in frame.columns}
    for hint in _TARGET_HINTS:
        if hint in lowered and target_problem(frame[lowered[hint]]) is None:
            return lowered[hint]
    for name in reversed(frame.columns):
        if target_problem(frame[name]) is None:
            return name
    return frame.columns[-1]


def identifier_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that identify rows rather than describe them (ids, names)."""
    found = []
    n = len(frame)
    for name in frame.columns:
        lowered = str(name).lower()
        column = frame[name]
        if lowered in _ID_NAMES or lowered.endswith("_id"):
            found.append(name)
        elif n >= 20 and column.nunique(dropna=True) == n:
            if is_text(column):
                found.append(name)
            elif pd.api.types.is_integer_dtype(column) and (column.diff().dropna() == 1).all():
                found.append(name)
    return found


def target_problem(column: pd.Series) -> str | None:
    """Why a column cannot serve as a classification target, if it can't."""
    known = column.dropna()
    n_classes = known.nunique()
    if n_classes < 2:
        return "has fewer than two distinct values"
    if pd.api.types.is_float_dtype(known) and np.any(np.mod(known.to_numpy(), 1) != 0) and n_classes > 20:
        return "looks continuous (a regression target), not a set of classes"
    if n_classes > 50:
        return f"has {n_classes} distinct values, too many classes for this tool"
    if len(known) > 20 and n_classes > 0.5 * len(known):
        return "takes a different value on most rows, so it looks like an identifier"
    return None


# -- discretisation ----------------------------------------------------------
def bin_edges(values: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Inner cut points that split ``values`` into at most ``n_bins`` intervals."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.array([])
    distinct = np.unique(values)
    if distinct.size <= n_bins:
        edges = (distinct[:-1] + distinct[1:]) / 2
    elif strategy == "uniform":
        edges = np.linspace(values.min(), values.max(), n_bins + 1)[1:-1]
    elif strategy == "quantile":
        edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    elif strategy == "kmeans":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans").fit(values.reshape(-1, 1))
        edges = fitted.bin_edges_[0][1:-1]
    else:
        raise ValueError(f"Unknown binning strategy {strategy!r}.")
    edges = np.unique(edges)
    return edges[(edges >= values.min()) & (edges < values.max())]


def interval_labels(edges: np.ndarray) -> list[str]:
    """Readable, unique labels for the intervals defined by ``edges``.

    Intervals are right-closed, and the outer ones are open-ended so values
    outside the training range still fall into a bin.
    """
    if len(edges) == 0:
        return ["any value"]
    labels: list[str] = []
    for digits in range(3, 10):
        def fmt(x: float) -> str:
            return format_number(x, digits)

        labels = (
            [f"≤ {fmt(edges[0])}"]
            + [f"({fmt(a)}, {fmt(b)}]" for a, b in zip(edges[:-1], edges[1:])]
            + [f"> {fmt(edges[-1])}"]
        )
        if len(set(labels)) == len(labels):
            return labels
    return [f"bin {i + 1}: {label}" for i, label in enumerate(labels)]


# -- the transformer ---------------------------------------------------------
class TreePreprocessor(TransformerMixin, BaseEstimator):
    """Type coercion, imputation and (for ID3) discretisation.

    Everything learned from data (fill values, bin edges) is learned in
    ``fit`` on the training split and replayed unchanged in ``transform``.

    Parameters
    ----------
    numeric_features, categorical_features : tuple of str
        Column roles. Numeric columns are coerced to float, categorical
        columns to text.
    impute : bool
        Fill missing values: numeric with the median or mean, categorical
        with the mode. Turn off to let C4.5 handle unknown values natively.
    numeric_strategy : {"median", "mean"}
    discretize : bool
        Replace each numeric column by an ordered categorical of intervals.
    n_bins : int
        Upper bound on intervals per numeric column.
    binning : {"quantile", "uniform", "kmeans"}
    """

    def __init__(
        self,
        numeric_features: tuple[str, ...] = (),
        categorical_features: tuple[str, ...] = (),
        impute: bool = True,
        numeric_strategy: str = "median",
        discretize: bool = False,
        n_bins: int = 4,
        binning: str = "quantile",
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.impute = impute
        self.numeric_strategy = numeric_strategy
        self.discretize = discretize
        self.n_bins = n_bins
        self.binning = binning

    def fit(self, X: pd.DataFrame, y=None):
        X = self._coerce(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.missing_counts_ = {name: int(X[name].isna().sum()) for name in X.columns}

        self.fill_values_: dict[str, object] = {}
        if self.impute:
            for name in self.numeric_features:
                statistic = X[name].median() if self.numeric_strategy == "median" else X[name].mean()
                self.fill_values_[name] = float(statistic) if pd.notna(statistic) else 0.0
            for name in self.categorical_features:
                mode = X[name].mode(dropna=True)
                self.fill_values_[name] = mode.iloc[0] if len(mode) else "(missing)"

        self.bin_edges_: dict[str, np.ndarray] = {}
        self.bin_labels_: dict[str, list[str]] = {}
        if self.discretize:
            for name in self.numeric_features:
                values = X[name]
                if self.impute:
                    values = values.fillna(self.fill_values_[name])
                edges = bin_edges(values.to_numpy(dtype=float), self.n_bins, self.binning)
                self.bin_edges_[name] = edges
                self.bin_labels_[name] = interval_labels(edges)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fill_values_")
        X = self._coerce(X)
        for name, value in self.fill_values_.items():
            X[name] = X[name].fillna(value)
        for name, edges in self.bin_edges_.items():
            values = X[name].to_numpy(dtype=float)
            codes = np.digitize(values, edges, right=True)
            codes[np.isnan(values)] = -1
            X[name] = pd.Categorical.from_codes(codes, categories=self.bin_labels_[name], ordered=True)
        return X

    def _coerce(self, X: pd.DataFrame) -> pd.DataFrame:
        wanted = set(self.numeric_features) | set(self.categorical_features)
        absent = sorted(wanted - set(X.columns))
        if absent:
            raise ValueError(f"Input is missing columns: {absent}")
        numeric = set(self.numeric_features)
        out = {}
        for name in X.columns:
            if name not in wanted:
                continue
            if name in numeric:
                out[name] = pd.to_numeric(X[name], errors="coerce").astype(float)
            else:
                out[name] = to_text(X[name])
        return pd.DataFrame(out, index=X.index)

    def binning_table(self) -> pd.DataFrame:
        """One row per discretised column: its intervals and cut points."""
        check_is_fitted(self, "fill_values_")
        rows = [
            {
                "attribute": name,
                "intervals": len(self.bin_labels_[name]),
                "cut points": ", ".join(format_number(e) for e in edges) or "none",
                "labels": "  ·  ".join(self.bin_labels_[name]),
            }
            for name, edges in self.bin_edges_.items()
        ]
        return pd.DataFrame(rows, columns=["attribute", "intervals", "cut points", "labels"])
