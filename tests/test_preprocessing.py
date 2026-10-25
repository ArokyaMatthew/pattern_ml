import numpy as np
import pandas as pd
import pytest

from arbor.preprocessing import (
    TreePreprocessor,
    bin_edges,
    identifier_columns,
    infer_kind,
    interval_labels,
    read_csv,
    suggest_target,
    target_problem,
)


def test_read_csv_detects_semicolons_and_question_marks():
    raw = b"colour ; size ; label\nred;1.5;a\n?;2,5;b\nblue; 3 ;a\n"
    frame = read_csv(raw)
    assert list(frame.columns) == ["colour", "size", "label"]
    assert frame["colour"].isna().sum() == 1
    assert frame.index[0] == 1


def test_read_csv_turns_numeric_text_into_numbers():
    frame = read_csv(b"a,b\n 1 ,x\n2,y\n")
    assert pd.api.types.is_numeric_dtype(frame["a"])


def test_read_csv_rejects_single_column_files():
    with pytest.raises(ValueError):
        read_csv(b"only\n1\n2\n")


def test_infer_kind():
    assert infer_kind(pd.Series([1.0, 2.5, 3.1, 4.2])) == "numeric"
    assert infer_kind(pd.Series(["a", "b", "a"])) == "categorical"
    assert infer_kind(pd.Series([0, 1, 1, 0])) == "categorical"


def test_suggest_target_and_identifiers(play_tennis, iris):
    assert suggest_target(play_tennis) == "play"
    assert suggest_target(iris) == "species"
    frame = pd.DataFrame({"id": range(30), "x": np.arange(30) % 3, "y": ["a", "b"] * 15})
    assert identifier_columns(frame) == ["id"]


def test_target_problem():
    assert target_problem(pd.Series(["a", "a"])) is not None
    assert target_problem(pd.Series(np.linspace(0, 1, 100))) is not None
    assert target_problem(pd.Series(["a", "b"] * 10)) is None


def test_bin_edges_strategies():
    values = np.arange(1, 101, dtype=float)
    assert bin_edges(values, 4, "uniform") == pytest.approx([25.75, 50.5, 75.25])
    assert bin_edges(values, 4, "quantile") == pytest.approx([25.75, 50.5, 75.25])
    assert len(bin_edges(values, 4, "kmeans")) == 3
    # Fewer distinct values than bins: one interval per value.
    assert bin_edges(np.array([1.0, 1.0, 2.0, 3.0]), 5, "quantile") == pytest.approx([1.5, 2.5])


def test_interval_labels_are_open_ended_and_unique():
    labels = interval_labels(np.array([1.5, 2.5]))
    assert labels == ["≤ 1.5", "(1.5, 2.5]", "> 2.5"]
    assert len(set(interval_labels(np.array([1.00001, 1.00002])))) == 3


def test_preprocessor_learns_from_training_data_only():
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "c": ["a", None, "a", "b"]})
    test = pd.DataFrame({"x": [np.nan, 100.0], "c": [None, "z"]})
    prep = TreePreprocessor(("x",), ("c",), impute=True, discretize=True, n_bins=2).fit(train)
    assert prep.fill_values_ == {"x": 2.0, "c": "a"}
    out = prep.transform(test)
    assert out.loc[0, "c"] == "a"
    # A value far outside the training range still lands in the top bin.
    assert out.loc[1, "x"] == prep.bin_labels_["x"][-1]
    assert isinstance(out["x"].dtype, pd.CategoricalDtype)


def test_preprocessor_can_keep_missing_values():
    frame = pd.DataFrame({"x": [1.0, np.nan], "c": ["a", None]})
    out = TreePreprocessor(("x",), ("c",), impute=False).fit(frame).transform(frame)
    assert out.isna().sum().sum() == 2


def test_categorical_numbers_are_normalised_to_text():
    frame = pd.DataFrame({"code": [1.0, 2.0, np.nan]})
    out = TreePreprocessor((), ("code",), impute=False).fit(frame).transform(frame)
    assert out["code"].tolist()[:2] == ["1", "2"]
