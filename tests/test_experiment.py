import numpy as np
import pytest

from arbor import charts, metrics
from arbor.experiment import (
    ALGORITHMS,
    C45,
    CART,
    ID3,
    ExperimentConfig,
    cross_validate_models,
    depth_sweep,
    run_experiment,
)
from arbor.graph import tree_graph


def iris_config(**overrides) -> ExperimentConfig:
    features = ("sepal_length", "sepal_width", "petal_length", "petal_width")
    return ExperimentConfig(target="species", features=features, numeric_features=features, **overrides)


def credit_config(**overrides) -> ExperimentConfig:
    features = ("age", "annual_income", "employment", "credit_history", "housing", "purpose",
                "loan_amount", "term_months", "existing_loans")
    numeric = ("age", "annual_income", "loan_amount", "term_months", "existing_loans")
    return ExperimentConfig(target="risk", features=features, numeric_features=numeric, **overrides)


def test_holdout_experiment_trains_both_algorithms(iris):
    experiment = run_experiment(iris, iris_config())
    assert set(experiment.runs) == set(ALGORITHMS)
    assert len(experiment.y_test) == 45 and len(experiment.y_train) == 105
    scores = experiment.scores()
    assert ((scores >= 0) & (scores <= 1)).all().all()
    assert scores.loc[C45, "accuracy"] > 0.8


def test_experiments_are_reproducible(iris):
    a = run_experiment(iris, iris_config(seed=7)).scores()
    b = run_experiment(iris, iris_config(seed=7)).scores()
    assert a.equals(b)


def test_id3_receives_discretised_complete_data(credit):
    experiment = run_experiment(credit, credit_config(missing_strategy="native"))
    id3_input = experiment.runs[ID3].prep.transform(experiment.X_train)
    c45_input = experiment.runs[C45].prep.transform(experiment.X_train)
    assert id3_input.isna().sum().sum() == 0
    assert all(str(id3_input[c].dtype) == "category" for c in credit_config().numeric_features)
    assert c45_input.isna().sum().sum() > 0


def test_drop_strategy_removes_incomplete_rows(credit):
    experiment = run_experiment(credit, credit_config(missing_strategy="drop", test_size=0.0))
    assert len(experiment.y_train) == len(credit.dropna())
    assert any("Dropped" in note for note in experiment.notes)


def test_cross_validation_scores_every_fold(iris):
    cv = cross_validate_models(iris, iris_config(), folds=4)
    assert len(cv) == 4 * len(ALGORITHMS)
    assert set(cv.columns) >= set(metrics.METRICS)


def test_depth_sweep_covers_each_depth(iris):
    sweep = depth_sweep(run_experiment(iris, iris_config()), max_depth=3)
    assert sorted(sweep["depth"].unique()) == [1, 2, 3]
    assert (sweep.groupby("algorithm")["leaves"].max() > 1).all()


def test_reference_cart_runs_on_mixed_data(credit):
    experiment = run_experiment(credit, credit_config(), algorithms=(CART,))
    assert 0.5 < experiment.scores().loc[CART, "accuracy"] <= 1


def test_metrics_agree_with_hand_counts():
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    scores = metrics.score(y_true, y_pred, "macro")
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert scores["recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics.confusion(y_true, y_pred, ["a", "b"]).tolist() == [[1, 1], [0, 2]]


def test_tree_drawing_collapses_hidden_levels(iris):
    model = run_experiment(iris, iris_config()).runs[ID3].model
    full = tree_graph(model).source
    shallow = tree_graph(model, max_depth=1).source
    assert full.count("->") == model.n_nodes_ - 1
    assert "nodes</FONT>" in shallow and shallow.count("->") < full.count("->")


def test_charts_render(iris):
    experiment = run_experiment(iris, iris_config())
    run = experiment.runs[C45]
    cm = metrics.confusion(experiment.y_test, run.y_pred, experiment.labels)
    for figure in (
        charts.confusion_matrix(cm, experiment.labels),
        charts.metric_comparison(experiment.scores()),
        charts.class_distribution(iris["species"].value_counts(), experiment.labels),
    ):
        assert figure.get_axes()
