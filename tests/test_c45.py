import numpy as np
import pandas as pd
import pytest

from arbor import C45Classifier
from arbor.c45 import added_errors
from arbor.tree import CONTINUOUS, LEFT


def fit(frame: pd.DataFrame, target: str = "play", **params) -> C45Classifier:
    return C45Classifier(**params).fit(frame.drop(columns=target), frame[target])


def test_learns_quinlans_golf_tree(weather):
    # Same structure as C4.5 / Weka J48 on the golf data.
    model = fit(weather, prune=False)
    assert model.export_text() == (
        "outlook = overcast: yes (4/0)\n"
        "outlook = rainy\n"
        "|   windy = False: yes (3/0)\n"
        "|   windy = True: no (2/0)\n"
        "outlook = sunny\n"
        "|   humidity <= 77.5: yes (2/0)\n"
        "|   humidity > 77.5: no (3/0)"
    )


def test_numeric_attributes_get_binary_threshold_splits(weather):
    model = fit(weather, prune=False)
    sunny = model.root_.children["sunny"]
    assert sunny.kind == CONTINUOUS
    assert sunny.threshold == pytest.approx(77.5)
    assert set(sunny.children) == {LEFT, ">"}


def test_gain_ratio_and_average_gain_rule(weather):
    model = fit(weather, prune=False)
    scores = {c.feature: c for c in model.root_.candidates}
    assert scores["outlook"].gain_ratio == pytest.approx(0.1564, abs=1e-4)
    assert scores["outlook"].selected
    # windy has positive gain but less than the average, so it is not admitted.
    assert scores["windy"].note == "below average gain"


def test_threshold_penalty_can_make_gain_negative(weather):
    model = fit(weather, prune=False)
    temperature = next(c for c in model.root_.candidates if c.feature == "temperature")
    assert temperature.gain < 0
    assert temperature.note.startswith("best of")


def test_handles_missing_values_with_fractional_cases(weather):
    damaged = weather.copy()
    damaged.loc[damaged.index[[0, 5, 9]], "humidity"] = np.nan
    damaged.loc[damaged.index[[2]], "outlook"] = np.nan
    model = fit(damaged, prune=False)

    # Fractional weights reach the leaves but the totals are preserved.
    leaf_total = sum(node.n_samples for node in model.nodes_ if node.is_leaf)
    assert leaf_total == pytest.approx(len(damaged))
    assert any(not float(node.n_samples).is_integer() for node in model.nodes_)

    proba = model.predict_proba(damaged.drop(columns="play"))
    assert proba.sum(axis=1) == pytest.approx(np.ones(len(damaged)))


def test_unknown_value_at_prediction_blends_branches(weather):
    model = fit(weather, prune=False)
    case = pd.DataFrame({"outlook": ["sunny"], "temperature": [70.0], "humidity": [np.nan], "windy": ["False"]})
    proba = model.predict_proba(case)[0]
    # Sunny days: 2 of 5 had humidity <= 77.5 (all "yes"), 3 of 5 above (all "no").
    assert dict(zip(model.classes_, proba)) == pytest.approx({"no": 0.6, "yes": 0.4})


def test_added_errors_match_quinlans_tables():
    z_squared = 0.6744897501960817 ** 2
    # Quinlan (1993) p. 41: U25%(0, 6) = 0.206 and U25%(0, 9) = 0.143.
    assert (0 + added_errors(6, 0, 0.25, z_squared)) / 6 == pytest.approx(0.206, abs=1e-3)
    assert (0 + added_errors(9, 0, 0.25, z_squared)) / 9 == pytest.approx(0.143, abs=1e-3)
    # U25%(1, 16) = 0.157, via the normal approximation.
    assert (1 + added_errors(16, 1, 0.25, z_squared)) / 16 == pytest.approx(0.157, abs=3e-3)


def test_pruning_shrinks_the_tree(credit):
    target = "risk"
    full = fit(credit, target, prune=False)
    pruned = fit(credit, target, prune=True, confidence=0.25)
    harder = fit(credit, target, prune=True, confidence=0.05)
    assert full.n_pruned_ == 0
    assert pruned.n_leaves_ < full.n_leaves_
    assert harder.n_leaves_ <= pruned.n_leaves_
    assert pruned.n_nodes_grown_ == full.n_nodes_


def test_min_samples_leaf_is_respected(iris):
    model = C45Classifier(min_samples_leaf=10, prune=False).fit(iris.drop(columns="species"), iris["species"])
    for node in model.nodes_:
        if not node.is_leaf and node.kind == CONTINUOUS:
            sizes = [child.n_samples for child in node.children.values()]
            assert min(sizes) >= 10


def test_continuous_attribute_can_be_split_again(iris):
    model = fit(iris, "species", prune=False, min_samples_leaf=1)
    features_on_paths = [[c.split(" ")[0] for c in rule.conditions] for rule in model.rules()]
    assert any(len(path) != len(set(path)) for path in features_on_paths)


@pytest.mark.parametrize("params", [{"confidence": 0.0}, {"confidence": 0.6}, {"min_samples_leaf": 0}])
def test_rejects_invalid_parameters(weather, params):
    with pytest.raises(ValueError):
        fit(weather, **params)
