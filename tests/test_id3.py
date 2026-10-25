import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from arbor import ID3Classifier


def fit(frame: pd.DataFrame, target: str = "play", **params) -> ID3Classifier:
    return ID3Classifier(**params).fit(frame.drop(columns=target), frame[target])


def test_root_scores_match_mitchell(play_tennis):
    model = fit(play_tennis)
    gains = {c.feature: c.gain for c in model.root_.candidates}
    assert gains["outlook"] == pytest.approx(0.2467, abs=1e-4)
    assert gains["humidity"] == pytest.approx(0.1518, abs=1e-4)
    assert gains["wind"] == pytest.approx(0.0481, abs=1e-4)
    assert gains["temperature"] == pytest.approx(0.0292, abs=1e-4)
    assert model.root_.feature == "outlook"


def test_learns_the_textbook_tree(play_tennis):
    model = fit(play_tennis)
    assert model.export_text() == (
        "outlook = overcast: yes (4/0)\n"
        "outlook = rain\n"
        "|   wind = strong: no (2/0)\n"
        "|   wind = weak: yes (3/0)\n"
        "outlook = sunny\n"
        "|   humidity = high: no (3/0)\n"
        "|   humidity = normal: yes (2/0)"
    )
    assert (model.get_depth(), model.get_n_leaves(), model.n_nodes_) == (2, 5, 8)
    assert (model.predict(play_tennis.drop(columns="play")) == play_tennis["play"]).all()


def test_attribute_is_used_once_per_path(play_tennis):
    model = fit(play_tennis)
    for rule in model.rules():
        features = [condition.split(" ")[0] for condition in rule.conditions]
        assert len(features) == len(set(features))


def test_max_depth_and_min_gain_limit_growth(play_tennis):
    assert fit(play_tennis, max_depth=1).get_depth() == 1
    # Outlook gains 0.247 bits; nothing is worth more than 0.3.
    stump = fit(play_tennis, min_gain=0.3)
    assert stump.root_.is_leaf
    assert stump.predict(play_tennis.drop(columns="play").head(1))[0] == "yes"


def test_unseen_value_falls_back_to_node_distribution(play_tennis):
    model = fit(play_tennis)
    case = pd.DataFrame({"outlook": ["foggy"], "temperature": ["hot"], "humidity": ["high"], "wind": ["weak"]})
    proba = model.predict_proba(case)[0]
    assert proba == pytest.approx([5 / 14, 9 / 14])
    path = model.decision_path(case)
    assert path[0][0] is model.root_ and path[0][1] is None


def test_missing_values_become_their_own_category():
    X = pd.DataFrame({"colour": ["red", "red", None, None, "blue", "blue"]})
    y = ["a", "a", "b", "b", "c", "c"]
    model = ID3Classifier().fit(X, y)
    assert "(missing)" in model.categories_["colour"]
    assert list(model.predict(pd.DataFrame({"colour": [None, "red"]}))) == ["b", "a"]


def test_rules_cover_every_leaf(play_tennis):
    model = fit(play_tennis)
    rules = model.rules()
    assert len(rules) == model.n_leaves_
    assert sum(r.support for r in rules) == len(play_tennis)
    assert rules[0].as_text("play") == "IF outlook = overcast THEN play = yes"


def test_feature_importances_sum_to_one(play_tennis):
    model = fit(play_tennis)
    importance = dict(zip(model.feature_names_in_, model.feature_importances_))
    assert sum(importance.values()) == pytest.approx(1.0)
    # Weighted gain: outlook 14/14 * 0.247; humidity and wind 5/14 * 0.971 each.
    assert importance["humidity"] == pytest.approx(importance["wind"])
    assert importance["humidity"] > importance["outlook"] > 0
    assert importance["temperature"] == 0


def test_works_with_scikit_learn_tooling(play_tennis):
    X, y = play_tennis.drop(columns="play"), play_tennis["play"]
    copy = clone(ID3Classifier(max_depth=3))
    assert copy.get_params()["max_depth"] == 3
    scores = cross_val_score(copy, X, y, cv=2)
    assert len(scores) == 2 and all(0 <= s <= 1 for s in scores)


@pytest.mark.parametrize("params", [{"max_depth": 0}, {"min_samples_split": 1}, {"min_gain": -1}])
def test_rejects_invalid_parameters(play_tennis, params):
    with pytest.raises(ValueError):
        fit(play_tennis, **params)
