"""Comparison tab: ID3 against C4.5 on the same split, across folds and depths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from arbor import charts, metrics
from arbor.experiment import ALGORITHMS, C45, CART, ID3, Experiment
from ui import cache
from ui.components import ROW_HEIGHT, figure, fmt_seconds, section, table_height, verdict


def render(frame: pd.DataFrame, experiment: Experiment) -> None:
    scores = experiment.scores("macro")
    section(
        "Head to head",
        eyebrow="Same data, same split",
        description="Both trees were trained on identical training rows and scored on identical test rows. "
        "Precision, recall and F1 are macro averages.",
        first=True,
    )
    verdict(_verdict(experiment, scores))
    figure(charts.metric_comparison(scores))

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        section("Scorecard")
        card = _scorecard(experiment, scores)
        st.dataframe(card, hide_index=True, width="stretch", height=table_height(len(card), 600), row_height=ROW_HEIGHT)
    with right:
        section("What the trees rely on", description="Share of total gain contributed by each attribute.")
        importance = pd.DataFrame(
            {name: run.model.feature_importances_ for name, run in experiment.runs.items()},
            index=experiment.runs[ID3].model.feature_names_in_,
        )
        figure(charts.feature_importance(importance))

    _cross_validation(frame, experiment)
    _depth_sweep(frame, experiment)
    _reference(frame, experiment, scores)


def _verdict(experiment: Experiment, scores: pd.DataFrame) -> str:
    id3, c45 = scores.loc[ID3], scores.loc[C45]
    leaves = {name: run.model.n_leaves_ for name, run in experiment.runs.items()}
    diff = c45["accuracy"] - id3["accuracy"]
    if abs(diff) < 0.0005:
        text = f"ID3 and C4.5 are level on accuracy at <b>{id3['accuracy']:.3f}</b>."
    else:
        winner, loser = (C45, ID3) if diff > 0 else (ID3, C45)
        text = (
            f"<b>{winner}</b> is more accurate on this split: {scores.loc[winner, 'accuracy']:.3f} against "
            f"{scores.loc[loser, 'accuracy']:.3f} ({abs(diff) * 100:.1f} percentage points)."
        )
    if leaves[C45] != leaves[ID3]:
        smaller, larger = (C45, ID3) if leaves[C45] < leaves[ID3] else (ID3, C45)
        text += f" {smaller} needs {leaves[smaller]} leaves where {larger} uses {leaves[larger]}."
    text += f" Macro F1: ID3 {id3['f1']:.3f}, C4.5 {c45['f1']:.3f}."
    n_test = len(experiment.y_test)
    if experiment.holdout and n_test < 40:
        text += (
            f" <span style='color:var(--arbor-ink-2)'>With {n_test} test rows each case moves accuracy by "
            f"{100 / n_test:.1f} points; the cross-validation below is the steadier comparison.</span>"
        )
    elif not experiment.holdout:
        text += " <span style='color:var(--arbor-ink-2)'>Scored on training data, so this mostly measures fit, not generalisation.</span>"
    return text


def _scorecard(experiment: Experiment, scores: pd.DataFrame) -> pd.DataFrame:
    runs = experiment.runs
    rows = []

    def add(label, values, higher_is_better, fmt):
        a, b = values[ID3], values[C45]
        if np.isclose(a, b):
            better = "tie"
        else:
            better = ID3 if (a > b) == higher_is_better else C45
        rows.append({"Measure": label, ID3: fmt(a), C45: fmt(b), "Better": better})

    for metric in metrics.METRICS:
        add(metrics.METRIC_LABELS[metric], scores[metric], True, lambda v: f"{v:.3f}")
    train = {name: run.train_accuracy for name, run in runs.items()}
    add("Training accuracy", train, True, lambda v: f"{v:.3f}")
    gap = {name: train[name] - scores.loc[name, "accuracy"] for name in runs}
    add("Train − test gap", gap, False, lambda v: f"{v:+.3f}")
    add("Depth", {n: r.model.depth_ for n, r in runs.items()}, False, str)
    add("Leaves", {n: r.model.n_leaves_ for n, r in runs.items()}, False, str)
    add("Nodes", {n: r.model.n_nodes_ for n, r in runs.items()}, False, str)
    add("Fit time (tree)", {n: r.model.fit_time_ for n, r in runs.items()}, False, fmt_seconds)
    add("Predict time (test set)", {n: r.predict_seconds for n, r in runs.items()}, False, fmt_seconds)
    return pd.DataFrame(rows)


@st.fragment
def _cross_validation(frame: pd.DataFrame, experiment: Experiment) -> None:
    section(
        "k-fold cross-validation",
        eyebrow="Stability",
        description="Every row is used for testing exactly once. Preprocessing is refitted inside each fold, "
        "so no fold sees another's statistics.",
    )
    n_rows = len(experiment.y_train) + (len(experiment.y_test) if experiment.holdout else 0)
    max_folds = int(min(10, n_rows))
    if max_folds < 3:
        st.info("Too few rows for cross-validation.")
        return
    controls = st.columns([1.3, 1, 1, 2], vertical_alignment="bottom", gap="medium")
    run_cv = controls[0].toggle("Run cross-validation", value=n_rows <= 3000, key="cv_run")
    folds = controls[1].slider("Folds", 3, max_folds, value=min(5, max_folds), key=f"cv_folds::{max_folds}")
    metric = controls[2].selectbox("Metric", metrics.METRICS, format_func=metrics.METRIC_LABELS.get, key="cv_metric")
    if not run_cv:
        st.caption("Switched off by default for large files; turn it on to run.")
        return

    cv = cache.cross_validation(frame, experiment.config, folds, "macro")
    figure(charts.cv_scores(cv, metric))

    wide = cv.pivot(index="fold", columns="algorithm", values=metric)
    wins = int((wide[C45] > wide[ID3] + 1e-12).sum())
    losses = int((wide[ID3] > wide[C45] + 1e-12).sum())
    summary = cv.groupby("algorithm")[list(metrics.METRICS)].agg(["mean", "std"])
    table = pd.DataFrame(
        {
            metrics.METRIC_LABELS[m]: [f"{summary.loc[a, (m, 'mean')]:.3f} ± {summary.loc[a, (m, 'std')]:.3f}" for a in ALGORITHMS]
            for m in metrics.METRICS
        },
        index=list(ALGORITHMS),
    )
    st.dataframe(table, width="stretch")
    st.caption(
        f"{metrics.METRIC_LABELS[metric]}: C4.5 beat ID3 on {wins} of {folds} folds, lost on {losses}"
        f"{'' if wins + losses == folds else f', tied on {folds - wins - losses}'}. Mean ± standard deviation across folds."
    )


@st.fragment
def _depth_sweep(frame: pd.DataFrame, experiment: Experiment) -> None:
    section(
        "Depth limit sweep",
        eyebrow="Overfitting",
        description="Both trees retrained with a maximum depth of 1, 2, 3… on the same split. Where training "
        "accuracy keeps climbing but test accuracy stalls or drops, the extra depth is memorising noise.",
    )
    deepest = max(run.model.depth_ for run in experiment.runs.values())
    max_depth = int(min(12, max(2, deepest + 1)))
    n_rows = len(experiment.y_train)
    if not st.toggle("Run depth sweep", value=n_rows <= 3000, key="sweep_run"):
        st.caption("Switched off by default for large files; turn it on to run.")
        return
    sweep = cache.sweep(frame, experiment.config, max_depth)
    figure(charts.depth_curves(sweep))
    if not experiment.holdout:
        st.caption("No test set is held out, so the test curve repeats the training curve.")


def _reference(frame: pd.DataFrame, experiment: Experiment, scores: pd.DataFrame) -> None:
    section(
        "Reference point",
        eyebrow="Optional",
        description="scikit-learn's <code>DecisionTreeClassifier</code> (binary CART splits, entropy criterion) "
        "with C4.5's depth and leaf-size limits, on the same split. Useful as a sanity check, not part of the assignment's comparison.",
    )
    if not st.toggle("Include scikit-learn CART", value=False, key="reference_run"):
        return
    reference = cache.reference(frame, experiment.config)
    run = reference.runs[CART]
    all_scores = pd.concat([scores, reference.scores("macro")])
    rows = []
    for name in list(ALGORITHMS) + [CART]:
        model = experiment.runs[name].model if name in experiment.runs else run.model
        rows.append(
            {
                "Model": name,
                **{metrics.METRIC_LABELS[m]: round(float(all_scores.loc[name, m]), 3) for m in metrics.METRICS},
                "Depth": model.get_depth(),
                "Leaves": model.get_n_leaves(),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("CART grows binary trees and reads categorical attributes as ordinal codes, so its shape differs from both.")
