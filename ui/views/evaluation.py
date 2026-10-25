"""Evaluation tab: metrics, confusion matrices and per-class scores."""

from __future__ import annotations

import streamlit as st

from arbor import charts, metrics
from arbor.experiment import ALGORITHMS, C45, ID3, Experiment
from ui.components import chip, figure, section, stats

DESCRIPTIONS = {
    ID3: "information gain · multiway splits",
    C45: "gain ratio · thresholds · pruning",
}


def evaluation_scope(experiment: Experiment) -> str:
    n = len(experiment.y_test)
    if experiment.holdout:
        return f"Scored on {n:,} held-out test rows the trees never saw during training."
    return f"Scored on the same {n:,} rows used for training (no test set), so these numbers are optimistic."


@st.fragment
def render(experiment: Experiment) -> None:
    head = st.columns([1.2, 3], vertical_alignment="bottom", gap="medium")
    average = head[0].segmented_control(
        "Averaging for precision, recall, F1",
        list(metrics.AVERAGES),
        format_func=metrics.AVERAGES.get,
        default="macro",
        key="eval_average",
        help="**Macro** weighs every class equally. **Weighted** weighs classes by their support.",
    ) or "macro"
    head[1].caption(evaluation_scope(experiment))
    if not experiment.holdout:
        st.warning("Switch on *Hold out a test set* in the sidebar for an honest estimate.", icon=":material/warning:")

    y_true = experiment.y_test.to_numpy()
    columns = st.columns(len(ALGORITHMS), gap="large")
    for algorithm, column in zip(ALGORITHMS, columns):
        run = experiment.runs[algorithm]
        scores = metrics.score(y_true, run.y_pred, average)
        correct = int((run.y_pred == y_true).sum())
        label = metrics.AVERAGES[average].lower()
        with column:
            chip(algorithm, DESCRIPTIONS[algorithm])
            stats(
                [
                    ("Accuracy", f"{scores['accuracy']:.3f}", f"{correct:,} of {len(y_true):,} correct"),
                    ("Precision", f"{scores['precision']:.3f}", f"{label} average"),
                    ("Recall", f"{scores['recall']:.3f}", f"{label} average"),
                    ("F1-score", f"{scores['f1']:.3f}", f"{label} average"),
                ]
            )
            section("Confusion matrix", description="Counts; shading shows the share of each actual class.")
            cm = metrics.confusion(y_true, run.y_pred, experiment.labels)
            figure(charts.confusion_matrix(cm, experiment.labels))
            section("Per class")
            table = metrics.per_class(y_true, run.y_pred, experiment.labels)
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "class": st.column_config.TextColumn("Class"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                    "support": st.column_config.NumberColumn("Support", format="%d"),
                },
            )
