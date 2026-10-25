"""Predict tab: classify a hand-entered case, inspect test predictions, score a file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from arbor.experiment import ALGORITHMS, C45, ID3, Experiment
from arbor.graph import tree_graph
from arbor.preprocessing import read_csv
from arbor.style import mix
from arbor.tree import format_number
from ui.components import ROW_HEIGHT, class_legend, prediction_card, section, table_height

UNKNOWN = "(unknown)"
WRONG = mix("#d03b3b", "#ffffff", 0.86)


def render(experiment: Experiment, scope: str) -> None:
    section(
        "Classify a new case",
        eyebrow="Single prediction",
        description="Set the attribute values and both trees answer immediately. Leave a value unknown to see "
        "how each algorithm copes: imputed inputs get the training median or mode, while with "
        "<i>C4.5 native</i> missing values C4.5 follows every branch and blends the leaves.",
        first=True,
    )
    _new_case(experiment, scope)
    _test_predictions(experiment)
    _batch(experiment)


def _typical(experiment: Experiment) -> dict:
    X, numeric = experiment.X_train, set(experiment.config.numeric_features)
    values = {}
    for name in experiment.config.features:
        column = X[name]
        if name in numeric:
            median = pd.to_numeric(column, errors="coerce").median()
            values[name] = float(median) if pd.notna(median) else None
        else:
            mode = column.dropna().astype(str).mode()
            values[name] = mode.iloc[0] if len(mode) else UNKNOWN
    return values


def _decimals(column: pd.Series) -> int:
    values = pd.to_numeric(column, errors="coerce").dropna().to_numpy()
    for digits in range(0, 5):
        if np.allclose(values, np.round(values, digits)):
            return digits
    return 4


@st.fragment
def _new_case(experiment: Experiment, scope: str) -> None:
    cfg = experiment.config
    numeric = set(cfg.numeric_features)
    keys = {name: f"case::{scope}::{name}" for name in cfg.features}
    X_test = experiment.X_test

    def load(values: dict) -> None:
        for name, value in values.items():
            if name in numeric:
                st.session_state[keys[name]] = None if value is None or pd.isna(value) else float(value)
            else:
                st.session_state[keys[name]] = UNKNOWN if value is None or pd.isna(value) else str(value)

    if any(key not in st.session_state for key in keys.values()):
        load(_typical(experiment))

    presets = ["Typical values (median / mode)"] + [f"Row {i}" for i in X_test.index[:300]]

    def apply_preset() -> None:
        choice = st.session_state[f"case_preset::{scope}"]
        if choice.startswith("Row "):
            row = X_test.loc[type(X_test.index[0])(choice[4:])]
            load({name: row[name] for name in cfg.features})
        else:
            load(_typical(experiment))

    top = st.columns([1.4, 2.6], vertical_alignment="bottom")
    top[0].selectbox(
        "Fill the form from", presets, key=f"case_preset::{scope}", on_change=apply_preset,
        help="Load a row from the test set to check a known answer, or reset to typical values.",
    )

    grid = st.columns(3, gap="medium")
    for i, name in enumerate(cfg.features):
        with grid[i % 3]:
            column = experiment.X_train[name]
            if name in numeric:
                digits = _decimals(column)
                st.number_input(
                    name, value=None, key=keys[name], placeholder="unknown",
                    step=float(10 ** -digits) if digits else 1.0, format=f"%.{digits}f",
                )
            else:
                options = [UNKNOWN] + sorted(column.dropna().astype(str).unique())
                if st.session_state[keys[name]] not in options:
                    st.session_state[keys[name]] = UNKNOWN
                st.selectbox(name, options, key=keys[name])

    row = {}
    for name in cfg.features:
        value = st.session_state[keys[name]]
        if name in numeric:
            row[name] = [np.nan if value is None else float(value)]
        else:
            row[name] = [np.nan if value == UNKNOWN else value]
    case = pd.DataFrame(row)
    unknown = [name for name in cfg.features if pd.isna(case.loc[0, name])]

    cards = st.columns(len(ALGORITHMS), gap="medium")
    paths = {}
    for algorithm, column in zip(ALGORITHMS, cards):
        run = experiment.runs[algorithm]
        proba = run.pipeline.predict_proba(case)[0]
        transformed = run.prep.transform(case)
        path = run.model.decision_path(transformed)
        paths[algorithm] = [node.id for node, _ in path]
        steps, note = [], ""
        for node, key in path:
            if node.is_leaf:
                break
            if key is None:
                note = (
                    f"{node.feature} is unknown at node {node.id}, so the case went down every branch "
                    "weighted by branch size, and the leaf distributions were blended."
                    if algorithm == C45
                    else f"{node.feature} took a value this node never saw in training, so it answered with the node's class mix."
                )
                steps.append(f"{node.feature} = ?")
                break
            steps.append(_step_text(run.model, node, key, case))
        if run.prep.impute and unknown:
            filled = ", ".join(unknown)
            note = (note + " " if note else "") + f"Filled with the training median or mode before {algorithm} saw them: {filled}."
        with column:
            blended = algorithm == C45 and any(key is None and not node.is_leaf for node, key in path)
            prediction_card(algorithm, list(run.model.classes_), proba, steps, note, blended)

    with st.expander("Show the decision path in the tree", expanded=False):
        algorithm = st.segmented_control("Tree", ALGORITHMS, default=C45, key=f"case_tree::{scope}") or C45
        model = experiment.runs[algorithm].model
        # Expand only the nodes on the path; every other branch collapses to a stub.
        expand = None if model.n_leaves_ <= 12 else set(paths[algorithm])
        st.graphviz_chart(
            tree_graph(model, highlight=paths[algorithm], expand=expand, show_stats=False),
            width="content",
        )
        class_legend(model.classes_, hint="Outlined nodes = the path this case took")


def _step_text(model, node, key, case: pd.DataFrame) -> str:
    """Condition as the user entered it, e.g. 'petal_length = 4.8 → (4.75, 5.1]' for binned ID3."""
    raw = case.loc[0, node.feature]
    condition = model.condition(node, key)
    if model.algorithm == ID3 and isinstance(raw, (float, int, np.number)) and not pd.isna(raw):
        return f"{node.feature} {format_number(raw)} ∈ {key}"
    return condition


@st.fragment
def _test_predictions(experiment: Experiment) -> None:
    section(
        "Test-set predictions" if experiment.holdout else "Training-set predictions",
        eyebrow="Every case",
        description="Actual class next to both predictions. Wrong answers are shaded.",
    )
    actual = experiment.y_test.to_numpy()
    table = experiment.X_test.copy()
    table.insert(0, "actual", actual)
    for algorithm in ALGORITHMS:
        table.insert(table.columns.get_loc("actual") + 1 + ALGORITHMS.index(algorithm), algorithm, experiment.runs[algorithm].y_pred)
    id3_ok = table[ID3].to_numpy() == actual
    c45_ok = table[C45].to_numpy() == actual

    counts = {
        "All": len(table),
        "Disagreements": int((table[ID3] != table[C45]).sum()),
        "ID3 wrong": int((~id3_ok).sum()),
        "C4.5 wrong": int((~c45_ok).sum()),
    }
    view = st.segmented_control(
        "Show", list(counts), default="All", key="pred_filter", format_func=lambda k: f"{k}  {counts[k]}",
        label_visibility="collapsed",
    ) or "All"
    masks = {
        "All": np.ones(len(table), bool),
        "Disagreements": (table[ID3] != table[C45]).to_numpy(),
        "ID3 wrong": ~id3_ok,
        "C4.5 wrong": ~c45_ok,
    }
    shown = table[masks[view]]

    def shade(column: pd.Series) -> list[str]:
        if column.name not in ALGORITHMS:
            return [""] * len(column)
        return [f"background-color: {WRONG}" if p != a else "" for p, a in zip(column, shown["actual"])]

    st.dataframe(shown.style.apply(shade, axis=0), height=table_height(len(shown), 400), row_height=ROW_HEIGHT, width="stretch")
    st.download_button(
        "Download predictions (.csv)", table.to_csv().encode("utf-8"), file_name="predictions.csv",
        mime="text/csv", icon=":material/download:", on_click="ignore",
    )


@st.fragment
def _batch(experiment: Experiment) -> None:
    section(
        "Score a file",
        eyebrow="Batch prediction",
        description="Upload new cases with the same attribute columns (a target column is optional and ignored). "
        "Both trees predict every row.",
    )
    upload = st.file_uploader("CSV of new cases", type=["csv", "txt"], key="batch_upload", label_visibility="collapsed")
    if upload is None:
        return
    try:
        frame = read_csv(upload.getvalue())
    except Exception as error:  # noqa: BLE001 - report any parse failure
        st.error(f"Could not read {upload.name}: {error}")
        return
    absent = [name for name in experiment.config.features if name not in frame.columns]
    if absent:
        st.error(f"The file is missing {len(absent)} attribute column(s): {', '.join(absent)}.")
        return
    result = frame[list(experiment.config.features)].copy()
    for algorithm in ALGORITHMS:
        pipeline = experiment.runs[algorithm].pipeline
        proba = pipeline.predict_proba(result[list(experiment.config.features)])
        result[algorithm] = pipeline.classes_[proba.argmax(axis=1)]
        result[f"{algorithm} confidence"] = proba.max(axis=1).round(3)
    st.dataframe(result, height=table_height(len(result), 400), row_height=ROW_HEIGHT, width="stretch")
    st.download_button(
        "Download scored file (.csv)", result.to_csv().encode("utf-8"), file_name=f"scored_{upload.name}",
        mime="text/csv", icon=":material/download:", on_click="ignore",
    )
