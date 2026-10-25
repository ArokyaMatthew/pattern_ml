"""Arbor: an interactive workbench for ID3 and C4.5 decision trees.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from arbor.experiment import ExperimentConfig
from arbor.preprocessing import target_problem
from ui import cache, sidebar, theme
from ui.components import empty_state, esc, header
from ui.views import comparison, data, evaluation, method, predict, trees

st.set_page_config(
    page_title="Arbor · ID3 & C4.5 decision trees",
    page_icon=":material/account_tree:",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject()
sidebar.init_state()

frame, source = sidebar.dataset()
if frame is None:
    header()
    empty_state(
        "Upload a classification dataset",
        "<p>Choose a CSV file in the sidebar, or switch back to a sample dataset.</p>"
        "<ul><li>The first row must hold column names.</li>"
        "<li>One column is the class to predict; any number of other columns can be numeric or categorical.</li>"
        "<li>Empty cells and <code>?</code> are read as missing values.</li>"
        "<li>Comma, semicolon, tab and pipe delimiters are detected automatically.</li></ul>",
    )
    st.stop()

target = sidebar.target(frame, source)
settings = sidebar.settings()
n_classes = frame[target].nunique()
header(
    source.name,
    [
        ("Dataset", source.name),
        ("Rows", f"{len(frame):,}"),
        ("Attributes", str(frame.shape[1] - 1)),
        ("Target", target),
        ("Classes", str(n_classes)),
    ],
)

problem = target_problem(frame[target])
if problem:
    st.error(f"**{esc(target)}** cannot be the target: it {problem}. Pick another target attribute in the sidebar.")
    st.stop()

tab_data, tab_trees, tab_eval, tab_compare, tab_predict, tab_method = st.tabs(
    ["Data", "Trees", "Evaluation", "Comparison", "Predict", "Method"]
)
scope = f"{source.id}::{target}"

with tab_data:
    summary_slot = st.container()
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        schema = data.schema_editor(frame, target, scope)
    with right:
        data.class_balance(frame, target)
    with summary_slot:
        data.summary(frame, target, schema)
    details_slot = st.container()

with tab_method:
    method.render()

if not schema.features:
    with tab_data:
        st.warning("Tick at least one attribute in the table above to train the trees.", icon=":material/warning:")
    st.stop()

config = ExperimentConfig(target=target, features=schema.features, numeric_features=schema.numeric, **settings)
try:
    experiment = cache.experiment(frame, config)
except ValueError as error:
    with tab_data:
        st.error(f"Training failed: {error}")
    st.stop()

with details_slot:
    data.preprocessing(experiment)
    data.preview(frame, experiment)
with tab_trees:
    trees.render(experiment)
with tab_eval:
    evaluation.render(experiment)
with tab_compare:
    comparison.render(frame, experiment)
with tab_predict:
    predict.render(experiment, scope)
