"""Cached wrappers around the expensive calls.

Streamlit re-runs the script on every interaction; these keep a slider
tweak in one tab from retraining everything. Cache keys are the dataset
frame plus the frozen ``ExperimentConfig``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from arbor.datasets import load_sample
from arbor.experiment import (
    ALGORITHMS,
    CART,
    Experiment,
    ExperimentConfig,
    cross_validate_models,
    depth_sweep,
    run_experiment,
)
from arbor.preprocessing import read_csv


@st.cache_data(show_spinner=False, max_entries=16)
def sample(key: str) -> pd.DataFrame:
    return load_sample(key)


@st.cache_data(show_spinner=False, max_entries=8)
def uploaded(data: bytes, name: str) -> pd.DataFrame:
    return read_csv(data)


@st.cache_data(show_spinner="Training ID3 and C4.5…", max_entries=48)
def experiment(frame: pd.DataFrame, cfg: ExperimentConfig) -> Experiment:
    return run_experiment(frame, cfg, ALGORITHMS)


@st.cache_data(show_spinner="Training the scikit-learn reference…", max_entries=16)
def reference(frame: pd.DataFrame, cfg: ExperimentConfig) -> Experiment:
    return run_experiment(frame, cfg, (CART,))


@st.cache_data(show_spinner="Running k-fold cross-validation…", max_entries=32)
def cross_validation(frame: pd.DataFrame, cfg: ExperimentConfig, folds: int, average: str) -> pd.DataFrame:
    return cross_validate_models(frame, cfg, folds=folds, average=average)


@st.cache_data(show_spinner="Retraining at each depth limit…", max_entries=32)
def sweep(frame: pd.DataFrame, cfg: ExperimentConfig, max_depth: int) -> pd.DataFrame:
    return depth_sweep(experiment(frame, cfg), max_depth)
