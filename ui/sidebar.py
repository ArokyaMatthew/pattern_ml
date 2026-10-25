"""Sidebar: the experiment setup. Every control feeds ``ExperimentConfig``."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from arbor.datasets import SAMPLES
from arbor.experiment import MISSING_STRATEGIES
from arbor.preprocessing import BINNING_STRATEGIES, suggest_target
from ui import cache
from ui.components import side_label

DEFAULTS = {
    "cfg_missing": "impute",
    "cfg_numeric_impute": "median",
    "cfg_bins": 4,
    "cfg_binning": "quantile",
    "cfg_dedupe": False,
    "cfg_holdout": True,
    "cfg_test_size": 30,
    "cfg_stratify": True,
    "cfg_seed": 42,
    "cfg_id3_depth": 0,
    "cfg_id3_split": 2,
    "cfg_id3_gain": 0.0,
    "cfg_c45_depth": 0,
    "cfg_c45_leaf": 2,
    "cfg_c45_prune": True,
    "cfg_c45_cf": 0.25,
}


DEFAULT_SAMPLE = "iris"
# Below this many rows the hold-out split is switched off when a dataset loads.
SMALL_DATASET = 30


@dataclass(frozen=True)
class Source:
    id: str
    name: str


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _reset() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def dataset() -> tuple[pd.DataFrame | None, Source | None]:
    with st.sidebar:
        side_label("01", "Dataset", first=True)
        mode = st.segmented_control(
            "Source", ["Sample", "Upload CSV"], default="Sample", key="data_mode",
            label_visibility="collapsed", width="stretch",
        )
        if mode != "Upload CSV":
            st.session_state.setdefault("sample_key", DEFAULT_SAMPLE)
            key = st.selectbox("Sample dataset", list(SAMPLES), format_func=lambda k: SAMPLES[k].title, key="sample_key")
            st.caption(SAMPLES[key].description)
            frame, source = cache.sample(key), Source(f"sample:{key}", SAMPLES[key].file)
            _on_new_source(frame, source)
            return frame, source

        upload = st.file_uploader(
            "CSV file", type=["csv", "txt"], key="upload",
            help="Comma, semicolon, tab or pipe separated, with column names in the first row. "
            "Empty cells and '?' are read as missing values.",
        )
        if upload is None:
            return None, None
        data = upload.getvalue()
        try:
            frame = cache.uploaded(data, upload.name)
        except Exception as error:  # noqa: BLE001 - any parse failure is reported to the user
            st.error(f"Could not read **{upload.name}**: {error}")
            return None, None
        source = Source(f"upload:{hashlib.sha1(data).hexdigest()[:12]}", upload.name)
        _on_new_source(frame, source)
        return frame, source


def _on_new_source(frame: pd.DataFrame, source: Source) -> None:
    """Pick a sensible validation default whenever the dataset changes.

    Toy datasets such as Play Tennis (14 rows) are meant to be learned whole;
    holding out 30 % of them leaves too little to learn from or to score.
    """
    if st.session_state.get("_source_id") == source.id:
        return
    st.session_state["_source_id"] = source.id
    st.session_state["cfg_holdout"] = len(frame) >= SMALL_DATASET


def target(frame: pd.DataFrame, source: Source) -> str:
    with st.sidebar:
        side_label("02", "Target")
        columns = list(frame.columns)
        return st.selectbox(
            "Target attribute",
            columns,
            index=columns.index(suggest_target(frame)),
            key=f"target::{source.id}",
            help="The class to predict. Choose which other columns the trees may use in the Data tab.",
        )


def settings() -> dict:
    """Render the remaining controls; return ``ExperimentConfig`` keyword arguments."""
    s = st.session_state
    with st.sidebar:
        side_label("03", "Preprocessing")
        st.selectbox(
            "Missing values", list(MISSING_STRATEGIES), format_func=MISSING_STRATEGIES.get, key="cfg_missing",
            help="**Impute** fills gaps from the training split. **Drop** removes incomplete rows. "
            "**C4.5 native** keeps gaps for C4.5, which spreads such cases across branches with fractional "
            "weights; ID3 still gets imputed data.",
        )
        st.radio(
            "Numeric fill value", ["median", "mean"], format_func=str.capitalize, key="cfg_numeric_impute",
            horizontal=True, disabled=s.cfg_missing == "drop",
        )
        st.slider(
            "ID3 intervals per numeric attribute", 2, 10, key="cfg_bins",
            help="ID3 only handles categorical attributes, so numeric columns are cut into intervals. "
            "Cut points are learned on the training split.",
        )
        st.selectbox("ID3 binning strategy", list(BINNING_STRATEGIES), format_func=BINNING_STRATEGIES.get, key="cfg_binning")
        st.checkbox("Remove duplicate rows", key="cfg_dedupe")

        side_label("04", "Validation")
        st.toggle("Hold out a test set", key="cfg_holdout", help="Off: train on every row and score on the same rows.")
        if not s.cfg_holdout:
            st.caption("Training and scoring on every row. Fine for textbook examples; optimistic as an estimate.")
        st.slider("Test set size", 10, 50, step=5, format="%d%%", key="cfg_test_size", disabled=not s.cfg_holdout)
        st.checkbox("Stratify by class", key="cfg_stratify", disabled=not s.cfg_holdout)
        st.number_input("Random seed", min_value=0, max_value=2**31 - 1, step=1, key="cfg_seed")

        side_label("05", "ID3")
        st.number_input("Max depth", min_value=0, max_value=50, key="cfg_id3_depth", help="0 means no limit.")
        st.number_input("Min cases to split a node", min_value=2, max_value=1000, key="cfg_id3_split")
        st.slider("Min information gain (bits)", 0.0, 0.5, step=0.01, key="cfg_id3_gain")

        side_label("06", "C4.5")
        st.number_input("Max depth", min_value=0, max_value=50, key="cfg_c45_depth",
                        help="0 means no limit. C4.5 itself has no depth limit.")
        st.number_input("Min cases per branch (m)", min_value=1, max_value=1000, key="cfg_c45_leaf",
                        help="A split must send at least this many cases down two of its branches. C4.5's default is 2.")
        st.toggle("Error-based pruning", key="cfg_c45_prune")
        st.slider("Confidence factor (CF)", 0.01, 0.50, step=0.01, key="cfg_c45_cf", disabled=not s.cfg_c45_prune,
                  help="Lower values estimate more pessimistic error rates and prune harder. C4.5's default is 0.25.")

        st.button("Reset settings", on_click=_reset, type="tertiary", icon=":material/restart_alt:")
        st.markdown(
            '<div class="arbor-side-foot">Arbor 1.0 · ID3 and C4.5 implemented from scratch in NumPy; '
            "scikit-learn provides splitting, metrics and cross-validation.</div>",
            unsafe_allow_html=True,
        )

    return {
        "missing_strategy": s.cfg_missing,
        "numeric_impute": s.cfg_numeric_impute,
        "n_bins": int(s.cfg_bins),
        "binning": s.cfg_binning,
        "drop_duplicates": bool(s.cfg_dedupe),
        "test_size": s.cfg_test_size / 100 if s.cfg_holdout else 0.0,
        "stratify": bool(s.cfg_stratify),
        "seed": int(s.cfg_seed),
        "id3_max_depth": int(s.cfg_id3_depth) or None,
        "id3_min_samples_split": int(s.cfg_id3_split),
        "id3_min_gain": float(s.cfg_id3_gain),
        "c45_max_depth": int(s.cfg_c45_depth) or None,
        "c45_min_samples_leaf": int(s.cfg_c45_leaf),
        "c45_prune": bool(s.cfg_c45_prune),
        "c45_confidence": float(s.cfg_c45_cf),
    }
