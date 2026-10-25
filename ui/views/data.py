"""Data tab: schema editor, class balance, preprocessing log and previews."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from arbor import charts
from arbor.experiment import C45, ID3, Experiment
from arbor.preprocessing import BINNING_STRATEGIES, identifier_columns, profile_columns
from arbor.tree import format_number
from ui.components import ROW_HEIGHT, chip, esc, figure, section, stats, steps, table_height

KIND_LABELS = {"numeric": "Numeric", "categorical": "Categorical"}


@dataclass(frozen=True)
class Schema:
    features: tuple[str, ...]
    numeric: tuple[str, ...]


def schema_editor(frame: pd.DataFrame, target: str, scope: str) -> Schema:
    section(
        "Attributes",
        eyebrow="Schema",
        description="Tick the columns the trees may split on and set how each is treated. "
        "Numeric attributes are cut into intervals for ID3 and split on learned thresholds by C4.5.",
        first=True,
    )
    profile = profile_columns(frame.drop(columns=[target]))
    ids = set(identifier_columns(frame.drop(columns=[target])))
    table = pd.DataFrame(
        {
            "use": [name not in ids for name in profile["column"]],
            "attribute": profile["column"],
            "treat_as": profile["kind"].map(KIND_LABELS),
            "missing": profile["missing"],
            "unique": profile["unique"],
            "examples": profile["examples"],
        }
    )
    edited = st.data_editor(
        table,
        key=f"schema::{scope}",
        hide_index=True,
        width="stretch",
        height=table_height(len(table)),
        row_height=ROW_HEIGHT,
        disabled=["attribute", "missing", "unique", "examples"],
        column_config={
            "use": st.column_config.CheckboxColumn("Use", width="small"),
            "attribute": st.column_config.TextColumn("Attribute"),
            "treat_as": st.column_config.SelectboxColumn(
                "Treat as", options=list(KIND_LABELS.values()), required=True, width="small"
            ),
            "missing": st.column_config.NumberColumn("Missing", format="%d", width="small"),
            "unique": st.column_config.NumberColumn("Distinct", format="%d", width="small"),
            "examples": st.column_config.TextColumn("Example values"),
        },
    )

    features, numeric = [], []
    for row in edited.itertuples(index=False):
        if not row.use:
            continue
        features.append(row.attribute)
        if row.treat_as == "Numeric":
            column = frame[row.attribute]
            if pd.to_numeric(column, errors="coerce").notna().sum() < 0.9 * column.notna().sum():
                st.warning(f"**{row.attribute}** is mostly text, so it stays categorical.", icon=":material/info:")
            else:
                numeric.append(row.attribute)
    if ids:
        st.caption(f"Left out by default because they look like row identifiers: {', '.join(sorted(ids))}.")
    return Schema(tuple(features), tuple(numeric))


def summary(frame: pd.DataFrame, target: str, schema: Schema) -> None:
    counts = frame[target].dropna().astype(str).value_counts()
    features = list(schema.features)
    missing_cells = int(frame[features].isna().sum().sum()) if features else 0
    total_cells = max(1, len(frame) * len(features))
    stats(
        [
            ("Rows", f"{len(frame):,}", f"{frame[target].isna().sum():,} without a target" if frame[target].isna().any() else ""),
            ("Attributes in use", f"{len(features)} of {frame.shape[1] - 1}",
             f"{len(schema.numeric)} numeric · {len(features) - len(schema.numeric)} categorical"),
            ("Classes", f"{len(counts)}", f"in {target}"),
            ("Missing values", f"{missing_cells:,}", f"{missing_cells / total_cells:.1%} of attribute cells"),
            ("Majority baseline", f"{counts.iloc[0] / counts.sum():.1%}", f"always predict {counts.index[0]}"),
        ]
    )


def class_balance(frame: pd.DataFrame, target: str) -> None:
    section("Class balance", eyebrow="Target", description=f"How often each value of <b>{esc(target)}</b> occurs.", first=True)
    counts = frame[target].dropna().astype(str).value_counts()
    classes = sorted(counts.index)
    figure(charts.class_distribution(counts, classes))
    ratio = counts.max() / counts.min()
    if ratio >= 3:
        st.caption(
            f"The largest class is {ratio:.1f}× the smallest. Accuracy alone can flatter a model here; "
            "macro-averaged recall and F1 weigh every class equally."
        )
    else:
        st.caption(f"Largest to smallest class ratio {ratio:.1f} : 1.")


def preprocessing(experiment: Experiment) -> None:
    cfg = experiment.config
    section(
        "Preprocessing",
        eyebrow="Pipeline",
        description="What each algorithm receives. Fill values and cut points are learned on the training split "
        "only, then replayed unchanged on test rows and on new cases.",
    )
    shared = [(esc(note), "") for note in experiment.notes]
    if shared:
        steps(shared)

    left, right = st.columns(2, gap="large")
    id3_prep, c45_prep = experiment.runs[ID3].prep, experiment.runs[C45].prep
    with left:
        chip(ID3, "categorical attributes only")
        id3_steps = [_types_step(cfg), _impute_step(id3_prep, cfg)]
        if cfg.numeric_features:
            id3_steps.append(
                (
                    f"Discretise {len(cfg.numeric_features)} numeric attribute"
                    f"{'s' if len(cfg.numeric_features) != 1 else ''} into at most {cfg.n_bins} intervals",
                    f"{BINNING_STRATEGIES[cfg.binning]} binning; outer intervals are open so unseen extremes still land in a bin.",
                )
            )
        detail = f"stop at depth {cfg.id3_max_depth}" if cfg.id3_max_depth else "grow until pure or attributes run out"
        id3_steps.append(("Grow by information gain, one branch per value", detail))
        steps(id3_steps)
    with right:
        chip(C45, "numeric and categorical attributes")
        c45_steps = [_types_step(cfg)]
        if cfg.missing_strategy == "native":
            unknown = sum(c45_prep.missing_counts_.values())
            c45_steps.append(
                (
                    f"Keep {unknown:,} missing training value{'s' if unknown != 1 else ''} as unknown",
                    "Gain is scaled by the share of known values; unknown cases go down every branch with fractional weight.",
                )
            )
        else:
            c45_steps.append(_impute_step(c45_prep, cfg))
        if cfg.numeric_features:
            c45_steps.append(("Use numeric attributes as they are", "Each node searches every cut point between distinct values."))
        grow = "Grow by gain ratio among above-average-gain attributes"
        if cfg.c45_prune:
            c45_steps.append((grow, f"then prune bottom-up with confidence factor {cfg.c45_confidence:.2f}."))
        else:
            c45_steps.append((grow, "pruning is switched off."))
        steps(c45_steps)

    if cfg.numeric_features:
        section("ID3 intervals", description="Cut points learned from the training split for each numeric attribute.")
        bins = id3_prep.binning_table()
        st.dataframe(
            bins, hide_index=True, width="stretch", height=table_height(len(bins), 300), row_height=ROW_HEIGHT,
            column_config={"labels": st.column_config.TextColumn("intervals as ID3 sees them", width="large")},
        )


def _types_step(cfg) -> tuple[str, str]:
    n_num, n_cat = len(cfg.numeric_features), len(cfg.categorical_features)
    return (f"Read {n_num} attribute{'s' if n_num != 1 else ''} as numbers and {n_cat} as categories", "")


def _impute_step(prep, cfg) -> tuple[str, str]:
    filled = [(name, prep.fill_values_[name]) for name, n in prep.missing_counts_.items() if n and name in prep.fill_values_]
    if cfg.missing_strategy == "drop":
        return ("Missing values: incomplete rows were dropped", "")
    if not filled:
        return ("No missing values in the training split", "")
    parts = []
    for name, value in filled[:6]:
        shown = format_number(value) if isinstance(value, float) else value
        parts.append(f"<code>{esc(name)}</code> ← {esc(shown)}")
    more = f" and {len(filled) - 6} more" if len(filled) > 6 else ""
    return (f"Fill gaps in {len(filled)} attribute{'s' if len(filled) != 1 else ''}", ", ".join(parts) + more)


def preview(frame: pd.DataFrame, experiment: Experiment) -> None:
    section("Rows", eyebrow="Preview", description="The raw file, or the training rows exactly as each algorithm receives them.")
    view = st.segmented_control(
        "View", ["Raw data", "ID3 input", "C4.5 input"], default="Raw data", key="data_preview",
        label_visibility="collapsed",
    ) or "Raw data"
    if view == "Raw data":
        shown = frame
    else:
        run = experiment.runs[ID3 if view == "ID3 input" else C45]
        shown = run.prep.transform(experiment.X_train)
        shown[experiment.config.target] = experiment.y_train
        shown = shown.sort_index()
    st.dataframe(shown.head(500), height=330, width="stretch")
    if len(shown) > 500:
        st.caption(f"First 500 of {len(shown):,} rows.")
