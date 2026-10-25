"""Trees tab: the drawing, split-by-split reasoning, and the tree as rules."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from arbor.experiment import ALGORITHMS, C45, Experiment
from arbor.graph import render_svg, tree_graph
from arbor.style import ACCENT, mix
from arbor.tree import BaseDecisionTree, Node
from ui.components import ROW_HEIGHT, class_legend, fmt_count, fmt_seconds, section, stats, table_height

LAYOUTS = {"Top-down": "TB", "Left-right": "LR"}


def _level_widths(model: BaseDecisionTree) -> dict[int, int]:
    widths: dict[int, int] = {}
    for node in model.nodes_:
        widths[node.depth] = widths.get(node.depth, 0) + 1
    return widths


def default_levels(model: BaseDecisionTree, widest: int = 8) -> int:
    """As many levels as keep the widest drawn row to about ``widest`` nodes.

    The drawing is scaled to the page width, so this is what keeps labels
    legible; the slider and the full-screen button show the rest.
    """
    widths = _level_widths(model)
    levels = 1
    while levels < model.depth_ and widths.get(levels + 1, 0) <= widest:
        levels += 1
    return max(levels, 1)


@st.fragment
def render(experiment: Experiment) -> None:
    controls = st.columns([1.1, 1.1, 1.6, 1], vertical_alignment="bottom", gap="medium")
    algorithm = controls[0].segmented_control(
        "Algorithm", ALGORITHMS, default=C45, key="tree_algorithm", width="stretch"
    ) or C45
    run = experiment.runs[algorithm]
    model = run.model
    shape = f"{algorithm}::{model.depth_}::{model.n_nodes_}"
    if model.depth_ > 1:
        levels = controls[2].slider(
            "Levels shown", 1, model.depth_, value=default_levels(model), key=f"tree_levels::{shape}"
        )
    else:
        levels = model.depth_
    layout = controls[1].segmented_control(
        "Layout", list(LAYOUTS), default="Top-down", key="tree_layout", width="stretch"
    ) or "Top-down"
    show_stats = controls[3].toggle("Node details", value=True, key="tree_details")

    tiles = [
        ("Depth", str(model.depth_)),
        ("Leaves", str(model.n_leaves_)),
        ("Nodes", str(model.n_nodes_)),
        ("Training accuracy", f"{run.train_accuracy:.3f}"),
        ("Fit time", fmt_seconds(model.fit_time_), "tree only, excluding preprocessing"),
    ]
    if algorithm == C45 and model.prune:
        tiles.insert(3, ("Pruned away", str(model.n_pruned_), f"of {model.n_nodes_grown_} grown nodes"))
    stats(tiles)

    graph = tree_graph(model, orientation=LAYOUTS[layout], max_depth=levels, show_stats=show_stats)
    st.graphviz_chart(graph, width="content")
    hint = "Leaf colour = predicted class · bar = class mix of the training cases at that node"
    if levels < model.depth_:
        hint += f" · showing {levels} of {model.depth_} levels; use the slider or the full-screen button for more"
    class_legend(model.classes_, hint=hint)
    _downloads(model, graph, algorithm)

    _split_analysis(model, algorithm)
    _rules(model, experiment.config.target)


def _downloads(model: BaseDecisionTree, graph, algorithm: str) -> None:
    slug = algorithm.lower().replace(".", "")
    cols = st.columns([1, 1, 1, 3], gap="small")
    cols[0].download_button(
        "DOT source", graph.source, file_name=f"{slug}_tree.dot", mime="text/vnd.graphviz",
        icon=":material/download:", on_click="ignore", width="stretch",
    )
    cols[1].download_button(
        "Rules (.txt)", "\n".join(r.as_text() for r in model.rules()), file_name=f"{slug}_rules.txt",
        mime="text/plain", icon=":material/download:", on_click="ignore", width="stretch",
    )
    svg = render_svg(graph)
    if svg is not None:
        cols[2].download_button(
            "SVG image", svg, file_name=f"{slug}_tree.svg", mime="image/svg+xml",
            icon=":material/download:", on_click="ignore", width="stretch",
        )
    else:
        cols[2].button(
            "SVG image", disabled=True, width="stretch", icon=":material/download:",
            help="Install the Graphviz system package (graphviz.org/download) to export SVG files.",
        )


def _parents(model: BaseDecisionTree) -> dict[int, tuple[Node, object]]:
    parents = {}
    for node in model.nodes_:
        for key, child in node.children.items():
            parents[child.id] = (node, key)
    return parents


def _path_text(model: BaseDecisionTree, node: Node, parents) -> str:
    conditions = []
    while node.id in parents:
        parent, key = parents[node.id]
        conditions.append(model.condition(parent, key))
        node = parent
    return " AND ".join(reversed(conditions)) if conditions else "all training cases (root)"


def _split_analysis(model: BaseDecisionTree, algorithm: str) -> None:
    section(
        "Why each split",
        eyebrow="Split analysis",
        description="Every attribute the algorithm scored at a node. "
        + ("ID3 takes the highest information gain." if algorithm != C45 else
           "C4.5 admits attributes with at least average gain, then takes the highest gain ratio."),
    )
    nodes = [node for node in model.nodes_ if node.candidates]
    if not nodes:
        st.info("The root is already pure or too small to split, so no attribute was scored.")
        return
    parents = _parents(model)

    def label(node_id: int) -> str:
        node = model.nodes_[node_id]
        what = node.feature or (f"pruned ({node.pruned_feature})" if node.pruned_feature else "leaf, no useful split")
        return f"Node {node.id} · {what} · n {fmt_count(node.n_samples)}"

    node_id = st.selectbox("Node", [n.id for n in nodes], format_func=label, key=f"split_node::{algorithm}::{model.n_nodes_}")
    node = model.nodes_[node_id]
    mix_text = ", ".join(f"{c} {fmt_count(n)}" for c, n in zip(model.classes_, node.counts) if n > 0)
    st.caption(f"Reached by: {_path_text(model, node, parents)}  \nClass mix: {mix_text} · entropy {node.entropy:.3f} bits")

    rows = []
    for c in node.candidates:
        if c.selected and node.pruned_feature:
            status = "chosen, later pruned"
        elif c.selected:
            status = "chosen"
        else:
            status = c.note
        rows.append(
            {
                "Attribute": c.feature,
                "Gain": c.gain,
                "Split info": c.split_info,
                "Gain ratio": c.gain_ratio,
                "Cut point": c.threshold,
                "Status": status,
            }
        )
    table = pd.DataFrame(rows).sort_values("Gain ratio" if algorithm == C45 else "Gain", ascending=False)
    if table["Cut point"].isna().all():
        table = table.drop(columns="Cut point")
    highlight = mix(ACCENT, "#ffffff", 0.88)
    styled = (
        table.style.format(precision=4, na_rep="")
        .apply(lambda row: [f"background-color: {highlight}" if row["Status"].startswith("chosen") else "" for _ in row], axis=1)
    )
    st.dataframe(styled, hide_index=True, width="stretch")

    if node.leaf_error_estimate is not None:
        verdict = "kept" if node.children else "pruned"
        st.caption(
            f"Pruning check: as one leaf this node would make an estimated {node.leaf_error_estimate:.2f} errors, "
            f"as a subtree {node.subtree_error_estimate:.2f}. Subtree {verdict}."
        )


def _rules(model: BaseDecisionTree, target: str) -> None:
    section("The tree as rules", eyebrow="Rules", description="One rule per leaf. Cases can be fractional for C4.5.")
    view = st.segmented_control(
        "Format", ["Rules", "Text tree"], default="Rules", key="rules_view", label_visibility="collapsed"
    ) or "Rules"
    if view == "Text tree":
        st.code(model.export_text(), language=None, height=380)
        return
    rules = model.rules()
    table = pd.DataFrame(
        {
            "IF": [" AND ".join(r.conditions) or "(always)" for r in rules],
            "THEN": [f"{target} = {r.prediction}" for r in rules],
            "Cases": [r.support for r in rules],
            "Confidence": [r.confidence for r in rules],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=table_height(len(table), 380),
        row_height=ROW_HEIGHT,
        column_config={
            "IF": st.column_config.TextColumn("IF"),
            "Cases": st.column_config.NumberColumn("Cases", format="%.4g", width="small"),
            "Confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1, width="small"),
        },
    )
