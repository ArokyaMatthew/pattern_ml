"""Graphviz drawings of fitted ID3 / C4.5 trees.

Only DOT source is produced here. Streamlit renders DOT in the browser, so
the Graphviz system binaries are needed only for exporting SVG/PNG files
(``render_svg``).
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape

import graphviz

from arbor.style import AXIS, BORDER, INK, INK_MUTED, INK_SECONDARY, SURFACE, class_color, mix
from arbor.tree import BaseDecisionTree, Node, format_number

# Graphviz sizes text with Helvetica metrics; the app's stylesheet swaps the
# rendered face for IBM Plex Sans, which has very similar widths.
FONT = "Helvetica"
BAR_WIDTH = 96


def tree_graph(
    model: BaseDecisionTree,
    *,
    orientation: str = "TB",
    max_depth: int | None = None,
    highlight: Iterable[int] = (),
    expand: Iterable[int] | None = None,
    show_stats: bool = True,
) -> graphviz.Digraph:
    """Build a styled ``graphviz.Digraph`` of a fitted tree.

    Parameters
    ----------
    orientation : {"TB", "LR"}
        Top-to-bottom or left-to-right layout.
    max_depth : int, optional
        Draw only this many levels; deeper subtrees collapse into a stub.
    highlight : node ids
        Nodes on a decision path, drawn with a heavier outline.
    expand : node ids, optional
        Draw children only for these nodes; every other subtree collapses
        into a stub. Used to show a single decision path in a large tree.
    show_stats : bool
        Include case counts, entropy and split scores in each node.
    """
    highlight = set(highlight)
    expand = None if expand is None else set(expand)
    colors = [class_color(i) for i in range(len(model.classes_))]
    graph = graphviz.Digraph(
        name=model.algorithm,
        graph_attr={
            "rankdir": orientation,
            "bgcolor": "transparent",
            "nodesep": "0.22",
            "ranksep": "0.42" if orientation == "TB" else "0.6",
            "pad": "0.15",
            "ordering": "out",
            "fontname": FONT,
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": FONT,
            "margin": "0.10,0.06",
            "penwidth": "1",
        },
        edge_attr={
            "fontname": FONT,
            "color": AXIS,
            "arrowsize": "0.55",
            "penwidth": "1.1",
        },
    )

    def add(node: Node) -> None:
        on_path = node.id in highlight
        fill = mix(colors[node.majority], SURFACE, 0.84) if node.is_leaf else SURFACE
        graph.node(
            str(node.id),
            label=_node_label(model, node, colors, show_stats),
            fillcolor=fill,
            color=INK if on_path else BORDER,
            penwidth="2.2" if on_path else "1",
            tooltip=_tooltip(model, node),
        )
        if node.is_leaf:
            return
        if (max_depth is not None and node.depth >= max_depth) or (expand is not None and node.id not in expand):
            hidden = sum(1 for _ in node.iter_nodes()) - 1
            stub = f"{node.id}_more"
            graph.node(
                stub,
                label=f"<<FONT POINT-SIZE=\"9\" COLOR=\"{INK_MUTED}\">+{hidden} nodes</FONT>>",
                style="rounded,dashed",
                color=AXIS,
                fillcolor=SURFACE,
            )
            graph.edge(str(node.id), stub, style="dashed")
            return
        for key, child in node.children.items():
            add(child)
            both = on_path and child.id in highlight
            graph.edge(
                str(node.id),
                str(child.id),
                label=_edge_label(model.branch_label(node, key)),
                color=INK if both else AXIS,
                penwidth="2" if both else "1.1",
            )

    add(model.root_)
    return graph


def render_svg(graph: graphviz.Digraph) -> bytes | None:
    """SVG bytes via the Graphviz binaries, or ``None`` if they're not installed."""
    try:
        return graph.pipe(format="svg")
    except (graphviz.ExecutableNotFound, graphviz.CalledProcessError, OSError):
        return None


def _html(text: str) -> str:
    return escape(str(text)).replace("≤", "&#8804;")


def _node_label(model: BaseDecisionTree, node: Node, colors: list[str], show_stats: bool) -> str:
    if node.is_leaf:
        title = _html(model.classes_[node.majority])
    else:
        title = _html(node.feature)
    rows = [f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="11" COLOR="{INK}"><B>{title}</B></FONT></TD></TR>']

    if show_stats:
        details = [f"n {format_number(node.n_samples, 3)}"]
        if node.is_leaf:
            details.append(f"{node.distribution.max():.0%} pure")
        else:
            details.append(f"H {node.entropy:.3f}")
        rows.append(_small_row(" · ".join(details)))
        if not node.is_leaf:
            if model.algorithm == "C4.5":
                rows.append(_small_row(f"gain ratio {node.gain_ratio:.3f}"))
            else:
                rows.append(_small_row(f"gain {node.gain:.3f}"))
        elif node.pruned_feature:
            rows.append(_small_row(f"pruned {_html(node.pruned_feature)}"))

    rows.append(f'<TR><TD ALIGN="LEFT" CELLPADDING="3">{_distribution_bar(node, colors)}</TD></TR>')
    return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">' + "".join(rows) + "</TABLE>>"


def _small_row(text: str) -> str:
    return f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8.5" COLOR="{INK_SECONDARY}">{text}</FONT></TD></TR>'


def _distribution_bar(node: Node, colors: list[str]) -> str:
    """Class mix of a node as a thin segmented bar, segments split by 1pt gaps."""
    shares = [(p, colors[i]) for i, p in enumerate(node.distribution) if p > 0]
    gaps = len(shares) - 1
    usable = BAR_WIDTH - gaps
    widths = [max(2, round(p * usable)) for p, _ in shares]
    widths[widths.index(max(widths))] += usable - sum(widths)
    cells = "".join(
        f'<TD WIDTH="{w}" HEIGHT="5" FIXEDSIZE="TRUE" BGCOLOR="{color}"></TD>'
        for w, (_, color) in zip(widths, shares)
    )
    return f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="0"><TR>{cells}</TR></TABLE>'


def _edge_label(text: str) -> str:
    if len(text) > 26:
        text = text[:25] + "…"
    return f'<<FONT POINT-SIZE="9" COLOR="{INK_SECONDARY}"> {_html(text)} </FONT>>'


def _tooltip(model: BaseDecisionTree, node: Node) -> str:
    counts = ", ".join(
        f"{label}: {format_number(count, 3)}" for label, count in zip(model.classes_, node.counts) if count > 0
    )
    head = f"Node {node.id}"
    if not node.is_leaf:
        head += f" · split on {node.feature}"
    return f"{head}\n{counts}"
