"""Small HTML building blocks rendered through ``st.markdown``.

Every piece of user data that ends up in markup goes through ``esc``.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.figure import Figure

from arbor.style import algorithm_color, class_color
from arbor.tree import format_number
from ui.theme import LOGO


def esc(value) -> str:
    return escape(str(value))


def html(markup: str) -> None:
    # Markdown would read indented lines as code and stop an HTML block at a
    # blank line, so flatten the markup before handing it over.
    flat = "".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def header(dataset: str | None = None, meta: Sequence[tuple[str, str]] = ()) -> None:
    items = "".join(f'<div><span class="k">{esc(k)}</span><span class="v" title="{esc(v)}">{esc(v)}</span></div>' for k, v in meta)
    html(
        f"""<div class="arbor-header">
  <div class="arbor-brand">{LOGO}
    <div><div class="arbor-wordmark">Arbor</div>
    <div class="arbor-tagline">Decision tree classification workbench · ID3 and C4.5</div></div>
  </div>
  <div class="arbor-meta">{items}</div>
</div>"""
    )


def section(title: str, eyebrow: str | None = None, description: str | None = None, first: bool = False) -> None:
    parts = [f'<div class="arbor-section{" first" if first else ""}">']
    if eyebrow:
        parts.append(f'<div class="arbor-eyebrow">{esc(eyebrow)}</div>')
    parts.append(f'<div class="arbor-title">{esc(title)}</div>')
    if description:
        parts.append(f"<p>{description}</p>")
    parts.append("</div>")
    html("".join(parts))


def side_label(number: str, text: str, first: bool = False) -> None:
    html(f'<div class="arbor-side-label{" first" if first else ""}"><span class="num">{esc(number)}</span>{esc(text)}</div>')


def stats(items: Iterable[tuple[str, str] | tuple[str, str, str]]) -> None:
    cells = []
    for item in items:
        label, value, *rest = item
        sub = f'<div class="sub">{esc(rest[0])}</div>' if rest and rest[0] else ""
        cells.append(f'<div class="arbor-stat"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>{sub}</div>')
    html(f'<div class="arbor-stats">{"".join(cells)}</div>')


def chip(algorithm: str, description: str = "") -> None:
    desc = f'<span class="desc">{esc(description)}</span>' if description else ""
    html(
        f'<div class="arbor-chip" style="--c:{algorithm_color(algorithm)}"><span class="dot"></span>'
        f'<span class="name">{esc(algorithm)}</span>{desc}</div>'
    )


def class_legend(classes: Sequence, hint: str = "") -> None:
    swatches = "".join(f'<span style="--c:{class_color(i)}"><i></i>{esc(c)}</span>' for i, c in enumerate(classes))
    tail = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    html(f'<div class="arbor-legend">{swatches}{tail}</div>')


def steps(items: Iterable[tuple[str, str]]) -> None:
    """Numbered list; each item is ``(markup, detail markup)``. Callers escape."""
    rows = "".join(
        f"<li>{text}{f'<span class=detail>{detail}</span>' if detail else ''}</li>" for text, detail in items
    )
    html(f'<ol class="arbor-steps">{rows}</ol>')


def verdict(markup: str) -> None:
    html(f'<div class="arbor-verdict">{markup}</div>')


def empty_state(title: str, body: str) -> None:
    html(f'<div class="arbor-empty"><div class="arbor-title">{esc(title)}</div>{body}</div>')


def prediction_card(
    algorithm: str, classes: Sequence, proba: np.ndarray, path: Sequence[str], note: str = "", blended: bool = False
) -> None:
    best = int(np.argmax(proba))
    order = np.argsort(-proba)
    bars = "".join(
        f'<span class="cls" style="--c:{class_color(i)}"><i></i>{esc(classes[i])}</span>'
        f'<span class="track"><span class="fill" style="--c:{class_color(i)};width:{proba[i] * 100:.1f}%;display:block"></span></span>'
        f'<span class="pct">{proba[i]:.0%}</span>'
        for i in order[:6]
    )
    arrow = '<span class="arrow">→</span>'
    route = arrow.join(f"<code>{esc(step)}</code>" for step in path)
    route += f'{arrow}<span class="end">{esc(classes[best])}</span>'
    extra = f'<div class="conf" style="margin-top:6px">{esc(note)}</div>' if note else ""
    html(
        f"""<div class="arbor-card" style="--c:{algorithm_color(algorithm)}">
  <div class="eyebrow"><i></i>{esc(algorithm)}</div>
  <div class="pred">{esc(classes[best])}</div>
  <div class="conf">{proba[best]:.0%} {'after blending the reachable leaves' if blended else 'of the training cases at the leaf'}</div>
  <div class="arbor-prob">{bars}</div>
  <div class="arbor-path">{route}</div>{extra}
</div>"""
    )


def figure(fig: Figure) -> None:
    """Show a Matplotlib figure at its designed size (100 px per inch), crisp on high-DPI screens."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200)
    width = int(round(fig.get_figwidth() * 100))
    plt.close(fig)
    st.image(buffer.getvalue(), width=width)


ROW_HEIGHT = 34


def table_height(n_rows: int, cap: int = 420) -> int:
    """Pixel height that shows ``n_rows`` rows plus the header without a blank tail."""
    return min(ROW_HEIGHT * (n_rows + 1) + 3, cap)


def fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms" if seconds < 0.1 else f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def fmt_count(value: float) -> str:
    return format_number(value, 3)
