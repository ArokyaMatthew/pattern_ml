"""Stylesheet for the workbench.

Streamlit's theme (``.streamlit/config.toml``) sets fonts and base colours;
this adds the layout, typography and the few custom components (header,
stat strips, prediction cards) that the theme options cannot express.
"""

from __future__ import annotations

import streamlit as st

from arbor.style import (
    ACCENT,
    ALGORITHM_COLORS,
    BORDER,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PAGE,
    SUNKEN,
    SURFACE,
)

CSS = f"""
<style>
:root {{
  --arbor-ink: {INK};
  --arbor-ink-2: {INK_SECONDARY};
  --arbor-muted: {INK_MUTED};
  --arbor-border: {BORDER};
  --arbor-grid: {GRID};
  --arbor-surface: {SURFACE};
  --arbor-sunken: {SUNKEN};
  --arbor-page: {PAGE};
  --arbor-accent: {ACCENT};
  --arbor-id3: {ALGORITHM_COLORS["ID3"]};
  --arbor-c45: {ALGORITHM_COLORS["C4.5"]};
  --arbor-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}}

/* ---- page frame ---------------------------------------------------- */
[data-testid="stMainBlockContainer"] {{
  padding-top: 1.75rem;
  padding-bottom: 5rem;
  max-width: 1400px;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebarContent"] {{ padding-top: 0.25rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}

/* ---- header -------------------------------------------------------- */
.arbor-header {{
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 18px 32px; flex-wrap: wrap;
  padding-bottom: 16px; margin-bottom: 6px;
  border-bottom: 1px solid var(--arbor-border);
}}
.arbor-brand {{ display: flex; align-items: center; gap: 12px; }}
.arbor-brand svg {{ flex: none; }}
.arbor-wordmark {{ font-size: 1.3rem; font-weight: 600; color: var(--arbor-ink); line-height: 1.15; }}
.arbor-tagline {{ font-size: 0.84rem; color: var(--arbor-ink-2); }}
.arbor-meta {{ display: flex; flex-wrap: wrap; gap: 6px 28px; }}
.arbor-meta > div {{ display: flex; flex-direction: column; min-width: 0; }}
.arbor-meta .k {{
  font-size: 0.66rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--arbor-muted);
}}
.arbor-meta .v {{
  font-size: 0.9rem; font-weight: 500; color: var(--arbor-ink);
  font-variant-numeric: tabular-nums; max-width: 260px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}

/* ---- tabs ---------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 12px 9px; height: auto; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 0.93rem; font-weight: 500; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 0.6rem; }}

/* ---- section headings ---------------------------------------------- */
.arbor-section {{ margin: 1.5rem 0 0.7rem; }}
.arbor-section.first {{ margin-top: 0.5rem; }}
.arbor-eyebrow {{
  font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--arbor-muted); margin-bottom: 2px;
}}
.arbor-title {{ font-size: 1.06rem; font-weight: 600; color: var(--arbor-ink); line-height: 1.35; }}
.arbor-section p {{ font-size: 0.87rem; color: var(--arbor-ink-2); margin: 4px 0 0; max-width: 80ch; line-height: 1.5; }}

/* ---- sidebar section labels ---------------------------------------- */
.arbor-side-label {{
  display: flex; align-items: baseline; gap: 8px;
  margin: 1.1rem 0 0.15rem; padding-top: 0.85rem;
  border-top: 1px solid var(--arbor-border);
  font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--arbor-ink-2);
}}
.arbor-side-label.first {{ border-top: none; padding-top: 0; margin-top: 0.2rem; }}
.arbor-side-label .num {{ font-family: var(--arbor-mono); color: var(--arbor-accent); letter-spacing: 0; }}
.arbor-side-foot {{ font-size: 0.74rem; color: var(--arbor-muted); margin-top: 1.4rem; line-height: 1.5; }}

/* ---- stat strip ---------------------------------------------------- */
.arbor-stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  border: 1px solid var(--arbor-border); border-radius: 8px;
  background: var(--arbor-surface); overflow: hidden; margin: 0.25rem 0 0.9rem;
}}
.arbor-stat {{ padding: 11px 16px 12px; box-shadow: 1px 0 0 var(--arbor-grid); }}
.arbor-stat .label {{ font-size: 0.74rem; color: var(--arbor-ink-2); white-space: nowrap; }}
.arbor-stat .value {{ font-size: 1.32rem; font-weight: 600; color: var(--arbor-ink); line-height: 1.3; }}
.arbor-stat .sub {{ font-size: 0.74rem; color: var(--arbor-muted); font-variant-numeric: tabular-nums; }}

/* ---- algorithm chip ------------------------------------------------- */
.arbor-chip {{ display: flex; align-items: baseline; gap: 10px; margin: 0.2rem 0 0.55rem; }}
.arbor-chip .dot {{ width: 10px; height: 10px; border-radius: 3px; background: var(--c); align-self: center; flex: none; }}
.arbor-chip .name {{ font-size: 1.02rem; font-weight: 600; color: var(--arbor-ink); }}
.arbor-chip .desc {{ font-size: 0.84rem; color: var(--arbor-ink-2); }}

/* ---- legend --------------------------------------------------------- */
.arbor-legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.8rem; color: var(--arbor-ink-2); margin: 0.4rem 0 0.2rem; }}
.arbor-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.arbor-legend i {{ width: 10px; height: 10px; border-radius: 3px; background: var(--c); display: inline-block; }}
.arbor-legend .hint {{ color: var(--arbor-muted); }}

/* ---- notes / pipeline log ------------------------------------------ */
.arbor-steps {{ list-style: none; counter-reset: step; margin: 0.2rem 0 0; padding: 0; }}
.arbor-steps li {{
  counter-increment: step; position: relative; padding: 7px 0 7px 30px;
  font-size: 0.86rem; color: var(--arbor-ink); line-height: 1.45;
  border-bottom: 1px solid var(--arbor-grid);
}}
.arbor-steps li:last-child {{ border-bottom: none; }}
.arbor-steps li::before {{
  content: counter(step, decimal-leading-zero); position: absolute; left: 0; top: 8px;
  font-family: var(--arbor-mono); font-size: 0.72rem; color: var(--arbor-muted);
}}
.arbor-steps .detail {{ display: block; color: var(--arbor-ink-2); font-size: 0.8rem; }}
.arbor-steps code, .arbor-path code {{ font-family: var(--arbor-mono); font-size: 0.78rem; background: var(--arbor-sunken); padding: 1px 5px; border-radius: 4px; }}

/* ---- prediction card ----------------------------------------------- */
.arbor-card {{
  border: 1px solid var(--arbor-border); border-radius: 8px; background: var(--arbor-surface);
  padding: 15px 18px 16px; height: 100%;
}}
.arbor-card .eyebrow {{ display: flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: 600; color: var(--arbor-ink); }}
.arbor-card .eyebrow i {{ width: 10px; height: 10px; border-radius: 3px; background: var(--c); }}
.arbor-card .pred {{ font-size: 1.55rem; font-weight: 600; color: var(--arbor-ink); margin: 6px 0 0; line-height: 1.2; word-break: break-word; }}
.arbor-card .conf {{ font-size: 0.82rem; color: var(--arbor-ink-2); }}
.arbor-prob {{
  display: grid; grid-template-columns: minmax(70px, max-content) 1fr 46px;
  gap: 7px 12px; align-items: center; margin-top: 14px; font-size: 0.82rem; color: var(--arbor-ink);
}}
.arbor-prob .cls {{ display: inline-flex; align-items: center; gap: 7px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.arbor-prob .cls i {{ width: 8px; height: 8px; border-radius: 2px; background: var(--c); flex: none; }}
.arbor-prob .track {{ height: 8px; background: var(--arbor-sunken); border-radius: 4px; overflow: hidden; }}
.arbor-prob .fill {{ height: 100%; background: var(--c); border-radius: 0 4px 4px 0; }}
.arbor-prob .pct {{ text-align: right; font-variant-numeric: tabular-nums; color: var(--arbor-ink-2); }}
.arbor-path {{
  margin-top: 14px; padding-top: 11px; border-top: 1px solid var(--arbor-grid);
  font-size: 0.8rem; color: var(--arbor-ink-2); line-height: 1.9;
}}
.arbor-path .arrow {{ color: var(--arbor-muted); padding: 0 4px; }}
.arbor-path code {{ white-space: nowrap; }}
.arbor-path .end {{ font-weight: 600; color: var(--arbor-ink); }}

/* ---- callouts ------------------------------------------------------- */
.arbor-verdict {{
  border-left: 3px solid var(--arbor-accent); background: var(--arbor-surface);
  border-radius: 0 8px 8px 0; padding: 12px 16px; font-size: 0.92rem; color: var(--arbor-ink);
  line-height: 1.55; margin: 0.3rem 0 1rem; border-top: 1px solid var(--arbor-border);
  border-right: 1px solid var(--arbor-border); border-bottom: 1px solid var(--arbor-border);
}}
.arbor-empty {{
  border: 1px dashed var(--arbor-border); border-radius: 10px; background: var(--arbor-surface);
  padding: 44px 32px; margin-top: 1.5rem; max-width: 760px;
}}
.arbor-empty .arbor-title {{ font-size: 1.1rem; margin-bottom: 6px; }}
.arbor-empty p, .arbor-empty li {{ font-size: 0.9rem; color: var(--arbor-ink-2); line-height: 1.6; }}
.arbor-empty code {{ font-family: var(--arbor-mono); font-size: 0.82rem; }}

/* ---- tree drawing --------------------------------------------------- */
[data-testid="stGraphVizChart"] {{
  border: 1px solid var(--arbor-border); border-radius: 8px; background: var(--arbor-surface);
  padding: 14px; overflow: auto;
}}
[data-testid="stGraphVizChart"] svg text {{
  font-family: "IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
}}

/* ---- widgets -------------------------------------------------------- */
[data-testid="stExpander"] summary p {{ font-weight: 500; }}
[data-testid="stCaptionContainer"] {{ color: var(--arbor-ink-2); }}
[data-testid="stImage"] img {{ max-width: 100%; height: auto; }}
code {{ font-family: var(--arbor-mono); }}
</style>
"""

LOGO = """
<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect width="34" height="34" rx="8" fill="#1f5c46"/>
  <path d="M17 10.5V14.5M10.5 14.5H23.5M10.5 14.5V18.5M23.5 14.5V18.5M23.5 22V23.5M19.5 23.5H27M19.5 23.5V25M27 23.5V25"
        stroke="#ffffff" stroke-opacity="0.85" stroke-width="1.4" stroke-linecap="round"/>
  <rect x="13.5" y="6" width="7" height="4.5" rx="1.4" fill="#ffffff"/>
  <rect x="7" y="18.5" width="7" height="4.5" rx="1.4" fill="#ffffff" fill-opacity="0.9"/>
  <rect x="20" y="18.5" width="7" height="4" rx="1.4" fill="#ffffff"/>
  <circle cx="19.5" cy="26.6" r="1.7" fill="#ffffff" fill-opacity="0.9"/>
  <circle cx="27" cy="26.6" r="1.7" fill="#ffffff" fill-opacity="0.9"/>
</svg>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
