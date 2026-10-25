"""Visual tokens shared by the charts, the tree drawings and the app theme.

The categorical palette and its order were validated for colour-vision
deficiency separation and contrast against the page surface; extend it by
folding extra classes into a neutral grey, never by generating new hues.
"""

from __future__ import annotations

PAGE = "#fbfbf9"
SURFACE = "#ffffff"
SUNKEN = "#f3f2ee"
INK = "#1a1a19"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e6e5df"
AXIS = "#c3c2b7"
BORDER = "#dedcd4"
ACCENT = "#1f5c46"

ALGORITHM_COLORS = {
    "ID3": "#2a78d6",
    "C4.5": "#eb6834",
    "CART (scikit-learn)": "#898781",
}

CLASS_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
OTHER_CLASS_COLOR = "#b5b3ab"

# Single-hue blue ramp, near-surface to dark, for magnitudes (confusion matrices).
SEQUENTIAL = ("#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def class_color(index: int) -> str:
    """Colour of the ``index``-th class; classes past the eighth share grey."""
    return CLASS_COLORS[index] if index < len(CLASS_COLORS) else OTHER_CLASS_COLOR


def algorithm_color(name: str) -> str:
    return ALGORITHM_COLORS.get(name, INK_MUTED)


def mix(color: str, other: str = SURFACE, amount: float = 0.5) -> str:
    """Blend ``color`` toward ``other``; ``amount`` 0 keeps ``color``, 1 gives ``other``."""
    a, b = _rgb(color), _rgb(other)
    return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(a, b))


def text_on(fill: str) -> str:
    """White or ink, whichever reads better on ``fill``."""
    r, g, b = (c / 255 for c in _rgb(fill))
    luminance = 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)
    return "#ffffff" if luminance < 0.36 else INK


def _rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
