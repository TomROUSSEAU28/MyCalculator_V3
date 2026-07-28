"""
Figure layout and export, built on matplotlib.

Nothing domain-specific here: the themes, the chrome, and the export path that
every figure of the project shares. A plotting module of its own (MOSFET,
thermal, magnetics...) imports these, then only writes what its data means —
the same split as SRC/OTHERS/terminal.py, which does this for the report side.

    fig, ax = new_figure("dark", (8.0, 4.4))
    ax.barh(labels, values, height=0.3, color=series("dark")[0], zorder=3)
    grid(ax, "dark", axis="x")
    add_title(ax, "dark", "Loss breakdown", "total 4.81 W")
    save_figure(fig, OUTPUT / "losses")

Colours live in THEMES (chrome) and SERIES (the categorical slots). The theme
names match those of SRC/OTHERS/terminal.py, so a report and its figures can be
set to the same one and look like they belong together.

Adding a theme is NOT a matter of taste — see the note above SERIES.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "FONT",
    "SERIES",
    "THEMES",
    "add_title",
    "annotate",
    "axis_labels",
    "grid",
    "new_figure",
    "reference_line",
    "save_figure",
    "series",
    "style",
]


# ============================================================================ #
#  Themes
# ============================================================================ #

# Validated categorical slots. The ORDER is the colour-blind-safety mechanism,
# not decoration — adjacent pairs are the ones a stacked bar puts side by side,
# and this order is what clears the separation gates. Do not reorder to taste.
#
# The three accent themes ("red", "blue", "vivid") draw the SAME hue steps as
# "dark" — only the order changes, so slot 1, the colour most figures actually
# plot, sets the mood. Each order was picked by enumerating the permutations
# and keeping only those clearing every gate against that theme's own surface;
# the worst adjacent pair lands at CVD ΔE 13.0-13.2 (target 8) and normal-vision
# ΔE 19.3 (floor 15). Re-run the enumeration if you change a step.
#
# Two limits a new plotting module has to know:
#   - six slots is the cap for bars/lines/stacks, where only neighbours touch.
#     For a scatter or a map ANY two marks can sit side by side, and six of
#     these cannot be told apart pairwise (worst pair CVD ΔE 1.9). Past three
#     series there, fold the rest into "Other" or facet — do not add hues.
#   - three of the light-mode slots sit below 3:1 against their surface, so the
#     palette may only carry meaning next to visible labels or a table.
SERIES: dict[str, list[str]] = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    "red": ["#e66767", "#9085e9", "#c98500", "#d55181", "#3987e5", "#d95926"],
    "blue": ["#3987e5", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    "vivid": ["#d55181", "#c98500", "#9085e9", "#e66767", "#3987e5", "#d95926"],
}

# Chrome. The accent themes tint their near-black surface toward the accent
# hue; ink and labels are stepped to clear the "dark" theme's contrast (muted
# text 4.85:1), so nothing gets harder to read in exchange for the mood.
#
# `critical` is a status colour, not a series one: it is FIXED across every
# theme, because a reader has to be able to learn "that red rule is the limit"
# once and have it hold everywhere. Same rule as the alert colours of
# SRC/OTHERS/terminal.py. Never spend it on a series.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "critical": "#d03b3b",
    },
    # Red on a warm black. Caveat worth knowing: the series red and the fixed
    # `critical` red are the same hue family (ΔE 9.9, under the 15 series
    # floor), so a figure drawing both has to separate them by more than
    # colour — a dashed rule and labels, as reference_line() does.
    "red": {
        "surface": "#151011",
        "ink": "#ffffff",
        "ink_secondary": "#cdc0c0",
        "muted": "#9a8d8d",
        "grid": "#2a2021",
        "axis": "#3a2c2d",
        "critical": "#d03b3b",
    },
    # Blue on a cool black. The one accent theme with no status clash: its
    # slot 1 sits ΔE 31.9 off the limit red.
    "blue": {
        "surface": "#0e1117",
        "ink": "#ffffff",
        "ink_secondary": "#c2cad6",
        "muted": "#8b95a3",
        "grid": "#1e2530",
        "axis": "#2b3440",
        "critical": "#d03b3b",
    },
    # The loud one: the highest-chroma steps first, on a plum black.
    "vivid": {
        "surface": "#121014",
        "ink": "#ffffff",
        "ink_secondary": "#cec7d6",
        "muted": "#968ea1",
        "grid": "#241f2a",
        "axis": "#332c3b",
        "critical": "#d03b3b",
    },
}

FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def style(theme: str) -> dict[str, str]:
    """The chrome of a theme, by role name. Raises on an unknown theme."""
    if theme not in THEMES:
        known = ", ".join(repr(name) for name in THEMES)
        raise ValueError(f"theme must be one of {known}, got {theme!r}")
    return THEMES[theme]


def series(theme: str) -> list[str]:
    """The categorical slots of a theme, in their validated order."""
    style(theme)  # same error for the same mistake, whichever way you ask
    return SERIES[theme]


# ============================================================================ #
#  Chrome
# ============================================================================ #


def new_figure(theme: str, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """A themed figure and its axes, chrome already recessive."""
    c = style(theme)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    # Recessive chrome: hairline, solid, one step off the surface. The data is
    # the only thing allowed to be loud.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT)
    return fig, ax


def add_title(
    ax: Axes,
    theme: str,
    title: str,
    subtitle: str | None = None,
) -> None:
    """
    Left-aligned title, with the subtitle set under it.

    The subtitle is where a single-series figure says what is plotted — which
    is what lets it skip the legend box entirely.
    """
    c = style(theme)
    ax.set_title(
        title,
        color=c["ink"],
        fontsize=13,
        fontweight="bold",
        fontfamily=FONT,
        loc="left",
        pad=22 if subtitle else 12,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xytext=(0, 9),
            xycoords="axes fraction",
            textcoords="offset points",
            color=c["ink_secondary"],
            fontsize=10,
            fontfamily=FONT,
            va="bottom",
        )


def axis_labels(
    ax: Axes,
    theme: str,
    x: str | None = None,
    y: str | None = None,
) -> None:
    """Axis titles, in the muted ink. Pass only the ones you need."""
    c = style(theme)
    if x is not None:
        ax.set_xlabel(x, color=c["muted"], fontsize=9, fontfamily=FONT)
    if y is not None:
        ax.set_ylabel(y, color=c["muted"], fontsize=9, fontfamily=FONT)


def grid(ax: Axes, theme: str, axis: str = "y") -> None:
    """
    Gridlines on one axis only — the one the reader measures along.

    Always behind the data: a gridline crossing a mark reads as part of it.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    c = style(theme)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.grid(True, color=c["grid"], linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def annotate(
    ax: Axes,
    theme: str,
    text: str,
    xy: tuple[float, float],
    offset: tuple[float, float] = (6, 0),
    role: str = "ink_secondary",
    size: float = 9,
    bold: bool = False,
    va: str | None = None,
) -> None:
    """
    A direct label in data coordinates, nudged by `offset` in points.

    Direct labels are how a figure here stays readable without a tooltip and
    without a legend — and they are the mandatory relief wherever a series
    colour sits below 3:1 on its surface.

    role : key of THEMES ("ink", "ink_secondary", "muted", "critical"...).
           Text wears text colours; it never takes a series colour, or it
           starts competing with the marks for identity.
    """
    c = style(theme)
    if role not in c:
        raise ValueError(f"unknown role {role!r} (available: {', '.join(c)})")
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        fontsize=size,
        fontweight="bold" if bold else "normal",
        color=c[role],
        fontfamily=FONT,
        **({"va": va} if va else {}),
    )


def reference_line(
    ax: Axes,
    theme: str,
    y: float,
    label: str,
    kind: str = "context",
    offset: tuple[float, float] | None = None,
) -> None:
    """
    A labelled horizontal rule.

    kind "limit"   : a real threshold (a rating, a budget). Drawn dashed in the
                     fixed `critical` colour and labelled in bold — a dashed
                     rule is only honest when the line really is a limit.
    kind "context" : the baseline a reading is relative to (an ambient, a
                     nominal). Solid, in the axis colour, so it recedes.

    The label is what keeps a limit distinguishable from a series of the same
    hue family — see the "red" theme note above. Do not draw the rule without it.
    """
    c = style(theme)
    if kind == "limit":
        ax.axhline(y, color=c["critical"], linewidth=1.5, linestyle="--", zorder=2)
        annotate(
            ax, theme, label, (0, y), offset or (2, 6), role="critical", bold=True
        )
    elif kind == "context":
        ax.axhline(y, color=c["axis"], linewidth=1.0, zorder=1)
        annotate(ax, theme, label, (0, y), offset or (8, -14), role="muted")
    else:
        raise ValueError(f"kind must be 'limit' or 'context', got {kind!r}")


# ============================================================================ #
#  Export
# ============================================================================ #

# Vector formats: the figure is stored as geometry, so it stays sharp at any
# zoom and any print size. `dpi` is irrelevant for them (it only sets the
# nominal size), which is why PNG is the odd one out below.
_VECTOR_FORMATS = frozenset({"svg", "pdf", "eps"})
_RASTER_FORMATS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "webp"})

# Text-as-text, not text-as-outlines:
#   pdf/ps.fonttype 42 -> embed TrueType, so the PDF stays searchable and the
#                         labels remain editable in Illustrator / Inkscape.
#   svg.fonttype "none" -> the SVG references the font by name instead of
#                         converting it to paths. Smaller and editable, but the
#                         viewer needs the font installed (see FONT: Segoe UI
#                         on Windows, DejaVu Sans as fallback).
_EXPORT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "pdf.compression": 6,
}


def save_figure(
    fig: Figure,
    path: str | Path,
    formats: tuple[str, ...] = ("png", "svg", "pdf"),
    dpi: int = 300,
    transparent: bool = False,
) -> list[Path]:
    """
    Write a figure once per requested format, at publication quality.

    path    : destination WITHOUT extension (any extension given is dropped) —
              one file per entry of `formats` is written next to it.
    formats : "svg" / "pdf" / "eps" are vector (infinite zoom, the right choice
              for a report, LaTeX, or Word); "png" and the other raster formats
              honour `dpi`.
    dpi     : raster resolution. 300 is print quality, 600 for a figure that
              will be blown up. Ignored by the vector formats.
    transparent : drop the theme surface so the figure takes the background of
              the page it is dropped into. Careful with the dark themes — light
              text on a white page is unreadable.

    Returns the paths actually written.
    """
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    unknown = [f for f in formats if f.lower() not in _VECTOR_FORMATS | _RASTER_FORMATS]
    if unknown:
        raise ValueError(f"unsupported format(s): {', '.join(unknown)}")

    written: list[Path] = []
    with mpl.rc_context(_EXPORT_RC):
        for fmt in formats:
            fmt = fmt.lower()
            target = base.with_suffix(f".{fmt}")
            fig.savefig(
                target,
                format=fmt,
                # Vector formats ignore dpi for the geometry, but it still sets
                # the size any rasterised element is written at.
                dpi=dpi,
                # Crop to the ink: kills the whitespace tight_layout leaves and
                # rescues a long label that would otherwise be clipped.
                bbox_inches="tight",
                pad_inches=0.06,
                transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor(),
                edgecolor="none",
            )
            written.append(target)
    return written
