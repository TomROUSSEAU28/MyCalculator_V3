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

Colours live in PLOT_THEMES (chrome) and PLOT_SERIES (the categorical slots).
The theme names match those of SRC/OTHERS/terminal.py, so a report and its
figures can be set to the same one and look like they belong together.

Adding a theme is NOT a matter of taste — see the note above PLOT_SERIES.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter

__all__ = [
    "FONT",
    "PLOT_SERIES",
    "PLOT_THEMES",
    "add_title",
    "annotate",
    "axis_labels",
    "engineering_ticks",
    "figure_to_svg",
    "grid",
    "legend",
    "new_figure",
    "new_grid",
    "reference_line",
    "save_figure",
    "series",
    "style",
    "vertical_reference_line",
]


PLOT_SERIES: dict[str, list[str]] = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    "paper": ["#2f6f9f", "#c05a2b", "#2f7d63", "#a2801c", "#a75277", "#4b7a2e"],
    "blueprint": ["#1f5f9c", "#c25a25", "#0f7a6c", "#9a7714", "#7a4fa8", "#a03f5e"],
    "sage": ["#c0532c", "#3a6ea5", "#2c6a52", "#96771a", "#8a4a76", "#5b6b7a"],
}


PLOT_THEMES: dict[str, dict[str, str]] = {
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
    "paper": {
        "surface": "#f7f3ea",
        "ink": "#14120d",
        "ink_secondary": "#57534a",
        "muted": "#8b8578",
        "grid": "#e5dfd1",
        "axis": "#cfc7b4",
        "critical": "#c0392b",
    },
    "blueprint": {
        "surface": "#eaf1f7",
        "ink": "#0d2438",
        "ink_secondary": "#3d566b",
        "muted": "#7b8fa1",
        "grid": "#d7e2ea",
        "axis": "#b6c6d3",
        "critical": "#c0392b",
    },
    "sage": {
        "surface": "#edf0e9",
        "ink": "#12160f",
        "ink_secondary": "#4d5449",
        "muted": "#838a7c",
        "grid": "#dde2d8",
        "axis": "#c3cabb",
        "critical": "#c0392b",
    },
}
FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def style(theme: str) -> dict[str, str]:
    """The chrome of a theme, by role name. Raises on an unknown theme."""
    if theme not in PLOT_THEMES:
        known = ", ".join(repr(name) for name in PLOT_THEMES)
        raise ValueError(f"theme must be one of {known}, got {theme!r}")
    return PLOT_THEMES[theme]


def series(theme: str) -> list[str]:
    """The categorical slots of a theme, in their validated order."""
    style(theme)  # same error for the same mistake, whichever way you ask
    return PLOT_SERIES[theme]


# ============================================================================ #
#  Chrome
# ============================================================================ #


def _dress(ax: Axes, c: dict[str, str]) -> None:
    """
    Recessive chrome on one axes: hairline, solid, one step off the surface.

    The data is the only thing allowed to be loud. Shared by new_figure() and
    new_grid(), so a panel of a grid is indistinguishable from a lone figure.
    """
    ax.set_facecolor(c["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT)


def new_figure(theme: str, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """A themed figure and its axes, chrome already recessive."""
    c = style(theme)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(c["surface"])
    _dress(ax, c)
    return fig, ax


def new_grid(
    theme: str,
    figsize: tuple[float, float],
    rows: int = 2,
    columns: int = 1,
    height_ratios: Sequence[float] | None = None,
    share_x: bool = False,
) -> tuple[Figure, list[Axes]]:
    """
    Several themed panels on one figure, returned flat in reading order.

    For the case where two views answer ONE question and have to be read
    together — a signal and its spectrum, a waveform and its error. Two plots
    that merely happen to be about the same part belong in two figures.

    height_ratios : relative panel heights, e.g. (3, 2) to give the top panel
                    more room. One entry per row.
    share_x       : couple the x axes — only when the panels really share a
                    quantity AND a range, otherwise it silently rescales one.
    """
    c = style(theme)
    if height_ratios is not None and len(height_ratios) != rows:
        raise ValueError(
            f"height_ratios needs {rows} entries, got {len(height_ratios)}"
        )

    fig, _ = plt.subplots(
        rows,
        columns,
        figsize=figsize,
        sharex=share_x,
        gridspec_kw=None if height_ratios is None else {"height_ratios": height_ratios},
    )
    fig.patch.set_facecolor(c["surface"])
    # fig.axes rather than the returned array: always a flat list, whatever the
    # shape, so a caller never has to know if it got a 1-D or 2-D grid back.
    for ax in fig.axes:
        _dress(ax, c)
    return fig, fig.axes


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

    Call it LAST, after legend(): both write into the strip above the axes, and
    the title makes room for the legend only if the legend is already there.
    The reading order that comes out is title, subtitle, legend, then the data.
    """
    c = style(theme)
    # 22 points of pad clears a subtitle, 12 clears nothing; a legend() strip
    # is one 9 pt line plus its air.
    pad = 22 if subtitle else 12
    if ax.get_legend() is not None:
        pad += 16
    ax.set_title(
        title,
        color=c["ink"],
        fontsize=13,
        fontweight="bold",
        fontfamily=FONT,
        loc="left",
        pad=pad,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            # Same 16 points of clearance: the subtitle sits between the title
            # and the legend, not on top of it.
            xytext=(0, 9 if ax.get_legend() is None else 25),
            xycoords="axes fraction",
            textcoords="offset points",
            color=c["ink_secondary"],
            fontsize=10,
            fontfamily=FONT,
            va="bottom",
        )


def legend(
    ax: Axes,
    theme: str,
    entries: Sequence[tuple[str, str]],
    kind: str = "patch",
    columns: int | None = None,
) -> None:
    """
    A frameless legend above the axes, built from explicit (label, colour) pairs.

    Only for a figure that cannot be direct-labelled — 3+ series that touch
    (stacked bars, overlapping lines). One or two series: use annotate(), or
    say it in the subtitle.

    kind "patch" : bars, areas, stacks.   kind "line" : curves.

    Never list a reference_line() here: `critical` is a status colour, and
    the rule already carries its own label.
    """
    c = style(theme)
    if kind not in ("patch", "line"):
        raise ValueError(f"kind must be 'patch' or 'line', got {kind!r}")

    handles = [
        Patch(facecolor=colour, edgecolor="none", label=label)
        if kind == "patch"
        else Line2D([], [], color=colour, linewidth=2.0, label=label)
        for label, colour in entries
    ]

    box = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),  # sit above the axes, aligned with the title
        ncols=columns or len(handles),
        frameon=False,  # the box is chrome we already removed
        borderpad=0,
        handlelength=1.1,
        handletextpad=0.6,
        columnspacing=1.5,
        labelcolor=c["ink_secondary"],
        fontsize=9,
    )
    for text in box.get_texts():
        text.set_fontfamily(FONT)


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


def engineering_ticks(
    ax: Axes,
    theme: str,
    axis: str = "x",
    unit: str = "",
    places: int | None = None,
) -> None:
    """
    Tick labels in engineering notation: 1e-05 becomes "10 µs", 2.4e6 "2.4 MHz".

    Mandatory on any electronics axis. A time base of a few microseconds and a
    spectrum reaching megahertz are unreadable in either scientific notation or
    plain decimals — and a caller who rescales the DATA to get nice numbers
    (dividing by 1e-6 to "plot in µs") then has to keep the unit of the label
    and the unit of the values in sync by hand. Plot in SI, label here.

    unit   : appended to every tick, e.g. "s", "Hz", "V", "A".
    places : digits after the decimal point. None keeps matplotlib's own
             choice, which drops the trailing zeros.
    """
    style(theme)  # same error for the same mistake, whichever way you ask
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    # Thin space before the unit: "10 µs" reads as one quantity, "10µs" does
    # not, and a normal space is wide enough to look like two tokens.
    target.set_major_formatter(EngFormatter(unit=unit, places=places, sep=" "))


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
    va: str = "baseline",
) -> None:
    """
    A direct label in data coordinates, nudged by `offset` in points.

    Direct labels are how a figure here stays readable without a tooltip and
    without a legend — and they are the mandatory relief wherever a series
    colour sits below 3:1 on its surface.

    role : key of PLOT_THEMES ("ink", "ink_secondary", "muted", "critical"...).
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
        verticalalignment=va,
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
        annotate(ax, theme, label, (0, y), offset or (2, 6), role="critical", bold=True)
    elif kind == "context":
        ax.axhline(y, color=c["axis"], linewidth=1.0, zorder=1)
        annotate(ax, theme, label, (0, y), offset or (8, -14), role="muted")
    else:
        raise ValueError(f"kind must be 'limit' or 'context', got {kind!r}")


def vertical_reference_line(
    ax: Axes,
    theme: str,
    x: float,
    label: str,
    kind: str = "context",
    offset: tuple[float, float] = (4, -4),
) -> None:
    """
    A labelled vertical rule — reference_line(), turned a quarter turn.

    Same two kinds, same reason for each: "limit" for a real threshold (a
    bandwidth, a maximum frequency), "context" for the abscissa a reading is
    relative to (the fundamental of a spectrum, an edge of a waveform).

    The label rides at the top of the axes rather than at a y value, because
    the rule spans the whole height and has no natural one.
    """
    c = style(theme)
    if kind == "limit":
        ax.axvline(x, color=c["critical"], linewidth=1.5, linestyle="--", zorder=2)
        role, bold = "critical", True
    elif kind == "context":
        ax.axvline(x, color=c["axis"], linewidth=1.0, zorder=1)
        role, bold = "muted", False
    else:
        raise ValueError(f"kind must be 'limit' or 'context', got {kind!r}")

    ax.annotate(
        label,
        xy=(x, 1.0),
        # x in data coordinates, y in axes fraction: the label tracks the rule
        # sideways and stays pinned to the top edge however the y limits move.
        xycoords=ax.get_xaxis_transform(),
        xytext=offset,
        textcoords="offset points",
        color=c[role],
        fontsize=9,
        fontweight="bold" if bold else "normal",
        fontfamily=FONT,
        verticalalignment="top",
    )


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
# Typed loosely on purpose: matplotlib's stubs want a Literal union for the
# keys, which a plain dict of strings cannot satisfy.
_EXPORT_RC: dict[Any, Any] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "pdf.compression": 6,
}


def figure_to_svg(fig: Figure) -> str:
    """
    The figure as a standalone SVG string, cropped and themed like a saved one.

    Same settings as save_figure(), nothing written to disk — for embedding a
    figure into a page or a notebook cell (see SRC/OTHERS/notebook.py).
    """
    buffer = io.StringIO()
    with mpl.rc_context(_EXPORT_RC):
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
    return buffer.getvalue()


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
