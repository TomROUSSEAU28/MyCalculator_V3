"""
Plots for the MOSFET loss model.

Two figures, matching the two questions the model answers:
  plot_loss_breakdown()   -> where does the power go?
  plot_thermal_iteration()-> where does the junction temperature settle?

Both render in a light or a dark theme.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from SRC.MOSFET.mosfet_loss import MOSFET_LOSS_RESULT, THERMAL_ITERATION_RESULT

# region theme

# Validated categorical slots. The ORDER is the colour-blind-safety mechanism,
# not decoration — adjacent pairs are the ones a stacked bar puts side by side,
# and this order is what clears the separation gates. Do not reorder to taste.
_SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
}

_THEME = {
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
}

# Loss categories, in a fixed order so a colour always means the same thing
# whichever figure you are looking at.
_LOSS_KEYS: list[tuple[str, str]] = [
    ("p_cond", "Conduction"),
    ("p_sw", "Switching"),
    ("p_coss", "Coss"),
    ("p_body_cond", "Body diode conduction"),
    ("p_rr", "Reverse recovery"),
    ("p_gate_int", "Gate (internal Rg)"),
]

_FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def _style(theme: str) -> dict:
    if theme not in _THEME:
        raise ValueError(f"theme must be 'light' or 'dark', got {theme!r}")
    return _THEME[theme]


def _new_figure(theme: str, figsize: tuple[float, float]) -> tuple[Figure, plt.Axes]:
    c = _style(theme)
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
        label.set_fontfamily(_FONT)
    return fig, ax


def _title(ax: plt.Axes, theme: str, title: str, subtitle: str | None = None) -> None:
    c = _style(theme)
    ax.set_title(
        title,
        color=c["ink"],
        fontsize=13,
        fontweight="bold",
        fontfamily=_FONT,
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
            fontfamily=_FONT,
            va="bottom",
        )


# endregion


# region data extraction


def _components(result: MOSFET_LOSS_RESULT) -> dict[str, float]:
    """The six loss buckets, summing exactly to what the device dissipates."""
    return {
        "p_cond": result.p_cond,
        "p_sw": result.p_sw,
        "p_coss": result.p_coss,
        "p_body_cond": result.p_body_cond,
        "p_rr": result.p_body_rr,
        "p_gate_int": result.p_gate_int,
    }


def _devices(result: MOSFET_LOSS_RESULT) -> list[tuple[str, dict[str, float], float]]:
    """Normalise to the [(device name, buckets, total), ...] shape the plots use."""
    return [("MOSFET", _components(result), result.p_total)]


def loss_table(result: MOSFET_LOSS_RESULT) -> pd.DataFrame:
    """
    The chart's table twin — every plotted value in readable form.

    Three of the light-mode series colours sit below 3:1 against the surface,
    so the palette is only allowed to carry meaning alongside visible labels or
    this table. Print it next to the figure; never let colour be the only way
    to read a number.
    """
    devices = _devices(result)
    frame = pd.DataFrame(
        {name: [buckets[key] for key, _ in _LOSS_KEYS] for name, buckets, _ in devices},
        index=[label for _, label in _LOSS_KEYS],
    )
    frame.loc["TOTAL"] = [total for _, _, total in devices]
    frame.index.name = "Loss [W]"
    return frame.round(4)


# endregion


# region plots


def plot_loss_breakdown(
    result: MOSFET_LOSS_RESULT,
    title: str = "Loss breakdown",
    subtitle: str | None = None,
    theme: str = "light",
) -> Figure:
    """
    Where the power goes.

    A horizontal bar per loss mechanism, sorted worst first, all in one colour.
    The categories have no natural order, so colouring them by size would only
    re-encode the bar length.
    """
    c = _style(theme)
    palette = _SERIES[theme]
    _, buckets, total = _devices(result)[0]

    items = sorted(
        ((label, buckets[key]) for key, label in _LOSS_KEYS),
        key=lambda kv: kv[1],
    )
    labels = [label for label, _ in items]
    values = [value for _, value in items]

    fig, ax = _new_figure(theme, (8.0, 4.4))
    # Thin marks: the bar never fills its band, the leftover is air.
    ax.barh(labels, values, height=0.3, color=palette[0], zorder=3)

    span = max(values) if max(values) > 0 else 1.0
    for y, value in enumerate(values):
        # Bars carry their value at the tip — no tooltip to hide behind.
        ax.annotate(
            f"{value:.3f} W",
            xy=(value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=c["ink_secondary"],
            fontfamily=_FONT,
        )
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel("Power [W]", color=c["muted"], fontsize=9, fontfamily=_FONT)
    ax.xaxis.grid(True, color=c["grid"], linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for label in ax.get_yticklabels():
        label.set_color(c["ink_secondary"])
        label.set_fontsize(10)
    # A single series needs no legend box; the subtitle names what is plotted.
    _title(ax, theme, title, subtitle or f"Total dissipated in the die: {total:.3f} W")

    fig.tight_layout()
    return fig


def plot_thermal_iteration(
    result: THERMAL_ITERATION_RESULT,
    t_j_max: float | None = None,
    title: str = "Junction temperature convergence",
    subtitle: str | None = None,
    theme: str = "light",
) -> Figure:
    """
    How Tj settles (or does not).

    The fixed point is only meaningful if you can see it converge, so the whole
    trajectory is plotted, not just the final number. T_j,max is drawn as a
    threshold — the one place a dashed rule is honest, because it really is a
    limit rather than a gridline.

    t_j_max : limit to draw [°C]. Pass the datasheet value; omitted, no limit
              line is drawn.
    """
    c = _style(theme)
    palette = _SERIES[theme]

    # Iteration 0 is the seed, T_ambient — prepending it shows the whole climb
    # instead of dropping the reader into an already-hot plot.
    series = [("Junction", [result.t_ambient] + list(result.history), palette[0])]
    steps = range(0, len(series[0][1]))

    fig, ax = _new_figure(theme, (8.2, 4.4))
    ax.yaxis.grid(True, color=c["grid"], linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    # Ambient is where every trajectory starts — context, not a series.
    ax.axhline(result.t_ambient, color=c["axis"], linewidth=1.0, zorder=1)
    ax.annotate(
        f"Ambient {result.t_ambient:.0f} °C",
        xy=(0, result.t_ambient),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=9,
        color=c["muted"],
        fontfamily=_FONT,
    )

    if t_j_max is not None:
        # A dashed rule is honest here: this really is a limit, not a gridline.
        ax.axhline(t_j_max, color=c["critical"], linewidth=1.5, linestyle="--", zorder=2)
        ax.annotate(
            f"T_j,max {t_j_max:.0f} °C",
            xy=(0, t_j_max),
            xytext=(2, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=c["critical"],
            fontfamily=_FONT,
        )

    for name, values, colour in series:
        ax.plot(
            list(steps),
            values,
            color=colour,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            marker="o",
            markersize=7,
            # 2px surface ring keeps markers legible where the two lines cross
            markeredgecolor=c["surface"],
            markeredgewidth=2.0,
            zorder=4,
            label=name,
        )

    ax.set_xlabel("Iteration", color=c["muted"], fontsize=9, fontfamily=_FONT)
    ax.set_ylabel("Junction temperature [°C]", color=c["muted"], fontsize=9, fontfamily=_FONT)
    ax.set_xlim(-0.25, len(series[0][1]) - 1 + 0.75)
    ax.set_xticks(list(steps))

    # Label the endpoints only — the value the reader actually came for. Done
    # after the limits are settled so a label sitting on top of the T_j,max
    # rule can be nudged clear of it instead of overprinting it.
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for _name, values, _colour in series:
        final = values[-1]
        dy = 0
        if t_j_max is not None and abs(final - t_j_max) < 0.07 * y_span:
            dy = -13 if final <= t_j_max else 13
        ax.annotate(
            f"{final:.1f} °C",
            xy=(len(values) - 1, final),
            xytext=(8, dy),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=c["ink"],
            fontfamily=_FONT,
        )

    # One curve only, so no legend box: the subtitle says what is plotted.
    if subtitle is None:
        if result.converged:
            subtitle = f"Converged in {result.iterations} iterations"
        else:
            subtitle = f"DID NOT CONVERGE in {result.iterations} iterations — thermal runaway"
    _title(ax, theme, title, subtitle)

    fig.tight_layout()
    return fig


# endregion


# region export

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
#                         viewer needs the font installed (see _FONT: Segoe UI
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
              the page it is dropped into. Careful with the dark theme — light
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


# endregion
