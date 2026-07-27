"""
Plots for the MOSFET loss model.

Two figures, matching the two questions the model answers:
  plot_loss_breakdown()   -> where does the power go?
  plot_thermal_iteration()-> where does the junction temperature settle?

Both accept either the single-MOSFET result or the half-bridge one, and both
render in a light or a dark theme.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from SRC.MOSFET.mosfet_loss import (
    HALF_BRIDGE_LOSS_RESULT,
    HALF_BRIDGE_THERMAL_RESULT,
    MOSFET_LOSS_RESULT,
    THERMAL_ITERATION_RESULT,
)

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


def _components(result: MOSFET_LOSS_RESULT, p_rr_extra: float = 0.0) -> dict[str, float]:
    """
    The six loss buckets, summing exactly to what the device dissipates.

    p_rr_extra is the recovery share allocated to this device's OWN body diode
    by the half-bridge model — it lives outside MOSFET_LOSS_RESULT because a
    lone MOSFET has no opposite switch to be triggered by.
    """
    return {
        "p_cond": result.p_cond,
        "p_sw": result.p_sw,
        "p_coss": result.p_coss,
        "p_body_cond": result.p_body_cond,
        "p_rr": result.p_body_rr + p_rr_extra,
        "p_gate_int": result.p_gate_int,
    }


def _devices(
    result: MOSFET_LOSS_RESULT | HALF_BRIDGE_LOSS_RESULT,
) -> list[tuple[str, dict[str, float], float]]:
    """Normalise both result types to [(device name, buckets, total), ...]."""
    if isinstance(result, HALF_BRIDGE_LOSS_RESULT):
        return [
            ("High side", _components(result.high_side, result.p_rr_diode_hs), result.p_hs_total),
            ("Low side", _components(result.low_side, result.p_rr_diode_ls), result.p_ls_total),
        ]
    return [("MOSFET", _components(result), result.p_total)]


def loss_table(result: MOSFET_LOSS_RESULT | HALF_BRIDGE_LOSS_RESULT) -> pd.DataFrame:
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
    result: MOSFET_LOSS_RESULT | HALF_BRIDGE_LOSS_RESULT,
    title: str = "Loss breakdown",
    subtitle: str | None = None,
    theme: str = "light",
) -> Figure:
    """
    Where the power goes.

    One MOSFET  -> a horizontal bar per loss mechanism, sorted worst first, all
                   in one colour. The categories have no natural order, so
                   colouring them by size would only re-encode the bar length.
    Half bridge -> one stacked bar per device, segments coloured by mechanism,
                   so HS and LS are directly comparable at a glance.
    """
    c = _style(theme)
    palette = _SERIES[theme]
    devices = _devices(result)

    if len(devices) == 1:
        # --- single MOSFET: magnitude comparison, one series ------------------
        _, buckets, total = devices[0]
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

    else:
        # --- half bridge: part-to-whole, two devices --------------------------
        fig, ax = _new_figure(theme, (9.0, 3.2))
        names = [name for name, _, _ in devices]
        y_pos = range(len(devices))

        left = [0.0] * len(devices)
        for slot, (key, _label) in enumerate(_LOSS_KEYS):
            widths = [buckets[key] for _, buckets, _ in devices]
            ax.barh(
                list(y_pos),
                widths,
                left=left,
                height=0.28,
                color=palette[slot],
                zorder=3,
                # 2px of surface separates touching segments — white does the
                # separating, never a stroke drawn around the mark.
                edgecolor=c["surface"],
                linewidth=2.0,
            )
            left = [a + b for a, b in zip(left, widths)]

        span = max(left) if max(left) > 0 else 1.0
        for y, (_, _, total) in enumerate(devices):
            ax.annotate(
                f"{total:.3f} W",
                xy=(total, y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=c["ink"],
                fontfamily=_FONT,
            )
        ax.set_yticks(list(y_pos), names)
        ax.invert_yaxis()
        ax.set_xlim(0, span * 1.14)
        ax.set_xlabel("Power [W]", color=c["muted"], fontsize=9, fontfamily=_FONT)
        ax.xaxis.grid(True, color=c["grid"], linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        for label in ax.get_yticklabels():
            label.set_color(c["ink_secondary"])
            label.set_fontsize(10)

        # Interior segments get no inline label — they would be clipped. The
        # legend carries identity, loss_table() carries the numbers.
        legend = ax.legend(
            handles=[
                Patch(facecolor=palette[slot], label=label)
                for slot, (_, label) in enumerate(_LOSS_KEYS)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            # 2 columns, because matplotlib fills a legend column-major: this
            # is the only ncol where reading order matches the stack order.
            ncol=2,
            frameon=False,
            fontsize=9,
            handlelength=1.6,
            columnspacing=3.0,
        )
        for text in legend.get_texts():
            text.set_color(c["ink_secondary"])
            text.set_fontfamily(_FONT)

        total_leg = sum(total for _, _, total in devices)
        _title(ax, theme, title, subtitle or f"Leg total: {total_leg:.3f} W")

    fig.tight_layout()
    return fig


def plot_thermal_iteration(
    result: THERMAL_ITERATION_RESULT | HALF_BRIDGE_THERMAL_RESULT,
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
    half_bridge = isinstance(result, HALF_BRIDGE_THERMAL_RESULT)

    # Iteration 0 is the seed, T_ambient — prepending it shows the whole climb
    # instead of dropping the reader into an already-hot plot.
    t_a = result.t_ambient
    if half_bridge:
        series = [
            ("High side", [t_a] + [hs for hs, _ in result.history], palette[0]),
            ("Low side", [t_a] + [ls for _, ls in result.history], palette[1]),
        ]
    else:
        series = [("Junction", [t_a] + list(result.history), palette[0])]

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

    if len(series) > 1:
        legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(c["ink_secondary"])
            text.set_fontfamily(_FONT)

    if subtitle is None:
        if result.converged:
            subtitle = f"Converged in {result.iterations} iterations"
        else:
            subtitle = f"DID NOT CONVERGE in {result.iterations} iterations — thermal runaway"
    _title(ax, theme, title, subtitle)

    fig.tight_layout()
    return fig


# endregion
