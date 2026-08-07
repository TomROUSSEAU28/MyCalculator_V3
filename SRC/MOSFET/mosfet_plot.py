"""
Plots for the MOSFET loss model.

Two figures, matching the two questions the model answers:
  plot_loss_breakdown()   -> where does the power go?
  plot_thermal_iteration()-> where does the junction temperature settle?

Only what is MOSFET-specific lives here. The themes, the chrome and the export
path are shared by every figure of the project and sit in SRC/OTHERS/plot.py —
including save_figure(), which is where you import it from.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from SRC.MOSFET.mosfet_loss import MOSFET_LOSS_RESULT, THERMAL_ITERATION_RESULT
from SRC.OTHERS.plot import (
    add_title,
    annotate,
    axis_labels,
    engineering_ticks,
    grid,
    new_figure,
    quantity,
    reference_line,
    series,
    style,
)

__all__ = [
    "loss_table",
    "plot_loss_breakdown",
    "plot_thermal_iteration",
]

# region data extraction

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
    palette = series(theme)
    _, buckets, total = _devices(result)[0]

    items = sorted(
        ((label, buckets[key]) for key, label in _LOSS_KEYS),
        key=lambda kv: kv[1],
    )
    labels = [label for label, _ in items]
    values = [value for _, value in items]

    fig, ax = new_figure(theme, (8.0, 4.4))
    # Thin marks: the bar never fills its band, the leftover is air.
    ax.barh(labels, values, height=0.3, color=palette[0], zorder=3)

    span = max(values) if max(values) > 0 else 1.0
    for y, value in enumerate(values):
        # Bars carry their value at the tip — no tooltip to hide behind. In
        # engineering notation, like the axis: a gate loss of 0.004 W and a
        # conduction loss of 4.2 W belong on the same chart, and only "4 mW"
        # next to "4.2 W" keeps both of them readable.
        annotate(ax, theme, quantity(value, "W"), (value, y), (6, 0), va="center")
    ax.set_xlim(0, span * 1.18)
    engineering_ticks(ax, theme, axis="x", unit="W")
    axis_labels(ax, theme, x="Power")
    grid(ax, theme, axis="x")
    for label in ax.get_yticklabels():
        label.set_color(style(theme)["ink_secondary"])
        label.set_fontsize(10)
    # A single series needs no legend box; the subtitle names what is plotted.
    add_title(
        ax,
        theme,
        title,
        subtitle or f"Total dissipated in the die: {quantity(total, 'W')}",
    )

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
    c = style(theme)
    palette = series(theme)

    # Iteration 0 is the seed, T_ambient — prepending it shows the whole climb
    # instead of dropping the reader into an already-hot plot.
    curves = [("Junction", [result.t_ambient] + list(result.history), palette[0])]
    steps = range(len(curves[0][1]))

    fig, ax = new_figure(theme, (8.2, 4.4))
    grid(ax, theme, axis="y")

    # Ambient is where every trajectory starts — context, not a series.
    reference_line(ax, theme, result.t_ambient, f"Ambient {result.t_ambient:.0f} °C")

    if t_j_max is not None:
        reference_line(ax, theme, t_j_max, f"T_j,max {t_j_max:.0f} °C", kind="limit")

    for name, values, colour in curves:
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

    axis_labels(ax, theme, x="Iteration", y="Junction temperature [°C]")
    ax.set_xlim(-0.25, len(curves[0][1]) - 1 + 0.75)
    ax.set_xticks(list(steps))

    # Label the endpoints only — the value the reader actually came for. Done
    # after the limits are settled so a label sitting on top of the T_j,max
    # rule can be nudged clear of it instead of overprinting it.
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for _name, values, _colour in curves:
        final = values[-1]
        dy = 0
        if t_j_max is not None and abs(final - t_j_max) < 0.07 * y_span:
            dy = -13 if final <= t_j_max else 13
        annotate(
            ax,
            theme,
            f"{final:.1f} °C",
            (len(values) - 1, final),
            (8, dy),
            role="ink",
            size=10,
            bold=True,
            va="center",
        )

    # One curve only, so no legend box: the subtitle says what is plotted.
    if subtitle is None:
        if result.converged:
            subtitle = f"Converged in {result.iterations} iterations"
        else:
            subtitle = (
                f"DID NOT CONVERGE in {result.iterations} iterations — thermal runaway"
            )
    add_title(ax, theme, title, subtitle)

    fig.tight_layout()
    return fig


# endregion
