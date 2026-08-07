"""
Plots for the Dowell winding model.

The questions a winding is designed against, one figure each:

    plot_layer_stack()      -> what is stacked against what (the cut-away),
                               with the MMF laid over it on request
    plot_mmf_profile()      -> the field the stacking order builds
    plot_winding_section()  -> both, stacked and sharing their abscissa

    plot_resistance_vs_frequency()      -> where the copper stops being copper
    plot_losses_vs_frequency()          -> what that costs, winding by winding
                                           and in total
    plot_current_density_vs_frequency() -> and whether the wire can take it

    layer_table() / winding_table()     -> the numbers behind all of them

The cut-away is the figure the rest depend on: Dowell's loss is driven by the
MMF at the two faces of each layer, so a layer's loss is a property of WHERE it
sits, not of the layer alone. Two windings interleaved and the same two stacked
have identical DC resistance and wildly different AC loss — which is only
visible once the stack and its MMF are drawn side by side.

Reading the cut-away:
    colour  = winding      (primary RED, secondary BLUE)
    hatch   = wire type    (round, square, foil, Litz)
    x       = build-up, core on the left, outside on the right
    y       = window breadth bw

The x axis is NOT to the same scale as y — a real cut of a 0.7 mm build in a
24 mm window is a sliver nobody can read. Layers are drawn as bands of their
true relative thickness, stretched horizontally.


ONE FIGURE PER CONDUCTION PHASE
-------------------------------
Every function here takes ONE `currents` dict and draws the winding as if that
set of currents flowed at once. That is the truth for a forward or a push-pull,
where both windings conduct together. It is NOT the truth for a flyback, where
they never do: the MMF there is not one profile, it is one profile per phase.

So a flyback gets TWO figures — primary alone, then secondary alone — and the
loss is the sum of the two. Each winding's RMS over the whole period already
carries its own duty cycle, so the two phases add directly (see the `phases`
tuple of the demo at the foot of this module). On that transformer:

    ON  (primary 229 mA, secondary 0)     H:  -641 ->    0 ->    0 A/m
    OFF (primary 0, secondary 331 mA)     H:   360 ->  360 ->    0 A/m

Two things only the pair of figures says. In ON the secondary sits at H = 0
across its whole thickness and dissipates nothing — it is shielded. In OFF the
primary sits in the FULL MMF, flat at 360 A/m on both its faces, while carrying
no current at all: 2.2 mW of pure proximity loss, a fifth of that phase, in a
winding that is switched off.

Drawing the two currents together instead does not merely blur that — it gets
the number wrong in the optimistic direction. The opposite polarities partially
cancel, and the single profile comes out at 23.2 mW against the 27.1 mW the two
phases really cost. Loss goes as H squared, so nothing about the combined
figure can be unpicked afterwards.

The split is a rule about FIGURES, and it stops there. It must NOT be carried
into the loss arithmetic: Dowell_Winding_Structure.harmonic_losses() keeps the
complex phasor of every rank, so one call with both windings is exact — for a
flyback as for anything else. Summing one call per phase instead gives 43.7 mW
against the 53.5 mW of the joint call, 18 % low, because it treats the ON -> OFF
transition as if the field went back to zero and rested there. It does not: the
primary's inner face swings from -641 to +360 A/m in one step, and eddy currents
answer to that whole excursion. Diffusion in copper has memory; the two phases
are disjoint in time, their fields are not independent.

Read against that reference, the f_sw-only numbers above are the shape of the
answer and roughly half its size.

plot_resistance_vs_frequency() takes `currents` for the same reason, and its
default (None) is the one-winding-at-a-time convention — which IS the flyback
phase. Give it an operating point for anything that conducts simultaneously:
without it the curve cannot even see interleaving, the main lever there is.
P-S-S-P and P-P-S-S have the same self resistance to the milliohm and a loss
31 % apart.

Only what is winding-specific lives here. The themes, the chrome and the export
path are shared by every figure of the project and sit in SRC/OTHERS/plot.py —
including save_figure(), which is where you import it from.
"""

from __future__ import annotations

import sys
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Doit preceder les imports 'SRC.*' : sans cela le script n'est importable
# qu'en module (python -m SRC.MAGNETIC.dowell_plot), pas en direct.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from SRC.constant import MU0
from SRC.MAGNETIC.dowell import (
    WINDING_TYPE,
    WIRE_TYPE,
    Dowell_Winding_Structure,
    Layer,
    skin_depth,
)
from SRC.OTHERS.plot import (
    FONT,
    add_title,
    annotate,
    axis_labels,
    engineering_ticks,
    grid,
    legend,
    new_figure,
    new_grid,
    quantity,
    reference_line,
    series,
    style,
    vertical_reference_line,
)

__all__ = [
    "layer_table",
    "plot_current_density_vs_frequency",
    "plot_layer_stack",
    "plot_losses_vs_frequency",
    "plot_mmf_profile",
    "plot_resistance_vs_frequency",
    "plot_winding_section",
    "winding_colour",
    "winding_table",
]


# ============================================================================ #
#  Encodings
# ============================================================================ #

# Primary RED, secondary BLUE: the convention of every winding drawing, and the
# one thing a reader is allowed to assume without checking the legend.
#
# The two hues are taken FROM the theme's own palette rather than written in
# hex, so a cut-away still belongs to the same family as the rest of the
# project's figures. The slot holding the red and the slot holding the blue are
# not the same from one theme to the next (see PLOT_SERIES), hence the table.
# Third slot: the neutral a total line takes, chosen to be neither red nor blue.
_ROLE_SLOTS: dict[str, tuple[int, int, int]] = {
    #             red  blue  total
    "light": (1, 0, 5),
    "dark": (1, 0, 5),
    "paper": (1, 0, 5),
    "blueprint": (1, 0, 4),
    "sage": (0, 1, 4),
}

# Texture is the SECOND variable of the cut-away: colour already carries the
# winding, so the wire type has to be readable without it — which is also what
# keeps the figure legible printed in black and white.
#
# Each pattern is the section it stands for, seen the way the cut-away cuts:
#   round  : circles                 square : the grid of a stack of rectangles
#   foil   : flat sheets, edge on    Litz   : a dust of strands
_WIRE_HATCH: dict[WIRE_TYPE, str] = {
    WIRE_TYPE.ROUND: "oo",
    WIRE_TYPE.SQUARE: "++",
    WIRE_TYPE.FOIL: "---",
    WIRE_TYPE.LITZ: "....",
}

_WIRE_LABEL: dict[WIRE_TYPE, str] = {
    WIRE_TYPE.ROUND: "Round",
    WIRE_TYPE.SQUARE: "Square",
    WIRE_TYPE.FOIL: "Foil",
    WIRE_TYPE.LITZ: "Litz",
}


def _slots(theme: str) -> tuple[int, int, int]:
    if theme not in _ROLE_SLOTS:
        known = ", ".join(repr(name) for name in _ROLE_SLOTS)
        raise ValueError(f"no winding colours for theme {theme!r} (known: {known})")
    return _ROLE_SLOTS[theme]


def winding_colour(theme: str, winding: WINDING_TYPE) -> str:
    """The colour a winding wears in every figure of this module."""
    red, blue, _ = _slots(theme)
    palette = series(theme)
    return palette[red if winding is WINDING_TYPE.PRIMARY else blue]


def _total_colour(theme: str) -> str:
    """The neutral of a summed curve — never one of the two winding hues."""
    return series(theme)[_slots(theme)[2]]


def _winding_label(winding: WINDING_TYPE) -> str:
    return winding.value.capitalize()


# ============================================================================ #
#  Geometry
# ============================================================================ #


def _thickness(layer: Layer) -> float:
    """
    Radial build of one layer [m] — the room it eats in the window.

    A Litz layer is as thick as the sqrt(k) strand sub-layers Dowell splits it
    into (Layer.expand()), so summing this over list_of_layers and over
    effective_layers() gives the same total. That identity is what lets the
    MMF staircase, computed on the sub-layers, land exactly on the band edges
    drawn from the parent layers.
    """
    wire = layer.wire
    if wire.wire_type in (WIRE_TYPE.ROUND, WIRE_TYPE.LITZ):
        return len(layer.expand()) * wire.diameter
    return wire.height


def _edges(layers: list[Layer]) -> np.ndarray:
    """Cumulative build [m] at every layer boundary — len(layers) + 1 values."""
    if not layers:
        raise ValueError("no layer to plot")
    return np.concatenate(([0.0], np.cumsum([_thickness(lay) for lay in layers])))


def _windings(structure: Dowell_Winding_Structure) -> list[WINDING_TYPE]:
    """The windings present, innermost first — the order every figure iterates in."""
    order = structure.windings()
    if not order:
        raise ValueError("the structure has no layer")
    return order


def _delta_one_frequency(layer: Layer) -> float:
    """
    Frequency at which the skin depth equals the layer's effective height.

    Delta = 1 is the knee: below it the current still fills the conductor,
    above it the layer only conducts on its skin and R_ac climbs as sqrt(f).
    Solving skin_depth(f) = h_eff gives f = 1 / (pi mu0 sigma eta h_eff^2).
    """
    h_eff = layer.effective_height()
    eta = layer.porosity()
    if h_eff <= 0.0 or eta <= 0.0:
        return float("inf")
    return 1.0 / (pi * MU0 * layer.wire.sigma * eta * h_eff**2)


def _equivalent_area(
    structure: Dowell_Winding_Structure, winding: WINDING_TYPE
) -> float:
    """
    The copper section [m2] a winding behaves as, whatever it is wound with.

    A winding whose layers do not share one wire has no single "wire section" to
    divide the current by. The area that DOES mean something is the one its own
    DC resistance implies: R_dc = L / (sigma A_eq) over the total wire length,
    i.e. the length-weighted harmonic mean of the layer sections. For a winding
    of one wire it is exactly that wire's section.
    """
    layers = [lay for lay in structure.list_of_layers if lay.winding_type is winding]
    if not layers:
        raise ValueError(f"no layer belongs to {winding.value}")
    length = sum(lay.number_of_turns * lay.mlt for lay in layers)
    r_dc = sum(lay.dc_resistance() for lay in layers)
    sigma = layers[0].wire.sigma
    if r_dc <= 0.0 or not np.isfinite(r_dc):
        return float("nan")
    return length / (sigma * r_dc)


def _frequencies(f_min: float, f_max: float, points: int) -> np.ndarray:
    """A log sweep: decades matter here, not hertz."""
    if f_min <= 0.0:
        raise ValueError(f"f_min must be > 0 on a log sweep, got {f_min}")
    if f_max <= f_min:
        raise ValueError(f"f_max must be > f_min, got {f_max} <= {f_min}")
    if points < 2:
        raise ValueError(f"points must be >= 2, got {points}")
    return np.geomspace(f_min, f_max, points)


def _currents_or_raise(currents: dict[WINDING_TYPE, float]) -> dict[WINDING_TYPE, float]:
    if not currents or all(value == 0.0 for value in currents.values()):
        raise ValueError("every current is zero — nothing to draw")
    return currents


# ============================================================================ #
#  Panels
# ============================================================================ #
#
# One function per panel, drawing into an axes it does not own — which is what
# lets plot_winding_section() stack the cut-away and the MMF without
# duplicating either of them.


def _twin_axes(ax: Axes, theme: str) -> Axes:
    """A right-hand axis wearing the same recessive chrome as its host."""
    c = style(theme)
    twin = ax.twinx()
    twin.set_facecolor("none")
    for side in ("top", "left", "bottom"):
        twin.spines[side].set_visible(False)
    twin.spines["right"].set_color(c["axis"])
    twin.spines["right"].set_linewidth(1.0)
    twin.tick_params(colors=c["muted"], labelsize=9, length=0)
    for label in twin.get_yticklabels():
        label.set_fontfamily(FONT)
    return twin


def _draw_section(
    ax: Axes,
    theme: str,
    structure: Dowell_Winding_Structure,
    core: bool,
    labels: bool,
    label_top: bool = False,
) -> None:
    """The cut-away: one band per layer, coloured by winding, hatched by wire."""
    c = style(theme)
    layers = structure.list_of_layers
    build = _edges(layers)
    total = float(build[-1])
    window = max(lay.bw for lay in layers)

    for layer, x0, x1 in zip(layers, build, build[1:]):
        colour = winding_colour(theme, layer.winding_type)
        band = Rectangle(
            # Layers of different bw are centred on the window rather than
            # sitting on its floor: a short layer is short at BOTH ends.
            (x0, 0.5 * (window - layer.bw)),
            x1 - x0,
            layer.bw,
            facecolor=to_rgba(colour, 0.22),
            edgecolor=colour,
            linewidth=1.3,
            hatch=_WIRE_HATCH[layer.wire.wire_type],
            zorder=3,
        )
        band.set_hatch_linewidth(0.6)
        ax.add_patch(band)

        if not labels:
            continue
        # A band narrower than a sixth of the build has no room for horizontal
        # text; turned a quarter turn it always fits, because the window is the
        # long side of every layer.
        narrow = (x1 - x0) / total < 0.16
        name = layer.name or layer.winding_type.value.lower()
        ax.text(
            0.5 * (x0 + x1),
            # Pushed to the ceiling when an MMF curve is coming: the curve owns
            # the middle of the panel, and a staircase struck through a label
            # is two things ruined for the price of one.
            window * (0.96 if label_top else 0.5),
            f"{name}\n{layer.number_of_turns:g} t",
            rotation=90 if narrow else 0,
            ha="center",
            va="top" if label_top else "center",
            color=c["ink"],
            fontsize=9,
            fontfamily=FONT,
            # The hatch runs under the text: a surface-coloured pad is what
            # keeps the label readable without hiding the texture around it.
            bbox={"facecolor": c["surface"], "edgecolor": "none", "pad": 2.0},
            zorder=5,
        )

    left = 0.0
    if core:
        # The core is not a layer, but without it nothing says which end of the
        # build is the inside — and the MMF profile is read from that end.
        left = -0.09 * total
        ax.add_patch(
            Rectangle(
                (left, 0.0),
                -left,
                window,
                facecolor=c["grid"],
                edgecolor=c["axis"],
                linewidth=1.0,
                zorder=2,
            )
        )
        ax.text(
            0.5 * left,
            0.5 * window,
            "core",
            rotation=90,
            ha="center",
            va="center",
            color=c["muted"],
            fontsize=9,
            fontfamily=FONT,
            zorder=5,
        )

    ax.set_xlim(left, total * 1.02)
    ax.set_ylim(0.0, window * 1.02)
    engineering_ticks(ax, theme, axis="x", unit="m")
    engineering_ticks(ax, theme, axis="y", unit="m")
    axis_labels(ax, theme, x="Build-up (core → outside)", y="Window breadth bw")


def _draw_mmf(
    ax: Axes,
    theme: str,
    structure: Dowell_Winding_Structure,
    currents: dict[WINDING_TYPE, float],
    bands: bool,
    grid_lines: bool = True,
    headroom: float = 0.18,
) -> None:
    """The MMF staircase against the build, layer bands shaded behind it."""
    c = style(theme)
    field = structure.field_profile(currents)
    # The staircase is sampled on the sub-layers, the bands drawn on the parent
    # ones: _thickness() guarantees the two abscissae end at the same build.
    build = _edges(structure.effective_layers())

    if bands:
        # Same colours as the cut-away, washed out to the point of being a
        # background: here the curve is the data and the layers are the ruler.
        parents = _edges(structure.list_of_layers)
        for layer, x0, x1 in zip(structure.list_of_layers, parents, parents[1:]):
            ax.axvspan(
                x0,
                x1,
                color=winding_colour(theme, layer.winding_type),
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )

    if grid_lines:
        grid(ax, theme, axis="y")
    # H changes sign as soon as the two windings oppose: zero is a real level
    # here, not a convenience.
    ax.axhline(0.0, color=c["axis"], linewidth=1.0, zorder=2)
    ax.plot(
        build,
        field,
        color=c["ink"],
        linewidth=1.8,
        marker="o",
        markersize=4.5,
        markeredgecolor=c["surface"],
        markeredgewidth=1.2,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )

    peak = max(field, key=abs)
    span = max(abs(value) for value in field) or 1.0
    # headroom : empty room left above the curve. On its own panel a fifth of
    # the span is enough to clear the peak label; overlaid on the cut-away it
    # is what keeps the staircase clear of the layer names.
    ax.set_ylim(
        min(min(field), 0.0) - 0.18 * span, max(max(field), 0.0) + headroom * span
    )
    annotate(
        ax,
        theme,
        quantity(peak, "A/m"),
        (float(build[field.index(peak)]), peak),
        (6, 6 if peak >= 0 else -14),
        role="ink",
        bold=True,
    )
    engineering_ticks(ax, theme, axis="x", unit="m")
    engineering_ticks(ax, theme, axis="y", unit="A/m")
    axis_labels(ax, theme, x="Build-up (core → outside)", y="MMF field H")


def _section_legend(
    ax: Axes, theme: str, structure: Dowell_Winding_Structure, wires: bool = True
) -> None:
    """
    Windings by colour, then wire types by texture — one strip, two groups.

    wires : list the wire types. Only for a figure that actually draws the
            hatches; on a panel where the layers are flat colour washes, a
            texture nobody can see is a legend entry that lies.
    """
    c = style(theme)
    entries = [
        (_winding_label(winding), winding_colour(theme, winding))
        for winding in _windings(structure)
    ]
    hatches = [""] * len(entries)

    if wires:
        seen: list[WIRE_TYPE] = []
        for layer in structure.list_of_layers:
            if layer.wire.wire_type not in seen:
                seen.append(layer.wire.wire_type)
        for wire_type in seen:
            # Wire entries take the muted ink, never a winding hue: the texture
            # is the whole message, and a colour on it would claim a winding it
            # does not have.
            entries.append((_WIRE_LABEL[wire_type], c["muted"]))
            hatches.append(_WIRE_HATCH[wire_type])

    legend(ax, theme, entries, kind="patch", hatches=hatches)


def _stack_subtitle(structure: Dowell_Winding_Structure) -> str:
    layers = structure.list_of_layers
    build = float(_edges(layers)[-1])
    text = (
        f"{len(layers)} layers · build {quantity(build, 'm')}"
        f" · bw {quantity(max(lay.bw for lay in layers), 'm')}"
    )
    # Only worth saying when a Litz layer has been split: everywhere else the
    # two counts are the same number twice.
    seen = len(structure.effective_layers())
    if seen != len(layers):
        text += f" · {seen} sub-layers seen by Dowell"
    return text


def _mmf_subtitle(
    structure: Dowell_Winding_Structure, currents: dict[WINDING_TYPE, float]
) -> str:
    net = structure.net_ampere_turns(currents)
    balance = "Σn·i balanced" if abs(net) < 1e-9 else f"Σn·i = {quantity(net, 'A')}"
    windings = " · ".join(
        f"{_winding_label(w)} {quantity(currents.get(w, 0.0), 'A')}"
        for w in _windings(structure)
    )
    return (
        f"{windings} · {balance} · "
        f"{structure.outer_mmf_fraction:.0%} of the MMF returns outside"
    )


# ============================================================================ #
#  Figures — the winding itself
# ============================================================================ #


def plot_layer_stack(
    structure: Dowell_Winding_Structure,
    currents: dict[WINDING_TYPE, float] | None = None,
    mmf: bool = False,
    title: str = "Winding cross-section",
    subtitle: str | None = None,
    theme: str = "light",
    core: bool = True,
    labels: bool = True,
) -> Figure:
    """
    The cut-away: layers stacked from the core outwards.

    Colour is the winding (primary red, secondary blue), hatch is the wire type,
    band width is the layer's true share of the build. The x axis is stretched
    against the y axis — see the module docstring.

    currents : RMS per winding [A]. Only needed for `mmf`.
    mmf      : draw the MMF profile ON the cut-away, on a right-hand axis. The
               one figure that shows a layer sitting where the field is highest
               at the same time as what it is made of. Costs the reader a second
               scale: for a figure meant to be studied rather than glanced at,
               prefer plot_winding_section(), which gives the MMF its own panel.
    core     : draw the core as a grey band on the left. It is the only thing
               saying which end of the build is the inside.
    labels   : name and turn count inside each band. Turn it off for a stack of
               many thin layers, where the labels outnumber the bands.
    """
    fig, ax = new_figure(theme, (8.4, 4.6))
    _draw_section(ax, theme, structure, core=core, labels=labels, label_top=mmf)

    text = subtitle if subtitle is not None else _stack_subtitle(structure)
    if mmf:
        if currents is None:
            raise ValueError("mmf=True needs the currents to build the field from")
        twin = _twin_axes(ax, theme)
        # No gridlines and no shading from the overlay: the twin is drawn over
        # the whole host axes, so anything it lays down crosses the bands.
        _draw_mmf(
            twin,
            theme,
            structure,
            currents,
            bands=False,
            grid_lines=False,
            headroom=0.55,
        )
        # The host axes owns the abscissa and its labelling; the twin only
        # brings its own ordinate.
        twin.set_xlim(*ax.get_xlim())
        twin.set_xlabel("")
        if subtitle is None:
            text = _mmf_subtitle(structure, currents)

    _section_legend(ax, theme, structure)
    add_title(ax, theme, title, text)
    fig.tight_layout()
    return fig


def plot_mmf_profile(
    structure: Dowell_Winding_Structure,
    currents: dict[WINDING_TYPE, float],
    title: str = "MMF profile",
    subtitle: str | None = None,
    theme: str = "light",
    bands: bool = True,
) -> Figure:
    """
    The field the stacking order builds, across the window.

    H is what Dowell squares: a layer sitting where |H| is large pays for it
    whether or not it carries current, which is why a passive secondary can
    dissipate. The staircase is piecewise linear because each layer is assumed
    to carry its current uniformly across its own thickness.

    The two faces of the stack are set by outer_mmf_fraction — where the flux
    return path is. Read it off the ends of the curve.

    `currents` is ONE conduction phase. A flyback has two and they never
    overlap: draw it twice — see "one figure per conduction phase" at the top
    of this module.

    bands : shade the layers behind the curve, so a knee can be attributed to
            the layer that caused it.
    """
    fig, ax = new_figure(theme, (8.4, 4.4))
    _draw_mmf(ax, theme, structure, _currents_or_raise(currents), bands=bands)
    if bands:
        # Flat washes here, no hatching: the legend names the colours only.
        _section_legend(ax, theme, structure, wires=False)
    add_title(ax, theme, title, subtitle or _mmf_subtitle(structure, currents))
    fig.tight_layout()
    return fig


def plot_winding_section(
    structure: Dowell_Winding_Structure,
    currents: dict[WINDING_TYPE, float],
    title: str = "Winding cross-section",
    subtitle: str | None = None,
    theme: str = "light",
    core: bool = True,
    labels: bool = True,
) -> Figure:
    """
    The cut-away and its MMF, stacked on one abscissa.

    The two panels answer ONE question — why this stack loses what it loses —
    and cannot be read apart: the top says what a layer is, the bottom says
    what field it sits in. Sharing the x axis is what makes a knee in the MMF
    point at the band that caused it.

    `currents` is ONE conduction phase — a flyback needs one figure per phase.
    See "one figure per conduction phase" at the top of this module.
    """
    currents = _currents_or_raise(currents)
    fig, (top, bottom) = new_grid(
        theme, (8.6, 7.6), rows=2, height_ratios=(3.0, 2.2), share_x=True
    )

    _draw_section(top, theme, structure, core=core, labels=labels)
    _draw_mmf(bottom, theme, structure, currents, bands=True)
    # share_x makes the core band of the top panel part of the bottom one's
    # range too; the MMF simply has nothing to draw there, which is correct.
    bottom.set_xlim(*top.get_xlim())
    axis_labels(top, theme, x="")

    _section_legend(top, theme, structure)
    add_title(top, theme, title, subtitle or _stack_subtitle(structure))
    add_title(bottom, theme, "MMF profile", _mmf_subtitle(structure, currents))

    fig.tight_layout()
    # Room for the MMF panel's own title block, which tight_layout sizes for
    # the axes only and would otherwise let the cut-away sit on top of.
    fig.subplots_adjust(hspace=0.34)
    return fig


# ============================================================================ #
#  Figures — against frequency
# ============================================================================ #


def _label_ends(
    ax: Axes, theme: str, x: float, entries: list[tuple[float, str]]
) -> None:
    """
    Name every curve at its right end — a legend for two series is a box too many.

    Two windings can end a decade apart or a hair apart depending on the wire:
    normalised into R_ac/R_dc they very nearly land on each other. So the
    labels are placed in DISPLAY space and pushed up until they clear the one
    below, which is the only way to know whether they collide at all.
    """
    ax.get_ylim()  # settles the autoscale, so transData is the final one
    placed: list[float] = []
    # 26 points: two 9 pt lines and the air between them.
    spacing = 26.0
    for value, text in sorted(entries):
        pixel = float(ax.transData.transform((x, value))[1])
        shift = 0.0
        for other in placed:
            if abs(pixel + shift - other) < spacing:
                shift = other + spacing - pixel
        placed.append(pixel + shift)
        annotate(ax, theme, text, (x, value), (7, shift), role="ink", bold=True, va="center")


def _resistance_sweep(
    structure: Dowell_Winding_Structure,
    frequencies: np.ndarray,
    currents: dict[WINDING_TYPE, float] | None = None,
) -> dict[WINDING_TYPE, np.ndarray]:
    """
    R_ac per winding over the sweep. One model call per frequency, not per curve.

    `currents` picks the convention — see Dowell_Winding_Structure
    .ac_resistances(). None gives the self resistance of each winding; an
    operating point gives the effective one, the only correct choice as soon as
    two windings conduct together.
    """
    sampled = [structure.ac_resistances(float(f), currents) for f in frequencies]
    return {
        winding: np.array([point[winding] for point in sampled])
        for winding in _windings(structure)
    }


def plot_resistance_vs_frequency(
    structure: Dowell_Winding_Structure,
    currents: dict[WINDING_TYPE, float] | None = None,
    f_min: float = 1e3,
    f_max: float = 10e6,
    points: int = 160,
    normalise: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    theme: str = "light",
    mark_delta_one: bool = True,
) -> Figure:
    """
    R_ac against frequency, one curve per winding.

    currents : WHICH resistance, and the answer depends on the topology.

        None — each winding energised alone with 1 A, the others open. Its
        SELF resistance: what an LCR bridge reads, and what a flyback winding
        really sees, since a flyback never has two windings conducting at once
        (split its phases outside and call this once per phase).

        An operating point — the EFFECTIVE resistance, R = P/I², every field
        present. Mandatory as soon as the windings conduct TOGETHER (forward,
        push-pull, LLC): their MMFs partially cancel and the self resistance
        then overstates the loss. It is also the only version that can see
        interleaving at all — P-S-S-P and P-P-S-S have the SAME self
        resistance and a loss 31 % apart. Windings carrying no current are
        dropped: their loss is real but no current of theirs explains it.

    normalise      : plot R_ac / R_dc instead of ohms. The shape is identical;
                     the ratio is what tells you whether 2 mΩ is a good 2 mΩ,
                     and it puts two windings of very different sizes on one
                     readable scale.
    mark_delta_one : a rule at the frequency where the skin depth falls to the
                     effective height of the winding's thickest layer. Left of
                     it the curve is flat, right of it it climbs — that rule is
                     the whole design constraint on the wire.
    """
    frequencies = _frequencies(f_min, f_max, points)
    r_dc = structure.dc_resistances()
    r_ac = _resistance_sweep(structure, frequencies, currents)
    # A winding with no current has no effective resistance (nan) — it is not
    # missing data, it is a quantity that does not exist for it.
    windings = [w for w in _windings(structure) if np.isfinite(r_ac[w][0])]
    if not windings:
        raise ValueError("no winding carries current — nothing to draw")

    fig, ax = new_figure(theme, (8.2, 4.6))
    grid(ax, theme, axis="y")
    ax.set_xscale("log")
    ax.set_yscale("log")

    if normalise:
        reference_line(ax, theme, 1.0, "R_dc", offset=(8, 6))

    ends: list[tuple[float, str]] = []
    for winding in windings:
        colour = winding_colour(theme, winding)
        values = r_ac[winding] / r_dc[winding] if normalise else r_ac[winding]
        ax.plot(
            frequencies,
            values,
            color=colour,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )
        final = float(values[-1])
        ends.append(
            (
                final,
                f"{_winding_label(winding)}\n"
                + (f"×{final:.1f}" if normalise else quantity(final, "Ω")),
            )
        )
        if not normalise:
            # Its own floor, per winding: a shared one would be meaningless
            # between a 67-turn primary and a 26-turn secondary.
            ax.axhline(r_dc[winding], color=colour, linewidth=1.0, alpha=0.45, zorder=2)
    _label_ends(ax, theme, float(frequencies[-1]), ends)

    if mark_delta_one:
        for index, winding in enumerate(windings):
            thickest = max(
                (lay for lay in structure.effective_layers() if lay.winding_type is winding),
                key=lambda lay: lay.effective_height(),
            )
            knee = _delta_one_frequency(thickest)
            if f_min <= knee <= f_max:
                vertical_reference_line(
                    ax,
                    theme,
                    knee,
                    f"Δ=1 · {_winding_label(winding)}",
                    # Two knees can land within a hair of each other; the second
                    # label drops a line rather than overprinting the first.
                    offset=(4, -4 - 14 * index),
                )

    engineering_ticks(ax, theme, axis="x", unit="Hz")
    if normalise:
        # A bare EngFormatter on the ratio, so the decades read "1", "10",
        # "100" and not the 10^n of matplotlib's log default.
        engineering_ticks(ax, theme, axis="y")
        axis_labels(ax, theme, x="Frequency", y="R_ac / R_dc")
    else:
        engineering_ticks(ax, theme, axis="y", unit="Ω")
        axis_labels(ax, theme, x="Frequency", y="AC resistance")

    if subtitle is None:
        floors = " · ".join(
            f"R_dc {_winding_label(w)} {quantity(r_dc[w], 'Ω')}" for w in windings
        )
        # Which resistance is on the axes is not a detail the reader can infer
        # from the curves — the two conventions differ by 31 % on an
        # interleaved winding and not at all on a stacked one.
        convention = (
            "self, one winding energised at a time"
            if currents is None
            else "effective, "
            + " + ".join(
                f"{_winding_label(w)} {quantity(currents.get(w, 0.0), 'A')}"
                for w in windings
            )
            + " together"
        )
        subtitle = f"{floors} · {convention}"

    add_title(
        ax,
        theme,
        title or ("AC resistance factor" if normalise else "AC resistance"),
        subtitle,
    )
    fig.tight_layout()
    return fig


def plot_losses_vs_frequency(
    structure: Dowell_Winding_Structure,
    currents: dict[WINDING_TYPE, float],
    f_min: float = 1e3,
    f_max: float = 10e6,
    points: int = 160,
    title: str = "Copper loss against frequency",
    subtitle: str | None = None,
    theme: str = "light",
    dc_reference: bool = True,
) -> Figure:
    """
    What the given currents cost, winding by winding, plus the total — the
    total only when there is more than one winding for it to sum.

    The currents are held constant and the whole of each is placed at the
    frequency of the abscissa: the curve answers "what would this winding cost
    if its RMS sat entirely at f", which is the question a switching frequency
    is chosen against. It is NOT the loss of a real waveform — that one is
    spread over a spectrum and is what Dowell_Winding_Structure.harmonic_losses()
    sums up.

    A winding's curve holds every watt dissipated in ITS layers, including the
    proximity loss the other winding's field forces on it. That is why a
    winding carrying no current can still have a curve above zero, and why the
    total is the only number a heatsink cares about.

    `currents` is ONE conduction phase. On a flyback, plot the two phases
    separately and ADD their totals: passing both currents at once lets the
    opposite MMFs cancel and understates the loss. See "one figure per
    conduction phase" at the top of this module.

    dc_reference : rule at the DC loss of the same currents. The gap between it
                   and the total curve is the entire cost of the AC effects.
    """
    currents = _currents_or_raise(currents)
    frequencies = _frequencies(f_min, f_max, points)
    windings = _windings(structure)

    per_winding: dict[WINDING_TYPE, list[float]] = {w: [] for w in windings}
    total: list[float] = []
    for frequency in frequencies:
        bucket = structure.losses_by_winding(float(frequency), currents)
        for winding in windings:
            per_winding[winding].append(bucket[winding])
        total.append(sum(bucket.values()))

    fig, ax = new_figure(theme, (8.2, 4.6))
    grid(ax, theme, axis="y")
    ax.set_xscale("log")
    ax.set_yscale("log")

    if dc_reference:
        p_dc = structure.dc_loss(currents)
        if p_dc > 0.0:
            reference_line(
                ax, theme, p_dc, f"P_dc {quantity(p_dc, 'W')}", offset=(8, 6)
            )

    # A winding dissipating exactly nothing — one shielded by another in a
    # phase where it carries no current — has no curve a log axis can draw.
    # Keeping it would leave a legend entry pointing at nothing.
    drawn = [w for w in windings if max(per_winding[w]) > 0.0]
    if not drawn:
        raise ValueError("nothing is dissipated at any frequency — nothing to draw")

    curves = [
        (_winding_label(w), per_winding[w], winding_colour(theme, w), 1.8)
        for w in drawn
    ]
    # Total last and thicker: it is drawn over the two it sums, and being the
    # top curve is not enough to say which one it is. Only worth a curve when
    # there IS more than one: on an inductor, or on a flyback phase where the
    # idle winding is shielded, it lands exactly on the single winding below it
    # and puts a second copy of the same number in the margin.
    if len(drawn) > 1:
        curves.append(("Total", total, _total_colour(theme), 2.4))

    for _name, values, colour, width in curves:
        ax.plot(
            frequencies,
            values,
            color=colour,
            linewidth=width,
            solid_capstyle="round",
            zorder=3,
        )
    # Values at the right end, names in the legend below: a loss plot spanning
    # less than two decades gets one or two labelled gridlines and no more, so
    # without these the reader can see the shape and not read a single watt.
    _label_ends(
        ax,
        theme,
        float(frequencies[-1]),
        [(values[-1], quantity(values[-1], "W")) for _, values, _, _ in curves],
    )

    engineering_ticks(ax, theme, axis="x", unit="Hz")
    engineering_ticks(ax, theme, axis="y", unit="W")
    axis_labels(ax, theme, x="Frequency", y="Copper loss")

    # Three series that touch at the left of the plot: this is the case the
    # legend box exists for. One curve names itself at its right end already.
    if len(curves) > 1:
        legend(
            ax,
            theme,
            [(name, colour) for name, _, colour, _ in curves],
            kind="line",
        )
    add_title(
        ax,
        theme,
        title,
        subtitle
        or " · ".join(
            f"{_winding_label(w)} {quantity(currents.get(w, 0.0), 'A')} RMS"
            for w in windings
        ),
    )
    fig.tight_layout()
    return fig


def plot_current_density_vs_frequency(
    structure: Dowell_Winding_Structure,
    currents: dict[WINDING_TYPE, float],
    f_min: float = 1e3,
    f_max: float = 10e6,
    points: int = 160,
    j_limit: float | None = None,
    title: str = "Current density against frequency",
    subtitle: str | None = None,
    theme: str = "light",
) -> Figure:
    """
    The current density each winding really works at, in A/mm2.

    At DC the current fills the copper and J = I / A. Above the knee it does
    not: the same current crowds into a skin, and the density in that skin is
    higher than the section suggests. What is plotted is the EQUIVALENT uniform
    density — the one that would dissipate the measured loss in the whole
    section:

        J_eq(f) = (I / A_eq) * sqrt(R_ac(f) / R_dc)

    so the usual 4-6 A/mm2 rule of thumb, which is a thermal rule about loss
    per unit of copper, can still be applied to it. The peak density inside the
    skin is higher still; this is the number a temperature rise follows.

    R_ac is the EFFECTIVE resistance under `currents`, not the winding's self
    resistance: the density has to follow the loss the winding really has, and
    on an interleaved build the two differ by tens of percent (see
    plot_resistance_vs_frequency). On a flyback phase, where one winding
    conducts alone, the two coincide exactly.

    A_eq is the section the winding's own DC resistance implies (_equivalent_area),
    so a winding of mixed wires still gets one honest number.

    j_limit : the design ceiling [A/mm2], drawn as a threshold. Where a curve
              crosses it is the frequency at which the wire has to change.

    Only windings actually carrying current are drawn — a zero current is a
    flat zero, and it would only steal a colour.

    NB: this is the one axis of the project not in base SI. A/mm2 is the unit
    the rule of thumb, the datasheets and the designer all use; the same values
    plotted in A/m2 would read as "5 MA/m2" and be understood by nobody.
    """
    currents = _currents_or_raise(currents)
    frequencies = _frequencies(f_min, f_max, points)
    r_dc = structure.dc_resistances()
    r_ac = _resistance_sweep(structure, frequencies, currents)

    windings = [w for w in _windings(structure) if currents.get(w, 0.0) > 0.0]
    if not windings:
        raise ValueError("no winding of this structure carries current")

    fig, ax = new_figure(theme, (8.2, 4.6))
    grid(ax, theme, axis="y")
    ax.set_xscale("log")

    if j_limit is not None:
        reference_line(
            ax, theme, j_limit, f"limit {j_limit:.1f} A/mm²", kind="limit", offset=(2, 6)
        )

    top = 0.0
    floors: list[str] = []
    ends: list[tuple[float, str]] = []
    for winding in windings:
        colour = winding_colour(theme, winding)
        # 1e6 : m2 -> mm2, the axis unit. The model stays in SI throughout.
        area = _equivalent_area(structure, winding)
        j_dc = currents[winding] / area / 1e6
        values = j_dc * np.sqrt(r_ac[winding] / r_dc[winding])
        floors.append(
            f"{_winding_label(winding)} {quantity(currents[winding], 'A')} "
            f"in {area * 1e6:.3g} mm² → {j_dc:.2f} A/mm² at DC"
        )
        ax.plot(
            frequencies,
            values,
            color=colour,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )
        # Its DC density as a floor: the vertical gap to the curve is the whole
        # penalty the frequency imposes, read straight off the axis.
        ax.axhline(j_dc, color=colour, linewidth=1.0, alpha=0.45, zorder=2)
        ends.append(
            (
                float(values[-1]),
                f"{_winding_label(winding)}\n{values[-1]:.1f} A/mm²",
            )
        )
        top = max(top, float(values[-1]))

    ceiling = max(top, j_limit or 0.0)
    ax.set_ylim(0.0, ceiling * 1.15)
    # After the limits: the labels are pushed apart in display space, which
    # only means anything once the axes knows how tall it is.
    _label_ends(ax, theme, float(frequencies[-1]), ends)
    engineering_ticks(ax, theme, axis="x", unit="Hz")
    axis_labels(ax, theme, x="Frequency", y="Equivalent current density [A/mm²]")

    add_title(ax, theme, title, subtitle or " · ".join(floors))
    fig.tight_layout()
    return fig


# ============================================================================ #
#  Tables
# ============================================================================ #


def layer_table(
    structure: Dowell_Winding_Structure, freq: float, unit_turns: str = "t"
) -> pd.DataFrame:
    """
    The cut-away's table twin: one row per layer AS DOWELL SEES IT.

    Litz layers appear split into their sqrt(k) strand sub-layers, because that
    is what the loss is computed on — a row per parent layer would hide the
    only thing that makes Litz worth its price.

    Three of the light-mode series colours sit below 3:1 against the surface,
    so the palette is only allowed to carry meaning alongside visible labels or
    this table. Print it next to the figure.

    The cells stay inside cp1252, unlike the figures: a Windows console still
    defaults to it, and a table that raises UnicodeEncodeError on "Ω" is a
    table nobody can print. Hence "Ohm" and "eta" spelled out.
    """
    rows = []
    index = []
    for layer in structure.effective_layers():
        eta = layer.porosity()
        index.append(layer.name or layer.winding_type.value.lower())
        rows.append(
            [
                _winding_label(layer.winding_type),
                _WIRE_LABEL[layer.wire.wire_type],
                f"{layer.number_of_turns:.4g} {unit_turns}",
                quantity(_thickness(layer), "m"),
                quantity(layer.effective_height(), "m"),
                f"{eta:.3f}",
                quantity(skin_depth(freq, layer.wire.sigma, eta), "m"),
                f"{layer.delta(freq):.3f}",
                quantity(layer.dc_resistance(), "Ohm"),
            ]
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "Winding",
            "Wire",
            "Turns",
            "Build",
            "h_eff",
            "Porosity eta",
            "Skin depth",
            "Delta",
            "R_dc",
        ],
        index=index,
    )
    frame.index.name = f"Layer @ {quantity(freq, 'Hz')}"
    return frame


def winding_table(
    structure: Dowell_Winding_Structure,
    freq: float,
    currents: dict[WINDING_TYPE, float] | None = None,
) -> pd.DataFrame:
    """
    The frequency figures' table twin, at one frequency: what each winding is.

    Without `currents` it stops at the geometry and the SELF resistances, which
    do not depend on what flows. With them it switches to the EFFECTIVE ones —
    the operating point, every field present, same convention as the figures —
    and adds the densities and the loss, plus a TOTAL row, the only line a
    thermal budget takes.

    A winding that carries no current keeps its loss (proximity is real) and
    gets a dash everywhere a current would have to divide.

    Cells inside cp1252, for the reason given on layer_table().
    """
    windings = _windings(structure)
    r_dc = structure.dc_resistances()
    r_ac = structure.ac_resistances(freq, currents)

    def cell(value: float, unit: str = "", digits: int = 2) -> str:
        """A dash for what does not exist, rather than a nan pretending to."""
        if not np.isfinite(value):
            return "-"
        return quantity(value, unit) if unit else f"{value:.{digits}f}"

    columns = ["R_dc", f"R_ac @ {quantity(freq, 'Hz')}", "R_ac/R_dc", "Copper section"]
    rows = {
        _winding_label(w): [
            quantity(r_dc[w], "Ohm"),
            cell(r_ac[w], "Ohm"),
            cell(r_ac[w] / r_dc[w], digits=3),
            f"{_equivalent_area(structure, w) * 1e6:.3f} mm2",
        ]
        for w in windings
    }

    if currents is not None:
        loss = structure.losses_by_winding(freq, currents)
        columns += ["I_rms", "J_dc", "J_eq", "Loss"]
        for w in windings:
            current = currents.get(w, 0.0)
            j_dc = current / _equivalent_area(structure, w) / 1e6
            rows[_winding_label(w)] += [
                quantity(current, "A"),
                f"{j_dc:.2f} A/mm2" if current else "-",
                f"{j_dc * sqrt(r_ac[w] / r_dc[w]):.2f} A/mm2" if current else "-",
                quantity(loss[w], "W"),
            ]
        rows["TOTAL"] = ["-"] * (len(columns) - 1) + [
            quantity(sum(loss.values()), "W")
        ]

    frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    frame.index.name = "Winding"
    return frame


# ============================================================================ #
#  Demo
# ============================================================================ #

if __name__ == "__main__":
    from SRC.MAGNETIC.dowell import POLARITY, Wire
    from SRC.OTHERS.plot import save_figure
    from SRC.OTHERS.terminal import alert, dataframe, kv, section, table, use_theme
    from SRC.SIGNAL_PROCESSING.signal import ElectronicPeriodicSignal
    from SRC.SIGNAL_PROCESSING.signal_plot import (
        plot_spectrum,
        plot_time_domain,
        signal_table,
    )

    OUTPUT = Path(__file__).parent.parent.parent / "OUTPUT"
    THEME = "light"
    use_theme(THEME)

    PRI, SEC = WINDING_TYPE.PRIMARY, WINDING_TYPE.SECONDARY

    F_SW = 100e3
    T_SW = 1.0 / F_SW
    # 200 ranks = 20 MHz at 100 kHz. Ideal edges give a 1/n spectrum and a
    # per-rank loss going as n^-1.5: the sum converges, but slowly, so a
    # truncated one is always a lower bound.
    N_MAX = 200

    figures: dict[str, Figure] = {}

    # ======================================================================== #
    #  1 — Flyback: two windings that never conduct together
    # ======================================================================== #
    section(1, "Flyback — one figure per conduction phase")

    bw = pi * 7.62e-3  # RM10 bobbin, window breadth
    mlt = 17.5e-3
    N_PRI, N_SEC = 67, 26
    D_PRI, D_SEC = 0.3e-3, 0.4e-3

    flyback = Dowell_Winding_Structure()
    flyback.add_layer(
        Layer(
            name="Primary",
            number_of_turns=N_PRI,
            bw=bw,
            mlt=mlt,
            wire=Wire(wire_type=WIRE_TYPE.ROUND, diameter=D_PRI),
            winding_type=PRI,
            polarity=POLARITY.POSITIVE,
        )
    )
    flyback.add_layer(
        Layer(
            name="Secondary",
            number_of_turns=N_SEC,
            bw=bw,
            # Wound OVER the primary: its mean turn is longer by the build it
            # sits on, which is the only reason the two windings of the same
            # copper do not have the same resistance per turn.
            mlt=mlt + 8 * D_PRI,
            wire=Wire(wire_type=WIRE_TYPE.ROUND, diameter=D_SEC),
            winding_type=SEC,
            polarity=POLARITY.NEGATIVE,
        )
    )

    # DCM flyback. The flux does not jump when the switch opens, so the
    # ampere-turns hand over: N_p·I_p,pk = N_s·I_s,pk. The secondary peak is
    # NOT a free parameter — pick it independently and the MMF profile drawn
    # below is of a transformer that cannot exist.
    t_on = 6.30e-6  # primary ramps 0 -> I_pk
    t_dem = 1.98e-6  # secondary demagnetises I_pk·n -> 0
    i_pri_pk = 0.5
    i_sec_pk = i_pri_pk * N_PRI / N_SEC
    assert abs(N_PRI * i_pri_pk - N_SEC * i_sec_pk) < 1e-9

    i_pri = ElectronicPeriodicSignal.from_breakpoints(
        "I primary",
        times=[0.0, t_on, t_on, T_SW],
        values=[0.0, i_pri_pk, 0.0, 0.0],
        n_samples=8192,
        period=T_SW,
    )
    i_sec = ElectronicPeriodicSignal.from_breakpoints(
        "I secondary",
        times=[0.0, t_on, t_on, t_on + t_dem, T_SW],
        values=[0.0, 0.0, i_sec_pk, 0.0, 0.0],
        n_samples=8192,
        period=T_SW,
    )
    signals = {PRI: i_pri, SEC: i_sec}

    kv("Topology", f"flyback DCM, {quantity(F_SW, 'Hz')}, Np:Ns = {N_PRI}:{N_SEC}")
    kv("Primary", f"{quantity(i_pri_pk, 'A')} peak over {quantity(t_on, 's')}")
    kv("Secondary", f"{quantity(i_sec_pk, 'A')} peak over {quantity(t_dem, 's')}")
    kv("Dead time", quantity(T_SW - t_on - t_dem, "s"))
    dataframe(signal_table([i_pri, i_sec], unit="A"))
    dataframe(layer_table(flyback, F_SW))

    # Each winding conducts only during its own phase, so its RMS over the
    # WHOLE period already carries its duty cycle. That is what makes ONE
    # FIGURE per phase readable — and it is as far as the split goes. The two
    # phases do NOT add directly in watts: see the comparison further down.
    # Currents, figure-name suffix, and the one-winding spectrum of the phase.
    phases = (
        ("on", "ON · primary conducts", {PRI: i_pri.rms(), SEC: 0.0}, {PRI: i_pri}),
        ("off", "OFF · secondary conducts", {PRI: 0.0, SEC: i_sec.rms()}, {SEC: i_sec}),
    )

    p_phase_split = 0.0
    for suffix, label, currents, phase_signal in phases:
        section(None, f"Phase {label}")
        H = flyback.field_profile(currents)
        kv("MMF at the boundaries", " → ".join(quantity(h, "A/m") for h in H))
        dataframe(winding_table(flyback, F_SW, currents))
        p_phase_split += flyback.loss_at_frequency(F_SW, currents)

        figures[f"dowell_flyback_section_{suffix}"] = plot_winding_section(
            flyback, currents, title=f"Flyback — phase {label}", theme=THEME
        )
        figures[f"dowell_flyback_losses_{suffix}"] = plot_losses_vs_frequency(
            flyback,
            currents,
            title=f"Copper loss — phase {label}",
            theme=THEME,
        )
        # One winding alone: self and effective resistance coincide, so this is
        # both the LCR-bridge number and the loss-bearing one.
        figures[f"dowell_flyback_density_{suffix}"] = (
            plot_current_density_vs_frequency(
                flyback,
                currents,
                j_limit=6.0,
                title=f"Current density — phase {label}",
                theme=THEME,
            )
        )

    figures["dowell_flyback_currents"] = plot_time_domain(
        [i_pri, i_sec],
        title="Flyback winding currents",
        unit="A",
        cycles=2.0,
        theme=THEME,
    )
    figures["dowell_flyback_spectrum"] = plot_spectrum(
        [i_pri, i_sec], title="Flyback current spectra", unit="A", theme=THEME
    )
    # No `currents`: on a flyback the self resistance IS what each phase sees,
    # since the other winding is open while this one conducts.
    figures["dowell_flyback_resistance"] = plot_resistance_vs_frequency(
        flyback, theme=THEME
    )
    figures["dowell_flyback_resistance_factor"] = plot_resistance_vs_frequency(
        flyback, normalise=True, theme=THEME
    )

    # --- What the shortcuts cost ------------------------------------------ #
    # Four ways to price the same transformer on the same operating point.
    # Only the last one is right; the three others are what gets used.
    #
    # The reference is ONE harmonic_losses() call carrying both windings. It
    # keeps the complex phasor of each rank, so the true angle between the two
    # currents is what decides how much their MMFs cancel — and on a flyback
    # that angle is 94° at the fundamental, nowhere near the 180° a polarity
    # declaration suggests.
    p_together = flyback.loss_at_frequency(F_SW, {PRI: i_pri.rms(), SEC: i_sec.rms()})
    p_harm_joint = flyback.harmonic_losses(signals, n_max=N_MAX)["P_total"]
    p_harm_split = sum(
        flyback.harmonic_losses(phase_signal, n_max=N_MAX)["P_total"]
        for _, _, _, phase_signal in phases
    )

    section(None, "What the shortcuts cost")
    table(
        ["Method", "Frequency", "Windings", "Loss", "vs reference"],
        [
            [
                "Both RMS at once",
                "f_sw only",
                "together",
                quantity(p_together, "W"),
                f"{100 * (p_together / p_harm_joint - 1.0):+.1f} %",
            ],
            [
                "Phase by phase",
                "f_sw only",
                "split",
                quantity(p_phase_split, "W"),
                f"{100 * (p_phase_split / p_harm_joint - 1.0):+.1f} %",
            ],
            [
                "Phase by phase",
                f"n ≤ {N_MAX}",
                "split",
                quantity(p_harm_split, "W"),
                f"{100 * (p_harm_split / p_harm_joint - 1.0):+.1f} %",
            ],
            [
                "One call, both spectra",
                f"n ≤ {N_MAX}",
                "together",
                quantity(p_harm_joint, "W"),
                "reference",
            ],
        ],
    )
    alert(
        "warn",
        "Splitting the phases is a rule about FIGURES — a drawing shows one "
        "instant, so it needs one per phase. Carried into the arithmetic it "
        "costs "
        f"{100 * (1.0 - p_harm_split / p_harm_joint):.0f} %: it treats the "
        "ON → OFF transition as if the field rested at zero, when the "
        "primary's inner face swings from −641 to +360 A/m in one step.",
    )
    alert(
        "warn",
        "Feeding both RMS to a single frequency is worse still, and for the "
        "opposite reason: at f_sw the two currents are declared in exact "
        "opposition, so their MMFs cancel far more than they really do.",
    )
    alert(
        "ok",
        f"Reference {quantity(p_harm_joint, 'W')} — one harmonic_losses() "
        f"call, every winding, every rank, phases kept. Exact for any "
        f"topology; the only approximation left is the truncation at "
        f"n = {N_MAX}.",
    )

    # ======================================================================== #
    #  2 — Boost inductor: one winding, nothing to cancel the MMF
    # ======================================================================== #
    section(2, "Boost inductor — DC bias against ripple")

    V_IN, V_OUT, P_OUT = 12.0, 24.0, 36.0
    L_BOOST = 47e-6
    duty = 1.0 - V_IN / V_OUT
    i_avg = P_OUT / V_IN  # the boost inductor carries the INPUT current
    d_i = V_IN * duty / (L_BOOST * F_SW)  # peak-to-peak ripple

    i_boost = ElectronicPeriodicSignal.from_breakpoints(
        "I_L boost",
        times=[0.0, duty * T_SW, T_SW],
        values=[i_avg - 0.5 * d_i, i_avg + 0.5 * d_i, i_avg - 0.5 * d_i],
        n_samples=8192,
        period=T_SW,
    )

    # A transformer's MMF comes back to zero because its ampere-turns balance.
    # An inductor's cannot: there is one winding and Σn·i is the whole point of
    # it. The staircase therefore ramps monotonically across the build, the
    # outermost turn sits in the full field, and interleaving — the main lever
    # of the flyback above — does not exist here. Wire choice is all there is.
    #
    # Default outer_mmf_fraction = 0: gap in the centre leg, so the return path
    # is inside and H falls to zero on the OUTER face.
    BW_L, MLT_L, D_L = 19.5e-3, 53e-3, 0.9e-3
    TURNS_PER_LAYER, LAYERS = 20, 2

    boost = Dowell_Winding_Structure()
    for index in range(LAYERS):
        boost.add_layer(
            Layer(
                name=f"L{index + 1}",
                number_of_turns=TURNS_PER_LAYER,
                bw=BW_L,
                # Each layer sits on the one below: one wire diameter of build
                # adds 2·pi·d to the mean turn.
                mlt=MLT_L + index * 2.0 * pi * D_L,
                wire=Wire(wire_type=WIRE_TYPE.ROUND, diameter=D_L),
                winding_type=PRI,
                polarity=POLARITY.POSITIVE,
            )
        )

    i_rms = i_boost.rms()
    i_ac = i_boost.ac_rms()
    i_dc = i_boost.mean()

    kv("Topology", f"boost {V_IN:.0f} V → {V_OUT:.0f} V, {P_OUT:.0f} W")
    kv("Inductor", f"{quantity(L_BOOST, 'H')}, {LAYERS}×{TURNS_PER_LAYER} t of "
       f"{quantity(D_L, 'm')} round, gapped centre leg")
    kv("Current", f"{quantity(i_dc, 'A')} DC + {quantity(d_i, 'A')} pk-pk ripple "
       f"→ {quantity(i_rms, 'A')} RMS, of which {quantity(i_ac, 'A')} is AC")
    dataframe(signal_table(i_boost, unit="A"))
    dataframe(layer_table(boost, F_SW))
    # The ripple alone: what the AC effects actually get to work on.
    dataframe(winding_table(boost, F_SW, {PRI: i_ac}))

    # --- Three ways to price it ------------------------------------------- #
    # The whole RMS placed at f_sw is the reflex, and on a DC-biased inductor
    # it is wrong by the ratio of the two currents squared: 88 % of the RMS
    # here is DC, which sees R_dc and nothing else.
    p_naive = boost.loss_at_frequency(F_SW, {PRI: i_rms})
    p_hand_split = boost.dc_loss({PRI: i_dc}) + boost.loss_at_frequency(
        F_SW, {PRI: i_ac}
    )
    harm = boost.harmonic_losses({PRI: i_boost}, n_max=N_MAX)

    section(None, "What the shortcuts cost")
    table(
        ["Method", "Current placed at f_sw", "Loss", "vs reference"],
        [
            [
                "Whole RMS at f_sw",
                quantity(i_rms, "A"),
                quantity(p_naive, "W"),
                f"{100 * (p_naive / harm['P_total'] - 1.0):+.1f} %",
            ],
            [
                "DC apart, ripple at f_sw",
                quantity(i_ac, "A"),
                quantity(p_hand_split, "W"),
                f"{100 * (p_hand_split / harm['P_total'] - 1.0):+.1f} %",
            ],
            [
                "Harmonic sum",
                f"rank by rank, n ≤ {N_MAX}",
                quantity(harm["P_total"], "W"),
                "reference",
            ],
        ],
    )
    kv("of which DC (rank 0)", quantity(harm["P_dc"], "W"))
    kv("of which AC (rank ≥ 1)", quantity(harm["P_ac"], "W"))

    r_dc_boost = boost.dc_resistances()[PRI]
    r_ac_boost = boost.ac_resistances(F_SW)[PRI]
    area_boost = _equivalent_area(boost, PRI)
    # The thermal number, and the one the 4-6 A/mm2 rule is about: the DC bias
    # fills the copper, so its density is the plain I/A whatever f_sw is.
    kv("J at the DC bias", f"{i_dc / area_boost / 1e6:.2f} A/mm²")
    alert(
        "info",
        f"R_ac/R_dc = {r_ac_boost / r_dc_boost:.1f} at {quantity(F_SW, 'Hz')} — "
        f"alarming on paper, and it only ever multiplies the "
        f"{quantity(i_ac, 'A')} of ripple: {harm['P_ac'] / harm['P_total']:.0%} "
        f"of the total. Solid wire survives here BECAUSE the current is mostly "
        f"DC. Widen the ripple towards DCM, where the AC RMS approaches the "
        f"whole of it, and the same winding becomes unusable without Litz.",
    )
    alert(
        "warn",
        f"Feeding the whole RMS to a single frequency prices this inductor at "
        f"{quantity(p_naive, 'W')} instead of {quantity(harm['P_total'], 'W')} "
        f"— a factor {p_naive / harm['P_total']:.1f} of copper bought for "
        f"nothing.",
    )

    figures["dowell_boost_current"] = plot_time_domain(
        i_boost,
        title="Boost inductor current",
        unit="A",
        cycles=2.0,
        levels=True,
        theme=THEME,
    )
    figures["dowell_boost_spectrum"] = plot_spectrum(
        i_boost, title="Boost inductor current spectrum", unit="A", theme=THEME
    )
    # Drawn at the DC bias: that is the field standing in the window at every
    # instant, the ripple only wobbling it by ±d_i/2.
    figures["dowell_boost_section"] = plot_winding_section(
        boost,
        {PRI: i_dc},
        title="Boost inductor cross-section",
        theme=THEME,
    )
    figures["dowell_boost_stack_mmf"] = plot_layer_stack(
        boost, {PRI: i_dc}, mmf=True, title="Boost inductor — stack and MMF", theme=THEME
    )
    figures["dowell_boost_resistance_factor"] = plot_resistance_vs_frequency(
        boost, normalise=True, theme=THEME
    )
    figures["dowell_boost_losses"] = plot_losses_vs_frequency(
        boost,
        {PRI: i_ac},
        title="Ripple loss against frequency",
        subtitle=f"{quantity(i_ac, 'A')} of AC ripple only — the "
        f"{quantity(i_dc, 'A')} DC bias adds a flat "
        f"{quantity(r_dc_boost * i_dc**2, 'W')} at every frequency",
        theme=THEME,
    )

    # ======================================================================== #

    section(None, "Figures")
    for name, figure in figures.items():
        save_figure(figure, OUTPUT / name, formats=("png",))
        print(f"  {OUTPUT / name}.png")
