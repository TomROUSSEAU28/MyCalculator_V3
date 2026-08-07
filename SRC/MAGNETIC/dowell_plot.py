"""
Figures du modele de Dowell.

    plot_layer_stack()                  -> l'empilement (coupe)
    plot_mmf_profile()                  -> le champ que cet empilement construit
    plot_winding_section()              -> les deux, meme abscisse
    plot_resistance_vs_frequency()      -> ou le cuivre cesse d'etre du cuivre
    plot_losses_vs_frequency()          -> ce que ca coute, par enroulement
    plot_current_density_vs_frequency() -> et si le fil tient
    layer_table() / winding_table()     -> les memes chiffres, en tableau

La coupe est la figure dont les autres dependent : la perte d'une couche vient
du champ a ses deux faces, donc de l'endroit ou elle est empilee. Deux
enroulements entrelaces et les memes empiles ont la meme R_dc et des pertes AC
tres differentes.

Lecture de la coupe :
    couleur = enroulement (primaire ROUGE, secondaire BLEU)
    hachure = type de fil
    x = build-up, noyau a gauche, exterieur a droite
    y = largeur de fenetre bw
L'axe x n'est pas a la meme echelle que y : un build de 0.7 mm dans une fenetre
de 24 mm serait illisible.

UNE FIGURE PAR PHASE DE CONDUCTION
----------------------------------
Chaque fonction prend UN dict `currents` et dessine le bobinage comme si ces
courants circulaient au meme instant. Vrai pour un forward ou un DAB, faux pour
un flyback : la, il faut une figure par phase.

Cette regle s'arrete aux figures. Les watts se calculent avec
Dowell_Winding_Structure.harmonic_losses(), un seul appel, tous les
enroulements ensemble, quelle que soit la topologie.

Les themes et save_figure() sont partages par tout le projet et vivent dans
SRC/OTHERS/plot.py.
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
#  Codage visuel
# ============================================================================ #

# Primaire ROUGE, secondaire BLEU : la convention de tout schema de bobinage.
# Les teintes viennent de la palette du theme, pas d'un hex ecrit en dur, pour
# que la coupe reste de la meme famille que les autres figures du projet.
# Le slot du rouge et celui du bleu changent d'un theme a l'autre, d'ou la table.
# Troisieme slot : le neutre d'une courbe de total, ni rouge ni bleu.
_ROLE_SLOTS: dict[str, tuple[int, int, int]] = {
    #          rouge  bleu  total
    "light": (1, 0, 5),
    "dark": (1, 0, 5),
    "paper": (1, 0, 5),
    "blueprint": (1, 0, 4),
    "sage": (0, 1, 4),
}

# La texture porte le type de fil, la couleur porte deja l'enroulement.
# C'est aussi ce qui garde la figure lisible imprimee en noir et blanc.
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
    """La couleur d'un enroulement, la meme dans toutes les figures."""
    red, blue, _ = _slots(theme)
    palette = series(theme)
    return palette[red if winding is WINDING_TYPE.PRIMARY else blue]


def _total_colour(theme: str) -> str:
    return series(theme)[_slots(theme)[2]]


def _winding_label(winding: WINDING_TYPE) -> str:
    return winding.value.capitalize()


# ============================================================================ #
#  Geometrie
# ============================================================================ #


def _thickness(layer: Layer) -> float:
    """Build radial d'une couche [m] : la place qu'elle prend dans la fenetre.

    Une couche Litz est aussi epaisse que les sqrt(k) sous-couches en
    lesquelles Dowell la decoupe, donc la somme est la meme sur
    list_of_layers et sur effective_layers(). C'est ce qui fait tomber
    l'escalier de MMF exactement sur les bords des bandes dessinees.
    """
    wire = layer.wire
    if wire.wire_type in (WIRE_TYPE.ROUND, WIRE_TYPE.LITZ):
        return len(layer.expand()) * wire.diameter
    return wire.height


def _edges(layers: list[Layer]) -> np.ndarray:
    """Build cumule [m] a chaque frontiere : len(layers) + 1 valeurs."""
    if not layers:
        raise ValueError("no layer to plot")
    return np.concatenate(([0.0], np.cumsum([_thickness(lay) for lay in layers])))


def _windings(structure: Dowell_Winding_Structure) -> list[WINDING_TYPE]:
    order = structure.windings()
    if not order:
        raise ValueError("the structure has no layer")
    return order


def _delta_one_frequency(layer: Layer) -> float:
    """Frequence ou l'epaisseur de peau egale h_eff.

    C'est le coude : a gauche le courant remplit encore le conducteur, a droite
    la couche ne conduit plus que sur sa peau et R_ac monte en sqrt(f).
    skin_depth(f) = h_eff donne f = 1 / (pi.mu0.sigma.eta.h_eff^2).
    """
    h_eff = layer.effective_height()
    eta = layer.porosity()
    if h_eff <= 0.0 or eta <= 0.0:
        return float("inf")
    return 1.0 / (pi * MU0 * layer.wire.sigma * eta * h_eff**2)


def _equivalent_area(
    structure: Dowell_Winding_Structure, winding: WINDING_TYPE
) -> float:
    """Section de cuivre [m2] equivalente d'un enroulement.

    Un enroulement dont les couches n'ont pas le meme fil n'a pas de "section
    de fil" unique. Celle qui a un sens est celle qu'implique sa propre R_dc :
    R_dc = L / (sigma.A_eq). Pour un enroulement d'un seul fil, c'est
    exactement la section de ce fil.
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
    """Balayage log : ici ce sont les decades qui comptent, pas les hertz."""
    if f_min <= 0.0:
        raise ValueError(f"f_min must be > 0 on a log sweep, got {f_min}")
    if f_max <= f_min:
        raise ValueError(f"f_max must be > f_min, got {f_max} <= {f_min}")
    if points < 2:
        raise ValueError(f"points must be >= 2, got {points}")
    return np.geomspace(f_min, f_max, points)


def _currents_or_raise(currents: dict) -> dict:
    if not currents or all(value == 0.0 for value in currents.values()):
        raise ValueError("every current is zero — nothing to draw")
    return currents


# ============================================================================ #
#  Panneaux
# ============================================================================ #
# Une fonction par panneau, dessinant dans un axes qu'elle ne possede pas :
# c'est ce qui permet a plot_winding_section() d'empiler la coupe et la MMF
# sans dupliquer ni l'une ni l'autre.


def _twin_axes(ax: Axes, theme: str) -> Axes:
    """Un axe de droite avec la meme chrome discrete que son hote."""
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
    """La coupe : une bande par couche, couleur = enroulement, hachure = fil."""
    c = style(theme)
    layers = structure.list_of_layers
    build = _edges(layers)
    total = float(build[-1])
    window = max(lay.bw for lay in layers)

    for layer, x0, x1 in zip(layers, build, build[1:]):
        colour = winding_colour(theme, layer.winding_type)
        band = Rectangle(
            # Une couche plus courte que la fenetre est centree, pas posee au
            # fond : elle est courte des DEUX cotes.
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

        # Une bande plus etroite qu'un sixieme du build n'a pas la place pour
        # du texte horizontal ; tournee d'un quart de tour elle passe toujours.
        narrow = (x1 - x0) / total < 0.16
        name = layer.name or layer.winding_type.value.lower()
        ax.text(
            0.5 * (x0 + x1),
            # Colle au plafond quand une courbe de MMF arrive : la courbe
            # occupe le milieu du panneau.
            window * (0.96 if label_top else 0.5),
            f"{name}\n{layer.number_of_turns:g} t",
            rotation=90 if narrow else 0,
            ha="center",
            va="top" if label_top else "center",
            color=c["ink"],
            fontsize=9,
            fontfamily=FONT,
            # La hachure passe sous le texte : le fond de la boite le garde
            # lisible sans masquer la texture autour.
            bbox={"facecolor": c["surface"], "edgecolor": "none", "pad": 2.0},
            zorder=5,
        )

    left = 0.0
    if core:
        # Le noyau n'est pas une couche, mais sans lui rien ne dit quel bout du
        # build est l'interieur — et la MMF se lit depuis ce bout-la.
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
    currents: dict,
    bands: bool,
    grid_lines: bool = True,
    headroom: float = 0.18,
) -> None:
    """L'escalier de MMF contre le build, couches en fond."""
    c = style(theme)
    field = [float(np.real(h)) for h in structure.field_profile(currents)]
    # L'escalier est echantillonne sur les sous-couches, les bandes dessinees
    # sur les couches parentes : _thickness() garantit la meme abscisse finale.
    build = _edges(structure.effective_layers())

    if bands:
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

    # H change de signe des que les deux enroulements s'opposent : le zero est
    # un vrai niveau ici.
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
    """Enroulements par couleur, puis types de fil par texture.

    wires : n'activer que sur une figure qui dessine vraiment les hachures.
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
            # Encre neutre, jamais une teinte d'enroulement : la texture est
            # tout le message.
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
    seen = len(structure.effective_layers())
    if seen != len(layers):  # seulement si une couche Litz a ete eclatee
        text += f" · {seen} sub-layers seen by Dowell"
    return text


def _mmf_subtitle(structure: Dowell_Winding_Structure, currents: dict) -> str:
    net = float(np.real(structure.net_ampere_turns(currents)))
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
#  Figures — le bobinage
# ============================================================================ #


def plot_layer_stack(
    structure: Dowell_Winding_Structure,
    currents: dict | None = None,
    mmf: bool = False,
    title: str = "Winding cross-section",
    subtitle: str | None = None,
    theme: str = "light",
    core: bool = True,
    labels: bool = True,
) -> Figure:
    """La coupe : les couches empilees depuis le noyau.

    currents : RMS par enroulement [A]. Uniquement necessaire si `mmf`.
    mmf      : superpose le profil de MMF sur un axe de droite. Pour une figure
               a etudier plutot qu'a survoler, preferer plot_winding_section(),
               qui donne un panneau propre a la MMF.
    core     : bande grise a gauche. C'est la seule chose qui dit quel bout du
               build est l'interieur.
    labels   : nom et nombre de spires dans chaque bande. A couper sur un
               empilement de nombreuses couches fines.
    """
    fig, ax = new_figure(theme, (8.4, 4.6))
    _draw_section(ax, theme, structure, core=core, labels=labels, label_top=mmf)
    text = subtitle if subtitle is not None else _stack_subtitle(structure)

    if mmf:
        if currents is None:
            raise ValueError("mmf=True needs the currents to build the field from")
        twin = _twin_axes(ax, theme)
        # Ni grille ni fond depuis la surcouche : le twin couvre tout l'axes
        # hote, donc tout ce qu'il pose barre les bandes.
        _draw_mmf(
            twin,
            theme,
            structure,
            currents,
            bands=False,
            grid_lines=False,
            headroom=0.55,
        )
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
    currents: dict,
    title: str = "MMF profile",
    subtitle: str | None = None,
    theme: str = "light",
    bands: bool = True,
) -> Figure:
    """Le champ construit par l'ordre d'empilement, en travers de la fenetre.

    H est ce que Dowell eleve au carre : une couche placee la ou |H| est grand
    paie, qu'elle conduise ou non. L'escalier est lineaire par morceaux parce
    que chaque couche est supposee porter son courant uniformement.

    Les deux extremites sont fixees par outer_mmf_fraction, c'est-a-dire par
    l'endroit ou passe le retour de flux.

    `currents` est UNE phase de conduction : un flyback se dessine deux fois.

    bands : ombre les couches derriere la courbe, pour attribuer un coude.
    """
    fig, ax = new_figure(theme, (8.4, 4.4))
    _draw_mmf(ax, theme, structure, _currents_or_raise(currents), bands=bands)
    if bands:
        _section_legend(ax, theme, structure, wires=False)
    add_title(ax, theme, title, subtitle or _mmf_subtitle(structure, currents))
    fig.tight_layout()
    return fig


def plot_winding_section(
    structure: Dowell_Winding_Structure,
    currents: dict,
    title: str = "Winding cross-section",
    subtitle: str | None = None,
    theme: str = "light",
    core: bool = True,
    labels: bool = True,
) -> Figure:
    """La coupe et sa MMF sur une abscisse commune.

    Le haut dit ce qu'est une couche, le bas dit dans quel champ elle est. Le
    partage de l'axe x est ce qui fait pointer un coude de MMF vers la bande
    qui l'a cause.

    `currents` est UNE phase de conduction.
    """
    currents = _currents_or_raise(currents)
    fig, (top, bottom) = new_grid(
        theme, (8.6, 7.6), rows=2, height_ratios=(3.0, 2.2), share_x=True
    )

    _draw_section(top, theme, structure, core=core, labels=labels)
    _draw_mmf(bottom, theme, structure, currents, bands=True)
    # share_x met la bande de noyau du panneau haut dans la plage du bas ; la
    # MMF n'y a simplement rien a tracer, ce qui est correct.
    bottom.set_xlim(*top.get_xlim())

    axis_labels(top, theme, x="")
    _section_legend(top, theme, structure)
    add_title(top, theme, title, subtitle or _stack_subtitle(structure))
    add_title(bottom, theme, "MMF profile", _mmf_subtitle(structure, currents))

    fig.tight_layout()
    # Place pour le bloc de titre du panneau bas, que tight_layout ne compte pas.
    fig.subplots_adjust(hspace=0.34)
    return fig


# ============================================================================ #
#  Figures — contre la frequence
# ============================================================================ #


def _label_ends(
    ax: Axes, theme: str, x: float, entries: list[tuple[float, str]]
) -> None:
    """Nomme chaque courbe a son extremite droite.

    Deux enroulements peuvent finir a une decade ou a un cheveu l'un de
    l'autre. Les etiquettes sont donc placees en espace DISPLAY et poussees
    vers le haut jusqu'a degager celle du dessous.
    """
    ax.get_ylim()  # fige l'autoscale, donc transData est la bonne
    placed: list[float] = []
    spacing = 26.0  # deux lignes de 9 pt et l'air entre elles

    for value, text in sorted(entries):
        pixel = float(ax.transData.transform((x, value))[1])
        shift = 0.0
        for other in placed:
            if abs(pixel + shift - other) < spacing:
                shift = other + spacing - pixel
        placed.append(pixel + shift)
        annotate(
            ax, theme, text, (x, value), (7, shift), role="ink", bold=True, va="center"
        )


def _resistance_sweep(
    structure: Dowell_Winding_Structure,
    frequencies: np.ndarray,
    currents: dict | None = None,
) -> dict[WINDING_TYPE, np.ndarray]:
    """R_ac par enroulement sur le balayage. Un appel modele par frequence."""
    sampled = [structure.ac_resistances(float(f), currents) for f in frequencies]
    return {
        winding: np.array([point[winding] for point in sampled])
        for winding in _windings(structure)
    }


def plot_resistance_vs_frequency(
    structure: Dowell_Winding_Structure,
    currents: dict | None = None,
    f_min: float = 1e3,
    f_max: float = 10e6,
    points: int = 160,
    normalise: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    theme: str = "light",
    mark_delta_one: bool = True,
) -> Figure:
    """R_ac contre la frequence, une courbe par enroulement.

    currents : QUELLE resistance.
        None — chaque enroulement excite seul sous 1 A. Resistance PROPRE :
        ce que lit un pont LCR, et ce que voit une phase de flyback.
        Un point de fonctionnement — resistance EFFECTIVE, R = P/I2, tous les
        champs presents. Obligatoire des que les enroulements conduisent
        ENSEMBLE, et seule version qui voit l'entrelacement. Les enroulements
        sans courant sont ecartes.

    normalise : trace R_ac/R_dc. Meme forme, mais dit si 2 mOhm est un bon
        2 mOhm, et met deux enroulements de tailles differentes sur une echelle
        lisible.

    mark_delta_one : trait a la frequence ou l'epaisseur de peau tombe a h_eff
        de la couche la plus epaisse. A gauche la courbe est plate, a droite
        elle monte : c'est toute la contrainte de choix du fil.
    """
    frequencies = _frequencies(f_min, f_max, points)
    r_dc = structure.dc_resistances()
    r_ac = _resistance_sweep(structure, frequencies, currents)

    # Un enroulement sans courant n'a pas de resistance effective (nan) : ce
    # n'est pas une donnee manquante, c'est une grandeur qui n'existe pas.
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
            # Son propre plancher : un plancher commun entre un primaire de
            # 67 spires et un secondaire de 26 ne voudrait rien dire.
            ax.axhline(r_dc[winding], color=colour, linewidth=1.0, alpha=0.45, zorder=2)

    _label_ends(ax, theme, float(frequencies[-1]), ends)

    if mark_delta_one:
        for index, winding in enumerate(windings):
            thickest = max(
                (
                    lay
                    for lay in structure.effective_layers()
                    if lay.winding_type is winding
                ),
                key=lambda lay: lay.effective_height(),
            )
            knee = _delta_one_frequency(thickest)
            if f_min <= knee <= f_max:
                vertical_reference_line(
                    ax,
                    theme,
                    knee,
                    f"Δ=1 · {_winding_label(winding)}",
                    # Deux coudes peuvent tomber a un cheveu l'un de l'autre :
                    # la seconde etiquette descend d'une ligne.
                    offset=(4, -4 - 14 * index),
                )

    engineering_ticks(ax, theme, axis="x", unit="Hz")
    if normalise:
        engineering_ticks(ax, theme, axis="y")
        axis_labels(ax, theme, x="Frequency", y="R_ac / R_dc")
    else:
        engineering_ticks(ax, theme, axis="y", unit="Ω")
        axis_labels(ax, theme, x="Frequency", y="AC resistance")

    if subtitle is None:
        floors = " · ".join(
            f"R_dc {_winding_label(w)} {quantity(r_dc[w], 'Ω')}" for w in windings
        )
        # Laquelle des deux resistances est sur l'axe ne se devine pas depuis
        # les courbes.
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
    currents: dict,
    f_min: float = 1e3,
    f_max: float = 10e6,
    points: int = 160,
    title: str = "Copper loss against frequency",
    subtitle: str | None = None,
    theme: str = "light",
    dc_reference: bool = True,
) -> Figure:
    """Ce que couteraient ces courants si toute leur RMS etait a la frequence
    de l'abscisse. C'est la question contre laquelle on choisit une f_sw.

    Ce n'est PAS la perte d'une forme d'onde reelle : celle-la est etalee sur
    un spectre et se calcule avec harmonic_losses().

    La courbe d'un enroulement contient tous les watts dissipes dans SES
    couches, proximite imposee par l'autre comprise. D'ou une courbe au-dessus
    de zero pour un enroulement qui ne conduit pas.

    `currents` est UNE phase de conduction.

    dc_reference : trait a la perte DC des memes courants. L'ecart avec la
        courbe de total est tout le cout des effets AC.
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

    # Un enroulement qui ne dissipe exactement rien — blinde par l'autre — n'a
    # pas de courbe qu'un axe log puisse tracer.
    drawn = [w for w in windings if max(per_winding[w]) > 0.0]
    if not drawn:
        raise ValueError("nothing is dissipated at any frequency — nothing to draw")

    curves = [
        (_winding_label(w), per_winding[w], winding_colour(theme, w), 1.8)
        for w in drawn
    ]
    # Total en dernier et plus epais : il est trace par-dessus ceux qu'il somme.
    # Inutile s'il n'y a qu'une courbe : il tomberait dessus.
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

    # Valeurs a droite : un axe log sur moins de deux decades ne donne qu'une
    # ou deux graduations etiquetees.
    _label_ends(
        ax,
        theme,
        float(frequencies[-1]),
        [(values[-1], quantity(values[-1], "W")) for _, values, _, _ in curves],
    )

    engineering_ticks(ax, theme, axis="x", unit="Hz")
    engineering_ticks(ax, theme, axis="y", unit="W")
    axis_labels(ax, theme, x="Frequency", y="Copper loss")

    if len(curves) > 1:
        legend(
            ax, theme, [(name, colour) for name, _, colour, _ in curves], kind="line"
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
    currents: dict,
    f_min: float = 1e3,
    f_max: float = 10e6,
    points: int = 160,
    j_limit: float | None = None,
    title: str = "Current density against frequency",
    subtitle: str | None = None,
    theme: str = "light",
) -> Figure:
    """La densite de courant reellement vue, en A/mm2.

    En DC le courant remplit le cuivre et J = I/A. Au-dessus du coude, non : il
    se tasse dans une peau. Ce qui est trace est la densite uniforme
    EQUIVALENTE, celle qui dissiperait la meme perte dans toute la section :

        J_eq(f) = (I / A_eq) . sqrt(R_ac(f) / R_dc)

    donc la regle du pouce des 4-6 A/mm2, qui est une regle thermique, lui
    s'applique encore. La densite de pointe dans la peau est plus elevee.

    R_ac est la resistance EFFECTIVE sous `currents`, pas la propre : la
    densite doit suivre la perte que l'enroulement a vraiment.

    j_limit : plafond de conception [A/mm2]. La ou une courbe le croise est la
        frequence a laquelle le fil doit changer.

    NB : seul axe du projet hors SI de base. A/mm2 est l'unite des datasheets ;
    les memes valeurs en A/m2 se liraient "5 MA/m2".
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
            ax,
            theme,
            j_limit,
            f"limit {j_limit:.1f} A/mm²",
            kind="limit",
            offset=(2, 6),
        )

    top = 0.0
    floors: list[str] = []
    ends: list[tuple[float, str]] = []
    for winding in windings:
        colour = winding_colour(theme, winding)
        area = _equivalent_area(structure, winding)
        j_dc = currents[winding] / area / 1e6  # 1e6 : m2 -> mm2, le modele reste en SI
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
        # Sa densite DC en plancher : l'ecart vertical a la courbe est toute la
        # penalite imposee par la frequence.
        ax.axhline(j_dc, color=colour, linewidth=1.0, alpha=0.45, zorder=2)
        ends.append(
            (float(values[-1]), f"{_winding_label(winding)}\n{values[-1]:.1f} A/mm²")
        )
        top = max(top, float(values[-1]))

    ceiling = max(top, j_limit or 0.0)
    ax.set_ylim(0.0, ceiling * 1.15)
    # Apres les limites : les etiquettes sont ecartees en espace display, ce
    # qui ne veut dire quelque chose qu'une fois la hauteur connue.
    _label_ends(ax, theme, float(frequencies[-1]), ends)
    engineering_ticks(ax, theme, axis="x", unit="Hz")
    axis_labels(ax, theme, x="Frequency", y="Equivalent current density [A/mm²]")

    add_title(ax, theme, title, subtitle or " · ".join(floors))
    fig.tight_layout()
    return fig


# ============================================================================ #
#  Tableaux
# ============================================================================ #
# Les cellules restent en cp1252 : une console Windows y est encore par defaut,
# et un tableau qui leve UnicodeEncodeError sur "Ω" est un tableau imprimable
# par personne. D'ou "Ohm" et "eta" ecrits en toutes lettres.


def layer_table(
    structure: Dowell_Winding_Structure, freq: float, unit_turns: str = "t"
) -> pd.DataFrame:
    """Le jumeau tabulaire de la coupe : une ligne par couche VUE PAR DOWELL.

    Les couches Litz apparaissent eclatees en sous-couches de brins, parce que
    c'est la-dessus que la perte est calculee.
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
    currents: dict | None = None,
) -> pd.DataFrame:
    """Ce qu'est chaque enroulement, a une frequence.

    Sans `currents` : geometrie et resistances PROPRES, qui ne dependent pas de
    ce qui circule. Avec : resistances EFFECTIVES, densites, perte, et une
    ligne TOTAL — la seule que prend un budget thermique.

    Un enroulement sans courant garde sa perte (la proximite est reelle) et
    recoit un tiret partout ou il faudrait diviser par un courant.
    """
    windings = _windings(structure)
    r_dc = structure.dc_resistances()
    r_ac = structure.ac_resistances(freq, currents)

    def cell(value: float, unit: str = "", digits: int = 2) -> str:
        """Un tiret pour ce qui n'existe pas, plutot qu'un nan qui fait semblant."""
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
        rows["TOTAL"] = ["-"] * (len(columns) - 1) + [quantity(sum(loss.values()), "W")]

    frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    frame.index.name = "Winding"
    return frame


# ============================================================================ #
#  Demo
# ============================================================================ #

if __name__ == "__main__":
    from SRC.MAGNETIC.dowell import POLARITY, Wire
    from SRC.OTHERS.plot import save_figure
    from SRC.OTHERS.terminal import alert, dataframe, kv, section, use_theme
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
    # 200 rangs = 20 MHz a 100 kHz. Des fronts ideaux donnent un spectre en 1/n
    # et une perte par rang en n^-1.5 : la somme converge lentement, donc une
    # somme tronquee est toujours une borne inferieure.
    N_MAX = 200

    figures: dict[str, Figure] = {}

    # ======================================================================== #
    #  1 — Flyback : deux enroulements qui ne conduisent jamais ensemble
    # ======================================================================== #
    section(1, "Flyback — une figure par phase de conduction")

    bw = pi * 7.62e-3  # carcasse RM10
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
            # Bobine PAR-DESSUS le primaire : sa spire moyenne est plus longue
            # du build sur lequel il repose.
            mlt=mlt + 8 * D_PRI,
            wire=Wire(wire_type=WIRE_TYPE.ROUND, diameter=D_SEC),
            winding_type=SEC,
            polarity=POLARITY.NEGATIVE,
        )
    )

    # Flyback DCM. Le flux ne saute pas a l'ouverture, donc les ampere-tours se
    # passent le relais : N_pri.I_pri,pk = N_sec.I_sec,pk. Le pic secondaire
    # n'est pas un parametre libre.
    t_on = 6.30e-6
    t_dem = 1.98e-6
    i_pri_pk = 0.5
    i_sec_pk = i_pri_pk * N_PRI / N_SEC

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

    # --- Les watts : un seul appel, les deux enroulements ------------------- #
    # Le fait que les deux ne conduisent jamais ensemble est deja porte par
    # l'angle des phaseurs. Rien a decouper ici.
    harm = flyback.harmonic_losses(signals, n_max=N_MAX)
    section(None, "Pertes cuivre")
    kv("P_dc (rang 0)", quantity(harm["P_dc"], "W"))
    kv("P_ac (rang >= 1)", quantity(harm["P_ac"], "W"))
    kv("P_total", quantity(harm["P_total"], "W"))
    alert(
        "ok",
        f"{quantity(harm['P_total'], 'W')} — un appel harmonic_losses(), tous "
        f"les enroulements, tous les rangs jusqu'a n = {N_MAX}. Seule "
        f"approximation restante : la troncature.",
    )

    # --- Les figures : une par phase ---------------------------------------- #
    # La RMS de chaque enroulement sur la periode ENTIERE porte deja son
    # rapport cyclique, donc chaque phase se dessine avec un seul courant.
    phases = (
        ("on", "ON · primary conducts", {PRI: i_pri.rms(), SEC: 0.0}),
        ("off", "OFF · secondary conducts", {PRI: 0.0, SEC: i_sec.rms()}),
    )

    for suffix, label, currents in phases:
        section(None, f"Phase {label}")
        H = flyback.field_profile(currents)
        kv("MMF at the boundaries", " → ".join(quantity(float(h), "A/m") for h in H))
        dataframe(winding_table(flyback, F_SW, currents))

        figures[f"dowell_flyback_section_{suffix}"] = plot_winding_section(
            flyback, currents, title=f"Flyback — phase {label}", theme=THEME
        )
        figures[f"dowell_flyback_losses_{suffix}"] = plot_losses_vs_frequency(
            flyback, currents, title=f"Copper loss — phase {label}", theme=THEME
        )
        figures[f"dowell_flyback_density_{suffix}"] = plot_current_density_vs_frequency(
            flyback,
            currents,
            j_limit=6.0,
            title=f"Current density — phase {label}",
            theme=THEME,
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
    # Pas de `currents` : sur un flyback la resistance propre EST celle que
    # voit chaque phase, l'autre enroulement etant ouvert.
    figures["dowell_flyback_resistance"] = plot_resistance_vs_frequency(
        flyback, theme=THEME
    )
    figures["dowell_flyback_resistance_factor"] = plot_resistance_vs_frequency(
        flyback, normalise=True, theme=THEME
    )

    # ======================================================================== #
    #  2 — Inductance de boost : un seul enroulement, rien pour compenser la MMF
    # ======================================================================== #
    section(2, "Inductance de boost — biais DC contre ondulation")

    V_IN, V_OUT, P_OUT = 12.0, 24.0, 36.0
    L_BOOST = 47e-6
    duty = 1.0 - V_IN / V_OUT
    i_avg = P_OUT / V_IN  # l'inductance de boost porte le courant d'ENTREE
    d_i = V_IN * duty / (L_BOOST * F_SW)  # ondulation crete-crete

    i_boost = ElectronicPeriodicSignal.from_breakpoints(
        "I_L boost",
        times=[0.0, duty * T_SW, T_SW],
        values=[i_avg - 0.5 * d_i, i_avg + 0.5 * d_i, i_avg - 0.5 * d_i],
        n_samples=8192,
        period=T_SW,
    )

    # La MMF d'un transformateur revient a zero parce que ses ampere-tours
    # s'equilibrent. Celle d'une inductance ne peut pas : Sigma n.i est tout
    # l'interet de l'objet. L'escalier monte donc de facon monotone, la spire
    # la plus externe est dans le champ maximal, et l'entrelacement — le grand
    # levier du flyback — n'existe pas ici. Il ne reste que le choix du fil.
    #
    # outer_mmf_fraction = 0 par defaut : entrefer en jambe centrale, donc le
    # retour est a l'interieur et H tombe a zero sur la face EXTERNE.
    BW_L, MLT_L, D_L = 19.5e-3, 53e-3, 0.9e-3
    TURNS_PER_LAYER, LAYERS = 20, 2

    boost = Dowell_Winding_Structure()
    for index in range(LAYERS):
        boost.add_layer(
            Layer(
                name=f"L{index + 1}",
                number_of_turns=TURNS_PER_LAYER,
                bw=BW_L,
                # Chaque couche repose sur la precedente : un diametre de build
                # ajoute 2.pi.d a la spire moyenne.
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
    kv(
        "Inductor",
        f"{quantity(L_BOOST, 'H')}, {LAYERS}×{TURNS_PER_LAYER} t of "
        f"{quantity(D_L, 'm')} round, gapped centre leg",
    )
    kv(
        "Current",
        f"{quantity(i_dc, 'A')} DC + {quantity(d_i, 'A')} pk-pk ripple "
        f"→ {quantity(i_rms, 'A')} RMS, of which {quantity(i_ac, 'A')} is AC",
    )
    dataframe(signal_table(i_boost, unit="A"))
    dataframe(layer_table(boost, F_SW))
    # L'ondulation seule : ce sur quoi les effets AC ont vraiment prise.
    dataframe(winding_table(boost, F_SW, {PRI: i_ac}))

    harm_boost = boost.harmonic_losses({PRI: i_boost}, n_max=N_MAX)
    section(None, "Pertes cuivre")
    kv("P_dc (rang 0)", quantity(harm_boost["P_dc"], "W"))
    kv("P_ac (rang >= 1)", quantity(harm_boost["P_ac"], "W"))
    kv("P_total", quantity(harm_boost["P_total"], "W"))

    r_dc_boost = boost.dc_resistances()[PRI]
    r_ac_boost = boost.ac_resistances(F_SW)[PRI]
    area_boost = _equivalent_area(boost, PRI)
    # Le chiffre thermique, celui dont parle la regle des 4-6 A/mm2 : le biais
    # DC remplit le cuivre, donc sa densite est le I/A nu quelle que soit f_sw.
    kv("J at the DC bias", f"{i_dc / area_boost / 1e6:.2f} A/mm²")
    alert(
        "info",
        f"R_ac/R_dc = {r_ac_boost / r_dc_boost:.1f} a {quantity(F_SW, 'Hz')} — "
        f"alarmant sur le papier, et ca ne multiplie jamais que les "
        f"{quantity(i_ac, 'A')} d'ondulation : "
        f"{harm_boost['P_ac'] / harm_boost['P_total']:.0%} du total. Le fil "
        f"plein tient ici PARCE QUE le courant est surtout du DC. Elargir "
        f"l'ondulation vers le DCM et le meme bobinage devient inutilisable "
        f"sans Litz.",
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
    # Dessine au biais DC : c'est le champ present dans la fenetre a chaque
    # instant, l'ondulation ne le faisant osciller que de ±d_i/2.
    figures["dowell_boost_section"] = plot_winding_section(
        boost, {PRI: i_dc}, title="Boost inductor cross-section", theme=THEME
    )
    figures["dowell_boost_stack_mmf"] = plot_layer_stack(
        boost,
        {PRI: i_dc},
        mmf=True,
        title="Boost inductor — stack and MMF",
        theme=THEME,
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
