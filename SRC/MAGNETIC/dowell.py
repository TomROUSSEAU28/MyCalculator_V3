"""
Modele de Dowell : pertes cuivre (peau + proximite) d'un bobinage, couche par
couche.

Principe : la perte d'une couche depend du champ H a ses DEUX faces, donc de
l'endroit ou elle est empilee, pas seulement de ce qu'elle est.

Regle d'usage :
    harmonic_losses(tous_les_signaux) -> le nombre a mettre dans un budget
    thermique. Un seul appel, tous les enroulements, toutes les topologies.

Tout le reste (loss_at_frequency, ac_resistances, field_profile) sert a lire
un point de fonctionnement ou a tracer, pas a chiffrer.
"""

from __future__ import annotations

import sys
from enum import Enum
from math import pi, sqrt
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Doit preceder les imports 'SRC.*' : sans cela le script n'est importable
# qu'en module (python -m SRC.MAGNETIC.dowell), pas en direct.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from SRC.constant import MU0, SIGMA_CU_20C

__all__ = [
    "POLARITY",
    "WINDING_TYPE",
    "WIRE_TYPE",
    "Dowell_Winding_Structure",
    "Layer",
    "Wire",
    "skin_depth",
]


def skin_depth(
    freq: float, sigma: float = SIGMA_CU_20C, porosity: float = 1.0
) -> float:
    """Epaisseur de peau [m]. La porosite corrige la conductivite vue par
    la couche : moins de cuivre en largeur, peau apparente plus grande."""
    if freq < 1e-3:
        return float("inf")
    return sqrt(2.0 / (2.0 * pi * freq * MU0 * sigma * porosity))


class WINDING_TYPE(Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"


class WIRE_TYPE(Enum):
    ROUND = "ROUND"
    SQUARE = "SQUARE"
    FOIL = "FOIL"
    LITZ = "LITZ"


class POLARITY(Enum):
    POSITIVE = 1
    NEGATIVE = -1


# ============================================================================ #
#  Conducteur
# ============================================================================ #


class Wire(BaseModel):
    """Un conducteur. ROUND/LITZ -> diameter. SQUARE/FOIL -> width + height."""

    wire_type: WIRE_TYPE
    diameter: float = 0.0
    width: float = 0.0
    height: float = 0.0
    sigma: float = SIGMA_CU_20C
    number_of_strands: int = 1  # LITZ uniquement

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _check(self) -> Wire:
        t = self.wire_type
        if self.sigma <= 0.0:
            raise ValueError("sigma doit etre > 0")

        match t:
            case WIRE_TYPE.ROUND | WIRE_TYPE.LITZ:
                if self.diameter <= 0.0:
                    raise ValueError(f"{t.value} : 'diameter' obligatoire et > 0")
                if self.width or self.height:
                    raise ValueError(f"{t.value} : utiliser 'diameter', pas w/h")

            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                if self.width <= 0.0 or self.height <= 0.0:
                    raise ValueError(f"{t.value} : 'width' et 'height' > 0")
                if self.diameter:
                    raise ValueError(f"{t.value} : utiliser w/h, pas 'diameter'")

        if t is WIRE_TYPE.LITZ:
            if self.number_of_strands < 2:
                raise ValueError("LITZ : au moins 2 brins, sinon utiliser ROUND")
        elif self.number_of_strands != 1:
            raise ValueError(f"{t.value} : 'number_of_strands' est propre au LITZ")

        return self


# ============================================================================ #
#  Couche
# ============================================================================ #


class Layer(BaseModel):
    """Une couche de bobinage.

    bw  : largeur de fenetre [m]
    mlt : longueur moyenne d'une spire [m]
    polarity : sens du bobinage, decide si sa MMF s'ajoute ou se retranche
    current_divider : k pour une sous-couche Litz, 1 sinon
    """

    name: str = ""
    number_of_turns: float
    bw: float
    mlt: float
    wire: Wire
    winding_type: WINDING_TYPE
    polarity: POLARITY
    current_divider: float = 1.0

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _check(self) -> Layer:
        for f in ("number_of_turns", "bw", "mlt", "current_divider"):
            if getattr(self, f) <= 0.0:
                raise ValueError(f"'{f}' doit etre > 0 (couche '{self.name}')")

        if self.wire.wire_type is WIRE_TYPE.LITZ:
            return self  # le parent Litz n'est jamais evalue, seul expand() l'est

        # Le cuivre doit tenir dans bw. Au-dela, porosity() sature a 1.0 et le
        # modele rend un resultat faux en silence : on refuse la couche.
        cu = self.number_of_turns * (
            self.effective_height()
            if self.wire.wire_type is WIRE_TYPE.ROUND
            else self.wire.width
        )
        if cu > self.bw * (1.0 + 1e-12):
            raise ValueError(
                f"couche '{self.name}' : {cu * 1e3:.2f} mm de cuivre pour "
                f"bw = {self.bw * 1e3:.2f} mm -> reduire le nombre de spires "
                f"ou repartir sur plusieurs couches"
            )

        # Invariant de Dowell : la section implicite (bw x h_eff x eta) doit
        # egaler la vraie section de cuivre. Verifie ici, une fois, plutot que
        # dans la boucle de calcul.
        implied = self.bw * self.effective_height() * self.porosity()
        true_area = self.number_of_turns * self.copper_area()
        if true_area > 0.0 and abs(implied / true_area - 1.0) > 1e-9:
            raise ValueError(f"couche '{self.name}' : geometrie incoherente")

        return self

    def effective_height(self) -> float:
        """Epaisseur equivalente [m]. Un fil rond est remplace par le carre de
        meme section, c'est l'hypothese de base de Dowell."""
        match self.wire.wire_type:
            case WIRE_TYPE.ROUND | WIRE_TYPE.LITZ:
                return self.wire.diameter * sqrt(pi / 4.0)
            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                return self.wire.height
            case _:
                return 0.0

    def porosity(self) -> float:
        """eta : part de la largeur de fenetre reellement occupee par du cuivre."""
        match self.wire.wire_type:
            case WIRE_TYPE.ROUND:
                cu = self.number_of_turns * self.effective_height()
            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                cu = self.number_of_turns * self.wire.width
            case _:
                return 0.0
        return min(cu / self.bw, 1.0)

    def delta(self, freq: float) -> float:
        """Epaisseur normalisee de Dowell : h_eff / epaisseur de peau.
        Delta < 1 -> le courant remplit le conducteur. Delta > 1 -> il ne
        conduit plus que sur sa peau."""
        return self.effective_height() / skin_depth(
            freq, self.wire.sigma, self.porosity()
        )

    def copper_area(self) -> float:
        """Section de cuivre d'un conducteur [m2]."""
        match self.wire.wire_type:
            case WIRE_TYPE.ROUND:
                return pi * (self.wire.diameter / 2.0) ** 2
            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                return self.wire.width * self.wire.height
            case WIRE_TYPE.LITZ:
                return (
                    self.wire.number_of_strands * pi * (self.wire.diameter / 2.0) ** 2
                )
            case _:
                return 0.0

    def dc_resistance(self) -> float:
        """R = L / (sigma . S) sur toute la couche [Ohm]."""
        area = self.copper_area()
        if area <= 0.0:
            return float("inf")
        return self.number_of_turns * self.mlt / (self.wire.sigma * area)

    def expand(self) -> list[Layer]:
        """Litz -> sqrt(k) sous-couches de brins (Geng eq. 3).
        Les autres types se renvoient eux-memes."""
        if self.wire.wire_type is not WIRE_TYPE.LITZ:
            return [self]

        k = self.wire.number_of_strands
        n_sub = max(1, round(sqrt(k)))
        strand = Wire(
            wire_type=WIRE_TYPE.ROUND,
            diameter=self.wire.diameter,
            sigma=self.wire.sigma,
        )
        return [
            Layer(
                name=f"{self.name}[{j + 1}/{n_sub}]",
                number_of_turns=self.number_of_turns * k / n_sub,
                bw=self.bw,
                mlt=self.mlt,
                wire=strand,
                winding_type=self.winding_type,
                polarity=self.polarity,
                current_divider=k,
            )
            for j in range(n_sub)
        ]


def _dowell_G(delta: float) -> tuple[float, float]:
    """Les deux fonctions de Dowell : g1 pour la peau, g2 pour la proximite."""
    if delta < 1e-3:
        return 1.0, 0.5  # limite DC
    if delta > 50.0:
        return delta, 0.0  # asymptote, evite l'overflow des cosh
    den = np.cosh(2 * delta) - np.cos(2 * delta)
    g1 = delta * (np.sinh(2 * delta) + np.sin(2 * delta)) / den
    g2 = delta * (np.sinh(delta) * np.cos(delta) + np.cosh(delta) * np.sin(delta)) / den
    return g1, g2


# ============================================================================ #
#  Bobinage
# ============================================================================ #


class Dowell_Winding_Structure(BaseModel):
    list_of_layers: list[Layer] = Field(default_factory=list)

    outer_mmf_fraction: float = 0.0
    """Part de la MMF absorbee par le chemin de retour EXTERIEUR.

    0.0 : entrefer cote noyau interieur (jambe centrale, tore) -> H = 0 sur la
          face externe. Cas normal.
    1.0 : tout cote exterieur -> H = 0 sur la face interne.
    0.5 : entrefer reparti sur les 3 jambes d'un E-core.

    Sans effet si Sigma n.i = 0 (vrai transformateur) : les deux faces sont
    alors nulles. Decisif sur une inductance ou un flyback.
    """

    def add_layer(self, layer: Layer) -> None:
        """Ajouter DE L'INTERIEUR VERS L'EXTERIEUR : celle contre le noyau en
        premier. field_profile() depend de cet ordre."""
        self.list_of_layers.append(layer)

    def effective_layers(self) -> list[Layer]:
        """Les couches vues par le modele, Litz eclate en sous-couches."""
        return [sub for lay in self.list_of_layers for sub in lay.expand()]

    def windings(self) -> list[WINDING_TYPE]:
        """Les enroulements presents, dans l'ordre de bobinage (pas un set :
        l'ordre doit etre stable d'une execution a l'autre)."""
        return list(dict.fromkeys(lay.winding_type for lay in self.list_of_layers))

    # ---------------------------------------------------------------- champ --

    def field_profile(self, currents: dict) -> list[complex]:
        """MMF [A/m] aux frontieres de couche, de l'interieur vers l'exterieur.

        `currents` accepte des reels (un point de fonctionnement) ou des
        phaseurs complexes (un rang harmonique) : le profil n'est qu'une somme
        ponderee, il porte l'angle de chacun sans rien avoir a changer."""
        boundaries: list[complex] = [0.0]
        ampere_turns: complex = 0.0
        for layer in self.effective_layers():
            ampere_turns += (
                layer.number_of_turns
                * currents.get(layer.winding_type, 0.0)
                / layer.current_divider
                * layer.polarity.value
            )
            boundaries.append(ampere_turns / layer.bw)
        shift = (1.0 - self.outer_mmf_fraction) * boundaries[-1]
        return [h - shift for h in boundaries]

    def net_ampere_turns(self, currents: dict) -> complex:
        """Somme signee des ampere-tours. ~0 pour un vrai transformateur,
        non nul pour une inductance ou un flyback."""
        return sum(
            layer.number_of_turns
            * currents.get(layer.winding_type, 0.0)
            * layer.polarity.value
            for layer in self.list_of_layers
        )

    # --------------------------------------------------------------- pertes --

    def layer_losses(self, freq: float, currents: dict) -> list[float]:
        """Perte [W] de chaque couche effective, a une frequence.

        Forme hermitienne : sur des courants reels elle redonne la formule
        classique, sur des phaseurs elle garde l'angle. Deux MMF en opposition
        se compensent, deux MMF en quadrature ne se compensent pas.
        Prendre le module a la place detruirait cette information."""
        H = self.field_profile(currents)

        losses = []
        for i, layer in enumerate(self.effective_layers()):
            H_in, H_out = complex(H[i]), complex(H[i + 1])
            h_eff, eta = layer.effective_height(), layer.porosity()
            if (H_in == 0.0 and H_out == 0.0) or h_eff == 0.0 or eta == 0.0:
                losses.append(0.0)
                continue

            g1, g2 = _dowell_G(layer.delta(freq))
            bracket = (abs(H_out) ** 2 + abs(H_in) ** 2) * g1 - 4.0 * (
                H_out * H_in.conjugate()
            ).real * g2
            losses.append(
                layer.bw * layer.mlt * bracket / (h_eff * eta * layer.wire.sigma)
            )
        return losses

    def losses_by_layer(self, freq: float, currents: dict) -> list[float]:
        """Pertes regroupees sur les couches d'origine (Litz recolle)."""
        flat = self.layer_losses(freq, currents)
        out, i = [], 0
        for lay in self.list_of_layers:
            n = len(lay.expand())
            out.append(sum(flat[i : i + n]))
            i += n
        return out

    def losses_by_winding(self, freq: float, currents: dict) -> dict:
        """Pertes par enroulement.

        Contient TOUT ce que dissipent ses couches, y compris la proximite que
        le champ de l'autre lui impose : un enroulement qui ne conduit pas peut
        etre au-dessus de zero."""
        totals = dict.fromkeys(self.windings(), 0.0)
        for lay, loss in zip(self.list_of_layers, self.losses_by_layer(freq, currents)):
            totals[lay.winding_type] += loss
        return totals

    def loss_at_frequency(self, freq: float, currents: dict) -> float:
        """Perte totale [W] si tout le courant donne se trouvait a `freq`.
        Outil de lecture : pour un chiffre, passer par harmonic_losses()."""
        return sum(self.layer_losses(freq, currents))

    def dc_loss(self, currents_rms: dict) -> float:
        """Pertes Joule pures, sans effet de peau ni de proximite."""
        return sum(
            layer.dc_resistance() * (currents_rms.get(layer.winding_type, 0.0) ** 2)
            for layer in self.list_of_layers
        )

    # ---------------------------------------------------------- resistances --

    def dc_resistances(self) -> dict:
        """R_dc par enroulement [Ohm]. Constante purement geometrique."""
        out: dict = {}
        for lay in self.list_of_layers:
            out[lay.winding_type] = out.get(lay.winding_type, 0.0) + lay.dc_resistance()
        return out

    def ac_resistances(self, freq: float, currents: dict | None = None) -> dict:
        """R_ac par enroulement [Ohm]. Deux conventions, parce qu'une
        resistance AC n'a pas de sens hors du courant qui la traverse.

        currents = None : chaque enroulement excite seul sous 1 A, les autres
            ouverts. Resistance PROPRE, celle que lit un pont LCR.

        currents = {...} : resistance EFFECTIVE au point de fonctionnement,
            R = P / I^2, tous les champs presents. Obligatoire des que deux
            enroulements conduisent ensemble, et seule version qui voit
            l'entrelacement. Un enroulement sans courant renvoie nan : sa perte
            existe mais aucun courant a lui ne l'explique -> losses_by_winding().
        """
        if currents is None:
            return {
                w: self.losses_by_winding(freq, {w: 1.0})[w] for w in self.windings()
            }

        losses = self.losses_by_winding(freq, currents)
        return {
            w: (losses[w] / i**2 if (i := currents.get(w, 0.0)) else float("nan"))
            for w in self.windings()
        }

    # ------------------------------------------------------------ reference --

    def harmonic_losses(
        self,
        signals: dict,
        n_max: int = 40,
        threshold: float = 0.0,
    ) -> dict[str, float]:
        """LA methode de calcul des pertes. Superposition harmonique, phases
        conservees.

        Exact au sens du modele de Dowell, pour n'importe quelle forme d'onde
        et n'importe quelle topologie : le milieu est lineaire et deux rangs
        differents sont orthogonaux sur la periode, donc leurs pertes
        s'additionnent sans terme croise. Seule approximation : la troncature
        a n_max.

        PASSER TOUS LES ENROULEMENTS DANS UN SEUL APPEL, y compris quand ils
        ne conduisent jamais en meme temps (flyback). Le fait qu'ils soient
        disjoints dans le temps est deja code dans l'angle des phaseurs.
        Sommer un appel par phase jette le terme croise entre les deux, et
        sous-estime la perte.

        Le decoupage par phase reste la bonne facon de DESSINER : un profil de
        MMF est celui d'un instant (voir dowell_plot.py).

        Tous les signaux doivent partager la meme origine des temps et la meme
        periode. C'est relatif : un decalage entre deux signaux devient un
        dephasage, et rien ne le signalerait ensuite.
        """
        if not signals:
            return {"P_dc": 0.0, "P_ac": 0.0, "P_total": 0.0}

        fundamentals = {w: sig.fundamental() for w, sig in signals.items()}
        f0 = next(iter(fundamentals.values()))
        for w, f in fundamentals.items():
            if abs(f / f0 - 1.0) > 1e-9:
                raise ValueError(
                    f"'{w.value}' a f0 = {f:.6g} Hz au lieu de {f0:.6g} Hz : "
                    f"les signaux doivent partager la meme periode et la meme "
                    f"origine des temps, sinon leur dephasage est faux"
                )

        spectra = {w: sig.harmonics_rms(n_max, threshold) for w, sig in signals.items()}

        p_dc = 0.0
        p_ac = 0.0
        for rank in range(n_max + 1):
            currents = {w: spectrum.get(rank, 0j) for w, spectrum in spectra.items()}
            if all(i == 0.0 for i in currents.values()):
                continue

            if rank == 0:
                # 0 Hz : pas de courant de Foucault, donc pas de phase a garder.
                p_dc += self.dc_loss({w: abs(c) for w, c in currents.items()})
            else:
                p_ac += self.loss_at_frequency(rank * f0, currents)

        return {"P_dc": p_dc, "P_ac": p_ac, "P_total": p_dc + p_ac}
