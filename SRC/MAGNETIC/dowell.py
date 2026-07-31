import sys
from enum import Enum
from math import pi, sqrt
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))
from SRC.constant import MU0, SIGMA_CU_20C


def skin_depth(
    freq: float, sigma: float = SIGMA_CU_20C, porosity: float = 1.0
) -> float:
    if freq < 1e-3:
        return float("inf")
    omega = 2.0 * pi * freq
    return sqrt(2.0 / (omega * MU0 * sigma * porosity))


def porosity(number_of_turns: int, wire_diameter: float, winding_width: float) -> float:
    return min((number_of_turns * wire_diameter) / winding_width, 1.0)


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


class Wire(BaseModel):
    wire_type: WIRE_TYPE
    diameter: float = 0.0
    width: float = 0.0  # Width of the wire (for square or foil wire)
    height: float = 0.0  # Height of the wire
    sigma: float = SIGMA_CU_20C
    number_of_strands: int = 1  # Number of strands in the wire (for Litz wire)


class Layer(BaseModel):
    name: str = ""
    number_of_turns: int  # Number of turns in the winding layer
    bw: float  # Bandwidth of the winding layer (in meters)
    mlt: float  # Mean Length Turn (MLT) of the winding layer
    wire: Wire
    winding_type: WINDING_TYPE
    polarity: POLARITY  # Polarity of the layer (+1 or -1)

    def porosity(self) -> float:
        """
        Calcule le facteur de porosité (eta) de la couche.
        C'est le ratio entre la largeur totale de cuivre et la largeur de la fenêtre (bw).
        """
        if self.bw <= 0.0:
            return 1.0  # Sécurité pour éviter une division par zéro

        match self.wire.wire_type:
            case WIRE_TYPE.ROUND:
                cu_width = self.number_of_turns * self.effective_height()

            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                # Pour un feuillard ou fil rectangulaire, on utilise sa largeur (width)
                cu_width = self.number_of_turns * self.wire.width

            case WIRE_TYPE.LITZ:
                # Modèle de Dowell 1D : on étale virtuellement tous les brins face au champ
                cu_width = (
                    self.number_of_turns
                    * self.effective_height()
                    * sqrt(self.wire.number_of_strands)
                )

            case _:
                cu_width = 0.0

        # La porosité ne peut théoriquement pas dépasser 1.0 (100% de remplissage)
        return min(cu_width / self.bw, 1.0)

    def delta(self, freq: float) -> float:
        """Calcule le ratio d'épaisseur normalisée de Dowell (Delta)"""
        h_eff = self.effective_height()
        return h_eff / skin_depth(freq, self.wire.sigma, self.porosity())

    def effective_height(self) -> float:
        """
        Calcule l'épaisseur effective de la couche (h_eff).
        Applique la transformation en carré équivalent pour les fils ronds et Litz.
        """
        match self.wire.wire_type:
            case WIRE_TYPE.ROUND | WIRE_TYPE.LITZ:
                return self.wire.diameter * sqrt(pi / 4.0)

            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                return self.wire.height

            case _:
                return 0.0

    def copper_area(self) -> float:
        """Surface de cuivre de la section du fil (en m²)"""
        match self.wire.wire_type:
            case WIRE_TYPE.ROUND:
                return pi * (self.wire.diameter / 2.0) ** 2

            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                if self.wire.width <= 0.0 or self.wire.height <= 0.0:
                    raise ValueError(
                        "Width and height must be positive for square or foil wire."
                    )
                return self.wire.width * self.wire.height

            case WIRE_TYPE.LITZ:
                # La surface totale est la somme de la surface de tous les brins
                return (
                    self.wire.number_of_strands * pi * (self.wire.diameter / 2.0) ** 2
                )

            case _:
                return 0.0

    def dc_resistance(self) -> float:
        """Vraie résistance DC de la couche entière (en Ohms)"""
        area = self.copper_area()

        # Sécurité pour éviter une division par zéro
        if area == 0.0 or self.wire.sigma == 0.0:
            return float("inf")

        # Longueur totale du fil = (Nombre de tours) x (Longueur moyenne d'un tour)
        total_length = self.number_of_turns * self.mlt

        # Formule universelle : R = L / (sigma * Surface)
        return total_length / (self.wire.sigma * area)

    def dc_resistance_dowell(self) -> float:
        """Résistance DC du conducteur virtuel (carré ou feuillard) vu par Dowell"""
        match self.wire.wire_type:
            case WIRE_TYPE.ROUND:
                # Le carré équivalent a pour côté h_eff
                dowell_area = self.effective_height() ** 2

            case WIRE_TYPE.LITZ:
                # Le Litz a N brins, donc N carrés équivalents de côté h_eff
                dowell_area = (
                    self.effective_height() ** 2
                ) * self.wire.number_of_strands

            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                # Pour un feuillard, l'aire reste la largeur * la hauteur
                dowell_area = self.wire.width * self.wire.height

            case _:
                dowell_area = 0.0

        if dowell_area == 0.0 or self.wire.sigma == 0.0:
            return float("inf")

        return (self.mlt * self.number_of_turns) / (self.wire.sigma * dowell_area)

    def dc_true_dc_dowell_ratio(self) -> float:
        """Ratio entre la vraie résistance DC et la résistance DC de Dowell"""
        true_dc = self.dc_resistance()
        dowell_dc = self.dc_resistance_dowell()

        # Sécurité pour éviter une division par zéro
        if dowell_dc == 0.0:
            return float("inf")

        return true_dc / dowell_dc

    def dowell_area_ratio(self) -> float:
        """Doit valoir 1.0 : coherence entre la porosite et la vraie
        section de cuivre. Toute derive signale une erreur de definition."""
        implied_area = self.bw * self.effective_height() * self.porosity()
        true_area = self.number_of_turns * self.copper_area()

        if true_area == 0.0:
            return 1.0

        return implied_area / true_area


class Dowell_Winding_Structure(BaseModel):
    list_of_layers: list[Layer] = Field(default_factory=list)
    outer_mmf_fraction: float = 0.0
    """Fraction de la MMF totale absorbée par le chemin de retour EXTÉRIEUR.

    0.0 : toute la MMF côté noyau intérieur (entrefer jambe centrale,
          tore, noyau poudre) -> H = 0 sur la face externe. Cas normal.
    1.0 : toute la MMF côté extérieur -> H = 0 sur la face interne.
    0.5 : entrefer réparti également (E-core entrefer é sur 3 jambes).

    Sans effet si Sigma n.I = 0 : les deux faces sont alors nulles.
    """

    def add_layer(self, layer: Layer):
        """Les couches DOIVENT etre ajoutees de l'interieur vers l'exterieur
        (celle contre le noyau en premier). field_profile() et
        outer_mmf_fraction dependent de cet ordre."""
        self.list_of_layers.append(layer)

    def field_profile(self, currents: dict[WINDING_TYPE, float]) -> list[float]:
        """Escalier de MMF signe. Renvoie n+1 valeurs : H[i] = face interne
        de la couche i, H[i+1] = sa face externe.

        Les marches viennent d'Ampere ; le calage vertical vient de
        outer_mmf_fraction (condition aux limites imposee par le noyau).
        """
        boundaries = [0.0]
        ampere_turns = 0.0
        for layer in self.list_of_layers:
            ampere_turns += (
                layer.number_of_turns
                * currents.get(layer.winding_type, 0.0)
                * layer.polarity.value
            )
            boundaries.append(ampere_turns / layer.bw)
        shift = (1.0 - self.outer_mmf_fraction) * boundaries[-1]
        return [h - shift for h in boundaries]

    def _dowell_G(self, delta: float) -> tuple[float, float]:
        if delta < 1e-3:
            return 1.0, 0.5
        if delta > 50.0:  # asymptotic, avoids overflow
            return delta, 0.0
        den = np.cosh(2 * delta) - np.cos(2 * delta)
        g1 = delta * (np.sinh(2 * delta) + np.sin(2 * delta)) / den
        g2 = (
            delta
            * (np.sinh(delta) * np.cos(delta) + np.cosh(delta) * np.sin(delta))
            / den
        )
        return g1, g2

    def net_ampere_turns(self, currents: dict[WINDING_TYPE, float]) -> float:
        """Somme signee des ampere-tours. Doit valoir ~0 pour un vrai
        transformateur, non nul pour une inductance couplee (flyback)."""
        return sum(
            layer.number_of_turns
            * currents.get(layer.winding_type, 0.0)
            * layer.polarity.value
            for layer in self.list_of_layers
        )

    def layer_losses(self, freq, currents) -> list[float]:
        """Per-layer loss in W. Same physics as loss_at_frequency."""
        H = self.field_profile(currents)

        losses = []
        for i, layer in enumerate(self.list_of_layers):
            assert abs(layer.dowell_area_ratio() - 1.0) < 1e-9, layer.name
            H_in, H_out = H[i], H[i + 1]
            h_eff, eta = layer.effective_height(), layer.porosity()
            if (H_in == 0.0 and H_out == 0.0) or h_eff == 0.0 or eta == 0.0:
                losses.append(0.0)
                continue
            g1, g2 = self._dowell_G(layer.delta(freq))
            bracket = (H_out**2 + H_in**2) * g1 - 4.0 * H_out * H_in * g2
            losses.append(
                layer.bw * layer.mlt * bracket / (h_eff * eta * layer.wire.sigma)
            )
        return losses

    def loss_at_frequency(self, freq, currents) -> float:
        return sum(self.layer_losses(freq, currents))

    def dc_loss(self, currents_rms: dict[WINDING_TYPE, float]) -> float:
        return sum(
            layer.dc_resistance() * (currents_rms.get(layer.winding_type, 0.0) ** 2)
            for layer in self.list_of_layers
        )


if __name__ == "__main__":
    freq_sw = 100e3
    bw_total = pi * 7.62e-3
    mlt = 17.5e-3

    phases = {
        "ON  (primaire conduit)": {
            WINDING_TYPE.PRIMARY: 0.3,
            WINDING_TYPE.SECONDARY: 0.0,
        },
        "OFF (secondaire conduit)": {
            WINDING_TYPE.PRIMARY: 0.0,
            WINDING_TYPE.SECONDARY: 0.5,
        },
    }

    flyback_transfo = Dowell_Winding_Structure()
    flyback_transfo.add_layer(
        Layer(
            name="Primaire",
            number_of_turns=67,
            bw=bw_total,
            mlt=mlt,
            wire=Wire(wire_type=WIRE_TYPE.ROUND, diameter=0.3e-3),
            winding_type=WINDING_TYPE.PRIMARY,
            polarity=POLARITY.POSITIVE,
        )
    )
    flyback_transfo.add_layer(
        Layer(
            name="Secondaire",
            number_of_turns=26,
            bw=bw_total,
            mlt=mlt,
            wire=Wire(wire_type=WIRE_TYPE.ROUND, diameter=0.4e-3),
            winding_type=WINDING_TYPE.SECONDARY,
            polarity=POLARITY.NEGATIVE,
        )
    )

    layers = flyback_transfo.list_of_layers

    # --- Géométrie ---
    print("=== GÉOMÉTRIE DES COUCHES ===")
    print(
        f"{'Couche':<12}{'d [mm]':>9}{'h_eff [µm]':>12}{'eta':>8}"
        f"{'delta_pk [µm]':>15}{'Delta':>8}"
    )
    for lay in layers:
        eta = lay.porosity()
        print(
            f"{lay.name:<12}{lay.wire.diameter * 1e3:>9.2f}"
            f"{lay.effective_height() * 1e6:>12.1f}{eta:>8.3f}"
            f"{skin_depth(freq_sw, lay.wire.sigma, eta) * 1e6:>15.1f}"
            f"{lay.delta(freq_sw):>8.3f}"
        )
    print(f"\nÉpaisseur de peau nue (eta=1) : {skin_depth(freq_sw) * 1e6:.1f} µm\n")

    # --- Bilan par phase ---
    p_dc_total = 0.0
    p_ac_total = 0.0
    print("=== PERTES PAR PHASE ET PAR COUCHE ===")
    for phase_name, currents in phases.items():
        losses = flyback_transfo.layer_losses(freq_sw, currents)
        H = flyback_transfo.field_profile(currents)
        print(f"\n--- Phase {phase_name} ---")
        print(f"  MMF aux frontières : {[f'{h:.0f}' for h in H]} A/m")
        for i, (lay, p_ac) in enumerate(zip(layers, losses)):
            i_rms = currents.get(lay.winding_type, 0.0)
            p_dc = lay.dc_resistance() * i_rms**2
            p_dc_total += p_dc
            p_ac_total += p_ac
            if i_rms > 0.0:
                fr = p_ac / p_dc if p_dc > 0 else 1.0
                note = f"Fr = {fr:6.3f}"
            else:
                note = "PASSIVE (proximité seule)" if p_ac > 0 else "PASSIVE (blindée)"
            print(
                f"  {lay.name:<12} P_dc = {p_dc * 1e3:6.2f} mW   "
                f"P_ac = {p_ac * 1e3:6.2f} mW   {note}"
            )

    # --- Synthèse ---
    fr_global = p_ac_total / p_dc_total if p_dc_total > 0 else 1.0
    print("\n=== SYNTHÈSE ===")
    print(f"  Pertes DC totales     : {p_dc_total * 1e3:6.2f} mW")
    print(f"  Pertes AC totales     : {p_ac_total * 1e3:6.2f} mW")
    print(f"  Facteur Dowell global : {fr_global:6.3f}")
    print(f"  Surcoût AC            : {(p_ac_total - p_dc_total) * 1e3:6.2f} mW")

    # --- Contrôles de cohérence ---
    p_dc_check = sum(flyback_transfo.dc_loss(c) for c in phases.values())
    assert abs(p_dc_check - p_dc_total) < 1e-12, "incohérence DC"
    p_zero_f = sum(flyback_transfo.loss_at_frequency(1e-6, c) for c in phases.values())
    assert abs(p_zero_f - p_dc_total) / p_dc_total < 1e-6, (
        f"Dowell ne converge pas vers le DC : {p_zero_f * 1e3:.2f} vs {p_dc_total * 1e3:.2f} mW"
    )
    print("\n  [OK] Dowell converge vers les pertes DC quand f -> 0")
