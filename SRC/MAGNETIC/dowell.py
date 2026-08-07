import logging
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
from SRC.OTHERS.plot import *
from SRC.OTHERS.terminal import *
from SRC.SIGNAL_PROCESSING.signal import *
from SRC.SIGNAL_PROCESSING.signal_plot import *

logger = logging.getLogger(__name__)


def skin_depth(
    freq: float, sigma: float = SIGMA_CU_20C, porosity: float = 1.0
) -> float:
    if freq < 1e-3:
        return float("inf")
    omega = 2.0 * pi * freq
    return sqrt(2.0 / (omega * MU0 * sigma * porosity))


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,  # objet immuable apres creation
        extra="forbid",  # rejette un champ inconnu au lieu de l'ignorer
    )

    @model_validator(mode="after")
    def _check(self) -> "Wire":
        t = self.wire_type
        if self.sigma <= 0.0:
            raise ValueError("sigma doit etre > 0")

        match t:
            case WIRE_TYPE.ROUND | WIRE_TYPE.LITZ:
                if self.diameter <= 0.0:
                    raise ValueError(f"{t.value} : 'diameter' est obligatoire et > 0")
                if self.width or self.height:
                    raise ValueError(
                        f"{t.value} : 'width'/'height' n'ont pas de sens pour un fil rond ou Litz, utiliser 'diameter'"
                    )

            case WIRE_TYPE.SQUARE | WIRE_TYPE.FOIL:
                if self.width <= 0.0 or self.height <= 0.0:
                    raise ValueError(
                        f"{t.value} : 'width' et 'height' sont obligatoires et > 0"
                    )
                if self.diameter:
                    raise ValueError(
                        f"{t.value} : 'diameter' n'a pas de sens pour un fil rectangulaire ou feuillard, utiliser 'width' et 'height'"
                    )

        if t is WIRE_TYPE.LITZ:
            if self.number_of_strands < 2:
                raise ValueError(
                    "LITZ : 'number_of_strands' doit valoir au moins 2 sinon utiliser ROUND"
                )
        elif self.number_of_strands != 1:
            raise ValueError(
                f"{t.value} : 'number_of_strands' ne s'applique qu'au LITZ"
            )

        return self


class Layer(BaseModel):
    name: str = ""
    number_of_turns: float  # Number of turns in the winding layer
    bw: float  # Bandwidth of the winding layer (in meters)
    mlt: float  # Mean Length Turn (MLT) of the winding layer
    wire: Wire
    winding_type: WINDING_TYPE
    polarity: POLARITY  # Polarity of the layer (+1 or -1)
    current_divider: float = 1.0  # k pour une sous-couche Litz, 1 sinon

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,  # objet immuable apres creation
        extra="forbid",  # rejette un champ inconnu au lieu de l'ignorer
    )

    @model_validator(mode="after")
    def _check(self) -> "Layer":
        for f in ("number_of_turns", "bw", "mlt", "current_divider"):
            if getattr(self, f) <= 0.0:
                raise ValueError(f"'{f}' doit etre > 0 (couche '{self.name}')")

        w = self.wire
        strands = sqrt(w.number_of_strands)
        if w.wire_type in (WIRE_TYPE.ROUND, WIRE_TYPE.LITZ):
            cu_phys = self.number_of_turns * w.diameter * strands
            cu_model = self.number_of_turns * self.effective_height() * strands
        else:
            cu_phys = cu_model = self.number_of_turns * w.width

        if cu_model > self.bw:
            logger.warning(
                "couche '%s' : porosite plafonnee (%.1f mm de cuivre equivalent "
                "pour bw = %.1f mm) -> resultat faux",
                self.name,
                cu_model * 1e3,
                self.bw * 1e3,
            )
        elif cu_phys > self.bw and self.current_divider == 1.0:
            logger.warning(
                "couche '%s' : %.1f mm de fil pour bw = %.1f mm -> ne tient pas "
                "sur une seule couche physique (le calcul reste valide)",
                self.name,
                cu_phys * 1e3,
                self.bw * 1e3,
            )
        return self

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

    def dowell_area_ratio(self) -> float:
        """Doit valoir 1.0 : coherence entre la porosite et la vraie
        section de cuivre. Toute derive signale une erreur de definition."""
        implied_area = self.bw * self.effective_height() * self.porosity()
        true_area = self.number_of_turns * self.copper_area()

        if true_area == 0.0:
            return 1.0

        return implied_area / true_area

    def expand(self) -> list["Layer"]:
        """Litz -> sqrt(k) sous-couches de brins (Geng eq. 3).
        Les autres types se renvoient eux-memes."""
        if self.wire.wire_type != WIRE_TYPE.LITZ:
            return [self]

        k = self.wire.number_of_strands
        n_sub = max(1, round(sqrt(k)))
        strands_per_sub = k / n_sub
        strand = Wire(
            wire_type=WIRE_TYPE.ROUND,
            diameter=self.wire.diameter,
            sigma=self.wire.sigma,
        )
        return [
            Layer(
                name=f"{self.name}[{j + 1}/{n_sub}]",
                number_of_turns=self.number_of_turns * strands_per_sub,
                bw=self.bw,
                mlt=self.mlt,
                wire=strand,
                winding_type=self.winding_type,
                polarity=self.polarity,
                current_divider=k,
            )
            for j in range(n_sub)
        ]


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

    def effective_layers(self) -> list[Layer]:
        return [sub for lay in self.list_of_layers for sub in lay.expand()]

    def field_profile(self, currents) -> list[float]:
        boundaries = [0.0]
        ampere_turns = 0.0
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
        for i, layer in enumerate(self.effective_layers()):
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

    def losses_by_layer(self, freq, currents) -> list[float]:
        """Pertes regroupees sur les couches d'origine (pas les sous-couches)."""
        flat = self.layer_losses(freq, currents)
        out, i = [], 0
        for lay in self.list_of_layers:
            n = len(lay.expand())
            out.append(sum(flat[i : i + n]))
            i += n
        return out

    def loss_at_frequency(self, freq, currents) -> float:
        return sum(self.layer_losses(freq, currents))

    def dc_loss(self, currents_rms: dict[WINDING_TYPE, float]) -> float:
        return sum(
            layer.dc_resistance() * (currents_rms.get(layer.winding_type, 0.0) ** 2)
            for layer in self.list_of_layers
        )

    def dc_resistances(self) -> dict[WINDING_TYPE, float]:
        """
        Calcule la résistance DC totale propre à chaque enroulement.
        Cette valeur est une constante purement géométrique.
        """
        resistances = {}
        for lay in self.list_of_layers:
            w_type = lay.winding_type
            # On additionne la résistance de la couche à son enroulement
            resistances[w_type] = resistances.get(w_type, 0.0) + lay.dc_resistance()

        return resistances

    def ac_resistances(self, freq: float) -> dict[WINDING_TYPE, float]:
        """
        Calcule la résistance AC propre à chaque enroulement à une fréquence donnée.

        Méthodologie : On injecte 1A RMS dans l'enroulement à tester (et 0A dans
        les autres). La perte active (en Watts) dans les couches de cet enroulement
        devient alors mathématiquement égale à sa résistance AC (en Ohms).
        """
        resistances_ac = {}
        # On identifie tous les types d'enroulements présents dans le transfo
        windings = {lay.winding_type for lay in self.list_of_layers}

        for w_type in windings:
            # 1. On crée un scénario de test avec 1 Ampère dans cet enroulement
            test_current = {w_type: 1.0}

            # 2. On calcule les pertes de toutes les couches avec ce courant
            losses = self.losses_by_layer(freq, test_current)

            # 3. On additionne les pertes uniquement pour les couches qui
            # appartiennent à l'enroulement que l'on est en train de tester.
            r_ac = 0.0
            for lay, p_loss in zip(self.list_of_layers, losses):
                if lay.winding_type == w_type:
                    r_ac += p_loss

            resistances_ac[w_type] = r_ac

        return resistances_ac

    def harmonic_losses(
        self,
        signals: dict[WINDING_TYPE, "ElectronicPeriodicSignal"],
        n_max: int = 40,
        threshold: float = 0.0,
    ) -> dict[str, float]:
        """
        Calcule les pertes totales par superposition harmonique.
        Exploite le fait que tous les enroulements d'un convertisseur
        partagent la même fréquence fondamentale (f_sw).
        """
        if not signals:
            return {"P_dc": 0.0, "P_ac": 0.0, "P_total": 0.0}

        # 1. On récupère la fréquence fondamentale (f_sw) partagée par tous
        first_signal = next(iter(signals.values()))
        f0 = first_signal.fundamental()

        # 2. On extrait le spectre complet de chaque signal
        spectra = {
            w_type: sig.harmonics_rms(n_max, threshold)
            for w_type, sig in signals.items()
        }

        total_dc_loss = 0.0
        total_ac_loss = 0.0

        # 3. Parcours harmonique par harmonique (de 0 à n_max)
        for rank in range(n_max + 1):
            # On extrait les courants RMS (en module) pour ce rang.
            currents_rank = {
                w_type: abs(spectrum.get(rank, 0j))
                for w_type, spectrum in spectra.items()
            }

            # S'il n'y a aucune énergie significative sur ce rang pour
            # aucun des enroulements, on économise du temps de calcul
            if all(i == 0.0 for i in currents_rank.values()):
                continue

            if rank == 0:
                # Rang 0 : Composante continue (0 Hz) -> Pertes Joules DC
                total_dc_loss += self.dc_loss(currents_rank)
            else:
                # Rang n : Effet de peau et de proximité AC
                freq = rank * f0
                total_ac_loss += self.loss_at_frequency(freq, currents_rank)

        return {
            "P_dc": total_dc_loss,
            "P_ac": total_ac_loss,
            "P_total": total_dc_loss + total_ac_loss,
        }


if __name__ == "__main__":
    OUTPUT = Path(__file__).parent / "OUTPUT"

    # One name for both palettes: SRC/OTHERS/plot.py and SRC/OTHERS/terminal.py
    # carry the same theme names on purpose, so the report and the figures match.
    THEME = "light"

    freq_sw = 100e3
    bw_total = pi * 7.62e-3
    mlt = 17.5e-3

    # --- Le signal reel : flyback 100 kHz --------------------------------
    T = 10.0e-6  # periode
    t_on = 6.30e-6  # primaire conduit   : rampe 0 -> 500 mA
    t_dem = 1.98e-6  # secondaire conduit : rampe 2.52 A -> 0
    t_off = T - t_on - t_dem  # 1.72 us : les deux a zero

    i_pri = ElectronicPeriodicSignal.from_breakpoints(
        "I_primaire",
        times=[0.0, t_on, t_on, T],
        values=[0.0, 0.5, 0.0, 0.0],
        n_samples=8192,
        period=T,
    )
    i_sec = ElectronicPeriodicSignal.from_breakpoints(
        "I_secondaire",
        times=[0.0, t_on, t_on, t_on + t_dem, T],
        values=[0.0, 0.0, 2.52, 0.0, 0.0],
        n_samples=8192,
        period=T,
    )
    signals = {WINDING_TYPE.PRIMARY: i_pri, WINDING_TYPE.SECONDARY: i_sec}

    fig = plot_time_domain(
        [i_pri, i_sec], title="Courants du flyback", unit="A", theme=THEME
    )

    fig2 = plot_spectrum(
        [i_pri, i_sec], title="Spectre du flyback", unit="A", theme=THEME
    )
    save_figure(fig, OUTPUT / "dowell_flyback_currents.png", dpi=300)
    save_figure(fig2, OUTPUT / "dowell_flyback_currents_spectrum.pdf", dpi=300)

    dataframe(signal_table(i_pri))
    dataframe(signal_table(i_sec))

    # Chaque enroulement ne conduit que pendant sa phase : sa RMS sur la
    # periode entiere est donc deja la contribution ponderee par le rapport
    # cyclique. Les pertes des deux phases s'additionnent directement.
    phases = {
        "ON  (primaire conduit)": {
            WINDING_TYPE.PRIMARY: i_pri.rms(),
            WINDING_TYPE.SECONDARY: 0.0,
        },
        "OFF (secondaire conduit)": {
            WINDING_TYPE.PRIMARY: 0.0,
            WINDING_TYPE.SECONDARY: i_sec.rms(),
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
            mlt=mlt + 8 * 0.3e-3,  # bobine par-dessus le primaire
            wire=Wire(wire_type=WIRE_TYPE.ROUND, diameter=0.4e-3),
            winding_type=WINDING_TYPE.SECONDARY,
            polarity=POLARITY.NEGATIVE,
        )
    )

    layers = flyback_transfo.list_of_layers

    # --- Géométrie (sur les sous-couches vues par Dowell) ---
    print("=== GÉOMÉTRIE DES COUCHES ===")
    print(
        f"{'Couche':<16}{'d [mm]':>9}{'h_eff [µm]':>12}{'eta':>8}"
        f"{'delta_pk [µm]':>15}{'Delta':>8}"
    )
    for lay in flyback_transfo.effective_layers():
        eta = lay.porosity()
        print(
            f"{lay.name:<16}{lay.wire.diameter * 1e3:>9.2f}"
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
        losses = flyback_transfo.losses_by_layer(freq_sw, currents)
        assert len(losses) == len(layers)
        H = flyback_transfo.field_profile(currents)
        print(f"\n--- Phase {phase_name} ---")
        print(f"  MMF aux frontières : {[f'{h:.0f}' for h in H]} A/m")
        for lay, p_ac in zip(layers, losses):
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

    # =====================================================================
    #  SANS DECOMPOSITION HARMONIQUE  vs  AVEC
    # =====================================================================
    # Methode 1 (ci-dessus) : une seule frequence, f_sw, alimentee par la
    #   RMS de chaque enroulement -> p_ac_total.
    # Methode 2 : le meme signal eclate en son spectre, chaque rang evalue
    #   a sa propre frequence puis somme.
    N_MAX = 200  # rang harmonique max (20 MHz a f0 = 100 kHz)
    f0 = i_pri.fundamental()
    harm = flyback_transfo.harmonic_losses(signals, n_max=N_MAX)

    print("\n" + "=" * 72)
    print("PERTES SANS vs AVEC DECOMPOSITION HARMONIQUE")
    print("=" * 72)
    print(
        f"  T = {T * 1e6:.2f} us   t_on = {t_on * 1e6:.2f} us   "
        f"t_dem = {t_dem * 1e6:.2f} us   t_off = {t_off * 1e6:.2f} us   "
        f"f0 = {f0 / 1e3:.1f} kHz\n"
    )
    for sig in (i_pri, i_sec):
        nfo = sig.info()
        print(
            f"  {nfo.name:<14} I_rms = {nfo.rms * 1e3:7.1f} mA   "
            f"I_pk = {nfo.peak * 1e3:7.1f} mA   "
            f"H1_rms = {nfo.fundamental_rms * 1e3:6.1f} mA   "
            f"THD = {nfo.thd * 100:5.1f} %"
        )

    # --- Ou passe l'energie, rang par rang -------------------------------
    spectra = {w: s.harmonics_rms(N_MAX) for w, s in signals.items()}
    print(
        f"\n  {'rang':>5}{'f [kHz]':>10}{'I_pri [mA]':>13}{'I_sec [mA]':>13}"
        f"{'P [mW]':>11}{'cumul [%]':>11}"
    )
    cumul = 0.0
    for rank in range(N_MAX + 1):
        currents = {w: abs(sp.get(rank, 0j)) for w, sp in spectra.items()}
        if all(i == 0.0 for i in currents.values()):
            continue
        cumul += (
            flyback_transfo.dc_loss(currents)
            if rank == 0
            else flyback_transfo.loss_at_frequency(rank * f0, currents)
        )
        if rank > 10 and rank % 20:  # au-dela du rang 10, un point sur 20
            continue
        print(
            f"  {rank:>5}{rank * f0 / 1e3:>10.0f}"
            f"{currents[WINDING_TYPE.PRIMARY] * 1e3:>13.1f}"
            f"{currents[WINDING_TYPE.SECONDARY] * 1e3:>13.1f}"
            f"{cumul * 1e3:>11.2f}{100 * cumul / harm['P_total']:>11.1f}"
        )

    # --- Verdict ----------------------------------------------------------
    print("\n  --- COMPARAISON ---")
    print(f"  Sans harmoniques (f_sw seule) : {p_ac_total * 1e3:6.2f} mW")
    print(f"  Avec harmoniques (n <= {N_MAX})   : {harm['P_total'] * 1e3:6.2f} mW")
    print(
        f"  Surcout des harmoniques       : "
        f"{(harm['P_total'] - p_ac_total) * 1e3:+6.2f} mW  "
        f"({100 * (harm['P_total'] / p_ac_total - 1.0):+.1f} %)"
    )

    # --- Convergence : jusqu'ou faut-il monter en rang ? -------------------
    # Fronts ideaux => spectre en 1/n et pertes par rang en n^-1.5 : la somme
    # converge, mais lentement. Le vrai signal a un di/dt fini qui coupe la
    # queue du spectre ; sans cette information, la somme reste une borne
    # inferieure. L'erreur de Parseval dit ce que la troncature a laisse.
    print("\n  --- CONVERGENCE ---")
    print(f"  {'n_max':>7}{'P_total [mW]':>15}{'ecart/prec [%]':>16}{'Parseval':>12}")
    prev = None
    for n in (20, 40, 60, 100, 200, 400):
        p_n = flyback_transfo.harmonic_losses(signals, n_max=n)["P_total"]
        drift = f"{100 * (p_n / prev - 1.0):>15.1f}" if prev else f"{'-':>15}"
        print(f"  {n:>7}{p_n * 1e3:>15.2f}{drift} {i_sec.parseval_error(n):>11.2e}")
        prev = p_n

    # NB : harmonic_losses() prend le module des harmoniques, donc la phase
    # relative primaire/secondaire est perdue. Sur un flyback les deux ne
    # conduisent jamais ensemble, mais rang par rang ils apparaissent en
    # phase et leurs MMF (polarites opposees) se compensent partiellement :
    # les pertes de proximite sont donc sous-estimees.
