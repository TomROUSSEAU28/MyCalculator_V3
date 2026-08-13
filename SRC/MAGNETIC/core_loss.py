import sys
from enum import Enum
from math import pi, sqrt
from pathlib import Path
import numpy as np

from scipy.integrate import cumulative_trapezoid, simpson
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Doit preceder les imports 'SRC.*' : sans cela le script n'est importable
# qu'en module (python -m SRC.MAGNETIC.dowell), pas en direct.
sys.path.append(str(Path(__file__).resolve().parents[2]))

# ==============================================================================
# 0. UTILITAIRES DE VALIDATION (Vérifications de sécurité)
# ==============================================================================


def _validate_and_format_signals(time_array, *signal_arrays):
    """
    Vérifie que tous les tableaux ont la même taille, qu'ils sont assez grands,
    et que le temps est strictement croissant. Force la conversion en np.ndarray.
    """
    t = np.asarray(time_array, dtype=float)

    if len(t) < 2:
        raise ValueError(
            "Les tableaux doivent contenir au moins 2 points pour l'intégration numérique."
        )

    if not np.all(np.diff(t) > 0):
        raise ValueError("Le vecteur time_array doit être strictement croissant.")

    formatted_signals = []
    for sig in signal_arrays:
        s = np.asarray(sig, dtype=float)
        if s.shape != t.shape:
            raise ValueError(
                f"Erreur de dimension : time_array a {len(t)} éléments, mais un signal a {len(s)} éléments."
            )
        formatted_signals.append(s)

    return (t, *formatted_signals)


def _validate_physics(N, Ae_mm2):
    """Vérifie la cohérence des paramètres géométriques du noyau."""
    if N <= 0:
        raise ValueError(
            f"Le nombre de spires N doit être strictement positif. Reçu : {N}"
        )
    if Ae_mm2 <= 0:
        raise ValueError(
            f"La section effective Ae_mm2 doit être strictement positive. Reçu : {Ae_mm2}"
        )


# ==============================================================================
# 1. PRÉ-TRAITEMENT DES SIGNAUX (TENSION MAGNÉTISANTE)
# ==============================================================================


def compute_v_lm(time_array, v_in_array, i_pri_array, Lk):
    """
    Isole la tension purement magnétisante v_Lm(t) pour un transformateur (ex: DAB).
    Soustrait la chute de tension due à l'inductance de fuite Lk.

    Paramètres :
    - time_array : array numpy des instants (en s)
    - v_in_array : array numpy de la tension totale appliquée au primaire (en V)
    - i_pri_array : array numpy du courant total mesuré au primaire (en A)
    - Lk : Inductance de fuite ramenée au primaire (en H)

    Retourne :
    - v_Lm_array : array numpy de la tension aux bornes de l'inductance magnétisante
    """
    t, v_in, i_pri = _validate_and_format_signals(time_array, v_in_array, i_pri_array)

    # Calcul de la dérivée du courant primaire : di/dt
    di_dt_array = np.gradient(i_pri, t)

    # Loi des mailles : v_Lm = v_in - v_Lk
    v_Lm_array = v_in - (Lk * di_dt_array)
    return v_Lm_array


# ==============================================================================
# 2. CALCUL DU FLUX MAGNÉTIQUE (B)
# ==============================================================================


def compute_b_from_current(I, Lm, N, Ae_mm2):
    """
    Calcule l'induction magnétique B à partir du courant.
    Valide pour un point scalaire I ou un array temporel I(t).
    Utilise la relation : B = (Lm * I) / (N * Ae)

    ATTENTION : Pour un transformateur, I doit être le courant magnétisant (I_mag),
                pas le courant total du primaire.

    Paramètres :
    - I : Courant magnétisant (scalaire ou array numpy, en A)
    - Lm : Inductance magnétisante (en H)
    - N : Nombre de spires
    - Ae_mm2 : Section effective du noyau (en mm²)

    Retourne :
    - B : Induction magnétique (en Tesla)
    """
    _validate_physics(N, Ae_mm2)
    Ae_m2 = Ae_mm2 * 1e-6
    return (Lm * I) / (N * Ae_m2)


def compute_flux_density(time_array, v_Lm_array, N, Ae_mm2, B_init=0.0):
    """
    Calcule l'évolution de la densité de flux magnétique B(t) par intégration temporelle.
    Universel : fonctionne pour inductances, Flyback, DAB, etc.
    Utilise la relation : B(t) = B(0) + (1 / N*Ae) * integral(v_Lm(t) dt)

    Paramètres :
    - time_array : array numpy des instants de simulation (en s)
    - v_Lm_array : array numpy de la tension magnétisante (en V).
                   (Générée par compute_v_lm pour un transfo, ou v_in direct pour inductance)
    - N : Nombre de spires
    - Ae_mm2 : Section du noyau en mm² (sera convertie en m² en interne)
    - B_init : Induction initiale à t=0 en Tesla (défaut: 0.0)

    Retourne :
    - B_array : array numpy de l'induction B(t) en Tesla
    """
    t, v_Lm = _validate_and_format_signals(time_array, v_Lm_array)
    _validate_physics(N, Ae_mm2)

    Ae_m2 = Ae_mm2 * 1e-6
    flux_linkage = cumulative_trapezoid(v_Lm, t, initial=0)
    return B_init + (1 / (N * Ae_m2)) * flux_linkage


# ==============================================================================
# 3. EXTRACTION DES AMPLITUDES
# ==============================================================================


def get_b_peak_to_peak(B_array):
    """
    Retourne l'excursion totale de flux (Delta B) sur le cycle.
    Correspond à la différence entre le maximum et le minimum absolus.
    """
    b = np.asarray(B_array)
    if len(b) == 0:
        raise ValueError("Le tableau B_array est vide.")
    return np.max(b) - np.min(b)


def get_b_amplitude_steinmetz(B_array):
    """
    Retourne l'amplitude crête (Bac) typiquement utilisée pour l'équation de Steinmetz.
    Correspond à la moitié de l'excursion totale : (Delta B) / 2.
    """
    return get_b_peak_to_peak(B_array) / 2.0


# ==============================================================================
# 4. CALCUL DES PERTES FER (CORE LOSSES)
# ==============================================================================


def calculate_classic_steinmetz_losses(f, B_amplitude, k, alpha, beta):
    """
    Calcule les pertes volumiques avec l'équation de Steinmetz classique (SSE).
    Idéal pour des signaux purement sinusoïdaux.

    Paramètres :
    - f : Fréquence de découpage (Hz)
    - B_amplitude : Amplitude crête du flux (Tesla), issue de get_b_amplitude_steinmetz()
    - k, alpha, beta : Coefficients empiriques fournis dans la datasheet de la ferrite

    Retourne :
    - Pertes volumiques (unités dépendantes de 'k', souvent mW/cm³ ou kW/m³)
    """
    if f <= 0:
        raise ValueError(f"La fréquence doit être strictement positive. Reçue : {f}")
    if B_amplitude < 0:
        raise ValueError(
            f"L'amplitude de flux ne peut pas être négative. Reçue : {B_amplitude}"
        )

    return k * (f**alpha) * (B_amplitude**beta)


def calculate_igse_losses(time_array, v_Lm_array, B_array, N, Ae_mm2, k, alpha, beta):
    """
    Calcule les pertes fer selon l'iGSE (Improved Generalized Steinmetz Equation).
    Indispensable pour estimer les pertes sous des signaux non sinusoïdaux
    (carrés, triangulaires, avec temps morts).

    Paramètres :
    - time_array : array numpy des instants (en s)
    - v_Lm_array : array numpy de la tension aux bornes de l'inductance magnétisante (en V)
                   ATTENTION : pour un transfo, utiliser compute_v_lm(v_in, i_pri)
    - B_array : array numpy d'induction magnétique B(t) (en Tesla)
    - N : Nombre de spires
    - Ae_mm2 : Section effective du noyau (en mm²)
    - k, alpha, beta : Paramètres de Steinmetz du matériau (datasheet)

    Retourne :
    - Puissance volumique moyenne dissipée sur la période (unités selon 'k')
    """
    t, v_Lm, b = _validate_and_format_signals(time_array, v_Lm_array, B_array)
    _validate_physics(N, Ae_mm2)

    Ae_m2 = Ae_mm2 * 1e-6
    T = t[-1] - t[0]

    delta_B = get_b_peak_to_peak(b)

    # Sécurité : Si l'excursion est nulle, les pertes sont nulles
    # (Évite les divisions par zéro ou puissances problématiques plus bas)
    if delta_B == 0:
        return 0.0

    # Calcul instantané de |dB/dt| via la loi de Faraday (évite le bruit de np.gradient)
    abs_dBdt_array = np.abs(v_Lm / (N * Ae_m2))

    # Conversion du coefficient 'k' classique en 'ki' pour l'iGSE
    theta = np.linspace(0, 2 * np.pi, 1000)
    integral_cos = np.trapezoid(np.abs(np.cos(theta)) ** alpha, theta)

    if integral_cos == 0:
        raise ValueError(
            "Erreur mathématique : l'intégrale de conversion iGSE est nulle (vérifiez la valeur de alpha)."
        )

    ki = k / (((2 * np.pi) ** (alpha - 1)) * integral_cos * (2 ** (beta - alpha)))

    # Construction du signal instantané de pertes à intégrer :
    # ki * |dB/dt|^alpha * (Delta B)^(beta - alpha)
    integrand = ki * (abs_dBdt_array**alpha) * (delta_B ** (beta - alpha))

    # Intégration sur la période T pour avoir la puissance moyenne (méthode de Simpson)
    power_loss_density = (1 / T) * simpson(y=integrand, x=t)

    return power_loss_density
