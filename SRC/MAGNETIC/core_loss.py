import sys
from enum import Enum
from math import pi, sqrt
from pathlib import Path
import numpy as np

from scipy.integrate import cumulative_trapezoid

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Doit preceder les imports 'SRC.*' : sans cela le script n'est importable
# qu'en module (python -m SRC.MAGNETIC.dowell), pas en direct.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from SRC.constant import MU0, SIGMA_CU_20C


# il faut une fonction qui prend le signal courant en electronic periodic signal et qui le transforme en flux B,
# ensuite une classe de material, core geometrie
# ensuite une fonction IGSE

# v (signal) → flux_density(v, N, core) → B (signal aussi)
# B → core_loss(B, core, T) → CORE_LOSS_INFO


def compute_b_init_from_current(I_init, L, N, Ae_mm2):
    """
    Calcule l'induction initiale B(0) à partir du courant initial.
    Utilise la relation : B = (L * I) / (N * Ae)

    Paramètres :
    - I_init : Courant initial (continu ou de vallée) dans la bobine (en A)
    - L : Inductance du composant (en H)
    - N : Nombre de spires
    - Ae_mm2 : Section du noyau en mm²

    Retourne :
    - B_init : Induction initiale (en Tesla)
    """
    # Conversion de la section en m²
    Ae_m2 = Ae_mm2 * 1e-6

    # Calcul de B(0)
    B_init = (L * I_init) / (N * Ae_m2)
    return B_init


def compute_flux_density(time_array, voltage_array, N, Ae_mm2, B_init=0.0):
    """
    Calcule l'évolution de la densité de flux magnétique B(t).

    Paramètres :
    - time_array : array numpy des instants de simulation (en s)
    - voltage_array : array numpy de la tension v(t) (en V)
    - N : Nombre de spires
    - Ae_mm2 : Section du noyau en mm² (sera convertie en m² en interne)
    - B_init : Induction initiale à t=0 en Tesla (défaut: 0.0)

    Retourne :
    - B_array : array numpy de l'induction B(t) en Tesla
    """

    # 1. Conversion de la section en m² (1 mm² = 1e-6 m²)
    Ae_m2 = Ae_mm2 * 1e-6

    # 2. Intégration numérique de la tension : flux_linkage = intégrale(v(t) dt)
    # cumulative_trapezoid renvoie un tableau de la même taille grâce à initial=0
    flux_linkage = cumulative_trapezoid(voltage_array, time_array, initial=0)

    # 3. Application de la loi de Faraday
    B_array = B_init + (1 / (N * Ae_m2)) * flux_linkage

    return B_array


import numpy as np
from scipy.integrate import (
    simpson,
)  # simpson est meilleur que trapezoid pour les puissances


def calculate_igse_losses(
    time_array, voltage_array, B_array, N, Ae_mm2, k, alpha, beta
):
    """
    Calcule la densité volumique de pertes fer (mW/cm³ ou kW/m³) selon l'iGSE.
    Attention: k, alpha, beta doivent correspondre aux unités cibles (souvent données pour B en Tesla, f en Hz).
    """
    Ae_m2 = Ae_mm2 * 1e-6

    # 1. Calcul de la période T
    T = time_array[-1] - time_array[0]

    # 2. Excursion globale delta_B (pic-à-pic)
    delta_B = np.max(B_array) - np.min(B_array)

    # 3. Calcul instantané de |dB/dt| grâce à la tension
    abs_dBdt_array = np.abs(voltage_array / (N * Ae_m2))

    # 4. Calcul du coefficient k_i (conversion du k classique de Steinmetz)
    # Formule mathématique standard d'intégration pour passer de k à k_i
    # (Certains constructeurs donnent directement k_i, sinon on l'évalue numériquement)
    theta = np.linspace(0, 2 * np.pi, 1000)
    integral_cos = np.trapz(np.abs(np.cos(theta)) ** alpha, theta)
    ki = k / ((2 * np.pi) ** (alpha - 1) * integral_cos * 2 ** (beta - alpha))

    # 5. Construction du signal à intégrer
    # ki * |dB/dt|^alpha * (delta_B)^(beta - alpha)
    integrand = ki * (abs_dBdt_array**alpha) * (delta_B ** (beta - alpha))

    # 6. Intégration sur la période T pour avoir la puissance moyenne
    # Utilisation de la méthode de Simpson pour plus de précision numérique
    power_loss_density = (1 / T) * simpson(y=integrand, x=time_array)

    return power_loss_density
