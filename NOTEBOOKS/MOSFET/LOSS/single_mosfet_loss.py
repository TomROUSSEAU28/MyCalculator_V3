"""
Pertes d'un MOSFET seul — version « feuille de calcul » python.

Même modèle que le notebook `single_mosfet_loss.ipynb`, sans widget : tout ce qui
se réglait dans le panneau est ici une constante de la section CONFIGURATION.
On édite les valeurs en haut du fichier, on lance, on lit le rapport.

    python NOTEBOOKS/MOSFET/LOSS/single_mosfet_loss.py

Les figures partent dans OUTPUT/ (SAVE_PLOTS) et/ou s'ouvrent (SHOW_PLOTS).

Rappel du modèle (détaillé dans le notebook) :
    P_cond  = R_DS(on)(Tj) * I_rms²          R_DS(on)(Tj) = R_25 [1 + a_R (Tj - 25)]
    P_sw    = ½ (V_on I_on t_on + V_off I_off t_off) f_sw
    P_oss   = ½ C_oss,er(V_on) V_on² f_sw
    P_body  = V_F I_body D_body + Q_rr V_on f_sw
    P_g,int = Q_g dVgs f_sw * R_g,int / R_g,tot     (seule part qui chauffe la puce)
    Tj(k+1) = T_amb + R_th * P_total(Tj(k))
"""

from __future__ import annotations

import sys
from pathlib import Path

# Remonte jusqu'à la racine du projet (le dossier qui contient DATABASE/)
ROOT = Path(__file__).resolve().parent
while not (ROOT / "DATABASE").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================ #
#  CONFIGURATION — tout ce qui se réglait dans le panneau du notebook
# ============================================================================ #

# --- Composants --------------------------------------------------------------
MOSFET = "BSC016N06NS"
DRIVER = "UCC27714"

# --- Point de fonctionnement -------------------------------------------------
# Pas de bouton « hard / soft switching » : on saisit la tension réellement
# balayée sur chaque front.
#   commutation dure   -> V_TURN_ON = V_TURN_OFF = V_bus
#   ZVS à l'amorçage   -> V_TURN_ON = 0   (annule P_oss ET le recouvrement)
#   ZVS des deux côtés -> V_TURN_ON = V_TURN_OFF = 0
V_TURN_ON = 48.0  # V_ds balayée à l'amorçage [V]
V_TURN_OFF = 48.0  # V_ds balayée au blocage [V]
I_RMS = 15.8  # courant efficace de conduction [A]
I_ON = 25.0  # courant commuté à l'amorçage [A]
I_OFF = 25.0  # courant commuté au blocage [A]
F_SW_KHZ = 100.0  # fréquence de découpage [kHz]

# --- Boucle de grille --------------------------------------------------------
R_G_EXT_ON = 2.2  # résistance de grille externe, amorçage [ohm]
R_G_EXT_OFF = 1.0  # résistance de grille externe, blocage [ohm]

# --- Body diode / recouvrement -----------------------------------------------
I_BODY = 0.0  # courant de diode pendant le temps mort [A]
D_BODY_PCT = 0.0  # rapport cyclique de conduction diode [%]

# Charge que l'amorçage doit balayer dans la diode d'en face, saisie à la main.
#
# Pour un MOSFET seul il n'y a pas de « mode auto » possible : la diode qui se
# recouvre n'est PAS la sienne, c'est celle du composant d'en face (Schottky
# parallèle, diode de roue libre, MOSFET opposé...) — et ce modèle ne la connaît
# pas. On saisit donc directement la charge, relevée sur la courbe Q_rr de sa
# datasheet au di/dt réel du montage (voir section 4 du rapport), ou mesurée.
#
#   0 -> pas de recouvrement du tout (ZCS, ZVS, GaN sans body diode)
#
# `BODY_DIODE.q_rr_at_condition(di_dt, v_r, i_f)` transpose un point de test
# datasheet vers le di/dt réel, que la section 4 du rapport affiche.
Q_RR_NC = 0.0  # [nC]

# --- Thermique ---------------------------------------------------------------
T_J_EVAL = 100.0  # Tj d'évaluation du bilan de pertes [°C]
T_AMBIENT = 40.0  # température ambiante de l'itération thermique [°C]

# R_TH_SOURCE :
#   "dissip"    -> calculé face par face à partir de FACES (voir plus bas)
#   "datasheet" -> R_thJA nu de la datasheet
#   "manual"    -> valeur saisie dans R_TH_MANUAL
R_TH_SOURCE = "dissip"
R_TH_MANUAL = 20.0  # R_th jonction -> ambiante [°C/W], si R_TH_SOURCE == "manual"

# --- Refroidissement, un bloc par face du boîtier ----------------------------
# Chaque face active ouvre une branche R_thJC(face) + R_ext(face) ; les branches
# se somment en parallèle. Le R_thJA datasheet n'est alors plus utilisé (il
# suppose déjà un plan de cuivre : le garder compterait deux fois le même PCB).
#
# mode :  "none" (face non refroidie) | "pcb" (plan de cuivre) | "std" (radiateur)
#
# Substrats courants — (k [W/(m·K)], épaisseur typique [mm]) :
#   FR4 standard   (0.3,  1.6)      DBC Al2O3  (25.0,  0.38)
#   IMS base alu   (2.0,  0.1)      DBC AlN    (170.0, 0.63)
#
# Deux pièges :
#   - deux faces en "pcb" déclarant chacune une surface opposée comptent le même
#     cuivre deux fois : mettre A_cu_other_side_cm2 à 0 sur l'une des deux ;
#   - la face "top" est presque toujours mauvaise (R_thJC,top >> R_thJC,bottom),
#     elle n'apporte quelque chose qu'avec un vrai radiateur collé dessus.
FACES = {
    "bottom": {
        "mode": "pcb",
        "pcb": {
            "A_cu_total_side_cm2": 10.0,  # cuivre total de cette face [cm²]
            "A_cu_other_side_cm2": 10.0,  # cuivre face opposée [cm²] (0 = aucun)
            "A_pad_mosfet_cm2": 0.3,  # surface du pad thermique [cm²]
            "copper_thickness_um": 70.0,  # épaisseur cuivre [µm] (35 = 1 oz)
            "pcb_thickness_mm": 1.6,
            "pcb_k_W_mK": 0.3,
            "k_conductor_W_mK": 385.0,
            "n_vias": 24,
            "via_diameter_mm": 0.3,
            "via_plating_um": 25.0,
            "via_filled": False,
            "h_conv_eff_W_m2K": 25.0,  # convection + rayonnement [W/(m²·K)]
        },
        "std": {"R_th": 5.0, "R_tim": 0.5},  # radiateur, si mode == "std"
    },
    "top": {
        "mode": "none",
        "pcb": {
            "A_cu_total_side_cm2": 2.0,
            "A_cu_other_side_cm2": 0.0,
            "A_pad_mosfet_cm2": 0.3,
            "copper_thickness_um": 70.0,
            "pcb_thickness_mm": 1.6,
            "pcb_k_W_mK": 0.3,
            "k_conductor_W_mK": 385.0,
            "n_vias": 0,
            "via_diameter_mm": 0.3,
            "via_plating_um": 25.0,
            "via_filled": False,
            "h_conv_eff_W_m2K": 25.0,
        },
        "std": {"R_th": 5.0, "R_tim": 0.5},
    },
}

# --- Sorties -----------------------------------------------------------------
SHOW_PLOTS = False  # ouvre les figures dans une fenêtre
SAVE_PLOTS = True  # écrit les PNG dans OUTPUT/
PLOT_THEME = "light"  # "light" ou "dark"
OUTPUT_DIR = ROOT / "OUTPUT"


# ============================================================================ #
#  Imports du projet — après le sys.path, et après le choix du backend
# ============================================================================ #

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")  # pas de fenêtre : on écrit directement les fichiers

import matplotlib.pyplot as plt

from DATABASE.db_driver_mosfet import DRIVER_LIBRARY, load_driver
from DATABASE.db_mosfet import MOSFET_LIBRARY, load_mosfet
from SRC.MOSFET.mosfet_loss import (
    OPERATING_POINT,
    loss_single_mosfet_at_temp,
    loss_thermal_iteration,
)
from SRC.MOSFET.mosfet_plot import (
    loss_table,
    plot_loss_breakdown,
    plot_thermal_iteration,
)
from SRC.OTHERS.terminal import alert, kv, section, table
from SRC.THERMAL.dissipator import PCBDissipator, Placement, StandardDissipator


def fmt_r(value: float) -> str:
    """Résistance thermique, infini compris (via absent = chemin ouvert)."""
    return "inf" if value > 1e6 else f"{value:.1f}"


# Une alerte est un couple (niveau, message) : "stop" interrompt le calcul,
# "warn" l'accompagne. Les deux fabriques ci-dessous évitent d'écrire le niveau
# à la main à chaque fois.
def stop(message: str) -> tuple[str, str]:
    return "stop", message


def warn(message: str) -> tuple[str, str]:
    return "warn", message


# ============================================================================ #
#  Construction du point de fonctionnement et des dissipateurs
# ============================================================================ #


def build_operating_point() -> OPERATING_POINT:
    return OPERATING_POINT(
        v_turn_on=V_TURN_ON,
        v_turn_off=V_TURN_OFF,
        i_rms=I_RMS,
        f_sw=F_SW_KHZ * 1e3,
        i_on=I_ON,
        i_off=I_OFF,
        r_g_ext_on=R_G_EXT_ON,
        r_g_ext_off=R_G_EXT_OFF,
        i_body=I_BODY,
        d_body=D_BODY_PCT / 100.0,
        q_rr_opposite=Q_RR_NC * 1e-9,
    )


def build_dissipators() -> list:
    """Les dissipateurs des faces actives, dans l'ordre de FACES."""
    built = []
    for face, cfg in FACES.items():
        mode = cfg["mode"]
        if mode == "none":
            continue
        placement = Placement(face)
        if mode == "std":
            built.append(
                StandardDissipator(
                    name=f"Radiateur {face}", placement=placement, **cfg["std"]
                )
            )
        elif mode == "pcb":
            params = dict(cfg["pcb"])
            # 0 cm² côté opposé = pas de plan opposé du tout (chemin B fermé)
            params["A_cu_other_side_cm2"] = params.get("A_cu_other_side_cm2") or None
            built.append(
                PCBDissipator(name=f"Plan cuivre {face}", placement=placement, **params)
            )
        else:
            raise ValueError(f"FACES['{face}']['mode'] inconnu : {mode!r}")
    return built


def resolve_r_th(mosfet) -> tuple[float | None, list[tuple[str, str]], list]:
    """
    (r_th, alertes, dissipateurs) selon R_TH_SOURCE.

    r_th = None laisse loss_thermal_iteration prendre le R_thJA datasheet.
    """
    if R_TH_SOURCE == "manual":
        return R_TH_MANUAL, [], []
    if R_TH_SOURCE == "datasheet":
        return None, [], []
    if R_TH_SOURCE != "dissip":
        raise ValueError(f"R_TH_SOURCE inconnu : {R_TH_SOURCE!r}")

    try:
        active = build_dissipators()
    except ValueError as err:  # géométrie refusée par pydantic (ValidationError)
        detail = str(err).split("Value error, ")[-1].split(" [type=")[0].strip()
        return None, [stop(f"Géométrie de dissipateur invalide — {detail}")], []

    if not active:
        message = warn(
            "Aucune face refroidie : repli sur le R_thJA datasheet "
            f"({mosfet.thermal.r_thja[0]:.0f} °C/W). Activer au moins une face "
            "dans FACES."
        )
        return None, [message], []

    faces_dispo = [face for _, face in mosfet.thermal.r_thjc]
    missing = [
        d.placement.value for d in active if d.placement.value not in faces_dispo
    ]
    if missing:
        message = stop(
            f"{MOSFET} n'a pas de R_thJC pour la face « {missing[0]} » "
            f"(disponibles : {', '.join(faces_dispo)}). Compléter `r_thjc` dans "
            "MOSFET_LIBRARY."
        )
        return None, [message], []

    alerts: list[tuple[str, str]] = []
    pcbs = [d for d in active if isinstance(d, PCBDissipator)]

    # Le PCB n'a que deux plans : les déclarer des deux côtés compte le cuivre 2x.
    if len(pcbs) > 1 and all(p.A_cu_other_side_cm2 for p in pcbs):
        alerts.append(
            warn(
                "Les deux faces sont en « pcb » et déclarent chacune un plan opposé : "
                "le même cuivre est compté deux fois. Mettre A_cu_other_side_cm2 à 0 "
                "sur l'une des deux."
            )
        )
    # Cuivre au-delà du rayon d'étalement : payé en surface, inutile en thermique.
    for p in pcbs:
        if p.A_cu_effective_side_cm2 < p.A_cu_total_side_cm2 - 1e-9:
            alerts.append(
                warn(
                    f"{p.name} : seuls {p.A_cu_effective_side_cm2:.1f} cm² des "
                    f"{p.A_cu_total_side_cm2:.1f} cm² déclarés participent (rayon "
                    f"d'étalement à {p.copper_thickness_um:.0f} µm). Élargir le plan "
                    "n'aidera plus : cuivre plus épais, vias, ou flux d'air."
                )
            )

    paths = [(d.get_rth(), d.placement.value) for d in active]
    return mosfet.thermal.r_thja_value(paths), alerts, active


def check_operating_point(mosfet, op: OPERATING_POINT) -> list[tuple[str, str]]:
    """Cohérence du point de fonctionnement vis-à-vis des limites du composant."""
    alerts: list[tuple[str, str]] = []
    v_max_coss = mosfet.c_oss.vds_points[-1]
    v_sw = max(op.v_turn_on, op.v_turn_off)

    if v_sw > mosfet.v_dss_max:
        alerts.append(
            stop(
                f"Tension commutée {v_sw:.0f} V au-dessus du V_DSS max du "
                f"{mosfet.component_info.part_number} ({mosfet.v_dss_max:.0f} V) "
                "— claquage."
            )
        )
    if op.v_turn_on > v_max_coss:
        alerts.append(
            stop(
                f"La courbe C_oss de la datasheet s'arrête à {v_max_coss:.0f} V : "
                f"impossible de calculer P_oss à {op.v_turn_on:.0f} V sans extrapoler. "
                "Étendre `vds_points` / `coss_points` dans MOSFET_LIBRARY."
            )
        )
    for name, value in (
        ("I_rms", op.i_rms),
        ("I amorçage", op.i_commutated_on()),
        ("I blocage", op.i_commutated_off()),
    ):
        if value > mosfet.i_max:
            alerts.append(
                warn(
                    f"{name} = {value:.0f} A au-dessus du calibre "
                    f"({mosfet.i_max:.0f} A)."
                )
            )
    if T_J_EVAL > mosfet.thermal.t_j_max:
        alerts.append(
            warn(
                f"Tj d'évaluation {T_J_EVAL:.0f} °C au-dessus de T_j,max "
                f"({mosfet.thermal.t_j_max:.0f} °C)."
            )
        )
    # P_rr = Q_rr * V_turn_on * f_sw : sous ZVS il n'y a rien pour forcer un
    # recouvrement, la charge saisie ne coûte rien. Autant le dire.
    if Q_RR_NC > 0.0 and op.v_turn_on == 0.0:
        alerts.append(
            warn(
                f"Q_rr = {Q_RR_NC:.0f} nC saisi mais V_TURN_ON = 0 (ZVS) : P_rr = "
                "Q_rr · V_turn_on · f_sw est nul de toute façon. La valeur n'a aucun "
                "effet sur le bilan."
            )
        )
    return alerts


# ============================================================================ #
#  Rapport
# ============================================================================ #


def report_operating_point(mosfet) -> None:
    section(1, "Point de fonctionnement")
    kv("MOSFET", f"{MOSFET}  ({mosfet.component_info.package})")
    kv("Driver", DRIVER)
    kv("Tension commutée on / off", f"{V_TURN_ON:.0f} V / {V_TURN_OFF:.0f} V")
    kv("Courant I_rms / I_on / I_off", f"{I_RMS:.1f} A / {I_ON:.1f} A / {I_OFF:.1f} A")
    kv("Fréquence de découpage", f"{F_SW_KHZ:.0f} kHz")
    kv("R grille externe on / off", f"{R_G_EXT_ON:.2f} / {R_G_EXT_OFF:.2f} ohm")
    kv("Diode : I_body / D_body", f"{I_BODY:.1f} A / {D_BODY_PCT:.1f} %")
    kv(
        "Q_rr de la diode d'en face",
        f"{Q_RR_NC:.0f} nC"
        + ("  (aucun recouvrement)" if Q_RR_NC == 0 else " (saisi)"),
    )
    kv("Tj d'évaluation / T_ambiante", f"{T_J_EVAL:.0f} °C / {T_AMBIENT:.0f} °C")


def report_thermal_paths(mosfet, active: list, r_th: float) -> None:
    """Une ligne par branche, puis le détail interne des plans PCB."""
    section(2, "Chemins thermiques")
    rows = []
    for d in active:
        r_jc = mosfet.thermal.r_thjc_value(d.placement.value)
        r_ext = d.get_rth()
        rows.append((d.name, f"{r_jc:.2f}", fmt_r(r_ext), f"{r_jc + r_ext:.1f}"))
    table(["Branche", "R_thJC", "R_ext", "Total [°C/W]"], rows)
    print()
    kv("Branches en parallèle -> R_thJA", f"{r_th:.1f} °C/W")
    kv("(datasheet nu, non utilisé ici)", f"{mosfet.thermal.r_thja[0]:.0f} °C/W")

    detail = [
        ("r_spreading_max_cm", "rayon d'étalement max [cm]"),
        ("A_cu_effective_side_cm2", "cuivre utile côté pad [cm²]"),
        ("A_cu_exposed_side_cm2", "cuivre exposé à l'air [cm²]"),
        ("R_conv_side", "R convection côté pad (chemin A)"),
        ("R_pcb_thru", "R substrat nu sous le pad"),
        ("R_vias", "R vias thermiques"),
        ("R_through", "R traversée (substrat || vias)"),
        ("R_conv_other", "R convection face opposée"),
        ("R_path_B", "chemin B total"),
        ("R_total", "R_ext de la face (A || B)"),
    ]
    for d in active:
        if not isinstance(d, PCBDissipator):
            continue
        b = d.breakdown()
        print(f"\n  Détail — {d.name}   [cm, cm², °C/W]")
        table(
            ["Poste", "Valeur"],
            [(label, fmt_r(b[key])) for key, label in detail if key in b],
        )


def report_losses(res) -> None:
    section(3, "Bilan de pertes [W]")
    print(loss_table(res).to_string())
    print()
    kv("Total dissipé dans la puce", f"{res.p_total:.3f} W  (à Tj = {T_J_EVAL:.0f} °C)")
    kv("Hors boîtier — driver", f"{res.p_gate_drv * 1e3:.0f} mW")
    kv("Hors boîtier — R grille externe", f"{res.p_gate_ext * 1e3:.0f} mW")


def report_switching(mosfet, res) -> None:
    section(4, "Commutation")
    table(
        ["Sous-intervalle", "Durée [ns]", "Vitesse"],
        [
            (
                "t_ri (montée courant)",
                f"{res.t_ri * 1e9:.1f}",
                f"di/dt on  {res.di_dt_on / 1e9:.2f} A/ns",
            ),
            (
                "t_fv (descente tension)",
                f"{res.t_fv * 1e9:.1f}",
                f"dv/dt on  {res.dv_dt_on / 1e9:.2f} V/ns",
            ),
            (
                "t_rv (montée tension)",
                f"{res.t_rv * 1e9:.1f}",
                f"dv/dt off {res.dv_dt_off / 1e9:.2f} V/ns",
            ),
            (
                "t_fi (descente courant)",
                f"{res.t_fi * 1e9:.1f}",
                f"di/dt off {res.di_dt_off / 1e9:.2f} A/ns",
            ),
        ],
    )
    print()
    kv("t_on / t_off", f"{res.t_on * 1e9:.1f} ns / {res.t_off * 1e9:.1f} ns")
    kv(
        f"R_DS(on) à {T_J_EVAL:.0f} °C",
        f"{res.r_ds_on * 1e3:.3f} mohm  "
        f"({res.r_ds_on / mosfet.r_ds_on.r_ds_on_25:.2f} x la valeur à 25 °C)",
    )


def report_thermal(mosfet, th, r_th_source_label: str) -> None:
    section(5, "Couplage thermique")
    t_j_max = mosfet.thermal.t_j_max
    kv("R_th jonction -> ambiante", f"{th.r_th:.1f} °C/W  ({r_th_source_label})")
    kv("Itérations", f"{th.iterations}  (converged={th.converged})")
    kv("Pertes à Tj convergé", f"{th.loss.p_total:.3f} W")
    print()
    if not th.converged:
        alert("error", "DIVERGE — emballement thermique.", prefix="Tj")
    elif th.t_j_max_exceeded:
        alert(
            "error",
            f"Tj = {th.t_j:.1f} °C, au-dessus de T_j,max = {t_j_max:.0f} °C.",
            prefix="Tj",
        )
    else:
        alert(
            "info",
            f"Tj = {th.t_j:.1f} °C, marge {t_j_max - th.t_j:.0f} °C sous "
            f"T_j,max = {t_j_max:.0f} °C.",
            prefix="Tj",
        )


def report_figures(mosfet, res, th) -> None:
    if not (SHOW_PLOTS or SAVE_PLOTS):
        return
    section(6, "Figures")

    fig_loss = plot_loss_breakdown(
        res,
        title=f"{MOSFET} — bilan de pertes",
        subtitle=f"{V_TURN_ON:.0f} V / {I_ON:.0f} A / {F_SW_KHZ:.0f} kHz "
        f"à Tj = {T_J_EVAL:.0f} °C — total {res.p_total:.2f} W",
        theme=PLOT_THEME,
    )
    fig_th = plot_thermal_iteration(
        th,
        t_j_max=mosfet.thermal.t_j_max,
        title=f"{MOSFET} — convergence de Tj",
        subtitle=f"R_th = {th.r_th:.0f} °C/W, T_amb = {T_AMBIENT:.0f} °C",
        theme=PLOT_THEME,
    )

    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(exist_ok=True)
        for fig, name in (
            (fig_loss, "single_mosfet_loss_breakdown.png"),
            (fig_th, "single_mosfet_thermal.png"),
        ):
            path = OUTPUT_DIR / name
            fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
            kv("écrit", str(path))

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_loss)
        plt.close(fig_th)


# ============================================================================ #
#  Calcul
# ============================================================================ #


def main() -> int:
    kv("Racine du projet", str(ROOT))
    kv("MOSFET disponibles", ", ".join(sorted(MOSFET_LIBRARY)))
    kv("Drivers disponibles", ", ".join(sorted(DRIVER_LIBRARY)))

    mosfet = load_mosfet(MOSFET)
    driver = load_driver(DRIVER)
    op = build_operating_point()

    r_th, alerts_th, active = resolve_r_th(mosfet)
    alerts = check_operating_point(mosfet, op) + alerts_th
    if alerts:
        print()
        for level, message in alerts:
            if level == "stop":
                alert("error", message, prefix="CALCUL IMPOSSIBLE")
            else:
                alert("warn", message)
    if any(level == "stop" for level, _ in alerts):
        return 1

    try:
        res = loss_single_mosfet_at_temp(mosfet, driver, op, t_j=T_J_EVAL)
    except ValueError as err:
        print()
        alert("error", str(err), prefix="CALCUL IMPOSSIBLE")
        return 1

    th = loss_thermal_iteration(mosfet, driver, op, t_ambient=T_AMBIENT, r_th=r_th)

    label = {"dissip": "dissipateurs", "datasheet": "datasheet", "manual": "saisi"}
    r_th_label = label[R_TH_SOURCE] if r_th is not None else "datasheet"

    report_operating_point(mosfet)
    # Le détail des branches n'a de sens que quand ce sont elles qui donnent R_th.
    if active and r_th is not None:
        report_thermal_paths(mosfet, active, r_th)
    report_losses(res)
    report_switching(mosfet, res)
    report_thermal(mosfet, th, r_th_label)
    report_figures(mosfet, res, th)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
