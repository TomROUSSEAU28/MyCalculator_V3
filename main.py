"""
Demo run of the MOSFET loss model.

Synchronous buck leg, BSC016N06NS driven by a UCC27714:
  - high side hard switched
  - low side as a synchronous rectifier, ZVS on both edges

Run from the project root:  python main.py
Figures land in OUTPUT/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no interactive window needed

from DATABASE.db_driver_mosfet import load_driver
from DATABASE.db_mosfet import load_mosfet
from SRC.MOSFET.mosfet_loss import (
    HALF_BRIDGE_OPERATING_POINT,
    OPERATING_POINT,
    QRR_ALLOCATION,
    loss_half_bridge_at_temp,
    loss_half_bridge_thermal_iteration,
    loss_single_mosfet_at_temp,
    loss_thermal_iteration,
)
from SRC.MOSFET.mosfet_plot import (
    loss_table,
    plot_loss_breakdown,
    plot_thermal_iteration,
)

OUTPUT = Path(__file__).parent / "OUTPUT"

# --- design point ------------------------------------------------------------
V_BUS = 48.0
I_OUT = 25.0
F_SW = 100e3
DUTY = 0.4
T_AMBIENT = 40.0
R_G_EXT = 2.2

mosfet = load_mosfet("BSC016N06NS")
driver = load_driver("UCC27714")

# High side: hard switched, conducts for D
high_side = OPERATING_POINT(
    v_turn_on=V_BUS,
    v_turn_off=V_BUS,
    i_rms=I_OUT * DUTY**0.5,
    f_sw=F_SW,
    i_on=I_OUT,
    i_off=I_OUT,
    r_g_ext_on=R_G_EXT,
    r_g_ext_off=1.0,
)

# Low side: synchronous rectifier. Turns on and off at ~0 V (the inductor
# current has already swung the node), and conducts its body diode during the
# dead time.
low_side = OPERATING_POINT(
    v_turn_on=0.0,
    v_turn_off=0.0,
    i_rms=I_OUT * (1.0 - DUTY) ** 0.5,
    f_sw=F_SW,
    i_on=I_OUT,
    i_off=I_OUT,
    r_g_ext_on=R_G_EXT,
    r_g_ext_off=1.0,
    i_body=I_OUT,
    d_body=0.02,  # 2 % of the period spent in the dead time
)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    t_j_max = mosfet.thermal.t_j_max
    part = mosfet.component_info.part_number

    # === 1. single MOSFET, hard switched ====================================
    single = loss_single_mosfet_at_temp(mosfet, driver, high_side, t_j=100.0)
    print(f"\n--- {part} alone, hard switched, Tj = 100 C ---")
    print(loss_table(single).to_string())
    # No reverse recovery here, and that is correct: this operating point has
    # i_body = 0, i.e. a high side whose own body diode never freewheels. The
    # recovery it really suffers comes from the LOW side diode, which only the
    # half-bridge model below can account for.
    print(f"  t_on = {single.t_on * 1e9:.1f} ns    t_off = {single.t_off * 1e9:.1f} ns")
    print(f"  dv/dt on = {single.dv_dt_on / 1e9:.2f} V/ns   di/dt on = {single.di_dt_on / 1e9:.2f} A/ns")

    fig = plot_loss_breakdown(
        single,
        title=f"{part} — loss breakdown",
        subtitle=f"Hard switched, {V_BUS:.0f} V / {I_OUT:.0f} A / {F_SW / 1e3:.0f} kHz, "
        f"Tj = 100 °C — total {single.p_total:.2f} W",
    )
    fig.savefig(OUTPUT / "single_loss.png", dpi=160, facecolor=fig.get_facecolor())

    thermal = loss_thermal_iteration(
        mosfet, driver, high_side, t_ambient=T_AMBIENT, r_th=20.0
    )
    print(f"  Tj = {thermal.t_j:.1f} C after {thermal.iterations} iterations "
          f"(converged={thermal.converged}, over T_j,max={thermal.t_j_max_exceeded})")

    fig = plot_thermal_iteration(
        thermal,
        t_j_max=t_j_max,
        title=f"{part} — junction temperature",
        subtitle=f"R_th = {thermal.r_th:.0f} °C/W, T_amb = {T_AMBIENT:.0f} °C — "
        f"settles at {thermal.t_j:.0f} °C",
    )
    fig.savefig(OUTPUT / "single_thermal.png", dpi=160, facecolor=fig.get_facecolor())

    # === 2. half-bridge leg =================================================
    leg = HALF_BRIDGE_OPERATING_POINT(
        high_side=high_side,
        low_side=low_side,
        qrr_allocation=QRR_ALLOCATION(fraction_to_switch=1.0),
    )

    hb = loss_half_bridge_at_temp(
        mosfet, driver, mosfet, driver, leg, t_j_hs=100.0, t_j_ls=100.0
    )
    print(f"\n--- Half-bridge leg, {part} x2, Tj = 100 C ---")
    print(loss_table(hb).to_string())
    print(f"  Q_rr LS = {hb.q_rr_ls * 1e9:.0f} nC recovered by the HS turn-on")
    print(f"  Q_rr HS = {hb.q_rr_hs * 1e9:.0f} nC recovered by the LS turn-on")

    fig = plot_loss_breakdown(
        hb,
        title="Half-bridge leg — loss breakdown",
        subtitle=f"Sync buck {V_BUS:.0f} V / {I_OUT:.0f} A / {F_SW / 1e3:.0f} kHz, "
        f"D = {DUTY:.0%}, Tj = 100 °C — leg total {hb.p_total:.2f} W",
    )
    fig.savefig(OUTPUT / "half_bridge_loss.png", dpi=160, facecolor=fig.get_facecolor())

    # Shared heatsink: the hard-switched HS drags the cool LS up with it.
    hb_thermal = loss_half_bridge_thermal_iteration(
        mosfet, driver, mosfet, driver, leg,
        t_ambient=T_AMBIENT,
        r_th_hs=5.0,
        r_th_ls=5.0,
        r_th_shared=12.0,
    )
    print(f"  Tj_hs = {hb_thermal.t_j_hs:.1f} C   Tj_ls = {hb_thermal.t_j_ls:.1f} C "
          f"({hb_thermal.iterations} iterations, converged={hb_thermal.converged})")

    fig = plot_thermal_iteration(
        hb_thermal,
        t_j_max=t_j_max,
        title="Half-bridge leg — junction temperatures",
        subtitle=f"Shared heatsink R_th = {hb_thermal.r_th_shared:.0f} °C/W "
        f"+ {hb_thermal.r_th_hs:.0f} °C/W per device, T_amb = {T_AMBIENT:.0f} °C",
    )
    fig.savefig(OUTPUT / "half_bridge_thermal.png", dpi=160, facecolor=fig.get_facecolor())

    # Dark theme renders from the same data — the palette is re-stepped for the
    # dark surface, not flipped.
    fig = plot_loss_breakdown(
        hb, title="Half-bridge leg — loss breakdown", theme="dark"
    )
    fig.savefig(OUTPUT / "half_bridge_loss_dark.png", dpi=160, facecolor=fig.get_facecolor())

    print(f"\nFigures written to {OUTPUT}")


if __name__ == "__main__":
    main()
