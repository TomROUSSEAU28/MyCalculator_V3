"""
Demo run of the MOSFET loss model.

One BSC016N06NS driven by a UCC27714, hard switched on a 48 V bus.

Run from the project root:  python main.py
Figures land in OUTPUT/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no interactive window needed

from DATABASE.db_driver_mosfet import load_driver
from DATABASE.db_mosfet import load_mosfet
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

# Hard switched, conducts for D. q_rr_opposite left at 0: no diode facing this
# switch here. Give it the charge of the real one (read off ITS datasheet at the
# di/dt below) when there is one.
operating_point = OPERATING_POINT(
    v_turn_on=V_BUS,
    v_turn_off=V_BUS,
    i_rms=I_OUT * DUTY**0.5,
    f_sw=F_SW,
    i_on=I_OUT,
    i_off=I_OUT,
    r_g_ext_on=R_G_EXT,
    r_g_ext_off=1.0,
)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    t_j_max = mosfet.thermal.t_j_max
    part = mosfet.component_info.part_number

    # === 1. loss breakdown at a fixed Tj =====================================
    single = loss_single_mosfet_at_temp(mosfet, driver, operating_point, t_j=100.0)
    print(f"\n--- {part}, hard switched, Tj = 100 C ---")
    print(loss_table(single).to_string())
    print(f"  t_on = {single.t_on * 1e9:.1f} ns    t_off = {single.t_off * 1e9:.1f} ns")
    print(
        f"  dv/dt on = {single.dv_dt_on / 1e9:.2f} V/ns   "
        f"di/dt on = {single.di_dt_on / 1e9:.2f} A/ns"
    )

    fig = plot_loss_breakdown(
        single,
        title=f"{part} — loss breakdown",
        subtitle=f"Hard switched, {V_BUS:.0f} V / {I_OUT:.0f} A / {F_SW / 1e3:.0f} kHz, "
        f"Tj = 100 °C — total {single.p_total:.2f} W",
    )
    fig.savefig(OUTPUT / "single_loss.png", dpi=160, facecolor=fig.get_facecolor())

    # === 2. self-consistent junction temperature =============================
    thermal = loss_thermal_iteration(
        mosfet, driver, operating_point, t_ambient=T_AMBIENT, r_th=20.0
    )
    print(
        f"  Tj = {thermal.t_j:.1f} C after {thermal.iterations} iterations "
        f"(converged={thermal.converged}, over T_j,max={thermal.t_j_max_exceeded})"
    )

    fig = plot_thermal_iteration(
        thermal,
        t_j_max=t_j_max,
        title=f"{part} — junction temperature",
        subtitle=f"R_th = {thermal.r_th:.0f} °C/W, T_amb = {T_AMBIENT:.0f} °C — "
        f"settles at {thermal.t_j:.0f} °C",
    )
    fig.savefig(OUTPUT / "single_thermal.png", dpi=160, facecolor=fig.get_facecolor())

    # Dark theme renders from the same data — the palette is re-stepped for the
    # dark surface, not flipped.
    fig = plot_loss_breakdown(
        single, title=f"{part} — loss breakdown", theme="dark"
    )
    fig.savefig(OUTPUT / "single_loss_dark.png", dpi=160, facecolor=fig.get_facecolor())

    print(f"\nFigures written to {OUTPUT}")


if __name__ == "__main__":
    main()
