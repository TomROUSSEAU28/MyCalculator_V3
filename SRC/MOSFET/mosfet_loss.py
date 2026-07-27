from pydantic import BaseModel, Field

from DATABASE.db_driver_mosfet import DRIVER_MOSFET_SPEC, DRIVING_MODEL_ENUM
from DATABASE.db_mosfet import MOSFET_SPEC


# region mosfet_loss


def conduction_loss(rds_on_tj: float, i_rms: float) -> float:
    """
    Conduction loss.
    P_cond = Rds(on)(Tj) * I_rms^2

    rds_on_tj : on-resistance at junction temperature Tj [ohm]
    i_rms     : RMS current through the MOSFET [A]
    """
    return rds_on_tj * i_rms**2


def switching_loss(
    v_on: float,
    i_on: float,
    t_on: float,
    v_off: float,
    i_off: float,
    t_off: float,
    f_sw: float,
) -> float:
    """
    Switching loss.
    P_sw = 0.5 * (V_on*I_on*t_on + V_off*I_off*t_off) * f_sw

    v_on, i_on, t_on   : voltage, current, time during turn-on
    v_off, i_off, t_off: voltage, current, time during turn-off
    f_sw               : switching frequency [Hz]
    """
    e_on = v_on * i_on * t_on
    e_off = v_off * i_off * t_off
    return 0.5 * (e_on + e_off) * f_sw


def gate_loss(q_g: float, delta_v_gs: float, f_sw: float) -> float:
    """
    Total gate drive loss.
    P_gate = Q_g * delta_Vgs * f_sw

    q_g        : total gate charge [C]
    delta_v_gs : gate-source voltage swing [V]
    f_sw       : switching frequency [Hz]
    """
    return q_g * delta_v_gs * f_sw


def gate_loss_split(
    q_g: float,
    delta_v_gs: float,
    f_sw: float,
    r_drv: float,
    r_ext: float,
    r_gint: float,
) -> tuple[float, float, float]:
    """
    Gate loss split between driver, external gate resistor,
    and internal gate resistance, proportional to their
    share of the total gate resistance path.

    r_drv  : driver output resistance [ohm]
    r_ext  : external gate resistor [ohm]
    r_gint : MOSFET internal gate resistance [ohm]

    Returns (P_drv, P_ext, P_gint)
    """
    p_gate_total = gate_loss(q_g, delta_v_gs, f_sw)
    r_total = r_drv + r_ext + r_gint

    p_drv = p_gate_total * (r_drv / r_total)
    p_ext = p_gate_total * (r_ext / r_total)
    p_gint = p_gate_total * (r_gint / r_total)

    return p_drv, p_ext, p_gint


def coss_loss(c_oss_er: float, v_on: float, f_sw: float) -> float:
    """
    Output capacitance (Coss) loss.
    P_oss = 0.5 * C_oss_er * V_on^2 * f_sw

    c_oss_er : energy-related output capacitance [F]
    v_on     : blocking voltage before turn-on [V]
    f_sw     : switching frequency [Hz]
    """
    return 0.5 * c_oss_er * v_on**2 * f_sw


def body_diode_loss(
    v_f: float, i_body: float, d_body: float, q_rr: float, v_rr: float, f_sw: float
) -> float:
    """
    Body diode loss (conduction + reverse recovery).
    P_body = V_F * I_body * D_body + Q_rr * V_rr * f_sw

    v_f     : body diode forward voltage drop [V]
    i_body  : body diode conduction current [A]
    d_body  : body diode conduction duty cycle [-]
    q_rr    : reverse recovery charge [C]
    v_rr    : reverse voltage during recovery [V]
    f_sw    : switching frequency [Hz]
    """
    p_cond_diode = v_f * i_body * d_body
    p_rr = q_rr * v_rr * f_sw
    return p_cond_diode + p_rr


# endregion


# region helper_loss


def total_gate_resistance(r_drv: float, r_ext: float, r_gint: float) -> float:
    """
    Total resistance in the gate loop.
    R_g,total = R_drv + R_ext + R_gint

    r_drv  : driver output resistance (source or sink) [ohm]
    r_ext  : external gate resistor [ohm]
    r_gint : MOSFET internal gate resistance [ohm]
    """
    return r_drv + r_ext + r_gint


def gate_current_resistive(v_drive: float, v_gate: float, r_g_total: float) -> float:
    """
    Gate current for a RESISTIVE driver (Model A).
    I_g = (V_drive - V_gate) / R_g,total

    This is just Ohm's law across the gate loop: the driver pulls the gate
    toward V_drive, and the instantaneous gate voltage opposes it.

    v_drive   : driver rail the gate is being pulled toward [V]
                (V_on for turn-on, V_off for turn-off)
    v_gate    : instantaneous gate voltage during the region of interest [V]
                (typically V_plateau during the Miller region)
    r_g_total : total gate loop resistance [ohm]
    """
    return (v_drive - v_gate) / r_g_total


def gate_current_effective(
    v_drive: float,
    v_gate: float,
    r_g_total: float,
    i_peak: float | None = None,
    i_source_ideal: float | None = None,
) -> float:
    """
    Effective gate current, selecting the right driver model.

    Three cases:
      1. IDEAL CURRENT SOURCE (i_source_ideal set):
         the driver forces a constant current regardless of R_g,total.
         The gate resistance does NOT limit the current here.

      2. RESISTIVE with PEAK CLAMP (i_peak set):
         Ohm's law would give (V_drive - V_gate)/R_g,total, but the driver
         physically cannot deliver more than i_peak. The real current is
         whichever is SMALLER — the driver saturates.

      3. PURELY RESISTIVE (neither set):
         plain Ohm's law, no limit.

    Returns the magnitude of the gate current [A] (always positive).
    """
    if i_source_ideal is not None:
        # Case 1: constant-current driver — R_g,total is irrelevant
        return abs(i_source_ideal)

    i_resistive = abs(gate_current_resistive(v_drive, v_gate, r_g_total))

    if i_peak is not None:
        # Case 2: clamp — the driver cannot exceed its peak capability
        return min(i_resistive, abs(i_peak))

    # Case 3: unlimited resistive
    return i_resistive


def is_driver_current_limited(
    v_drive: float, v_gate: float, r_g_total: float, i_peak: float
) -> bool:
    """
    True if the DRIVER is the bottleneck, False if the GATE RESISTANCE is.

    Useful diagnostic: if this returns True, increasing R_ext will NOT slow
    down switching (the driver was already saturating), so your gate resistor
    is doing nothing until you drop below the clamp point.
    """
    i_resistive = abs(gate_current_resistive(v_drive, v_gate, r_g_total))
    return i_resistive > abs(i_peak)


def switching_time_from_charge(q_region: float, i_gate: float) -> float:
    """
    Time to move a given gate charge at a given gate current.
    t = Q / I_g

    This is the definition of current: charge per unit time. If the gate
    current is constant over the region, the time is simply charge divided
    by current.

    q_region : charge to be moved in this region [C]
    i_gate   : effective gate current during this region [A]
    """
    if i_gate <= 0.0:
        raise ValueError(f"i_gate must be > 0, got {i_gate}")
    return q_region / i_gate


def turn_on_time(
    q_sw: float,
    v_on: float,
    v_plateau: float,
    r_g_total: float,
    i_peak: float | None = None,
    i_source_ideal: float | None = None,
) -> float:
    """
    Turn-on switching time from the datasheet SWITCHING charge Q_sw.

    NOTE — this is the coarse one-region estimate and it does NOT match the
    t_on reported by loss_single_mosfet_at_temp(), which resolves the edge in
    two sub-intervals (t_ri from Q_gs-Q_gth, t_fv from Q_gd) and is the one the
    loss model uses. Expect a difference wherever Q_sw != (Q_gs-Q_gth) + Q_gd.
    Use this only for a quick datasheet sanity check.

    The gate sits at V_plateau while the drain voltage swings, so the driver
    sees a constant voltage difference (V_on - V_plateau) across R_g,total.

    q_sw           : switching charge [C]
    v_on           : driver turn-on rail [V]
    v_plateau      : gate plateau (Miller) voltage [V]
    r_g_total      : total gate loop resistance [ohm]
    i_peak         : driver peak source current, if it clamps [A]
    i_source_ideal : constant source current, if ideal current driver [A]
    """
    i_g = gate_current_effective(
        v_drive=v_on,
        v_gate=v_plateau,
        r_g_total=r_g_total,
        i_peak=i_peak,
        i_source_ideal=i_source_ideal,
    )
    return switching_time_from_charge(q_sw, i_g)


def turn_off_time(
    q_sw: float,
    v_off: float,
    v_plateau: float,
    r_g_total: float,
    i_peak: float | None = None,
    i_sink_ideal: float | None = None,
) -> float:
    """
    Turn-off switching time from Q_sw. Same caveat as turn_on_time(): this is
    the coarse estimate, not the t_off the loss model reports.

    Same idea as turn-on, but the driver pulls DOWN toward V_off, so the
    driving voltage is (V_plateau - V_off). Note this is often much smaller
    than (V_on - V_plateau), which is why turn-off is usually slower unless
    you use a negative V_off or a lower sink resistance.
    """
    i_g = gate_current_effective(
        v_drive=v_off,
        v_gate=v_plateau,
        r_g_total=r_g_total,
        i_peak=i_peak,
        i_source_ideal=i_sink_ideal,
    )
    return switching_time_from_charge(q_sw, i_g)


def current_rise_time(
    q_gs_miller, v_on, v_th, v_plateau, r_g_total, i_peak=None, i_source_ideal=None
) -> float:
    """V_th -> V_plateau. Current rises, V_ds still high."""
    v_gate_avg = 0.5 * (v_th + v_plateau)
    i_g = gate_current_effective(v_on, v_gate_avg, r_g_total, i_peak, i_source_ideal)
    return q_gs_miller / i_g


def voltage_fall_time(
    q_gd, v_on, v_plateau, r_g_total, i_peak=None, i_source_ideal=None
) -> float:
    """Miller plateau. V_ds falls, gate pinned at V_plateau."""
    i_g = gate_current_effective(v_on, v_plateau, r_g_total, i_peak, i_source_ideal)
    return q_gd / i_g


def voltage_rise_time(
    q_gd, v_off, v_plateau, r_g_total, i_peak=None, i_sink_ideal=None
) -> float:
    """Miller plateau. V_ds rises."""
    i_g = gate_current_effective(v_off, v_plateau, r_g_total, i_peak, i_sink_ideal)
    return q_gd / i_g


def current_fall_time(
    q_gs_miller, v_off, v_th, v_plateau, r_g_total, i_peak=None, i_sink_ideal=None
) -> float:
    """V_plateau -> V_th. Current falls, V_ds already high."""
    v_gate_avg = 0.5 * (v_th + v_plateau)
    i_g = gate_current_effective(v_off, v_gate_avg, r_g_total, i_peak, i_sink_ideal)
    return q_gs_miller / i_g


def dead_time_zvs(q_oss: float, i_transition: float) -> float:
    """
    Dead time required to fully discharge Coss for ZVS.
    t_dead = Q_oss(V_bus) / I_transition

    Use C_OSS_INFO.q_oss_at_voltage(v_bus) to get q_oss at the real bus
    voltage rather than the datasheet test point.

    i_transition : current available to discharge the node [A]
    """
    if i_transition <= 0.0:
        raise ValueError(f"i_transition must be > 0, got {i_transition}")
    return q_oss / i_transition


def di_dt_from_switching(i_switched: float, t_rise: float) -> float:
    """
    di/dt during turn-on — feeds BODY_DIODE.q_rr_at_condition().
    di/dt = I_switched / t_rise
    """
    if t_rise <= 0.0:
        raise ValueError(f"t_rise must be > 0, got {t_rise}")
    return i_switched / t_rise


def dv_dt_from_switching(v_switched: float, t_transition: float) -> float:
    """
    dv/dt across a switching transition.
    dv/dt = V_switched / t_transition

    Drives EMI, the dV/dt immunity of the opposite switch (Cgd-induced
    spurious turn-on) and the common-mode current through the isolation
    barrier. Feed it the VOLTAGE sub-interval (t_fv at turn-on, t_rv at
    turn-off), not the whole edge.

    Returns 0 when nothing is switched (v_switched = 0, i.e. ZVS).
    """
    if t_transition <= 0.0:
        raise ValueError(f"t_transition must be > 0, got {t_transition}")
    return v_switched / t_transition


# endregion


# region operating_point


class OPERATING_POINT(BaseModel):
    """
    Everything about the circuit that the datasheet does not know.

    The MOSFET_SPEC gives the component, the DRIVER_MOSFET_SPEC gives the
    gate loop; this gives the working point they are used at.
    """

    # Drain-source voltage actually swung on each edge, given separately so
    # that any switching scheme can be described without a mode flag:
    #   hard switching   -> v_turn_on = v_turn_off = V_bus
    #   ZVS at turn-on   -> v_turn_on = 0   (Coss and Q_rr terms vanish on
    #                                        their own, no special case needed)
    #   ZVS at turn-off  -> v_turn_off = 0
    #   snubbed / partial-> whatever fraction of V_bus is really seen
    v_turn_on: float  # V_ds falling across the turn-on transition [V]
    v_turn_off: float  # V_ds rising across the turn-off transition [V]

    i_rms: float  # RMS drain current (conduction) [A]
    f_sw: float  # switching frequency [Hz]

    i_on: float | None = None  # commutated current at turn-on [A]
    i_off: float | None = None  # commutated current at turn-off [A]

    r_g_ext_on: float = 0.0  # external gate resistor, turn-on path [ohm]
    r_g_ext_off: float = 0.0  # external gate resistor, turn-off path [ohm]

    # Body diode conduction during the dead time. d_body is a duty cycle, not
    # a time. i_body does DOUBLE DUTY and the two uses are the same physical
    # current, so there is only one field:
    #   - conduction loss:  P = V_F * i_body * d_body
    #   - recovery scaling: it is the I_F of the Q_rr law (the forward current
    #     the diode carried right before being commutated off)
    # Leaving it at 0 declares a diode that never conducts: no conduction loss
    # AND no reverse recovery, which is the physically consistent pair.
    i_body: float = 0.0
    d_body: float = 0.0

    # Charge this turn-on has to sweep out of the diode facing it — given, not
    # derived. That diode belongs to ANOTHER device (parallel Schottky, discrete
    # freewheel...) which this model does not know, so there is nothing to infer:
    # read the charge off that diode's datasheet at the real di/dt, e.g. with
    # BODY_DIODE.q_rr_at_condition().
    #   0.0 -> no recovery at all (ZCS, ZVS, SiC/GaN without body diode)
    # The loss is driven by v_turn_on, so it dies out under ZVS either way.
    q_rr_opposite: float = 0.0

    def i_commutated_on(self) -> float:
        return self.i_on if self.i_on is not None else self.i_rms

    def i_commutated_off(self) -> float:
        return self.i_off if self.i_off is not None else self.i_rms


class MOSFET_LOSS_RESULT(BaseModel):
    """Loss breakdown at one junction temperature."""

    t_j: float  # temperature the losses were evaluated at [°C]
    r_ds_on: float  # R_DS(on) at that temperature [ohm]

    p_cond: float
    p_sw: float
    p_coss: float

    p_body: float  # = p_body_cond + p_body_rr
    p_body_cond: float  # dead-time conduction, V_F * I * D
    p_body_rr: float  # reverse recovery, Q_rr * V_turn_on * f_sw

    # Gate loss is split: only the internal share heats the die.
    p_gate_total: float
    p_gate_drv: float  # dissipated in the driver
    p_gate_ext: float  # dissipated in the external gate resistor
    p_gate_int: float  # dissipated in the MOSFET die

    p_total: float  # total dissipated IN the MOSFET

    # Timing, kept for diagnostics
    t_on: float  # t_ri + t_fv [s]
    t_off: float  # t_rv + t_fi [s]
    t_ri: float  # current rise, turn-on [s]
    t_fv: float  # voltage fall, turn-on [s]
    t_rv: float  # voltage rise, turn-off [s]
    t_fi: float  # current fall, turn-off [s]

    # Slew rates — what EMI, dV/dt immunity and Q_rr scaling actually depend on
    di_dt_on: float  # [A/s]
    di_dt_off: float  # [A/s]
    dv_dt_on: float  # [V/s]
    dv_dt_off: float  # [V/s]


class THERMAL_ITERATION_RESULT(BaseModel):
    """Outcome of the self-consistent Tj <-> losses loop."""

    converged: bool
    iterations: int
    t_j: float
    t_ambient: float
    r_th: float
    loss: MOSFET_LOSS_RESULT
    history: list[float] = Field(default_factory=list)  # Tj at each iteration
    t_j_max_exceeded: bool


# endregion


# region single_mosfet_loss


def _driver_edge_params(
    driver: DRIVER_MOSFET_SPEC, turn_on: bool
) -> tuple[float, float | None, float | None]:
    """
    Translate a DRIVER_MOSFET_SPEC into the (r_out, i_peak, i_ideal) triplet
    expected by gate_current_effective(), for one edge.

    DRIVING_MODEL_ENUM is named <turn_on>_<turn_off>:
      SOURCE -> ideal current source, the value in current_info is forced
      PEAK   -> resistive output, the value in current_info is the clamp
    """
    kind = driver.Current_information.kind
    i_on_spec, i_off_spec = driver.Current_information.current_info

    if turn_on:
        r_out = driver.r_out_source
        i_spec = i_on_spec
        is_ideal = kind in (
            DRIVING_MODEL_ENUM.SOURCE_SOURCE,
            DRIVING_MODEL_ENUM.SOURCE_PEAK,
        )
    else:
        r_out = driver.r_in_source
        i_spec = i_off_spec
        is_ideal = kind in (
            DRIVING_MODEL_ENUM.SOURCE_SOURCE,
            DRIVING_MODEL_ENUM.PEAK_SOURCE,
        )

    if is_ideal:
        # R_out is irrelevant, but total_gate_resistance() still needs a number
        return (r_out or 0.0), None, i_spec

    if r_out is None:
        raise ValueError(
            "Driver is resistive on this edge but the corresponding output "
            "resistance (r_out_source / r_in_source) is None."
        )
    return r_out, i_spec, None


def loss_single_mosfet_at_temp(
    mosfet: MOSFET_SPEC,
    driver: DRIVER_MOSFET_SPEC,
    op: OPERATING_POINT,
    t_j: float,
) -> MOSFET_LOSS_RESULT:
    """
    Full loss breakdown of ONE MOSFET at a GIVEN junction temperature.

    Nothing here is iterative: Tj is an input, and the only thing it changes
    is R_DS(on). Use loss_thermal_iteration() when Tj itself is unknown.

    mosfet : component from DATABASE.db_mosfet
    driver : gate driver from DATABASE.db_driver_mosfet
    op     : circuit operating point
    t_j    : junction temperature to evaluate at [°C]
    """
    gate = mosfet.gate_charge
    delta_v_gs = driver.v_on - driver.v_off

    # --- gate loop, per edge -------------------------------------------------
    r_out_on, i_peak_on, i_ideal_on = _driver_edge_params(driver, turn_on=True)
    r_out_off, i_peak_off, i_ideal_off = _driver_edge_params(driver, turn_on=False)

    r_g_on = total_gate_resistance(r_out_on, op.r_g_ext_on, mosfet.r_g_int)
    r_g_off = total_gate_resistance(r_out_off, op.r_g_ext_off, mosfet.r_g_int)

    # --- switching times -----------------------------------------------------
    # Turn-on  = current rise (Q_gs,miller) then voltage fall (Q_gd)
    # Turn-off = voltage rise (Q_gd) then current fall (Q_gs,miller)
    if gate.q_gs is None or gate.q_gd is None:
        raise ValueError(
            f"{mosfet.component_info.part_number}: q_gs and q_gd are needed to "
            "split the switching transition. Give at least one of them in the "
            "library entry — GATE_CHARGE_INFO derives the other from q_g."
        )

    q_gs_miller = gate.q_gs - (gate.q_gth or 0.0)
    q_gd = gate.q_gd
    if q_gs_miller <= 0.0:
        raise ValueError(
            f"{mosfet.component_info.part_number}: q_gs ({gate.q_gs}) must be "
            f"greater than q_gth ({gate.q_gth}) — the current rise happens "
            "between V_th and V_plateau."
        )

    t_ri = current_rise_time(
        q_gs_miller,
        driver.v_on,
        gate.v_th,
        gate.v_plateau,
        r_g_on,
        i_peak_on,
        i_ideal_on,
    )
    t_fv = voltage_fall_time(
        q_gd, driver.v_on, gate.v_plateau, r_g_on, i_peak_on, i_ideal_on
    )
    t_rv = voltage_rise_time(
        q_gd, driver.v_off, gate.v_plateau, r_g_off, i_peak_off, i_ideal_off
    )
    t_fi = current_fall_time(
        q_gs_miller,
        driver.v_off,
        gate.v_th,
        gate.v_plateau,
        r_g_off,
        i_peak_off,
        i_ideal_off,
    )

    t_on = t_ri + t_fv
    t_off = t_rv + t_fi

    i_sw_on = op.i_commutated_on()
    i_sw_off = op.i_commutated_off()

    # Slew rates: each one belongs to its own sub-interval, not to t_on/t_off.
    # The current moves during t_ri / t_fi, the voltage during t_fv / t_rv.
    di_dt_on = di_dt_from_switching(i_sw_on, t_ri)
    di_dt_off = di_dt_from_switching(i_sw_off, t_fi)
    dv_dt_on = dv_dt_from_switching(op.v_turn_on, t_fv)
    dv_dt_off = dv_dt_from_switching(op.v_turn_off, t_rv)

    # --- losses --------------------------------------------------------------
    r_ds_on_tj = mosfet.r_ds_on.calculate_at_temperature(t_j)
    p_cond = conduction_loss(r_ds_on_tj, op.i_rms)

    p_sw = switching_loss(
        v_on=op.v_turn_on,
        i_on=i_sw_on,
        t_on=t_on,
        v_off=op.v_turn_off,
        i_off=i_sw_off,
        t_off=t_off,
        f_sw=op.f_sw,
    )

    # Coss energy stored at v_turn_on is dumped in the channel at turn-on.
    # v_turn_on = 0 (ZVS) makes this vanish, so no mode flag is needed — but
    # the Coss curve cannot be integrated down to zero, hence the guard.
    if op.v_turn_on > 0.0:
        c_oss_er = mosfet.c_oss.coss_effective_energy(op.v_turn_on)
        p_coss = coss_loss(c_oss_er, op.v_turn_on, op.f_sw)
    else:
        p_coss = 0.0

    # Body diode: conduction from the dead time, plus the recovery of the
    # charge that this turn-on has to sweep out. That charge belongs to the
    # diode FACING this MOSFET, not to its own body diode, so it is an input.
    # The recovery term is proportional to v_turn_on, so ZVS zeroes it.
    q_rr = op.q_rr_opposite

    p_body = body_diode_loss(
        v_f=mosfet.body_diode.vf,
        i_body=op.i_body,
        d_body=op.d_body,
        q_rr=q_rr,
        v_rr=op.v_turn_on,
        f_sw=op.f_sw,
    )
    # Same two terms kept apart: they are driven by completely different things
    # (dead-time management vs turn-on speed) and are fixed by different means.
    p_body_cond = mosfet.body_diode.vf * op.i_body * op.d_body
    p_body_rr = p_body - p_body_cond

    # Gate loss: averaged over both edges, since the resistance split differs
    # between turn-on and turn-off.
    p_gate_total = gate_loss(gate.q_g, delta_v_gs, op.f_sw)
    p_drv_on, p_ext_on, p_int_on = gate_loss_split(
        gate.q_g, delta_v_gs, op.f_sw, r_out_on, op.r_g_ext_on, mosfet.r_g_int
    )
    p_drv_off, p_ext_off, p_int_off = gate_loss_split(
        gate.q_g, delta_v_gs, op.f_sw, r_out_off, op.r_g_ext_off, mosfet.r_g_int
    )
    # Each edge carries half of Q_g * dVgs, hence the 0.5 weighting.
    p_gate_drv = 0.5 * (p_drv_on + p_drv_off)
    p_gate_ext = 0.5 * (p_ext_on + p_ext_off)
    p_gate_int = 0.5 * (p_int_on + p_int_off)

    # Only the internal gate resistance heats the die — the driver and the
    # external resistor dissipate their share outside the package.
    p_total = p_cond + p_sw + p_coss + p_body + p_gate_int

    return MOSFET_LOSS_RESULT(
        t_j=t_j,
        r_ds_on=r_ds_on_tj,
        p_cond=p_cond,
        p_sw=p_sw,
        p_coss=p_coss,
        p_body=p_body,
        p_body_cond=p_body_cond,
        p_body_rr=p_body_rr,
        p_gate_total=p_gate_total,
        p_gate_drv=p_gate_drv,
        p_gate_ext=p_gate_ext,
        p_gate_int=p_gate_int,
        p_total=p_total,
        t_on=t_on,
        t_off=t_off,
        t_ri=t_ri,
        t_fv=t_fv,
        t_rv=t_rv,
        t_fi=t_fi,
        di_dt_on=di_dt_on,
        di_dt_off=di_dt_off,
        dv_dt_on=dv_dt_on,
        dv_dt_off=dv_dt_off,
    )


def loss_thermal_iteration(
    mosfet: MOSFET_SPEC,
    driver: DRIVER_MOSFET_SPEC,
    op: OPERATING_POINT,
    t_ambient: float = 25.0,
    r_th: float | None = None,
    max_iter: int = 50,
    tol: float = 0.01,
    relaxation: float = 1.0,
) -> THERMAL_ITERATION_RESULT:
    """
    Self-consistent junction temperature.

    R_DS(on) rises with Tj, Tj rises with the losses, and the losses rise with
    R_DS(on) — so neither can be computed alone. Fixed-point iteration:

        Tj(k+1) = T_amb + R_th * P_total(Tj(k))

    until |Tj(k+1) - Tj(k)| < tol. If the loop diverges the design is in
    thermal runaway; that shows up as converged=False (and usually as
    t_j_max_exceeded=True on the way there).

    t_ambient  : ambient / reference temperature [°C]
    r_th       : junction-to-ambient thermal resistance [°C/W].
                 Defaults to the datasheet R_thJA of the MOSFET.
    tol        : convergence threshold on Tj [°C]
    relaxation : damping factor in ]0, 1]. Use < 1 (e.g. 0.5) when the loop
                 oscillates instead of settling — it takes more iterations but
                 stays stable near the runaway point.
    """
    if not (0.0 < relaxation <= 1.0):
        raise ValueError(f"relaxation must be in ]0, 1], got {relaxation}")

    if r_th is None:
        r_th = mosfet.thermal.r_thja_value()

    t_j = t_ambient
    history: list[float] = []
    converged = False
    iterations = 0
    loss = loss_single_mosfet_at_temp(mosfet, driver, op, t_j)

    for iterations in range(1, max_iter + 1):
        loss = loss_single_mosfet_at_temp(mosfet, driver, op, t_j)
        t_j_new = t_ambient + r_th * loss.p_total

        # Damped update: full step when relaxation == 1
        t_j_next = t_j + relaxation * (t_j_new - t_j)
        history.append(t_j_next)

        delta = abs(t_j_next - t_j)
        t_j = t_j_next

        if delta < tol:
            converged = True
            # Re-evaluate so the returned losses match the returned Tj
            loss = loss_single_mosfet_at_temp(mosfet, driver, op, t_j)
            break

        # Runaway guard: R_DS(on) is exponential-ish, Tj can blow up
        if t_j > 10.0 * mosfet.thermal.t_j_max:
            break

    return THERMAL_ITERATION_RESULT(
        converged=converged,
        iterations=iterations,
        t_j=t_j,
        t_ambient=t_ambient,
        r_th=r_th,
        loss=loss,
        history=history,
        t_j_max_exceeded=t_j > mosfet.thermal.t_j_max,
    )


# endregion
