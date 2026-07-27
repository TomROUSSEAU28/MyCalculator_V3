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


def rr_loss_low_side(q_rr_ls: float, v_bus: float, f_sw: float) -> float:
    """
    Recovery power for a charge swept out of the LOW-SIDE diode.
    P_rr,LS = Q_rr,LS * V_bus * f_sw

    The event is the HS turning on and forcing the LS body diode to recover.
    This returns the power that event represents — it does NOT say which
    device dissipates it: the reverse current flows through the recovering
    diode AND the switch forcing it, in series across the bus, so the split
    is a separate question. That is what QRR_ALLOCATION decides; pass this
    function the already-allocated share.

    q_rr_ls : LS diode reverse recovery charge, or the share of it [C]
    v_bus   : voltage commutated by the HS turn-on [V]
    f_sw    : switching frequency [Hz]
    """
    return q_rr_ls * v_bus * f_sw


def rr_loss_high_side(q_rr_hs: float, v_bus: float, f_sw: float) -> float:
    """
    Recovery power for a charge swept out of the HIGH-SIDE diode.
    P_rr,HS = Q_rr,HS * V_bus * f_sw

    Mirror of rr_loss_low_side(): the event is the LS turning on. Same
    caveat — this is the energy of the event, not an attribution to a
    device. See QRR_ALLOCATION.

    q_rr_hs : HS diode reverse recovery charge, or the share of it [C]
    v_bus   : voltage commutated by the LS turn-on [V]
    f_sw    : switching frequency [Hz]
    """
    return q_rr_hs * v_bus * f_sw


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

    # Charge this turn-on has to sweep out of the opposite diode.
    #   None -> unknown, fall back to this MOSFET's own body diode scaled to
    #           the real di/dt (single-switch estimate)
    #   0.0  -> explicitly no recovery (ZCS, SiC/GaN without body diode, or a
    #           half-bridge that allocates the charge elsewhere)
    # The loss is driven by v_turn_on, so it dies out under ZVS either way.
    q_rr_opposite: float | None = None

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
    # charge that this turn-on has to sweep out. q_rr_opposite is the OTHER
    # switch's diode; left at 0, this MOSFET's own diode is scaled to di/dt.
    # The recovery term is proportional to v_turn_on, so ZVS zeroes it too.
    if op.q_rr_opposite is None:
        q_rr = mosfet.body_diode.q_rr_at_condition(
            di_dt=di_dt_on,
            # The diode blocks whatever this turn-on commutates...
            v_r=op.v_turn_on,
            # ...and I_F IS i_body: same diode, same forward current. The
            # inductor current barely moves over a dead time, so the current
            # the diode was conducting is the current being commutated out of
            # it. i_body = 0 therefore gives Q_rr = 0, and correctly so: a
            # diode that never conducted has no charge to recover.
            i_f=op.i_body,
        )
    else:
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


# region half_bridge


class QRR_ALLOCATION(BaseModel):
    """
    Where the reverse recovery energy Q_rr * V_bus actually ends up.

    During recovery the reverse current flows through the switch that is
    turning on AND through the diode that is recovering, in series across the
    bus. How the energy splits depends on how the bus voltage divides between
    them over the transient:

      - at the start the diode is still conducting (~0 V across it), so the
        whole bus sits across the active switch -> all the loss is in the
        SWITCH;
      - at the end the diode blocks the full bus and the switch is fully
        enhanced -> the loss is in the DIODE.

    fraction_to_switch = 1.0 is the usual conservative engineering choice
    (all of it charged to the switch forcing the recovery). Lower it when you
    have measurements or a diode with a soft recovery (large tb/ta ratio),
    where a meaningful share is dissipated in the diode itself.
    """

    fraction_to_switch: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def fraction_to_diode(self) -> float:
        return 1.0 - self.fraction_to_switch


class HALF_BRIDGE_OPERATING_POINT(BaseModel):
    """
    Working point of a half-bridge leg.

    The two switches get their own OPERATING_POINT because in any real leg
    they do NOT see the same thing: different duty cycle, different RMS
    current, and usually only the low side conducts its body diode during the
    dead time.

    Their q_rr_opposite fields are IGNORED — the whole point of the half
    bridge is that the recovery charge is a property of the OTHER device, so
    it is resolved here and cross-allocated.
    """

    high_side: OPERATING_POINT
    low_side: OPERATING_POINT
    qrr_allocation: QRR_ALLOCATION = Field(default_factory=QRR_ALLOCATION)


class HALF_BRIDGE_LOSS_RESULT(BaseModel):
    """Loss breakdown of a half-bridge leg at given junction temperatures."""

    high_side: MOSFET_LOSS_RESULT
    low_side: MOSFET_LOSS_RESULT

    # Recovered charges, each evaluated at the di/dt the OPPOSITE turn-on
    # actually imposes on it.
    q_rr_hs: float  # HS body diode, recovered when the LS turns on [C]
    q_rr_ls: float  # LS body diode, recovered when the HS turns on [C]

    # Recovery energy allocation.
    # The "switch" share is already inside high_side/low_side .p_body, because
    # it is charged to the device that forces the recovery. The "diode" share
    # is added on top of the device that owns the recovering diode.
    p_rr_switch_hs: float  # in the HS, forced by the HS turning on
    p_rr_switch_ls: float  # in the LS, forced by the LS turning on
    p_rr_diode_hs: float  # in the HS body diode, forced by the LS turning on
    p_rr_diode_ls: float  # in the LS body diode, forced by the HS turning on

    p_hs_total: float  # everything dissipated in the HS device [W]
    p_ls_total: float  # everything dissipated in the LS device [W]
    p_total: float  # whole leg [W]


class HALF_BRIDGE_THERMAL_RESULT(BaseModel):
    """Outcome of the coupled two-node Tj loop."""

    converged: bool
    iterations: int
    t_j_hs: float
    t_j_ls: float
    t_ambient: float
    r_th_hs: float
    r_th_ls: float
    r_th_shared: float
    loss: HALF_BRIDGE_LOSS_RESULT
    history: list[tuple[float, float]] = Field(default_factory=list)  # (Tj_hs, Tj_ls)
    t_j_max_exceeded_hs: bool
    t_j_max_exceeded_ls: bool

    @property
    def t_j_max_exceeded(self) -> bool:
        return self.t_j_max_exceeded_hs or self.t_j_max_exceeded_ls


def loss_half_bridge_at_temp(
    mosfet_hs: MOSFET_SPEC,
    driver_hs: DRIVER_MOSFET_SPEC,
    mosfet_ls: MOSFET_SPEC,
    driver_ls: DRIVER_MOSFET_SPEC,
    op: HALF_BRIDGE_OPERATING_POINT,
    t_j_hs: float,
    t_j_ls: float,
) -> HALF_BRIDGE_LOSS_RESULT:
    """
    Loss breakdown of a half-bridge leg at GIVEN junction temperatures.

    What a half bridge adds over two independent MOSFETs is the reverse
    recovery coupling, and it is a CROSS coupling:

        HS turns on -> the LOW side body diode recovers
        LS turns on -> the HIGH side body diode recovers

    So each Q_rr is a property of one device but is triggered by the other,
    at the di/dt the other one imposes. That charge is then split between the
    switch forcing the recovery and the diode recovering, per op.qrr_allocation.

    The two temperatures are separate inputs: in a real leg the HS and LS rarely
    run at the same Tj. Use loss_half_bridge_thermal_iteration() to solve them.
    """
    # --- pass 1: timings only -----------------------------------------------
    # di/dt comes from the gate loop alone, it does not depend on Q_rr, so a
    # recovery-free probe is enough to resolve the coupling.
    probe_hs = loss_single_mosfet_at_temp(
        mosfet_hs,
        driver_hs,
        op.high_side.model_copy(update={"q_rr_opposite": 0.0}),
        t_j_hs,
    )
    probe_ls = loss_single_mosfet_at_temp(
        mosfet_ls,
        driver_ls,
        op.low_side.model_copy(update={"q_rr_opposite": 0.0}),
        t_j_ls,
    )

    # --- cross-allocated recovery charges ------------------------------------
    # The LS diode recovers under the conditions the HS turn-on imposes on it:
    # its di/dt, the voltage it commutates, and the current the LS diode had
    # been freewheeling. And symmetrically for the HS diode.
    # I_F is each diode's OWN i_body — the current it was freewheeling before
    # the opposite switch forced it off. A device whose i_body is left at 0
    # declares a diode that never conducts, hence nothing to recover.
    q_rr_ls = mosfet_ls.body_diode.q_rr_at_condition(
        di_dt=probe_hs.di_dt_on,
        v_r=op.high_side.v_turn_on,
        i_f=op.low_side.i_body,
    )
    q_rr_hs = mosfet_hs.body_diode.q_rr_at_condition(
        di_dt=probe_ls.di_dt_on,
        v_r=op.low_side.v_turn_on,
        i_f=op.high_side.i_body,
    )

    to_switch = op.qrr_allocation.fraction_to_switch
    to_diode = op.qrr_allocation.fraction_to_diode

    # --- pass 2: real losses, each switch charged with the OPPOSITE diode ----
    result_hs = loss_single_mosfet_at_temp(
        mosfet_hs,
        driver_hs,
        op.high_side.model_copy(update={"q_rr_opposite": q_rr_ls * to_switch}),
        t_j_hs,
    )
    result_ls = loss_single_mosfet_at_temp(
        mosfet_ls,
        driver_ls,
        op.low_side.model_copy(update={"q_rr_opposite": q_rr_hs * to_switch}),
        t_j_ls,
    )

    p_rr_switch_hs = result_hs.p_body_rr
    p_rr_switch_ls = result_ls.p_body_rr

    # The DIODE share has no such home: it belongs to the device that owns the
    # recovering diode, which is not the device the call above was made for.
    # It is the only piece that still has to be computed here.
    p_rr_diode_ls = rr_loss_low_side(
        q_rr_ls * to_diode, op.high_side.v_turn_on, op.high_side.f_sw
    )
    p_rr_diode_hs = rr_loss_high_side(
        q_rr_hs * to_diode, op.low_side.v_turn_on, op.low_side.f_sw
    )

    # result_*.p_total already carries the switch share (through p_body); only
    # the diode share still has to be added to its owner.
    p_hs_total = result_hs.p_total + p_rr_diode_hs
    p_ls_total = result_ls.p_total + p_rr_diode_ls

    return HALF_BRIDGE_LOSS_RESULT(
        high_side=result_hs,
        low_side=result_ls,
        q_rr_hs=q_rr_hs,
        q_rr_ls=q_rr_ls,
        p_rr_switch_hs=p_rr_switch_hs,
        p_rr_switch_ls=p_rr_switch_ls,
        p_rr_diode_hs=p_rr_diode_hs,
        p_rr_diode_ls=p_rr_diode_ls,
        p_hs_total=p_hs_total,
        p_ls_total=p_ls_total,
        p_total=p_hs_total + p_ls_total,
    )


def loss_half_bridge_thermal_iteration(
    mosfet_hs: MOSFET_SPEC,
    driver_hs: DRIVER_MOSFET_SPEC,
    mosfet_ls: MOSFET_SPEC,
    driver_ls: DRIVER_MOSFET_SPEC,
    op: HALF_BRIDGE_OPERATING_POINT,
    t_ambient: float = 25.0,
    r_th_hs: float | None = None,
    r_th_ls: float | None = None,
    r_th_shared: float = 0.0,
    max_iter: int = 50,
    tol: float = 0.01,
    relaxation: float = 1.0,
) -> HALF_BRIDGE_THERMAL_RESULT:
    """
    Self-consistent junction temperatures of a half-bridge leg.

    Same fixed point as the single-MOSFET case, but on TWO coupled nodes —
    and the coupling runs both ways: each switch's Q_rr allocation depends on
    the other's di/dt, and a shared heatsink makes each one's temperature
    depend on the other's losses.

        T_sink  = T_amb  + R_th,shared * (P_hs + P_ls)
        Tj_hs   = T_sink + R_th,hs     *  P_hs
        Tj_ls   = T_sink + R_th,ls     *  P_ls

    r_th_hs / r_th_ls : per-device thermal resistance [°C/W]. With
                        r_th_shared = 0 these are plain junction-to-ambient
                        paths and the nodes only couple through Q_rr; default
                        is each MOSFET's datasheet R_thJA.
    r_th_shared       : common path both devices push through, e.g. a shared
                        heatsink or PCB copper pour [°C/W]. Non-zero makes the
                        hotter switch heat the cooler one — that is what turns
                        a marginal leg into a runaway.
    relaxation        : damping in ]0, 1]; lower it if the loop oscillates.
    """
    if not (0.0 < relaxation <= 1.0):
        raise ValueError(f"relaxation must be in ]0, 1], got {relaxation}")

    if r_th_hs is None:
        r_th_hs = mosfet_hs.thermal.r_thja_value()
    if r_th_ls is None:
        r_th_ls = mosfet_ls.thermal.r_thja_value()

    t_j_hs = t_ambient
    t_j_ls = t_ambient
    history: list[tuple[float, float]] = []
    converged = False
    iterations = 0
    loss = loss_half_bridge_at_temp(
        mosfet_hs, driver_hs, mosfet_ls, driver_ls, op, t_j_hs, t_j_ls
    )

    runaway_limit = 10.0 * max(mosfet_hs.thermal.t_j_max, mosfet_ls.thermal.t_j_max)

    for iterations in range(1, max_iter + 1):
        loss = loss_half_bridge_at_temp(
            mosfet_hs, driver_hs, mosfet_ls, driver_ls, op, t_j_hs, t_j_ls
        )

        t_sink = t_ambient + r_th_shared * (loss.p_hs_total + loss.p_ls_total)
        t_j_hs_new = t_sink + r_th_hs * loss.p_hs_total
        t_j_ls_new = t_sink + r_th_ls * loss.p_ls_total

        t_j_hs_next = t_j_hs + relaxation * (t_j_hs_new - t_j_hs)
        t_j_ls_next = t_j_ls + relaxation * (t_j_ls_new - t_j_ls)
        history.append((t_j_hs_next, t_j_ls_next))

        # Both nodes must settle, not just the hotter one
        delta = max(abs(t_j_hs_next - t_j_hs), abs(t_j_ls_next - t_j_ls))
        t_j_hs, t_j_ls = t_j_hs_next, t_j_ls_next

        if delta < tol:
            converged = True
            loss = loss_half_bridge_at_temp(
                mosfet_hs, driver_hs, mosfet_ls, driver_ls, op, t_j_hs, t_j_ls
            )
            break

        if max(t_j_hs, t_j_ls) > runaway_limit:
            break

    return HALF_BRIDGE_THERMAL_RESULT(
        converged=converged,
        iterations=iterations,
        t_j_hs=t_j_hs,
        t_j_ls=t_j_ls,
        t_ambient=t_ambient,
        r_th_hs=r_th_hs,
        r_th_ls=r_th_ls,
        r_th_shared=r_th_shared,
        loss=loss,
        history=history,
        t_j_max_exceeded_hs=t_j_hs > mosfet_hs.thermal.t_j_max,
        t_j_max_exceeded_ls=t_j_ls > mosfet_ls.thermal.t_j_max,
    )


# endregion
