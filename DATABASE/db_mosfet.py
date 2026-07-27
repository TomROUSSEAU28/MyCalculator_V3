import numpy as np
from pydantic import BaseModel, model_validator
from DATABASE.db_component import COMPONENT_INFO, THERMAL_INFO


class RDSON(BaseModel):
    r_ds_on_25: float
    r_ds_on_100: float | None = None
    alpha_R: float = 0.008  # (0.8% / °C)

    @model_validator(mode="after")
    def compute_missing_thermal_values(self) -> "RDSON":
        # NOTE: if both r_ds_on_100 and a custom alpha_R are provided,
        # r_ds_on_100 wins and alpha_R is silently recomputed/overwritten.
        delta_t_ref = 100.0 - 25.0

        if self.r_ds_on_100 is not None:
            self.alpha_R = (self.r_ds_on_100 / self.r_ds_on_25 - 1.0) / delta_t_ref
        else:
            self.r_ds_on_100 = self.r_ds_on_25 * (1.0 + self.alpha_R * delta_t_ref)

        return self

    def calculate_at_temperature(self, target_temperature: float) -> float:
        """
        Calcule la R_DS(on) pour n'importe quelle température cible (en °C).
        Formule : R(T) = R(25) * (1 + alpha_R * (T - 25))
        """
        delta_t = target_temperature - 25.0
        return self.r_ds_on_25 * (1.0 + self.alpha_R * delta_t)


class GATE_CHARGE_INFO(BaseModel):
    q_g: float
    q_gs: float | None = None
    q_gd: float | None = None
    q_gth: float | None = None
    q_sw: float | None = None

    v_th: float
    v_plateau: float

    @model_validator(mode="after")
    def compute_missing_charge_values(self) -> "GATE_CHARGE_INFO":
        """
        Datasheets don't always give every charge component.
        Fill in what's missing from what's known:
          q_g ≈ q_gs + q_gd   (common simplification)
          q_sw ≈ q_gd         (switching charge ≈ Miller charge)
        """
        if self.q_gd is None and self.q_gs is not None:
            self.q_gd = self.q_g - self.q_gs
        elif self.q_gs is None and self.q_gd is not None:
            self.q_gs = self.q_g - self.q_gd

        if self.q_sw is None and self.q_gd is not None:
            self.q_sw = self.q_gd

        return self


class C_OSS_INFO(BaseModel):
    q_oss: float  # Output capacitance
    v_test_oss: float

    # Capacitance curves
    vds_points: list[float]  # VDS voltage points
    coss_points: list[float]  # Output capacitance points

    def coss_at_voltage(self, v_ds: float) -> float:
        """
        Interpolate Coss at any Vds from the datasheet curve.
        Uses log-log interpolation since Coss vs Vds is highly non-linear
        (Coss drops fast at low Vds, then flattens).

        Raises ValueError if v_ds falls outside the datasheet's measured
        range, instead of silently clamping to the nearest known point.
        """
        v_min, v_max = self.vds_points[0], self.vds_points[-1]
        if not (v_min <= v_ds <= v_max):
            raise ValueError(
                f"v_ds={v_ds} is outside the datasheet Coss curve range "
                f"[{v_min}, {v_max}]. Extrapolation is not supported."
            )

        log_vds = np.log(self.vds_points)
        log_coss = np.log(self.coss_points)
        log_v = np.log(v_ds)
        log_c = np.interp(log_v, log_vds, log_coss)
        return float(np.exp(log_c))

    def coss_effective_energy(self, v_on: float) -> float:
        """
        Energy-related effective Coss, used in P_oss = 0.5 * Coss_er * V_on^2 * f_sw.

        Coss_er(V) = (2 / V^2) * ∫[0 to V] Coss(v) * v dv
        """
        v_grid = np.linspace(self.vds_points[0], v_on, 200)
        coss_grid = np.array([self.coss_at_voltage(v) for v in v_grid])
        integral = np.trapezoid(coss_grid * v_grid, v_grid)
        return (2.0 / v_on**2) * integral

    def coss_effective_time(self, v_on: float) -> float:
        """
        Time-related effective Coss: C_o(tr)
        Used for charge/discharge TIME, e.g. ZVS transition or dead-time sizing:
            t_transition ≈ C_o(tr) * V_on / I_transition

        C_o(tr)(V) = (1 / V) * ∫[0 to V] Coss(v) dv
        """
        v_grid = np.linspace(self.vds_points[0], v_on, 200)
        coss_grid = np.array([self.coss_at_voltage(v) for v in v_grid])
        integral = np.trapezoid(coss_grid, v_grid)
        return integral / v_on

    def q_oss_at_voltage(self, v_on: float) -> float:
        """
        Actual stored charge at v_on — the direct integral.
        Q_oss(V) = ∫[0 to V] Coss(v) dv
        """
        v_grid = np.linspace(self.vds_points[0], v_on, 200)
        coss_grid = np.array([self.coss_at_voltage(v) for v in v_grid])
        return float(np.trapezoid(coss_grid, v_grid))


class BODY_DIODE(BaseModel):
    vf: float = 0.7
    q_rr_typ: float  # Reverse recovery charge, at the three reference conditions
    t_rr_typ: float  # Reverse recovery time

    # The datasheet test point. Q_rr is only meaningful together with all three.
    v_r_ref: float  # reverse voltage the charge was measured at [V]
    i_f_ref: float  # forward current before commutation [A]
    di_dt_ref: float  # commutation slope [A/s]

    # Exponents of the scaling law below. Defaults are the common engineering
    # values; they are device-dependent, so override them per part whenever the
    # datasheet gives a Q_rr curve to fit against.
    exp_di_dt: float = 0.5
    exp_v_r: float = 0.5
    exp_i_f: float = 0.5

    def q_rr_at_condition(
        self,
        di_dt: float,
        v_r: float | None = None,
        i_f: float | None = None,
    ) -> float:
        """
        Scale Q_rr away from the datasheet test point.

            Q_rr = Q_rr,typ * (di/dt / di/dt_ref)^a
                            * (V_R  / V_R,ref )^b
                            * (I_F  / I_F,ref )^c

        Q_rr is NOT a property of the diode alone — it is the stored charge the
        commutation manages to sweep out, and all three conditions move it:

        - **di/dt**: a faster commutation extracts more charge before it can
          recombine. Steepest, best-known dependence.
        - **V_R**: a higher reverse voltage sweeps harder and widens the
          depletion region, so more charge comes out. Corroboration for b = 0.5:
          E_rr = Q_rr * V_R then scales as V_R^1.5, close to the V^1.4 commonly
          reported for diode recovery energy.
        - **I_F**: more forward current means more stored minority carriers to
          begin with.

        v_r / i_f left as None skip their factor (pure di/dt scaling, the old
        behaviour). v_r = 0 correctly returns 0: with no reverse voltage there
        is nothing forcing a recovery — that is what ZVS buys you.

        di_dt : actual commutation slope [A/s]
        v_r   : actual reverse voltage across the diode after recovery [V]
        i_f   : actual forward current just before commutation [A]
        """
        factor = (di_dt / self.di_dt_ref) ** self.exp_di_dt
        if v_r is not None:
            factor *= (v_r / self.v_r_ref) ** self.exp_v_r
        if i_f is not None:
            factor *= (i_f / self.i_f_ref) ** self.exp_i_f
        return self.q_rr_typ * factor

    def t_rr_at_condition(self, di_dt: float) -> float:
        """
        Same di/dt scaling applied to the recovery TIME.

        Deliberately not extended to V_R and I_F: t_rr and Q_rr do not move
        together (a snappier diode recovers a similar charge in less time), so
        reusing the charge exponents here would not be justified.
        """
        return self.t_rr_typ * (di_dt / self.di_dt_ref) ** self.exp_di_dt


class MOSFET_SPEC(BaseModel):
    component_info: COMPONENT_INFO
    i_max: float
    v_dss_max: float

    r_ds_on: RDSON
    gate_charge: GATE_CHARGE_INFO
    c_oss: C_OSS_INFO
    body_diode: BODY_DIODE
    thermal: THERMAL_INFO

    r_g_int: float = 0


def load_mosfet(part_number: str) -> MOSFET_SPEC:
    return MOSFET_SPEC(**MOSFET_LIBRARY[part_number])


MOSFET_LIBRARY: dict[str, dict] = {
    "BSC016N06NS": {
        "component_info": {
            "part_number": "BSC016N06NS",
            "manufacturer": "Infineon",
            "package": "SuperSO8",
            "packages_available": ["SuperSO8"],
            "cost": 1.2,
        },
        "i_max": 234.0,
        "v_dss_max": 60.0,
        "r_ds_on": {
            "r_ds_on_25": 0.0016,
            "alpha_R": 0.005,
        },
        "gate_charge": {
            "q_g": 7.1e-08,
            "q_gs": 1.5e-08,
            "q_gd": 1.7e-08,
            "q_gth": 4e-09,
            "q_sw": 2.1e-08,
            "v_th": 2.0,
            "v_plateau": 4.3,
        },
        "c_oss": {
            "q_oss": 8.1e-08,
            "v_test_oss": 30.0,
            "vds_points": [0.1, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "coss_points": [
                4.5e-09,
                3e-09,
                2.1e-09,
                1.4e-09,
                1.2e-09,
                1.1e-09,
                1.05e-09,
                1e-09,
            ],
        },
        "body_diode": {
            "vf": 0.9,
            "q_rr_typ": 7.8e-08,
            "t_rr_typ": 6.1e-08,
            "v_r_ref": 30.0,
            "i_f_ref": 50.0,
            "di_dt_ref": 100000000.0,
        },
        "thermal": {
            "r_thjc": [(0.5, "bottom"), (20.0, "top")],
            "r_thja": [50.0, "PCB FR4 6cm2 (one layer, 70 µm thick) copper area"],
            "t_j_max": 175.0,
        },
        "r_g_int": 2.9,
    },
    # more parts go here...
}
