from itertools import product
from typing import override

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

# Règles du pouce utilisées quand les contraintes ne sont pas fournies.
DEFAULT_CURRENT_RIPPLE_PERCENT = 30.0
DEFAULT_VOLTAGE_RIPPLE_PERCENT = 1.0

class ValueTolerance(BaseModel):
    nominal: float
    min_val: float
    max_val: float

    @model_validator(mode='after')
    def check_logic(self):
        if self.min_val >= self.max_val:
            raise ValueError("min_val doit être strictement inférieur à max_val")
        if self.nominal <= 0:
            raise ValueError("nominal doit être strictement positif")
        return self

    @classmethod
    def from_percentage(cls, nominal: float, percentage: float) -> "ValueTolerance":
        if percentage < 0 or percentage > 100:
            raise ValueError("percentage doit être compris entre 0 et 100")
        min_val = nominal * (1 - percentage / 100)
        max_val = nominal * (1 + percentage / 100)
        return cls(nominal=nominal, min_val=min_val, max_val=max_val)


class ValueConstraints(BaseModel):
    min_val: float
    max_val: float

    @model_validator(mode='after')
    def check_logic(self):
        if self.min_val > self.max_val:
            raise ValueError("min_val doit être inférieur ou égal à max_val")
        return self

    @classmethod
    def from_fixed_constraint(cls, fixed_value: float) -> "ValueConstraints":
        return cls(min_val=fixed_value, max_val=fixed_value)


class ConverterSpec(BaseModel):

    @staticmethod
    def pout_from_vout_iout(vout: float, iout: float) -> float:
        return vout * iout

    def get_spec(self) -> dict[str, float | ValueTolerance]:
        return self.model_dump()

    def get_keys(self) -> tuple[str, ...]:
        return tuple(type(self).model_fields.keys())

    def get_values(self) -> tuple[float | ValueTolerance, ...]:
        return tuple(getattr(self, k) for k in type(self).model_fields.keys())


class ConverterConstraints(BaseModel):
    def get_constraints(self) -> dict[str, float | ValueConstraints | bool]:
        return self.model_dump()

    def get_keys(self) -> tuple[str, ...]:
        return tuple(type(self).model_fields.keys())

    def get_values(self) -> tuple[float | ValueTolerance | bool, ...]:
        return tuple(getattr(self, k) for k in type(self).model_fields.keys())

class ConverterDesign(BaseModel):

    spec: ConverterSpec
    constraints: ConverterConstraints | None = None

    @staticmethod
    def _corners(value: float | ValueTolerance) -> list[float]:
        """
        Valeurs à balayer pour une entrée du cahier des charges.

        float          -> [value]
        ValueTolerance -> [min_val, nominal, max_val] (doublons retirés, trié)
        """
        if isinstance(value, ValueTolerance):
            return sorted({value.min_val, value.nominal, value.max_val})
        return [float(value)]

    def _corner_is_valid(self, values: tuple[float, ...]) -> bool:
        return True

    def _operating_points(self) -> list[dict[str, float]]:
        """
        Produit cartésien des coins de tolérance de vin, vout, pout et f_sw.
        Une entrée = un point de fonctionnement à dimensionner.
        """
        spec = self.spec

        keys = spec.get_keys()
        values = spec.get_values()
        corners = (
            self._corners(value) for value in values
        )
        return [
            dict(zip(keys, values))
            for values in product(*corners)
            if self._corner_is_valid(values)
        ]

    def _solve(self, point: dict[str, float]) -> dict[str, float | bool]:
        return {"Empty, need to be implemented in the subclass" : False}

    def run(self) -> "ConverterDesignResult":
        """Dimensionne tous les coins de tolérance et renvoie le résultat."""
        rows = [self._solve(point) for point in self._operating_points()]
        return ConverterDesignResult(solutions=pd.DataFrame(rows))

class ConverterDesignResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    solutions: pd.DataFrame


class BuckSpec(ConverterSpec):
    vin: float | ValueTolerance
    vout: float | ValueTolerance
    pout: float | ValueTolerance
    switching_frequency: float | ValueTolerance


class BuckConstraints(ConverterConstraints):
    max_current_ripple_percent: float | None = None
    max_voltage_ripple_percent: float | None = None
    l: ValueConstraints | None = None
    c: ValueConstraints | None = None


class BuckDesign(ConverterDesign):
    diode_vf: float | None = None

    # region helpers
    @override
    def _corner_is_valid(self, values: tuple[float, ...]) -> bool:
        # vin > vout (modifié pour correspondre à la condition stricte de _duty_cycle)
        if values[0] > values[1]:

            return True
        print("-"*20)
        print(f"Invalid corner: vin={values[0]} vout={values[1]}, vin must be greater than vout for a buck converter")
        print("-"*20)
        return False

    def _duty_cycle(self, vin: float, vout: float) -> float:
        """
        Rapport cyclique en CCM.
        Buck asynchrone (diode) : D = (Vout + Vf) / (Vin + Vf)
        Buck synchrone (Vf None): D = Vout / Vin
        """
        if vin <= vout:
            raise ValueError(f"vin ({vin}) doit être strictement supérieur à vout ({vout})")
        v_f = self.diode_vf or 0.0
        return (vout + v_f) / (vin + v_f)

    @staticmethod
    def _output_current(pout: float, vout: float) -> float:
        """I_out = P_out / V_out [A]"""
        return pout / vout

    def _current_ripple(self, iout: float) -> float:
        """
        Ondulation crête-à-crête de l'inductance, imposée par la contrainte.
        delta_IL = ripple% * I_out
        """
        percent = DEFAULT_CURRENT_RIPPLE_PERCENT
        if self.constraints is not None and self.constraints.max_current_ripple_percent is not None:
            percent = self.constraints.max_current_ripple_percent
        return iout * percent / 100.0

    def _voltage_ripple(self, vout: float) -> float:
        """
        Ondulation crête-à-crête admissible sur la sortie.
        delta_Vout = ripple% * V_out
        """
        percent = DEFAULT_VOLTAGE_RIPPLE_PERCENT
        if self.constraints is not None and self.constraints.max_voltage_ripple_percent is not None:
            percent = self.constraints.max_voltage_ripple_percent
        return vout * percent / 100.0

    @staticmethod
    def _inductance(vin: float, vout: float, d: float, f_sw: float, delta_il: float) -> float:
        """
        Inductance minimale pour tenir l'ondulation de courant.
        L = (Vin - Vout) * D / (f_sw * delta_IL)
        """
        return (vin - vout) * d / (f_sw * delta_il)

    @staticmethod
    def _boundary_inductance(vout: float, iout: float, d: float, f_sw: float) -> float:
        """
        Inductance à la limite CCM/DCM.
        L_ccm = (1 - D) * Vout / (2 * f_sw * I_out)
        """
        return (1.0 - d) * vout / (2.0 * f_sw * iout)

    @staticmethod
    def _capacitance(delta_il: float, f_sw: float, delta_vout: float) -> float:
        """
        Capacité de sortie minimale (contribution capacitive seule, ESR négligée).
        C = delta_IL / (8 * f_sw * delta_Vout)
        """
        return delta_il / (8.0 * f_sw * delta_vout)

    @staticmethod
    def _rms_current(iout: float, delta_il: float, duty_fraction: float) -> float:
        """
        Courant efficace d'un segment trapézoïdal conduisant pendant duty_fraction.
        I_rms = sqrt(duty * (I_out^2 + delta_IL^2 / 12))
        """
        return (duty_fraction * (iout**2 + delta_il**2 / 12.0)) ** 0.5

    def _check_constraints(self, l_min: float, c_min: float) -> dict[str, bool]:
        """
        Confronte L et C minimaux aux fenêtres de composants disponibles.
        Sans contrainte déclarée, le point est considéré réalisable.
        """
        l_ok = True
        c_ok = True
        if self.constraints is not None:
            if self.constraints.l is not None:
                l_ok = l_min <= self.constraints.l.max_val
            if self.constraints.c is not None:
                c_ok = c_min <= self.constraints.c.max_val
        return {"l_ok": l_ok, "c_ok": c_ok, "valid": l_ok and c_ok}

    @override
    def _solve(self, point: dict[str, float]) -> dict[str, float | bool]:
        """Dimensionne un point de fonctionnement -> une ligne du DataFrame."""
        vin, vout = point["vin"], point["vout"]
        pout, f_sw = point["pout"], point["switching_frequency"]

        d = self._duty_cycle(vin, vout)
        iout = self._output_current(pout, vout)
        delta_il = self._current_ripple(iout)
        delta_vout = self._voltage_ripple(vout)

        l_min = self._inductance(vin, vout, d, f_sw, delta_il)
        l_ccm = self._boundary_inductance(vout, iout, d, f_sw)
        c_min = self._capacitance(delta_il, f_sw, delta_vout)

        row: dict[str, float | bool] = {
            **point,
            "iout": iout,
            "duty": d,
            "delta_il": delta_il,
            "delta_vout": delta_vout,
            "il_rms": self._rms_current(iout, delta_il, 1.0),
            "il_pk": iout + delta_il / 2.0,
            "l_min": l_min,
            "l_ccm": l_ccm,
            "c_min": c_min,
            "isw_rms": self._rms_current(iout, delta_il, d),
            "idiode_rms": self._rms_current(iout, delta_il, 1.0 - d),
            "idiode_avg": iout * (1.0 - d),
            "ccm": iout >= delta_il / 2.0,
        }
        row.update(self._check_constraints(l_min, c_min))
        return row

    # endregion




class BuckDesignResult(ConverterDesignResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    solutions: pd.DataFrame


if __name__ == "__main__":
    buck_spec = BuckSpec(
        vin=12,
        vout=ValueTolerance.from_percentage(nominal=5, percentage=25),
        pout=10,
        switching_frequency=100000,
    )

    buck_constraints = BuckConstraints(
        max_current_ripple_percent=50,
        max_voltage_ripple_percent=1,
        l=ValueConstraints(min_val=1e-6, max_val=100e-6),
        c=ValueConstraints(min_val=1e-6, max_val=1e-3),
    )

    buck_design = BuckDesign(
        spec=buck_spec,
        constraints=buck_constraints,
        diode_vf=0.7,
    )

    result = buck_design.run()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(result.solutions)
