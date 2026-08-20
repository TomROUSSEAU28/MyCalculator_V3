from pydantic import BaseModel, model_validator, ConfigDict
import pandas as pd

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


class ConverterConstraints(BaseModel):
    pass

class ConverterDesign(BaseModel):
    pass

class ConverterDesignResult(BaseModel):
    pass



class BuckSpec(ConverterSpec):
    vin: float | ValueTolerance
    vout: float | ValueTolerance
    pout: float | ValueTolerance
    switching_frequency: float | ValueTolerance | None = None


class BuckConstraints(ConverterConstraints):
    max_current_ripple_percent: float | None = None
    l: ValueConstraints | None = None
    c: ValueConstraints | None = None


class BuckDesign(ConverterDesign):
    spec: BuckSpec
    constraints: BuckConstraints | None = None
    diode_vf: float | None = None

class BuckDesignResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    solutions: pd.DataFrame

    # def run(self) -> BuckDesignResult:
    #     raise NotImplementedError



if __name__ == "__main__":
    buck_spec = BuckSpec(
        vin=12,
        vout=ValueTolerance.from_percentage(nominal=5, percentage=25),
        pout=10,
        switching_frequency=100000,
    )

    buck_constraints = BuckConstraints(
        max_current_ripple_percent=50,
    )

    buck_design = BuckDesign(
        spec=buck_spec,
        constraints=buck_constraints,
        diode_vf=0.7,
    )

    print(buck_design)
