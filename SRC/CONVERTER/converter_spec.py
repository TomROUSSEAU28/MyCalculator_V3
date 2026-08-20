import sys
from enum import Enum
from math import pi, sqrt
from pathlib import Path
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ValueTolerance(BaseModel):
    nominal: float
    min_val: float
    max_val: float

    @model_validator(mode='after')
    def check_logic(self):
        if self.min_val >= self.max_val:
            raise ValueError("min_val doit être inférieur à max_val")
        if self.nominal <= 0:
            raise ValueError("nominal doit être strictement positif")
        return self

    @classmethod
    def from_percentage(cls, nominal: float, percentage: float) -> "ValueTolerance" :
        if percentage < 0 and percentage > 100:
            raise ValueError("percentage doit être positif et inférieur à 100")
        min_val = nominal * (1 - percentage / 100)
        max_val = nominal * (1 + percentage / 100)
        return cls(nominal=nominal, min_val=min_val, max_val=max_val)

class ValueConstraints(BaseModel):
    min_val: float
    max_val: float

    @model_validator(mode='after')
    def check_logic(self):
        if self.min_val >= self.max_val:
            raise ValueError("min_val doit être inférieur à max_val")
        return self

class ConverterSpec(BaseModel):
    pass
class ConverterConstraints(BaseModel):
    pass
class ConverterDesign(BaseModel):
    pass

class BuckSpec(ConverterSpec):
    Vin: float | ValueTolerance
    Vout: float | ValueTolerance
    Pout: float | ValueTolerance

class BuckConstraints(ConverterConstraints):
    l : ValueConstraints | None
    c : ValueConstraints | None



if __name__ == "__main__":
    buck_spec = BuckSpec(Vin=12, Vout=ValueTolerance.from_percentage(5, 25), Pout=10)
    print(buck_spec)
