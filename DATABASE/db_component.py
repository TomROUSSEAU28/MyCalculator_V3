from pydantic import BaseModel


class COMPONENT_INFO(BaseModel):
    part_number: str
    manufacturer: str
    package: str
    packages_available: list[str]
    cost: float


class THERMAL_INFO(BaseModel):
    r_thjc: list[tuple[float, str]]  # Junction to case + description
    r_thja: list[float | str]  # Junction to ambient + description
    t_j_max: float  # Maximum junction temperature

    def r_thja_value(self) -> float:
        """Numeric part of r_thja (the rest of the list is the description)."""
        for entry in self.r_thja:
            if isinstance(entry, (int, float)):
                return float(entry)
        raise ValueError("r_thja contains no numeric value")

    def r_thjc_value(self, side: str = "bottom") -> float:
        """
        Junction-to-case resistance for a given side ("bottom", "top", ...).
        """
        for value, description in self.r_thjc:
            if description == side:
                return float(value)
        available = [d for _, d in self.r_thjc]
        raise ValueError(f"r_thjc has no entry for '{side}', available: {available}")
