from pydantic import BaseModel


class COMPONENT_INFO(BaseModel):
    part_number: str
    manufacturer: str
    package: str
    packages_available: list[str]
    cost: float


class THERMAL_INFO(BaseModel):
    r_thjc: list[tuple[float, str]]  # Junction to case + description
    r_thja: tuple[
        float, str
    ]  # Junction to ambient + description (e.g., (40.0, "Junction to ambient"))
    t_j_max: float  # Maximum junction temperature

    def r_thja_value(
        self, external_paths: list[tuple[float, str]] | None = None
    ) -> float:
        """
        Calculates the global thermal resistance (Junction-to-Ambient) across multiple parallel cooling paths.

        Each external cooling path is placed in series with its corresponding internal Junction-to-Case
        resistance (matched by the string identifier). All resulting complete branches
        (Junction-to-Ambient) are then calculated in parallel.

        Args:
            external_paths: List of tuples (r_th_ext, path_id), e.g., [(15.0, "top"), (5.0, "bottom")]

        Returns:
            The total equivalent Junction-to-Ambient thermal resistance in [°C/W].
        """
        # Extract the default datasheet Junction-to-Ambient value
        bare_r_thja = float(self.r_thja[0])

        if not external_paths:
            return bare_r_thja

        thermal_conductance_sum = 0.0
        seen_paths = set()

        for r_ext, path_id in external_paths:
            if path_id in seen_paths:
                raise ValueError(
                    f"Duplicate external path provided for '{path_id}'. Only one allowed per path."
                )
            seen_paths.add(path_id)

            # Retrieve the corresponding internal RthJC using your existing method.
            r_int = self.r_thjc_value(side=path_id)

            # The total resistance of this specific branch (Junction -> Case -> Ambient)
            r_branch_total = r_int + r_ext

            if r_branch_total <= 0.0:
                # Perfect cooling on this branch means 0 total resistance overall
                return 0.0

            # Add this branch's thermal conductance to the total parallel sum
            thermal_conductance_sum += 1.0 / r_branch_total

        if thermal_conductance_sum == 0.0:
            return bare_r_thja

        # Total equivalent resistance is the inverse of the sum of conductances
        return 1.0 / thermal_conductance_sum

    def r_thjc_value(self, side: str = "bottom") -> float:
        """
        Junction-to-case resistance for a given side ("bottom", "top", ...).
        """
        for value, description in self.r_thjc:
            if description == side:
                return float(value)
        available = [d for _, d in self.r_thjc]
        raise ValueError(f"r_thjc has no entry for '{side}', available: {available}")
