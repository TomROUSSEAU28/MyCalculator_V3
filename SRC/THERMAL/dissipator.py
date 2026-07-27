from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional  # noqa: UP035

import numpy as np
from pydantic import BaseModel, Field, model_validator

# ================================================================== #
#  Dissipator.py — Modelling of external heat sinks                   #
# ================================================================== #
#
#  Usable with any component that has a thermal pad:
#    MOSFET, IGBT, diode, planar magnetic (inductor/transformer),
#    linear regulator, gate driver, etc.
#
#  Class hierarchy:
#
#    Dissipator (ABC)
#      ├── StandardDissipator   — R_th given directly (heat sink datasheet)
#      └── PCBDissipator        — computed from the PCB copper plane geometry
#
#  Convention: get_rth() returns ONLY the resistance EXTERNAL to the package.
#  The component's R_thJC adds in series, and is handled by each component's
#  own thermal method (e.g. MOSFET.get_effective_rth(), or its equivalent).
#
#  Integration (example with a MOSFET, identical for any component):
#    pcb = PCBDissipator(...)
#    mos.solve_thermal_equilibrium(op, T_amb=25, **pcb.to_mosfet_kwargs())
#
#    For several heat sinks (dual-side, or plane + heat sink):
#    mos.solve_thermal_equilibrium(
#        op, T_amb=25,
#        **Dissipator.combine_to_mosfet_kwargs(pcb_bot, heatsink_top),
#    )
# ================================================================== #


class Placement(str, Enum):
    """
    Side of the component package the heat sink is attached to.

    BOTTOM : bottom pad (typical case: D-PAK, D2PAK, PowerPAK, TO-220, planar core)
    TOP    : top of the package (top-cool case: some GaN packages, top heat sink)
    """

    BOTTOM = "bottom"
    TOP = "top"


# ================================================================== #
#  Dissipator (abstract base)                                         #
# ================================================================== #


class Dissipator(BaseModel, ABC):
    """
    Abstract base for any heat sink EXTERNAL to a component package.

    Applies to any component with a thermal pad: MOSFET, IGBT, diode, planar
    inductor / transformer, linear regulator, gate driver…

    Convention:
      get_rth() returns only the resistance external to the package, i.e. the
      resistance seen by the component pad on its way to the ambient air. The
      component's R_thJC adds in series, and is handled by the thermal method
      of the component itself (e.g. MOSFET.get_effective_rth(), or the
      equivalent in other classes).
    """

    name: str = Field(description="Heat sink name")
    placement: Placement = Field(
        default=Placement.BOTTOM,
        description="Side of the package the heat sink is attached to",
    )

    @abstractmethod
    def get_rth(self) -> float:
        """Return the external R_th [K/W] seen from the package on side `placement`."""
        ...


# ================================================================== #
#  StandardDissipator                                                 #
# ================================================================== #


class StandardDissipator(Dissipator):
    """
    Heat sink whose thermal resistance is given directly (manufacturer
    datasheet). Typical case: extruded aluminium heat sink, encapsulated
    thermal pad, thermoelectric module.

    Thermal model (series):

        [ MOSFET T_case ]
               │
           [ R_tim ]   ← TIM: silicone grease, SilPad, insulating pad...
               │           Typical value: 0.5-5 K/W depending on area and thickness
               │           Set to 0 if R_tim is already included in the datasheet R_th
           [ R_th  ]   ← Heat sink resistance (manufacturer datasheet)
               │
        [ T_ambient ]

        get_rth() = R_tim + R_th
    """

    R_th: float = Field(gt=0, description="Heat sink thermal resistance [K/W]")
    R_tim: float = Field(
        default=0.0,
        ge=0,
        description="Thermal resistance of the TIM (grease, pad) in series [K/W]",
    )

    def get_rth(self) -> float:
        return self.R_th + self.R_tim


# ================================================================== #
#  PCBDissipator                                                      #
# ================================================================== #

#                       [ Heat from the component ]
#                                  │
#                              ( T_Case )  <-- Starting point (the pad)
#                                  │
#            ┌─────────────────────┴─────────────────────┐
#            │                                           │
#      ** PATH A **                                ** PATH B **
#  (direct dissipation)                     (crossing to the opposite side)
#            │                                           │
#            │                             ┌─────────────┴─────────────┐
#            │                             │                           │
#      [ R_conv_side ]                [ R_pcb_thru ]               [ R_vias ]
#   (air on top conductor)             (bare substrate)          (thermal vias)
#            │                             │                           │
#            │                             └─────────────┬─────────────┘
#            │                                           │
#            │                                  ( T_Copper_Opposite )
#            │                                           │
#            │                                   [ R_conv_other ]
#            │                              (air on the bottom copper)
#            │                                           │
#            └─────────────────────┬─────────────────────┘
#                                  │
#                             ( T_Ambient )


class PCBDissipator(Dissipator):
    """
    Dissipation through a conductive plane on a substrate (PCB, IMS, DBC,
    ceramic...).

    First-order model, 2 thermal paths in parallel between the component pad
    and the ambient air:

        Path A : pad → conductive plane, same side → air   (direct convection)
        Path B : pad → (bare substrate ∥ vias) → opposite plane → air

    Assumptions:
      - The conductive plane is treated as isothermal; spreading resistance is
        neglected as long as A_cu ≲ 10 cm². Beyond that, the computed R_th is
        underestimated.
      - Natural convection and radiation are lumped together in h_conv_eff_W_m2K.
        Typical: 10-15 W/(m²·K) natural; 30-100 W/(m²·K) with forced airflow.
      - No intermediate inner plane is modelled. If you have a dedicated inner
        plane, treat it as an opposite plane (it reduces the effective substrate
        thickness).

    Common substrate / conductor combinations:
      ┌────────────────────┬───────────────┬──────────────────┬──────────────────┐
      │ Technology         │ pcb_k [W/mK]  │ k_conductor      │ Use case         │
      ├────────────────────┼───────────────┼──────────────────┼──────────────────┤
      │ Standard FR4       │ 0.3           │ 385 (Cu)         │ General, low     │
      │                    │               │                  │ density          │
      ├────────────────────┼───────────────┼──────────────────┼──────────────────┤
      │ IMS (alu base)     │ 1 - 3         │ 385 (Cu)         │ LED, DC/DC conv. │
      │                    │ (dielectric)  │                  │ medium density   │
      ├────────────────────┼───────────────┼──────────────────┼──────────────────┤
      │ DBC Al₂O₃          │ 25            │ 385 (Cu)         │ IGBT/SiC modules │
      │                    │               │                  │ high power       │
      ├────────────────────┼───────────────┼──────────────────┼──────────────────┤
      │ DBC AlN            │ 170           │ 385 (Cu)         │ High density,    │
      │                    │               │                  │ SiC/GaN          │
      └────────────────────┴───────────────┴──────────────────┴──────────────────┘

    Orders of magnitude (h=12 W/m²K, 4 cm² each side):
      - FR4 1.6 mm, no vias            → ~100-150 K/W
      - FR4 1.6 mm, 16 vias Ø0.3 mm   → ~20-30 K/W
      - IMS 0.1 mm dielectric          → ~5-15 K/W
      - DBC Al₂O₃ 0.38 mm             → ~2-5 K/W
    """

    # --- Copper geometry ---
    A_cu_total_side_cm2: float = Field(
        gt=0, description="Total copper area on the same side (whole plane) [cm²]"
    )
    A_cu_other_side_cm2: Optional[float] = Field(
        default=None,
        description="Copper area on the opposite side exposed to air [cm²]. None = no opposite plane",
    )
    A_pad_mosfet_cm2: float = Field(
        gt=0,
        description="Area of the MOSFET thermal pad (for direct FR4 conduction) [cm²]",
    )
    copper_thickness_um: float = Field(
        default=35.0,
        gt=0,
        description=(
            "Copper plane thickness [µm]. "
            "1 oz/ft² = 35 µm  |  2 oz = 70 µm  |  3 oz = 105 µm. "
            "Sets the effective spreading radius: r_max = 1.5 cm × √(t_cu / 35 µm)."
        ),
    )

    # --- Substrate ---
    pcb_thickness_mm: float = Field(
        default=1.6, gt=0, description="Substrate thickness [mm]"
    )
    pcb_k_W_mK: float = Field(
        default=0.3,
        gt=0,
        description=(
            "Thermal conductivity of the SUBSTRATE [W/(m·K)]. "
            "FR4≈0.3  |  IMS dielectric≈1-3  |  Al2O3≈25  |  AlN≈170"
        ),
    )

    # --- Conductor (planes + vias) ---
    k_conductor_W_mK: float = Field(
        default=385.0,
        gt=0,
        description=(
            "Thermal conductivity of the conductor (planes and vias) [W/(m·K)]. "
            "Copper≈385  |  Aluminium≈205  |  Silver≈429"
        ),
    )

    # Effective and exposed areas — computed automatically by _compute_exposed_area.
    # A_cu_effective_side_cm2 is the useful area on the pad side, capped by the
    # spreading radius: beyond r_max, the extra copper no longer takes part in the
    # dissipation (the heat does not reach it). A_cu_exposed_side_cm2 derives from it
    # the area actually in contact with air (effective − pad).
    A_cu_effective_side_cm2: float = Field(
        default=0.0,
        description=(
            "Effective copper area on the pad side, limited by the spreading radius [cm²]. "
            "= min(A_cu_total_side_cm2, A_pad_mosfet + π·r_max²) — computed automatically."
        ),
    )
    A_cu_exposed_side_cm2: float = Field(
        default=0.0,
        description=(
            "Copper area exposed to air on the pad side [cm²]. "
            "= A_cu_effective_side_cm2 − A_pad_mosfet_cm2 — computed automatically."
        ),
    )

    @model_validator(mode="after")
    def _compute_exposed_area(self) -> "PCBDissipator":
        # Maximum spreading radius: empirical law based on the copper thickness.
        # For 1 oz (35 µm), r_max = 1.5 cm; for 2 oz (70 µm), r_max ≈ 2.1 cm.
        r_max_cm = 1.5 * np.sqrt(self.copper_thickness_um / 35.0)
        A_max_cm2 = self.A_pad_mosfet_cm2 + np.pi * r_max_cm**2

        # Effective area on the pad side: capped by the spreading
        self.A_cu_effective_side_cm2 = min(self.A_cu_total_side_cm2, A_max_cm2)

        exposed = self.A_cu_effective_side_cm2 - self.A_pad_mosfet_cm2
        if exposed <= 0:
            raise ValueError(
                f"A_cu_effective_side ({self.A_cu_effective_side_cm2:.2f} cm²) must be "
                f"> A_pad_mosfet ({self.A_pad_mosfet_cm2} cm²)"
            )
        self.A_cu_exposed_side_cm2 = exposed

        # Same cap on the opposite plane: the heat reaches it from the pad through
        # the vias, so its useful area is limited by A_max in exactly the same way.
        if self.A_cu_other_side_cm2 is not None:
            self.A_cu_other_side_cm2 = min(self.A_cu_other_side_cm2, A_max_cm2)

        return self

    # --- Thermal vias ---
    n_vias: int = Field(
        default=0, ge=0, description="Number of thermal vias under the pad"
    )
    via_diameter_mm: float = Field(default=0.3, gt=0, description="Via diameter [mm]")
    via_plating_um: float = Field(
        default=25.0, gt=0, description="Copper plating thickness of the vias [μm]"
    )
    via_filled: bool = Field(
        default=False,
        description="True for copper / conductive resin filled vias, False for plating only",
    )

    #   ┌──────────────────────────────────────────────────────────────┬──────────────────┐
    #   │                          Situation                           │      h_conv      │
    #   ├──────────────────────────────────────────────────────────────┼──────────────────┤
    #   │ PCB horizontal, face up, no airflow                          │ ~5-8 W/(m²·K)    │
    #   ├──────────────────────────────────────────────────────────────┼──────────────────┤
    #   │ PCB vertical, or in an enclosure with natural circulation    │ ~8-12 W/(m²·K)   │
    #   ├──────────────────────────────────────────────────────────────┼──────────────────┤
    #   │ Light fan (low airflow inside the case)                      │ ~20-40 W/(m²·K)  │
    #   ├──────────────────────────────────────────────────────────────┼──────────────────┤
    #   │ Direct forced airflow                                        │ ~50-100 W/(m²·K) │
    #   └──────────────────────────────────────────────────────────────┴──────────────────┘
    # --- Convection ---
    h_conv_eff_W_m2K: float = Field(
        default=12.0,
        gt=0,
        description=(
            "EFFECTIVE convection coefficient (convection + radiation) [W/(m²·K)]. "
            "Natural: 10-15; forced, light fan: 30-50; forced, blown: 50-100"
        ),
    )

    # ============================================================== #
    #  Copper conversion helpers                                     #
    # ============================================================== #

    @staticmethod
    def oz_to_um(oz: float) -> float:
        """Convert a copper weight into a thickness [µm]. Basis: 1 oz/ft² = 35 µm."""
        return oz * 35.0

    @staticmethod
    def um_to_oz(um: float) -> float:
        """Convert a copper thickness [µm] into a weight [oz/ft²]. Basis: 1 oz/ft² = 35 µm."""
        return um / 35.0

    # ============================================================== #
    #  Computation                                                   #
    # ============================================================== #

    def _R_conv(self, A_cm2: float) -> float:
        """Convection resistance to air for a given area [K/W]."""
        return 1.0 / (self.h_conv_eff_W_m2K * (A_cm2 * 1e-4))

    def _R_pcb_thru(self) -> float:
        """Heat conduction through the substrate under the pad (bare FR4) [K/W]."""
        e_pcb = self.pcb_thickness_mm * 1e-3
        A_pad = self.A_pad_mosfet_cm2 * 1e-4
        return e_pcb / (self.pcb_k_W_mK * A_pad)

    def _R_vias(self) -> float:
        """Conduction through the thermal vias in parallel [K/W]."""
        if self.n_vias <= 0:
            return float("inf")
        D = self.via_diameter_mm * 1e-3
        if self.via_filled:
            A_one = np.pi * (D / 2.0) ** 2
        else:
            t = self.via_plating_um * 1e-6
            A_one = np.pi * ((D / 2.0) ** 2 - max(D / 2.0 - t, 0.0) ** 2)
        A_tot = self.n_vias * A_one
        e_pcb = self.pcb_thickness_mm * 1e-3
        return e_pcb / (self.k_conductor_W_mK * A_tot)

    def get_rth(self) -> float:
        # Path A: direct convection on the pad side
        R_conv_side = self._R_conv(self.A_cu_exposed_side_cm2)

        if not self.A_cu_other_side_cm2 or self.A_cu_other_side_cm2 == 0:
            # No opposite plane: path A is the only one dissipating
            return R_conv_side

        # Path B: through-PCB → opposite plane → convection
        R_pcb = self._R_pcb_thru()
        R_via = self._R_vias()
        # Bare FR4 and vias in parallel (two thermal paths through the PCB)
        R_through = (
            (R_pcb * R_via) / (R_pcb + R_via) if R_via != float("inf") else R_pcb
        )

        R_conv_other = self._R_conv(self.A_cu_other_side_cm2)
        R_path_B = R_through + R_conv_other

        # Parallel between path A (top) and path B (bottom)
        return (R_conv_side * R_path_B) / (R_conv_side + R_path_B)

    def breakdown(self) -> Dict[str, float]:
        """
        Detail the thermal components [K/W], for debugging.

        Shows what is limiting:
          - If R_conv_side >> R_through+R_conv_other: the top surface is the limit
          - If R_through >> R_conv: add vias, or increase their diameter
          - If R_vias >> R_conv: the vias are already good, convection is the limit
          - If A_cu_effective < A_cu_total: spreading is the limit (copper underused)

        Returns:
            dict with keys: r_spreading_max_cm, A_cu_effective_side_cm2,
            A_cu_exposed_side_cm2, R_conv_side, R_pcb_thru, R_vias, R_through,
            R_conv_other, R_path_B, R_total
        """
        r_max_cm = 1.5 * np.sqrt(self.copper_thickness_um / 35.0)
        R_conv_side = self._R_conv(self.A_cu_exposed_side_cm2)
        out: Dict[str, float] = {
            "r_spreading_max_cm": r_max_cm,
            "A_cu_effective_side_cm2": self.A_cu_effective_side_cm2,
            "A_cu_exposed_side_cm2": self.A_cu_exposed_side_cm2,
            "R_conv_side": R_conv_side,
        }

        if not self.A_cu_other_side_cm2:
            out["R_total"] = R_conv_side
            return out

        R_pcb = self._R_pcb_thru()
        R_via = self._R_vias()
        R_through = (
            (R_pcb * R_via) / (R_pcb + R_via) if R_via != float("inf") else R_pcb
        )
        R_conv_other = self._R_conv(self.A_cu_other_side_cm2)
        R_path_B = R_through + R_conv_other
        R_total = (R_conv_side * R_path_B) / (R_conv_side + R_path_B)

        out.update(
            {
                "R_pcb_thru": R_pcb,
                "R_vias": R_via,
                "R_through": R_through,
                "R_conv_other": R_conv_other,
                "R_path_B": R_path_B,
                "R_total": R_total,
            }
        )
        return out
