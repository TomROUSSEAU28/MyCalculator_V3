from enum import Enum

from pydantic import BaseModel, Field

from DATABASE.db_component import COMPONENT_INFO


# source_sink
class DRIVING_MODEL_ENUM(Enum):
    """
    How the driver behaves on each edge.

    Naming is <turn_on>_<turn_off>:
      SOURCE = ideal current source (forces the current, R_g,total irrelevant)
      PEAK   = resistive output, clamped at the driver peak current
    """

    SOURCE_SOURCE = 1
    SOURCE_PEAK = 2
    PEAK_SOURCE = 3
    PEAK_PEAK = 4


class DRIVING_MODE(BaseModel):
    kind: DRIVING_MODEL_ENUM = Field(default=DRIVING_MODEL_ENUM.PEAK_PEAK)
    current_info: tuple[float, float]  # (turn-on current [A], turn-off current [A])


class DRIVIER_TYPE(Enum):
    HALF_BRIDGE = 1
    HIGH_SIDE = 2
    LOW_SIDE = 3


class DRIVER_MOSFET_SPEC(BaseModel):
    """Gate driver specification for power MOSFETs."""

    component_info: COMPONENT_INFO
    driver_type: DRIVIER_TYPE
    v_on: float
    v_off: float
    r_out_source: float | None = None  # output resistance, turn-on (source) path
    r_in_source: float | None = None  # output resistance, turn-off (sink) path
    Current_information: DRIVING_MODE


def load_driver(part_number: str) -> DRIVER_MOSFET_SPEC:
    return DRIVER_MOSFET_SPEC(**DRIVER_LIBRARY[part_number])


DRIVER_LIBRARY: dict[str, dict] = {
    "UCC27714": {
        "component_info": {
            "part_number": "UCC27714",
            "manufacturer": "Texas Instruments",
            "package": "SOIC-14",
            "packages_available": ["SOIC-14"],
            "cost": 2.1,
        },
        "driver_type": DRIVIER_TYPE.HALF_BRIDGE,
        # 600 V half-bridge driver, VDD 10-20 V. v_off = 0: no negative
        # turn-off rail, which is why the sink path is the slower one.
        "v_on": 12.0,
        "v_off": 0.0,
        # Output resistance is asymmetric by design — the pull-down is much
        # stronger than the pull-up, to hold the off device off against dv/dt.
        "r_out_source": 4.0,
        "r_in_source": 1.0,
        "Current_information": {
            "kind": DRIVING_MODEL_ENUM.PEAK_PEAK,
            "current_info": (4.0, 4.0),  # 4 A source / 4 A sink peak
        },
    },
    # more parts go here...
}
