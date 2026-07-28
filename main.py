"""
Demo run of the MOSFET loss model.

One BSC016N06NS driven by a UCC27714, hard switched on a 48 V bus.

Run from the project root:  python main.py
Figures land in OUTPUT/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no interactive window needed

from DATABASE.db_driver_mosfet import load_driver
from DATABASE.db_mosfet import load_mosfet
from SRC.MOSFET.mosfet_loss import (
    OPERATING_POINT,
    loss_single_mosfet_at_temp,
    loss_thermal_iteration,
)
from SRC.MOSFET.mosfet_plot import (
    loss_table,
    plot_loss_breakdown,
    plot_thermal_iteration,
)
from SRC.OTHERS.plot import *

import numpy as np

OUTPUT = Path(__file__).parent / "OUTPUT"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    main()
