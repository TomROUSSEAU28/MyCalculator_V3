"""
Demo run of the MOSFET loss model.

One BSC016N06NS driven by a UCC27714, hard switched on a 48 V bus.

Run from the project root:  python main.py
Figures land in OUTPUT/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no interactive window needed

import numpy as np

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
from SRC.OTHERS.terminal import *
from SRC.SIGNAL_PROCESSING.signal import *
from SRC.SIGNAL_PROCESSING.signal_plot import *

OUTPUT = Path(__file__).parent / "OUTPUT"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)

    # triangle 10uS
    signal = ElectronicPeriodicSignal.from_breakpoints(
        "triangle", [0.0, 5e-6, 10e-6], [0.0, 1.0, 0.0], n_samples=2048
    )
    fig = plot_periodic_signal_time_domain(
        signal, title="Triangle Signal", subtitle="10 μs period", theme="dark"
    )
    save_figure(fig, OUTPUT / "triangle_signal_time_domain")


if __name__ == "__main__":
    main()
