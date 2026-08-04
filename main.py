import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no interactive window needed

import numpy as np

from SRC.OTHERS.plot import *
from SRC.OTHERS.terminal import *
from SRC.SIGNAL_PROCESSING.signal import *
from SRC.SIGNAL_PROCESSING.signal_plot import *

OUTPUT = Path(__file__).parent / "OUTPUT"

# One name for both palettes: SRC/OTHERS/plot.py and SRC/OTHERS/terminal.py
# carry the same theme names on purpose, so the report and the figures match.
THEME = "light"


# ============================================================================ #
#  Design point
# ============================================================================ #


def main():
    OUTPUT.mkdir(exist_ok=True)
    use_theme(THEME)

    triangle = ElectronicPeriodicSignal.from_breakpoints(
        name="triangle", times=[0, 1e-6, 2e-6], values=[0, 1, 0]
    )
    section(2, "Triangle signal")
    dataframe(signal_table(triangle, "V"))
    dataframe(harmonics_table(triangle, "V", n_max=100))


if __name__ == "__main__":
    main()
