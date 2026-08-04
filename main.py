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

    time = np.linspace(0, 5e-6, 1000)
    sin = np.sin(2 * np.pi * 200e3 * time) * 5
    sinusoid = ElectronicPeriodicSignal.from_breakpoints(
        name="sinusoid",
        times=time,
        values=sin,
    )

    figure = plot_time_domain(
        [triangle, sinusoid],
        title="Triangle vs sinusoid",
        unit="V",
        levels=False,
        theme=THEME,
        cycles=5,
    )

    section(2, "Triangle signal")
    save_figure(
        figure,
        OUTPUT / "triangle.png",
        formats=("png",),
    )
    figure2 = plot_spectrum(
        [triangle, sinusoid],
        title="Triangle vs sinusoid",
        unit="V",
        theme=THEME,
    )
    save_figure(
        figure2,
        OUTPUT / "triangle_spectrum.png",
        formats=("png",),
    )

    figure3 = plot_signal(
        triangle, title="Triangle vs sinusoid", unit="V", theme=THEME, fill=False
    )
    save_figure(
        figure3,
        OUTPUT / "triangle_signal.png",
        formats=("png",),
    )

    dataframe(signal_table([triangle, sinusoid], unit="V"))


if __name__ == "__main__":
    main()
