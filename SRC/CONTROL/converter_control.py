import logging
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

import control as ct

sys.path.append(str(Path(__file__).resolve().parents[2]))
from SRC.OTHERS.plot import save_figure
from SRC.OTHERS.terminal import kv, section, table, use_theme

logger = logging.getLogger(__name__)
OUTPUT = Path(__file__).resolve().parents[2] / "OUTPUT"
THEME = "light"
s = ct.tf("s")
TWO_PI = 2.0 * np.pi


if __name__ == "__main__":
    # https://python-control.readthedocs.io/en/0.10.2/functions.html#function-ref
    plant = 1 / (1 + s * 5)
    freq = np.linspace(10, 10000)
    ct.bode_plot(plant, dB=True, deg=True)
    plt.show()
