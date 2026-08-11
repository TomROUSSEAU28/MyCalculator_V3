import sys
from enum import Enum
from math import pi, sqrt
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Doit preceder les imports 'SRC.*' : sans cela le script n'est importable
# qu'en module (python -m SRC.MAGNETIC.dowell), pas en direct.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from SRC.constant import MU0, SIGMA_CU_20C


# il faut une fonction qui prend le signal courant en electronic periodic signal et qui le transforme en flux B,
# ensuite une classe de material, core geometrie
# ensuite une fonction IGSE

# v (signal) → flux_density(v, N, core) → B (signal aussi)
# B → core_loss(B, core, T) → CORE_LOSS_INFO
