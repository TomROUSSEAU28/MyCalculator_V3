import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, warnings

logger = logging.getLogger(__name__)


class ElectronicPeriodicSignal(BaseModel):
    name: str = Field(..., description="Nom du signal")
    value: np.ndarray = Field(..., description="Valeur du signal")
    time: np.ndarray = Field(..., description="Temps du signal")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,  # objet immuable apres creation
        extra="forbid",  # rejette un champ inconnu au lieu de l'ignorer)
        # doit forcement se construire avec from_breakpoints() pour garantir un echantillonnage uniforme
    )

    @classmethod
    def from_breakpoints(
        cls, name, times, values, n_samples: int = 2048, period: float | None = None
    ) -> "ElectronicPeriodicSignal":
        """Construit le signal a partir de points de rupture, puis reechantillonne
        uniformement. Un temps duplique exprime un saut vertical."""
        t_bp = np.asarray(times, dtype=float)
        v_bp = np.asarray(values, dtype=float)

        if t_bp.shape != v_bp.shape or t_bp.ndim != 1:
            raise ValueError("'times' et 'values' : meme forme, 1D")
        if np.any(np.diff(t_bp) < 0):
            raise ValueError("'times' doit etre croissant (doublon = saut vertical)")
        if t_bp[0] != 0.0:
            raise ValueError("le premier point doit etre a t = 0")

        T = float(period) if period is not None else float(t_bp[-1])
        if T <= 0.0 or t_bp[-1] > T:
            raise ValueError("periode incoherente avec les points fournis")

        if not np.isclose(
            v_bp[-1], v_bp[0], rtol=0.0, atol=1e-12 * max(1.0, float(np.ptp(v_bp)))
        ):
            logger.warning(
                "'%s' : v(T)=%.4g != v(0)=%.4g -> discontinuite au raccord de cycle",
                name,
                v_bp[-1],
                v_bp[0],
            )

        t = np.arange(n_samples) * T / n_samples
        return cls(name=name, value=np.interp(t, t_bp, v_bp), time=t)

    @model_validator(mode="after")
    def check_arrays(self):
        if self.value.shape != self.time.shape:
            raise ValueError("'value' et 'time' doivent avoir la meme dimension.")
        if self.value.ndim != 1 or self.value.size < 8:
            raise ValueError("Signal 1D d'au moins 8 points attendu.")
        if np.iscomplexobj(self.value):
            raise ValueError("Un signal electrique est reel.")
        dt = np.diff(self.time)
        if np.ptp(dt) > 1e-9 * np.mean(dt):
            raise ValueError("Echantillonnage non uniforme : la FFT serait fausse.")
        return self

    def sample_period(self) -> float:
        return float(self.time[1] - self.time[0])

    def period(self) -> float:
        """Duree d'un cycle : N*dt, et non time[-1]-time[0].
        Le dernier echantillon ne doit pas repeter le premier."""
        return self.value.size * self.sample_period()

    def fundamental(self) -> float:
        return 1.0 / self.period()

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.value**2)))

    def mean(self) -> float:
        return float(np.mean(self.value))

    def real_fft(self) -> tuple[np.ndarray, np.ndarray]:
        """(frequences, coefficients complexes crete) — signal reel, donc rfft."""
        spectrum = np.fft.rfft(self.value) / self.value.size
        freqs = np.fft.rfftfreq(self.value.size, d=self.sample_period())
        return freqs, spectrum

    def harmonics_rms(
        self, n_max: int = 40, threshold: float = 0.0
    ) -> dict[int, complex]:
        """Composantes RMS complexes indexees par rang harmonique.
        Rang 0 = composante continue (valeur moyenne, reelle)."""
        _, spectrum = self.real_fft()
        out: dict[int, complex] = {0: complex(spectrum[0].real, 0.0)}
        i_ref = abs(out[0]) or 1.0
        for n in range(1, min(n_max, spectrum.size - 1) + 1):
            c = spectrum[n] * np.sqrt(2.0)  # crete/2 -> RMS
            if abs(c) >= threshold * i_ref:
                out[n] = complex(c)
        return out

    def parseval_error(self, n_max: int = 40) -> float:
        """Ecart relatif entre RMS temporel et RMS spectral.
        Tend vers 0 quand n_max suffit."""
        h = self.harmonics_rms(n_max)
        spectral = np.sqrt(sum(abs(v) ** 2 for v in h.values()))
        return abs(spectral - self.rms()) / self.rms()
