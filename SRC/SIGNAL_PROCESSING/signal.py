import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, warnings

logger = logging.getLogger(__name__)


def _ratio(numerator: float, denominator: float) -> float:
    """
    numerator / denominator, or nan when the denominator is zero.

    A crest factor with no RMS, a THD with no fundamental: these are not
    infinite, they are undefined. nan says so, and it says it without stopping
    the report that was only asking for a number.
    """
    return float(numerator / denominator) if denominator else float("nan")


class SIGNAL_INFO(BaseModel):
    """
    Every standard measurement of one periodic signal, in one place.

    What a bench instrument shows for the same waveform, and in the same
    units — SI throughout, degrees for the one phase.
    """

    name: str

    # --- sampling: what the numbers below are allowed to say ----------------
    n_samples: int
    sample_period: float  # dt [s]
    sample_rate: float  # 1/dt [Hz]
    nyquist: float  # highest frequency the sampling can carry [Hz]
    period: float  # T [s]
    fundamental: float  # f0 = 1/T [Hz]

    # --- amplitude ----------------------------------------------------------
    minimum: float
    maximum: float
    peak: float  # max |x|, what a rating is compared against
    peak_to_peak: float
    mean: float  # DC component
    rectified_mean: float  # mean |x|, what a moving-coil meter reads
    rms: float  # heating value: I_rms^2 * R is the loss
    ac_rms: float  # RMS once the DC is removed — the ripple

    # --- shape: dimensionless, and the reason two signals of equal RMS do
    #     not stress a part equally ---------------------------------------
    crest_factor: float  # peak / rms
    form_factor: float  # rms / rectified_mean
    ripple_factor: float  # ac_rms / |mean|
    duty_cycle: float  # fraction of the period above the mid-level

    # --- spectrum -----------------------------------------------------------
    fundamental_rms: float  # RMS of harmonic 1 [same unit as the signal]
    fundamental_phase: float  # phase of harmonic 1 [deg]
    thd: float  # total harmonic distortion, fraction (0.05 = 5 %)
    n_harmonics: int  # harmonics the two numbers above were taken over
    parseval_error: float  # residual between temporal and spectral RMS


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

    def sample_rate(self) -> float:
        return 1.0 / self.sample_period()

    def nyquist(self) -> float:
        """Highest frequency the sampling can carry. Anything above it in the
        real signal came back folded onto a lower one — it is not measurable
        here, whatever the FFT shows."""
        return 0.5 * self.sample_rate()

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.value**2)))

    def mean(self) -> float:
        return float(np.mean(self.value))

    # region amplitude

    def minimum(self) -> float:
        return float(np.min(self.value))

    def maximum(self) -> float:
        return float(np.max(self.value))

    def peak(self) -> float:
        """max |x| : the excursion a rating is compared against, which is the
        negative one for a signal that swings below zero."""
        return float(np.max(np.abs(self.value)))

    def peak_to_peak(self) -> float:
        return float(np.ptp(self.value))

    def rectified_mean(self) -> float:
        """mean |x| : what an average-reading meter shows."""
        return float(np.mean(np.abs(self.value)))

    def ac_rms(self) -> float:
        """
        RMS of the signal once its DC component is removed — the ripple.

        Computed as the standard deviation rather than sqrt(rms^2 - mean^2):
        the two are equal on paper, but the subtraction loses every significant
        digit when a small ripple rides on a large DC, which is exactly the
        case this measurement exists for.
        """
        return float(np.std(self.value))

    # endregion

    # region shape

    def crest_factor(self) -> float:
        """peak / RMS. 1.41 for a sine, 1 for a square, high for a pulse — and
        it is what says whether a part sized on RMS still survives the peak."""
        return _ratio(self.peak(), self.rms())

    def form_factor(self) -> float:
        """RMS / rectified mean. 1.11 for a sine — the factor an average-reading
        meter is calibrated with, and therefore its error on anything else."""
        return _ratio(self.rms(), self.rectified_mean())

    def ripple_factor(self) -> float:
        """
        AC RMS / |DC|. The usual figure of merit of a rectifier output.

        Undefined for a waveform that has no DC to be a ripple ON — a capacitor
        current, anything AC-coupled. A mean sitting at the round-off floor of
        the sum that produced it is not a small DC, it is no DC, and dividing by
        it yields 1e15 rather than a measurement. Reported as nan instead.

        The floor is relative to the signal's own peak: 1e-12 of it is some
        three orders of magnitude above the arithmetic's resolution, and far
        below any DC a converter could actually be carrying.
        """
        mean = abs(self.mean())
        return _ratio(self.ac_rms(), mean if mean > 1e-12 * self.peak() else 0.0)

    def duty_cycle(self) -> float:
        """
        Fraction of the period spent above the mid-level (max + min) / 2.

        The mid-level, not zero: it is what makes the answer the expected one
        for a signal that never crosses zero, e.g. a 0 V / 5 V logic waveform
        or an inductor current with a DC bias. Meaningful for a two-level
        waveform; on a triangle or a sine it just reports 0.5.
        """
        mid = 0.5 * (self.maximum() + self.minimum())
        return float(np.count_nonzero(self.value > mid) / self.value.size)

    # endregion

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
        return _ratio(abs(spectral - self.rms()), self.rms())

    def full_rank(self) -> int:
        """Highest harmonic rank the sampling supports — the one at Nyquist.
        Default of every spectral measurement that must not be truncated."""
        return self.value.size // 2

    def fundamental_component(self) -> complex:
        """RMS phasor of harmonic 1: magnitude is its RMS, angle its phase."""
        return self.harmonics_rms(n_max=1).get(1, 0j)

    def thd(self, n_max: int | None = None) -> float:
        """
        Total harmonic distortion: sqrt(sum of H2..Hn squared) / H1, in RMS.

        A fraction, not a percentage — 0.05 is 5 %.

        n_max None takes every rank up to Nyquist, which is the only value that
        does not silently flatter the signal: truncating the sum can only lower
        the result, so a THD quoted over 40 harmonics is a lower bound.
        """
        h = self.harmonics_rms(n_max if n_max is not None else self.full_rank())
        distortion = np.sqrt(sum(abs(v) ** 2 for rank, v in h.items() if rank >= 2))
        return _ratio(float(distortion), abs(h.get(1, 0j)))

    def info(self, n_max: int | None = None) -> SIGNAL_INFO:
        """
        Every standard measurement of the signal, in one object.

        n_max : harmonics the spectral figures are taken over. None (default)
                goes up to Nyquist — see thd().

        The temporal figures come from the samples and the spectral ones from
        the FFT, on purpose: `parseval_error` compares the two, and is the
        check that the spectrum really accounts for the whole signal. Anything
        but a near-zero value there means the waveform carries detail the
        sampling cannot resolve, and every harmonic number is then suspect.
        """
        rank = n_max if n_max is not None else self.full_rank()
        first = self.fundamental_component()
        return SIGNAL_INFO(
            name=self.name,
            n_samples=int(self.value.size),
            sample_period=self.sample_period(),
            sample_rate=self.sample_rate(),
            nyquist=self.nyquist(),
            period=self.period(),
            fundamental=self.fundamental(),
            minimum=self.minimum(),
            maximum=self.maximum(),
            peak=self.peak(),
            peak_to_peak=self.peak_to_peak(),
            mean=self.mean(),
            rectified_mean=self.rectified_mean(),
            rms=self.rms(),
            ac_rms=self.ac_rms(),
            crest_factor=self.crest_factor(),
            form_factor=self.form_factor(),
            ripple_factor=self.ripple_factor(),
            duty_cycle=self.duty_cycle(),
            fundamental_rms=abs(first),
            fundamental_phase=float(np.degrees(np.angle(first))),
            thd=self.thd(rank),
            n_harmonics=rank,
            parseval_error=self.parseval_error(rank),
        )
