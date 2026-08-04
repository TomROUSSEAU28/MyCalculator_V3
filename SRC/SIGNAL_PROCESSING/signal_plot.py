"""
Plots for a periodic electronic signal.

The two views a waveform is read through, and the tables that go with them:

    plot_time_domain()  -> what the probe shows: shape, levels, duty cycle
    plot_spectrum()     -> what the circuit sees: harmonic ranks and weights
    plot_signal()       -> both, stacked, for one signal
    signal_table()      -> every measurement of ElectronicPeriodicSignal.info()
    harmonics_table()   -> rank by rank, the numbers behind the spectrum

A periodic signal has a DISCRETE spectrum: energy only at multiples of f0.
That is why the spectrum is drawn as stems and not as a continuous curve — a
line joining harmonic 3 to harmonic 4 would draw energy at frequencies where
the signal has none.

Only what is signal-specific lives here. The themes, the chrome and the export
path are shared by every figure of the project and sit in SRC/OTHERS/plot.py —
including save_figure(), which is where you import it from.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter

from SRC.OTHERS.plot import (
    add_title,
    annotate,
    axis_labels,
    engineering_ticks,
    grid,
    legend,
    new_figure,
    new_grid,
    reference_line,
    series,
    style,
    vertical_reference_line,
)
from SRC.SIGNAL_PROCESSING.signal import ElectronicPeriodicSignal

__all__ = [
    "harmonics_table",
    "plot_signal",
    "plot_spectrum",
    "plot_time_domain",
    "signal_table",
]

# region helpers


def _quantity(value: float, unit: str = "", digits: int = 4) -> str:
    """
    A number the way an engineer writes it: 1e-05 -> "10 µs", 2.4e6 -> "2.4 MHz".

    Same formatter as the tick labels of SRC/OTHERS/plot.py, so a value quoted
    in a subtitle or a table reads in the same units as the axis it came from.

    Rounded to `digits` SIGNIFICANT digits rather than to a fixed number of
    decimals: a peak resampled onto the time grid comes out at 5.99951 A, and
    the last two digits are the sampling, not the signal. Fixed decimals cannot
    do this — the same setting that tidies 5.99951 destroys 4.88281 ns.
    """
    rounded = float(f"{value:.{digits}g}") if np.isfinite(value) else value
    return EngFormatter(unit=unit, sep=" ")(rounded)


def _percent(value: float, places: int = 2) -> str:
    """A fraction as a percentage, nan included — info() returns nan for a
    ratio that is undefined rather than pretending it is zero."""
    return "-" if np.isnan(value) else f"{100.0 * value:.{places}f} %"


def _signals(
    signal: ElectronicPeriodicSignal | Sequence[ElectronicPeriodicSignal],
) -> list[ElectronicPeriodicSignal]:
    """
    Accept one signal or several, always hand back a list.

    Capped at the six categorical slots of a theme: past that the palette
    starts repeating, and two curves of the same colour on one axes is not a
    figure anybody can read. Overlaying more than three is already a lot —
    facet into several figures instead.
    """
    items = [signal] if isinstance(signal, ElectronicPeriodicSignal) else list(signal)
    if not items:
        raise ValueError("no signal to plot")
    if len(items) > 6:
        raise ValueError(
            f"{len(items)} signals for 6 colour slots — plot them in several figures"
        )
    return items


def _fills(fill: bool | Sequence[bool], count: int) -> list[bool]:
    """One fill flag per signal, from a flag or a list of them."""
    if isinstance(fill, bool):
        return [fill] * count
    flags = list(fill)
    if len(flags) != count:
        raise ValueError(f"'fill' needs {count} entries, got {len(flags)}")
    return flags


def _repeat(
    signal: ElectronicPeriodicSignal, cycles: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    The waveform laid out over `cycles` periods.

    The stored signal holds exactly one period and does NOT repeat its first
    sample at the end — see ElectronicPeriodicSignal.period(). Plotted as is,
    the segment from the last sample back to t = T is therefore missing, which
    on a square wave means the closing edge is simply not drawn. Tiling and
    appending value[0] at t = cycles*T restores it.
    """
    if cycles <= 0:
        raise ValueError(f"cycles must be > 0, got {cycles}")
    whole = int(np.ceil(cycles))
    value = np.append(np.tile(signal.value, whole), signal.value[0])
    time = np.arange(value.size) * signal.sample_period()
    keep = time <= cycles * signal.period() * (1.0 + 1e-12)
    return time[keep], value[keep]


def _harmonics(
    signal: ElectronicPeriodicSignal,
    n_max: int,
    threshold: float,
    include_dc: bool,
) -> tuple[list[int], np.ndarray]:
    """
    (ranks, RMS magnitudes) worth drawing, quietest ones dropped.

    `threshold` is a fraction of the FUNDAMENTAL, which is the reference a
    harmonic is judged against. ElectronicPeriodicSignal.harmonics_rms() scales
    its own threshold on the DC component instead, so it is asked for
    everything here and the filtering is done below.
    """
    spectrum = signal.harmonics_rms(n_max, threshold=0.0)
    reference = abs(spectrum.get(1, 0j)) or 1.0
    ranks = [
        rank
        for rank in sorted(spectrum)
        if (rank != 0 or include_dc) and abs(spectrum[rank]) >= threshold * reference
    ]
    return ranks, np.array([abs(spectrum[rank]) for rank in ranks])


# endregion


# region panels
#
# One function per panel, drawing into an axes it does not own. The public
# plot_* below only pick a figure size and a title — which is what lets
# plot_signal() stack the same two panels without duplicating a line of them.


def _draw_time(
    ax,
    theme: str,
    signals: list[ElectronicPeriodicSignal],
    unit: str,
    fill: list[bool],
    cycles: float,
    levels: bool,
) -> None:
    """The waveform(s) against time."""
    palette = series(theme)
    c = style(theme)

    grid(ax, theme, axis="y")

    # 1. On calcule la durée totale à afficher (le signal le plus lent * cycles)
    max_duration = max(s.period() * cycles for s in signals)

    span = 0.0
    for position, signal in enumerate(signals):
        colour = palette[position]

        # 2. On adapte le nombre de cycles pour que chaque signal aille jusqu'au bout
        local_cycles = max_duration / signal.period()

        time, value = _repeat(signal, local_cycles)
        span = max(span, float(time[-1]))
        ax.plot(
            time,
            value,
            color=colour,
            linewidth=1.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        if fill[position]:
            ax.fill_between(
                time, value, 0.0, color=colour, alpha=0.18, linewidth=0, zorder=2
            )

    if levels:
        if len(signals) != 1:
            raise ValueError("levels= draws on one signal only; plot it alone")
        mean, rms = signals[0].mean(), signals[0].rms()
        # Context, not limits: these are the levels the waveform is read
        # relative to, so they take the recessive rule and never the dashed one.
        #
        # RMS labelled above its rule and the mean below its own: RMS >= |mean|
        # always, so this is the one arrangement that cannot collide — and a
        # small ripple on a large DC puts the two rules a hair apart, which is
        # exactly when the labels are worth reading.
        reference_line(
            ax, theme, mean, f"mean {_quantity(mean, unit)}", offset=(8, -14)
        )
        reference_line(ax, theme, rms, f"RMS {_quantity(rms, unit)}", offset=(8, 6))

    # The period is only worth marking once the plot shows more than one of
    # them — on a single cycle the axis already ends exactly there.
    if len(signals) == 1 and cycles > 1:
        period = signals[0].period()
        vertical_reference_line(ax, theme, period, f"T = {_quantity(period, 's')}")

    # Zero is a real level for a signal that swings both ways, and the one the
    # fill is measured from. Drawn under everything, in the axis colour.
    if min(float(np.min(s.value)) for s in signals) < 0.0:
        ax.axhline(0.0, color=c["axis"], linewidth=1.0, zorder=1)

    ax.set_xlim(0.0, span)
    engineering_ticks(ax, theme, axis="x", unit="s")
    engineering_ticks(ax, theme, axis="y", unit=unit)
    axis_labels(ax, theme, x="Time", y="Amplitude")


def _draw_spectrum(
    ax,
    theme: str,
    signals: list[ElectronicPeriodicSignal],
    unit: str,
    n_max: int,
    threshold: float,
    scale: str,
    floor_db: float,
    include_dc: bool,
) -> None:
    """The harmonic content, as stems at n*f0."""
    if scale not in ("linear", "db"):
        raise ValueError(f"scale must be 'linear' or 'db', got {scale!r}")
    palette = series(theme)
    c = style(theme)

    grid(ax, theme, axis="y")
    top = 0.0
    right = 0.0
    for position, signal in enumerate(signals):
        colour = palette[position]
        ranks, magnitudes = _harmonics(signal, n_max, threshold, include_dc)
        if not ranks:
            continue
        f0 = signal.fundamental()
        frequency = np.array(ranks, dtype=float) * f0

        if scale == "db":
            reference = abs(signal.fundamental_component()) or 1.0
            with np.errstate(divide="ignore"):
                # A rank that filtered through at exactly zero (a missing
                # harmonic of a symmetric waveform) is -inf dB: clamp it to the
                # floor so it draws as an empty slot instead of vanishing.
                magnitudes = 20.0 * np.log10(magnitudes / reference)
            magnitudes = np.maximum(magnitudes, floor_db)
            baseline = floor_db
        else:
            baseline = 0.0

        # Several signals sharing an f0 would stack their stems on the exact
        # same abscissa. Nudged apart by a fraction of f0 — the true frequency
        # of every stem is still n*f0, the offset is drawing, not data.
        offset = 0.0
        if len(signals) > 1:
            offset = (position - 0.5 * (len(signals) - 1)) * 0.14 * f0

        ax.vlines(
            frequency + offset,
            baseline,
            magnitudes,
            color=colour,
            linewidth=2.0,
            zorder=3,
        )
        ax.plot(
            frequency + offset,
            magnitudes,
            linestyle="none",
            marker="o",
            markersize=4.5,
            color=colour,
            # Surface-coloured ring: keeps the heads readable where the stems
            # of two signals land close together.
            markeredgecolor=c["surface"],
            markeredgewidth=1.5,
            zorder=4,
        )
        top = max(top, float(np.max(magnitudes)))
        right = max(right, float(np.max(frequency)))

    ax.set_xlim(-0.04 * right if right else -1.0, right * 1.04 if right else 1.0)
    engineering_ticks(ax, theme, axis="x", unit="Hz")
    if scale == "db":
        ax.set_ylim(floor_db, max(top, 0.0) + 6.0)
        axis_labels(ax, theme, x="Frequency", y="Level [dBc]")
    else:
        ax.set_ylim(0.0, top * 1.18 if top else 1.0)
        engineering_ticks(ax, theme, axis="y", unit=unit)
        axis_labels(ax, theme, x="Frequency", y="RMS amplitude")

    # One signal: label the fundamental, the value the reader came for. Several:
    # the labels would collide, and the table is the place to read them off.
    if len(signals) == 1 and scale == "linear":
        first = abs(signals[0].fundamental_component())
        annotate(
            ax,
            theme,
            _quantity(first, unit),
            (signals[0].fundamental(), first),
            (7, 0),
            role="ink",
            bold=True,
            va="center",
        )


# endregion


# region figures


def _time_subtitle(signals: list[ElectronicPeriodicSignal], unit: str) -> str | None:
    """What a single-signal time plot says instead of carrying a legend."""
    if len(signals) != 1:
        return None
    signal = signals[0]
    return (
        f"{_quantity(signal.fundamental(), 'Hz')}"
        f" · {_quantity(signal.peak_to_peak(), unit)} pk-pk"
        f" · RMS {_quantity(signal.rms(), unit)}"
        f" · mean {_quantity(signal.mean(), unit)}"
    )


def _spectrum_subtitle(
    signals: list[ElectronicPeriodicSignal], n_max: int
) -> str | None:
    if len(signals) != 1:
        return None
    signal = signals[0]
    return (
        f"f0 = {_quantity(signal.fundamental(), 'Hz')}"
        f" · THD {_percent(signal.thd(n_max))} over {n_max} harmonics"
    )


def plot_time_domain(
    signal: ElectronicPeriodicSignal | Sequence[ElectronicPeriodicSignal],
    title: str = "Time domain",
    subtitle: str | None = None,
    theme: str = "light",
    unit: str = "",
    fill: bool | Sequence[bool] = False,
    cycles: float = 1.0,
    levels: bool = False,
) -> Figure:
    """
    The waveform against time — one signal or several overlaid.

    unit   : the physical unit of the samples, "V" or "A". It goes on the tick
             labels, so the axis reads "5 mA" rather than "0.005".
    fill   : shade between the curve and zero. One flag, or one per signal.
             Worth it for a current whose area is the charge; noise otherwise.
    cycles : periods to draw. 2 makes the periodicity visible, which a single
             cycle cannot show. Fractional values are allowed.
    levels : draw the mean and the RMS as rules. Single signal only — two sets
             of level rules on one axes cannot be told apart.
    """
    signals = _signals(signal)
    fig, ax = new_figure(theme, (8.0, 4.4))
    _draw_time(ax, theme, signals, unit, _fills(fill, len(signals)), cycles, levels)

    if len(signals) > 1:
        legend(
            ax,
            theme,
            [(s.name, series(theme)[i]) for i, s in enumerate(signals)],
            kind="line",
        )
    add_title(ax, theme, title, subtitle or _time_subtitle(signals, unit))
    fig.tight_layout()
    return fig


def plot_spectrum(
    signal: ElectronicPeriodicSignal | Sequence[ElectronicPeriodicSignal],
    title: str = "Harmonic spectrum",
    subtitle: str | None = None,
    theme: str = "light",
    unit: str = "",
    n_max: int = 20,
    threshold: float = 0.01,
    scale: str = "linear",
    floor_db: float = -60.0,
    include_dc: bool = True,
) -> Figure:
    """
    The harmonic content, as stems at multiples of the fundamental.

    n_max     : highest rank drawn. The ranks, not the frequencies — two
                signals of different f0 are still compared harmonic to
                harmonic.
    threshold : drop anything below this fraction of the fundamental. 0.01 is
                1 %; pass 0.0 to keep every rank, including the numerically
                zero ones a symmetric waveform has.
    scale     : "linear" plots RMS in the unit of the signal — the values that
                add up, and the ones a loss calculation needs. "db" plots
                20*log10 relative to the fundamental (dBc), which is the only
                way to see a harmonic 60 dB down next to one at 0 dB.
    floor_db  : bottom of the dB axis. Stems below it are clamped, not hidden.
    include_dc: keep rank 0. It is not a harmonic, but on a signal with a bias
                it is usually the largest component and dropping it silently
                would misrepresent the spectrum.
    """
    signals = _signals(signal)
    fig, ax = new_figure(theme, (8.0, 4.4))
    _draw_spectrum(
        ax, theme, signals, unit, n_max, threshold, scale, floor_db, include_dc
    )

    if len(signals) > 1:
        legend(
            ax,
            theme,
            [(s.name, series(theme)[i]) for i, s in enumerate(signals)],
            kind="line",
        )
    add_title(ax, theme, title, subtitle or _spectrum_subtitle(signals, n_max))
    fig.tight_layout()
    return fig


def plot_signal(
    signal: ElectronicPeriodicSignal,
    title: str | None = None,
    subtitle: str | None = None,
    theme: str = "light",
    unit: str = "",
    fill: bool = True,
    cycles: float = 2.0,
    n_max: int = 20,
    threshold: float = 0.01,
    scale: str = "linear",
    floor_db: float = -60.0,
) -> Figure:
    """
    One signal, both views stacked: the waveform, and its spectrum under it.

    The two panels answer one question — "what is this signal made of" — and
    have to be read together, which is what earns them a single figure. Two
    signals compared belong in plot_time_domain() and plot_spectrum(), one
    figure each: stacking those would put four series on two panels.

    Arguments carry the meaning they have in the two functions above.
    """
    fig, (top, bottom) = new_grid(theme, (8.2, 7.4), rows=2, height_ratios=(3.0, 2.0))

    _draw_time(top, theme, [signal], unit, [fill], cycles, levels=False)
    _draw_spectrum(
        bottom, theme, [signal], unit, n_max, threshold, scale, floor_db, True
    )

    add_title(
        top,
        theme,
        title or signal.name,
        subtitle or _time_subtitle([signal], unit),
    )
    add_title(bottom, theme, "Harmonic spectrum", _spectrum_subtitle([signal], n_max))

    fig.tight_layout()
    # Room for the spectrum's own title block, which tight_layout sizes for the
    # axes only and would otherwise let the waveform sit on top of.
    fig.subplots_adjust(hspace=0.42)
    return fig


# endregion


# region tables

# Every field of SIGNAL_INFO, in reading order: what was sampled, how big it
# is, what shape it has, what it is made of. (label, attribute, unit) — a unit
# of None marks a dimensionless number, "%" a fraction to show as a percentage.
_INFO_ROWS: list[tuple[str, str, str | None]] = [
    ("Samples", "n_samples", None),
    ("Sample period", "sample_period", "s"),
    ("Sample rate", "sample_rate", "Hz"),
    ("Nyquist frequency", "nyquist", "Hz"),
    ("Period", "period", "s"),
    ("Fundamental", "fundamental", "Hz"),
    ("Minimum", "minimum", ""),
    ("Maximum", "maximum", ""),
    ("Peak |x|", "peak", ""),
    ("Peak to peak", "peak_to_peak", ""),
    ("Mean (DC)", "mean", ""),
    ("Rectified mean", "rectified_mean", ""),
    ("RMS", "rms", ""),
    ("AC RMS (ripple)", "ac_rms", ""),
    ("Crest factor", "crest_factor", None),
    ("Form factor", "form_factor", None),
    ("Ripple factor", "ripple_factor", None),
    ("Duty cycle", "duty_cycle", "%"),
    ("Fundamental RMS", "fundamental_rms", ""),
    ("Fundamental phase", "fundamental_phase", "deg"),
    ("THD", "thd", "%"),
    ("Harmonics used", "n_harmonics", None),
    ("Parseval error", "parseval_error", "%"),
]


def _info_cell(value, unit: str | None, signal_unit: str) -> str:
    """One measurement, formatted for reading rather than for arithmetic."""
    # info() returns nan for a ratio that is undefined — a crest factor with no
    # RMS. One dash for all of them: "nan" in a report reads as a failure.
    if isinstance(value, float) and not np.isfinite(value):
        return "-"
    if unit == "%":
        return _percent(value)
    if unit is None:
        if isinstance(value, int):
            return f"{value:d}"
        # A dimensionless number in the thousands is a ratio against something
        # that is nearly zero — a ripple factor on an AC-coupled waveform. Three
        # decimals there is false precision; the exponent says what it is.
        return f"{value:.3g}" if abs(value) >= 1000.0 else f"{value:.3f}"
    if unit == "deg":
        return f"{value:.1f}°"
    # "" means the unit of the samples themselves, which only the caller knows.
    return _quantity(value, signal_unit if unit == "" else unit)


def signal_table(
    signal: ElectronicPeriodicSignal | Sequence[ElectronicPeriodicSignal],
    unit: str = "",
    n_max: int | None = None,
) -> pd.DataFrame:
    """
    The figures' table twin — every measurement of info(), one column per signal.

    Three of the light-mode series colours sit below 3:1 against the surface,
    so the palette is only allowed to carry meaning alongside visible labels or
    this table. Print it next to the figures.

    The cells are STRINGS in engineering notation, not floats: a 10 µs period
    rounded to four decimals is 0.0000, which is how a table of correct numbers
    ends up saying nothing. Read the numbers off info() when you need to
    compute with them.

    Feed it straight to SRC/OTHERS/terminal.py's dataframe().
    """
    signals = _signals(signal)
    measurements = [s.info(n_max) for s in signals]
    frame = pd.DataFrame(
        {
            f"{s.name} [{position}]" if len(signals) > 1 else s.name: [
                _info_cell(getattr(info, attribute), row_unit, unit)
                for _, attribute, row_unit in _INFO_ROWS
            ]
            for position, (s, info) in enumerate(zip(signals, measurements))
        },
        index=[label for label, _, _ in _INFO_ROWS],
    )
    frame.index.name = "Measurement"
    return frame


def harmonics_table(
    signal: ElectronicPeriodicSignal,
    unit: str = "",
    n_max: int = 20,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """1
    Rank by rank, the numbers the spectrum plot draws.

    One row per harmonic kept: its frequency, its RMS, its weight relative to
    the fundamental in percent and in dBc, and its phase. Rank 0 is the DC
    component — listed, but with no phase and no dBc, because neither means
    anything for it.
    """
    ranks, magnitudes = _harmonics(signal, n_max, threshold, include_dc=True)
    spectrum = signal.harmonics_rms(n_max, threshold=0.0)
    reference = abs(spectrum.get(1, 0j)) or 1.0
    f0 = signal.fundamental()

    rows = []
    for rank, magnitude in zip(ranks, magnitudes):
        share = magnitude / reference
        rows.append(
            [
                _quantity(rank * f0, "Hz"),
                _quantity(float(magnitude), unit),
                _percent(float(share)),
                "-" if rank == 0 or share == 0 else f"{20.0 * np.log10(share):.1f}",
                "-" if rank == 0 else f"{np.degrees(np.angle(spectrum[rank])):.1f}°",
            ]
        )

    frame = pd.DataFrame(
        rows,
        columns=["Frequency", "RMS", "of fundamental", "dBc", "Phase"],
        index=[("DC" if rank == 0 else f"H{rank}") for rank in ranks],
    )
    frame.index.name = f"{signal.name} — rank"
    return frame


# endregion
