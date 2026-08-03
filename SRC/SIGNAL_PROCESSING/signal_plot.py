from matplotlib.figure import Figure

from SRC.OTHERS.plot import *
from SRC.SIGNAL_PROCESSING.signal import *


def plot_periodic_signal_time_domain(
    signal: ElectronicPeriodicSignal,
    title: str = "Loss breakdown",
    subtitle: str | None = None,
    theme: str = "light",
) -> Figure:

    fig, ax = new_figure(theme, (8.0, 4.4))
    ax.plot(signal.time, signal.value, color=series(theme)[0], linewidth=1.5)
    axis_labels(ax, theme, x="time [s]", y="signal value")
    add_title(ax, theme, title, subtitle)
    fig.tight_layout()
    return fig
