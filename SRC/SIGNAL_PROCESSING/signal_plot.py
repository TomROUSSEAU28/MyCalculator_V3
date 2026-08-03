from matplotlib.figure import Figure

from SRC.OTHERS.plot import *
from SRC.SIGNAL_PROCESSING.signal import *


def plot_periodic_signal_time_domain(
    signal: list[ElectronicPeriodicSignal],
    title: str = "Loss breakdown",
    subtitle: str | None = None,
    theme: str = "light",
    fill: bool = False,
) -> Figure:

    fig, ax = new_figure(theme, (8.0, 4.4))
    for i, s in enumerate(signal):
        color_index = i % len(series(theme))
        ax.plot(
            s.time,
            s.value,
            color=series(theme)[color_index],
            linewidth=1.5,
            label=s.name,
        )
        if fill:
            ax.fill_between(
                s.time, s.value, 0.0, color=series(theme)[color_index], alpha=0.2
            )

        # legend with color adaptation
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        for text in ax.legend().get_texts():
            text.set_color(style(theme)["ink"])
            text.set_fontsize(10)

    axis_labels(ax, theme, x="time [s]", y="signal value")
    add_title(ax, theme, title, subtitle)
    fig.tight_layout()

    return fig
