"""
Terminal report layout, built on rich.

Nothing domain-specific here: only the presentation of the project's
"spreadsheet" scripts (main.py, NOTEBOOKS/**/*.py). One set of helpers, so
every report of the project reads the same way — in a terminal and in a
Jupyter cell alike, since rich renders to both.

    use_theme("dark")
    section(1, "Operating point")
    kv("Switching frequency", "100 kHz")
    table(["Edge", "t [ns]", "Slew rate"], [("t_ri", "11.3", "2.21 A/ns")])
    alert("error", "Voltage above the V_DSS rating — breakdown.")

Colours live in THEMES: one theme is a plain {style name: rich style} dict.
Add an entry there, call use_theme("your-name") once at the top of your
script, and every helper below follows — nothing else to change.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

__all__ = [
    "THEMES",
    "active_theme",
    "alert",
    "blank",
    "console",
    "dataframe",
    "kv",
    "paragraph",
    "rule",
    "section",
    "table",
    "use_theme",
]

WIDTH = 78  # report width, in characters
INDENT = 2  # left margin of everything but the section rules
LABEL_WIDTH = 38  # width of the label column of kv()


# ============================================================================ #
#  Themes
# ============================================================================ #

# Every theme defines the same style names, so no helper can ever land on a
# missing colour. The names are semantic on purpose: "report.label" says what
# the text is, not what it looks like — that is what makes a theme swappable.
#
#   report.title  section headings          report.header  table headers
#   report.text   body text                 report.border  table rules
#   report.label  the left column of kv()   report.rule    section rules
#   report.value  the right column of kv()  report.muted   asides, units
#
# "light" and "dark" mirror the plot palettes of SRC/MOSFET/mosfet_plot.py, so
# a report and its figures look like they belong together. "mono" drops colour
# entirely — for log files, CI output, and anything piped elsewhere.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "report.title": "bold #0b0b0b",
        "report.text": "#0b0b0b",
        "report.label": "#52514e",
        "report.value": "bold #0b0b0b",
        "report.muted": "#898781",
        "report.header": "bold #52514e",
        "report.border": "#c3c2b7",
        "report.rule": "#c3c2b7",
        "report.error": "bold #d03b3b",
        "report.warn": "bold #b26a00",
        "report.info": "bold #2a78d6",
        "report.ok": "bold #007a3d",
    },
    "dark": {
        "report.title": "bold #ffffff",
        "report.text": "#e6e5e0",
        "report.label": "#c3c2b7",
        "report.value": "bold #ffffff",
        "report.muted": "#898781",
        "report.header": "bold #c3c2b7",
        "report.border": "#383835",
        "report.rule": "#383835",
        "report.error": "bold #ff6b6b",
        "report.warn": "bold #eda100",
        "report.info": "bold #3987e5",
        "report.ok": "bold #34c759",
    },
    "mono": {
        "report.title": "bold",
        "report.text": "none",
        "report.label": "none",
        "report.value": "bold",
        "report.muted": "dim",
        "report.header": "bold",
        "report.border": "dim",
        "report.rule": "dim",
        "report.error": "bold",
        "report.warn": "bold",
        "report.info": "bold",
        "report.ok": "bold",
    },
}

DEFAULT_THEME = "dark"

# One console for the whole project, created once. highlight=False keeps rich
# from colouring numbers on its own: in a report every value is already placed
# by hand, and a second, uninvited colour code would only compete with it.
console = Console(theme=Theme(THEMES[DEFAULT_THEME]), width=WIDTH, highlight=False)

_active_theme = DEFAULT_THEME


def use_theme(name: str) -> None:
    """
    Switch the palette used by every helper below.

    Call it once, before the first report line. The console object itself is
    never replaced, so modules that imported it earlier keep working.
    """
    global _active_theme
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r} (available: {', '.join(THEMES)})")
    console.push_theme(Theme(THEMES[name]))
    _active_theme = name


def active_theme() -> str:
    """Name of the palette currently in use."""
    return _active_theme


# ============================================================================ #
#  Blocks
# ============================================================================ #


def blank() -> None:
    """One empty line, so every report breathes the same way."""
    console.print()


def _print_indented(renderable, indent: int) -> None:
    """
    Print a block shifted right by `indent`.

    expand=False matters: left to itself, Padding stretches the block to the
    full console width and pads every line with trailing spaces — invisible in
    a terminal, but noise as soon as the report is piped into a file.
    """
    console.print(Padding(renderable, (0, 0, 0, indent), expand=False))


def rule(title: str = "") -> None:
    """Horizontal rule, with the title written into it when there is one."""
    heading = Text(title, style="report.title") if title else ""
    console.print(Rule(heading, style="report.rule", align="left"))


def section(number: int | str | None, title: str) -> None:
    """Section heading. number=None for a heading without one."""
    blank()
    rule(title if number is None else f"{number}.  {title}")


def kv(
    label: str,
    value,
    indent: int = INDENT,
    label_width: int = LABEL_WIDTH,
) -> None:
    """
    One "label ....... value" line.

    soft_wrap keeps a long value (a file path, mostly) on its own line instead
    of folding it back under the label column, where it would read as a second
    label. It runs past the report width; the terminal deals with it.
    """
    line = Text(" " * indent)
    line.append(f"{label:<{label_width}}", style="report.label")
    line.append(str(value), style="report.value")
    console.print(line, soft_wrap=True)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence],
    indent: int = INDENT,
) -> None:
    """
    Column table, widths handled by rich.

    First column left-aligned (labels), the others right-aligned (numbers).
    Cells are str()-ed as they come: format your floats before calling, the
    caller is the only one that knows the units.

    Everything goes in as Text rather than as a plain string, because a report
    is full of "[W]", "[ns]", "[°C/W]" — which rich would otherwise read as
    markup tags and swallow.
    """
    grid = Table(
        box=box.SIMPLE_HEAD,  # a single rule under the header, nothing else
        header_style="report.header",
        border_style="report.border",
        show_edge=False,
        pad_edge=False,
        padding=(0, 2, 0, 0),
    )
    for position, header in enumerate(headers):
        grid.add_column(
            Text(str(header)), justify="left" if position == 0 else "right"
        )
    for row in rows:
        grid.add_row(*(Text(str(cell)) for cell in row))
    _print_indented(grid, indent)


def dataframe(
    frame,
    float_format: str = "{:.4f}",
    indent: int = INDENT,
) -> None:
    """
    A pandas DataFrame as a report table — the index becomes the first column.

    Duck-typed on purpose (index / columns / itertuples): this module stays
    free of a pandas import, and of the load time that comes with it.
    """
    headers = [str(frame.index.name or "")] + [str(column) for column in frame.columns]
    rows = [
        [str(row[0])] + [_cell(value, float_format) for value in row[1:]]
        for row in frame.itertuples(index=True, name=None)
    ]
    table(headers, rows, indent=indent)


def _cell(value, float_format: str) -> str:
    return float_format.format(value) if isinstance(value, float) else str(value)


def paragraph(text: str, indent: int = INDENT) -> None:
    """Free text, wrapped to the report width."""
    _print_indented(Text(text, style="report.text"), indent)


# Tag written by alert() for each level, with the style that goes with it.
# `prefix=` replaces the tag on the spot, without inventing a new level.
ALERT_LEVELS: dict[str, tuple[str, str]] = {
    "error": ("ERROR", "report.error"),
    "warn": ("WARNING", "report.warn"),
    "info": ("INFO", "report.info"),
    "ok": ("OK", "report.ok"),
}


def alert(
    level: str,
    message: str,
    prefix: str | None = None,
    indent: int = INDENT,
) -> None:
    """
    A flagged message, wrapped and hanging under its own tag.

    level  : key of ALERT_LEVELS ("error", "warn", "info", "ok")
    prefix : replacement tag, for when the level goes by another name in
             context (e.g. "CANNOT COMPUTE" for a blocking error)

    The tag and the message are two cells of a grid, which is what keeps the
    continuation lines aligned under the first word rather than under the tag.
    """
    default_label, style = ALERT_LEVELS.get(level, (level.upper(), "report.text"))
    label = default_label if prefix is None else prefix

    line = Table.grid(padding=(0, 1))
    line.add_column(no_wrap=True)
    line.add_column(overflow="fold", max_width=WIDTH - indent - len(label) - 3)
    line.add_row(Text(f"[{label}]", style=style), Text(message, style="report.text"))
    _print_indented(line, indent)
