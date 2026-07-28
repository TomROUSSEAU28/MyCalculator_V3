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
from rich.style import Style

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

WIDTH = 200  # report width, in characters
INDENT = 4  # left margin of everything but the section rules
LABEL_WIDTH = 50  # width of the label column of kv()


THEMES: dict[str, dict[str, str]] = {
    # Inspiré par Tailwind CSS (Gris ardoise profond) : très lisible et doux pour les yeux.
    "light": {
        "report.title": "bold #111827",
        "report.text": "#374151",
        "report.label": "#6b7280",
        "report.value": "bold #111827",
        "report.muted": "italic #9ca3af",
        "report.header": "bold #4b5563",
        "report.border": "#d1d5db",
        "report.rule": "#e5e7eb",
        "report.error": "bold #ef4444",
        "report.warn": "bold #f59e0b",
        "report.info": "bold #3b82f6",
        "report.ok": "bold #10b981",
    },
    # Inspiré par GitHub Dark : contraste net, bordures discrètes, statuts percutants.
    "dark": {
        "report.title": "bold #e1e4e8",
        "report.text": "#c9d1d9",
        "report.label": "#8b949e",
        "report.value": "bold #ffffff",
        "report.muted": "italic #6e7681",
        "report.header": "bold #8b949e",
        "report.border": "#30363d",
        "report.rule": "#30363d",
        "report.error": "bold #f85149",
        "report.warn": "bold #d29922",
        "report.info": "bold #58a6ff",
        "report.ok": "bold #3fb950",
    },
    # Inspiré par Dracula : teintes chaudes et accents vibrants.
    "red": {
        "report.title": "bold #ff5555",
        "report.text": "#f8f8f2",
        "report.label": "#d4c4c4",
        "report.value": "bold #ffffff",
        "report.muted": "italic #887979",
        "report.header": "bold #ffb86c",
        "report.border": "#442c2c",
        "report.rule": "#442c2c",
        "report.error": "bold #ff5555",
        "report.warn": "bold #f1fa8c",
        "report.info": "bold #8be9fd",
        "report.ok": "bold #50fa7b",
    },
    # Inspiré par Nord Theme : bleus arctiques, teintes froides et reposantes.
    "blue": {
        "report.title": "bold #88c0d0",
        "report.text": "#d8dee9",
        "report.label": "#92a1b9",
        "report.value": "bold #eceff4",
        "report.muted": "italic #4c566a",
        "report.header": "bold #81a1c1",
        "report.border": "#3b4252",
        "report.rule": "#3b4252",
        "report.error": "bold #bf616a",
        "report.warn": "bold #ebcb8b",
        "report.info": "bold #5e81ac",
        "report.ok": "bold #a3be8c",
    },
    # Inspiré par Cyberpunk / Synthwave : néon rose, cyan et violet intense.
    "vivid": {
        "report.title": "bold #ff007f",
        "report.text": "#e0e0e0",
        "report.label": "#b366ff",
        "report.value": "bold #00ffff",
        "report.muted": "italic #7a429c",
        "report.header": "bold #ff007f",
        "report.border": "#4d0066",
        "report.rule": "#4d0066",
        "report.error": "bold #ff003c",
        "report.warn": "bold #ffea00",
        "report.info": "bold #00f0ff",
        "report.ok": "bold #39ff14",
    },
    # Mono affiné : ajout de l'italique pour mieux séparer visuellement le texte secondaire sans couleur.
    "mono": {
        "report.title": "bold",
        "report.text": "none",
        "report.label": "none",
        "report.value": "bold",
        "report.muted": "dim italic",
        "report.header": "bold",
        "report.border": "dim",
        "report.rule": "dim",
        "report.error": "bold",
        "report.warn": "bold",
        "report.info": "bold",
        "report.ok": "bold",
    },
    # Altium Orange : Inspiré des calques "Mechanical", "Keep-Out" ou des sélections de composants.
    # Un orange cuivré et tranchant, parfait pour attirer l'œil sans éblouir.
    "orange_altium": {
        "report.title": "bold #ff8c00",  # Orange technique vif (Mechanical layer)
        "report.text": "#d4d4d4",
        "report.label": "#858585",
        "report.value": "bold #ffffff",
        "report.muted": "dim #5c5c5c",
        "report.header": "bold #cc7000",  # Orange cuivré plus dense
        "report.border": "#2d2d30",
        "report.rule": "#2d2d30",
        "report.error": "bold #ff2a2a",  # Rouge alerte DRC
        "report.warn": "bold #ffdd00",  # Jaune "Silkscreen" (Top Overlay)
        "report.info": "bold #00bfff",  # Bleu cyan
        "report.ok": "bold #00ff00",  # Vert de validation standard (universel en ingénierie)
    },
    "good": {
        "report.title": "bold #c58af9",  # Violet étincelant (l'étincelle de l'IA)
        "report.text": "#e8eaed",  # Blanc cassé ultra-lisible, anti-fatigue
        "report.label": "#9aa0a6",  # Gris structurel, neutre et analytique
        "report.value": "bold #8ab4f8",  # Bleu lumineux pour détacher les variables et résultats
        "report.muted": "dim italic #5f6368",  # Discrétion absolue pour le bruit de fond
        "report.header": "bold #a8c7fa",  # Bleu pastel pour une hiérarchie douce
        "report.border": "#3c4043",  # Bordure grise anthracite, nette mais subtile
        "report.rule": "#3c4043",
        "report.error": "bold #f28b82",  # Rouge doux, n'agresse pas l'œil mais reste clair
        "report.warn": "bold #fdd663",  # Jaune ambre contrastant
        "report.info": "bold #24c1e0",  # Cyan très froid pour les flux de données
        "report.ok": "bold #81c995",  # Vert pastel de validation
    },
}
DEFAULT_THEME = "dark"

# One console for the whole project, created once. highlight=False keeps rich
# from colouring numbers on its own: in a report every value is already placed
# by hand, and a second, uninvited colour code would only compete with it.
console = Console(
    theme=Theme(THEMES[DEFAULT_THEME]),
    width=WIDTH,
    highlight=False,
    color_system="truecolor",
    force_jupyter=False,
    force_terminal=True,
)

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

    """
    grid = Table.grid(padding=(0, 0))

    grid.add_column(width=label_width, no_wrap=True)

    grid.add_column()

    grid.add_row(
        Text(str(label), style="report.label"), Text(str(value), style="report.value")
    )

    _print_indented(grid, indent)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence],
    indent: int = INDENT,
) -> None:
    """
    Column table, styled like a modern spreadsheet (Zebra striping).

    First column left-aligned (labels), the others center-aligned.
    Alternating row backgrounds for better readability.
    """
    # Récupération dynamique du style du thème actif + ajout de l'inversion
    header_band_style = console.get_style("report.header") + Style(reverse=True)

    grid = Table(
        box=box.HEAVY_HEAD,
        header_style=header_band_style,  # <-- Utilisation de l'objet Style combiné
        border_style="report.border",
        row_styles=["none", "dim"],
        show_edge=False,
        pad_edge=True,
        padding=(0, 2, 0, 2),
    )

    for position, header in enumerate(headers):
        alignment = "left" if position == 0 else "center"
        grid.add_column(Text(str(header)), justify=alignment)

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
