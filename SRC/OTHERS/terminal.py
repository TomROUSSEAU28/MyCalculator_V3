"""
Mise en forme des sorties terminal.

Rien de métier ici : uniquement la mise en page des scripts « feuille de
calcul » (main.py, NOTEBOOKS/**/*.py). Un seul jeu de fonctions pour que tous
les rapports du projet se lisent pareil.

    section(1, "Point de fonctionnement")
    kv("Fréquence de découpage", "100 kHz")
    table(["Front", "t [ns]", "vitesse"], [("t_ri", "11.3", "2.21 A/ns")])
    alert("error", "Tension au-dessus du V_DSS max — claquage.")
"""

from __future__ import annotations

import textwrap

WIDTH = 78  # largeur de référence des filets et du retour à la ligne
INDENT = "  "
LABEL_WIDTH = 38  # colonne des libellés de kv()

# Étiquette affichée par alert() selon le niveau. `prefix=` permet de la
# remplacer ponctuellement sans inventer un nouveau niveau.
ALERT_LABELS = {
    "error": "ERREUR",
    "warn": "ATTENTION",
    "info": "INFO",
}


def rule(char: str = "=", width: int = WIDTH) -> None:
    """Filet horizontal."""
    print(char * width)


def section(n: int | str | None, title: str, width: int = WIDTH) -> None:
    """Titre de section encadré. n = None pour un titre sans numéro."""
    print()
    rule("=", width)
    print(f"  {title}" if n is None else f"  {n}.  {title}")
    rule("=", width)


def kv(
    label: str,
    value,
    indent: str = INDENT,
    label_width: int = LABEL_WIDTH,
) -> None:
    """Une ligne « libellé ....... valeur »."""
    print(f"{indent}{label:<{label_width}}{value}")


def table(
    headers: list[str],
    rows: list[tuple],
    indent: str = INDENT,
) -> None:
    """
    Tableau à largeur de colonne automatique.

    Première colonne alignée à gauche (des libellés), les autres à droite (des
    nombres). Les cellules sont converties en str : formate les flottants
    toi-même avant d'appeler, c'est le seul endroit qui connaît les unités.
    """
    grid = [[str(c) for c in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in grid) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        justified = [
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        ]
        return indent + "  ".join(justified).rstrip()

    print(line(grid[0]))
    print(indent + "  ".join("-" * w for w in widths))
    for row in grid[1:]:
        print(line(row))


def paragraph(text: str, indent: str = INDENT, width: int = WIDTH) -> None:
    """Texte libre replié à la largeur du rapport."""
    for chunk in textwrap.wrap(text, width=width - len(indent)):
        print(indent + chunk)


def alert(
    level: str,
    message: str,
    prefix: str | None = None,
    indent: str = INDENT,
    width: int = WIDTH,
) -> None:
    """
    Message signalé, replié et aligné sous son étiquette.

    level  : clé de ALERT_LABELS ("error", "warn", "info")
    prefix : étiquette de remplacement, quand le niveau se dit autrement dans
             le contexte (ex. "CALCUL IMPOSSIBLE" pour une erreur bloquante)
    """
    label = prefix if prefix is not None else ALERT_LABELS.get(level, level.upper())
    lead = f"{indent}[{label}] "
    lines = textwrap.wrap(message, width=width - len(lead)) or [""]
    print(lead + lines[0])
    for extra in lines[1:]:
        print(" " * len(lead) + extra)
