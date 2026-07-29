from __future__ import annotations

import base64
import html as _html
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import ipywidgets as W
from IPython.display import HTML, clear_output, display

__all__ = [
    "FIELD_WIDTH",
    "LABEL_WIDTH",
    "button",
    "choice",
    "column",
    "figures",
    "gallery",
    "group",
    "image",
    "image_picker",
    "integer",
    "number",
    "on_change",
    "output",
    "panel",
    "row",
    "tabs",
    "text",
    "toggle",
    "visible_when",
]

# One label column and one field width for the whole project, so panels built
# in different notebooks line up. Retune here, not field by field.
#
# A panel whose labels genuinely do not fit — "h convection + radiation
# [W/(m²·K)]:" — passes label_width= and width= instead, once per panel rather
# than once per field. Everything inside one panel must share them, or the
# fields stop lining up, which is the whole point of having defaults.
LABEL_WIDTH = "185px"
FIELD_WIDTH = "380px"


def _geometry(label_width: str | None, width: str | None):
    """(style, layout) for a field, falling back to the project defaults."""
    return (
        {"description_width": label_width or LABEL_WIDTH},
        W.Layout(width=width or FIELD_WIDTH),
    )


# ============================================================================ #
#  Pictures
# ============================================================================ #

# Pictures are embedded as base64 data URIs rather than linked. Three reasons,
# and they all bite in practice: a linked path breaks the moment the notebook
# is exported to HTML or moved; .webp and friends are not served by every
# Jupyter front-end; and an embedded notebook still shows its figures to
# someone who does not have the repository.
_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
}

# Captions and frames inherit the notebook's own text colour instead of naming
# one. A hard-coded grey is unreadable in half the JupyterLab themes; this
# follows whichever theme is actually in use.
# Double quotes around the font name, not single: every style attribute below
# is delimited by single quotes, and 'Segoe UI' would close it early — the rest
# of the rule would then be parsed as markup.
_CAPTION_CSS = (
    'font: 500 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;'
    "color: currentColor; opacity: 0.65; margin: 6px 2px 0;"
)
_IMG_CSS = "width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; border-radius: 6px;"
#without scale 
#_IMG_CSS = "width: 100%; height: auto; display: block; border-radius: 6px;"

def _data_uri(path: Path) -> str:
    mime = _MIME.get(path.suffix.lower())
    if mime is None:
        known = ", ".join(sorted(_MIME))
        raise ValueError(f"unsupported image type {path.suffix!r} (known: {known})")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _missing(path) -> str:
    return (
        f"<p style='{_CAPTION_CSS}opacity:1;color:#d03b3b'>"
        f"Image not found: {_html.escape(str(path))}</p>"
    )


def _cell(source: str, caption: str | None) -> str:
    """
    One <figure>: the picture, and its caption when there is one.

    The caption is escaped — it is ordinary text a caller wrote, and a stray
    "<" or "&" in it would otherwise be read as markup and eat the rest.
    """
    block = f"<figure style='margin:0'><img src='{source}' style='{_IMG_CSS}'>"
    if caption:
        escaped = _html.escape(str(caption))
        block += f"<figcaption style='{_CAPTION_CSS}'>{escaped}</figcaption>"
    return block + "</figure>"


def image(source, width: int | None = 720, caption: str | None = None) -> HTML:
    """
    One picture from a file.

    source  : path to the image. Any type in _MIME, .webp included.
    width   : max width in px. None lets it fill the output area.
    caption : shown underneath, in the notebook's own text colour.
    """
    path = Path(source)
    if not path.exists():
        return HTML(_missing(path))
    limit = f"max-width:{width}px;" if width else ""
    return HTML(f"<div style='{limit}'>{_cell(_data_uri(path), caption)}</div>")


def gallery(
    sources: Sequence,
    columns: int = 2,
    width: int | None = None,
    captions: Sequence[str] | None = None,
) -> HTML:
    """
    Several pictures on a grid — the "two side by side, or four" case.

    columns  : pictures per row. 2 sources + columns=2 -> side by side.
               4 sources + columns=2 -> a 2x2 block. 4 + columns=4 -> one row.
    width    : max width of the WHOLE block in px; the columns split it.
    captions : one per source, or None for no captions.

    The grid wraps on a narrow screen, so a 4-wide row degrades instead of
    overflowing.
    """
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")
    if captions is not None and len(captions) != len(sources):
        raise ValueError(
            f"{len(captions)} captions for {len(sources)} images — pass one each"
        )

    cells = []
    for position, source in enumerate(sources):
        path = Path(source)
        caption = captions[position] if captions else None
        if not path.exists():
            cells.append(_missing(path))
        else:
            cells.append(_cell(_data_uri(path), caption))

    limit = f"max-width:{width}px;" if width else ""
    return HTML(
        f"<div style='display:grid;gap:14px;{limit}"
        f"grid-template-columns:repeat({columns},minmax(0,1fr))'>"
        + "".join(cells)
        + "</div>"
    )


def figures(
    figs: Sequence,
    columns: int = 2,
    width: int | None = None,
    captions: Sequence[str] | None = None,
) -> HTML:
    """
    The same grid, for matplotlib figures instead of files.

    This is how you put two plots side by side: the notebook's own inline
    display always stacks them one under the other, one per output.

    Figures go in as SVG — vector, so they stay sharp — through
    SRC/OTHERS/plot.py's figure_to_svg(), so they carry the same font and crop
    settings as a saved one. Nothing is written to disk, and the figures are
    NOT closed: close them yourself if it matters.
    """
    if captions is not None and len(captions) != len(figs):
        raise ValueError(
            f"{len(captions)} captions for {len(figs)} figures — pass one each"
        )

    # Imported here, not at the top: a notebook that only lays out widgets and
    # pictures should not pay for matplotlib.
    from SRC.OTHERS.plot import figure_to_svg

    sources = []
    for fig in figs:
        encoded = base64.b64encode(figure_to_svg(fig).encode("utf-8")).decode()
        sources.append(f"data:image/svg+xml;base64,{encoded}")

    cells = [
        _cell(source, captions[position] if captions else None)
        for position, source in enumerate(sources)
    ]
    limit = f"max-width:{width}px;" if width else ""
    return HTML(
        f"<div style='display:grid;gap:14px;{limit}"
        f"grid-template-columns:repeat({max(columns, 1)},minmax(0,1fr))'>"
        + "".join(cells)
        + "</div>"
    )


def image_picker(
    folder,
    pattern: str = "*",
    width: int | None = 720,
    default: str | None = None,
    label: str = "Image:",
) -> tuple[W.VBox, W.Dropdown]:
    """
    A dropdown over every picture in a folder, with the selected one below it.

    folder  : searched recursively; anything _MIME knows is offered.
    pattern : glob to narrow it down, e.g. "*.png".
    default : substring of the file name to select first (the rest stay in the
              list). Falls back to the first picture.

    Returns (box, dropdown) — same shape as button(). display() the box; read
    the selection from `dropdown.value`, which is a Path.
    """
    root = Path(folder)
    found = sorted(
        (p for p in root.rglob(pattern) if p.suffix.lower() in _MIME),
        key=lambda p: p.name,
    )
    if not found:
        empty = W.Dropdown(options=[], description=label)
        return W.VBox([W.HTML(_missing(f"no image under {root}"))]), empty

    first = next((p for p in found if default and default in p.name), found[0])
    picker = W.Dropdown(
        options=[(p.name, p) for p in found],
        value=first,
        description=label,
        style={"description_width": "80px"},
        layout=W.Layout(width="620px"),
    )
    out = W.Output()

    def refresh(_=None):
        with out:
            clear_output(wait=True)
            display(image(picker.value, width=width))

    picker.observe(refresh, names="value")
    refresh()
    return W.VBox([picker, out]), picker


# ============================================================================ #
#  Widgets
# ============================================================================ #
#
# Thin wrappers, one per widget worth repeating. Each one only fixes the label
# width and the field width — everything else is the ipywidgets object, and it
# is what gets returned, so `.value`, `.observe`, `.disabled` all work as usual.
#
# The bounds are the point of number() and integer(): a panel that cannot be
# handed a negative area or a 900 % duty cycle never has to check for one.


def number(
    label: str,
    value: float,
    step: float | None = None,
    minimum: float = 0.0,
    maximum: float = 1e9,
    label_width: str | None = None,
    width: str | None = None,
) -> W.BoundedFloatText:
    """A float field, bounded. Put the unit in the label: "Voltage [V]:"."""
    style, layout = _geometry(label_width, width)
    return W.BoundedFloatText(
        value=value,
        description=label,
        style=style,
        layout=layout,
        step=step,
        min=minimum,
        max=maximum,
    )


def integer(
    label: str,
    value: int,
    minimum: int = 0,
    maximum: int = 10_000,
    label_width: str | None = None,
    width: str | None = None,
) -> W.BoundedIntText:
    """An int field, bounded — counts: vias, turns, phases."""
    style, layout = _geometry(label_width, width)
    return W.BoundedIntText(
        value=value,
        description=label,
        style=style,
        layout=layout,
        min=minimum,
        max=maximum,
    )


def choice(
    label: str,
    options,
    value=None,
    label_width: str | None = None,
    width: str | None = None,
) -> W.Dropdown:
    """
    A dropdown.

    options : ["a", "b"], or [("What the user reads", real_value), ...] when
              the value you want back is not the label you want shown.
    """
    style, layout = _geometry(label_width, width)
    picker = W.Dropdown(options=options, description=label, style=style, layout=layout)
    if value is not None:
        picker.value = value
    return picker


def toggle(
    label: str,
    value: bool = False,
    width: str | None = None,
) -> W.Checkbox:
    """A checkbox. indent=False keeps it aligned with the fields above it."""
    return W.Checkbox(
        value=value,
        description=label,
        indent=False,
        layout=W.Layout(width=width or FIELD_WIDTH),
    )


def text(
    label: str,
    value: str = "",
    label_width: str | None = None,
    width: str | None = None,
) -> W.Text:
    """A free text field — a part number, a note, a file name."""
    style, layout = _geometry(label_width, width)
    return W.Text(value=value, description=label, style=style, layout=layout)


def output() -> W.Output:
    """An output area to print into. Usually you want button() instead."""
    return W.Output()


def button(
    label: str,
    on_click: Callable[[], None],
    icon: str = "calculator",
    out: W.Output | None = None,
) -> tuple[W.Button, W.Output]:
    """
    A button and the output area it writes into.

    Clicking clears the area and runs `on_click()` inside it — which is what
    keeps a re-run replacing the previous result instead of stacking under it.
    `on_click` takes no argument: read the widgets it needs from the enclosing
    scope.

    Returns (button, output). Display both — see panel().
    """
    area = out or W.Output()
    go = W.Button(
        description=label,
        button_style="primary",
        icon=icon,
        layout=W.Layout(width="200px", height="38px"),
    )

    def run(_=None):
        with area:
            clear_output(wait=True)
            on_click()

    go.on_click(run)
    return go, area


def on_change(widget, callback: Callable[[], None]) -> None:
    """
    Run `callback()` whenever the widget's value changes.

    Wraps `.observe(..., names="value")`, whose handler signature is the thing
    everybody gets wrong. `callback` takes no argument.
    """
    widget.observe(lambda _: callback(), names="value")


def visible_when(target, control, *values) -> None:
    """
    Show `target` only while `control.value` is one of `values`.

    The alternative — leaving a field on screen and disabling it, or worse
    leaving it live — is how a panel ends up showing a number that took no
    part in the result. Applied immediately, then on every change.

    visible_when(rth_field, mode_dropdown, "manual")
    visible_when(pcb_box, mode_dropdown, "pcb", "hybrid")   # several values
    """

    def sync(_=None):
        target.layout.display = "" if control.value in values else "none"

    control.observe(sync, names="value")
    sync()


# ============================================================================ #
#  Layout
# ============================================================================ #


def group(title: str, widgets: Iterable) -> W.VBox:
    """A titled column of widgets — one block of a panel."""
    return W.VBox(
        [W.HTML(f"<b style='font-size:13px'>{title}</b>"), *widgets],
        layout=W.Layout(margin="0 28px 14px 0"),
    )


def row(*items) -> W.HBox:
    """
    Side by side, wrapping onto the next line when the window is too narrow.

    Wrapping is the default on purpose: a panel that overflows horizontally
    hides its right-hand fields with no scrollbar to hint at it.
    """
    return W.HBox(list(items), layout=W.Layout(flex_flow="row wrap"))


def column(*items) -> W.VBox:
    """Stacked, top to bottom."""
    return W.VBox(list(items))


def tabs(pages: dict) -> W.Tab:
    """
    Tabbed pages, from a {title: widget} mapping.

    For settings that repeat per instance — one tab per package face, per
    winding, per phase — where showing them all at once would drown the panel.
    """
    box = W.Tab(children=list(pages.values()), layout=W.Layout(width="470px"))
    for position, title in enumerate(pages):
        box.set_title(position, title)
    return box


def panel(*items, run_now: W.Button | None = None) -> W.VBox:
    """
    Assemble and display a panel in one call.

    run_now : a button to click once the panel is on screen, so the notebook
              opens on a filled-in result instead of an empty box. Pass the
              button() gave you — clicking it is what routes the output into
              its own area rather than into the cell.

    Returns the VBox, already displayed.
    """
    box = W.VBox(list(items))
    display(box)
    if run_now is not None:
        run_now.click()
    return box
