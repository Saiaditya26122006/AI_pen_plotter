"""
Pen plotter text writer: converts a text string to plotter G-code.

Uses outline-traced font glyphs (same rendering pipeline as text_art.py)
placed along a readable baseline at a fixed size, with word-wrap, font
selection, and paper positioning.
"""

import os
from modules.text_art import precompute_strokes

_PAPER_W = 287.0
_PAPER_H = 410.0
_MARGIN  = 10.0

_LETTER_SPACING = 0.12   # inter-glyph gap  (fraction of font_h)
_WORD_SPACING   = 0.40   # space character  (fraction of font_h)
_LINE_SPACING   = 1.55   # line pitch       (multiple  of font_h)

# ── Font catalog ──────────────────────────────────────────────────────────────
# Each entry: key (CLI/API id), name, path, tag (short), desc (one-liner)
FONT_CATALOG = [
    {
        "key":  "arial-bold",
        "name": "Arial Bold",
        "path": "C:/Windows/Fonts/arialbd.ttf",
        "tag":  "Modern Sans",
        "desc": "Clean, neutral — works at any size",
    },
    {
        "key":  "impact",
        "name": "Impact",
        "path": "C:/Windows/Fonts/impact.ttf",
        "tag":  "Heavy",
        "desc": "Tall, condensed — dramatic bold statements",
    },
    {
        "key":  "times-bold",
        "name": "Times New Roman Bold",
        "path": "C:/Windows/Fonts/timesbd.ttf",
        "tag":  "Classic Serif",
        "desc": "Traditional, elegant, literary",
    },
    {
        "key":  "calibri-bold",
        "name": "Calibri Bold",
        "path": "C:/Windows/Fonts/calibrib.ttf",
        "tag":  "Modern Office",
        "desc": "Friendly, rounded, contemporary",
    },
    {
        "key":  "verdana-bold",
        "name": "Verdana Bold",
        "path": "C:/Windows/Fonts/verdanab.ttf",
        "tag":  "Wide Sans",
        "desc": "Wide spacing — very legible at small sizes",
    },
    {
        "key":  "georgia-bold",
        "name": "Georgia Bold",
        "path": "C:/Windows/Fonts/georgiab.ttf",
        "tag":  "Warm Serif",
        "desc": "Elegant, warm, book-like quality",
    },
    {
        "key":  "courier-bold",
        "name": "Courier New Bold",
        "path": "C:/Windows/Fonts/courbd.ttf",
        "tag":  "Typewriter",
        "desc": "Monospace, retro, technical look",
    },
    {
        "key":  "comic-sans",
        "name": "Comic Sans MS",
        "path": "C:/Windows/Fonts/comic.ttf",
        "tag":  "Casual",
        "desc": "Informal, handwritten-style feel",
    },
    {
        "key":  "trebuchet-bold",
        "name": "Trebuchet MS Bold",
        "path": "C:/Windows/Fonts/trebucbd.ttf",
        "tag":  "Humanist Sans",
        "desc": "Friendly, modern, clear letterforms",
    },
    {
        "key":  "segoe-bold",
        "name": "Segoe UI Bold",
        "path": "C:/Windows/Fonts/segoeuib.ttf",
        "tag":  "Modern UI",
        "desc": "Clean, contemporary, Windows native",
    },
    {
        "key":  "tahoma-bold",
        "name": "Tahoma Bold",
        "path": "C:/Windows/Fonts/tahomabd.ttf",
        "tag":  "Screen Sans",
        "desc": "Compact, crisp, high legibility",
    },
    {
        "key":  "century-gothic",
        "name": "Century Gothic",
        "path": "C:/Windows/Fonts/GOTHIC.TTF",
        "tag":  "Geometric",
        "desc": "Circular letterforms — artistic and airy",
    },
]


def detect_available_fonts() -> list:
    """Return only catalog entries whose font file exists on this machine."""
    return [f for f in FONT_CATALOG if os.path.exists(f["path"])]


def resolve_font(font_key: str) -> str | None:
    """
    Resolve a font key (or 1-based number string) to a font file path.
    Returns None if not found or file missing.
    """
    available = detect_available_fonts()
    if not available:
        return None

    # Try as 1-based index
    try:
        idx = int(font_key) - 1
        if 0 <= idx < len(available):
            return available[idx]["path"]
    except (ValueError, TypeError):
        pass

    # Try as key or name (case-insensitive)
    key_lo = (font_key or "").lower().strip()
    for f in available:
        if f["key"] == key_lo or f["name"].lower() == key_lo:
            return f["path"]

    return None


def prompt_font_selection() -> str | None:
    """
    Interactive CLI font picker.  Prints a numbered menu, reads user input,
    returns the chosen font file path (or None to use the default).
    """
    available = detect_available_fonts()
    if not available:
        print("  (no additional fonts found — using default)")
        return None

    col_name = max(len(f["name"]) for f in available) + 2
    col_tag  = max(len(f["tag"])  for f in available) + 2

    print()
    print("  Available fonts:")
    print("  " + "-" * 62)
    for i, f in enumerate(available, 1):
        name_pad = f["name"].ljust(col_name)
        tag_pad  = f"[{f['tag']}]".ljust(col_tag + 2)
        print(f"  {i:2d}. {name_pad} {tag_pad} {f['desc']}")
    print("  " + "-" * 62)

    raw = input("  Choose a font [1-%d, default 1]: " % len(available)).strip()
    if not raw:
        chosen = available[0]
    else:
        try:
            idx = int(raw) - 1
            chosen = available[max(0, min(idx, len(available) - 1))]
        except ValueError:
            chosen = available[0]

    print(f"  -> Using: {chosen['name']}")
    return chosen["path"]


# ── Core G-code generation ────────────────────────────────────────────────────

def _glyph_advance(strokes: list) -> float:
    if not strokes:
        return 0.5
    xs = [p[0] for stroke in strokes for p in stroke]
    return (max(xs) - min(xs)) if xs else 0.5


def _measure_line(line: str, cache: dict, font_h: float) -> float:
    total = 0.0
    for ch in line:
        if ch == " ":
            total += _WORD_SPACING * font_h
        else:
            total += _glyph_advance(cache.get(ch, [])) * font_h + _LETTER_SPACING * font_h
    return total


def _word_wrap(text: str, cache: dict, font_h: float, max_w: float) -> list:
    words = text.split()
    lines, cur, cur_w = [], "", 0.0

    for word in words:
        word_w = _measure_line(word, cache, font_h)
        sep_w  = _WORD_SPACING * font_h if cur else 0.0

        if cur and cur_w + sep_w + word_w > max_w:
            lines.append(cur)
            cur, cur_w = word, word_w
        else:
            if cur:
                cur   += " "
                cur_w += sep_w
            cur   += word
            cur_w += word_w

    if cur:
        lines.append(cur)
    return lines or [""]


def text_write_gcode(
    text: str,
    font_h: float    = 10.0,
    position: str    = "bottom",
    align: str       = "center",
    font_path: str   = None,
    paper_w: float   = _PAPER_W,
    paper_h: float   = _PAPER_H,
    margin: float    = _MARGIN,
    feed: int        = 4500,
) -> list:
    """
    Convert *text* to a list of G-code lines that write it on paper.

    Parameters
    ----------
    text      : String to write.  Use \\n for explicit line breaks.
    font_h    : Glyph height in mm.
    position  : 'top' | 'center' | 'bottom'
    align     : 'left' | 'center' | 'right'
    font_path : Absolute path to a .ttf/.otf file, or None for the default.
    paper_w/h : Paper dimensions in mm (default A3 portrait 287 x 410).
    margin    : Minimum edge keep-out in mm.
    feed      : Drawing feed rate in mm/min.

    Returns a list of G-code strings (no header/footer — caller wraps in G21/G90/M2).
    Coordinate convention: Y=0 at bottom of paper, increases upward.
    """
    usable_w = paper_w - 2 * margin
    line_h   = _LINE_SPACING * font_h

    unique_chars = sorted(set(text) - {" ", "\n", "\r", "\t"})
    cache = precompute_strokes("".join(unique_chars), font_path=font_path)

    display_lines: list = []
    for para in text.replace("\r\n", "\n").split("\n"):
        if para.strip():
            display_lines.extend(_word_wrap(para, cache, font_h, usable_w))
        else:
            display_lines.append("")

    n_lines = len(display_lines)
    block_h = n_lines * line_h

    if position == "top":
        y_top = paper_h - margin
    elif position == "center":
        y_top = paper_h / 2.0 + block_h / 2.0
    else:   # bottom
        y_top = margin + block_h

    gcode: list = []

    for i, line in enumerate(display_lines):
        if not line.strip():
            continue

        line_top_y = y_top - i * line_h
        line_w     = _measure_line(line, cache, font_h)

        if align == "left":
            x_cursor = margin
        elif align == "right":
            x_cursor = paper_w - margin - line_w
        else:
            x_cursor = paper_w / 2.0 - line_w / 2.0

        for ch in line:
            if ch == " ":
                x_cursor += _WORD_SPACING * font_h
                continue

            strokes = cache.get(ch, [])
            if not strokes:
                x_cursor += _WORD_SPACING * font_h
                continue

            for stroke in strokes:
                if len(stroke) < 2:
                    continue

                sx0, sy0 = stroke[0]
                xm0 = x_cursor + sx0 * font_h
                ym0 = line_top_y - sy0 * font_h   # sy=0 top, sy=1 bottom; paper Y up

                gcode += [
                    "G0 Z5",
                    f"G0 X{xm0:.3f} Y{ym0:.3f}",
                    "G1 Z0 F100",
                ]
                for sx, sy in stroke[1:]:
                    gcode.append(
                        f"G1 X{x_cursor + sx * font_h:.3f} Y{line_top_y - sy * font_h:.3f} F{feed}"
                    )
                gcode.append("G0 Z5")

            x_cursor += _glyph_advance(strokes) * font_h + _LETTER_SPACING * font_h

    return gcode
