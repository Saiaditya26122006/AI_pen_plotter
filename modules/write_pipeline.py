"""Pipeline wrapper: text_write_gcode -> output/drawing_write.gcode"""

import os
from modules.text_writer import (
    detect_available_fonts,
    prompt_font_selection,
    resolve_font,
    text_write_gcode,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT_DIR   = os.path.join(_PROJECT_ROOT, "output")


def process_text_write(
    text: str,
    font_h: float    = 10.0,
    position: str    = "bottom",
    align: str       = "center",
    font_key: str    = None,
    font_path: str   = None,
    interactive: bool = True,
) -> str:
    """
    Render *text* as plotter G-code and save to output/drawing_write.gcode.

    Font resolution order:
      1. font_path (explicit path) — used directly if provided
      2. font_key  (key string or 1-based number) — resolved from the catalog
      3. If neither given and interactive=True  — CLI menu prompt
      4. Fallback to system default font

    Returns the complete G-code string.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # Resolve font
    resolved_path = font_path
    if not resolved_path and font_key:
        resolved_path = resolve_font(font_key)
        if not resolved_path:
            print(f"  Warning: font '{font_key}' not found — using default")

    if not resolved_path and interactive:
        resolved_path = prompt_font_selection()

    font_display = resolved_path or "(system default)"
    print(f"[1/2] Generating G-code  |  font: {os.path.basename(font_display) if resolved_path else font_display}")

    body = text_write_gcode(
        text,
        font_h    = font_h,
        position  = position,
        align     = align,
        font_path = resolved_path,
    )

    gcode_lines = ["G21", "G90", "G0 Z5"] + body + ["G0 Z5", "M2"]
    gcode_str   = "\n".join(gcode_lines) + "\n"

    out_path = os.path.join(_OUTPUT_DIR, "drawing_write.gcode")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gcode_str)

    draw_moves   = sum(1 for ln in gcode_lines if ln.startswith("G1 X"))
    travel_moves = sum(1 for ln in gcode_lines if ln.startswith("G0 X"))
    print(f"[2/2] Done -- {draw_moves} draw moves, {travel_moves} travel moves")
    print(f"  Saved -> {out_path}")
    return gcode_str
