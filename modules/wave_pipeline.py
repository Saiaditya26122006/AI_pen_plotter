import os

import cv2

from modules.subject_detection import detect_persons_advanced
from modules.wave_art import wave_art_gcode

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
_PAPER_W = 287.0
_PAPER_H = 410.0
_OFFSET = 5.0


def _make_to_mm(img_w: int, img_h: int):
    scale = min((_PAPER_W - 2 * _OFFSET) / img_w, (_PAPER_H - 2 * _OFFSET) / img_h)
    drawn_w = img_w * scale
    drawn_h = img_h * scale
    off_x = _OFFSET + (_PAPER_W - 2 * _OFFSET - drawn_w) / 2.0
    off_y = _OFFSET + (_PAPER_H - 2 * _OFFSET - drawn_h) / 2.0

    def to_mm(px: float, py: float):
        return (off_x + px * scale, off_y + (img_h - 1 - py) * scale)

    return to_mm


def process_wave_art(
    image_path: str,
    row_spacing: int = 8,
) -> str:
    """
    Full wave-art pipeline:
      1. YOLOv8 person detection + crop
      2. Sine-wave modulation G-code (amplitude encodes brightness)
      3. Save to output/drawing_wave.gcode
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    print("[1/3] YOLOv8 person detection...")
    img_bgr, person_mask = detect_persons_advanced(image_path)
    h, w = img_bgr.shape[:2]
    print(f"  Canvas: {w}x{h} px")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    to_mm = _make_to_mm(w, h)

    rows = h // row_spacing
    print(f"[2/3] Generating wave art ({rows} scan lines, {row_spacing}px row spacing)...")
    gcode = ["G21", "G90", "G0 Z5"]
    wave_lines = wave_art_gcode(gray, person_mask, to_mm, w, h, row_spacing=row_spacing)
    gcode.extend(wave_lines)
    gcode.append("M2")

    gcode_str = "\n".join(gcode) + "\n"

    out_path = os.path.join(_OUTPUT_DIR, "drawing_wave.gcode")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gcode_str)

    draw_moves  = sum(1 for ln in gcode if ln.startswith("G1 X"))
    travel_moves = sum(1 for ln in gcode if ln.startswith("G0 X"))
    print(f"[3/3] Done -- {draw_moves} draw moves, {travel_moves} travel moves")
    print(f"  Saved -> {out_path}")
    return gcode_str
