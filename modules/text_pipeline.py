import os

import cv2
import numpy as np

from modules.subject_detection import detect_persons_advanced
from modules.text_art import build_brightness_palette, precompute_strokes, text_art_gcode

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


def process_text_art(
    image_path: str,
    word: str = "VARSHEETHvarsheeth",
    cell_h: int = 14,
    mode: str = "word",
    chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
) -> str:
    """
    Full text-art pen plotter pipeline:
      1. YOLOv8 multi-person detection + crop
      2. Pre-render letter skeletons
      3. Tile image with letters, size ∝ local darkness
      4. Save G-code to output/drawing_text.gcode

    mode='word'  → cycle letters from `word` (original behaviour)
    mode='auto'  → pick letter by brightness using sorted A-Z density palette
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Person detection ───────────────────────────────────────────
    print("[1/4] YOLOv8 person detection...")
    img_bgr, person_mask = detect_persons_advanced(image_path)
    h, w = img_bgr.shape[:2]
    print(f"  Canvas: {w}×{h} px")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)

    to_mm = _make_to_mm(w, h)

    # ── Step 2: Build palette / pre-render letter skeletons ───────────────
    palette = None
    if mode == "auto":
        print(f"[2/4] Building brightness palette from chars: {chars}")
        palette = build_brightness_palette(chars)
        stroke_cache = precompute_strokes("".join(palette))
    else:
        print(f"[2/4] Pre-rendering strokes for word: {set(word)}")
        stroke_cache = precompute_strokes(word)

    # ── Step 3: Tile image with text ──────────────────────────────────────
    rows = h // cell_h
    print(f"[3/4] Tiling {w}×{h} image → ~{rows} rows of text...")
    gcode = ["G21", "G90", "G0 Z5"]
    art_lines = text_art_gcode(
        gray_enh,
        person_mask,
        word,
        stroke_cache,
        to_mm,
        w,
        h,
        cell_h=cell_h,
        palette=palette,
    )
    gcode.extend(art_lines)
    gcode.append("M2")

    gcode_str = "\n".join(gcode) + "\n"

    # ── Step 4: Save ──────────────────────────────────────────────────────
    out_path = os.path.join(_OUTPUT_DIR, "drawing_text.gcode")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gcode_str)

    draw_moves = sum(1 for ln in gcode if ln.startswith("G1 X"))
    travel_moves = sum(1 for ln in gcode if ln.startswith("G0 X"))
    print(f"[4/4] Done — {draw_moves} draw moves, {travel_moves} travel moves")
    print(f"  Saved → {out_path}")
    return gcode_str
