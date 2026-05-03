import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold  (best for plotter — thick strokes)
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
]
_RENDER_PX = 120   # render each letter at this height — higher = more contour detail


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_binary(letter: str, size: int = _RENDER_PX) -> np.ndarray:
    """Render a letter as a white-on-black tight binary image (minimal padding)."""
    pad = size // 6          # tight padding — was size//2, which wasted 75% of canvas
    canvas = size + pad * 2
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    font = _load_font(size)

    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - tw) // 2 - bbox[0]
    y = (canvas - th) // 2 - bbox[1]
    draw.text((x, y), letter, fill=255, font=font)

    arr = np.array(img)
    return (arr > 64).astype(np.uint8) * 255


def _letter_to_outline_strokes(letter: str) -> list:
    """
    Render a letter, find its outline contours, normalize to bounding box.

    Returns list of stroke paths — each path is a list of (x, y) in [0, 1]
    normalized coordinates, with aspect ratio preserved.

    Uses outline tracing (not skeleton) so letters look correct even at
    very small scale (6–8 px on screen).
    """
    binary = _render_binary(letter, _RENDER_PX)
    if not np.any(binary):
        return []

    # Slight close to fill any anti-alias gaps at the rendered edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # RETR_CCOMP: outer boundary + holes (e.g. inside of 'A', 'e', 'a')
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Simplify each contour with Douglas-Peucker before normalising
    epsilon = max(1.0, _RENDER_PX * 0.015)   # 1.5% of render height
    simplified = []
    for c in contours:
        s = cv2.approxPolyDP(c, epsilon=epsilon, closed=True)
        if len(s) >= 2:
            simplified.append(s.reshape(-1, 2))

    if not simplified:
        return []

    # ── Compute ACTUAL letter bounding box (not canvas size) ──────────────
    # This is the critical fix: old code divided by canvas size (160px),
    # meaning letters were only 35–40% of their intended size on paper.
    all_pts = np.vstack(simplified)
    x_min, y_min = all_pts.min(axis=0).astype(float)
    x_max, y_max = all_pts.max(axis=0).astype(float)
    x_range = max(1.0, x_max - x_min)
    y_range = max(1.0, y_max - y_min)
    # Normalise by height so letter fills [0,1] vertically
    norm = y_range

    strokes = []
    for pts in simplified:
        path = [((float(p[0]) - x_min) / norm,
                 (float(p[1]) - y_min) / norm) for p in pts]
        # Close the contour so the outline is complete
        path.append(path[0])
        strokes.append(path)

    return strokes


def precompute_strokes(word: str) -> dict:
    """
    Pre-render and outline-trace every unique character in word.
    Returns dict: char → list of normalized stroke paths.
    """
    cache: dict = {}
    unique = sorted(set(word))
    print(f"  Pre-rendering {len(unique)} characters: {unique}")
    for ch in unique:
        strokes = _letter_to_outline_strokes(ch)
        cache[ch] = strokes
        pts_total = sum(len(s) for s in strokes)
        print(f"    '{ch}': {len(strokes)} contours, {pts_total} pts")
    return cache


def text_art_gcode(
    gray: np.ndarray,
    person_mask: np.ndarray,
    word: str,
    stroke_cache: dict,
    to_mm_fn,
    img_w: int,
    img_h: int,
    cell_h: int = 20,
    max_scale: float = 1.0,
    min_scale: float = 0.28,
    skip_brightness: int = 215,
    gamma: float = 1.6,
) -> list:
    """
    Tile the image with letters from word.
    Letter height scales with local darkness:
        dark  → max_scale × cell_h  (big letter)
        light → min_scale × cell_h  (tiny letter)
        white → blank

    gamma > 1 sharpens the dark/light contrast in letter sizing.
    """
    lines: list = []
    letter_idx = 0
    n = len(word)

    # Pre-build a contrast-enhanced gray using CLAHE for better tonal separation
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gray_c = clahe.apply(gray)

    row_y = cell_h // 2

    while row_y < img_h:
        col_x = 2

        while col_x < img_w:
            r = max(2, cell_h // 2)
            y0, y1 = max(0, row_y - r), min(img_h, row_y + r)
            x0, x1 = max(0, col_x - r), min(img_w, col_x + r)

            gray_reg = gray_c[y0:y1, x0:x1]
            mask_reg = person_mask[y0:y1, x0:x1]

            if gray_reg.size == 0:
                col_x += cell_h
                continue

            brightness = float(np.mean(gray_reg))
            person_frac = float(np.mean(mask_reg > 0)) if mask_reg.size > 0 else 0.0

            letter = word[letter_idx % n]
            letter_idx += 1

            # Skip bright / background areas
            if brightness > skip_brightness or person_frac < 0.25:
                col_x += int(cell_h * 0.55) + 1
                continue

            # Brightness → letter scale with gamma contrast boost
            t = float(np.clip(brightness / skip_brightness, 0.0, 1.0))
            t = t ** (1.0 / gamma)          # gamma > 1 → bigger dark letters
            scale = max_scale * (1.0 - t) + min_scale * t
            lh = max(4, int(scale * cell_h))   # letter height in image pixels
            lw = max(3, int(lh * 0.60))        # letter width  in image pixels

            strokes = stroke_cache.get(letter, [])
            top_y = row_y - lh // 2            # top edge of letter in image pixels

            for stroke in strokes:
                if len(stroke) < 2:
                    continue

                sx0, sy0 = stroke[0]
                px0 = col_x + sx0 * lh
                py0 = top_y + sy0 * lh
                xm, ym = to_mm_fn(px0, py0)
                lines += ["G0 Z5", f"G0 X{xm:.3f} Y{ym:.3f}", "G1 Z0 F100"]

                for sx, sy in stroke[1:]:
                    px = col_x + sx * lh
                    py = top_y + sy * lh
                    xm, ym = to_mm_fn(px, py)
                    lines.append(f"G1 X{xm:.3f} Y{ym:.3f} F4500")

                lines.append("G0 Z5")

            col_x += lw + 2

        row_y += cell_h

    return lines
