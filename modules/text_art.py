import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/times.ttf",
]
_RENDER_SIZE = 80  # render letters at this pixel height for skeleton extraction


def _load_font(size: int = 80) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_binary(letter: str, size: int = 80) -> np.ndarray:
    """Render one letter as white-on-black binary numpy array."""
    pad = size
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


def _trace_skeleton(skeleton: np.ndarray) -> list:
    """
    Trace a skeleton (1-pixel-wide) image into ordered point paths.
    Each path is a list of (x, y) integer pixel coordinates forming one
    continuous pen stroke.
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return []

    remaining = set(zip(xs.tolist(), ys.tolist()))
    paths: list = []

    while remaining:
        # Start each new path from the topmost-leftmost unvisited pixel
        start = min(remaining, key=lambda p: (p[1], p[0]))
        path = [start]
        remaining.discard(start)

        while True:
            cx, cy = path[-1]
            best = None
            best_d = 3  # max squared distance for 8-connectivity is 2
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in remaining:
                        d = dx * dx + dy * dy
                        if d < best_d:
                            best_d = d
                            best = nb
            if best is None:
                break
            path.append(best)
            remaining.discard(best)

        if len(path) >= 2:
            paths.append(path)

    return paths


def precompute_strokes(word: str) -> dict:
    """
    Pre-render and skeleton-trace every unique character in word.
    Returns dict: char → list of normalized stroke paths.
    Each stroke path = list of (x, y) in [0.0, 1.0] normalized coords.
    """
    cache: dict = {}
    unique_chars = sorted(set(word))
    print(f"  Pre-rendering {len(unique_chars)} unique characters...")

    for ch in unique_chars:
        binary = _render_binary(ch, _RENDER_SIZE)

        if not np.any(binary):
            cache[ch] = []
            continue

        # Dilate slightly to ensure stroke connectivity before thinning
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, k, iterations=1)

        # Thin to 1-pixel skeleton
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skel = cv2.ximgproc.thinning(binary)
        else:
            skel = cv2.erode(binary, k, iterations=4)

        paths = _trace_skeleton(skel)

        h, w = skel.shape
        norm = float(max(h, w))
        norm_paths = [[(px / norm, py / norm) for px, py in p] for p in paths]
        cache[ch] = norm_paths

    return cache


def text_art_gcode(
    gray: np.ndarray,
    person_mask: np.ndarray,
    word: str,
    stroke_cache: dict,
    to_mm_fn,
    img_w: int,
    img_h: int,
    cell_h: int = 14,
    max_scale: float = 1.0,
    min_scale: float = 0.22,
    skip_brightness: int = 218,
) -> list:
    """
    Tile the image with letters from word, with letter height proportional
    to local darkness.

    Dark pixel   → big letter (max_scale × cell_h pixels tall)
    Light pixel  → tiny letter (min_scale × cell_h pixels tall)
    Very bright  → blank (no letter drawn)
    Background   → blank (outside person mask)
    """
    lines: list = []
    letter_idx = 0
    n = len(word)

    row_y = cell_h // 2

    while row_y < img_h:
        col_x = 2

        while col_x < img_w:
            r = max(2, cell_h // 2)
            y0, y1 = max(0, row_y - r), min(img_h, row_y + r)
            x0, x1 = max(0, col_x - r), min(img_w, col_x + r)

            gray_reg = gray[y0:y1, x0:x1]
            mask_reg = person_mask[y0:y1, x0:x1]

            if gray_reg.size == 0:
                col_x += cell_h // 2
                continue

            brightness = float(np.mean(gray_reg))
            person_frac = float(np.mean(mask_reg > 0)) if mask_reg.size > 0 else 0.0

            letter = word[letter_idx % n]
            letter_idx += 1

            # Skip background and near-white regions
            if brightness > skip_brightness or person_frac < 0.25:
                col_x += int(cell_h * 0.55) + 1
                continue

            # Brightness → letter scale
            t = float(np.clip(brightness / skip_brightness, 0.0, 1.0))
            scale = max_scale * (1.0 - t) + min_scale * t
            lh = max(3, int(scale * cell_h))   # letter height in pixels
            lw = max(2, int(lh * 0.62))         # letter width in pixels

            strokes = stroke_cache.get(letter, [])
            top_y = row_y - lh // 2             # top edge of letter in image pixels

            for stroke in strokes:
                if len(stroke) < 2:
                    continue

                sx, sy = stroke[0]
                px0 = col_x + sx * lh
                py0 = top_y + sy * lh
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
