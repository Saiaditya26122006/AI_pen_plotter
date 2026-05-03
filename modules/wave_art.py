import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def wave_art_gcode(
    gray: np.ndarray,
    person_mask: np.ndarray,
    to_mm_fn,
    img_w: int,
    img_h: int,
    row_spacing: int = 8,
    wavelength: float = None,      # px; default = row_spacing * 5
    max_amplitude: float = None,   # px; default = row_spacing * 0.88
    min_amplitude: float = None,   # px; default = row_spacing * 0.04
    smoothing: float = 5.0,        # horizontal Gaussian sigma in px
    gamma: float = 1.4,
) -> list:
    """
    Joy-Division-style wave modulation G-code.

    Each horizontal scan line is a sine wave whose amplitude encodes local brightness:
        dark pixels  → tall wave (high amplitude)
        bright pixels → nearly flat wave (low amplitude)

    Quality features:
    - 2D Gaussian pre-smoothing (horizontal + vertical) so waves follow face
      contours smoothly rather than tracking pixel-level noise.
    - Golden-ratio phase shift per row prevents vertical stripe artifacts.
    - Bidirectional scan (L→R / R→L alternating) minimizes pen travel.
    - Waves clip cleanly at person mask boundaries.
    """
    if wavelength is None:
        wavelength = row_spacing * 5.0
    if max_amplitude is None:
        max_amplitude = row_spacing * 0.88
    if min_amplitude is None:
        min_amplitude = row_spacing * 0.04

    # ── Contrast enhancement ──────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gray_c = clahe.apply(gray)

    # ── 2-D Gaussian pre-smooth ───────────────────────────────────────────
    # sigma_y (vertical) = row_spacing/2  → blends adjacent scan bands
    # sigma_x (horizontal) = smoothing    → smooths amplitude driver along row
    gray_smooth = gaussian_filter(
        gray_c.astype(np.float64),
        sigma=[row_spacing * 0.5, smoothing],
    )

    # ── Pre-smooth mask too (soft edges → cleaner wave clip) ──────────────
    mask_smooth = gaussian_filter(
        (person_mask > 0).astype(np.float64),
        sigma=[row_spacing * 0.5, 2.0],
    )

    lines: list = []
    row_ys = list(range(row_spacing // 2, img_h, row_spacing))
    xs = np.arange(img_w, dtype=np.float64)

    for row_idx, row_y in enumerate(row_ys):
        # Sample pre-smoothed rows
        gray_row = gray_smooth[row_y, :]   # shape (img_w,)
        mask_row = mask_smooth[row_y, :]   # shape (img_w,), values 0..1

        if not (mask_row > 0.15).any():
            continue

        # Gamma-corrected brightness → amplitude
        t = np.clip(gray_row / 255.0, 0.0, 1.0) ** (1.0 / gamma)
        amplitudes = max_amplitude * (1.0 - t) + min_amplitude * t

        # Golden-ratio phase offset — prevents vertical stripes
        phase = (row_idx * 0.6180339887) % 1.0 * 2.0 * np.pi

        wave_dys = amplitudes * np.sin(2.0 * np.pi * xs / wavelength + phase)
        actual_ys = np.clip(row_y + wave_dys, 0.0, img_h - 1.0)

        # Bidirectional: odd rows scan right → left
        x_range = range(img_w - 1, -1, -1) if row_idx % 2 else range(img_w)

        in_stroke = False
        for x in x_range:
            inside = mask_row[x] > 0.15

            if not inside:
                if in_stroke:
                    lines.append("G0 Z5")
                    in_stroke = False
                continue

            xm, ym = to_mm_fn(float(x), float(actual_ys[x]))

            if not in_stroke:
                lines += [f"G0 X{xm:.3f} Y{ym:.3f}", "G1 Z0 F100"]
                in_stroke = True
            else:
                lines.append(f"G1 X{xm:.3f} Y{ym:.3f} F4500")

        if in_stroke:
            lines.append("G0 Z5")

    return lines
