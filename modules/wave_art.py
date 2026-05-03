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
    wavelength: float = None,       # px; default = row_spacing * 8
    max_amplitude: float = None,    # px; default = row_spacing * 1.4
    min_amplitude: float = 0.0,     # px; bright areas → completely flat
    smoothing: float = 2.0,         # Gaussian sigma — lower keeps more face detail
    gamma: float = 2.5,             # high gamma = dark areas get dramatically taller waves
) -> list:
    """
    Joy-Division-style sine-wave portrait.

    Fixes vs v1:
    - All rows scan LEFT → RIGHT only (bidirectional caused a crisscross net).
    - Local contrast normalization inside the person mask: the darkest area
      of the face always maps to max_amplitude and the lightest always maps
      to min_amplitude, regardless of absolute image brightness.
      This makes eyes / shadows / hair produce tall dramatic peaks while
      forehead highlights stay flat.
    - Longer wavelength (8× row_spacing) for smoother, cleaner waves.
    - Higher gamma (2.5) for more aggressive dark-area emphasis.
    """
    if wavelength is None:
        wavelength = row_spacing * 8.0
    if max_amplitude is None:
        max_amplitude = row_spacing * 1.4

    # ── Contrast enhancement ──────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_c = clahe.apply(gray)

    # ── 2-D Gaussian smooth (mild — keeps face features sharp) ────────────
    gray_f = gaussian_filter(gray_c.astype(np.float64), sigma=smoothing)

    # ── Local contrast normalisation inside the person mask ───────────────
    # Stretch the brightness range that actually exists on the person so
    # the full amplitude range (0 → max_amplitude) is always used.
    person_vals = gray_f[person_mask > 0]
    if person_vals.size > 10:
        p_dark  = np.percentile(person_vals, 3)   # darkest 3 %
        p_light = np.percentile(person_vals, 97)  # lightest 97 %
        span = max(p_light - p_dark, 1.0)
        # gray_norm: 0 = darkest part of face, 1 = lightest part of face
        gray_norm = np.clip((gray_f - p_dark) / span, 0.0, 1.0)
    else:
        gray_norm = np.clip(gray_f / 255.0, 0.0, 1.0)

    # ── Smooth mask for clean wave clip at person boundary ────────────────
    mask_f = gaussian_filter((person_mask > 0).astype(np.float64), sigma=2.0)

    # ── Build amplitude map ───────────────────────────────────────────────
    # t=0 (darkest) → max_amplitude   t=1 (lightest) → min_amplitude
    t = gray_norm ** (1.0 / gamma)     # gamma > 1 makes dark areas dominate
    amp_map = max_amplitude * (1.0 - t) + min_amplitude * t

    # ── Scan lines ────────────────────────────────────────────────────────
    lines: list = []
    xs = np.arange(img_w, dtype=np.float64)

    for row_idx, row_y in enumerate(range(row_spacing // 2, img_h, row_spacing)):
        amp_row  = amp_map[row_y, :]
        mask_row = mask_f[row_y, :]

        if not (mask_row > 0.15).any():
            continue

        # Golden-ratio phase per row → prevents vertical stripes
        phase = (row_idx * 0.6180339887 % 1.0) * 2.0 * np.pi

        wave_dys  = amp_row * np.sin(2.0 * np.pi * xs / wavelength + phase)
        actual_ys = np.clip(row_y + wave_dys, 0.0, img_h - 1.0)

        # ALL rows LEFT → RIGHT — no bidirectional (bidirectional caused crisscross net)
        in_stroke = False
        for x in range(img_w):
            if mask_row[x] <= 0.15:
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
