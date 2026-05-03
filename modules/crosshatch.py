import numpy as np
import cv2


def crosshatch_region(
    gray: np.ndarray,
    mask: np.ndarray,
    to_mm_fn,
    img_w: int,
    img_h: int,
    base_spacing: int = 8,
) -> list:
    """
    Adaptive multi-directional crosshatching within a masked region.
    Dark zones get 4 angle layers, mid-tones 2, lights 1, very light none.
    """
    if not np.any(mask > 0):
        return []

    lines: list = []

    dark_m = np.where((mask > 0) & (gray < 60), 255, 0).astype(np.uint8)
    mid_dark_m = np.where((mask > 0) & (gray >= 60) & (gray < 120), 255, 0).astype(np.uint8)
    mid_m = np.where((mask > 0) & (gray >= 120) & (gray < 180), 255, 0).astype(np.uint8)
    light_m = np.where((mask > 0) & (gray >= 180), 255, 0).astype(np.uint8)

    if np.any(dark_m > 0):
        for angle in (0, 45, 90, 135):
            lines.extend(_hatch_angle(gray, dark_m, angle, base_spacing,
                                       to_mm_fn, img_w, img_h))

    if np.any(mid_dark_m > 0):
        for angle in (45, 135):
            lines.extend(_hatch_angle(gray, mid_dark_m, angle, base_spacing * 2,
                                       to_mm_fn, img_w, img_h))

    if np.any(mid_m > 0):
        lines.extend(_hatch_angle(gray, mid_m, 45, base_spacing * 3,
                                   to_mm_fn, img_w, img_h))

    if np.any(light_m > 0):
        lines.extend(_hatch_angle(gray, light_m, 45, base_spacing * 5,
                                   to_mm_fn, img_w, img_h))

    return lines


def _hatch_angle(
    gray: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
    spacing: int,
    to_mm_fn,
    img_w: int,
    img_h: int,
) -> list:
    """Draw parallel lines at angle_deg, only within mask pixels."""
    if not np.any(mask > 0):
        return []

    angle_rad = np.radians(angle_deg)
    dx = np.cos(angle_rad)   # along-line direction
    dy = np.sin(angle_rad)
    nx = -dy                 # perpendicular direction
    ny = dx

    diag = int(np.ceil(np.sqrt(img_w ** 2 + img_h ** 2)))
    cx, cy = img_w / 2.0, img_h / 2.0

    lines: list = []

    for d in range(-diag, diag + 1, spacing):
        # A point on this hatch line
        px = cx + d * nx
        py = cy + d * ny

        # Parametric clip: find t range where (px+t*dx, py+t*dy) stays in image
        t_min, t_max = float(-diag), float(diag)

        if abs(dx) > 1e-9:
            ta = (0.0 - px) / dx
            tb = (img_w - 1.0 - px) / dx
            t_min = max(t_min, min(ta, tb))
            t_max = min(t_max, max(ta, tb))
        else:
            if px < 0 or px >= img_w:
                continue

        if abs(dy) > 1e-9:
            ta = (0.0 - py) / dy
            tb = (img_h - 1.0 - py) / dy
            t_min = max(t_min, min(ta, tb))
            t_max = min(t_max, max(ta, tb))
        else:
            if py < 0 or py >= img_h:
                continue

        if t_min >= t_max:
            continue

        n_steps = max(2, int(t_max - t_min) + 1)
        ts = np.linspace(t_min, t_max, n_steps)
        xs = px + ts * dx
        ys = py + ts * dy

        valid = (xs >= 0) & (xs < img_w) & (ys >= 0) & (ys < img_h)
        xs, ys = xs[valid], ys[valid]
        if len(xs) < 2:
            continue

        xi = np.clip(xs.astype(int), 0, img_w - 1)
        yi = np.clip(ys.astype(int), 0, img_h - 1)
        in_mask = mask[yi, xi] > 0

        seg_x: list = []
        seg_y: list = []
        for x, y, im in zip(xs, ys, in_mask):
            if im:
                seg_x.append(x)
                seg_y.append(y)
            else:
                if len(seg_x) >= 2:
                    lines.extend(_seg_gcode(seg_x, seg_y, gray, to_mm_fn, img_w, img_h))
                seg_x, seg_y = [], []
        if len(seg_x) >= 2:
            lines.extend(_seg_gcode(seg_x, seg_y, gray, to_mm_fn, img_w, img_h))

    return lines


def _seg_gcode(seg_x, seg_y, gray, to_mm_fn, img_w, img_h):
    x0, y0 = to_mm_fn(seg_x[0], seg_y[0])
    out = ["G0 Z5", f"G0 X{x0:.3f} Y{y0:.3f}", "G1 Z0 F100"]
    for x, y in zip(seg_x[1:], seg_y[1:]):
        bv = int(gray[int(np.clip(y, 0, img_h - 1)),
                       int(np.clip(x, 0, img_w - 1))])
        feed = 1200 if bv < 80 else (2000 if bv < 160 else 3000)
        xm, ym = to_mm_fn(x, y)
        out.append(f"G1 X{xm:.3f} Y{ym:.3f} F{feed}")
    out.append("G0 Z5")
    return out
