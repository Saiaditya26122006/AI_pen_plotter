import os
import numpy as np
import cv2


def _compute_etf(gray: np.ndarray, num_iterations: int = 5, radius: int = 5) -> tuple:
    """Edge Tangent Flow (Kang et al. 2007).
    Iteratively aligns the initial Sobel tangent field so every vector
    pulls toward neighbours that face the same direction, giving a globally
    coherent flow that wraps around faces, hair, and clothing."""
    blurred = cv2.GaussianBlur(gray.astype(np.float64), (21, 21), 0)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mag /= grad_mag.max() + 1e-8

    # Tangent = gradient rotated 90° (flow runs along contours)
    tx = -gy
    ty = gx
    mag = np.sqrt(tx * tx + ty * ty)
    mag = np.where(mag < 1e-8, 1e-8, mag)
    tx /= mag
    ty /= mag

    for _ in range(num_iterations):
        new_tx = np.zeros_like(tx)
        new_ty = np.zeros_like(ty)
        total_w = np.zeros_like(tx)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > radius:
                    continue

                ws = 1.0 - dist / radius

                ntx = np.roll(np.roll(tx, -dy, axis=0), -dx, axis=1)
                nty = np.roll(np.roll(ty, -dy, axis=0), -dx, axis=1)
                ng  = np.roll(np.roll(grad_mag, -dy, axis=0), -dx, axis=1)

                dot  = tx * ntx + ty * nty
                wm   = np.abs(dot)
                sign = np.where(dot >= 0, 1.0, -1.0)

                w = ws * wm * ng
                new_tx  += w * sign * ntx
                new_ty  += w * sign * nty
                total_w += w

        total_w = np.where(total_w < 1e-8, 1e-8, total_w)
        tx = new_tx / total_w
        ty = new_ty / total_w
        mag = np.sqrt(tx * tx + ty * ty)
        mag = np.where(mag < 1e-8, 1e-8, mag)
        tx /= mag
        ty /= mag

    return tx, ty


def _poisson_disk_seeds(density_map: np.ndarray, base_radius: float = 6.0) -> list:
    """Bridson fast Poisson disk sampling with full-image coverage.

    Minimum separation radius scales with local brightness:
      black  → base_radius * 0.4  (dense seeds)
      white  → base_radius * 2.5  (sparse seeds, but never excluded)

    No pixel is ever fully excluded — the whole image is covered."""
    h, w = density_map.shape
    min_r  = base_radius * 0.4
    cell   = min_r / np.sqrt(2)
    gw = int(np.ceil(w / cell)) + 2
    gh = int(np.ceil(h / cell)) + 2

    grid    = np.full((gh, gw), -1, dtype=np.int32)
    samples: list = []
    active:  list = []
    rng = np.random.default_rng(42)

    def _radius(x: float, y: float) -> float:
        ix = int(np.clip(x, 0, w - 1))
        iy = int(np.clip(y, 0, h - 1))
        b  = float(density_map[iy, ix]) / 255.0
        return base_radius * (0.4 + 2.1 * b)   # dark→dense, bright→sparse

    def _too_close(x: float, y: float, r: float) -> bool:
        gx0 = max(0, int((x - 2 * r) / cell))
        gy0 = max(0, int((y - 2 * r) / cell))
        gx1 = min(gw - 1, int((x + 2 * r) / cell))
        gy1 = min(gh - 1, int((y + 2 * r) / cell))
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                idx = grid[gy, gx]
                if idx < 0:
                    continue
                sx, sy = samples[idx]
                if (sx - x) ** 2 + (sy - y) ** 2 < r * r:
                    return True
        return False

    x0, y0 = w / 2.0, h / 2.0
    samples.append((x0, y0))
    active.append(0)
    grid[min(gh - 1, int(y0 / cell)), min(gw - 1, int(x0 / cell))] = 0

    while active:
        i  = int(rng.integers(0, len(active)))
        sx, sy = samples[active[i]]
        r  = _radius(sx, sy)
        placed = False

        for _ in range(30):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            dist  = rng.uniform(r, 2.0 * r)
            nx    = sx + dist * np.cos(angle)
            ny    = sy + dist * np.sin(angle)

            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue

            r_new   = _radius(nx, ny)
            r_check = min(r, r_new)
            if not _too_close(nx, ny, r_check):
                new_idx = len(samples)
                samples.append((nx, ny))
                active.append(new_idx)
                grid[min(gh - 1, int(ny / cell)),
                     min(gw - 1, int(nx / cell))] = new_idx
                placed = True
                break

        if not placed:
            active.pop(i)

    return samples


def _trace_streamline(fx: np.ndarray, fy: np.ndarray,
                       start_x: float, start_y: float,
                       img_w: int, img_h: int,
                       max_steps: int = 200, step_size: float = 2.0) -> list:
    """Follow the ETF flow field from a seed point.
    Lines run until they leave the canvas — no brightness early-stop,
    so every part of the image is crossed by strokes."""
    sl = [(start_x, start_y)]
    cx, cy = start_x, start_y

    for _ in range(max_steps):
        ix = max(0, min(int(round(cx)), img_w - 1))
        iy = max(0, min(int(round(cy)), img_h - 1))

        cx += fx[iy, ix] * step_size
        cy += fy[iy, ix] * step_size

        if cx < 0 or cx >= img_w or cy < 0 or cy >= img_h:
            break

        sl.append((cx, cy))

    return sl


def add_hatching_to_gcode(gray_image: np.ndarray, existing_gcode: str) -> str:
    """Generate full-coverage flow-based hatching that replicates the style
    of high-end pen plotter portraits:
      • ETF flow field so lines follow face/body contours globally
      • Poisson-disk seeds covering the ENTIRE image at density ∝ darkness
      • Long streamlines that sweep across the whole canvas
      • No areas left blank — brightness controls spacing, not presence
      • Variable feed rate: slower in dark zones for richer ink deposit"""
    if not isinstance(gray_image, np.ndarray):
        raise TypeError("gray_image must be a numpy array")
    if gray_image.ndim == 3:
        if gray_image.shape[2] == 1:
            gray_image = gray_image[:, :, 0]
        elif gray_image.shape[2] == 4:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGRA2GRAY)
        else:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    if gray_image.ndim != 2:
        raise ValueError("gray_image must be a 2D grayscale array")

    img_h, img_w = gray_image.shape
    print(f"  Image: {img_w}x{img_h}")

    blurred = cv2.GaussianBlur(gray_image, (51, 51), 0)

    # ── 1. ETF flow field ────────────────────────────────────────────────────
    print("  Computing ETF flow field...")
    tx, ty = _compute_etf(gray_image, num_iterations=5, radius=5)

    x_scale = 287.0 / float(img_w)
    y_scale = 410.0 / float(img_h)

    def pixel_to_mm(px: float, py: float):
        return (
            5.0 + px * x_scale,
            5.0 + (img_h - 1 - py) * y_scale,
        )

    # ── 2. Full-image Poisson disk seeds ────────────────────────────────────
    print("  Placing seeds (full-image Poisson disk)...")
    seeds = _poisson_disk_seeds(blurred, base_radius=6.0)
    print(f"  {len(seeds)} seeds placed")

    hatch_lines: list = []
    streamline_count = 0

    # ── 3. Trace long streamlines — no brightness stop ───────────────────────
    for seed_x, seed_y in seeds:
        ix = int(np.clip(round(seed_x), 0, img_w - 1))
        iy = int(np.clip(round(seed_y), 0, img_h - 1))
        brightness = int(blurred[iy, ix])

        # Variable feed rate: dark = slower (more ink), bright = faster
        if brightness < 60:
            feed = 1800
            max_steps = 200
        elif brightness < 120:
            feed = 2500
            max_steps = 180
        elif brightness < 180:
            feed = 3000
            max_steps = 150
        else:
            feed = 3500
            max_steps = 120

        sl = _trace_streamline(tx, ty, seed_x, seed_y,
                               img_w, img_h, max_steps)
        if len(sl) < 2:
            continue

        x0, y0 = pixel_to_mm(sl[0][0], sl[0][1])
        hatch_lines.append("G0 Z5")
        hatch_lines.append(f"G0 X{x0:.3f} Y{y0:.3f}")
        hatch_lines.append("G1 Z0 F100")
        for px, py in sl[1:]:
            xm, ym = pixel_to_mm(px, py)
            hatch_lines.append(f"G1 X{xm:.3f} Y{ym:.3f} F{feed}")
        hatch_lines.append("G0 Z5")
        streamline_count += 1

    print(f"  {streamline_count} streamlines generated")

    # ── 4. Insert before M2 and save ────────────────────────────────────────
    base_lines = existing_gcode.splitlines()
    m2_idx = None
    for idx in range(len(base_lines) - 1, -1, -1):
        if base_lines[idx].strip() == "M2":
            m2_idx = idx
            break

    if m2_idx is None:
        combined_lines = base_lines + hatch_lines + ["M2"]
    else:
        combined_lines = base_lines[:m2_idx] + hatch_lines + base_lines[m2_idx:]

    combined_gcode = "\n".join(combined_lines) + "\n"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "drawing.gcode")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined_gcode)

    return combined_gcode


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    stylized_path = os.path.join(project_root, "output", "stylized.png")
    gcode_path    = os.path.join(project_root, "output", "drawing.gcode")

    gray = cv2.imread(stylized_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read: {stylized_path}")

    with open(gcode_path, "r", encoding="utf-8") as f:
        existing = f.read()

    combined  = add_hatching_to_gcode(gray, existing)
    draw_moves = sum(1 for line in combined.splitlines() if line.startswith("G1 X"))
    print(f"Total draw moves: {draw_moves}")
    print("Done. G-code saved to output/drawing.gcode")
