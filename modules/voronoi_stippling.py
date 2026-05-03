import numpy as np
import cv2
from scipy.spatial import cKDTree


def weighted_voronoi_stipple(
    gray: np.ndarray,
    n_points: int = 8000,
    n_iterations: int = 30,
    mask: np.ndarray = None,
) -> list:
    """
    Weighted Voronoi Stippling via Lloyd's relaxation (Secord 2002).
    Points cluster densely in dark zones, sparsely in bright zones.
    Returns list of (x, y) pixel coordinates.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a 2D grayscale array")

    h, w = gray.shape
    density = (255.0 - gray.astype(np.float64)) / 255.0

    if mask is not None:
        # Zero out background FIRST, then apply minimum only INSIDE the mask.
        # The previous code clipped to 0.02 after masking, which accidentally
        # gave background pixels a non-zero probability of receiving dots.
        inside = (mask > 0)
        density[~inside] = 0.0
        density[inside] = np.clip(density[inside], 0.02, 1.0)
    else:
        density = np.clip(density, 0.02, 1.0)

    flat_density = density.ravel()
    total = flat_density.sum()
    if total < 1e-10:
        return []
    flat_density = flat_density / total

    rng = np.random.default_rng(42)
    n_pts = min(n_points, int((flat_density > 0).sum()))
    indices = rng.choice(h * w, size=n_pts, p=flat_density, replace=False)
    ys_init, xs_init = np.unravel_index(indices, (h, w))
    points = np.stack([xs_init.astype(np.float64), ys_init.astype(np.float64)], axis=1)

    # Downsample pixel grid for speed on large images
    scale = 2 if (h * w > 400_000) else 1
    h_s, w_s = h // scale, w // scale
    if scale > 1:
        gray_s = cv2.resize(gray, (w_s, h_s), interpolation=cv2.INTER_AREA)
        mask_s = (cv2.resize(mask.astype(np.uint8), (w_s, h_s),
                             interpolation=cv2.INTER_NEAREST) if mask is not None else None)
    else:
        gray_s = gray
        mask_s = mask.astype(np.uint8) if mask is not None else None

    density_s = (255.0 - gray_s.astype(np.float64)) / 255.0
    if mask_s is not None:
        inside_s = (mask_s > 0)
        density_s[~inside_s] = 0.0
        density_s[inside_s] = np.clip(density_s[inside_s], 0.02, 1.0)
    else:
        density_s = np.clip(density_s, 0.02, 1.0)

    gy_s, gx_s = np.mgrid[0:h_s, 0:w_s]
    pixel_xy = np.stack(
        [gx_s.ravel() * scale, gy_s.ravel() * scale], axis=1
    ).astype(np.float64)
    weights = density_s.ravel()

    print(f"  Voronoi: {n_pts} pts, {n_iterations} iterations...")

    for it in range(n_iterations):
        tree = cKDTree(points)
        _, labels = tree.query(pixel_xy, workers=1)

        new_pts = np.zeros((n_pts, 2))
        tot_w = np.zeros(n_pts)
        np.add.at(tot_w, labels, weights)
        np.add.at(new_pts[:, 0], labels, weights * pixel_xy[:, 0])
        np.add.at(new_pts[:, 1], labels, weights * pixel_xy[:, 1])

        valid = tot_w > 1e-12
        new_pts[valid] /= tot_w[valid, np.newaxis]

        n_orphan = int((~valid).sum())
        if n_orphan > 0:
            idxs = rng.choice(h * w, size=n_orphan, p=flat_density)
            oy, ox = np.unravel_index(idxs, (h, w))
            new_pts[~valid, 0] = ox.astype(np.float64)
            new_pts[~valid, 1] = oy.astype(np.float64)

        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, h - 1)
        points = new_pts

        if (it + 1) % 10 == 0:
            print(f"    iteration {it + 1}/{n_iterations}")

    return [(float(p[0]), float(p[1])) for p in points]


def stipple_to_gcode_lines(
    points: list,
    img_w: int,
    img_h: int,
    gray: np.ndarray,
    to_mm_fn,
) -> list:
    """Convert stipple points to G-code with greedy NN ordering."""
    if not points:
        return []

    pts_arr = np.array(points)
    order = _greedy_nn_order(pts_arr)

    lines = []
    for i in order:
        px, py = points[i]
        xm, ym = to_mm_fn(px, py)
        ix = int(np.clip(round(px), 0, img_w - 1))
        iy = int(np.clip(round(py), 0, img_h - 1))
        bv = int(gray[iy, ix])
        feed = 300 if bv < 80 else (800 if bv < 160 else 1500)
        lines.append("G0 Z5")
        lines.append(f"G0 X{xm:.3f} Y{ym:.3f}")
        lines.append(f"G1 Z0 F{feed}")
        lines.append("G0 Z5")

    return lines


def _greedy_nn_order(pts: np.ndarray) -> list:
    n = len(pts)
    if n == 0:
        return []
    visited = np.zeros(n, dtype=bool)
    order: list = []
    current = 0
    tree = cKDTree(pts)
    k = min(30, n)

    for _ in range(n):
        order.append(current)
        visited[current] = True
        _, neighbors = tree.query(pts[current], k=k)
        neighbors_list = neighbors if hasattr(neighbors, '__iter__') else [neighbors]
        found = False
        for nb in neighbors_list:
            nb = int(nb)
            if not visited[nb]:
                current = nb
                found = True
                break
        if not found:
            remaining = np.where(~visited)[0]
            if len(remaining) == 0:
                break
            dists = np.sum((pts[remaining] - pts[current]) ** 2, axis=1)
            current = int(remaining[np.argmin(dists)])

    return order
