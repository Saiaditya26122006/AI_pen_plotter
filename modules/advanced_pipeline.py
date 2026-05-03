import os

import cv2
import numpy as np

from modules.subject_detection import detect_persons_advanced
from modules.edge_detection_hed import _CropLayer, _ensure_hed_files
from modules.voronoi_stippling import weighted_voronoi_stipple, stipple_to_gcode_lines
from modules.face_landmarks import get_face_regions
from modules.region_segmentation import segment_body_regions
from modules.crosshatch import crosshatch_region
from modules.hatching import _compute_etf, _poisson_disk_seeds, _trace_streamline

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


def _run_hed(img_bgr: np.ndarray, prototxt: str, model: str) -> np.ndarray:
    """Run HED and return a thinned binary edge map."""
    try:
        cv2.dnn_registerLayer("Crop", _CropLayer)
    except Exception:
        pass
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_bgr, 1.0, (w, h),
        (104.00698793, 116.66876762, 122.67891434), False, False,
    )
    net.setInput(blob)
    out = net.forward()
    edge = cv2.resize(out[0, 0], (w, h))
    # Higher threshold (0.55) reduces noise and spurious thin edge fragments
    binary = np.where(np.clip(edge, 0, 1) > 0.55, 255, 0).astype(np.uint8)

    # Thin to 1-pixel skeleton so findContours traces clean strokes not fat blobs
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        binary = cv2.ximgproc.thinning(binary)

    return binary


def _outlines_gcode(edges: np.ndarray, mask: np.ndarray, to_mm_fn) -> list:
    """
    Convert thinned HED edges to G-code outlines.
    Only draws the longest/most structurally significant contours.
    Draws open arcs — no closing stroke back to start.
    """
    masked = np.where(mask > 0, edges, 0).astype(np.uint8)

    # CHAIN_APPROX_SIMPLE collapses straight runs to their endpoints — far less noise
    contours, _ = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only contours above a meaningful perimeter threshold
    valid = [c for c in contours if cv2.arcLength(c, closed=False) >= 60]

    # Sort longest-first so the most important structural lines are drawn first
    valid.sort(key=lambda c: cv2.arcLength(c, closed=False), reverse=True)

    # Cap total contours to avoid thousands of tiny fragments
    valid = valid[:300]

    lines: list = []
    for contour in valid:
        # Higher epsilon (3.0) removes pixel-staircase jitter while keeping curves
        simp = cv2.approxPolyDP(contour, epsilon=3.0, closed=False)
        if len(simp) < 2:
            continue

        p0 = simp[0][0]
        x0, y0 = to_mm_fn(float(p0[0]), float(p0[1]))
        lines += ["G0 Z5", f"G0 X{x0:.3f} Y{y0:.3f}", "G1 Z0 F100"]
        for pt in simp[1:]:
            p = pt[0]
            xm, ym = to_mm_fn(float(p[0]), float(p[1]))
            lines.append(f"G1 X{xm:.3f} Y{ym:.3f} F3000")
        lines.append("G0 Z5")

    return lines


def _etf_hatch_masked(
    gray: np.ndarray,
    mask: np.ndarray,
    to_mm_fn,
    img_w: int,
    img_h: int,
) -> list:
    """ETF streamline hatching restricted to mask region (hair)."""
    if not np.any(mask > 0):
        return []

    tx, ty = _compute_etf(gray, num_iterations=5, radius=5)

    density = gray.copy()
    density[mask == 0] = 255
    seeds = _poisson_disk_seeds(density, base_radius=5.0)

    lines: list = []
    for sx, sy in seeds:
        ix = int(np.clip(round(sx), 0, img_w - 1))
        iy = int(np.clip(round(sy), 0, img_h - 1))
        if mask[iy, ix] == 0:
            continue

        bv = int(gray[iy, ix])
        feed = 1800 if bv < 80 else (2500 if bv < 140 else 3200)
        max_steps = 150 if bv < 80 else 100

        sl = _trace_streamline(tx, ty, sx, sy, img_w, img_h, max_steps)

        clipped: list = []
        for px, py in sl:
            cx_ = int(np.clip(round(px), 0, img_w - 1))
            cy_ = int(np.clip(round(py), 0, img_h - 1))
            if mask[cy_, cx_] > 0:
                clipped.append((px, py))
            elif clipped:
                break

        if len(clipped) < 2:
            continue

        x0, y0 = to_mm_fn(clipped[0][0], clipped[0][1])
        lines += ["G0 Z5", f"G0 X{x0:.3f} Y{y0:.3f}", "G1 Z0 F100"]
        for px, py in clipped[1:]:
            xm, ym = to_mm_fn(px, py)
            lines.append(f"G1 X{xm:.3f} Y{ym:.3f} F{feed}")
        lines.append("G0 Z5")

    return lines


def _valid_region_mask(mask: np.ndarray, person_mask: np.ndarray,
                        max_fraction: float = 0.65) -> np.ndarray:
    """
    Sanity-check a region mask.
    If it covers more than max_fraction of the person area, segmentation likely
    failed — return an empty mask rather than crosshatching the entire image.
    """
    person_px = int(np.sum(person_mask > 0))
    region_px = int(np.sum(mask > 0))
    if person_px == 0 or region_px == 0:
        return mask
    if region_px / person_px > max_fraction:
        print(f"  Region mask covers {region_px/person_px:.0%} of person — skipping (likely bad seg)")
        return np.zeros_like(mask)
    return mask


def _save_debug(name: str, img: np.ndarray) -> None:
    path = os.path.join(_OUTPUT_DIR, f"debug_{name}.png")
    cv2.imwrite(path, img)


def process_image_full_pipeline(image_path: str) -> str:
    """
    4-layer pen plotter pipeline:
      L1  HED outlines        — structure (thinned, longest-contour only)
      L2  Voronoi stippling   — photo-realistic tonal depth
      L3  Region hatching     — hair (ETF) + clothes (crosshatch, with sanity check)
      L4  Face emphasis       — denser stippling on eyes/mouth/nose
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Multi-person detection ────────────────────────────────────
    print("[1/5] YOLOv8 multi-person detection...")
    img_bgr, person_mask = detect_persons_advanced(image_path)

    h, w = img_bgr.shape[:2]
    print(f"  Cropped region: {w}x{h} px")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)
    to_mm = _make_to_mm(w, h)

    # Save debug crops for visual inspection
    _save_debug("cropped_bgr", img_bgr)
    _save_debug("person_mask", person_mask)

    gcode: list = ["G21", "G90", "G0 Z5"]

    # ── Layer 1: HED outlines ─────────────────────────────────────────────
    print("[2/5] Layer 1: HED edge outlines (thinned)...")
    try:
        models_dir = os.path.join(_PROJECT_ROOT, "models")
        proto, mdl = _ensure_hed_files(models_dir)
        hed_edges = _run_hed(img_bgr, proto, mdl)
        _save_debug("hed_edges", hed_edges)
        outline_lines = _outlines_gcode(hed_edges, person_mask, to_mm)
        gcode.extend(outline_lines)
        print(f"  {len(outline_lines)} G-code lines from outlines")
    except Exception as exc:
        print(f"  HED failed ({exc}) — skipping outlines")

    # ── Layer 2: Weighted Voronoi stippling ───────────────────────────────
    print("[3/5] Layer 2: Weighted Voronoi stippling (8 000 pts)...")
    try:
        pts = weighted_voronoi_stipple(
            gray_enh, n_points=8000, n_iterations=30, mask=person_mask
        )
        stip_lines = stipple_to_gcode_lines(pts, w, h, gray, to_mm)
        gcode.extend(stip_lines)
        print(f"  {len(pts)} stipple points")
    except Exception as exc:
        print(f"  Voronoi stippling failed ({exc}) — skipping")

    # ── Layer 3: Region-aware hatching ────────────────────────────────────
    print("[4/5] Layer 3: Region-aware hatching...")
    face_regions: dict = {}
    try:
        face_regions = get_face_regions(img_bgr)
        body = segment_body_regions(img_bgr, person_mask, face_regions)

        _save_debug("mask_hair", body['hair'])
        _save_debug("mask_skin", body['skin'])
        _save_debug("mask_clothes", body['clothes'])

        # Hair: ETF hatching (no sanity check needed — hair area is always small)
        hair_m = body['hair']
        if np.any(hair_m > 0):
            print("  Hair → ETF streamline hatching")
            gcode.extend(_etf_hatch_masked(gray_enh, hair_m, to_mm, w, h))

        # Clothes: crosshatch only when the mask is plausibly correct
        clothes_m = _valid_region_mask(body['clothes'], person_mask, max_fraction=0.65)
        if np.any(clothes_m > 0):
            print("  Clothes → adaptive crosshatching")
            gcode.extend(crosshatch_region(gray_enh, clothes_m, to_mm, w, h))

    except Exception as exc:
        print(f"  Region hatching failed ({exc}) — skipping")

    # ── Layer 4: Face feature emphasis ────────────────────────────────────
    print("[5/5] Layer 4: Face feature emphasis...")
    try:
        feat_mask = np.zeros((h, w), dtype=np.uint8)
        for key in ('eyes', 'mouth', 'eyebrows', 'nose'):
            if key in face_regions:
                feat_mask = np.maximum(feat_mask, face_regions[key])
        _save_debug("mask_face_features", feat_mask)

        if np.any(feat_mask > 0):
            feat_pts = weighted_voronoi_stipple(
                gray_enh, n_points=2000, n_iterations=20, mask=feat_mask
            )
            gcode.extend(stipple_to_gcode_lines(feat_pts, w, h, gray, to_mm))
            print(f"  {len(feat_pts)} feature-emphasis points")
        else:
            print("  No face features detected — skipping emphasis")
    except Exception as exc:
        print(f"  Face emphasis failed ({exc}) — skipping")

    # ── Finalize ──────────────────────────────────────────────────────────
    gcode.append("M2")
    gcode_str = "\n".join(gcode) + "\n"

    out_path = os.path.join(_OUTPUT_DIR, "drawing.gcode")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gcode_str)

    draw_moves = sum(1 for ln in gcode if ln.startswith("G1 X"))
    travel_moves = sum(1 for ln in gcode if ln.startswith("G0 X"))
    print(f"\nDone — {draw_moves} draw moves, {travel_moves} travel moves → {out_path}")
    return gcode_str
