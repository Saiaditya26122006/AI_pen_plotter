import os

import cv2
import numpy as np


def generate_gcode(edge_map: np.ndarray) -> str:
    if not isinstance(edge_map, np.ndarray):
        raise TypeError("edge_map must be a numpy array")
    if edge_map.ndim != 2:
        raise ValueError("edge_map must be a 2D array")

    height, width = edge_map.shape
    if height <= 0 or width <= 0:
        raise ValueError("edge_map must have non-zero dimensions")

    edge_u8 = edge_map if edge_map.dtype == np.uint8 else edge_map.astype(np.uint8)
    edge_u8 = np.where(edge_u8 > 0, 255, 0).astype(np.uint8)

    avail_w = 287.0
    avail_h = 410.0
    scale = min(avail_w / width, avail_h / height)
    drawn_w = width * scale
    drawn_h = height * scale
    off_x = 5.0 + (avail_w - drawn_w) / 2.0
    off_y = 5.0 + (avail_h - drawn_h) / 2.0

    def to_mm(px: float, py: float):
        # Image Y=0 is top; paper Y=0 is bottom — flip Y axis
        return (
            off_x + px * scale,
            off_y + drawn_h - py * scale,
        )

    # Find contours directly from the edge map.
    # Unlike potrace (which traces filled black blobs into closed SVG paths),
    # findContours gives us the actual boundary of each edge region as a
    # sequence of points we draw as a clean stroke.
    contours, _ = cv2.findContours(edge_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    gcode_lines = ["G21", "G90", "G0 Z5"]

    for contour in contours:
        # Skip tiny noise specks by minimum perimeter
        if cv2.arcLength(contour, closed=True) < 30:
            continue

        # Douglas-Peucker simplification: removes pixel-staircase jitter while
        # keeping the actual curve shape.
        simplified = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
        if len(simplified) < 3:
            continue

        # Travel to the start of this contour (pen up)
        p = simplified[0][0]
        x_mm, y_mm = to_mm(float(p[0]), float(p[1]))
        gcode_lines.append("G0 Z5")
        gcode_lines.append(f"G0 X{x_mm:.3f} Y{y_mm:.3f}")
        gcode_lines.append("G1 Z0 F100")

        # Draw each point in the simplified contour
        for pt in simplified[1:]:
            p = pt[0]
            x_mm, y_mm = to_mm(float(p[0]), float(p[1]))
            gcode_lines.append(f"G1 X{x_mm:.3f} Y{y_mm:.3f} F3000")

        # Close back to the start point
        p = simplified[0][0]
        x_mm, y_mm = to_mm(float(p[0]), float(p[1]))
        gcode_lines.append(f"G1 X{x_mm:.3f} Y{y_mm:.3f} F3000")
        gcode_lines.append("G0 Z5")

    gcode_lines.append("M2")
    gcode = "\n".join(gcode_lines) + "\n"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "drawing.gcode")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gcode)

    return gcode


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    edges_path = os.path.join(project_root, "output", "edges_hed.png")

    edge_map = cv2.imread(edges_path, cv2.IMREAD_GRAYSCALE)
    if edge_map is None:
        raise FileNotFoundError(
            f"Could not read edge map at: {edges_path}. Run edge_detection_hed.py first."
        )

    gcode = generate_gcode(edge_map)
    print("G-code saved to output/drawing.gcode")
    print(gcode[:500])
