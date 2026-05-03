import os

import cv2
import numpy as np


def reorder_contours_nearest_neighbor(contours: list) -> list:
    """
    Takes a list of contours (as returned by cv2.findContours) and reorders them
    using a nearest-neighbour greedy algorithm. After finishing one contour, it
    selects the next contour whose start point is closest to the current contour's
    end point.
    """
    if contours is None:
        raise ValueError("contours must not be None")
    if len(contours) <= 1:
        return list(contours)

    def start_xy(c):
        p = c[0][0]
        return float(p[0]), float(p[1])

    def end_xy(c):
        p = c[-1][0]
        return float(p[0]), float(p[1])

    remaining = [c for c in contours if c is not None and len(c) > 0]
    if len(remaining) <= 1:
        return remaining

    ordered = [remaining.pop(0)]
    curr_end = end_xy(ordered[0])

    while remaining:
        best_i = 0
        best_d2 = None
        cx, cy = curr_end
        for i, c in enumerate(remaining):
            sx, sy = start_xy(c)
            d2 = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i

        nxt = remaining.pop(best_i)
        ordered.append(nxt)
        curr_end = end_xy(nxt)

    return ordered


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    edges_path = os.path.join(project_root, "output", "edges.png")

    edge_map = cv2.imread(edges_path, cv2.IMREAD_GRAYSCALE)
    if edge_map is None:
        raise FileNotFoundError(
            f"Could not read edge map image at: {edges_path}. Run edge_detection.py first."
        )

    edge_u8 = edge_map
    if edge_u8.dtype != np.uint8:
        edge_u8 = edge_u8.astype(np.uint8)
    edge_u8 = np.where(edge_u8 > 0, 255, 0).astype(np.uint8)

    contours, _hier = cv2.findContours(
        edge_u8.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    reordered = reorder_contours_nearest_neighbor(contours)
    print(f"Contours reordered: {len(reordered)}")
    print("Path ordering completed.")
