import os
import sys

import cv2

from modules.hatching import add_hatching_to_gcode


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Usage: python main.py <image_path>")

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load full original image — background is part of the drawing,
    # exactly as seen in professional pen plotter portraits.
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"Input image: {gray.shape[1]}x{gray.shape[0]} px")

    # Minimal base G-code — hatching provides all the visual content.
    base_gcode = "G21\nG90\nG0 Z5\nM2\n"

    print("Generating full-coverage ETF contour hatching...")
    combined_gcode = add_hatching_to_gcode(gray, base_gcode)

    draw_moves = sum(
        1 for line in combined_gcode.splitlines() if line.startswith("G1 X")
    )
    print(f"Total draw moves: {draw_moves}")
    print("Done. G-code saved to output/drawing.gcode")


if __name__ == "__main__":
    main()
