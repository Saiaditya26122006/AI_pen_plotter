import os
import sys

from modules.advanced_pipeline import process_image_full_pipeline


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Usage: python main.py <image_path>")

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print(f"Input: {image_path}")
    gcode = process_image_full_pipeline(image_path)
    draw_moves = sum(1 for line in gcode.splitlines() if line.startswith("G1 X"))
    print(f"Total draw moves: {draw_moves}")


if __name__ == "__main__":
    main()
