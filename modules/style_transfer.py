import os
import sys

import cv2
import numpy as np

try:
    from modules.edge_detection_hed import detect_edges_from_sketch
except ModuleNotFoundError:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from modules.edge_detection_hed import detect_edges_from_sketch


def stylize_image(image_path: str) -> np.ndarray:
    """
    Pencil sketch via cv2.pencilSketch. Saves output/stylized.png, returns grayscale sketch.
    """
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    gray_for_brightness = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_for_brightness))

    # Apply CLAHE in LAB space to boost local contrast before sketching.
    # Dark portraits need a stronger clip limit so facial detail isn't lost.
    if mean_brightness < 100:
        clip = 3.0
    elif mean_brightness <= 160:
        clip = 2.0
    else:
        clip = 0.0  # bright image — no CLAHE needed

    if clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    sketch_gray, _sketch_color = cv2.pencilSketch(
        img_bgr, sigma_s=50, sigma_r=0.07, shade_factor=0.08
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "stylized.png")
    ok = cv2.imwrite(out_path, sketch_gray)
    if not ok:
        raise IOError(f"Failed to write stylized image to: {out_path}")

    return sketch_gray


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "input")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    candidates = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No files found in input folder: {input_dir}. Add an image and re-run."
        )

    stylized_img = stylize_image(candidates[0])
    edges = detect_edges_from_sketch(stylized_img)
    print(
        f"Stylized image saved to output/stylized.png. "
        f"Shape={stylized_img.shape}, dtype={stylized_img.dtype}"
    )
    print(f"Edge map saved to output/edges_hed.png. Shape={edges.shape}, dtype={edges.dtype}")
