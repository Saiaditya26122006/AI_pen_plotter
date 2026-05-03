import os

import cv2
import numpy as np


def detect_edges(image_path: str) -> np.ndarray:
    """Basic Canny edge detector — kept for compatibility."""
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "edges.png")
    ok = cv2.imwrite(out_path, edges)
    if not ok:
        raise IOError(f"Failed to write edge map to: {out_path}")

    return edges


def detect_edges_dog(image_path: str) -> np.ndarray:
    """XDoG (Extended Difference of Gaussians) edge detector.

    Gives smoother, more continuous artistic lines than Canny — closer to
    the kind of outlines a skilled human sketcher would draw.  Pipeline:
      1. CLAHE  — boosts local contrast so faint face details are detected
      2. Bilateral filter — smooths flat skin regions while keeping edges
      3. XDoG  — (1+p)*G_σ − p*G_kσ, thresholded with a soft tanh curve
      4. Morphological closing — bridges tiny gaps so contours are continuous
    """
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE — local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Bilateral filter — edge-preserving smoothing
    smooth = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    smooth_f = smooth.astype(np.float64)

    # 3. XDoG
    sigma_e = 0.4        # fine-detail scale
    k = 4.5              # ratio: larger → fewer, cleaner lines
    p = 19.0             # sharpness of the DoG response

    g1 = cv2.GaussianBlur(smooth_f, (0, 0), sigma_e)
    g2 = cv2.GaussianBlur(smooth_f, (0, 0), sigma_e * k)
    dog = (1.0 + p) * g1 - p * g2   # sharpened − blurred

    # Soft threshold (Winnemöller et al.): dark lines where dog < tau
    tau = 128.0
    phi = 0.1
    result = np.where(
        dog >= tau,
        0.0,
        255.0 * (1.0 - np.tanh(phi * (dog - tau))),
    )
    edges = np.clip(result, 0, 255).astype(np.uint8)

    # Binarise to white-on-black for findContours
    _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    # 4. Close small gaps so contours are continuous strokes
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "edges.png")
    ok = cv2.imwrite(out_path, edges)
    if not ok:
        raise IOError(f"Failed to write edge map to: {out_path}")

    return edges


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
        raise FileNotFoundError(f"No files found in input folder: {input_dir}")

    edges = detect_edges_dog(candidates[0])
    print(f"XDoG edge map saved to output/edges.png. Shape={edges.shape}")
