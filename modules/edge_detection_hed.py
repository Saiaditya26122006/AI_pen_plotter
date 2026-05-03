import os
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import cv2
import numpy as np


_HED_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/"
    "hed_pretrained_bsds.caffemodel"
)
_HED_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/"
    "deploy.prototxt"
)


class _CropLayer(cv2.dnn.Layer):
    def __init__(self, params, blobs):
        super().__init__()

    def getMemoryShapes(self, inputs):
        inp_shape, target_shape = inputs[0], inputs[1]
        batch_size, num_channels = inp_shape[0], inp_shape[1]
        height, width = target_shape[2], target_shape[3]
        return [[batch_size, num_channels, height, width]]

    def forward(self, inputs, outputs=None):
        inp, target = inputs[0], inputs[1]
        _, _, h, w = target.shape
        return [inp[:, :, :h, :w]]


def _ensure_hed_files(models_dir: str) -> tuple[str, str]:
    os.makedirs(models_dir, exist_ok=True)
    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
    model_path = os.path.join(models_dir, "hed_pretrained_bsds.caffemodel")

    model_urls = [
        _HED_MODEL_URL,
        "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel",
    ]
    prototxt_urls = [
        _HED_PROTOTXT_URL,
        "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt",
    ]

    if not os.path.isfile(model_path):
        last_err = None
        for url in model_urls:
            try:
                urlretrieve(url, model_path)
                last_err = None
                break
            except (HTTPError, URLError) as e:
                last_err = e
        if last_err is not None:
            raise last_err

    if not os.path.isfile(prototxt_path):
        last_err = None
        for url in prototxt_urls:
            try:
                urlretrieve(url, prototxt_path)
                last_err = None
                break
            except (HTTPError, URLError) as e:
                last_err = e
        if last_err is not None:
            raise last_err

    return prototxt_path, model_path


def detect_edges_hed(image_path: str) -> np.ndarray:
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    prototxt_path, model_path = _ensure_hed_files(models_dir)

    try:
        cv2.dnn_registerLayer("Crop", _CropLayer)
    except Exception:
        pass

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_bgr,
        scalefactor=1.0,
        size=(w, h),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    out = net.forward()

    edge = out[0, 0, :, :]
    edge = cv2.resize(edge, (w, h))
    edge = np.clip(edge, 0.0, 1.0)
    return np.where(edge > 0.5, 255, 0).astype(np.uint8)


def detect_edges_from_sketch(sketch_array: np.ndarray) -> np.ndarray:
    if not isinstance(sketch_array, np.ndarray):
        raise TypeError("sketch_array must be a numpy array")
    if sketch_array.ndim == 3:
        if sketch_array.shape[2] == 3:
            sketch_array = cv2.cvtColor(sketch_array, cv2.COLOR_BGR2GRAY)
        elif sketch_array.shape[2] == 1:
            sketch_array = sketch_array[:, :, 0]
    if sketch_array.ndim != 2:
        raise ValueError(f"sketch_array must be a 2D grayscale array, got shape {sketch_array.shape}")

    sketch_u8 = sketch_array.astype(np.uint8)

    # Bilateral filter instead of Gaussian: smooths interior texture while
    # keeping actual edges sharp. Gaussian blurs edges too, making potrace
    # trace a blurry gradient → jagged beziers. Bilateral keeps the edge
    # gradient steep → potrace traces a crisp, smooth curve.
    sketch_filtered = cv2.bilateralFilter(sketch_u8, d=5, sigmaColor=75, sigmaSpace=75)

    median = float(np.median(sketch_filtered))
    threshold1 = max(40, median * 0.25)
    threshold2 = max(100, median * 0.55)

    edges = cv2.Canny(sketch_filtered, threshold1=int(threshold1), threshold2=int(threshold2))

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "edges_hed.png")
    ok = cv2.imwrite(out_path, edges)
    if not ok:
        raise IOError(f"Failed to write edge map to: {out_path}")

    return edges


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    stylized_path = os.path.join(project_root, "output", "stylized.png")
    sketch = cv2.imread(stylized_path, cv2.IMREAD_GRAYSCALE)
    if sketch is None:
        raise FileNotFoundError(
            f"Could not read sketch image at: {stylized_path}. Run style_transfer.py first."
        )
    edges = detect_edges_from_sketch(sketch)
    print(
        f"Edge map saved to output/edges_hed.png. Shape={edges.shape}, dtype={edges.dtype}"
    )