import os

import cv2
import numpy as np


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


def _load_stylized_image_path(project_root: str) -> str:
    stylized_path = os.path.join(project_root, "output", "stylized.png")
    if not os.path.isfile(stylized_path):
        raise FileNotFoundError(
            f"Stylized image not found: {stylized_path}. Run modules/style_transfer.py first."
        )
    return stylized_path


def _run_hed_soft_map(img_bgr: np.ndarray, prototxt_path: str, model_path: str) -> np.ndarray:
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
    edge = np.clip(edge, 0.0, 1.0).astype(np.float32)
    return edge


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
    model_path = os.path.join(models_dir, "hed_pretrained_bsds.caffemodel")
    if not os.path.isfile(prototxt_path) or not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Missing HED model files in /models. Expected:\n"
            f"- {prototxt_path}\n"
            f"- {model_path}\n"
            "Run modules/edge_detection_hed.py once to download them."
        )

    image_path = _load_stylized_image_path(project_root)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    soft = _run_hed_soft_map(img_bgr, prototxt_path, model_path)  # float32, 0..1

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thinned_results = []

    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "thinning"):
        raise RuntimeError(
            "cv2.ximgproc.thinning() is not available in this OpenCV build. "
            "Install opencv-contrib-python to use thinning."
        )

    for t in thresholds:
        binary = np.where(soft > float(t), 255, 0).astype(np.uint8)
        thinned = cv2.ximgproc.thinning(binary)
        out_path = os.path.join(output_dir, f"threshold_test_{t:.1f}.png")
        ok = cv2.imwrite(out_path, thinned)
        if not ok:
            raise IOError(f"Failed to write: {out_path}")
        thinned_results.append((t, thinned))

    panels = []
    for t, img in thinned_results:
        panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            panel,
            f"t={t:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    row1 = np.hstack(panels[0:3])
    row2 = np.hstack(panels[3:6])
    comparison = np.vstack([row1, row2])

    comp_path = os.path.join(output_dir, "threshold_comparison.png")
    ok = cv2.imwrite(comp_path, comparison)
    if not ok:
        raise IOError(f"Failed to write: {comp_path}")

    print("Saved per-threshold outputs and output/threshold_comparison.png")
