import os
import cv2
import numpy as np
from ultralytics import YOLO


def detect_and_crop_subject(image_path: str) -> np.ndarray:
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    try:
        model = YOLO("yolov8n-seg.pt")
        seg_available = True
    except Exception:
        model = YOLO("yolov8n.pt")
        seg_available = False

    results = model(img)

    if len(results) == 0 or len(results[0].boxes) == 0:
        print("Warning: No objects detected. Using original image.")
        cropped = img
    else:
        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        # Collect all person detections (class 0) above confidence threshold.
        # If none qualify, fall back to the single highest-confidence detection.
        person_indices = [
            i for i, (cls, conf) in enumerate(zip(classes, confidences))
            if int(cls) == 0 and conf > 0.3
        ]
        if not person_indices:
            person_indices = [int(np.argmax(confidences))]

        print(f"Detected {len(person_indices)} person(s) — drawing all of them.")

        # Union bounding box covering every detected person
        all_xyxy = boxes.xyxy[person_indices].cpu().numpy()
        x1 = float(np.min(all_xyxy[:, 0]))
        y1 = float(np.min(all_xyxy[:, 1]))
        x2 = float(np.max(all_xyxy[:, 2]))
        y2 = float(np.max(all_xyxy[:, 3]))

        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = 0.05 * w, 0.05 * h
        new_x1 = int(max(0, x1 - pad_x))
        new_y1 = int(max(0, y1 - pad_y))
        new_x2 = int(min(img.shape[1], x2 + pad_x))
        new_y2 = int(min(img.shape[0], y2 + pad_y))

        masks = results[0].masks if seg_available else None
        if masks is not None:
            # Union of all per-person segmentation masks
            combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for idx in person_indices:
                if idx < len(masks.data):
                    mask_raw = masks.data[idx].cpu().numpy()
                    mask = cv2.resize(mask_raw, (img.shape[1], img.shape[0]))
                    combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

            white = np.full_like(img, 255)
            img_nobg = np.where(combined_mask[:, :, np.newaxis] == 1, img, white)
            cropped = img_nobg[new_y1:new_y2, new_x1:new_x2]
        else:
            cropped = img[new_y1:new_y2, new_x1:new_x2]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cropped.png")

    ok = cv2.imwrite(out_path, cropped)
    if not ok:
        raise IOError(f"Failed to write cropped image to: {out_path}")

    return cropped


def detect_persons_advanced(image_path: str) -> tuple:
    """
    Detect all persons, crop to their union bounding box, remove background.
    Returns (cropped_bgr, person_mask) where person_mask is in cropped coords.
    Falls back to full image when no person is detected.
    """
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]

    try:
        model = YOLO("yolov8n-seg.pt")
        seg_available = True
    except Exception:
        model = YOLO("yolov8n.pt")
        seg_available = False

    results = model(img)

    if len(results) == 0 or len(results[0].boxes) == 0:
        print("  No persons detected — using full image")
        mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
        _save_outputs(img, mask)
        return img, mask

    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    person_indices = [
        i for i, (cls, conf) in enumerate(zip(classes, confidences))
        if int(cls) == 0 and conf > 0.3
    ]
    if not person_indices:
        person_indices = [int(np.argmax(confidences))]

    print(f"  Detected {len(person_indices)} person(s)")

    all_xyxy = boxes.xyxy[person_indices].cpu().numpy()
    x1 = float(np.min(all_xyxy[:, 0]))
    y1 = float(np.min(all_xyxy[:, 1]))
    x2 = float(np.max(all_xyxy[:, 2]))
    y2 = float(np.max(all_xyxy[:, 3]))

    wb, hb = x2 - x1, y2 - y1
    pad_x, pad_y = 0.08 * wb, 0.08 * hb
    cx1 = int(max(0, x1 - pad_x))
    cy1 = int(max(0, y1 - pad_y))
    cx2 = int(min(img_w, x2 + pad_x))
    cy2 = int(min(img_h, y2 + pad_y))

    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    seg_masks = results[0].masks if seg_available else None

    if seg_masks is not None:
        for idx in person_indices:
            if idx < len(seg_masks.data):
                raw = seg_masks.data[idx].cpu().numpy()
                resized = cv2.resize(raw, (img_w, img_h))
                combined_mask = np.maximum(combined_mask, (resized > 0.5).astype(np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    else:
        for idx in person_indices:
            b = boxes.xyxy[idx].cpu().numpy()
            bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            combined_mask[by1:by2, bx1:bx2] = 1

    white = np.full_like(img, 255)
    img_nobg = np.where(combined_mask[:, :, np.newaxis] == 1, img, white)

    cropped_bgr = img_nobg[cy1:cy2, cx1:cx2]
    cropped_mask = combined_mask[cy1:cy2, cx1:cx2] * 255

    _save_outputs(cropped_bgr, cropped_mask)
    return cropped_bgr, cropped_mask


def _save_outputs(img: np.ndarray, mask: np.ndarray) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "cropped.png"), img)
    cv2.imwrite(os.path.join(out_dir, "person_mask.png"), mask)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "input")

    if not os.path.isdir(input_dir):
        print(f"Test skipped: Input folder not found: {input_dir}")
    else:
        images = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not images:
            print(f"Test skipped: No images found in {input_dir}")
        else:
            test_image = os.path.join(input_dir, images[0])
            print(f"Running test on: {test_image}")
            result = detect_and_crop_subject(test_image)
            print(f"Cropped image shape: {result.shape}")
            print("Saved to output/cropped.png")
