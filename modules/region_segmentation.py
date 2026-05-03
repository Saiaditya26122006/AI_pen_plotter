import cv2
import numpy as np


def segment_body_regions(
    img_bgr: np.ndarray,
    person_mask: np.ndarray,
    face_regions: dict,
) -> dict:
    """
    Estimate hair, skin, and clothes masks from person mask + face landmarks.
    Returns dict with uint8 masks (255=in region).
    Keys: 'hair', 'skin', 'clothes', 'person'
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    person = (person_mask > 0).astype(np.uint8) * 255

    # ── Skin = face oval from MediaPipe, clipped to person ────────────────
    if face_regions and 'face' in face_regions and np.any(face_regions['face'] > 0):
        skin = np.where((face_regions['face'] > 0) & (person > 0), 255, 0).astype(np.uint8)
    else:
        # Fallback: top third of person bounding box
        ys = np.where(person > 0)[0]
        if len(ys) > 0:
            y_min, y_max = int(ys.min()), int(ys.max())
            face_bot = y_min + (y_max - y_min) // 3
            skin = np.zeros((h, w), dtype=np.uint8)
            skin[y_min:face_bot] = person[y_min:face_bot]
        else:
            skin = np.zeros((h, w), dtype=np.uint8)

    # ── Hair = above face midpoint inside person, color-refined ───────────
    hair = np.zeros((h, w), dtype=np.uint8)
    p_ys = np.where(person > 0)[0]
    if len(p_ys) > 0:
        y_min_p, y_max_p = int(p_ys.min()), int(p_ys.max())

        if face_regions and 'face' in face_regions and np.any(face_regions['face'] > 0):
            f_ys = np.where(face_regions['face'] > 0)[0]
            face_top = int(f_ys.min())
            face_mid = int(f_ys.min() + (f_ys.max() - f_ys.min()) * 0.4)

            hair_base = np.zeros((h, w), dtype=np.uint8)
            hair_base[:face_mid] = person[:face_mid]
            # Include sides of face oval (sideburns / ears)
            if face_top < y_max_p:
                side = np.where(
                    (face_regions['face'] == 0) & (person > 0), 255, 0
                ).astype(np.uint8)
                hair_base[face_top:face_mid] = np.maximum(
                    hair_base[face_top:face_mid], side[face_top:face_mid]
                )
            hair = hair_base
        else:
            cut = y_min_p + (y_max_p - y_min_p) // 4
            hair[:cut] = person[:cut]

        # Color refine: hair pixels are typically dark (< 150)
        if np.any(hair > 0):
            dark = (gray < 150).astype(np.uint8) * 255
            hair = np.where((hair > 0) & (dark > 0), 255, 0).astype(np.uint8)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, k)
            hair = np.where(person > 0, hair, 0).astype(np.uint8)

    # ── Clothes = person minus skin minus hair ────────────────────────────
    surface = np.where((skin > 0) | (hair > 0), 1, 0).astype(np.uint8)
    clothes = np.where((person > 0) & (surface == 0), 255, 0).astype(np.uint8)

    return {'hair': hair, 'skin': skin, 'clothes': clothes, 'person': person}
