import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None

_LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
         308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
         185, 40, 39, 37, 0, 267, 269, 270, 409]
_NOSE = [1, 2, 5, 4, 6, 168, 195, 197, 48, 115, 220, 45, 275, 440, 344, 278]
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def get_face_regions(img_bgr: np.ndarray) -> dict:
    """
    Detect face regions via MediaPipe Face Mesh (468 landmarks, up to 10 faces).
    Returns dict of uint8 masks (255=in region, 0=outside).
    Keys: 'face', 'eyes', 'mouth', 'nose', 'eyebrows'
    Returns empty dict when MediaPipe is unavailable or no face found.
    """
    if not _MP_AVAILABLE:
        print("  MediaPipe not installed — skipping face landmark detection")
        return {}

    h, w = img_bgr.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    ) as mesh:
        results = mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print("  No faces detected by MediaPipe")
        return {}

    out = {k: np.zeros((h, w), dtype=np.uint8)
           for k in ('face', 'eyes', 'mouth', 'nose', 'eyebrows')}

    for face_lm in results.multi_face_landmarks:
        lm = face_lm.landmark

        def pts_px(idxs):
            return np.array(
                [[int(np.clip(lm[i].x * w, 0, w - 1)),
                  int(np.clip(lm[i].y * h, 0, h - 1))]
                 for i in idxs],
                dtype=np.int32,
            )

        def filled_expanded_rect(idxs, factor=2.0):
            pts = pts_px(idxs)
            x, y, rw, rh = cv2.boundingRect(pts)
            cx, cy = x + rw // 2, y + rh // 2
            nw, nh = int(rw * factor) // 2, int(rh * factor) // 2
            x0 = int(np.clip(cx - nw, 0, w - 1))
            y0 = int(np.clip(cy - nh, 0, h - 1))
            x1 = int(np.clip(cx + nw, 0, w - 1))
            y1 = int(np.clip(cy + nh, 0, h - 1))
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)
            return m

        cv2.fillPoly(out['face'], [pts_px(_FACE_OVAL)], 255)

        le_m = filled_expanded_rect(_LEFT_EYE, factor=2.5)
        re_m = filled_expanded_rect(_RIGHT_EYE, factor=2.5)
        out['eyes'] = np.maximum(out['eyes'], np.maximum(le_m, re_m))

        out['mouth'] = np.maximum(out['mouth'], filled_expanded_rect(_LIPS, factor=1.8))
        out['nose'] = np.maximum(out['nose'], filled_expanded_rect(_NOSE, factor=1.6))

        leb_m = filled_expanded_rect(_LEFT_EYEBROW, factor=1.8)
        reb_m = filled_expanded_rect(_RIGHT_EYEBROW, factor=1.8)
        out['eyebrows'] = np.maximum(out['eyebrows'], np.maximum(leb_m, reb_m))

    n_faces = len(results.multi_face_landmarks)
    print(f"  Face landmarks: {n_faces} face(s) detected")
    return out
