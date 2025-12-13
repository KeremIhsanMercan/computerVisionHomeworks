import os
import pyautogui
import time
import numpy as np
import cv2

_LAST_FACE_STATE = None


def _ensure_bgr(image: np.ndarray, image_is_rgb: bool = True) -> np.ndarray:
    """Convert an image to BGR for OpenCV ops.

    In this homework script, frames come from `pyautogui.screenshot()` which
    becomes an RGB ndarray via `np.array(...)`, so `image_is_rgb=True` is the
    right default.
    """
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR) if image_is_rgb else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image_is_rgb:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _find_face_roi(crop_bgr: np.ndarray) -> np.ndarray:
    """Find the face button ROI inside the provided crop.

    Uses a simple yellow-ish mask and picks the largest contour.
    Falls back to the full crop if segmentation fails.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Wide yellow range for classic Minesweeper-like face background.
    lower = np.array([15, 40, 80], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.05 * float(h * w):
        return crop_bgr

    x, y, bw, bh = cv2.boundingRect(best)
    pad = int(0.06 * max(bw, bh))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    return crop_bgr[y0:y1, x0:x1]


def _connected_components(binary: np.ndarray):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    comps = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = centroids[i]
        comps.append({"id": i, "x": x, "y": y, "w": w, "h": h, "area": area, "cx": float(cx), "cy": float(cy)})
    return comps


def face_recognition(image: np.ndarray, debug: bool = False, image_is_rgb: bool = True) -> str:
    """Classify the face state as 'shocked' or 'normal'.

    Strategy (landmark-like):
    - Extract dark facial features (eyes/mouth) via adaptive threshold.
    - Estimate mouth "openness" from the largest mouth blob height/area.

    Returns: 'shocked', 'normal', or 'unknown'.
    """
    global _LAST_FACE_STATE

    crop_bgr = _ensure_bgr(image, image_is_rgb=image_is_rgb)
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    face_bgr = _find_face_roi(crop_bgr)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # Trim border to reduce the button frame influence.
    h, w = gray.shape[:2]
    margin = int(0.08 * min(h, w))
    if margin > 0 and (h - 2 * margin) > 20 and (w - 2 * margin) > 20:
        gray = gray[margin : h - margin, margin : w - margin]

    # Segment dark facial features (eyes/mouth) against a relatively bright face.
    # Median-relative threshold is more stable here than adaptive thresholding.
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(g))
    # If the crop is unexpectedly dark, avoid collapsing everything into foreground.
    thr_val = max(10.0, 0.65 * med)
    binary = (g < thr_val).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Define regions for landmark extraction.
    bh, bw = binary.shape[:2]
    mouth_y0 = int(0.55 * bh)
    eye_y1 = int(0.48 * bh)
    mouth = binary[mouth_y0:bh, :]
    eyes = binary[0:eye_y1, :]

    # Mouth "openness" score: ratio + largest blob height.
    mouth_ratio = float(np.mean(mouth > 0)) if mouth.size else 0.0
    mouth_comps = _connected_components(mouth)
    mouth_comps = [c for c in mouth_comps if c["area"] >= 40]
    mouth_blob_h = max((c["h"] for c in mouth_comps), default=0)
    mouth_blob_area = max((c["area"] for c in mouth_comps), default=0)

    # Eye landmarks (not strictly needed for classification, but useful for debug).
    eye_comps = _connected_components(eyes)
    eye_comps = [c for c in eye_comps if c["area"] >= 30]
    eye_comps.sort(key=lambda c: c["area"], reverse=True)
    eye_centers = [(c["cx"], c["cy"]) for c in eye_comps[:2]]

    mouth_h_norm = 0.0
    if mouth.size:
        mouth_h_norm = float(mouth_blob_h) / float(max(1, mouth.shape[0]))

    # Calibrated against sample screenshots in part3/ss:
    # - normal: mouth_ratio ~ 0.10-0.14, mouth_blob_area ~ 1k
    # - shocked: mouth_ratio ~ 0.23-0.26, mouth_blob_area ~ 4-5k
    # Prefer area/ratio over blob height, since a thin border artifact can be tall.
    shocked = (mouth_ratio >= 0.18) or (mouth_blob_area >= 2500)
    state = "shocked" if shocked else "normal"

    if debug:
        print(
            f"state={state} med={med:.1f} thr={thr_val:.1f} mouth_ratio={mouth_ratio:.3f} mouth_blob_h={mouth_blob_h} "
            f"mouth_h_norm={mouth_h_norm:.3f} mouth_blob_area={mouth_blob_area} eyes={eye_centers}"
        )

    # Avoid spamming logs: only print when state changes.
    if _LAST_FACE_STATE != state:
        print(f"Face state: {state}")
        _LAST_FACE_STATE = state

    return state

def crop_image(image, crop_box):
    x, y, w, h = crop_box
    return image[y:h, x:w]

if __name__ == "__main__":
    print("Click on the game window.")
    time.sleep(3)  # Give user time to click on the game window
    while True:
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)

        crop_box = (2243, 1124, screenshot_np.shape[1], screenshot_np.shape[0])  # Example crop box (x, y, width, height)
        cropped_image = crop_image(screenshot_np, crop_box)
        face_recognition(cropped_image)

        time.sleep(0.5)  # Wait for 5 seconds before taking the next screenshot