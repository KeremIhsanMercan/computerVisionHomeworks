import os
import time
from collections import Counter
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyautogui


# --- Tunables ---
NUM_SNAPSHOTS_PER_DECISION = 3
SLEEP_BETWEEN_SHOTS_SEC = 0.03

# Hough settings for pip detection on the *warped top face*
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 18
HOUGH_PARAM1 = 120
HOUGH_PARAM2_CANDIDATES = (22, 20, 18, 16, 14)
HOUGH_MIN_RADIUS = 6
HOUGH_MAX_RADIUS = 28

# False-positive filter: a detected pip must be sufficiently darker than the face.
# Example: 0.80 means pip patch must be <= 80% of the face median intensity.
PIP_DARKNESS_RATIO = 0.80


def take_screenshot_pil():
    return pyautogui.screenshot()


def pil_to_bgr(pil_img) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def save_image(image_bgr: np.ndarray, output_path: str) -> None:
    cv2.imwrite(output_path, image_bgr)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4, 2)
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad_to_square(image_bgr: np.ndarray, quad: np.ndarray, out_size: Optional[int] = None) -> np.ndarray:
    quad = order_quad_points(quad.reshape(4, 2))
    (tl, tr, br, bl) = quad

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    size = int(max(max_w, max_h))
    if out_size is not None:
        size = int(out_size)
    size = int(np.clip(size, 100, 420))

    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image_bgr, M, (size, size), flags=cv2.INTER_LINEAR)


def find_top_face_quad(dice_bgr: np.ndarray) -> Optional[np.ndarray]:
    # Detect candidate quadrilaterals; choose a large one near the top.
    h, w = dice_bgr.shape[:2]
    gray = cv2.cvtColor(dice_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Use RETR_LIST to include internal quads (top-face border can be an inner contour).
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_quad = None
    best_score = -1.0

    img_area = float(h * w)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * img_area or area > 0.95 * img_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)

        # Reject extremely skinny quads (often noise / borders).
        ordered = order_quad_points(pts)
        side_lengths = [
            float(np.linalg.norm(ordered[1] - ordered[0])),
            float(np.linalg.norm(ordered[2] - ordered[1])),
            float(np.linalg.norm(ordered[3] - ordered[2])),
            float(np.linalg.norm(ordered[0] - ordered[3])),
        ]
        if min(side_lengths) / max(1.0, max(side_lengths)) < 0.35:
            continue

        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        # Prefer bigger quads and those higher up in the crop.
        top_bonus = 1.0 + (0.6 * (1.0 - (cy / max(1.0, h))))
        center_penalty = 1.0 - 0.15 * abs((cx / max(1.0, w)) - 0.5)
        score = float(area) * top_bonus * center_penalty

        if score > best_score:
            best_score = score
            best_quad = pts

    return best_quad


def extract_three_dice_crops(frame_bgr: np.ndarray) -> List[np.ndarray]:
    # Extract 3 main dice regions by contour area, then sort by x position (left->right).
    img = frame_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    candidates: List[Tuple[int, int, int, int, float]] = []
    h, w = img.shape[:2]
    img_area = float(h * w)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * img_area or area > 0.60 * img_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        aspect = float(bw) / float(bh)
        # Dice crops are roughly square-ish; reject long strips and global borders.
        if aspect < 0.45 or aspect > 2.2:
            continue
        # ignore likely text strip at bottom
        if y > int(0.70 * h):
            continue
        candidates.append((x, y, bw, bh, area))

    if not candidates:
        return []

    # Keep top 3 by area, then sort left->right.
    candidates.sort(key=lambda t: t[4], reverse=True)
    candidates = candidates[:3]
    candidates.sort(key=lambda t: t[0])

    crops = []
    for (x, y, bw, bh, _) in candidates:
        pad = 10
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        crops.append(img[y0:y1, x0:x1])
    return crops


def _count_pips_from_gray(gray: np.ndarray) -> int:
    gray = cv2.medianBlur(gray, 5)
    face_median = float(np.median(gray))
    if face_median <= 1.0:
        return 0

    size = min(gray.shape[:2])
    min_dist = max(10, int(0.18 * size))
    min_r = max(4, int(0.05 * size))
    max_r = max(min_r + 2, int(0.14 * size))

    best_count = 0
    for p2 in HOUGH_PARAM2_CANDIDATES:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=min_dist,
            param1=HOUGH_PARAM1,
            param2=float(p2),
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            continue

        circles = np.round(circles[0]).astype(int)  # (N, 3)
        accepted = []
        for (x, y, r) in circles:
            if x < 0 or y < 0 or x >= gray.shape[1] or y >= gray.shape[0]:
                continue
            rr = max(2, int(0.35 * r))
            x0 = max(0, x - rr)
            y0 = max(0, y - rr)
            x1 = min(gray.shape[1], x + rr + 1)
            y1 = min(gray.shape[0], y + rr + 1)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            if float(np.mean(patch)) <= PIP_DARKNESS_RATIO * face_median:
                accepted.append((x, y, r))

        count = len(accepted)
        if 0 <= count <= 6:
            best_count = max(best_count, count)
            # Early stop if perfect plausible max reached
            if best_count == 6:
                break

    return best_count


def count_top_face_pips(dice_bgr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
    quad = find_top_face_quad(dice_bgr)
    if quad is None:
        return 0, None
    

    warped = warp_quad_to_square(dice_bgr, quad)
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Crop a margin to reduce border influence
    m = int(0.08 * gray.shape[0])
    if m > 0:
        gray = gray[m:-m, m:-m]

    count = _count_pips_from_gray(gray)
    return count, warped


def vote_counts_across_frames(counts_per_frame: List[List[int]]) -> Tuple[List[int], List[float]]:
    # counts_per_frame: [frame_idx][die_idx]
    if not counts_per_frame:
        return [], []
    num_dice = len(counts_per_frame[0])
    final_counts = []
    error_rates = []
    for di in range(num_dice):
        values = [c[di] for c in counts_per_frame if di < len(c)]
        if not values:
            final_counts.append(0)
            error_rates.append(1.0)
            continue
        mode_val, mode_freq = Counter(values).most_common(1)[0]
        final_counts.append(int(mode_val))
        error_rates.append(float(1.0 - (mode_freq / max(1, len(values)))))
    return final_counts, error_rates



if __name__ == "__main__":
    print("Starting in 2 seconds. Click on the game window.")
    time.sleep(2)  # Initial delay to click on the game window
    ss_dir = "ss"
    cropped_image_dir = "dices"
    os.makedirs(ss_dir, exist_ok=True)
    os.makedirs(cropped_image_dir, exist_ok=True)

    while True:
        # Take multiple screenshots and vote, to reduce Hough instability.
        counts_per_frame: List[List[int]] = []

        for shot_idx in range(NUM_SNAPSHOTS_PER_DECISION):
            ss_path = os.path.join(ss_dir, "screenshot.png")
            pil_img = take_screenshot_pil()
            pil_img.save(ss_path)
            frame_bgr = pil_to_bgr(pil_img)

            dice_crops = extract_three_dice_crops(frame_bgr)
            if len(dice_crops) != 3:
                # If extraction fails, skip this shot.
                time.sleep(SLEEP_BETWEEN_SHOTS_SEC)
                continue

            shot_counts: List[int] = []
            for i, dice_crop in enumerate(dice_crops):
                dice_path = os.path.join(cropped_image_dir, f"dices{i}.png")
                save_image(dice_crop, dice_path)

                count, warped = count_top_face_pips(dice_crop)
                shot_counts.append(int(count))

                if warped is not None:
                    warped_path = os.path.join(cropped_image_dir, f"topface_warped{i}.png")
                    save_image(warped, warped_path)

            counts_per_frame.append(shot_counts)
            time.sleep(SLEEP_BETWEEN_SHOTS_SEC)

        final_counts, error_rates = vote_counts_across_frames(counts_per_frame)
        if len(final_counts) != 3:
            time.sleep(0.2)
            continue

        print(f"Top-face pip counts (A,S,D): {final_counts} | error: {error_rates}")

        # Decide: press the key for the die with the most pips.
        # Dice order is left->right which matches A,S,D on screen.
        keys = ['a', 's', 'd']
        best_idx = int(np.argmax(np.array(final_counts)))

        # Optional confidence gate: if too unstable, do nothing this loop.
        if error_rates[best_idx] <= 0.50:
            pyautogui.press(keys[best_idx])

        time.sleep(0.2)  # Delay before the next iteration