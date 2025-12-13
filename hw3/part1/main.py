import glob
import os
from typing import List, Tuple

import cv2
import moviepy.video.io.VideoFileClip as mpy
import numpy as np


def extract_frames(video_path: str, frames_dir: str) -> Tuple[int, float]:
    """Extract frames if they are not already on disk."""
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    existing = glob.glob(os.path.join(frames_dir, "frame_*.png"))
    if existing:
        sample_clip = mpy.VideoFileClip(video_path)
        return len(existing), sample_clip.fps

    clip = mpy.VideoFileClip(video_path)
    frame_count = clip.reader.n_frames
    fps = clip.fps
    for i in range(frame_count):
        rgb_frame = clip.get_frame(i * 1.0 / fps)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.png"), bgr_frame)
    return frame_count, fps


def list_frame_paths(frames_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not paths:
        raise FileNotFoundError("No frames found; run extract_frames first.")
    return paths


def get_green_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([63, 105, 178], dtype=np.uint8)
    upper_green = np.array([81, 203, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def detect_initial_right_hand(frame_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Find a seed point on the green ragdoll's right hand."""
    if mask is None:
        mask = get_green_mask(frame_bgr)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return np.array([[[240.0, 140.0]]], dtype=np.float32)  # conservative fallback

    valid_ids = []
    y_coords = mask.nonzero()[0]
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    top_cutoff = y_min + int(0.65 * (y_max - y_min))

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        cx, cy = centroids[label_id]
        if area < 15 or area > 1200:
            continue
        if cy > top_cutoff:
            continue
        valid_ids.append((cx, cy, area, label_id))

    if not valid_ids:
        # Fall back to the largest green blob if heuristic fails
        main_id = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        cx, cy = centroids[main_id]
        return np.array([[[cx, cy]]], dtype=np.float32)

    # Pick the rightmost centroid in the upper body as the hand seed
    cx, cy, _, _ = max(valid_ids, key=lambda t: t[0])
    return np.array([[[cx, cy]]], dtype=np.float32)


def blend_with_future(prev_gray: np.ndarray, prev_pt: np.ndarray, curr_gray: np.ndarray, future_gray: np.ndarray) -> np.ndarray:
    """Use an extra frame to stabilize the per-frame displacement."""
    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    next_pt, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pt, None, **lk_params)
    if st1[0] != 1:
        return None

    if future_gray is None:
        return next_pt

    future_pt, st2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, future_gray, prev_pt, None, **lk_params)
    if st2[0] != 1:
        return next_pt

    # Average the one-step flow with a halved two-step flow for stability
    one_step = next_pt
    two_step = prev_pt + 0.5 * (future_pt - prev_pt)
    blended = 0.7 * one_step + 0.3 * two_step
    return blended


def draw_arrow(frame_bgr: np.ndarray, start_pt: np.ndarray, end_pt: np.ndarray) -> np.ndarray:
    canvas = frame_bgr.copy()
    start = tuple(np.round(start_pt.ravel()).astype(int))
    end = tuple(np.round(end_pt.ravel()).astype(int))
    cv2.arrowedLine(canvas, start, end, (255, 255, 255), 2, tipLength=0.3)
    cv2.circle(canvas, start, 3, (255, 0, 0), -1)
    return canvas


def track_right_hand(frames_dir: str = "frames", output_dir: str = "tracked_frames", mask_dir: str = "masks") -> None:
    frame_paths = list_frame_paths(frames_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    first_frame = cv2.imread(frame_paths[0])
    first_mask = get_green_mask(first_frame)
    cv2.imwrite(os.path.join(mask_dir, "mask_0000.png"), first_mask)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_pt = detect_initial_right_hand(first_frame, first_mask)

    for idx in range(1, len(frame_paths)):
        curr_frame = cv2.imread(frame_paths[idx])
        curr_mask = get_green_mask(curr_frame)
        cv2.imwrite(os.path.join(mask_dir, f"mask_{idx:04d}.png"), curr_mask)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        future_gray = None
        if idx + 1 < len(frame_paths):
            future_frame = cv2.imread(frame_paths[idx + 1])
            future_gray = cv2.cvtColor(future_frame, cv2.COLOR_BGR2GRAY)

        # Use plain single-step LK (disable future blending for stability)
        lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        next_pt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pt, None, **lk_params)
        if st[0] != 1 or next_pt is None:
            # Re-detect on the current frame if tracking is lost
            next_pt = detect_initial_right_hand(curr_frame, curr_mask)

        flow_vec = next_pt - prev_pt
        end_pt = next_pt + 4.0 * flow_vec  # scale for visibility
        drawn = draw_arrow(curr_frame, next_pt, end_pt)

        out_path = os.path.join(output_dir, f"flow_{idx:04d}.png")
        cv2.imwrite(out_path, drawn)

        prev_pt = next_pt
        prev_gray = curr_gray


def frames_to_video(output_dir: str, fps: float, video_path: str = "tracked_output.avi") -> None:
    frame_paths = sorted(glob.glob(os.path.join(output_dir, "flow_*.png")))
    if not frame_paths:
        return
    sample = cv2.imread(frame_paths[0])
    h, w, _ = sample.shape
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for path in frame_paths:
        frame = cv2.imread(path)
        writer.write(frame)
    writer.release()


def main():
    video_path = "biped_1.avi"
    frames_dir = "frames"
    output_dir = "tracked_frames"

    frame_count, fps = extract_frames(video_path, frames_dir)
    print(f"Frames ready: {frame_count} at {fps:.2f} fps")

    track_right_hand(frames_dir, output_dir)
    frames_to_video(output_dir, fps)
    print(f"Tracked frames saved to '{output_dir}' and video exported.")


if __name__ == "__main__":
    main()
    