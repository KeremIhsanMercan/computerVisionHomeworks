from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
IMAGE_BASENAMES = ("first_image", "last_image")
BOUNDARY_TOLERANCE = 1.5  # pixels

_FACE_MESH = mp.solutions.face_mesh
_FEATURE_SAMPLING_PLAN = (
    (_FACE_MESH.FACEMESH_FACE_OVAL, 20),
    (_FACE_MESH.FACEMESH_LEFT_EYEBROW, 5),
    (_FACE_MESH.FACEMESH_RIGHT_EYEBROW, 5),
    (_FACE_MESH.FACEMESH_NOSE, 12),
    (_FACE_MESH.FACEMESH_LEFT_EYE, 6),
    (_FACE_MESH.FACEMESH_RIGHT_EYE, 6),
    (_FACE_MESH.FACEMESH_LIPS, 14),
)


def _ordered_vertices(edges: Iterable[Tuple[int, int]]) -> List[int]:
    ordered: List[int] = []
    for start, end in edges:
        if start not in ordered:
            ordered.append(start)
        if end not in ordered:
            ordered.append(end)
    return ordered


def _sample_vertices(ordered: Sequence[int], target: int) -> List[int]:
    if target <= 0:
        return []
    if target >= len(ordered):
        return list(ordered)
    if target == 1:
        return [ordered[len(ordered) // 2]]
    step = (len(ordered) - 1) / float(target - 1)
    return [ordered[int(round(i * step))] for i in range(target)]


def _build_landmark_indices() -> Tuple[int, ...]:
    selected: List[int] = []
    for edges, count in _FEATURE_SAMPLING_PLAN:
        ordered = _ordered_vertices(edges)
        for idx in _sample_vertices(ordered, count):
            if idx not in selected:
                selected.append(idx)
            if len(selected) == 68:
                return tuple(selected)
    for idx in range(468):
        if idx not in selected:
            selected.append(idx)
        if len(selected) == 68:
            return tuple(selected)
    raise RuntimeError("Failed to gather 68 landmark indices.")


LANDMARK_INDICES = _build_landmark_indices()


def resolve_image_path(stem: str) -> Path:
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = SCRIPT_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find an image for base name '{stem}'.")


def detect_landmarks(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with _FACE_MESH.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        raise RuntimeError(f"No faces were detected in {image_path}")

    face_landmarks = results.multi_face_landmarks[0]
    height, width = image.shape[:2]
    points: List[Tuple[float, float]] = []
    for idx in LANDMARK_INDICES:
        lm = face_landmarks.landmark[idx]
        x = min(max(lm.x * width, 0.0), width - 1)
        y = min(max(lm.y * height, 0.0), height - 1)
        points.append((x, y))
    landmarks = np.array(points, dtype=np.float32)
    if landmarks.shape[0] != 68:
        raise RuntimeError(f"Expected 68 landmarks, got {landmarks.shape[0]}")
    return image, landmarks


def boundary_points(width: int, height: int) -> np.ndarray:
    w, h = float(width - 1), float(height - 1)
    return np.array(
        [
            (0.0, 0.0),
            (w * 0.5, 0.0),
            (w, 0.0),
            (w, h * 0.5),
            (w, h),
            (w * 0.5, h),
            (0.0, h),
            (0.0, h * 0.5),
        ],
        dtype=np.float32,
    )


def append_boundary(points: np.ndarray, width: int, height: int) -> np.ndarray:
    extra = boundary_points(width, height)
    combined = np.vstack((points, extra))
    if combined.shape[0] != 76:
        raise RuntimeError(f"Expected 76 points after padding, got {combined.shape[0]}")
    return combined


def create_subdiv(width: int, height: int) -> cv2.Subdiv2D:
    rect = (0, 0, int(width), int(height))
    return cv2.Subdiv2D(rect)


def insert_points(subdiv: cv2.Subdiv2D, points: Sequence[Sequence[float]]) -> None:
    for x, y in points:
        subdiv.insert((float(x), float(y)))


def inside_image(point: Tuple[float, float], width: int, height: int) -> bool:
    x, y = point
    return 0 <= x < width and 0 <= y < height


def closest_point_index(point: Tuple[float, float], points: np.ndarray) -> int:
    deltas = points - np.array(point, dtype=np.float32)
    distances = np.sqrt(np.sum(deltas * deltas, axis=1))
    index = int(np.argmin(distances))
    if distances[index] > BOUNDARY_TOLERANCE:
        raise ValueError(f"Could not match triangle point {point} to any known landmark")
    return index


def delaunay_triangles(points: np.ndarray, width: int, height: int) -> List[Tuple[int, int, int]]:
    subdiv = create_subdiv(width, height)
    insert_points(subdiv, points)
    triangles = subdiv.getTriangleList()
    triangle_indices: List[Tuple[int, int, int]] = []
    for tri in triangles:
        pts = [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])]
        if not all(inside_image(pt, width, height) for pt in pts):
            continue
        indices = tuple(closest_point_index(pt, points) for pt in pts)
        triangle_indices.append(indices)
    if not triangle_indices:
        raise RuntimeError("No triangles were created. Check the inserted points.")
    return triangle_indices


def triangles_from_indices(points: np.ndarray, indices: Iterable[Tuple[int, int, int]]) -> np.ndarray:
    flattened: List[List[float]] = []
    for i0, i1, i2 in indices:
        flattened.append([*points[i0], *points[i1], *points[i2]])
    return np.array(flattened, dtype=np.float32)


def draw_triangles(image: np.ndarray, triangles_xy: np.ndarray) -> np.ndarray:
    output = image.copy()
    for triangle in triangles_xy:
        pts = triangle.reshape(3, 2).astype(np.int32)
        cv2.polylines(output, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    images: List[dict] = []
    for stem in IMAGE_BASENAMES:
        image_path = resolve_image_path(stem)
        image, raw_landmarks = detect_landmarks(image_path)
        h, w = image.shape[:2]
        padded_points = append_boundary(raw_landmarks, w, h)
        images.append({
            "stem": stem,
            "path": image_path,
            "image": image,
            "points": padded_points,
        })
        np.save(OUTPUT_DIR / f"{stem}_landmarks.npy", padded_points)
        print(f"Detected {raw_landmarks.shape[0]} landmarks (+8 boundary) for {image_path.name}.")

    width = images[0]["image"].shape[1]
    height = images[0]["image"].shape[0]
    triangle_indices = delaunay_triangles(images[0]["points"], width, height)
    np.save(OUTPUT_DIR / "triangle_indices.npy", np.array(triangle_indices, dtype=np.int32))
    print(f"Created {len(triangle_indices)} triangles on the first image.")

    first_triangles = triangles_from_indices(images[0]["points"], triangle_indices)
    second_triangles = triangles_from_indices(images[1]["points"], triangle_indices)
    np.save(OUTPUT_DIR / "img1_triangles.npy", first_triangles)
    np.save(OUTPUT_DIR / "img2_triangles.npy", second_triangles)

    first_overlay = draw_triangles(images[0]["image"], first_triangles)
    second_overlay = draw_triangles(images[1]["image"], second_triangles)
    cv2.imwrite(str(OUTPUT_DIR / f"{images[0]['stem']}_triangles.png"), first_overlay)
    cv2.imwrite(str(OUTPUT_DIR / f"{images[1]['stem']}_triangles.png"), second_overlay)
    print("Triangle visualizations saved under 'outputs/'.")


if __name__ == "__main__":
    main()
