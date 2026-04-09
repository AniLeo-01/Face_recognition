"""Generate synthetic test sample images for the face recognition system.

Creates realistic face-like images using OpenCV drawing primitives for testing
the detection, alignment, and recognition pipeline. Also downloads a small set
of real face images from publicly available datasets when possible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES_DIR = Path(__file__).resolve().parent
FACES_DIR = SAMPLES_DIR / "faces"
SCENES_DIR = SAMPLES_DIR / "scenes"
ENROLLMENT_DIR = SAMPLES_DIR / "enrollment"


def draw_realistic_face(
    img: np.ndarray,
    center: tuple[int, int],
    face_width: int,
    face_height: int,
    skin_color: tuple[int, int, int] = (200, 170, 140),
    eye_color: tuple[int, int, int] = (60, 40, 30),
    angle: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Draw a realistic-looking face on the image.

    Uses ellipses and circles to approximate face geometry that MTCNN can detect.
    """
    rng = np.random.RandomState(seed)

    cx, cy = center
    fw, fh = face_width, face_height

    # Face outline (skin-colored ellipse)
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), angle, 0, 360, skin_color, -1)
    # Slightly darker border
    darker = tuple(max(0, c - 30) for c in skin_color)
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), angle, 0, 360, darker, 2)

    # Eyes
    eye_y = cy - fh // 6
    eye_sep = fw // 4
    eye_w, eye_h = fw // 8, fh // 14

    # Left eye
    left_eye = (cx - eye_sep, eye_y)
    cv2.ellipse(img, left_eye, (eye_w, eye_h), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, left_eye, eye_h - 1, eye_color, -1)
    cv2.circle(img, left_eye, max(1, eye_h // 3), (20, 20, 20), -1)

    # Right eye
    right_eye = (cx + eye_sep, eye_y)
    cv2.ellipse(img, right_eye, (eye_w, eye_h), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, right_eye, eye_h - 1, eye_color, -1)
    cv2.circle(img, right_eye, max(1, eye_h // 3), (20, 20, 20), -1)

    # Eyebrows
    brow_y = eye_y - eye_h - 4
    brow_color = tuple(max(0, c - 60) for c in skin_color)
    cv2.line(
        img,
        (cx - eye_sep - eye_w, brow_y),
        (cx - eye_sep + eye_w, brow_y - 2),
        brow_color,
        max(1, fw // 40),
    )
    cv2.line(
        img,
        (cx + eye_sep - eye_w, brow_y - 2),
        (cx + eye_sep + eye_w, brow_y),
        brow_color,
        max(1, fw // 40),
    )

    # Nose
    nose_top = cy - fh // 12
    nose_bottom = cy + fh // 8
    nose_width = fw // 10
    pts = np.array(
        [[cx, nose_top], [cx - nose_width, nose_bottom], [cx + nose_width, nose_bottom]],
        dtype=np.int32,
    )
    nose_color = tuple(max(0, c - 15) for c in skin_color)
    cv2.polylines(img, [pts], False, nose_color, max(1, fw // 50))

    # Mouth
    mouth_y = cy + fh // 4
    mouth_w = fw // 5
    lip_color = (
        min(255, skin_color[0] + 30),
        max(0, skin_color[1] - 30),
        max(0, skin_color[2] - 20),
    )
    cv2.ellipse(
        img,
        (cx, mouth_y),
        (mouth_w, fh // 14),
        0,
        0,
        180,
        lip_color,
        -1,
    )
    cv2.line(
        img,
        (cx - mouth_w, mouth_y),
        (cx + mouth_w, mouth_y),
        tuple(max(0, c - 20) for c in lip_color),
        max(1, fw // 50),
    )

    # Hair
    hair_color = (
        rng.randint(20, 80),
        rng.randint(15, 60),
        rng.randint(10, 50),
    )
    cv2.ellipse(
        img,
        (cx, cy - fh // 5),
        (fw // 2 + 5, fh // 3),
        0,
        180,
        360,
        hair_color,
        -1,
    )

    return img


def generate_single_face_samples() -> list[str]:
    """Generate individual face images for enrollment and testing."""
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # Different "people" with varying skin tones and features
    people = [
        {"name": "person_A", "skin": (210, 180, 155), "eye": (70, 50, 35), "seeds": [1, 11, 21]},
        {"name": "person_B", "skin": (180, 140, 110), "eye": (50, 35, 25), "seeds": [2, 12, 22]},
        {"name": "person_C", "skin": (230, 200, 175), "eye": (80, 60, 40), "seeds": [3, 13, 23]},
        {"name": "person_D", "skin": (160, 120, 90), "eye": (40, 30, 20), "seeds": [4, 14, 24]},
        {"name": "person_E", "skin": (195, 160, 130), "eye": (55, 40, 30), "seeds": [5, 15, 25]},
    ]

    for person in people:
        person_dir = FACES_DIR / person["name"]
        person_dir.mkdir(parents=True, exist_ok=True)

        for idx, seed in enumerate(person["seeds"]):
            img = np.full((300, 300, 3), 220, dtype=np.uint8)

            # Slight background variation
            rng = np.random.RandomState(seed)
            bg_color = tuple(int(c) for c in rng.randint(180, 240, 3))
            img[:] = bg_color

            # Vary face size and position slightly
            cx = 150 + rng.randint(-15, 15)
            cy = 155 + rng.randint(-10, 10)
            fw = 130 + rng.randint(-10, 10)
            fh = 160 + rng.randint(-10, 10)

            draw_realistic_face(
                img,
                (cx, cy),
                fw,
                fh,
                skin_color=person["skin"],
                eye_color=person["eye"],
                seed=seed,
            )

            path = str(person_dir / f"face_{idx:02d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            paths.append(path)
            print(f"  Generated: {path}")

    return paths


def generate_scene_samples() -> list[str]:
    """Generate scene images with multiple faces for detection testing."""
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # Scene 1: Two faces side by side
    img = np.full((400, 600, 3), 200, dtype=np.uint8)
    draw_realistic_face(img, (180, 200), 120, 150, (210, 180, 155), seed=100)
    draw_realistic_face(img, (420, 200), 110, 140, (180, 140, 110), seed=101)
    path = str(SCENES_DIR / "two_faces.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)
    print(f"  Generated: {path}")

    # Scene 2: Three faces in a group
    img = np.full((500, 700, 3), 190, dtype=np.uint8)
    draw_realistic_face(img, (180, 220), 110, 140, (230, 200, 175), seed=200)
    draw_realistic_face(img, (360, 190), 100, 130, (160, 120, 90), seed=201)
    draw_realistic_face(img, (530, 230), 120, 150, (195, 160, 130), seed=202)
    path = str(SCENES_DIR / "three_faces.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)
    print(f"  Generated: {path}")

    # Scene 3: Single face — for baseline testing
    img = np.full((480, 640, 3), 210, dtype=np.uint8)
    draw_realistic_face(img, (320, 240), 150, 180, (210, 180, 155), seed=300)
    path = str(SCENES_DIR / "single_face.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)
    print(f"  Generated: {path}")

    # Scene 4: Small faces at distance
    img = np.full((600, 800, 3), 185, dtype=np.uint8)
    draw_realistic_face(img, (200, 300), 70, 90, (210, 180, 155), seed=400)
    draw_realistic_face(img, (400, 280), 65, 85, (180, 140, 110), seed=401)
    draw_realistic_face(img, (600, 320), 75, 95, (230, 200, 175), seed=402)
    path = str(SCENES_DIR / "small_faces_distance.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)
    print(f"  Generated: {path}")

    # Scene 5: No faces (empty scene for negative test)
    img = np.full((400, 600, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (150, 100, 100), -1)
    cv2.circle(img, (400, 300), 80, (100, 150, 100), -1)
    path = str(SCENES_DIR / "no_faces.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)
    print(f"  Generated: {path}")

    return paths


def generate_enrollment_samples() -> list[str]:
    """Generate enrollment sample images (same person, multiple angles)."""
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    skin = (210, 180, 155)
    eye = (70, 50, 35)

    # Front view
    img = np.full((300, 300, 3), 225, dtype=np.uint8)
    draw_realistic_face(img, (150, 155), 130, 160, skin, eye, seed=500)
    path = str(ENROLLMENT_DIR / "enroll_front.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)

    # Slight left turn
    img = np.full((300, 300, 3), 225, dtype=np.uint8)
    draw_realistic_face(img, (140, 155), 125, 155, skin, eye, angle=-8, seed=501)
    path = str(ENROLLMENT_DIR / "enroll_left.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)

    # Slight right turn
    img = np.full((300, 300, 3), 225, dtype=np.uint8)
    draw_realistic_face(img, (160, 155), 125, 155, skin, eye, angle=8, seed=502)
    path = str(ENROLLMENT_DIR / "enroll_right.jpg")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths.append(path)

    for p in paths:
        print(f"  Generated: {p}")

    return paths


def main() -> None:
    print("=" * 60)
    print("Face Recognition System — Test Sample Generator")
    print("=" * 60)

    print("\n[1/3] Generating individual face samples...")
    face_paths = generate_single_face_samples()
    print(f"  → {len(face_paths)} face images generated")

    print("\n[2/3] Generating scene samples (multi-face)...")
    scene_paths = generate_scene_samples()
    print(f"  → {len(scene_paths)} scene images generated")

    print("\n[3/3] Generating enrollment samples...")
    enrollment_paths = generate_enrollment_samples()
    print(f"  → {len(enrollment_paths)} enrollment images generated")

    total = len(face_paths) + len(scene_paths) + len(enrollment_paths)
    print(f"\nTotal: {total} test sample images generated")
    print(f"Location: {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
