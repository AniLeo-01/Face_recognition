#!/usr/bin/env python3
"""Demo script — demonstrates the full face recognition pipeline.

Usage:
    python scripts/demo.py

This script:
1. Initializes the pipeline
2. Enrolls test identities
3. Runs recognition on test scenes
4. Saves annotated output images
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    AlignmentConfig,
    DetectionConfig,
    RecognitionConfig,
    SystemConfig,
)
from src.pipeline.processor import FaceRecognitionPipeline

SAMPLES_DIR = ROOT / "test_samples"
OUTPUT_DIR = ROOT / "test_samples" / "demo_output"


def main() -> None:
    print("=" * 60)
    print("Face Recognition System — Pipeline Demo")
    print("=" * 60)

    # Initialize
    config = SystemConfig(
        detection=DetectionConfig(device="cpu", confidence_threshold=0.5),
        alignment=AlignmentConfig(output_size=(160, 160), min_blur_score=5.0),
        recognition=RecognitionConfig(device="cpu", similarity_threshold=0.4),
    )
    pipeline = FaceRecognitionPipeline(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Enroll identities ---
    print("\n[Step 1] Enrolling test identities...")
    faces_dir = SAMPLES_DIR / "faces"
    enrolled = 0
    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if images:
            result = pipeline.enroll(person_dir.name, person_dir.name, images)
            print(
                f"  Enrolled {person_dir.name}: "
                f"{result['embeddings_created']}/{result['images_processed']} images"
            )
            enrolled += 1

    print(f"  → {enrolled} identities enrolled, gallery size: {pipeline.gallery.size}")

    # --- Step 2: Run detection on scenes ---
    print("\n[Step 2] Running detection on scene images...")
    scenes_dir = SAMPLES_DIR / "scenes"
    for scene_path in sorted(scenes_dir.glob("*.jpg")):
        img = cv2.imread(str(scene_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pipeline.detector.detect(rgb)
        print(
            f"  {scene_path.name}: {result.num_faces} faces detected "
            f"({result.inference_time_ms:.1f}ms)"
        )

    # --- Step 3: Run full recognition pipeline ---
    print("\n[Step 3] Running full recognition pipeline...")
    for scene_path in sorted(scenes_dir.glob("*.jpg")):
        img = cv2.imread(str(scene_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pipeline.process_frame(rgb, skip_quality=True)

        print(f"\n  Scene: {scene_path.name}")
        print(f"    Detected: {result.num_detected}, Recognized: {result.num_recognized}")
        print(f"    Total time: {result.total_time_ms:.1f}ms")

        for face in result.recognized_faces:
            status = f"{face.identity_name} (sim={face.similarity:.3f})" if face.identity_id else "Unknown"
            print(f"    Face at {[int(v) for v in face.detected_face.bbox]}: {status}")

        # Save annotated output
        annotated = pipeline.annotate_frame(rgb, result)
        out_path = OUTPUT_DIR / f"annotated_{scene_path.name}"
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"    → Saved: {out_path}")

    # --- Step 4: Gallery info ---
    print(f"\n[Step 4] Gallery Summary:")
    print(f"  Total embeddings: {pipeline.gallery.size}")
    print(f"  Unique identities: {pipeline.gallery.identity_count}")
    for identity in pipeline.gallery.get_all_identities():
        print(f"    - {identity['name']}: {identity['num_embeddings']} embeddings")

    # --- Step 5: Save gallery ---
    gallery_path = OUTPUT_DIR / "demo_gallery.pkl"
    pipeline.gallery.save(gallery_path)
    print(f"\n  Gallery saved to: {gallery_path}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
