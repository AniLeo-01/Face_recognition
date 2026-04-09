"""
test_character.py
-----------------
Enroll character_1 and check if it is recognised in scene_1.

Usage
-----
  # Make sure the server is already running:
  #   make run   (or: python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000)

  python scripts/test_character.py \
      --enroll  test_samples/custom_test/enrollment/character_1.jpg \
      --scene   test_samples/custom_test/scenes/scene_1.jpg

  # Optional flags:
  #   --server  http://localhost:8000   (default)
  #   --name    character_1             (identity label, default: character_1)
  #   --output  test_samples/custom_test/output   (where to save annotated image)
  #   --no-annotate                     (skip saving annotated output)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_server(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def enroll_identity(base_url: str, name: str, image_path: Path) -> dict:
    """POST /enroll/ with one image and return the JSON response."""
    with image_path.open("rb") as fh:
        resp = requests.post(
            f"{base_url}/enroll/",
            files={"files": (image_path.name, fh, "image/jpeg")},
            data={"name": name},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()


def recognize_scene(base_url: str, image_path: Path) -> dict:
    """POST /recognize/ and return the JSON response."""
    with image_path.open("rb") as fh:
        resp = requests.post(
            f"{base_url}/recognize/",
            files={"file": (image_path.name, fh, "image/jpeg")},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()


def annotate_and_save(scene_path: Path, result: dict, output_dir: Path) -> Path | None:
    """Draw bounding boxes + labels on the scene and save to output_dir."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  [WARN] opencv-python not available — skipping annotation.")
        return None

    img = cv2.imread(str(scene_path))
    if img is None:
        print(f"  [WARN] Could not load {scene_path} for annotation.")
        return None

    faces = result.get("faces", [])
    for face in faces:
        identity = face.get("identity", {})
        name     = identity.get("name", "Unknown")
        conf     = identity.get("confidence", identity.get("similarity", 0.0))
        matched  = identity.get("matched", False)
        bbox     = face.get("bounding_box", face.get("bbox", {}))

        # Parse bounding box (handles dict or list)
        if isinstance(bbox, dict):
            x1 = int(bbox.get("x1", bbox.get("left",   0)))
            y1 = int(bbox.get("y1", bbox.get("top",    0)))
            x2 = int(bbox.get("x2", bbox.get("right",  img.shape[1])))
            y2 = int(bbox.get("y2", bbox.get("bottom", img.shape[0])))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        else:
            continue

        color = (0, 200, 0) if matched else (0, 0, 200)
        label = f"{name}  {conf:.2f}" if isinstance(conf, float) else name

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Background rectangle for text legibility
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "scene_1_annotated.jpg"
    cv2.imwrite(str(out_path), img)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll + recognise character in scene.")
    parser.add_argument("--enroll",  required=True, help="Path to character_1 enrollment image")
    parser.add_argument("--scene",   required=True, help="Path to scene_1 image")
    parser.add_argument("--server",  default="http://localhost:8000", help="API base URL")
    parser.add_argument("--name",    default="character_1", help="Identity name to use")
    parser.add_argument("--output",  default="test_samples/custom_test/output",
                        help="Directory to save annotated output")
    parser.add_argument("--no-annotate", action="store_true", help="Skip saving annotated image")
    args = parser.parse_args()

    enroll_path = Path(args.enroll)
    scene_path  = Path(args.scene)
    output_dir  = Path(args.output)

    # ── validation ────────────────────────────────────────────────────────────
    for p in (enroll_path, scene_path):
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  Face Recognition Test")
    print("=" * 60)

    # ── check server ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Checking server at {args.server} ...")
    if not _check_server(args.server):
        print(f"[ERROR] Server not reachable at {args.server}.")
        print("  Start it with:  make run")
        print("           or:    python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    print("      ✅  Server is up.")

    # ── enroll ────────────────────────────────────────────────────────────────
    print(f"\n[2/4] Enrolling '{args.name}' from {enroll_path.name} ...")
    try:
        enroll_result = enroll_identity(args.server, args.name, enroll_path)
    except requests.HTTPError as exc:
        print(f"[ERROR] Enrollment failed: {exc}\n{exc.response.text}")
        sys.exit(1)

    identity_id = enroll_result.get("id", enroll_result.get("identity_id", "?"))
    embeddings_stored = enroll_result.get("embeddings_stored", enroll_result.get("num_embeddings", "?"))
    print(f"      ✅  Enrolled successfully.")
    print(f"         Identity ID       : {identity_id}")
    print(f"         Embeddings stored : {embeddings_stored}")

    # ── recognise ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Running recognition on {scene_path.name} ...")
    try:
        recognize_result = recognize_scene(args.server, scene_path)
    except requests.HTTPError as exc:
        print(f"[ERROR] Recognition failed: {exc}\n{exc.response.text}")
        sys.exit(1)

    faces = recognize_result.get("faces", [])
    print(f"      Faces detected in scene : {len(faces)}")

    # ── results summary ───────────────────────────────────────────────────────
    print(f"\n[4/4] Results:")
    print("-" * 60)

    target_found = False
    for i, face in enumerate(faces, 1):
        identity = face.get("identity", {})
        name     = identity.get("name", "Unknown")
        conf     = identity.get("confidence", identity.get("similarity", 0.0))
        matched  = identity.get("matched", False)
        bbox     = face.get("bounding_box", face.get("bbox", {}))

        conf_str = f"{conf:.3f}" if isinstance(conf, float) else str(conf)
        print(f"  Face #{i}")
        print(f"    Recognised as : {name}")
        print(f"    Matched       : {matched}")
        print(f"    Confidence    : {conf_str}")
        print(f"    Bounding box  : {bbox}")
        print()

        if name == args.name and matched:
            target_found = True

    print("-" * 60)
    if target_found:
        print(f"\n  ✅  '{args.name}' DETECTED in scene_1!\n")
    else:
        print(f"\n  ❌  '{args.name}' NOT detected in scene_1.")
        print("      Possible reasons:")
        print("        • The face in the scene is at a different angle or lighting")
        print("        • Similarity threshold may need lowering")
        print("          (RecognitionConfig.similarity_threshold in src/config.py)")
        print("        • Try enrolling multiple photos (front, left, right angles)\n")

    # ── annotate ──────────────────────────────────────────────────────────────
    if not args.no_annotate:
        print("  Saving annotated image ...")
        out = annotate_and_save(scene_path, recognize_result, output_dir)
        if out:
            print(f"  📸  Annotated image → {out}")

    print("=" * 60 + "\n")

    # Pretty-print raw JSON for debugging
    print("Raw recognition JSON:")
    print(json.dumps(recognize_result, indent=2))


if __name__ == "__main__":
    main()
