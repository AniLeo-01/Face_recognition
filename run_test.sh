#!/usr/bin/env bash
# =============================================================================
# Face Recognition Test: Enroll character_1 → Detect in scene_1
# =============================================================================
# Usage:
#   1. Place character_1.jpg in  test_samples/custom_test/enrollment/
#   2. Place scene_1.jpg        in  test_samples/custom_test/scenes/
#   3. Run:  bash run_test.sh
# =============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ENROLL_IMG="$BASE_DIR/test_samples/custom_test/enrollment/character_1.jpg"
SCENE_IMG="$BASE_DIR/test_samples/custom_test/scenes/scene_1.jpg"
SERVER_URL="http://localhost:8000"
OUTPUT_DIR="$BASE_DIR/test_samples/custom_test/output"
LOG_FILE="$BASE_DIR/server.log"
SERVER_PID_FILE="/tmp/face_recognition_server.pid"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Face Recognition Test Runner"
echo "============================================================"
echo ""

if [[ ! -f "$ENROLL_IMG" ]]; then
  error "Enrollment image not found: $ENROLL_IMG"
  error "Please save character_1.jpg to test_samples/custom_test/enrollment/"
  exit 1
fi

if [[ ! -f "$SCENE_IMG" ]]; then
  error "Scene image not found: $SCENE_IMG"
  error "Please save scene_1.jpg to test_samples/custom_test/scenes/"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
success "Images found."

# ── Start server ──────────────────────────────────────────────────────────────
info "Starting Face Recognition API server..."

# Kill any existing server on port 8000
if lsof -ti:8000 &>/dev/null; then
  warn "Port 8000 already in use — killing existing process..."
  kill "$(lsof -ti:8000)" 2>/dev/null || true
  sleep 2
fi

cd "$BASE_DIR"
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$SERVER_PID_FILE"
info "Server PID: $SERVER_PID  (logs → $LOG_FILE)"

# ── Wait for server to be ready ───────────────────────────────────────────────
info "Waiting for server to become healthy..."
MAX_WAIT=60
ELAPSED=0
until curl -sf "$SERVER_URL/health" &>/dev/null; do
  sleep 2
  ELAPSED=$((ELAPSED + 2))
  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    error "Server did not start within ${MAX_WAIT}s."
    error "Check logs: $LOG_FILE"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi
  echo -n "."
done
echo ""
success "Server is up at $SERVER_URL"

# ── Health check ──────────────────────────────────────────────────────────────
echo ""
info "Health check:"
curl -s "$SERVER_URL/health" | python3 -m json.tool
echo ""

# ── Step 1: Enroll character_1 ────────────────────────────────────────────────
echo "------------------------------------------------------------"
info "STEP 1: Enrolling character_1 from $ENROLL_IMG"
echo "------------------------------------------------------------"

ENROLL_RESPONSE=$(curl -s -X POST "$SERVER_URL/enroll/" \
  -F "name=character_1" \
  -F "files=@${ENROLL_IMG}")

echo "$ENROLL_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$ENROLL_RESPONSE"

IDENTITY_ID=$(echo "$ENROLL_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('id', data.get('identity_id', 'unknown')))
except:
    print('unknown')
")

if [[ "$IDENTITY_ID" == "unknown" || -z "$IDENTITY_ID" ]]; then
  warn "Could not extract identity ID from enrollment response."
  warn "Check if enrollment succeeded."
else
  success "character_1 enrolled successfully! Identity ID: $IDENTITY_ID"
fi

# ── Step 2: List enrolled identities ─────────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
info "STEP 2: Listing enrolled identities"
echo "------------------------------------------------------------"
curl -s "$SERVER_URL/identities/" | python3 -m json.tool
echo ""

# ── Step 3: Run recognition on scene_1 ───────────────────────────────────────
echo "------------------------------------------------------------"
info "STEP 3: Running face recognition on scene_1 ($SCENE_IMG)"
echo "------------------------------------------------------------"

RECOGNIZE_RESPONSE=$(curl -s -X POST "$SERVER_URL/recognize/" \
  -F "file=@${SCENE_IMG}")

echo "$RECOGNIZE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RECOGNIZE_RESPONSE"

# ── Parse and summarise results ───────────────────────────────────────────────
# (Re-use $RECOGNIZE_RESPONSE already captured above — no extra curl call)
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"

export RECOGNIZE_RESPONSE SCENE_IMG SERVER_URL

python3 - "$SCENE_IMG" "$SERVER_URL" <<PYEOF
import sys, json, os

raw = os.environ.get("RECOGNIZE_RESPONSE", "")
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("Could not parse recognition response.")
    sys.exit(0)

faces = data.get("faces", [])
print("Faces detected in scene_1: {}".format(len(faces)))
print("")

found = False
for i, face in enumerate(faces, 1):
    name      = face.get("identity_name", "Unknown")
    similarity = face.get("similarity", 0.0)
    identity_id = face.get("identity_id")
    quality   = face.get("quality", {})
    bbox      = face.get("bbox", [])
    matched   = identity_id is not None

    print("  Face #{}:".format(i))
    print("    Name        : {}".format(name))
    print("    Matched     : {}".format(matched))
    print("    Similarity  : {:.3f}".format(similarity) if isinstance(similarity, float) else "    Similarity  : {}".format(similarity))
    print("    Quality OK  : {}".format(quality.get("passed", "?")))
    if not quality.get("passed", True):
        for reason in quality.get("rejection_reasons", []):
            print("      Rejected  : {}".format(reason))
    print("    Bounding box: {}".format(bbox))
    print("")

    if matched and name == "character_1":
        found = True

if found:
    print("character_1 DETECTED in scene_1!")
else:
    print("character_1 NOT detected in scene_1.")
    all_unknown = all(f.get("identity_id") is None for f in faces)
    any_quality_fail = any(not f.get("quality", {}).get("passed", True) for f in faces)
    if any_quality_fail:
        print("  -> Quality gate is still rejecting faces.")
        print("     Try: QUALITY_MAX_PITCH=90 QUALITY_MIN_BLUR=20 bash run_test.sh")
    elif all_unknown and faces:
        print("  -> Faces passed quality but similarity score is below threshold.")
        print("     Try lowering RecognitionConfig.similarity_threshold in src/config.py")
PYEOF

# ── Save annotated output ─────────────────────────────────────────────────────
echo ""
info "Saving annotated result..."

# Exports MUST come BEFORE the heredoc that uses them as env vars
export SCENE_IMG OUTPUT_DIR RECOGNIZE_RESPONSE

python3 - <<PYEOF
import json, os, sys

try:
    import cv2
    import numpy as np

    scene_img  = os.environ["SCENE_IMG"]
    output_dir = os.environ["OUTPUT_DIR"]
    raw        = os.environ.get("RECOGNIZE_RESPONSE", "")

    data  = json.loads(raw) if raw else {}
    faces = data.get("faces", [])

    img = cv2.imread(scene_img)
    if img is None:
        print("Could not load scene image for annotation.")
        sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)

    for face in faces:
        name        = face.get("identity_name", "Unknown")
        similarity  = face.get("similarity", 0.0)
        identity_id = face.get("identity_id")
        quality     = face.get("quality", {})
        bbox        = face.get("bbox", [])
        matched     = identity_id is not None

        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        color = (0, 200, 0) if matched else (0, 0, 200)
        tier  = quality.get("tier", "?")
        score = quality.get("quality_score", 0)
        label = "{} {:.2f} [{}:{:.0f}]".format(name, similarity, tier, score)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(output_dir, "scene_1_annotated.jpg")
    cv2.imwrite(out_path, img)
    print("Annotated image saved -> " + out_path)

except ImportError as e:
    print("Skipping annotation (missing package: {})".format(e))
except Exception as e:
    print("Annotation error: {}".format(e))
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
success "Test complete! Check output in: $OUTPUT_DIR"
echo ""
info "To stop the server: kill \$(cat $SERVER_PID_FILE)"
info "Server logs:        $LOG_FILE"
echo "============================================================"
echo ""
