#!/usr/bin/env bash
# =============================================================================
# Face Recognition Test Runner — multi-identity, multi-scene
# Compatible with bash 3.2+ (macOS default)
# =============================================================================
#
# ENROLLMENT LAYOUT  (test_samples/custom_test/enrollment/)
#   Flat files  →  one photo per identity, filename = identity name
#     char_1.jpg  →  "char_1"
#
#   Subdirectories  →  multiple photos per identity, dirname = identity name
#     char_1/front.jpg
#     char_1/side.jpg  →  "char_1"
#
# SCENES LAYOUT  (test_samples/custom_test/scenes/)
#   Any .jpg / .jpeg / .png  →  tested for recognition
#
# USAGE
#   bash run_test.sh               # start server automatically
#   bash run_test.sh --skip-server # assume server already running on :8000
# =============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ENROLL_DIR="$BASE_DIR/test_samples/custom_test/enrollment"
SCENE_DIR="$BASE_DIR/test_samples/custom_test/scenes"
OUTPUT_DIR="$BASE_DIR/test_samples/custom_test/output"
SERVER_URL="http://localhost:8000"
LOG_FILE="$BASE_DIR/server.log"
SERVER_PID_FILE="/tmp/face_recognition_server.pid"
TMP_DIR="/tmp/fr_test_$$"
SKIP_SERVER=false

mkdir -p "$TMP_DIR" "$OUTPUT_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${NC}"; }

for arg in "$@"; do [[ "$arg" == "--skip-server" ]] && SKIP_SERVER=true; done

# =============================================================================
# Discover identities → write one manifest file per identity
# Manifest: $TMP_DIR/id_<name>  containing one image path per line
# =============================================================================
discover_identities() {
  # Flat files: enrollment/alice.jpg  →  identity "alice"
  for img in "$ENROLL_DIR"/*.jpg "$ENROLL_DIR"/*.jpeg "$ENROLL_DIR"/*.png; do
    [[ -f "$img" ]] || continue
    name="$(basename "${img%.*}")"
    echo "$img" >> "$TMP_DIR/id_${name}"
  done

  # Subdirectories: enrollment/alice/*.jpg  →  identity "alice"
  for dir in "$ENROLL_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    name="$(basename "$dir")"
    for img in "$dir"*.jpg "$dir"*.jpeg "$dir"*.png; do
      [[ -f "$img" ]] || continue
      echo "$img" >> "$TMP_DIR/id_${name}"
    done
  done
}

# =============================================================================
# Enroll one identity given its manifest file
# =============================================================================
enroll_identity() {
  local manifest="$1"
  local name
  name="$(basename "$manifest" | sed 's/^id_//')"

  info "Enrolling: '$name'"

  # Build curl args — one -F files=@... per photo
  local args=(-s -X POST "$SERVER_URL/enroll/" -F "name=$name")
  while IFS= read -r photo; do
    [[ -f "$photo" ]] || continue
    args+=(-F "files=@${photo}")
    echo "    ↳ $(basename "$photo")"
  done < "$manifest"

  local resp
  resp=$(curl "${args[@]}")
  echo "$resp" | python3 -m json.tool 2>/dev/null || echo "$resp"

  # Extract identity_id and quality for the summary
  local iid qual
  iid=$(echo "$resp" | python3 -c "
import sys,json
try:
  d=json.load(sys.stdin); print(d.get('identity_id',''))
except: print('')
" 2>/dev/null)
  qual=$(echo "$resp" | python3 -c "
import sys,json
try:
  d=json.load(sys.stdin); print(d.get('avg_enrollment_quality',0))
except: print(0)
" 2>/dev/null)

  if [[ -n "$iid" && "$iid" != "null" ]]; then
    success "Enrolled '$name'  id=$iid  avg_quality=$qual"
    echo "$name=$iid" >> "$TMP_DIR/enrolled.txt"
  else
    warn "Enrollment may have failed for '$name' — check response above"
  fi
}

# =============================================================================
# Run recognition on one scene; annotate and return matched names
# =============================================================================
recognise_scene() {
  local scene="$1"
  local scene_name
  scene_name="$(basename "$scene")"
  info "Recognising: $scene_name"

  local resp
  resp=$(curl -s -X POST "$SERVER_URL/recognize/" -F "file=@${scene}")
  echo "$resp" | python3 -m json.tool 2>/dev/null || echo "$resp"

  # Parse, print breakdown, annotate image, write matched names to temp file
  # Pass JSON via env var — avoids shell-escaping issues with large/complex JSON
  export _FR_RESP="$resp" _FR_SCENE="$scene" _FR_OUT="$OUTPUT_DIR" _FR_NAME="$scene_name"
  python3 - <<'PYEOF'
import json, os, sys

scene_path  = os.environ["_FR_SCENE"]
output_dir  = os.environ["_FR_OUT"]
scene_name  = os.environ["_FR_NAME"]
raw         = os.environ.get("_FR_RESP", "")

try:
    data = json.loads(raw)
except Exception as e:
    print("  [WARN] Could not parse recognition JSON:", e)
    open("/tmp/fr_matched", "w").close()
    sys.exit(0)

faces   = data.get("faces", [])
matched = [f["identity_name"] for f in faces if f.get("identity_id")]

print("")
print("  Detected : {} face(s)".format(len(faces)))
print("  Matched  : {}".format(", ".join(matched) if matched else "none"))

for i, face in enumerate(faces, 1):
    name    = face.get("identity_name", "Unknown")
    sim     = face.get("similarity", 0.0)
    iid     = face.get("identity_id")
    quality = face.get("quality") or {}
    tier    = quality.get("tier", "?")
    qscore  = quality.get("quality_score", 0)
    reasons = quality.get("rejection_reasons", [])
    top     = face.get("top_candidate")
    flag    = (" | ! " + "; ".join(reasons)) if reasons else ""
    top_str = ""
    if top and not iid:
        top_str = " | closest: {} ({:.3f})".format(top.get("name","?"), top.get("similarity",0))
    print("  Face #{}: {} | sim={:.3f} | tier={} | q={:.1f}{}{}".format(
        i, name, sim, tier, qscore, flag, top_str))

# Write matched names for the shell summary
with open("/tmp/fr_matched", "w") as f:
    f.write(",".join(matched))

# Annotate image
try:
    import cv2
    img = cv2.imread(scene_path)
    if img is None:
        raise ValueError("cv2.imread returned None for " + scene_path)

    os.makedirs(output_dir, exist_ok=True)

    for face in faces:
        name    = face.get("identity_name", "Unknown")
        sim     = face.get("similarity", 0.0)
        iid     = face.get("identity_id")
        quality = face.get("quality") or {}
        tier    = quality.get("tier", "?")
        qscore  = quality.get("quality_score", 0)
        bbox    = face.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = (0, 200, 0) if iid else (0, 0, 200)
        label = "{} {:.2f} [{}:{:.0f}]".format(name, sim, tier, qscore)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    stem     = os.path.splitext(scene_name)[0]
    out_path = os.path.join(output_dir, stem + "_annotated.jpg")
    cv2.imwrite(out_path, img)
    print("  Annotated image saved -> " + out_path)
except ImportError as e:
    print("  [WARN] Skipping annotation (missing package: {})".format(e))
except Exception as e:
    print("  [WARN] Annotation error: {}".format(e))
PYEOF

  # Return matched names via temp file
  local matched_names=""
  [[ -f /tmp/fr_matched ]] && matched_names="$(cat /tmp/fr_matched)" && rm -f /tmp/fr_matched
  echo "$scene_name|${matched_names}" >> "$TMP_DIR/scene_results.txt"
}

# =============================================================================
# Main
# =============================================================================
echo ""
echo "============================================================"
echo "  Face Recognition Test Runner"
echo "============================================================"
echo ""

# ── Discover ──────────────────────────────────────────────────────────────────
discover_identities

IDENTITY_COUNT=$(ls "$TMP_DIR"/id_* 2>/dev/null | wc -l | tr -d ' ')
if [[ "$IDENTITY_COUNT" -eq 0 ]]; then
  error "No enrollment images found in: $ENROLL_DIR"
  echo ""
  echo "  Flat file layout:     $ENROLL_DIR/char_1.jpg"
  echo "  Multi-photo layout:   $ENROLL_DIR/char_1/front.jpg"
  exit 1
fi

SCENE_COUNT=0
for s in "$SCENE_DIR"/*.jpg "$SCENE_DIR"/*.jpeg "$SCENE_DIR"/*.png; do
  [[ -f "$s" ]] && SCENE_COUNT=$((SCENE_COUNT + 1))
done
if [[ "$SCENE_COUNT" -eq 0 ]]; then
  error "No scene images found in: $SCENE_DIR"
  exit 1
fi

info "Identities to enroll : $IDENTITY_COUNT"
for m in "$TMP_DIR"/id_*; do
  n=$(basename "$m" | sed 's/^id_//'); c=$(wc -l < "$m" | tr -d ' ')
  echo "    • $n  ($c photo(s))"
done
info "Scenes to test        : $SCENE_COUNT"
for s in "$SCENE_DIR"/*.jpg "$SCENE_DIR"/*.jpeg "$SCENE_DIR"/*.png; do
  [[ -f "$s" ]] && echo "    • $(basename "$s")"
done
echo ""

# ── Start server ──────────────────────────────────────────────────────────────
if [[ "$SKIP_SERVER" == "false" ]]; then
  info "Starting Face Recognition API server..."
  if lsof -ti:8000 &>/dev/null; then
    warn "Port 8000 in use — killing existing process..."
    kill "$(lsof -ti:8000)" 2>/dev/null || true
    sleep 2
  fi
  cd "$BASE_DIR"
  python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
  echo $! > "$SERVER_PID_FILE"
  info "Server PID: $(cat "$SERVER_PID_FILE")  (logs → $LOG_FILE)"

  info "Waiting for server..."
  MAX_WAIT=90; ELAPSED=0
  until curl -sf "$SERVER_URL/health" &>/dev/null; do
    sleep 2; ELAPSED=$((ELAPSED + 2)); echo -n "."
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
      echo ""; error "Server did not start in ${MAX_WAIT}s. Check: $LOG_FILE"; exit 1
    fi
  done
  echo ""
  success "Server is up"
else
  if ! curl -sf "$SERVER_URL/health" &>/dev/null; then
    error "Server not reachable at $SERVER_URL"; exit 1
  fi
  success "Server is reachable (--skip-server)"
fi

echo ""
curl -s "$SERVER_URL/health" | python3 -m json.tool
echo ""

# ── Enroll ────────────────────────────────────────────────────────────────────
header "━━  STEP 1: ENROLLMENT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
touch "$TMP_DIR/enrolled.txt"
for manifest in "$TMP_DIR"/id_*; do
  echo ""
  enroll_identity "$manifest"
done

echo ""
info "Gallery after enrollment:"
curl -s "$SERVER_URL/identities/" | python3 -m json.tool
echo ""

# ── Recognise ─────────────────────────────────────────────────────────────────
header "━━  STEP 2: RECOGNITION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
touch "$TMP_DIR/scene_results.txt"
for scene in "$SCENE_DIR"/*.jpg "$SCENE_DIR"/*.jpeg "$SCENE_DIR"/*.png; do
  [[ -f "$scene" ]] || continue
  echo ""
  recognise_scene "$scene"
done

# ── Summary ───────────────────────────────────────────────────────────────────
header "━━  SUMMARY  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "  %-30s  %s\n" "Scene" "Detected identities"
printf "  %-30s  %s\n" "------------------------------" "-------------------"
while IFS='|' read -r scene_name matched; do
  [[ -z "$matched" ]] && matched="none"
  printf "  %-30s  %s\n" "$scene_name" "$matched"
done < "$TMP_DIR/scene_results.txt"

echo ""
# Per-identity detection report
while IFS='=' read -r name _iid; do
  found=false
  while IFS='|' read -r _scene matched; do
    case ",$matched," in *",$name,"*) found=true; break;; esac
  done < "$TMP_DIR/scene_results.txt"
  if $found; then
    success "'$name' detected in at least one scene ✓"
  else
    warn    "'$name' NOT detected in any scene"
  fi
done < "$TMP_DIR/enrolled.txt"

echo ""
info "Annotated images → $OUTPUT_DIR"
[[ -f "$SERVER_PID_FILE" ]] && info "Stop server: kill \$(cat $SERVER_PID_FILE)"
echo "============================================================"
echo ""
