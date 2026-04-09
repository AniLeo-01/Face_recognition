# Custom Test: Enroll character_1 → Detect in scene_1

## Step 1 — Save the images

Save the two images you shared in the chat to these exact paths:

```
test_samples/custom_test/enrollment/character_1.jpg   ← the close-up portrait
test_samples/custom_test/scenes/scene_1.jpg           ← the hospital hallway scene
```

## Step 2 — Install dependencies (once)

```bash
pip install -r requirements.txt
```

## Step 3 — Run the server

```bash
# In one terminal:
make run
# or
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Step 4 — Run the test (new terminal)

```bash
python scripts/test_character.py \
    --enroll test_samples/custom_test/enrollment/character_1.jpg \
    --scene  test_samples/custom_test/scenes/scene_1.jpg
```

OR use the all-in-one shell script (starts server automatically):

```bash
bash run_test.sh
```

## Output

- Console: detection results + ✅/❌ match verdict
- `test_samples/custom_test/output/scene_1_annotated.jpg` — scene with bounding boxes
