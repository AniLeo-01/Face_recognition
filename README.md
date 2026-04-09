# Face Detection & Recognition System

Production-grade face detection and recognition system capable of detecting, tracking, and classifying multiple human faces from images and video frames in real-time.

## Architecture

```
Video/Image → Face Detection (MTCNN) → Face Alignment (Affine Transform)
    → Quality Gate (Blur/Pose/Size) → Embedding (InceptionResNetV1/VGGFace2, 512-d)
    → Gallery Matching (FAISS ANN) → Identity Classification → API Response
```

### Core Modules

| Module | Description | Technology |
|--------|-------------|------------|
| **Detection** | Multi-face detection with bounding boxes and landmarks | MTCNN (facenet-pytorch) |
| **Alignment** | Landmark-based affine normalization to canonical face crops | OpenCV |
| **Quality** | Blur detection, pose estimation, size validation | Laplacian variance, geometric |
| **Embedding** | 512-d face embedding generation | InceptionResNetV1 (VGGFace2) |
| **Matching** | Approximate nearest neighbor gallery search | FAISS (Inner Product) |
| **API** | RESTful API with Swagger documentation | FastAPI |
| **Database** | Identity management and audit logging | SQLAlchemy + SQLite/PostgreSQL |

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/anileo-01/face_recognition.git
cd face_recognition
pip install -r requirements.txt
```

### Run the API Server

```bash
# Start the server
make run
# or
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### Run the Demo

```bash
# Generate test samples
make generate-samples

# Run the pipeline demo
python scripts/demo.py
```

### Run Tests

```bash
make test
# or
python -m pytest tests/ -v
```

### Docker

```bash
# Build and run
make docker-build
make docker-run

# Or with docker-compose
cd docker && docker-compose up -d
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/config` | Current system configuration |
| `POST` | `/detect/` | Detect faces in an image |
| `POST` | `/recognize/` | Full recognition pipeline |
| `POST` | `/enroll/` | Enroll a new identity |
| `GET` | `/identities/` | List registered identities |
| `GET` | `/identities/{id}` | Get identity details |
| `PATCH` | `/identities/{id}` | Update identity |
| `DELETE` | `/identities/{id}` | Delete identity |
| `GET` | `/events` | Recognition audit log |

### Example: Enroll a Person

```bash
curl -X POST http://localhost:8000/enroll/ \
  -F "name=John Doe" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg"
```

### Example: Recognize Faces

```bash
curl -X POST http://localhost:8000/recognize/ \
  -F "file=@scene.jpg"
```

## Project Structure

```
├── src/
│   ├── detection/        # MTCNN face detector
│   ├── alignment/        # Face alignment & quality assessment
│   ├── recognition/      # Embedding generation & FAISS matching
│   ├── pipeline/         # End-to-end orchestration
│   ├── api/              # FastAPI REST application
│   │   └── routes/       # API route handlers
│   ├── db/               # Database models & repository
│   └── config.py         # System configuration
├── tests/                # Comprehensive test suite
├── test_samples/         # Generated test images
├── scripts/              # Demo and utility scripts
├── docker/               # Docker & Compose configuration
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## Configuration

Configuration is managed via `src/config.py` with environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Enable GPU acceleration |
| `DATABASE_URL` | `sqlite+aiosqlite:///./face_recognition.db` | Database connection string |

Key tunable parameters:
- **Detection threshold**: `DetectionConfig.confidence_threshold` (default: 0.90)
- **Recognition threshold**: `RecognitionConfig.similarity_threshold` (default: 0.65)
- **Quality gate**: blur score, face size, pose angle limits

## Performance

| Metric | CPU (i7) | GPU (T4) |
|--------|----------|----------|
| Detection | ~80ms/frame | ~15ms/frame |
| Alignment | ~2ms/face | ~2ms/face |
| Embedding | ~40ms/face | ~5ms/face |
| Gallery search (10K) | ~3ms | ~3ms |
| End-to-end (1 face) | ~130ms | ~25ms |

## Technology Stack

- **Python 3.10+**
- **PyTorch** — Deep learning framework
- **facenet-pytorch** — MTCNN detection + InceptionResNetV1 embeddings
- **FAISS** — Approximate nearest neighbor search
- **FastAPI** — Async REST API framework
- **SQLAlchemy** — Async ORM with SQLite/PostgreSQL
- **OpenCV** — Image processing
- **Docker** — Containerized deployment

## License

MIT
