.PHONY: install test run lint docker-build docker-run clean generate-samples help

# Default target
help:
	@echo "Face Detection & Recognition System"
	@echo "===================================="
	@echo ""
	@echo "  make install           Install dependencies"
	@echo "  make test              Run test suite"
	@echo "  make test-verbose      Run tests with verbose output"
	@echo "  make run               Start the API server"
	@echo "  make generate-samples  Generate test sample images"
	@echo "  make docker-build      Build Docker image"
	@echo "  make docker-run        Run with Docker Compose"
	@echo "  make clean             Remove generated files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python -m pytest tests/ -v --tb=short

test-verbose:
	python -m pytest tests/ -v --tb=long -s

# Start the API server
run:
	python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Generate test samples
generate-samples:
	python test_samples/generate_samples.py

# Docker
docker-build:
	docker build -t face-recognition-api:latest -f docker/Dockerfile .

docker-run:
	cd docker && docker-compose up -d

docker-stop:
	cd docker && docker-compose down

# Clean up
clean:
	rm -rf __pycache__ .pytest_cache data/ logs/ models/
	rm -f face_recognition.db test_face_recognition.db test_api.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
