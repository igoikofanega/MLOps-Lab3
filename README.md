# MLOps Lab3 - Transfer Learning with MLFlow

![CI Pipeline](https://github.com/igoikofanega/MLOps-Lab3/actions/workflows/CICD.yml/badge.svg)

Complete implementation of transfer learning on Oxford-IIIT Pet dataset with MLFlow experiment tracking, ONNX model serialization, and production deployment.

## Quick Start

### 1. Setup Environment

```bash
# Environment is already configured with Python 3.11
# Dependencies are installed in .venv
```

### 2. Download Dataset

```bash
uv run python scripts/prepare_dataset.py
```

### 3. Train Models

```bash
# Train ResNet18
uv run python scripts/train.py --model resnet18 --epochs 10

# Train EfficientNet-B0
uv run python scripts/train.py --model efficientnet_b0 --epochs 10

# Train with custom hyperparameters
uv run python scripts/train.py --model resnet18 --batch-size 64 --lr 0.01 --epochs 15
```

### 4. View MLFlow UI

```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Select Best Model and Export to ONNX

```bash
uv run python scripts/select_best_model.py
```

This creates:
- `results/model.onnx` - ONNX model
- `results/class_labels.json` - Class labels

### 6. Test API Locally

```bash
uv run uvicorn api.api:app --reload
# Open http://localhost:8000
```

### 7. Run Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run with coverage
uv run python -m pytest tests/ -vv --cov=mylib --cov=api --cov=cli
```

### 8. Deploy with Docker

```bash
# Build image
docker build -t mlops-lab3 .

# Run container
docker run -p 8000:8000 mlops-lab3
```

## Project Structure

```
├── api/                    # FastAPI application
├── cli/                    # Command-line interface
├── mylib/                  # Core library
│   ├── models.py          # Model utilities
│   ├── dataset.py         # Dataset utilities
│   ├── inference.py       # ONNX inference wrapper
│   └── operations.py      # Image processing operations
├── scripts/               # Training and utility scripts
│   ├── prepare_dataset.py # Dataset download and preparation
│   ├── train.py          # Training with MLFlow
│   └── select_best_model.py # Model selection and ONNX export
├── tests/                 # Test suite
├── results/              # Model artifacts (created after training)
├── data/                 # Dataset (created after download)
├── mlruns/              # MLFlow tracking data
└── plots/               # Training curves

```

## CI/CD Pipeline
The project uses GitHub Actions for automated testing and deployment:

- **Build**: Run tests, linting, and formatting checks.
- **Deploy**: Build and push Docker image to DockerHub, trigger Render deployment.
- **Deploy-HF**: Deploy Gradio frontend to HuggingFace Space.

## Deployment

The project uses a modern cloud deployment stack for production:

### DockerHub
Docker images are automatically built and pushed to DockerHub on every push to `main`.

### Render
The FastAPI backend is deployed on Render, providing a production-ready API with automatic HTTPS, health checks, and auto-deploy from DockerHub.

### Hugging Face Spaces
The Gradio UI is hosted on Hugging Face Spaces, connecting to the Render API to provide an interactive demo for end users.
