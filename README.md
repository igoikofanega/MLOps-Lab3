# MLOps-Lab1 - Image Processing Application

![CI Pipeline](https://github.com/igoikofanega/MLOps-Lab2/actions/workflows/CICD.yml/badge.svg)

## Description 

Image processing application developed for Lab1 of the MLOps course. The application allows basic image operations such as random class prediction, resizing, grayscale conversion, and retrieving image information.

## Project Structure

```
MLOps-Lab1/
├── .github/
│   └── workflows/
│       └── CICD.yml                  # CI/CD Pipeline (GitHub Actions)
├── mylib/
│   ├── __init__.py
│   └── operations.py               # Image processing logic
├── cli/
│   ├── __init__.py
│   └── cli.py                      # Command-line interface (Click)
├── api/
│   ├── __init__.py
│   └── api.py                      # FastAPI API
├── templates/
│   └── home.html                   # Home page template for the API
├── tests/
│   ├── __init__.py
│   ├── test_operations.py          # Unit tests for mylib
│   ├── test_cli.py                 # CLI integration tests
│   └── test_api.py                 # API integration tests
├── .gitignore
├── LICENSE                         # MIT License (o la que corresponda)
├── Makefile                        # Automated development commands
├── pyproject.toml                  # Project metadata & dependencies (PEP 621)
├── uv.lock                         # Lock file generado por uv (reproducible env)
└── README.md
```

## Installation

### Prerequisites
- Python 3.12 or higher
- uv (package and virtual environment manager)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/igoikofanega/MLOps-Lab1-demo.git
cd MLOps-Lab1
```

2. Create and activate the virtual environment:
```bash
uv init
uv sync
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
make install
```

## Usage

### Command Line Interface (CLI)

#### Predict class of an image:
```bash
uv run python -m cli.cli predict path/to/image.jpg
```

#### Resize an image:
```bash
uv run python -m cli.cli resize path/to/image.jpg 200 200 --output resized.jpg
```

#### Convert to grayscale:
```bash
uv run python -m cli.cli grayscale path/to/image.jpg --output gray.jpg
```

#### Get image information:
```bash
uv run python -m cli.cli info path/to/image.jpg
```

### API (FastAPI)

#### Start the server:
```bash
uv run python -m api.api
```

The API will be available at `http://localhost:8000`

#### Available Endpoints:

- **GET /** - Home page
- **POST /predict** - Predicts the class of an image  
  - Parameters: `file` (image)
- **POST /resize** - Resizes an image  
  - Parameters: `file` (image), `width` (int), `height` (int)
- **POST /grayscale** - Converts an image to grayscale  
  - Parameters: `file` (image)
- **POST /info** - Gets image information  
  - Parameters: `file` (image)

#### Interactive Documentation:
Once the server is running, visit `http://localhost:8000/docs` to access the interactive Swagger UI documentation.

## Development

### Makefile - Available Commands

```bash
make install    # Install all dependencies
make format     # Format code with Black
make lint       # Static analysis with Pylint
make test       # Run tests with pytest
make refactor   # Format + lint
make all        # Run everything (install + format + lint + test)
make clean      # Clean temporary files
```

### Testing

Run all tests:
```bash
make test
```

Run specific tests:
```bash
uv run pytest tests/test_operations.py -v
uv run pytest tests/test_cli.py -v
uv run pytest tests/test_api.py -v
```

View test coverage:
```bash
uv run pytest tests/ --cov=mylib --cov=cli --cov=api --cov-report=html
```

## CI/CD Pipeline

The project uses GitHub Actions to automatically run:

1. **Format**: Code formatting with Black
2. **Lint**: Static analysis with Pylint
3. **Test**: Test execution with pytest

The pipeline runs on every `push` and `pull_request` to the `main` branch.

## Prediction Classes

The available classes for random prediction are:
- dog
- cat
- car
- airplane
- ship
- bicycle
- person
- house

## Technologies Used

- **Python 3.12**
- **Pillow (PIL)**: Image processing
- **Click**: CLI creation
- **FastAPI**: API framework
- **Uvicorn**: ASGI server
- **Pytest**: Testing framework
- **Black**: Code formatting
- **Pylint**: Static analysis
- **uv**: Package and virtual environment manager

## Author

Iñigo Goioketxea - Public University of Navarre

## License

MIT License
