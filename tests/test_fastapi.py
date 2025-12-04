"""Integration testing with the API."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.api import app
from mylib.operations import CLASS_NAMES


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI application.

    Returns:
        TestClient: FastAPI test client with isolated application context.
    """
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> io.BytesIO:
    """Generate a 100x100 red JPEG image as bytes for testing.

    Returns:
        io.BytesIO: In-memory JPEG image file-like object, seeked to start.
    """
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_home_endpoint(test_client: TestClient) -> None:
    """Test that the root endpoint returns the expected HTML page.

    Args:
        test_client: FastAPI test client fixture.
    """
    response = test_client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_endpoint_success(
    test_client: TestClient, sample_image_bytes: io.BytesIO
) -> None:
    """Test successful image classification via /predict endpoint.

    Args:
        client: FastAPI test client fixture.
        sample_image_bytes: Valid JPEG image as bytes.
    """
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = test_client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in CLASS_NAMES
    assert "filename" in data
    assert "image_info" in data


def test_predict_endpoint_no_file(test_client: TestClient) -> None:
    """Test /predict endpoint when no file is uploaded.

    Args:
        client: FastAPI test client fixture.
    """
    response = test_client.post("/predict")

    assert response.status_code == 422  # Unprocessable Entity (missing required field)


def test_predict_endpoint_invalid_file(test_client: TestClient) -> None:
    """Test /predict endpoint with a non-image file upload.

    Args:
        client: FastAPI test client fixture.
    """
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = test_client.post("/predict", files=files)

    assert response.status_code == 400


def test_resize_endpoint_success(
    test_client: TestClient, sample_image_bytes: io.BytesIO
) -> None:
    """Test successful image resizing via /resize endpoint.

    Args:
        test_client: FastAPI test client fixture.
        sample_image_bytes: Valid JPEG image as bytes.
    """
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "50", "height": "75"}

    response = test_client.post("/resize", files=files, data=data)

    assert response.status_code == 200
    assert "image/" in response.headers["content-type"]

    img = Image.open(io.BytesIO(response.content))
    assert img.size == (50, 75)


def test_resize_endpoint_no_file(test_client: TestClient) -> None:
    """Test /resize endpoint when no file is provided.

    Args:
        test_client: FastAPI test client fixture.
    """
    data = {"width": "50", "height": "75"}
    response = test_client.post("/resize", data=data)

    assert response.status_code == 422


def test_resize_endpoint_missing_dimensions(
    test_client: TestClient, sample_image_bytes: io.BytesIO
) -> None:
    """Test /resize endpoint when width or height is missing.

    Args:
        test_client: FastAPI test client fixture.
        sample_image_bytes: Valid JPEG image as bytes.
    """
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = test_client.post("/resize", files=files)

    assert response.status_code == 422


def test_resize_endpoint_invalid_dimensions(
    test_client: TestClient, sample_image_bytes: io.BytesIO
) -> None:
    """Test /resize endpoint with negative or invalid dimensions.

    Args:
        test_client: FastAPI test client fixture.
        sample_image_bytes: Valid JPEG image as bytes.
    """
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "-50", "height": "75"}

    response = test_client.post("/resize", files=files, data=data)

    assert response.status_code == 400


def test_resize_endpoint_zero_dimensions(
    test_client: TestClient, sample_image_bytes: io.BytesIO
) -> None:
    """Test /resize endpoint with zero as dimension value.

    Args:
        test_client: FastAPI test client fixture.
        sample_image_bytes: Valid JPEG image as bytes.
    """
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": "0", "height": "75"}

    response = test_client.post("/resize", files=files, data=data)

    assert response.status_code == 400


def test_grayscale_endpoint_success(
    test_client: TestClient, sample_image_bytes: io.BytesIO
) -> None:
    """Test successful grayscale conversion via /grayscale endpoint.

    Args:
        test_client: FastAPI test client fixture.
        sample_image_bytes: Valid JPEG image as bytes.
    """
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}

    response = test_client.post("/grayscale", files=files)

    assert response.status_code == 200
    assert "image/" in response.headers["content-type"]

    img = Image.open(io.BytesIO(response.content))
    assert img.mode == "L"


def test_grayscale_endpoint_no_file(test_client: TestClient) -> None:
    """Test /grayscale endpoint when no file is uploaded.

    Args:
        test_client: FastAPI test client fixture.
    """
    response = test_client.post("/grayscale")

    assert response.status_code == 422


def test_grayscale_endpoint_invalid_file(test_client: TestClient) -> None:
    """Test /grayscale endpoint with a non-image file.

    Args:
        test_client: FastAPI test client fixture.
    """
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = test_client.post("/grayscale", files=files)

    assert response.status_code == 400


def test_info_endpoint_success(
    test_client: TestClient, sample_image_bytes: io.BytesIO
) -> None:
    """Test successful image info retrieval via /info endpoint.

    Args:
        test_client: FastAPI test client fixture.
        sample_image_bytes: Valid JPEG image as bytes.
    """
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}

    response = test_client.post("/info", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "image_info" in data
    assert "size" in data["image_info"]
    assert "width" in data["image_info"]
    assert "height" in data["image_info"]


def test_info_endpoint_no_file(test_client: TestClient) -> None:
    """Test /info endpoint when no file is uploaded.

    Args:
        test_client: FastAPI test client fixture.
    """
    response = test_client.post("/info")

    assert response.status_code == 422


def test_info_endpoint_invalid_file(test_client: TestClient) -> None:
    """Test /info endpoint with a non-image file.

    Args:
        test_client: FastAPI test client fixture.
    """
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = test_client.post("/info", files=files)

    assert response.status_code == 400
