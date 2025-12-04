"""Tests for image processing operations."""

import pytest
from PIL import Image

from mylib.operations import (
    predict_class,
    resize_image,
    convert_to_grayscale,
    get_image_info,
    CLASS_NAMES,
)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a 100x100 red RGB image for testing.

    Returns:
        Image.Image: A new PIL Image in RGB mode.
    """
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Create a 100x100 grayscale image (intensity 128) for testing.

    Returns:
        Image.Image: A new PIL Image in L mode.
    """
    return Image.new("L", (100, 100), color=128)


def test_predict_class_returns_valid_class(sample_image: Image.Image) -> None:
    """Test that predict_class returns a class name present in CLASS_NAMES.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    predicted = predict_class(sample_image)
    assert predicted in CLASS_NAMES


def test_predict_class_returns_string(sample_image: Image.Image) -> None:
    """Test that predict_class always returns a string.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    predicted = predict_class(sample_image)
    assert isinstance(predicted, str)


def test_resize_image_correct_dimensions(sample_image: Image.Image) -> None:
    """Test that resize_image produces an image with the requested dimensions.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    resized = resize_image(sample_image, 50, 75)
    assert resized.size == (50, 75)


def test_resize_image_maintains_mode(sample_image: Image.Image) -> None:
    """Test that resize_image preserves the original image mode.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    original_mode = sample_image.mode
    resized = resize_image(sample_image, 50, 50)
    assert resized.mode == original_mode


def test_resize_image_larger_dimensions(sample_image: Image.Image) -> None:
    """Test resizing an image to larger dimensions (upsampling).

    Args:
        sample_image: A valid RGB PIL Image.
    """
    resized = resize_image(sample_image, 200, 200)
    assert resized.size == (200, 200)


def test_convert_to_grayscale_mode(sample_image: Image.Image) -> None:
    """Test that convert_to_grayscale returns an image in grayscale ('L') mode.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    gray = convert_to_grayscale(sample_image)
    assert gray.mode == "L"


def test_convert_to_grayscale_dimensions(sample_image: Image.Image) -> None:
    """Test that grayscale conversion does not alter image dimensions.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    original_size = sample_image.size
    gray = convert_to_grayscale(sample_image)
    assert gray.size == original_size


def test_convert_to_grayscale_already_gray(sample_grayscale_image: Image.Image) -> None:
    """Test converting an image that is already in grayscale mode.

    Args:
        sample_grayscale_image: A valid grayscale PIL Image.
    """
    gray = convert_to_grayscale(sample_grayscale_image)
    assert gray.mode == "L"
    assert gray.size == sample_grayscale_image.size


def test_get_image_info_contains_required_fields(sample_image: Image.Image) -> None:
    """Test that get_image_info returns a dict with all expected keys.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    info = get_image_info(sample_image)
    expected_keys = {"size", "width", "height", "mode"}
    assert expected_keys.issubset(info.keys())


def test_get_image_info_correct_values(sample_image: Image.Image) -> None:
    """Test that get_image_info returns accurate metadata for the image.

    Args:
        sample_image: A 100x100 RGB PIL Image.
    """
    info = get_image_info(sample_image)
    assert info["size"] == (100, 100)
    assert info["width"] == 100
    assert info["height"] == 100
    assert info["mode"] == "RGB"


def test_get_image_info_returns_dict(sample_image: Image.Image) -> None:
    """Test that get_image_info always returns a dictionary.

    Args:
        sample_image: A valid RGB PIL Image.
    """
    info = get_image_info(sample_image)
    assert isinstance(info, dict)
    