"""Image processing operations for MLOps Lab1."""

import random
from PIL import Image
from typing import Tuple


# Define class names for random prediction
CLASS_NAMES = [
    "perro",
    "gato",
    "coche",
    "avión",
    "barco",
    "bicicleta",
    "persona",
    "casa",
]


def predict_class(image: Image.Image) -> str:
    """
    Predict the class of an image (randomly chosen for Lab1).

    Args:
        image: PIL Image object

    Returns:
        str: Randomly selected class name
    """
    return random.choice(CLASS_NAMES)


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Resize an image to specified dimensions.

    Args:
        image: PIL Image object
        width: Target width in pixels
        height: Target height in pixels

    Returns:
        PIL Image: Resized image
    """
    return image.resize((width, height))


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert an image to grayscale.

    Args:
        image: PIL Image object

    Returns:
        PIL Image: Grayscale image
    """
    return image.convert("L")


def get_image_info(image: Image.Image) -> dict:
    """
    Get basic information about an image.

    Args:
        image: PIL Image object

    Returns:
        dict: Dictionary containing image info (size, mode, format)
    """
    return {
        "size": image.size,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
    }
