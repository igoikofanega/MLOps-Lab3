"""Image processing operations for MLOps Lab3."""

import random
from PIL import Image
from typing import Tuple

# Import ONNX classifier
try:
    from mylib.inference import classifier
    CLASSIFIER_AVAILABLE = classifier is not None
except ImportError:
    CLASSIFIER_AVAILABLE = False
    classifier = None

# Fallback class names for when model is not available
FALLBACK_CLASS_NAMES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]


def predict_class(image: Image.Image) -> str:
    """
    Predict the class of an image using the ONNX model.
    Falls back to random prediction if model is not available.

    Args:
        image: PIL Image object

    Returns:
        str: Predicted class name
    """
    if CLASSIFIER_AVAILABLE:
        try:
            return classifier.predict(image)
        except Exception as e:
            print(f"Warning: Prediction failed with error: {e}")
            print("Falling back to random prediction")
            return random.choice(FALLBACK_CLASS_NAMES)
    else:
        # Fallback to random prediction if model not available
        return random.choice(FALLBACK_CLASS_NAMES)


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
