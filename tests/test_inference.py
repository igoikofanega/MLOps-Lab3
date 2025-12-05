"""Tests for ONNX inference."""

import os
import pytest
from PIL import Image

from mylib.inference import PetClassifier


def test_model_file_exists():
    """Test that the ONNX model file exists."""
    model_path = "results/model.onnx"
    # This test will fail until we train and export a model
    # That's expected - it ensures we don't deploy without a model
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}. Train a model first.")
    assert os.path.exists(model_path)


def test_class_labels_file_exists():
    """Test that the class labels JSON file exists."""
    labels_path = "results/class_labels.json"
    if not os.path.exists(labels_path):
        pytest.skip(f"Class labels file not found: {labels_path}. Train a model first.")
    assert os.path.exists(labels_path)


def test_classifier_initialization():
    """Test that the classifier can be initialized."""
    model_path = "results/model.onnx"
    labels_path = "results/class_labels.json"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        pytest.skip("Model files not found. Train a model first.")
    
    classifier = PetClassifier(model_path, labels_path)
    assert classifier is not None
    assert len(classifier.class_labels) > 0


def test_preprocessing():
    """Test image preprocessing."""
    model_path = "results/model.onnx"
    labels_path = "results/class_labels.json"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        pytest.skip("Model files not found. Train a model first.")
    
    classifier = PetClassifier(model_path, labels_path)
    
    # Create a test image
    test_image = Image.new("RGB", (100, 100), color="red")
    
    # Preprocess
    preprocessed = classifier.preprocess(test_image)
    
    # Check shape: (1, 3, 224, 224)
    assert preprocessed.shape == (1, 3, 224, 224)
    assert preprocessed.dtype.name == "float32"


def test_prediction_returns_string():
    """Test that prediction returns a string."""
    model_path = "results/model.onnx"
    labels_path = "results/class_labels.json"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        pytest.skip("Model files not found. Train a model first.")
    
    classifier = PetClassifier(model_path, labels_path)
    
    # Create a test image
    test_image = Image.new("RGB", (224, 224), color="blue")
    
    # Predict
    prediction = classifier.predict(test_image)
    
    assert isinstance(prediction, str)
    assert len(prediction) > 0


def test_prediction_returns_valid_class():
    """Test that prediction returns a class from the class labels."""
    model_path = "results/model.onnx"
    labels_path = "results/class_labels.json"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        pytest.skip("Model files not found. Train a model first.")
    
    classifier = PetClassifier(model_path, labels_path)
    
    # Create a test image
    test_image = Image.new("RGB", (300, 300), color="green")
    
    # Predict
    prediction = classifier.predict(test_image)
    
    assert prediction in classifier.class_labels
