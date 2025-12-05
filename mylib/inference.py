"""ONNX inference wrapper for Oxford-IIIT Pet classification."""

import json
import os
import numpy as np
from PIL import Image
import onnxruntime as ort


class PetClassifier:
    """ONNX-based classifier for Oxford-IIIT Pet dataset."""
    
    def __init__(self, model_path="results/model.onnx", labels_path="results/class_labels.json"):
        """Initialize the classifier.
        
        Args:
            model_path: Path to the ONNX model file
            labels_path: Path to the class labels JSON file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Class labels file not found: {labels_path}")
        
        # Initialize ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        
        # Get input name
        self.input_name = self.session.get_inputs()[0].name
        
        # Load class labels
        with open(labels_path, "r") as f:
            self.class_labels = json.load(f)
        
        print(f"PetClassifier initialized with {len(self.class_labels)} classes")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input.
        
        Args:
            image: PIL Image
            
        Returns:
            np.ndarray: Preprocessed image ready for inference
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format (channels, height, width)
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Image.Image) -> str:
        """Predict the class of an image.
        
        Args:
            image: PIL Image
            
        Returns:
            str: Predicted class label
        """
        # Preprocess image
        input_data = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Get logits (first output)
        logits = outputs[0]
        
        # Get predicted class index
        predicted_idx = int(logits.argmax())
        
        # Return class label
        return self.class_labels[predicted_idx]


# Global classifier instance (initialized when module is imported)
# This will be None until the model files exist
try:
    classifier = PetClassifier()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("Classifier will use fallback prediction until model is available")
    classifier = None
