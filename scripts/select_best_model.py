"""Select the best model from MLFlow and export to ONNX format."""

import json
import os
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Configuration
MODEL_REGISTRY_NAME = "oxford-pet-classifier"
RESULTS_DIR = "results"
ONNX_MODEL_PATH = os.path.join(RESULTS_DIR, "model.onnx")
CLASS_LABELS_PATH = os.path.join(RESULTS_DIR, "class_labels.json")


def select_best_model():
    """Query registered models and select the best one based on validation accuracy.
    
    Returns:
        ModelVersion: Best model version object
    """
    client = MlflowClient()
    
    # Search for all versions of the registered model
    print(f"Searching for models with name: {MODEL_REGISTRY_NAME}")
    model_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    
    if not model_versions:
        raise ValueError(f"No models found with name '{MODEL_REGISTRY_NAME}'")
    
    print(f"Found {len(model_versions)} model version(s)")
    
    # Compare models by validation accuracy
    best_version = None
    best_accuracy = -1.0
    
    print("\nComparing models:")
    print("-" * 80)
    
    for version in model_versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        # Get validation accuracy
        val_accuracy = metrics.get("final_val_accuracy", metrics.get("best_val_accuracy", 0.0))
        
        print(f"Version {version.version}:")
        print(f"  Run ID: {run_id}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  Model: {run.data.params.get('model_name', 'unknown')}")
        print(f"  Batch Size: {run.data.params.get('batch_size', 'unknown')}")
        print(f"  Learning Rate: {run.data.params.get('learning_rate', 'unknown')}")
        print("-" * 80)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_version = version
    
    print(f"\nBest model: Version {best_version.version}")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    return best_version


def export_to_onnx(best_version):
    """Load the best model and export to ONNX format.
    
    Args:
        best_version: Best model version from MLFlow
    """
    client = MlflowClient()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load the model
    print("\nLoading model...")
    model_uri = f"runs:/{best_version.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Move to CPU (required for deployment on Render)
    model = model.to("cpu")
    model.eval()
    
    print("Model loaded and moved to CPU")
    
    # Export to ONNX
    print(f"Exporting to ONNX format: {ONNX_MODEL_PATH}")
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    print(f"Model exported to: {ONNX_MODEL_PATH}")
    
    # Download and save class labels
    print("Downloading class labels...")
    class_labels_artifact = client.download_artifacts(
        best_version.run_id, "class_labels.json"
    )
    
    # Read the downloaded file and save to results directory
    with open(class_labels_artifact, "r") as f:
        class_labels = json.load(f)
    
    with open(CLASS_LABELS_PATH, "w") as f:
        json.dump(class_labels, f, indent=2)
    
    print(f"Class labels saved to: {CLASS_LABELS_PATH}")
    print(f"Number of classes: {len(class_labels)}")
    
    return model, class_labels


def verify_onnx_export():
    """Verify that the ONNX model works correctly."""
    import onnxruntime as ort
    import numpy as np
    
    print("\nVerifying ONNX export...")
    
    # Load ONNX model
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    
    session = ort.InferenceSession(
        ONNX_MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: dummy_input})
    logits = outputs[0]
    
    print(f"ONNX model verification successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Predicted class: {logits.argmax()}")
    
    # Load class labels
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    
    predicted_class = class_labels[logits.argmax()]
    print(f"Predicted class name: {predicted_class}")


def main():
    """Main function to select best model and export to ONNX."""
    print("=" * 80)
    print("Model Selection and ONNX Export")
    print("=" * 80)
    
    # Select best model
    best_version = select_best_model()
    
    # Export to ONNX
    model, class_labels = export_to_onnx(best_version)
    
    # Verify export
    verify_onnx_export()
    
    print("\n" + "=" * 80)
    print("Export complete!")
    print("=" * 80)
    print(f"ONNX model: {ONNX_MODEL_PATH}")
    print(f"Class labels: {CLASS_LABELS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
