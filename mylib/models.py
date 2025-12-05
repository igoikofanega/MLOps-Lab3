"""Model utilities for transfer learning."""

import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Load a pre-trained model and modify it for transfer learning.
    
    Args:
        model_name: Name of the model ('resnet18', 'efficientnet_b0', 'mobilenet_v2')
        num_classes: Number of output classes
        pretrained: Whether to load pre-trained weights
        
    Returns:
        torch.nn.Module: Modified model ready for transfer learning
    """
    model_name = model_name.lower()
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace the final layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace the classifier
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace the classifier
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Supported models: resnet18, efficientnet_b0, mobilenet_v2"
        )
    
    return model


def count_parameters(model):
    """Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params
