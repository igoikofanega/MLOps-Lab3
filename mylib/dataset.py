"""Dataset utilities for Oxford-IIIT Pet dataset."""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# ImageNet normalization (standard for pre-trained models)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_data_loaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def get_inverse_normalize():
    """Get inverse normalization transform for visualization.
    
    Returns:
        transforms.Normalize: Inverse normalization transform
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
