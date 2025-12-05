"""Download and prepare the Oxford-IIIT Pet dataset for training."""

import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Dataset configuration
DATA_DIR = "data"
TRAIN_SPLIT = 0.8
IMAGE_SIZE = 224

# ImageNet normalization (standard for pre-trained models)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_transforms():
    """Get data transformations for training and validation.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )

    return train_transform, val_transform


def prepare_dataset():
    """Download and prepare the Oxford-IIIT Pet dataset.
    
    Returns:
        tuple: (train_dataset, val_dataset, num_classes, class_names)
    """
    print("Downloading Oxford-IIIT Pet dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download the full dataset first (with default transform to get class info)
    full_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=True, transform=None
    )

    # Get class information
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    print(f"Dataset downloaded successfully!")
    print(f"Total images: {len(full_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names[:5]}... (showing first 5)")

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = total_size - train_size

    print(f"\nSplitting dataset:")
    print(f"  Training set: {train_size} images ({TRAIN_SPLIT * 100:.0f}%)")
    print(f"  Validation set: {val_size} images ({(1 - TRAIN_SPLIT) * 100:.0f}%)")

    # Split the dataset
    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = random_split(
        range(total_size), [train_size, val_size], generator=generator
    )

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets with proper transforms
    train_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=False, transform=train_transform
    )
    val_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=False, transform=val_transform
    )

    # Apply the indices to create subsets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    print("\nDataset preparation complete!")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Normalization: ImageNet stats")
    print(f"Random seed: {SEED}")

    return train_dataset, val_dataset, num_classes, class_names


if __name__ == "__main__":
    train_ds, val_ds, n_classes, classes = prepare_dataset()
    
    print("\n" + "=" * 50)
    print("Dataset Summary:")
    print("=" * 50)
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Number of classes: {n_classes}")
    print(f"All classes: {classes}")
    print("=" * 50)
