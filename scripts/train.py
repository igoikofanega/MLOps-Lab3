"""Train a transfer learning model on Oxford-IIIT Pet dataset with MLFlow tracking."""

import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mylib.models import get_model, count_parameters
from mylib.dataset import get_data_loaders
from scripts.prepare_dataset import get_transforms, SEED, DATA_DIR, TRAIN_SPLIT

# MLFlow configuration
EXPERIMENT_NAME = "oxford-pet-transfer-learning"
MODEL_REGISTRY_NAME = "oxford-pet-classifier"


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch.
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model.
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def train(args):
    """Main training function with MLFlow tracking."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_transform, val_transform = get_transforms()
    
    full_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=True, transform=None
    )
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size], generator=generator
    )
    
    # Create datasets with transforms
    train_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=False, transform=train_transform
    )
    val_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=False, transform=val_transform
    )
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        train_dataset, val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_classes, pretrained=True)
    model = model.to(device)
    
    trainable_params, total_params = count_parameters(model)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # MLFlow tracking
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    run_name = f"{args.model}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": args.model,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "seed": args.seed,
                "dataset": "OxfordIIITPet",
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "num_classes": num_classes,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "image_size": 224,
            }
        )
        
        # Training loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0
        
        print("\nStarting training...")
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Log metrics per epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )
            
            print(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Log final metrics
        mlflow.log_metrics(
            {
                "final_train_accuracy": train_accs[-1],
                "final_val_accuracy": val_accs[-1],
                "best_val_accuracy": best_val_acc,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
            }
        )
        
        # Plot and log training curves
        os.makedirs("plots", exist_ok=True)
        plot_path = f"plots/{run_name}_curves.png"
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
        mlflow.log_artifact(plot_path)
        
        # Save and log class labels
        class_labels_path = "class_labels.json"
        with open(class_labels_path, "w") as f:
            json.dump(class_names, f, indent=2)
        mlflow.log_artifact(class_labels_path)
        
        # Register model
        mlflow.pytorch.log_model(model, "model", registered_model_name=MODEL_REGISTRY_NAME)
        
        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model registered as: {MODEL_REGISTRY_NAME}")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train transfer learning model")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0", "mobilenet_v2"],
        help="Model architecture",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
