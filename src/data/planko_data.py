#!/usr/bin/env python3
"""
Planko dataset loader for binary classification (left vs right).

Dataset structure:
  singbaskv3/
    train/
      {num}_left_{id}.png  (label 0)
      {num}_right_{id}.png (label 1)
    test/
      {num}_left_{id}.png  (label 0)
      {num}_right_{id}.png (label 1)

Features:
- Binary classification task (left=0, right=1)
- RGB images (256x256)
- Pre-split into train and test sets
- ~75,000 training images, ~5,000 test images
- Balanced classes (~50/50)
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np


class PlankoDataset(Dataset):
    """Planko dataset for binary classification"""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: Path to train or test folder containing images
            transform: Optional transform to be applied on images
        """
        self.data_path = Path(data_path)
        self.transform = transform

        # Get all PNG files
        self.file_list = sorted([f for f in self.data_path.glob("*.png")])

        if len(self.file_list) == 0:
            raise ValueError(f"No PNG files found in {data_path}")

        # Extract labels from filenames
        self.labels = []
        for file_path in self.file_list:
            # Label 0 if "left" in filename, else 1
            label = 0 if "left" in file_path.name else 1
            self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """Load image and label"""
        img_path = self.file_list[idx]
        label = self.labels[idx]

        # Load image (already RGB)
        img = Image.open(img_path).convert('RGB')

        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    

class PlankoDatasetWithPath(Dataset):
    """
    CABC Dataset that also returns image paths (useful for tracking predictions).

    Args:
        data_path: Path to CABC dataset split
        transform: Optional transform to apply to images
        convert_to_rgb: Whether to convert grayscale to RGB (default: True)
    """
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: Path to train or test folder containing images
            transform: Optional transform to be applied on images
        """
        self.data_path = Path(data_path)
        self.transform = transform

        # Get all PNG files
        self.file_list = sorted([f for f in self.data_path.glob("*.png")])

        if len(self.file_list) == 0:
            raise ValueError(f"No PNG files found in {data_path}")

        # Extract labels from filenames
        self.labels = []
        for file_path in self.file_list:
            # Label 0 if "left" in filename, else 1
            label = 0 if "left" in file_path.name else 1
            self.labels.append(label)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Load image and label"""
        img_path = self.file_list[idx]
        label = self.labels[idx]

        # Load image (already RGB)
        img = Image.open(img_path).convert('RGB')

        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path


def create_train_val_split(dataset, train_ratio=0.9, val_ratio=0.1, seed=42):
    """
    Split dataset into train and validation subsets

    Args:
        dataset: Full dataset to split
        train_ratio: Fraction of data for training (default: 0.9)
        val_ratio: Fraction of data for validation (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset
    """
    assert train_ratio + val_ratio == 1.0, f"train_ratio + val_ratio must equal 1.0, got {train_ratio + val_ratio}"

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get dataset size
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Shuffle indices
    np.random.shuffle(indices)

    # Calculate split point
    train_size = int(train_ratio * dataset_size)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def get_planko_datasets(planko_root, transform=None, train_ratio=0.9, val_ratio=0.1):
    """
    Create train, val, and test datasets for Planko.

    Args:
        planko_root: Root directory containing train/ and test/ folders
        transform: Transform to apply to images
        train_ratio: Fraction of training data to use for training (default: 0.9)
        val_ratio: Fraction of training data to use for validation (default: 0.1)

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    planko_root = Path(planko_root)

    # Check if train and test directories exist
    train_path = planko_root / 'train'
    test_path = planko_root / 'test'

    if not train_path.exists():
        raise ValueError(f"Train directory not found: {train_path}")
    if not test_path.exists():
        raise ValueError(f"Test directory not found: {test_path}")

    # Load full training dataset
    full_train_dataset = PlankoDataset(train_path, transform=transform)

    # Split into train and validation
    train_dataset, val_dataset = create_train_val_split(
        full_train_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    # Load test dataset
    test_dataset = PlankoDataset(test_path, transform=transform)

    # Print dataset info
    print(f"Planko dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '.')
    from cabc_utils import get_transform_wo_crop
    import timm

    # Create a simple transform
    model = timm.create_model('resnet50.a1_in1k', pretrained=False)
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)

    # Test dataset loading
    planko_root = "/media/data_cifs_lrs/projects/prj_vis_sim/plankdatasets/singbaskv3"

    train_ds, val_ds, test_ds = get_planko_datasets(
        planko_root=planko_root,
        transform=transform
    )

    # Test a sample
    img, label = train_ds[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label}")
    print(f"\nLabel distribution (first 1000 train samples):")
    labels = [train_ds[i][1] for i in range(min(1000, len(train_ds)))]
    unique, counts = np.unique(labels, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Label {val}: {count} ({count/len(labels)*100:.1f}%)")
