"""
CABC Dataset Loader

Handles loading CABC dataset which has the following characteristics:
- Grayscale images (Mode: L) that need conversion to RGB for timm models
- Binary classification (positive/negative classes)
- Three difficulty levels: easy, medium, hard
- ImageFolder structure: /{difficulty}/images/{train|test}/{pos|neg}/{category}/sample_*.png
"""

import torch
from torch.utils.data import Dataset, Subset
import torchvision.datasets as datasets
from PIL import Image
import os
import numpy as np


class CABCDataset(Dataset):
    """
    CABC Dataset wrapper that converts grayscale images to RGB for timm compatibility.

    Args:
        data_path: Path to CABC dataset split (e.g., /path/to/cabc/easy/images/train)
        transform: Optional transform to apply to images
        convert_to_rgb: Whether to convert grayscale to RGB (default: True)
    """
    def __init__(self, data_path, transform=None, convert_to_rgb=True):
        self.image_folder = datasets.ImageFolder(data_path, transform=None)
        self.transform = transform
        self.convert_to_rgb = convert_to_rgb

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.imgs[idx]

        # Load image
        image = Image.open(path)

        # Convert grayscale to RGB if needed
        # CABC images are Mode 'L' (grayscale), timm models expect RGB
        if self.convert_to_rgb and image.mode == 'L':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class CABCDatasetWithPath(Dataset):
    """
    CABC Dataset that also returns image paths (useful for tracking predictions).

    Args:
        data_path: Path to CABC dataset split
        transform: Optional transform to apply to images
        convert_to_rgb: Whether to convert grayscale to RGB (default: True)
    """
    def __init__(self, data_path, transform=None, convert_to_rgb=True):
        self.image_folder = datasets.ImageFolder(data_path, transform=None)
        self.transform = transform
        self.convert_to_rgb = convert_to_rgb

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.imgs[idx]

        # Load image
        image = Image.open(path)

        # Convert grayscale to RGB if needed
        if self.convert_to_rgb and image.mode == 'L':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, path


def create_train_val_split(dataset, val_ratio=0.1, seed=42):
    """
    Create train/val split from a dataset.

    Args:
        dataset: PyTorch Dataset object
        val_ratio: Ratio of data to use for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset: Subset datasets for train and validation
    """
    np.random.seed(seed)

    # Get total number of samples
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # Shuffle indices
    np.random.shuffle(indices)

    # Split indices
    split_idx = int(num_samples * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    return train_dataset, val_dataset


def get_cabc_datasets(cabc_root, difficulty='easy', transform=None, val_ratio=0.1):
    """
    Create train, val, and test datasets for CABC.

    Args:
        cabc_root: Root path to CABC dataset (e.g., /path/to/cabc)
        difficulty: Difficulty level ('easy', 'medium', or 'hard')
        transform: Transform to apply to images
        val_ratio: Ratio of training data to use for validation

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert difficulty in ['easy', 'medium', 'hard'], \
        f"difficulty must be 'easy', 'medium', or 'hard', got {difficulty}"

    # Construct paths
    train_path = os.path.join(cabc_root, difficulty, 'images', 'train')
    test_path = os.path.join(cabc_root, difficulty, 'images', 'test')

    # Verify paths exist
    if not os.path.exists(train_path):
        raise ValueError(f"Train path does not exist: {train_path}")
    if not os.path.exists(test_path):
        raise ValueError(f"Test path does not exist: {test_path}")

    # Create full training dataset
    full_train_dataset = CABCDataset(train_path, transform=transform)

    # Split into train and val
    train_dataset, val_dataset = create_train_val_split(
        full_train_dataset, val_ratio=val_ratio
    )

    # Create test dataset
    test_dataset = CABCDataset(test_path, transform=transform)

    print(f"CABC {difficulty} dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset
