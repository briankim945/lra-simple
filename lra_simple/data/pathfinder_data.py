"""
PathFinder Dataset Loader

Handles loading PathFinder dataset which has the following characteristics:
- Grayscale images (Mode: L) that need conversion to RGB for timm models
- Binary classification (positive/negative classes)
- Three difficulty levels: 9, 14, 20 (contour length)
- ImageFolder structure: /curv_contour_length_{difficulty}/imgs/{neg|pos}/*.png
"""

import torch
from torch.utils.data import Dataset, Subset
import torchvision.datasets as datasets
from PIL import Image
import os
import numpy as np


class PathFinderDataset(Dataset):
    """
    PathFinder Dataset wrapper that converts grayscale images to RGB for timm compatibility.

    Args:
        data_path: Path to PathFinder dataset (e.g., /path/to/pathfinder/curv_contour_length_14/imgs)
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
        # PathFinder images are Mode 'L' (grayscale), timm models expect RGB
        if self.convert_to_rgb and image.mode == 'L':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class PathFinderDatasetWithPath(Dataset):
    """
    PathFinder Dataset that also returns image paths (useful for tracking predictions).

    Args:
        data_path: Path to PathFinder dataset
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


def create_train_val_test_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create train/val/test split from a dataset.

    Args:
        dataset: PyTorch Dataset object
        train_ratio: Ratio of data to use for training (default: 0.8 = 80%)
        val_ratio: Ratio of data to use for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset: Subset datasets for train, val, and test
    """
    np.random.seed(seed)

    # Get total number of samples
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # Shuffle indices
    np.random.shuffle(indices)

    # Calculate split indices
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    return train_dataset, val_dataset, test_dataset


def get_pathfinder_datasets(pathfinder_root, difficulty='14', transform=None,
                           train_ratio=0.8, val_ratio=0.1):
    """
    Create train, val, and test datasets for PathFinder.

    Args:
        pathfinder_root: Root path to PathFinder dataset (e.g., /path/to/pathfinder300_new_2025)
        difficulty: Difficulty level ('9', '14', or '20')
        transform: Transform to apply to images
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert difficulty in ['9', '14', '20', '25'], \
        f"difficulty must be '9', '14', or '20', got {difficulty}"

    # Construct path to imgs folder
    data_path = os.path.join(pathfinder_root, f'curv_contour_length_{difficulty}', 'imgs')

    # Verify path exists
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")

    # Create full dataset
    full_dataset = PathFinderDataset(data_path, transform=transform)

    # Split into train, val, and test
    train_dataset, val_dataset, test_dataset = create_train_val_test_split(
        full_dataset, train_ratio=train_ratio, val_ratio=val_ratio
    )

    print(f"PathFinder difficulty {difficulty} dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset
