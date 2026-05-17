"""
ImageNet Dataset Loader

Handles loading ImageNet-1k for linear probing of frozen foundation models.

Directory structure expected:
    imagenet_root/
        train/
            n01440764/
                *.JPEG
            n01443537/
                *.JPEG
            ...
        val/
            n01440764/
                *.JPEG
            n01443537/
                *.JPEG
            ...

Both train and val must use synset IDs (nXXXXXXXX) as directory names so that
ImageFolder produces consistent label mappings across both splits.

Since ImageNet's public test set has no labels, the official validation set is
used as the test set, and a portion of the training set is held out as a
validation set for hyperparameter selection.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.datasets as datasets
from PIL import Image


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset wrapper that ensures RGB conversion for timm models.

    A small fraction of ImageNet images are grayscale or CMYK; converting all
    images to RGB ensures consistent tensor shapes downstream.

    Args:
        data_path: Path to an ImageNet split directory (e.g. /path/to/train)
        transform: Optional transform to apply to images
    """
    def __init__(self, data_path, transform=None):
        self.image_folder = datasets.ImageFolder(data_path, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.imgs[idx]
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNetDatasetWithPath(Dataset):
    """ImageNet Dataset that also returns image paths (for XAI / debugging)."""
    def __init__(self, data_path, transform=None):
        self.image_folder = datasets.ImageFolder(data_path, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.imgs[idx]
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path


def create_train_val_holdout(dataset, val_ratio=0.02, seed=42):
    """
    Hold out a small portion of the training set for hyperparameter selection.

    Args:
        dataset: Full training dataset
        val_ratio: Fraction of training data to use as held-out val (default: 0.02 ~25K images)
        seed: Random seed for reproducibility

    Returns:
        train_subset, val_subset
    """
    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    print(f"Split ImageNet train: {len(train_subset)} train, {len(val_subset)} held-out val")

    return train_subset, val_subset


def get_imagenet_datasets(imagenet_root, transform=None, val_ratio=0.02):
    """
    Create train/val/test datasets for ImageNet.

    The official ImageNet validation set is used as the *test* set (since the
    public test set has no labels). A small portion of train is held out as the
    *validation* set used for hyperparameter selection.

    Args:
        imagenet_root: Root path containing 'train' and 'val' subdirectories
        transform: Transform to apply to images
        val_ratio: Fraction of training data to use as held-out val

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_path = os.path.join(imagenet_root, 'train')
    val_path = os.path.join(imagenet_root, 'val')

    if not os.path.exists(train_path):
        raise ValueError(f"Train path does not exist: {train_path}")
    if not os.path.exists(val_path):
        raise ValueError(f"Val path does not exist: {val_path}")

    # Build full train dataset, then split off held-out val
    full_train = ImageNetDataset(train_path, transform=transform)
    train_dataset, val_dataset = create_train_val_holdout(full_train, val_ratio=val_ratio)

    # Official val becomes the test set
    test_dataset = ImageNetDataset(val_path, transform=transform)

    # Sanity check: train and test should have the same class ordering
    train_classes = full_train.image_folder.classes
    test_classes = test_dataset.image_folder.classes
    if train_classes != test_classes:
        raise ValueError(
            f"ImageNet train/val class mismatch! Train has {len(train_classes)} classes, "
            f"val has {len(test_classes)}. First mismatch: "
            f"train[0]={train_classes[0]}, val[0]={test_classes[0]}. "
            f"Make sure both directories use synset IDs (nXXXXXXXX) as folder names."
        )

    print(f"ImageNet dataset loaded:")
    print(f"  Train:    {len(train_dataset)} samples")
    print(f"  Held val: {len(val_dataset)} samples")
    print(f"  Test:     {len(test_dataset)} samples (official val set)")
    print(f"  Classes:  {len(train_classes)}")

    return train_dataset, val_dataset, test_dataset