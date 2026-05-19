"""
PSVRT (Parametric Same-Different Visual Reasoning Test) Dataset Loader

Handles loading PSVRT datasets stored as numpy arrays with the following structure:
    data_dir/
        psvrt_sd_m{M}_n{N}/
            train_images.npy   # (N_train, H, W, 1), float values in [0, 1]
            train_labels.npy   # (N_train,), int 0/1
            val_images.npy
            val_labels.npy
            test_images.npy
            test_labels.npy

The task is binary classification: "same" (1) vs "different" (0) for pairs of
small squares placed in a larger image.

Parameters encoded in directory name:
    - M (e.g., m4): size of the squares in pixels
    - N (e.g., n40, n60, n120): size of the overall image in pixels
    Larger N with fixed M = harder (squares occupy smaller fraction of image)

Images are single-channel grayscale and are converted to 3-channel RGB for
compatibility with pretrained vision models.

IMPORTANT: NEAREST interpolation should be used when resizing these images,
as bilinear/bicubic interpolation blurs the small square patterns and
destroys the information needed for the task.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PSVRTDataset(Dataset):
    """
    PSVRT Dataset that loads from numpy arrays and outputs PIL images.

    Args:
        images_path: Path to .npy file containing images (N, H, W, 1)
        labels_path: Path to .npy file containing labels (N,)
        transform: Optional transform to apply to PIL images
    """
    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path)    # (N, H, W, 1), float [0, 1]
        self.labels = np.load(labels_path).astype(np.int64)
        self.transform = transform

        # Validate shapes
        assert len(self.images) == len(self.labels), \
            f"Image/label count mismatch: {len(self.images)} vs {len(self.labels)}"
        assert self.images.ndim == 4 and self.images.shape[3] == 1, \
            f"Expected shape (N, H, W, 1), got {self.images.shape}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]              # (H, W, 1), float [0, 1]
        img = np.repeat(img, 3, axis=2)     # (H, W, 3)
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        img = Image.fromarray(img, mode='RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


def get_psvrt_datasets(data_dir, split, transform=None):
    """
    Load train, val, and test datasets for a specific PSVRT configuration.

    Args:
        data_dir: Root directory containing PSVRT subdirectories
        split: Configuration name, e.g. 'sd_m4_n40' or 'psvrt_sd_m4_n40'
                (the 'psvrt_' prefix is added automatically if not present)
        transform: Transform to apply to images

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Handle split naming: accept both 'sd_m4_n40' and 'psvrt_sd_m4_n40'
    if not split.startswith('psvrt_'):
        split_dir = f'psvrt_{split}'
    else:
        split_dir = split

    data_path = os.path.join(data_dir, split_dir)

    if not os.path.exists(data_path):
        # Also try without prefix in case the directory is named differently
        alt_path = os.path.join(data_dir, split)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise ValueError(
                f"PSVRT data path does not exist: {data_path}\n"
                f"Also tried: {alt_path}"
            )

    # Verify all files exist
    required_files = [
        'train_images.npy', 'train_labels.npy',
        'val_images.npy', 'val_labels.npy',
        'test_images.npy', 'test_labels.npy',
    ]
    for f in required_files:
        fpath = os.path.join(data_path, f)
        if not os.path.exists(fpath):
            raise ValueError(f"Missing required file: {fpath}")

    train_dataset = PSVRTDataset(
        os.path.join(data_path, 'train_images.npy'),
        os.path.join(data_path, 'train_labels.npy'),
        transform=transform,
    )
    val_dataset = PSVRTDataset(
        os.path.join(data_path, 'val_images.npy'),
        os.path.join(data_path, 'val_labels.npy'),
        transform=transform,
    )
    test_dataset = PSVRTDataset(
        os.path.join(data_path, 'test_images.npy'),
        os.path.join(data_path, 'test_labels.npy'),
        transform=transform,
    )

    # Print dataset info
    img_shape = train_dataset.images.shape
    print(f"PSVRT {split_dir} dataset loaded:")
    print(f"  Image size: {img_shape[1]}x{img_shape[2]}")
    print(f"  Train: {len(train_dataset)} samples "
          f"({train_dataset.labels.sum()} same, {len(train_dataset) - train_dataset.labels.sum()} diff)")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset