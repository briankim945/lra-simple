"""
Tilt-orientation dataset loader.

Each split directory holds an imgs/ folder and a metadata.npy array of
entries with 'filename' and 'theta' (orientation in degrees). Since
orientation repeats every 180°, the regression target is (sin 2θ, cos 2θ).
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset


class TiltDataset(Dataset):
    """
    Tilt-orientation dataset returning images and (sin 2θ, cos 2θ) targets.

    Args:
        data_path: Path to a split directory (e.g. /path/to/train, with imgs/ and metadata.npy)
        transform: Optional transform to apply to PIL images
    """
    def __init__(self, data_path, transform=None):
        imgs_path = os.path.join(data_path, 'imgs')
        metadata = np.load(os.path.join(data_path, 'metadata.npy'), allow_pickle=True)

        self.entries = []
        missing = 0
        for entry in metadata:
            img_path = os.path.join(imgs_path, entry['filename'])
            if os.path.exists(img_path):
                self.entries.append((img_path, entry['theta']))
            else:
                missing += 1
        if missing:
            print(f"WARNING: {missing}/{len(metadata)} images listed in {data_path} "
                  f"metadata are missing and were skipped")

        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, theta = self.entries[idx]
        img = Image.open(img_path).convert('RGB')

        theta_rad = theta * np.pi / 180
        target = np.array([np.sin(2 * theta_rad), np.cos(2 * theta_rad)], dtype=np.float32)

        if self.transform:
            img = self.transform(img)

        return img, target


def create_train_val_holdout(dataset, val_ratio=0.1, seed=42):
    """
    Hold out a portion of the training set for hyperparameter selection.

    Args:
        dataset: Full training dataset
        val_ratio: Fraction of training data to use as held-out val
        seed: Random seed for reproducibility

    Returns:
        train_subset, val_subset
    """
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    val_size = int(len(dataset) * val_ratio)
    val_subset = Subset(dataset, indices[:val_size])
    train_subset = Subset(dataset, indices[val_size:])

    print(f"Split tilt train: {len(train_subset)} train, {len(val_subset)} held-out val")

    return train_subset, val_subset


def get_tilt_datasets(data_dir, transform=None, val_ratio=0.1, seed=42):
    """
    Load train, val, and test datasets for the tilt-orientation task.

    Args:
        data_dir: Root directory containing train/ and test/ split directories
        transform: Transform to apply to images
        val_ratio: Fraction of train/ held out as the validation set
        seed: Random seed for the train/val split

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Tilt-orientation data path does not exist: {data_dir}")

    full_train = TiltDataset(os.path.join(data_dir, 'train'), transform=transform)
    train_dataset, val_dataset = create_train_val_holdout(full_train, val_ratio=val_ratio, seed=seed)
    test_dataset = TiltDataset(os.path.join(data_dir, 'test'), transform=transform)

    print(f"Tilt-orientation dataset loaded from {data_dir}:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset
