"""
Minimal utilities for CABC evaluation (without submodule dependencies)
"""

import torch
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from timm.data.transforms import str_to_interp_mode


def binary_accuracy(output, target):
    """Binary classification accuracy"""
    with torch.no_grad():
        output = output > 0
        return torch.sum(output == target).item() / len(target) * 100


def get_transform_wo_crop(data_config, normalization='imagenet', dataset_mean=None, dataset_std=None):
    """
    Get transform without center crop for TIMM models

    Args:
        data_config: TIMM model data config from timm.data.resolve_model_data_config()
        normalization: Normalization scheme - 'imagenet' (default), 'none', 'simple', 'dataset'
            - 'imagenet': Use ImageNet mean/std from data_config
            - 'none': No normalization, just scale to [0, 1]
            - 'simple': Simple [-1, 1] normalization using mean=0.5, std=0.5
            - 'dataset': Use custom mean/std (requires dataset_mean, dataset_std)
        dataset_mean: Custom mean for 'dataset' normalization (tuple of 3 values)
        dataset_std: Custom std for 'dataset' normalization (tuple of 3 values)

    Returns:
        Transform composition
    """
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)

    interpolation = data_config['interpolation']

    tf = []
    tf += [transforms.Resize(input_size[0], interpolation=str_to_interp_mode(interpolation))]
    tf += [transforms.ToTensor()]

    # Add normalization based on scheme
    if normalization == 'imagenet':
        mean = data_config['mean']
        std = data_config['std']
        tf += [transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    elif normalization == 'none':
        # No normalization - images stay in [0, 1] range
        pass
    elif normalization == 'simple':
        # Simple standardization: (x - 0.5) / 0.5 -> maps [0,1] to [-1,1]
        tf += [transforms.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]),
                                    std=torch.tensor([0.5, 0.5, 0.5]))]
    elif normalization == 'dataset':
        if dataset_mean is None or dataset_std is None:
            raise ValueError("dataset_mean and dataset_std required for 'dataset' normalization")
        tf += [transforms.Normalize(mean=torch.tensor(dataset_mean),
                                    std=torch.tensor(dataset_std))]
    else:
        raise ValueError(f"Unknown normalization scheme: {normalization}")

    return transforms.Compose(tf)
