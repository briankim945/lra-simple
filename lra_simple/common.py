"""
Shared helpers for the linear-probe and fine-tuning pipelines:
dataset loading, model input configs/transforms, and error logging.
"""

import json
import os
import timm

from lra_simple.data.pathfinder_data import get_pathfinder_datasets
from lra_simple.data.cabc_data import get_cabc_datasets
from lra_simple.data.planko_data import get_planko_datasets
from lra_simple.data.imagenet_data import get_imagenet_datasets
from lra_simple.data.psvrt_data import get_psvrt_datasets


# =============================================================================
# DATA LOADING
# =============================================================================

def get_datasets(task, data_dir, split, transform, val_ratio=0.1, train_ratio=0.9):
    """Get train/val/test datasets for a given task and split."""
    
    if task == 'pathfinder':
        return get_pathfinder_datasets(data_dir, split, transform=transform)
    
    elif task == 'cabc':
        return get_cabc_datasets(
            cabc_root=data_dir,
            difficulty=split,
            transform=transform,
            val_ratio=val_ratio
        )
    
    elif task == 'planko':
        return get_planko_datasets(
            planko_root=data_dir,
            transform=transform,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

    elif task == 'imagenet':
        return get_imagenet_datasets(
            data_dir,
            transform=transform,
            val_ratio=0.02 # default should be 0.02
        )

    elif task == 'psvrt':
        return get_psvrt_datasets(
            data_dir,
            split=split,
            transform=transform,
        )

    else:
        raise ValueError(f"Unknown task: {task}")



def get_test_dataset(task, data_dir, split, transform):
    """Get just the test dataset for a given split (for cross-split evaluation)."""
    
    if task == 'pathfinder':
        _, _, test_dataset = get_pathfinder_datasets(data_dir, split, transform=transform)
        return test_dataset
    
    elif task == 'cabc':
        _, _, test_dataset = get_cabc_datasets(
            cabc_root=data_dir,
            difficulty=split,
            transform=transform,
            val_ratio=0.1
        )
        return test_dataset
    
    elif task == 'planko':
        # Planko only has one test set
        _, _, test_dataset = get_planko_datasets(
            planko_root=data_dir,
            transform=transform,
            train_ratio=0.9,
            val_ratio=0.1
        )
        return test_dataset

    elif task == 'imagenet':
        _, _, test_dataset = get_imagenet_datasets(
            data_dir,
            transform=transform,
            val_ratio=0.02
        )
        return test_dataset

    elif task == 'psvrt':
        _, _, test_dataset = get_psvrt_datasets(
            data_dir,
            split=split,
            transform=transform,
        )
        return test_dataset

    else:
        raise ValueError(f"Unknown task: {task}")



# =============================================================================
# MODEL INPUT CONFIG
# =============================================================================

def get_model_config(model_name):
    """
    Get model's expected input configuration from TIMM.
    
    Returns:
        dict with 'input_size', 'mean', 'std', 'interpolation', 'is_imagenet_default'
    """
    # Create model temporarily to get config
    model = timm.create_model(model_name, pretrained=False)
    cfg = model.default_cfg
    
    # Get input size (prefer test_input_size if available)
    if 'test_input_size' in cfg:
        input_size = cfg['test_input_size']
    else:
        input_size = cfg.get('input_size', (3, 224, 224))
    
    # Extract height/width (input_size is (C, H, W))
    img_height = input_size[1]
    img_width = input_size[2]
    
    # Get normalization values
    mean = cfg.get('mean', (0.485, 0.456, 0.406))
    std = cfg.get('std', (0.229, 0.224, 0.225))
    interpolation = cfg.get('interpolation', 'bilinear')
    
    # Check if this matches ImageNet defaults
    is_imagenet_default = (
        img_height == 224 and 
        img_width == 224 and
        mean == (0.485, 0.456, 0.406) and
        std == (0.229, 0.224, 0.225)
    )
    
    del model
    
    return {
        'input_size': (img_height, img_width),
        'mean': mean,
        'std': std,
        'interpolation': interpolation,
        'is_imagenet_default': is_imagenet_default,
    }



def get_transform(model_name=None, interpolation=None):
    """
    Get transform for a model.

    If model_name is provided, uses the model's expected input size and normalization.
    Otherwise falls back to ImageNet defaults (224x224).

    Args:
        model_name: TIMM model name (optional)
        interpolation: Resize interpolation mode string ('nearest', 'bilinear', etc.)
                       If None, uses PIL default (bilinear).
    """
    import torchvision.transforms as transforms
    
    if model_name:
        cfg = get_model_config(model_name)
        img_size = cfg['input_size']
        mean = cfg['mean']
        std = cfg['std']
    else:
        img_size = (224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # Build resize with specified interpolation
    if interpolation == 'nearest':
        resize = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
    elif interpolation == 'bicubic':
        resize = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)
    else:
        resize = transforms.Resize(img_size)  # default (bilinear)

    return transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])



# =============================================================================
# HANDLING ERRORS
# =============================================================================

def save_error(err_file, err_lock, model_name, error):
    with err_lock:
        errors = {}
        if os.path.exists(err_file):
            try:
                with open(err_file, 'r') as f:
                    errors = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        errors[model_name] = {"error": str(error)}
        with open(err_file, 'w') as f:
            json.dump(errors, f, indent=2)

