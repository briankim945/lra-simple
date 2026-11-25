"""
Shared helpers for the linear-probe and fine-tuning pipelines:
dataset loading, loss/metric handling, model input configs/transforms,
and error logging.
"""

import json
import os
import timm
import torch
import torch.nn as nn

from lra_simple.configs import TASK_CONFIGS
from lra_simple.data.pathfinder_data import get_pathfinder_datasets
from lra_simple.data.cabc_data import get_cabc_datasets
from lra_simple.data.planko_data import get_planko_datasets
from lra_simple.data.imagenet_data import get_imagenet_datasets
from lra_simple.data.psvrt_data import get_psvrt_datasets


# =============================================================================
# DATA LOADING
# =============================================================================

# Adapters giving every loader the same (data_dir, split, transform, **ratios)
# signature. Ratios are forwarded as given; a loader that doesn't take one
# (e.g. PSVRT, which has fixed splits) raises a TypeError.
DATASET_LOADERS = {
    'pathfinder': lambda data_dir, split, transform, **ratios: get_pathfinder_datasets(
        data_dir, split, transform=transform, **ratios),
    'cabc': lambda data_dir, split, transform, **ratios: get_cabc_datasets(
        cabc_root=data_dir, difficulty=split, transform=transform, **ratios),
    # Planko has a single split
    'planko': lambda data_dir, split, transform, **ratios: get_planko_datasets(
        planko_root=data_dir, transform=transform, **ratios),
    'imagenet': lambda data_dir, split, transform, **ratios: get_imagenet_datasets(
        data_dir, transform=transform, **ratios),
    'psvrt': lambda data_dir, split, transform, **ratios: get_psvrt_datasets(
        data_dir, split=split, transform=transform, **ratios),
}
assert set(DATASET_LOADERS) == set(TASK_CONFIGS), \
    "DATASET_LOADERS and TASK_CONFIGS must list the same tasks"


def get_datasets(task, data_dir, split, transform, val_ratio=None, train_ratio=None):
    """
    Get train/val/test datasets for a given task and split.

    val_ratio / train_ratio of None use the task loader's own default.
    """
    if task not in DATASET_LOADERS:
        raise ValueError(f"Unknown task: {task}")
    ratios = {k: v for k, v in [('val_ratio', val_ratio), ('train_ratio', train_ratio)]
              if v is not None}
    return DATASET_LOADERS[task](data_dir, split, transform, **ratios)


def get_test_dataset(task, data_dir, split, transform, val_ratio=None, train_ratio=None):
    """
    Get just the test dataset for a given split (for cross-split evaluation).

    Takes the same ratios as get_datasets, since some tasks (e.g. Pathfinder)
    carve the test set out of the same pool.
    """
    return get_datasets(task, data_dir, split, transform,
                        val_ratio=val_ratio, train_ratio=train_ratio)[2]


# =============================================================================
# LOSSES AND METRICS
# =============================================================================
# Each loss type defines how labels are shaped for the criterion, how a batch
# is scored, and which direction is better. Batch scores are summed over an
# epoch and converted to the reported metric by summarise_score().

def _bce_targets(labels):
    return labels.float().unsqueeze(1)


def _bce_correct(outputs, targets):
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    return (predicted == targets).sum().item()


def _ce_correct(outputs, targets):
    _, predicted = outputs.max(1)
    return predicted.eq(targets).sum().item()


LOSSES = {
    'cross_entropy': {
        'criterion': nn.CrossEntropyLoss,
        'prepare_targets': lambda labels: labels,
        'batch_score': _ce_correct,  # number correct
        'scale': 100.,               # report accuracy in %
        'higher_is_better': True,
        'unit': '%',
    },
    'bce': {
        'criterion': nn.BCEWithLogitsLoss,
        'prepare_targets': _bce_targets,
        'batch_score': _bce_correct,
        'scale': 100.,
        'higher_is_better': True,
        'unit': '%',
    },
}
assert all(cfg['loss'] in LOSSES for cfg in TASK_CONFIGS.values()), \
    "Every TASK_CONFIGS 'loss' must be a key of LOSSES"


def _loss(task):
    return LOSSES[TASK_CONFIGS[task]['loss']]


def make_criterion(task):
    return _loss(task)['criterion']()


def prepare_targets(task, labels):
    return _loss(task)['prepare_targets'](labels)


def batch_score(task, outputs, targets):
    """Summable score for one batch (e.g. number of correct predictions)."""
    return _loss(task)['batch_score'](outputs, targets)


def summarise_score(task, score_sum, count):
    """Turn a summed batch score into the reported metric (e.g. accuracy %)."""
    return _loss(task)['scale'] * score_sum / count


def is_better(task, new, old, margin=0.0):
    """Whether metric `new` beats `old` by more than `margin`."""
    if _loss(task)['higher_is_better']:
        return new > old + margin
    return new < old - margin


def worst_score(task):
    """Starting value for best-so-far tracking; any real result beats it."""
    return 0. if _loss(task)['higher_is_better'] else float('inf')


def format_score(task, value):
    return f"{value:.2f}{_loss(task)['unit']}"


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

