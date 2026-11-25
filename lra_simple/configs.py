# =============================================================================
# LINEAR PROBING
# =============================================================================

# Default grid for linear probing (smaller than fine-tuning grid)
LINEAR_PROBE_GRID = {
    "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1],
    "weight_decay": [0, 1e-4, 1e-3, 1e-2],
}

IN_LINEAR_PROBE_GRID = {
    "learning_rate": [5e-4],
    "weight_decay": [1e-5],
}

# No conditional grids needed for linear probing (simpler setup)
LINEAR_PROBE_CONDITIONAL_GRIDS = {}


# =============================================================================
# FINE-TUNING
# =============================================================================

# Default grid for fine-tuning
FINETUNE_GRID = {
    "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4],
    "batch_size": [16, 32, 64],
    "weight_decay": [0, 1e-4, 1e-2],
    "optimizer": ["adamw", "sgd"],
}

# Smaller batch sizes for memory-constrained fine-tuning
FINETUNE_SMALL_BATCH_GRID = {
    "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4],
    "batch_size": [4, 8, 16],
    "weight_decay": [0, 1e-4, 1e-2],
    "optimizer": ["adamw", "sgd"],
}

FINETUNE_CONDITIONAL_GRIDS = {
    "optimizer": {
        "sgd": {"momentum": [0.9, 0.99]},
    }
}


# =============================================================================
# TASK REGISTRY
# =============================================================================
# Every task-specific setting lives here. Dataset loaders are registered in
# common.py, and the behavior for each 'loss' type in common.LOSSES.
#
# Required keys:
#   num_classes          Output dimension of the head
#   test_splits          Splits evaluated after training
#   default_train_split  Split trained on when --train_split is not given
#   loss                 Key into common.LOSSES
# Optional keys:
#   resize_interpolation 'nearest' or 'bicubic' (default: bilinear)
#   linear_probe_grid    Overrides LINEAR_PROBE_GRID for this task

TASK_CONFIGS = {
    'pathfinder': {
        'num_classes': 2,
        'test_splits': ['9', '14', '20'],
        'default_train_split': '9',
        'loss': 'cross_entropy',
    },
    'cabc': {
        'num_classes': 1,  # Binary with BCEWithLogitsLoss
        'test_splits': ['easy', 'medium', 'hard'],
        'default_train_split': 'easy',
        'loss': 'bce',
    },
    'planko': {
        'num_classes': 1,  # Binary with BCEWithLogitsLoss
        'test_splits': ['test'],  # Single test set
        'default_train_split': 'train',
        'loss': 'bce',
    },
    'imagenet': {
        'num_classes': 1000,
        'test_splits': ['val'],
        'default_train_split': 'train',
        'loss': 'cross_entropy',
        'linear_probe_grid': IN_LINEAR_PROBE_GRID,
    },
    'psvrt': {
        'num_classes': 2,
        'test_splits': ['sd_m4_n40', 'sd_m4_n60', 'sd_m4_n120'],
        'default_train_split': 'sd_m4_n40',
        'loss': 'cross_entropy',
        'resize_interpolation': 'nearest',
    },
}
