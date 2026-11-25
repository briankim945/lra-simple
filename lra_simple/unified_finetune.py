#!/usr/bin/env python3
"""
Unified Fine-tuning Script for Vision Tasks

Supports: PathFinder, CABC, Planko, ImageNet, PSVRT, and Tilt
Can run single model or batch with grid search across multiple GPUs.
"""

import argparse
import json
import os
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from filelock import FileLock

# Task-specific imports
from lra_simple.grid_search import grid_search_with_conditionals
from lra_simple.configs import TASK_CONFIGS, FINETUNE_GRID, FINETUNE_CONDITIONAL_GRIDS, \
    FINETUNE_SMALL_BATCH_GRID, FINETUNE_REDUCED_GRID
from lra_simple.common import get_datasets, get_test_dataset, get_model_config, \
    get_transform, save_error, make_criterion, prepare_targets, batch_score, \
    summarise_score, is_better, worst_score, format_score, default_min_delta, \
    center_readout_spec, center_features, describe_center_readout


# =============================================================================
# MODEL CREATION
# =============================================================================

class CenterReadoutModel(nn.Module):
    """Backbone plus a linear head on its center feature (see common.center_features)."""

    def __init__(self, backbone, center_spec, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.center_spec = center_spec
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(center_features(self.backbone, x, self.center_spec))

    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)


def create_model(model_name, num_classes, pretrained=True, task=None, input_size=None):
    """
    Create a TIMM model with the appropriate head. Center-readout tasks get a
    linear head on the center feature instead of the model's pooled head.
    """
    if task is not None and TASK_CONFIGS[task].get('readout') == 'center':
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        center_spec, feature_dim = center_readout_spec(backbone, input_size)
        print(f"Center readout: {describe_center_readout(center_spec)}")
        return CenterReadoutModel(backbone, center_spec, feature_dim, num_classes)
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, use_amp=True, task='pathfinder', accumulation_steps=1):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0
    score = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Zero gradients at the start
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = prepare_targets(task, labels.to(device))

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            # Only step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            loss.backward()
            
            # Only step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Track unscaled loss for logging
        if accumulation_steps > 1:
            total_loss += loss.item() * accumulation_steps
        else:
            total_loss += loss.item()
        
        score += batch_score(task, outputs, labels)
        total += labels.size(0)
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1),
                          'score': summarise_score(task, score, total)})
    
    # Handle any remaining gradients (if dataset size not divisible by accumulation_steps)
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(train_loader), summarise_score(task, score, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True, task='pathfinder'):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    score = 0
    total = 0

    for images, labels in tqdm(loader, desc='Evaluating', leave=False):
        images = images.to(device)
        labels = prepare_targets(task, labels.to(device))

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        score += batch_score(task, outputs, labels)
        total += labels.size(0)

    return total_loss / len(loader), summarise_score(task, score, total)


# =============================================================================
# FINE-TUNING
# =============================================================================

def finetune_model(
    model_name,
    task,
    data_dir,
    train_split,
    epochs=30,
    batch_size=32,
    micro_batch_size=None,
    lr=1e-4,
    weight_decay=0.01,
    optimizer_name="adamw",
    momentum=None,
    gpu=0,
    use_amp=True,
    output_dir='results',
    prev_best_val_acc=None,
    num_workers=4,
    gradient_checkpointing=False,
    early_stopping=False,
    patience=5,
    min_delta=None
):
    """
    Fine-tune a model on a specific task.
    
    Args:
        batch_size: Effective batch size (for training dynamics)
        micro_batch_size: Actual batch size in memory (for gradient accumulation)
                         If None, uses batch_size directly (no accumulation)
    
    Returns history dict with training metrics and test accuracies.
    """
    task_config = TASK_CONFIGS[task]
    if prev_best_val_acc is None:
        prev_best_val_acc = worst_score(task)
    if min_delta is None:
        min_delta = default_min_delta(task)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Calculate gradient accumulation
    if micro_batch_size is not None and micro_batch_size < batch_size:
        actual_batch_size = micro_batch_size
        accumulation_steps = batch_size // micro_batch_size
    else:
        actual_batch_size = batch_size
        accumulation_steps = 1
    
    # Get model config for transform
    model_cfg = get_model_config(model_name)
    
    print(f"\n{'='*70}")
    print(f"Fine-tuning: {model_name}")
    print(f"Task: {task}, Train split: {train_split}")
    print(f"Input size: {model_cfg['input_size']}, ImageNet default: {model_cfg['is_imagenet_default']}")
    if accumulation_steps > 1:
        print(f"LR: {lr}, Effective BS: {batch_size} (micro={actual_batch_size}, accum={accumulation_steps}), WD: {weight_decay}, Opt: {optimizer_name}")
    else:
        print(f"LR: {lr}, BS: {batch_size}, WD: {weight_decay}, Opt: {optimizer_name}")
    print(f"Device: {device}, AMP: {use_amp}, GradCheckpoint: {gradient_checkpointing}")
    if early_stopping:
        print(f"Early stopping: patience={patience}, min_delta={format_score(task, min_delta)}")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir) / task / model_name / f'finetune_{train_split}'
    output_path.mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model(model_name, task_config['num_classes'], pretrained=True,
                         task=task, input_size=model_cfg['input_size'])
    
    # Enable gradient checkpointing if requested (saves memory, slower training)
    if gradient_checkpointing:
        try:
            model.set_grad_checkpointing(True)
            print("Gradient checkpointing enabled")
        except AttributeError:
            print("Warning: Model does not support gradient checkpointing, continuing without it")
    
    model = model.to(device)

    # Get transform and data
    resize_interp = task_config.get('resize_interpolation')
    transform = get_transform(model_name, interpolation=resize_interp)
    train_dataset, val_dataset, test_dataset = get_datasets(
        task, data_dir, train_split, transform
    )

    # Use actual_batch_size for DataLoader (may be smaller than effective batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=actual_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=actual_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Loss and optimizer
    criterion = make_criterion(task)

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            momentum=momentum if momentum else 0.9
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_acc = worst_score(task)
    best_epoch = 0
    saved_new_best = False
    # Best weights are held on CPU and only written to disk once this config
    # finishes, so a crash mid-config can't clobber a previous config's checkpoint
    best_state = None
    best_config = None
    # Patience is tracked against the last improvement larger than min_delta,
    # separately from best-epoch selection
    patience_best_val_acc = worst_score(task)
    epochs_without_improvement = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, task,
            accumulation_steps=accumulation_steps
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp, task)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train={format_score(task, train_acc)}, "
              f"Val={format_score(task, val_acc)}")

        scheduler.step()

        # Best-epoch selection: any strict improvement
        if is_better(task, val_acc, best_val_acc):
            best_val_acc = val_acc
            best_epoch = epoch

            if is_better(task, val_acc, prev_best_val_acc):
                # Clone to CPU: state_dict() shares storage with the live parameters
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_config = {
                    'model_name': model_name,
                    'task': task,
                    'train_split': train_split,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'weight_decay': weight_decay,
                    'optimizer': optimizer_name,
                    'momentum': momentum,
                    'use_amp': use_amp,
                    'best_epoch': epoch,
                    'best_val_acc': val_acc,
                }
                print(f"  ✓ New best snapshot (val: {format_score(task, val_acc)})")

        # Early-stopping patience: only improvements larger than min_delta reset it
        if is_better(task, val_acc, patience_best_val_acc, margin=min_delta):
            patience_best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if early_stopping and epochs_without_improvement >= patience:
            print(f"  Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Cross-split evaluation (only if this config beat the previous best)
    history['test_acc'] = {}

    if best_state is not None:
        print(f"\nCross-split evaluation:")
        model.load_state_dict(best_state)
        
        for test_split in task_config['test_splits']:
            # Reuse test_dataset from initial split when test_split == train_split
            # to avoid potential data leakage from re-splitting
            if test_split == train_split:
                eval_test_dataset = test_dataset
            else:
                eval_test_dataset = get_test_dataset(task, data_dir, test_split, transform)
            
            test_loader = DataLoader(
                eval_test_dataset, batch_size=actual_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
            _, test_acc = evaluate(model, test_loader, criterion, device, use_amp, task)
            history['test_acc'][test_split] = test_acc
            print(f"  {test_split}: {format_score(task, test_acc)}")

        # Config finished: now safe to overwrite the previous best on disk
        torch.save(best_state, output_path / 'best_model.pth')
        with open(output_path / 'config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        saved_new_best = True
        print(f"  ✓ New best model saved (val: {format_score(task, best_val_acc)})")

    # Package results
    history['best_val_acc'] = best_val_acc
    history['best_epoch'] = best_epoch
    history['saved_new_best'] = saved_new_best
    history['model_cfg'] = model_cfg
    history['config'] = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'optimizer': optimizer_name,
        'momentum': momentum,
        'use_amp': use_amp,
    }

    return history


def run_model_grid_search(
    model_name,
    task,
    data_dir,
    train_split=None,
    epochs=30,
    gpu=0,
    output_dir='results',
    log_file=None,
    num_workers=4,
    grid=None,
    conditional_grids=None,
    verbose=False,
    gradient_checkpointing=False,
    early_stopping=False,
    patience=5,
    min_delta=None,
    micro_batch_size=None
):
    """
    Run grid search for a single model on a specific task.
    
    Args:
        model_name: TIMM model name
        task: A key of TASK_CONFIGS
        data_dir: Path to task data
        train_split: Split to train on (uses task default if None)
        epochs: Training epochs per config
        gpu: GPU ID
        output_dir: Output directory
        log_file: JSON file for results (enables skip if already done)
        num_workers: DataLoader workers
        grid: Custom grid dict (uses FINETUNE_GRID if None)
        conditional_grids: Custom conditional grids (uses CONDITIONAL_GRIDS if None)
        verbose: Enable detailed logging
        gradient_checkpointing: Enable gradient checkpointing to save memory
        early_stopping: Enable early stopping
        patience: Epochs without improvement before stopping
        min_delta: Minimum improvement to reset patience, in the task's units
                   (None: the task's default, 0.1% or 0°)
        micro_batch_size: Actual batch size in memory (enables gradient accumulation)
        
    Returns:
        Dict with best_val_acc and best_result
    """
    import sys
    import gc
    import traceback

    def log(msg, override=False):
        """Print with timestamp and flush immediately (only if verbose)."""
        if verbose or override:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            sys.stdout.flush()
    
    task_config = TASK_CONFIGS[task]
    train_split = train_split or task_config['default_train_split']
    grid = grid or FINETUNE_GRID
    conditional_grids = conditional_grids or FINETUNE_CONDITIONAL_GRIDS
    
    print(f"\n{'='*70}")
    print(f"GRID SEARCH: {model_name}")
    print(f"Task: {task}, Train split: {train_split}")
    if micro_batch_size:
        print(f"Micro batch size: {micro_batch_size} (gradient accumulation enabled)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    if verbose:
        sys.stdout.flush()

    # Create lock for log_file
    json_lock = FileLock(log_file + ".lock")

    # Per-config results (one JSON line per config) sit next to the main log
    configs_file = str(Path(log_file).with_name(Path(log_file).stem + '_configs.jsonl'))
    configs_lock = FileLock(configs_file + ".lock")

    # Check if already done successfully (skip if it has a real val score)
    log("Checking for existing results...")
    if log_file and os.path.exists(log_file):
        with json_lock:
            try:
                with open(log_file, 'r') as f:
                    results = json.load(f)
                if model_name in results:
                    existing = results[model_name].get('best_val_acc')
                    if existing is not None and is_better(task, existing, worst_score(task)):
                        print(f"SKIP: {model_name} already completed (val={format_score(task, existing)})")
                        return None
                    else:
                        print(f"RETRY: {model_name} previously failed (no val score)")
            except:
                pass

    # Generate configs
    configs = grid_search_with_conditionals(grid, conditional_grids)
    print(f"Grid configurations: {len(configs)}")

    # Get model config once (for metadata)
    model_cfg = get_model_config(model_name)
    print(f"Model input size: {model_cfg['input_size']}, ImageNet default: {model_cfg['is_imagenet_default']}")

    best_val_acc = worst_score(task)
    best_result = {}
    status_counts = {'ok': 0, 'oom': 0, 'error': 0}

    for idx, config in enumerate(configs):
        print(f"\n>>> Config {idx+1}/{len(configs)}: "
              f"lr={config['learning_rate']:.0e}, bs={config['batch_size']}, "
              f"wd={config['weight_decay']:.0e}, opt={config['optimizer']}")

        record = {
            'model_name': model_name,
            'task': task,
            'train_split': train_split,
            'config_idx': idx,
            'config': config,
            'status': 'ok',
            'best_val_acc': None,
            'best_epoch': None,
            'test_acc': None,
            'error': None,
        }

        try:
            history = finetune_model(
                model_name=model_name,
                task=task,
                data_dir=data_dir,
                train_split=train_split,
                epochs=epochs,
                batch_size=config['batch_size'],
                micro_batch_size=micro_batch_size,
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                optimizer_name=config['optimizer'],
                momentum=config.get('momentum'),
                gpu=gpu,
                use_amp=True,
                output_dir=output_dir,
                prev_best_val_acc=best_val_acc,
                num_workers=num_workers,
                gradient_checkpointing=gradient_checkpointing,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta
            )

            record['best_val_acc'] = history['best_val_acc']
            record['best_epoch'] = history['best_epoch']
            # Test accuracies are only computed for configs that beat the previous best
            record['test_acc'] = history.get('test_acc') or None

            if is_better(task, history['best_val_acc'], best_val_acc):
                best_val_acc = history['best_val_acc']
                best_result = {
                    'best_val_acc': history['best_val_acc'],
                    'best_epoch': history['best_epoch'],
                    'test_acc': history.get('test_acc', {}),
                    'config': history['config']
                }

        except torch.cuda.OutOfMemoryError as e:
            # Keep going: a smaller batch size later in the grid may still fit
            record['status'] = 'oom'
            record['error'] = str(e).split('\n')[0]
            print(f"OOM: {record['error']}")

        except Exception as e:
            record['status'] = 'error'
            record['error'] = str(e)
            print(f"ERROR: {e}")
            traceback.print_exc()

        # Free GPU memory only after the except block has released the traceback,
        # which otherwise keeps the failed config's model alive
        if record['status'] == 'oom':
            gc.collect()
            torch.cuda.empty_cache()
        status_counts[record['status']] += 1

        record['timestamp'] = datetime.now().isoformat(timespec='seconds')
        with configs_lock:
            with open(configs_file, 'a') as f:
                f.write(json.dumps(record) + '\n')

    # Every config ran out of memory: raise so main() records the error and
    # counts it toward the consecutive-OOM exit
    if status_counts['oom'] == len(configs):
        raise torch.cuda.OutOfMemoryError(
            f"All {len(configs)} configs ran out of memory for {model_name}"
        )

    if status_counts['oom']:
        print(f"WARNING: {status_counts['oom']}/{len(configs)} configs ran out of memory; "
              f"best result is only among the configs that fit")

    # Save results
    final_result = {
        'best_val_acc': best_val_acc,
        'best_result': best_result,
        'input_size': model_cfg['input_size'],
        'is_imagenet_default': model_cfg['is_imagenet_default'],
    }
    
    with json_lock:
        if log_file:
            results = {}
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        results = json.load(f)
                except:
                    pass
            
            results[model_name] = final_result
            
            for attempt in range(5):
                try:
                    with open(log_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    break
                except Exception as e:
                    print(f"Save attempt {attempt+1} failed: {e}")

    print(f"\n{'='*70}")
    print(f"COMPLETE: {model_name}")
    print(f"Best val: {format_score(task, best_val_acc)}")
    print(f"{'='*70}")

    return final_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Fine-tuning for Vision Tasks')
    
    # Task configuration
    parser.add_argument('--task', type=str, required=True,
                        choices=list(TASK_CONFIGS),
                        help='Task to run')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to task data')
    parser.add_argument('--train_split', type=str, default=None,
                        help='Split to train on (uses task default if not specified)')
    
    # Model selection
    parser.add_argument('--model_name', type=str, default=None,
                        help='Single model to run (if not provided, uses --models_csv)')
    parser.add_argument('--models_csv', type=str, default=None,
                        help='CSV file with model names')
    parser.add_argument('--models_list', type=str, default=None,
                        help='Text file with model names (one per line)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index in model list')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index in model list (exclusive)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Model filtering
    parser.add_argument('--skip_non_224', action='store_true',
                        help='Skip models with input size > 224 (shorthand for --max_input_size 224)')
    parser.add_argument('--max_input_size', type=int, default=None,
                        help='Skip models with input size larger than this (overrides --skip_non_224)')
    parser.add_argument('--skip_errors', action='store_true',
                        help='Skip models that previously errored and are logged')
    
    # Grid selection (default: FINETUNE_GRID)
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument('--small_batch', action='store_true',
                            help='Use smaller batch sizes [4, 8, 16] instead of [16, 32, 64]')
    grid_group.add_argument('--reduced_grid', action='store_true',
                            help='Use the 6-config FINETUNE_REDUCED_GRID (smoke tests, quick sweeps)')

    # Memory optimization
    parser.add_argument('--micro_batch_size', type=int, default=None,
                        help='Actual batch size in GPU memory (enables gradient accumulation to simulate larger effective batch sizes)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing (saves memory, ~30%% slower)')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on validation accuracy')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience - epochs without improvement (default: 5)')
    parser.add_argument('--min_delta', type=float, default=None,
                        help='Minimum val improvement to reset early-stopping patience, in the '
                             "task's units (default: 0.1%% for accuracy, 0° for angular error)")
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--log_file', type=str, default=None,
                        help='JSON log file (default: output_dir/task_finetune.json)')
    parser.add_argument('--err_file', type=str, default=None,
                        help='JSON errors file')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed logging with timestamps (slower but useful for debugging)')

    args = parser.parse_args()
    if args.min_delta is None:
        args.min_delta = default_min_delta(args.task)

    # Determine max input size for filtering
    if args.max_input_size:
        max_input_size = args.max_input_size
    elif args.skip_non_224:
        max_input_size = 224
    else:
        max_input_size = None

    # Select grid
    if args.small_batch:
        grid = FINETUNE_SMALL_BATCH_GRID
    elif args.reduced_grid:
        grid = FINETUNE_REDUCED_GRID
    else:
        grid = FINETUNE_GRID

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = args.log_file or str(output_dir / f'{args.task}_finetune.json')
    err_file = args.err_file or str(output_dir / f'{args.task}_finetune_errors.json')

    # Single model mode
    if args.model_name:
        run_model_grid_search(
            model_name=args.model_name,
            task=args.task,
            data_dir=args.data_dir,
            train_split=args.train_split,
            epochs=args.epochs,
            gpu=args.gpu,
            output_dir=args.output_dir,
            log_file=log_file,
            num_workers=args.num_workers,
            grid=grid,
            verbose=args.verbose,
            gradient_checkpointing=args.gradient_checkpointing,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            micro_batch_size=args.micro_batch_size
        )
        return

    # Batch mode - load model list
    import sys
    import gc
    
    models = []
    
    if args.models_csv:
        import pandas as pd
        df = pd.read_csv(args.models_csv)
        models = df['model_name'].tolist()
    elif args.models_list:
        with open(args.models_list, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide --model_name, --models_csv, or --models_list")

    # Apply index range
    end_idx = args.end_idx or len(models)
    models = models[args.start_idx:end_idx]
    
    print(f"Processing {len(models)} models (idx {args.start_idx} to {end_idx-1})")
    if max_input_size:
        print(f"Skipping models with input size > {max_input_size}")
    if args.small_batch:
        print(f"Using small batch grid: {FINETUNE_SMALL_BATCH_GRID['batch_size']}")
    if args.reduced_grid:
        print("Using reduced grid (FINETUNE_REDUCED_GRID)")
    if args.micro_batch_size:
        print(f"Micro batch size: {args.micro_batch_size} (gradient accumulation enabled)")
    if args.gradient_checkpointing:
        print("Gradient checkpointing enabled")
    if args.early_stopping:
        print(f"Early stopping: patience={args.patience}, min_delta={format_score(args.task, args.min_delta)}")
    sys.stdout.flush()

    # Track OOM state
    consecutive_oom_failures = 0
    max_consecutive_oom = 3  # Exit after 3 consecutive OOM errors

    # Manage locks on the error file
    err_lock = FileLock(err_file + ".lock")

    # Run grid search for each model
    for i, model_name in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model_name}")
        sys.stdout.flush()

        if args.skip_errors:
            with err_lock:
                if err_file and os.path.exists(err_file):
                    with open(err_file, 'r') as f:
                        errors = json.load(f)
                    if model_name in errors:
                        continue
        
        # Check input size if filtering is enabled
        if max_input_size:
            try:
                cfg = get_model_config(model_name)
                height, width = cfg['input_size']
                if height > max_input_size or width > max_input_size:
                    print(f"  SKIP: Input size {cfg['input_size']} > {max_input_size}")
                    sys.stdout.flush()
                    continue
            except Exception as e:
                print(f"  SKIP: Could not get model config: {e}")
                sys.stdout.flush()
                continue
        
        try:
            run_model_grid_search(
                model_name=model_name,
                task=args.task,
                data_dir=args.data_dir,
                train_split=args.train_split,
                epochs=args.epochs,
                gpu=args.gpu,
                output_dir=args.output_dir,
                log_file=log_file,
                num_workers=args.num_workers,
                grid=grid,
                verbose=args.verbose,
                gradient_checkpointing=args.gradient_checkpointing,
                early_stopping=args.early_stopping,
                patience=args.patience,
                min_delta=args.min_delta,
                micro_batch_size=args.micro_batch_size
            )
            # Reset OOM counter on success
            consecutive_oom_failures = 0
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM ERROR: {model_name}: {e}")
            sys.stdout.flush()
            
            # Try to recover
            torch.cuda.empty_cache()
            gc.collect()
            
            # Test if GPU is recoverable
            try:
                device = torch.device(f'cuda:{args.gpu}')
                test_tensor = torch.zeros(1000, device=device)
                del test_tensor
                torch.cuda.empty_cache()
                print(f"  GPU recovered, continuing...")
                consecutive_oom_failures += 1
            except:
                consecutive_oom_failures += 1
                print(f"  GPU recovery failed ({consecutive_oom_failures}/{max_consecutive_oom})")

            save_error(err_file, err_lock, model_name, e)
            
            sys.stdout.flush()
            
            # Exit if too many consecutive OOM failures
            if consecutive_oom_failures >= max_consecutive_oom:
                print(f"\n{'='*70}")
                print(f"FATAL: {max_consecutive_oom} consecutive OOM errors")
                print(f"GPU memory likely corrupted. Exiting to prevent cascade failures.")
                print(f"Remaining models: {len(models) - i - 1}")
                print(f"{'='*70}")
                sys.stdout.flush()
                sys.exit(1)
                
        except Exception as e:
            print(f"  FAILED: {model_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            # Reset OOM counter - this was a different kind of error
            consecutive_oom_failures = 0

            save_error(err_file, err_lock, model_name, e)


if __name__ == '__main__':
    main()