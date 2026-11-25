#!/usr/bin/env python3
"""
Unified Linear Probing Script for Vision Tasks

Supports: PathFinder, CABC, Planko, ImageNet, PSVRT, and Tilt
Fast screening of models by training only a linear head on frozen features.
Can run single model or batch with grid search across multiple GPUs.
"""

import argparse
import json
import os
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from filelock import FileLock

# Task-specific imports
from lra_simple.grid_search import grid_search_with_conditionals
from lra_simple.configs import TASK_CONFIGS, LINEAR_PROBE_GRID, \
    LINEAR_PROBE_CONDITIONAL_GRIDS
from lra_simple.common import get_datasets, get_test_dataset, get_model_config, \
    get_transform, save_error, make_criterion, prepare_targets, batch_score, \
    summarise_score, is_better, worst_score, format_score, center_readout_spec, \
    center_features, describe_center_readout, seed_everything, save_checked_checkpoint


# =============================================================================
# FEATURE DATASET
# =============================================================================

class FeaturesDataset(Dataset):
    """Dataset wrapper for pre-extracted features."""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =============================================================================
# LINEAR MODEL
# =============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe with optional dropout."""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_features(model, data_loader, device, desc="Extracting", use_amp=True,
                     center_spec=None):
    """
    Extract features from a frozen backbone.

    By default this is the backbone's pooled output; with center_spec (from
    common.center_readout_spec) it is the feature at the image center.
    """
    model.eval()
    features = []
    labels = []

    def forward(images):
        if center_spec is not None:
            return center_features(model, images, center_spec)
        return model(images)

    for images, targets in tqdm(data_loader, desc=desc, leave=False):
        images = images.to(device)

        if use_amp:
            with torch.amp.autocast('cuda'):
                feats = forward(images)
        else:
            feats = forward(images)
        
        features.append(feats.cpu())
        labels.append(targets)
    
    # Ensure float32 (AMP may return float16)
    return torch.cat(features).float(), torch.cat(labels)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, task='pathfinder'):
    """Train linear probe for one epoch."""
    model.train()
    total_loss = 0
    score = 0
    total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = prepare_targets(task, labels.to(device))

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        score += batch_score(task, outputs, labels)
        total += labels.size(0)

    return total_loss / len(train_loader), summarise_score(task, score, total)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, task='pathfinder'):
    """Evaluate linear probe."""
    model.eval()
    total_loss = 0
    score = 0
    total = 0

    for features, labels in data_loader:
        features = features.to(device)
        labels = prepare_targets(task, labels.to(device))

        outputs = model(features)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        score += batch_score(task, outputs, labels)
        total += labels.size(0)

    return total_loss / len(data_loader), summarise_score(task, score, total)


# =============================================================================
# LINEAR PROBING
# =============================================================================

def run_model_grid_search(
    model_name,
    task,
    data_dir,
    train_split=None,
    epochs=30,
    gpu=0,
    output_dir='results',
    log_file=None,
    csv_file=None,
    num_workers=4,
    extract_batch_size=64,
    grid=None,
    conditional_grids=None,
    dropout_rate=0.0,
    verbose=False,
    seed=0
):
    """
    Run grid search for a single model on a specific task.
    
    Features are extracted once and reused across all grid configurations.
    Each configuration starts from the same seeded state.
    """
    import sys
    
    def log(msg, override=False):
        """Print with timestamp and flush immediately (only if verbose)."""
        if verbose or override:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            sys.stdout.flush()
    
    task_config = TASK_CONFIGS[task]
    train_split = train_split or task_config['default_train_split']
    grid = grid or LINEAR_PROBE_GRID
    conditional_grids = conditional_grids or LINEAR_PROBE_CONDITIONAL_GRIDS
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Always print the header
    print(f"\n{'='*70}")
    print(f"LINEAR PROBE GRID SEARCH: {model_name}")
    print(f"Task: {task}, Train split: {train_split}, GPU: {gpu}")
    print(f"{'='*70}")
    if verbose:
        sys.stdout.flush()

    # Create lock for log_file
    json_lock = FileLock(log_file + ".lock")
    csv_lock = FileLock(csv_file + ".lock")

    # Check if already done successfully (skip if it has a real val score)
    log("Checking for existing results...")
    if log_file and os.path.exists(log_file):
        with json_lock:
            try:
                with open(log_file, 'r') as f:
                    results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                results = {}
            if model_name in results:
                existing = results[model_name].get('best_val_acc')
                if existing is not None and is_better(task, existing, worst_score(task)):
                    log(f"SKIP: {model_name} already completed (val={format_score(task, existing)})", override=True)
                    return None
                else:
                    log(f"RETRY: {model_name} previously failed (no val score)", override=True)

    # Get model config for transform and metadata
    log("Getting model config...")
    model_cfg = get_model_config(model_name)
    log(f"Model input size: {model_cfg['input_size']}, ImageNet default: {model_cfg['is_imagenet_default']}", override=True)
    
    # Load model
    log("Loading pretrained model from TIMM...")
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    log("Moving model to device...")
    backbone = backbone.to(device)
    backbone.eval()
    log("Model loaded successfully")

    # Center-readout tasks read the center feature instead of the pooled output
    readout = {}
    if task_config.get('readout') == 'center':
        center_spec, _ = center_readout_spec(backbone, model_cfg['input_size'])
        readout = {'center_spec': center_spec}
        log(f"Center readout: {describe_center_readout(center_spec)}", override=True)

    # Load datasets
    log("Creating transform...")
    resize_interp = task_config.get('resize_interpolation')
    transform = get_transform(model_name, interpolation=resize_interp)
    log("Loading datasets...")
    train_dataset, val_dataset, train_split_test_dataset = get_datasets(task, data_dir, train_split, transform)
    log(f"Datasets loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(train_split_test_dataset)}", override=True)
    
    log("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=extract_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=extract_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    log("Data loaders created")
    
    # Extract features
    log("Extracting train features...")
    train_features, train_labels = extract_features(
        backbone, train_loader, device, "Train features", use_amp=True, **readout
    )
    log(f"Train features extracted: {train_features.shape}", override=True)
    
    log("Extracting val features...")
    val_features, val_labels = extract_features(
        backbone, val_loader, device, "Val features", use_amp=True, **readout
    )
    log(f"Val features extracted: {val_features.shape}", override=True)
    
    # Extract test features for all splits
    # Reuse the test dataset from initial split when test_split == train_split
    test_features = {}
    for test_split in task_config['test_splits']:
        log(f"Extracting test features for split '{test_split}'...")
        
        if test_split == train_split:
            # Use the test dataset from the initial train/val/test split
            test_dataset = train_split_test_dataset
            log(f"  (reusing test set from train_split to avoid data leakage)", override=True)
        else:
            # Different split, fetch separately
            test_dataset = get_test_dataset(task, data_dir, test_split, transform)
        
        test_loader = DataLoader(
            test_dataset, batch_size=extract_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_feats, test_labs = extract_features(
            backbone, test_loader, device, f"Test features ({test_split})", use_amp=True,
            **readout
        )
        test_features[test_split] = (test_feats, test_labs)
        log(f"Test features ({test_split}) extracted: {test_feats.shape}")
    
    feature_dim = train_features.shape[-1]
    log(f"Feature extraction complete. Feature dim: {feature_dim}", override=True)
    
    # Clean up backbone
    log("Cleaning up backbone model...")
    del backbone
    torch.cuda.empty_cache()
    log("Backbone cleaned up")

    # Generate configs
    configs = grid_search_with_conditionals(grid, conditional_grids)
    log(f"Starting grid search with {len(configs)} configurations", override=True)

    best_val_acc = worst_score(task)
    best_result = {}
    best_probe = None
    configs_failed = 0

    for idx, config in enumerate(configs):
        log(f"Config {idx+1}/{len(configs)}: lr={config['learning_rate']:.0e}, wd={config['weight_decay']:.0e}", override=True)

        # Same start for every config: probe initialization and batch order
        seed_everything(seed)

        try:
            # Create feature datasets
            train_feat_dataset = FeaturesDataset(train_features, train_labels)
            val_feat_dataset = FeaturesDataset(val_features, val_labels)
            
            batch_size = config.get('batch_size', 128)
            
            train_feat_loader = DataLoader(
                train_feat_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False
            )
            val_feat_loader = DataLoader(
                val_feat_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False
            )
            
            # Create fresh probe
            probe = LinearProbe(feature_dim, task_config['num_classes'], dropout_rate)
            probe = probe.to(device)
            criterion = make_criterion(task)

            optimizer = torch.optim.AdamW(
                probe.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
            
            # Train
            config_best_val_acc = worst_score(task)
            config_best_epoch = 0

            for epoch in range(epochs):
                train_loss, train_acc = train_epoch(
                    probe, train_feat_loader, criterion, optimizer, device, task
                )
                val_loss, val_acc = evaluate(
                    probe, val_feat_loader, criterion, device, task
                )

                if is_better(task, val_acc, config_best_val_acc):
                    config_best_val_acc = val_acc
                    config_best_epoch = epoch

                    if is_better(task, val_acc, best_val_acc):
                        best_val_acc = val_acc
                        # Clone tensors: state_dict() shares storage with the live
                        # parameters, so a shallow copy would track later updates
                        best_probe = {k: v.detach().clone() for k, v in probe.state_dict().items()}
                        best_result = {
                            'best_val_acc': best_val_acc,
                            'best_epoch': config_best_epoch,
                            'config': {
                                'learning_rate': config['learning_rate'],
                                'weight_decay': config['weight_decay'],
                            }
                        }
            
            log(f"  Config {idx+1} done: best_epoch={config_best_epoch+1}, val={format_score(task, config_best_val_acc)}", override=True)
            
            del probe
            torch.cuda.empty_cache()

        except Exception as e:
            configs_failed += 1

            log(f"  ERROR in config {idx+1}: {e}", override=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            continue

    # Every config failed: raise so main() records the error and nothing is
    # logged as a result (a later run retries the model)
    if configs_failed == len(configs):
        raise RuntimeError(f"All {len(configs)} grid configs failed for {model_name}; "
                           f"see the per-config tracebacks above")

    # Cross-split evaluation with best probe
    test_accs = {}
    if best_probe is not None:
        log(f"Cross-split evaluation with best probe (val={format_score(task, best_val_acc)})", override=True)

        probe = LinearProbe(feature_dim, task_config['num_classes'], dropout_rate)
        probe.load_state_dict(best_probe)
        probe = probe.to(device)
        criterion = make_criterion(task)

        for test_split in task_config['test_splits']:
            test_feats, test_labs = test_features[test_split]
            test_feat_dataset = FeaturesDataset(test_feats, test_labs)
            test_feat_loader = DataLoader(
                test_feat_dataset, batch_size=128, shuffle=False,
                num_workers=0, pin_memory=False
            )
            
            _, test_acc = evaluate(probe, test_feat_loader, criterion, device, task)
            test_accs[test_split] = test_acc
            log(f"  Test {test_split}: {format_score(task, test_acc)}")
        
        best_result['test_acc'] = test_accs
        
        # Save best probe
        log("Saving best probe...")
        output_path = Path(output_dir) / task / model_name / f'linear_probe_{train_split}'
        output_path.mkdir(parents=True, exist_ok=True)

        # Reload check: the saved file must reproduce the selected probe's
        # val score (guards against saving weights other than the ones chosen)
        val_check_loader = DataLoader(
            FeaturesDataset(val_features, val_labels), batch_size=128, shuffle=False,
            num_workers=0, pin_memory=False
        )
        def score_reloaded(state):
            reloaded = LinearProbe(feature_dim, task_config['num_classes'], dropout_rate)
            reloaded.load_state_dict(state)
            return evaluate(reloaded.to(device), val_check_loader, criterion, device, task)[1]
        reloaded_val = save_checked_checkpoint(
            best_probe, output_path / 'best_probe.pth', task, best_val_acc,
            len(val_labels), score_reloaded)
        log(f"Reload check passed (val {format_score(task, reloaded_val)})", override=True)
        
        with open(output_path / 'config.json', 'w') as f:
            json.dump({
                'model_name': model_name,
                'task': task,
                'train_split': train_split,
                'best_val_acc': best_val_acc,
                'test_acc': test_accs,
                'seed': seed,
                **best_result.get('config', {})
            }, f, indent=2)
        log("Best probe saved")

    # Package final result
    final_result = {
        'best_val_acc': best_val_acc,
        'best_result': best_result,
        'feature_dim': feature_dim,
        'input_size': model_cfg['input_size'],
        'is_imagenet_default': model_cfg['is_imagenet_default'],
        'seed': seed,
    }

    # Save to JSON log file
    log("Saving to JSON log file...", override=True)
    with json_lock:
        if log_file:
            results = {}
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        results = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    results = {}
            
            results[model_name] = final_result
            
            for attempt in range(5):
                try:
                    with open(log_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    log(f"JSON saved successfully")
                    break
                except Exception as e:
                    log(f"JSON save attempt {attempt+1} failed: {e}")

    # Save to CSV (load, update, write to handle overwrites)
    log("Saving to CSV file...", override=True)
    with csv_lock:
        if csv_file:
            import pandas as pd
            
            # Build new row as dict
            cfg = best_result.get('config', {})
            new_row = {
                'model_name': model_name,
                'task': task,
                'train_split': train_split,
                'best_val_acc': best_val_acc,
            }
            for s in task_config['test_splits']:
                new_row[f'test_acc_{s}'] = test_accs.get(s, 0)
            new_row['learning_rate'] = cfg.get('learning_rate', 0)
            new_row['weight_decay'] = cfg.get('weight_decay', 0)
            new_row['feature_dim'] = feature_dim
            new_row['input_size'] = f"{model_cfg['input_size'][0]}x{model_cfg['input_size'][1]}"
            new_row['is_imagenet_default'] = model_cfg['is_imagenet_default']
            new_row['seed'] = seed
            
            # Load existing or create new dataframe
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    # Remove existing row for this model if present
                    df = df[df['model_name'] != model_name]
                except:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
            
            # Append new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Write back
            for attempt in range(5):
                try:
                    df.to_csv(csv_file, index=False)
                    log(f"CSV saved successfully")
                    break
                except Exception as e:
                    log(f"CSV save attempt {attempt+1} failed: {e}")

    log(f"COMPLETE: {model_name} - best_val={format_score(task, best_val_acc)}", override=True)
    print(f"{'='*70}")
    sys.stdout.flush()

    return final_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Linear Probing for Vision Tasks')
    
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
                        help='Single model to run')
    parser.add_argument('--models_csv', type=str, default=None,
                        help='CSV file with model names')
    parser.add_argument('--models_list', type=str, default=None,
                        help='Text file with model names (one per line)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index in model list')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index in model list (exclusive)')
    parser.add_argument('--skip_errors', action='store_true',
                        help='Skip models that previously errored and are logged')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--extract_batch_size', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Model filtering
    parser.add_argument('--skip_non_224', action='store_true',
                        help='Skip models that expect input size other than 224x224')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed applied at the start of every grid configuration '
                             '(probe initialization, batch order); default: 0')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed logging with timestamps (slower but useful for debugging)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--log_file', type=str, default=None,
                        help='JSON log file')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='CSV results file')
    parser.add_argument('--err_file', type=str, default=None,
                        help='JSON errors file')

    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = args.log_file or str(output_dir / f'{args.task}_linear_probe.json')
    csv_file = args.csv_file or str(output_dir / f'{args.task}_linear_probe.csv')
    err_file = args.err_file or str(output_dir / f'{args.task}_linear_probe_errors.json')

    # Set grid
    grid = TASK_CONFIGS[args.task].get('linear_probe_grid', LINEAR_PROBE_GRID)

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
            csv_file=csv_file,
            num_workers=args.num_workers,
            extract_batch_size=args.extract_batch_size,
            dropout_rate=args.dropout_rate,
            verbose=args.verbose,
            seed=args.seed,
            grid=grid
        )
        return

    # Batch mode
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
    if args.skip_non_224:
        print("Skipping models with non-224x224 input size")
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
        
        # Check input size if skip_non_224 is enabled
        if args.skip_non_224:
            try:
                cfg = get_model_config(model_name)
                if cfg['input_size'] != (224, 224):
                    print(f"  SKIP: Input size {cfg['input_size']} != (224, 224)")
                    sys.stdout.flush()
                    continue
            except Exception as e:
                print(f"  SKIP: Could not get model config: {e}")
                sys.stdout.flush()
                continue

        if args.skip_errors:
            with err_lock:
                if err_file and os.path.exists(err_file):
                    with open(err_file, 'r') as f:
                        errors = json.load(f)
                    if model_name in errors:
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
                csv_file=csv_file,
                num_workers=args.num_workers,
                extract_batch_size=args.extract_batch_size,
                dropout_rate=args.dropout_rate,
                verbose=args.verbose,
                seed=args.seed,
                grid=grid
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