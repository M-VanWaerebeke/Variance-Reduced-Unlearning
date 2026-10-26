#!/usr/bin/env python3
"""
generate_shadow_csv.py - Generate CSV manifest for shadow model sampling in U-LiRA MIA.

Creates a sample-model provenance matrix where each row is a data sample and each column
is a trained shadow model. Entry (i,j) = 1 if shadow model j was trained on sample i, 0 otherwise.

IMPORTANT TERMINOLOGY:
- **Shadow models**: Models used for MIA evaluation (stored in CSV). Each trained on 50% of data.
- **Base models**: Models used for unlearning (full/scratch models). NOT related to shadow CSV.

Training Strategy:
- Adaptive sampling to ensure balanced IN/OUT coverage for all samples
- Each shadow model is trained on half the dataset (|D|/2 samples)
- Alternates between IN-priority (include under-covered samples) and OUT-priority (exclude under-covered samples)
- Continues until all samples have ≥(N × shadow_multiplier) IN and OUT models
- Uses all available data (no test/val split) for maximum efficiency

Features:
- Progressive expansion: Checks existing CSV coverage, only trains additional shadow models if needed
- Shadow multiplier: Generate M×N shadow models per sample for increased cross-run variability
- Hyperparameter caching: First shadow model runs full sweep, remaining reuse cached params (~73% cost reduction)

Usage Examples:
    # Initial generation with multiplier=1
    python generate_shadow_csv.py --dataset digits --N 8 --shadow_multiplier 1

    # Later: expand to multiplier=5 (only trains 32 additional shadow models, reuses existing 8)
    python generate_shadow_csv.py --dataset digits --N 8 --shadow_multiplier 5

    # Check coverage without training (will report and return if sufficient)
    python generate_shadow_csv.py --dataset digits --N 8 --shadow_multiplier 10
"""
from __future__ import annotations
import argparse
import os
import json
import sys
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure float64
torch.set_default_dtype(torch.float64)

# Detect if we're in a log file (not a terminal) to reduce tqdm output
IN_LOGFILE = not sys.stdout.isatty()

# Helper function to convert numpy types to JSON-serializable Python types
def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Import from existing codebase
from digit_train import (
    set_seed,
    get_data_loaders,
    build_model,
    train as train_model,
    Loaders,
)
from pretrain_model import MNIST_HP_CONFIGS, MNIST_DISCOUNT, MNIST_EPOCHS


@dataclass
class ShadowManifest:
    """Container for shadow model manifest data."""
    csv_path: str
    metadata_path: str
    n_models: int
    n_samples: int
    target_N: int
    dataset_name: str
    seed: int
    model_type: str
    weight_decay: float


def compute_coverage(csv_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IN and OUT coverage counts for each sample from CSV.

    Args:
        csv_df: DataFrame with 'sample_id' column and model_* columns (1=IN, 0=OUT)

    Returns:
        Tuple of (in_counts, out_counts) arrays of shape (n_samples,)
    """
    n_samples = len(csv_df)
    model_cols = [col for col in csv_df.columns if col.startswith('model_')]

    in_counts = np.zeros(n_samples, dtype=int)
    out_counts = np.zeros(n_samples, dtype=int)

    for col in model_cols:
        in_counts += csv_df[col].values
        out_counts += (1 - csv_df[col].values)

    return in_counts, out_counts


def check_existing_manifest(
    dataset_name: str,
    target_N: int,
    shadow_multiplier: int,
    model_type: str,
    weight_decay: float,
    output_dir: str,
) -> Tuple[bool, str, str, int, int]:
    """
    Check if existing shadow CSV has sufficient coverage.

    Returns:
        Tuple of (sufficient, csv_path, metadata_path, min_in, min_out)
        - sufficient: True if existing coverage >= required
        - csv_path: Path to existing CSV (or where it would be)
        - metadata_path: Path to existing metadata (or where it would be)
        - min_in: Minimum IN coverage per sample (-1 if CSV doesn't exist)
        - min_out: Minimum OUT coverage per sample (-1 if CSV doesn't exist)
    """
    wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
    csv_filename = f"{dataset_name}_shadow_manifest_N{target_N}_{model_type}_wd{wd_str}.csv"
    metadata_filename = f"{dataset_name}_shadow_metadata_N{target_N}_{model_type}_wd{wd_str}.json"

    csv_path = os.path.join(output_dir, csv_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    if not os.path.exists(csv_path):
        return False, csv_path, metadata_path, -1, -1

    # Load and check coverage
    csv_df = pd.read_csv(csv_path)
    in_counts, out_counts = compute_coverage(csv_df)

    min_in = int(in_counts.min())
    min_out = int(out_counts.min())
    required = target_N * shadow_multiplier

    sufficient = (min_in >= required and min_out >= required)

    return sufficient, csv_path, metadata_path, min_in, min_out


def circular_range(start: int, end: int, size: int) -> List[int]:
    """
    Generate a circular range of indices [start, end) with wraparound at size.

    Args:
        start: Starting index (inclusive)
        end: Ending index (exclusive)
        size: Total size for wraparound

    Returns:
        List of indices with wraparound

    Example:
        circular_range(8, 13, 10) -> [8, 9, 0, 1, 2]
    """
    if end > size:
        # Wraparound case (end exceeds size)
        return list(range(start, size)) + list(range(0, end - size))
    elif end <= start:
        # Wraparound case (end is modulo-ed)
        return list(range(start, size)) + list(range(0, end))
    else:
        # Normal case
        return list(range(start, end))


def check_coverage_complete(csv_df: pd.DataFrame, target_N: int) -> Tuple[bool, dict]:
    """
    Check if all samples have at least target_N IN models and target_N OUT models.

    Args:
        csv_df: DataFrame with columns [sample_id, model_0, model_1, ...]
        target_N: Target number of IN and OUT models per sample

    Returns:
        Tuple of (is_complete, stats_dict)
    """
    model_cols = [col for col in csv_df.columns if col.startswith('model_')]
    n_models = len(model_cols)

    min_in = float('inf')
    min_out = float('inf')
    max_in = 0
    max_out = 0

    for idx, row in csv_df.iterrows():
        in_count = row[model_cols].sum()
        out_count = n_models - in_count

        min_in = min(min_in, in_count)
        min_out = min(min_out, out_count)
        max_in = max(max_in, in_count)
        max_out = max(max_out, out_count)

        if in_count < target_N or out_count < target_N:
            stats = {
                'complete': False,
                'min_in': int(min_in),
                'min_out': int(min_out),
                'max_in': int(max_in),
                'max_out': int(max_out),
                'target_N': target_N,
            }
            return False, stats

    stats = {
        'complete': True,
        'min_in': int(min_in),
        'min_out': int(min_out),
        'max_in': int(max_in),
        'max_out': int(max_out),
        'target_N': target_N,
    }
    return True, stats


def train_shadow_base_model(
    train_indices: List[int],
    full_dataset,
    loaders_template: Loaders,
    device: torch.device,
    model_type: str,
    weight_decay: float,
    seed: int,
    model_idx: int,
    save_path: str,
    dataset_name: str = "digits",
    cached_hyperparams: dict = None,
    batch_size: int = 64,
) -> dict:
    """
    Train a single shadow base model on specified training indices.

    Uses the SAME hyperparameter sweep as main model training to ensure
    consistency between shadow models and attacked models.

    Args:
        train_indices: List of dataset indices to train on
        full_dataset: Complete dataset to subset from
        loaders_template: Template loaders for val/test (train will be replaced)
        device: Device to train on
        model_type: "LR", "NN", or "CNN"
        weight_decay: L2 regularization strength
        seed: Random seed
        model_idx: Model index for logging
        save_path: Path to save trained model
        dataset_name: Dataset name ("digits" or "mnist") for algo selection
        cached_hyperparams: Optional dict with cached hyperparameters from previous sweep.
                           If provided, skips hyperparameter sweep and uses cached params.
                           Format: {'lr': float, 'optimizer': str, 'epochs': int}

    Returns:
        dict: Best hyperparameters found/used. Format: {'lr': float, 'optimizer': str, 'epochs': int}
    """
    # Create subset dataset
    train_subset = Subset(full_dataset, train_indices)

    # Create loaders for this subset
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    full_train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)

    shadow_loaders = Loaders(
        train=train_loader,
        full_train=full_train_loader,
        val=loaders_template.val,
        test=loaders_template.test,
        n_features=loaders_template.n_features,
        n_classes=loaders_template.n_classes,
        n_train=len(train_subset),
    )

    print(f"  Training shadow model {model_idx} on {len(train_indices)} samples...")

    # Use SAME hyperparameter sweep as main model training (gradient_representation.py)
    # This ensures shadow models and attacked models have consistent training
    model_seed = seed + model_idx  # Unique seed per model

    if dataset_name == "mnist":
        # Grid sweep matching test_vru_retain_loss.py
        # Track best model across HP configs (save_folder="" disables disk tracking)
        best_loss = float('inf')
        best_state = None
        for lr_raw, optimizer, algo, scheduler in MNIST_HP_CONFIGS:
            lr_eff = lr_raw / 10 if model_type == "NN" else lr_raw
            model = build_model(model_type, loaders_template.n_features, loaders_template.n_classes)
            model = model.to(device=device, dtype=torch.float64)
            result = train_model(
                model,
                shadow_loaders,
                device,
                algo=algo,
                seed=model_seed,
                epochs=MNIST_EPOCHS,
                lr=lr_eff,
                scheduler=scheduler,
                optimizer=optimizer,
                discount_factor=MNIST_DISCOUNT,
                momentum=0.0,
                weight_decay=weight_decay,
                log_interval=0,
                save_folder="",
                model_type=model_type,
            )
            final_loss = min(result["train_metric"]) if result["train_metric"] else float('inf')
            if final_loss < best_loss:
                best_loss = final_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best model state
        model.load_state_dict(best_state)

        hyperparams_result = {}
    else:
        # Digits: Same sweep as main training
        if cached_hyperparams is None:
            # Run full hyperparameter sweep
            print(f"    Running hyperparameter sweep for model {model_idx}...")
            best_config = None
            best_loss = float('inf')

            for pretrain_lr in [0.1, 0.03, 0.01, 0.003]:
                lr_eff = pretrain_lr if (model_type == "LR") else pretrain_lr / 10
                for optimizer in ["sgd", "adam"]:
                    model = build_model(model_type, loaders_template.n_features, loaders_template.n_classes)
                    model = model.to(device=device, dtype=torch.float64)
                    result = train_model(
                        model,
                        shadow_loaders,
                        device,
                        algo="sgd",
                        seed=model_seed,
                        epochs=500,
                        lr=lr_eff,
                        scheduler="step",
                        optimizer=optimizer,
                        discount_factor=0.3,
                        momentum=0.0,
                        weight_decay=weight_decay,
                        log_interval=0,
                        save_folder="",
                        model_type=model_type,
                    )
                    final_loss = result["train_metric"][-1] if result["train_metric"] else float('inf')
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_config = (lr_eff, optimizer)

            print(f"    Best config: lr={best_config[0]}, optimizer={best_config[1]}")
        else:
            # Use cached hyperparameters (skip sweep)
            print(f"    Using cached hyperparameters: lr={cached_hyperparams['lr']}, optimizer={cached_hyperparams['optimizer']}")
            best_config = (cached_hyperparams['lr'], cached_hyperparams['optimizer'])

        # Train with best config
        model = build_model(model_type, loaders_template.n_features, loaders_template.n_classes)
        model = model.to(device=device, dtype=torch.float64)
        _ = train_model(
            model,
            shadow_loaders,
            device,
            algo="sgd",
            seed=model_seed,
            epochs=500,
            lr=best_config[0],
            scheduler="step",
            optimizer=best_config[1],
            discount_factor=0.3,
            momentum=0.0,
            weight_decay=weight_decay,
            log_interval=0,
            save_folder="",
            model_type=model_type,
        )

        # Return hyperparameters for caching
        hyperparams_result = {
            'lr': best_config[0],
            'optimizer': best_config[1],
            'epochs': 500,
        }

    # Save model manually
    torch.save(model.state_dict(), save_path)
    print(f"  Saved model to {save_path}")

    return hyperparams_result


def _plan_one_model(in_counts, out_counts, n_samples, train_size, rng, effective_target):
    """Plan training indices for one shadow model using adaptive sampling.

    Returns sorted train_indices, or None if coverage is already complete.
    """
    min_in_count = in_counts.min()
    min_out_count = out_counts.min()

    if min_in_count >= effective_target and min_out_count >= effective_target:
        return None

    if min_in_count <= min_out_count:
        samples_needing_in = np.where(in_counts == min_in_count)[0]
        priority_sample = rng.choice(samples_needing_in)
        other_samples = [i for i in range(n_samples) if i != priority_sample]
        rng.shuffle(other_samples)
        train_indices = [priority_sample] + other_samples[:train_size - 1]
    else:
        samples_needing_out = np.where(out_counts == min_out_count)[0]
        excluded_sample = rng.choice(samples_needing_out)
        available_samples = [i for i in range(n_samples) if i != excluded_sample]
        rng.shuffle(available_samples)
        train_indices = available_samples[:train_size]

    return sorted(train_indices)


def _train_shadow_worker(spec):
    """Worker for parallel shadow model training.

    Self-contained: loads dataset internally to avoid pickling large objects
    across process boundaries (spawn context).
    """
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    model_idx = spec['model_idx']
    save_path = spec['save_path']

    if os.path.exists(save_path):
        return {'model_idx': model_idx, 'hyperparams': None, 'existed': True}

    # Load dataset in worker (cheap for MNIST/digits)
    datasets_list, loaders = get_data_loaders(
        batch_size=spec['batch_size'],
        val_size=0.0,
        test_size=0.0,
        seed=spec['init_seed'],
        standardize=True,
        dataset=spec['dataset_name'],
    )
    full_dataset = datasets_list[0]

    hyperparams = train_shadow_base_model(
        train_indices=spec['train_indices'],
        full_dataset=full_dataset,
        loaders_template=loaders,
        device=torch.device('cpu'),
        model_type=spec['model_type'],
        weight_decay=spec['weight_decay'],
        seed=spec['init_seed'],
        model_idx=model_idx,
        save_path=save_path,
        dataset_name=spec['dataset_name'],
        cached_hyperparams=spec['cached_hyperparams'],
        batch_size=spec['batch_size'],
    )

    return {'model_idx': model_idx, 'hyperparams': hyperparams, 'existed': False}


def generate_shadow_csv(
    dataset_name: str,
    target_N: int,
    shadow_multiplier: int = 1,
    model_type: str = "LR",
    weight_decay: float = 0.1,
    device: torch.device = None,
    output_dir: str = "shadow_manifests",
    init_seed: int = 42,
    parallel_jobs: int = 1,
    batch_size: int = 64,
) -> ShadowManifest:
    """
    Generate CSV manifest with shadow model provenance (progressive expansion).

    This function implements progressive expansion: if an existing CSV has sufficient
    coverage (>= target_N * shadow_multiplier IN/OUT models per sample), it returns
    the existing manifest without training. Otherwise, it trains additional shadow
    models and appends them to the existing CSV.

    Args:
        dataset_name: "digits" or "mnist"
        target_N: Target number of IN/OUT models per sample (base target)
        shadow_multiplier: Multiplier for shadow pool size. Generates multiplier*target_N
                          IN/OUT models per sample for increased cross-run variability.
        model_type: "LR", "NN", or "CNN"
        weight_decay: L2 regularization
        device: Training device
        output_dir: Directory to save CSV and metadata
        init_seed: Random seed for model initialization only (default: 42)
        parallel_jobs: Number of parallel workers for shadow model training (default: 1)

    Returns:
        ShadowManifest with paths and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    effective_target = target_N * shadow_multiplier

    # Check if existing CSV has sufficient coverage
    print(f"Checking for existing shadow CSV...")
    sufficient, csv_path, metadata_path, min_in, min_out = check_existing_manifest(
        dataset_name=dataset_name,
        target_N=target_N,
        shadow_multiplier=shadow_multiplier,
        model_type=model_type,
        weight_decay=weight_decay,
        output_dir=output_dir,
    )

    if sufficient:
        print(f"✓ Found existing shadow CSV with sufficient coverage:")
        print(f"  Path: {csv_path}")
        print(f"  Min IN per sample:  {min_in} (required: {effective_target})")
        print(f"  Min OUT per sample: {min_out} (required: {effective_target})")
        print(f"  Reusing existing manifest (no training needed)\n")

        # Load metadata to return full manifest
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return ShadowManifest(
            csv_path=csv_path,
            metadata_path=metadata_path,
            n_models=len(metadata['models']),
            n_samples=metadata['n_samples'],
            target_N=target_N,
            dataset_name=dataset_name,
            seed=init_seed,
            model_type=model_type,
            weight_decay=weight_decay,
        )

    # If we reach here, need to generate (or expand) shadow models
    if min_in == -1:
        print(f"No existing CSV found. Generating new shadow manifest...")
    else:
        print(f"Existing CSV has insufficient coverage:")
        print(f"  Current: min_in={min_in}, min_out={min_out}")
        print(f"  Required: {effective_target}")
        print(f"  Expanding shadow pool with additional models...\n")

    set_seed(init_seed)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, f"{dataset_name}_{model_type}_shadow_base_models")
    os.makedirs(models_dir, exist_ok=True)

    # Load dataset (use all data - no test/val split needed)
    print(f"Loading {dataset_name} dataset...")
    datasets_list, loaders = get_data_loaders(
        batch_size=batch_size,
        val_size=0.0,
        test_size=0.0,
        seed=init_seed,
        standardize=True,
        dataset=dataset_name,
    )
    train_dataset = datasets_list[0]
    n_samples = len(train_dataset)

    print(f"Dataset size: {n_samples} training samples")
    print(f"Target: {target_N} IN and {target_N} OUT models per sample (base)")
    if shadow_multiplier > 1:
        print(f"Shadow multiplier: {shadow_multiplier}x → Effective target: {effective_target} IN/OUT per sample")
    print(f"Strategy: Adaptive sampling to ensure balanced IN/OUT coverage")

    # Load existing CSV and metadata if available, otherwise initialize new
    existing_csv = os.path.exists(csv_path)
    if existing_csv:
        print(f"\nLoading existing CSV to expand: {csv_path}")
        csv_df = pd.read_csv(csv_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Update metadata with new target (in case shadow_multiplier changed)
        metadata['shadow_multiplier'] = shadow_multiplier
        metadata['effective_target'] = effective_target

        # Compute existing coverage
        in_counts, out_counts = compute_coverage(csv_df)
        print(f"Existing coverage: min_in={int(in_counts.min())}, min_out={int(out_counts.min())}")

        # Next model index = number of existing model columns
        model_idx = len([col for col in csv_df.columns if col.startswith('model_')])
        print(f"Continuing from model index {model_idx}\n")
    else:
        print(f"\nInitializing new CSV manifest")
        # Initialize CSV with sample_id column
        csv_data = {'sample_id': list(range(n_samples))}
        csv_df = pd.DataFrame(csv_data)

        # Metadata for tracking model paths
        metadata = {
            'dataset': dataset_name,
            'n_samples': n_samples,
            'target_N': target_N,
            'shadow_multiplier': shadow_multiplier,
            'effective_target': effective_target,
            'init_seed': init_seed,
            'model_type': model_type,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'models': [],
            'cached_hyperparams': None,  # Will be set after first model
        }

        # Initialize coverage tracking
        in_counts = np.zeros(n_samples, dtype=int)
        out_counts = np.zeros(n_samples, dtype=int)
        model_idx = 0

    # Training loop: adaptive sampling with parallel batch execution
    # Strategy:
    #   1. Train first model sequentially (hyperparameter sweep)
    #   2. Batch remaining models and train in parallel with ProcessPoolExecutor
    #   Planning uses temporary coverage copies so batch decisions are consistent.
    train_size = n_samples // 2  # Train on half the data

    # RNG for random sampling (separate from model init seed for reproducibility)
    rng = np.random.default_rng(init_seed + 77777)

    # Estimate: need at least 2*effective_target models for full coverage
    estimated_models = 2 * effective_target + 10  # Small buffer
    print(f"\nEstimated models needed: ~{estimated_models}")
    print(f"Training models with T={train_size} samples each...")
    if parallel_jobs > 1:
        print(f"Using {parallel_jobs} parallel workers")
    print()

    if shadow_multiplier > 1:
        print(f"Note: First model will run full hyperparameter sweep.")
        print(f"      Remaining models will use cached hyperparameters for efficiency.\n")

    pbar = tqdm(desc="Training shadow models", unit="model",
                mininterval=float('inf') if IN_LOGFILE else 0.1)

    # Phase 1: Train first model sequentially for hyperparameter sweep (if needed)
    if model_idx == 0:
        indices = _plan_one_model(in_counts, out_counts, n_samples, train_size, rng, effective_target)
        if indices is not None:
            wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
            model_path = os.path.join(models_dir, f"base_model_idx{model_idx:04d}_wd{wd_str}.pt")

            if not os.path.exists(model_path):
                hyperparams = train_shadow_base_model(
                    train_indices=indices,
                    full_dataset=train_dataset,
                    loaders_template=loaders,
                    device=device,
                    model_type=model_type,
                    weight_decay=weight_decay,
                    seed=init_seed,
                    model_idx=0,
                    save_path=model_path,
                    dataset_name=dataset_name,
                    cached_hyperparams=None,
                    batch_size=batch_size,
                )
                metadata['cached_hyperparams'] = hyperparams
                print(f"  ✓ Cached hyperparameters from first model: {hyperparams}")
            else:
                print(f"  Model 0 already exists at {model_path}, skipping training")

            # Update coverage
            train_set = set(indices)
            for i in range(n_samples):
                if i in train_set:
                    in_counts[i] += 1
                else:
                    out_counts[i] += 1

            csv_df['model_0'] = 0
            csv_df.loc[indices, 'model_0'] = 1
            metadata['models'].append({
                'model_id': 0,
                'path': model_path,
                'train_indices': indices,
                'n_samples_trained': len(indices),
                'used_sweep': True,
                'hyperparams': metadata.get('cached_hyperparams'),
            })
            model_idx = 1
            pbar.update(1)

    # Phase 2: Parallel batch training for remaining models
    cached_hp = metadata.get('cached_hyperparams')

    while True:
        min_in = int(in_counts.min())
        min_out = int(out_counts.min())

        if min_in >= effective_target and min_out >= effective_target:
            print(f"\n✓ Coverage complete after {model_idx} models!")
            print(f"  Min IN per sample: {min_in} (target: {effective_target})")
            print(f"  Max IN per sample: {int(in_counts.max())}")
            print(f"  Min OUT per sample: {min_out} (target: {effective_target})")
            print(f"  Max OUT per sample: {int(out_counts.max())}")
            break

        if model_idx >= 10 * effective_target:
            print(f"\n⚠ Warning: Reached safety limit ({10 * effective_target} models) without full coverage.")
            print(f"  Current min IN: {min_in}, min OUT: {min_out}, target: {effective_target}")
            break

        # Plan a batch of models using temporary coverage
        batch = []
        plan_in = in_counts.copy()
        plan_out = out_counts.copy()
        wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
        batch_limit = max(1, parallel_jobs)

        for _ in range(batch_limit):
            if model_idx + len(batch) >= 10 * effective_target:
                break
            indices = _plan_one_model(plan_in, plan_out, n_samples, train_size, rng, effective_target)
            if indices is None:
                break

            midx = model_idx + len(batch)
            model_path = os.path.join(models_dir, f"base_model_idx{midx:04d}_wd{wd_str}.pt")

            batch.append({
                'model_idx': midx,
                'train_indices': indices,
                'save_path': model_path,
                'init_seed': init_seed,
                'dataset_name': dataset_name,
                'model_type': model_type,
                'weight_decay': weight_decay,
                'cached_hyperparams': cached_hp,
                'batch_size': batch_size,
            })

            # Update planning coverage so next model in batch sees updated state
            train_set = set(indices)
            for i in range(n_samples):
                if i in train_set:
                    plan_in[i] += 1
                else:
                    plan_out[i] += 1

        if not batch:
            break

        # Execute batch
        if parallel_jobs > 1 and len(batch) > 1:
            mp_context = multiprocessing.get_context('spawn')
            n_workers = min(parallel_jobs, len(batch))
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                futures = {executor.submit(_train_shadow_worker, spec): spec for spec in batch}
                for future in as_completed(futures):
                    result = future.result()
                    if result.get('hyperparams') and cached_hp is None:
                        cached_hp = result['hyperparams']
                        metadata['cached_hyperparams'] = cached_hp
        else:
            # Sequential fallback
            for spec in batch:
                if not os.path.exists(spec['save_path']):
                    hp = train_shadow_base_model(
                        train_indices=spec['train_indices'],
                        full_dataset=train_dataset,
                        loaders_template=loaders,
                        device=device,
                        model_type=model_type,
                        weight_decay=weight_decay,
                        seed=init_seed,
                        model_idx=spec['model_idx'],
                        save_path=spec['save_path'],
                        dataset_name=dataset_name,
                        cached_hyperparams=cached_hp,
                        batch_size=batch_size,
                    )
                    if cached_hp is None:
                        cached_hp = hp
                        metadata['cached_hyperparams'] = cached_hp
                else:
                    print(f"  Model {spec['model_idx']} already exists, skipping")

        # Update actual coverage, CSV, and metadata
        for spec in batch:
            indices = spec['train_indices']
            midx = spec['model_idx']
            train_set = set(indices)
            for i in range(n_samples):
                if i in train_set:
                    in_counts[i] += 1
                else:
                    out_counts[i] += 1

            col_name = f'model_{midx}'
            csv_df[col_name] = 0
            csv_df.loc[indices, col_name] = 1
            metadata['models'].append({
                'model_id': midx,
                'path': spec['save_path'],
                'train_indices': indices,
                'n_samples_trained': len(indices),
                'used_sweep': False,
                'hyperparams': cached_hp,
            })

        model_idx += len(batch)
        pbar.update(len(batch))

        pbar.set_postfix({
            'min_in': int(in_counts.min()),
            'min_out': int(out_counts.min()),
            'target': effective_target,
            'models': model_idx,
        })

    pbar.close()

    # Save CSV
    wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
    csv_filename = f"{dataset_name}_shadow_manifest_N{target_N}_{model_type}_wd{wd_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    csv_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV manifest to: {csv_path}")

    # Save metadata
    metadata_filename = f"{dataset_name}_shadow_metadata_N{target_N}_{model_type}_wd{wd_str}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, 'w') as f:
        # Convert all numpy types to Python native types for JSON serialization
        metadata_serializable = convert_to_json_serializable(metadata)
        json.dump(metadata_serializable, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")

    return ShadowManifest(
        csv_path=csv_path,
        metadata_path=metadata_path,
        n_models=model_idx,
        n_samples=n_samples,
        target_N=target_N,
        dataset_name=dataset_name,
        seed=init_seed,
        model_type=model_type,
        weight_decay=weight_decay,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate shadow model CSV manifest for U-LiRA MIA")
    p.add_argument("--dataset", type=str, default="digits", choices=["digits", "mnist", "breast_cancer"],
                   help="Dataset to use")
    p.add_argument("--N", type=int, default=8,
                   help="Target number of IN and OUT models per sample")
    p.add_argument("--shadow_multiplier", type=int, default=1,
                   help="Multiplier for shadow model pool size (default: 1). "
                        "With multiplier M, generates M*N IN/OUT models per sample for increased variability.")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN", "CNN"],
                   help="Model architecture: LR, NN, or CNN")
    p.add_argument("--weight_decay", type=float, default=0.1,
                   help="L2 regularization strength")
    p.add_argument("--init_seed", type=int, default=42,
                   help="Random seed for model initialization (does not affect data splits)")
    p.add_argument("--no_cuda", action="store_true",
                   help="Force CPU even if CUDA available")
    p.add_argument("--output_dir", type=str, default="shadow_manifests",
                   help="Output directory for CSV and models")
    p.add_argument("--parallel_jobs", type=int, default=1,
                   help="Number of parallel workers for shadow model training (default: 1)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Shadow training batch size; match the target model")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}\n")

    manifest = generate_shadow_csv(
        dataset_name=args.dataset,
        target_N=args.N,
        shadow_multiplier=args.shadow_multiplier,
        model_type=args.model,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=args.output_dir,
        init_seed=args.init_seed,
        parallel_jobs=args.parallel_jobs,
        batch_size=args.batch_size,
    )

    print(f"\n{'='*60}")
    print("Shadow CSV Generation Complete!")
    print(f"{'='*60}")
    print(f"CSV Path: {manifest.csv_path}")
    print(f"Metadata Path: {manifest.metadata_path}")
    print(f"Total Models: {manifest.n_models}")
    print(f"Total Samples: {manifest.n_samples}")
    print(f"Target N: {manifest.target_N}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
