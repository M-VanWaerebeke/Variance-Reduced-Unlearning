#!/usr/bin/env python3
"""
shadow_utils.py - Utility functions for CSV-based shadow model sampling in U-LiRA MIA.

Provides helper functions for:
- Loading shadow CSV manifests
- Sampling IN/OUT models for specific samples
- Creating forget sets containing specific samples
"""
from __future__ import annotations
import json
from typing import List, Tuple, Dict, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader, TensorDataset


def load_shadow_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load shadow CSV manifest and associated metadata.

    Args:
        csv_path: Path to shadow manifest CSV

    Returns:
        Tuple of (csv_dataframe, metadata_dict)

    Raises:
        FileNotFoundError: If CSV or metadata not found
        ValueError: If metadata path cannot be determined
    """
    csv_path_obj = Path(csv_path)

    if not csv_path_obj.exists():
        raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

    # Load CSV
    csv_df = pd.read_csv(csv_path)

    # Infer metadata path (same name, different extension)
    # Expected pattern: {dataset}_shadow_manifest_N{N}_{model}_wd{wd}.csv
    # Metadata pattern: {dataset}_shadow_metadata_N{N}_{model}_wd{wd}.json
    metadata_path = csv_path_obj.with_name(
        csv_path_obj.stem.replace('manifest', 'metadata') + '.json'
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return csv_df, metadata


def sample_shadows_from_csv(
    csv_df: pd.DataFrame,
    sample_id: int,
    N: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample N IN models and N OUT models for a given sample.

    Args:
        csv_df: DataFrame with columns [sample_id, model_0, model_1, ...]
        sample_id: ID of sample to attack
        N: Number of IN and OUT models to sample
        rng: NumPy random generator for reproducibility

    Returns:
        Tuple of (in_model_ids, out_model_ids) as numpy arrays

    Raises:
        ValueError: If sample_id not found or insufficient IN/OUT models
    """
    # Get row for this sample
    if sample_id >= len(csv_df):
        raise ValueError(f"Sample ID {sample_id} out of range (max: {len(csv_df) - 1})")

    row = csv_df.loc[sample_id]

    # Extract model columns
    model_cols = [col for col in csv_df.columns if col.startswith('model_')]

    # Find IN and OUT models
    in_models = []
    out_models = []

    for col in model_cols:
        model_id = int(col.split('_')[1])  # Extract ID from "model_123"
        if row[col] == 1:
            in_models.append(model_id)
        else:
            out_models.append(model_id)

    in_models = np.array(in_models)
    out_models = np.array(out_models)

    # Check coverage
    if len(in_models) < N:
        raise ValueError(
            f"Sample {sample_id} has only {len(in_models)} IN models, need {N}. "
            f"CSV may not have sufficient coverage."
        )

    if len(out_models) < N:
        raise ValueError(
            f"Sample {sample_id} has only {len(out_models)} OUT models, need {N}. "
            f"CSV may not have sufficient coverage."
        )

    # Sample N from each
    in_sample = rng.choice(in_models, size=N, replace=False)
    out_sample = rng.choice(out_models, size=N, replace=False)

    return in_sample, out_sample


def get_model_path_from_metadata(metadata: Dict, model_id: int) -> str:
    """
    Get model path for a given model ID from metadata.

    Args:
        metadata: Metadata dictionary from load_shadow_csv()
        model_id: Model ID to look up

    Returns:
        Path to model checkpoint

    Raises:
        ValueError: If model_id not found
    """
    for model_info in metadata['models']:
        if model_info['model_id'] == model_id:
            return model_info['path']

    raise ValueError(f"Model ID {model_id} not found in metadata")


def get_model_train_indices_from_metadata(metadata: Dict, model_id: int) -> List[int]:
    """
    Get training indices for a given model ID from metadata.

    Args:
        metadata: Metadata dictionary from load_shadow_csv()
        model_id: Model ID to look up

    Returns:
        List of sample indices the model was trained on

    Raises:
        ValueError: If model_id not found
    """
    for model_info in metadata['models']:
        if model_info['model_id'] == model_id:
            return model_info['train_indices']

    raise ValueError(f"Model ID {model_id} not found in metadata")


def sample_forget_set_with_sample(
    target_sample_idx: int,
    forget_size: int,
    full_dataset,
    rng: np.random.Generator,
    exclude_indices: List[int] = None,
    candidate_indices: Optional[Sequence[int]] = None,
    batch_size: int = 64,
) -> Tuple[List[int], DataLoader]:
    """
    Sample a forget set of specified size that includes a target sample.

    Args:
        target_sample_idx: Index of sample that must be in forget set
        forget_size: Total size of forget set
        full_dataset: Complete dataset to sample from
        rng: Random generator for sampling
        exclude_indices: Optional list of indices to exclude from sampling
        candidate_indices: Indices that the base model was trained on. If omitted,
            all dataset indices are eligible.
        batch_size: Batch size of the target unlearning run.

    Returns:
        Tuple of (forget_indices, forget_loader)

    Raises:
        ValueError: If forget_size < 1 or target_sample_idx in exclude_indices
    """
    if forget_size < 1:
        raise ValueError(f"forget_size must be >= 1, got {forget_size}")

    n_samples = len(full_dataset)

    if target_sample_idx >= n_samples:
        raise ValueError(f"target_sample_idx {target_sample_idx} out of range (max: {n_samples - 1})")

    if exclude_indices is None:
        exclude_indices = []

    if target_sample_idx in exclude_indices:
        raise ValueError(f"target_sample_idx {target_sample_idx} is in exclude_indices")

    candidates = list(range(n_samples)) if candidate_indices is None else list(candidate_indices)
    if len(set(candidates)) != len(candidates) or any(i < 0 or i >= n_samples for i in candidates):
        raise ValueError("candidate_indices must be unique valid dataset indices")
    if target_sample_idx not in candidates:
        raise ValueError("target_sample_idx must belong to the base model's training set")

    # Start with target sample
    forget_indices = [target_sample_idx]

    # Create pool of available indices (excluding target and excluded)
    available = [i for i in candidates
                 if i != target_sample_idx and i not in exclude_indices]

    # Sample remaining indices
    n_remaining = forget_size - 1
    if n_remaining > len(available):
        raise ValueError(
            f"Cannot sample {forget_size} samples: only {len(available) + 1} available "
            f"after excluding {len(exclude_indices)} indices"
        )

    if n_remaining > 0:
        sampled = rng.choice(available, size=n_remaining, replace=False)
        forget_indices.extend(sampled.tolist())

    # Create forget loader
    forget_subset = Subset(full_dataset, forget_indices)
    forget_loader = DataLoader(forget_subset, batch_size=batch_size, shuffle=True)

    return forget_indices, forget_loader


def create_retain_loader_excluding_forget(
    full_dataset,
    forget_indices: List[int],
    batch_size: int = 64,
    base_train_indices: List[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create retain loaders excluding specified forget indices.

    Args:
        full_dataset: Complete dataset
        forget_indices: Indices to exclude (forget set)
        batch_size: Batch size for loaders
        base_train_indices: Optional list of indices the base model was trained on.
                           If provided, retain set = base_train_indices - forget_indices.
                           If None, retain set = all_indices - forget_indices (legacy behavior).

    Returns:
        Tuple of (retain_loader, retain_full_loader)
    """
    forget_set = set(forget_indices)

    if base_train_indices is not None:
        # Use only the base model's training indices, excluding forget set
        retain_indices = [i for i in base_train_indices if i not in forget_set]
    else:
        # Legacy: use all dataset indices except forget set
        n_samples = len(full_dataset)
        retain_indices = [i for i in range(n_samples) if i not in forget_set]

    # Create retain loaders
    retain_subset = Subset(full_dataset, retain_indices)
    retain_loader = DataLoader(retain_subset, batch_size=batch_size, shuffle=True)
    retain_full_loader = DataLoader(retain_subset, batch_size=len(retain_subset), shuffle=False)

    return retain_loader, retain_full_loader


def verify_csv_coverage(csv_path: str, target_N: int) -> Dict:
    """
    Verify that a CSV manifest meets coverage requirements.

    Args:
        csv_path: Path to CSV manifest
        target_N: Required number of IN/OUT models per sample

    Returns:
        Dictionary with verification results
    """
    csv_df, metadata = load_shadow_csv(csv_path)

    model_cols = [col for col in csv_df.columns if col.startswith('model_')]
    n_models = len(model_cols)
    n_samples = len(csv_df)

    # Check coverage for all samples
    min_in = float('inf')
    min_out = float('inf')
    max_in = 0
    max_out = 0
    insufficient_samples = []

    for idx, row in csv_df.iterrows():
        in_count = row[model_cols].sum()
        out_count = n_models - in_count

        min_in = min(min_in, in_count)
        min_out = min(min_out, out_count)
        max_in = max(max_in, in_count)
        max_out = max(max_out, out_count)

        if in_count < target_N or out_count < target_N:
            insufficient_samples.append({
                'sample_id': idx,
                'in_count': int(in_count),
                'out_count': int(out_count),
            })

    is_valid = len(insufficient_samples) == 0

    return {
        'valid': is_valid,
        'n_samples': n_samples,
        'n_models': n_models,
        'target_N': target_N,
        'min_in': int(min_in),
        'max_in': int(max_in),
        'min_out': int(min_out),
        'max_out': int(max_out),
        'insufficient_samples': insufficient_samples,
        'dataset': metadata.get('dataset', 'unknown'),
        'seed': metadata.get('seed', 'unknown'),
    }


def print_csv_stats(csv_path: str):
    """
    Print detailed statistics about a shadow CSV manifest.

    Args:
        csv_path: Path to CSV manifest
    """
    csv_df, metadata = load_shadow_csv(csv_path)

    model_cols = [col for col in csv_df.columns if col.startswith('model_')]
    n_models = len(model_cols)
    n_samples = len(csv_df)

    print("=" * 60)
    print(f"Shadow CSV Manifest: {csv_path}")
    print("=" * 60)
    print(f"Dataset: {metadata.get('dataset', 'unknown')}")
    print(f"Seed: {metadata.get('seed', 'unknown')}")
    print(f"Model Type: {metadata.get('model_type', 'unknown')}")
    print(f"Weight Decay: {metadata.get('weight_decay', 'unknown')}")
    print(f"Target N: {metadata.get('target_N', 'unknown')}")
    print(f"\nDimensions:")
    print(f"  Samples: {n_samples}")
    print(f"  Models: {n_models}")

    # Compute IN/OUT statistics
    in_counts = []
    out_counts = []

    for idx, row in csv_df.iterrows():
        in_count = row[model_cols].sum()
        out_count = n_models - in_count
        in_counts.append(in_count)
        out_counts.append(out_count)

    in_counts = np.array(in_counts)
    out_counts = np.array(out_counts)

    print(f"\nCoverage Statistics:")
    print(f"  IN models per sample:")
    print(f"    Min: {in_counts.min()}")
    print(f"    Max: {in_counts.max()}")
    print(f"    Mean: {in_counts.mean():.2f}")
    print(f"    Median: {np.median(in_counts):.0f}")
    print(f"  OUT models per sample:")
    print(f"    Min: {out_counts.min()}")
    print(f"    Max: {out_counts.max()}")
    print(f"    Mean: {out_counts.mean():.2f}")
    print(f"    Median: {np.median(out_counts):.0f}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python shadow_utils.py <csv_path>")
        print("  Prints statistics about a shadow CSV manifest")
        sys.exit(1)

    csv_path = sys.argv[1]
    print_csv_stats(csv_path)

    # Optionally verify coverage
    if len(sys.argv) >= 3:
        target_N = int(sys.argv[2])
        print(f"\nVerifying coverage for N={target_N}...")
        result = verify_csv_coverage(csv_path, target_N)
        if result['valid']:
            print("✓ Coverage is valid!")
        else:
            print(f"✗ Coverage insufficient: {len(result['insufficient_samples'])} samples below threshold")
            print(f"  Details: {result}")
