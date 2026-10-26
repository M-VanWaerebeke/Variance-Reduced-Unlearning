#!/usr/bin/env python3
"""
risk_vs_privacy.py — Excess risk vs. Empirical Privacy Risk.

Plots the trade-off between utility (Excess Risk) and privacy (Empirical Privacy Risk)
for various unlearning methods. The ratio of forget data (r_f) is represented by the
marker size.
"""
import argparse
import copy
from collections import defaultdict
from typing import Dict
import numpy as np
import sys
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import os
import pickle
import hashlib
import shutil
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Ensure float64
torch.set_default_dtype(torch.float64)

# Detect if we're in a log file (not a terminal) to reduce tqdm output
IN_LOGFILE = not sys.stdout.isatty()

from fast_grid import (
    run_unlearning_with_checkpoints,
    averaged_noised_retain_loss,
    _evaluate_retain_loss,
    _evaluate_loader_loss,
    _param_vector,
    _load_full_base,
    apply_gaussian_noise_to_model,
)
from gradient_representation import (
    set_seed,
    get_data_loaders,
    get_retain_forget_loaders,
    UnlearnLoaders,
    build_model,
    train_model,
    Loaders,
    _ensure_pretrained_full_model,
    _ensure_scratch_model,
)
from shadow_utils import (
    load_shadow_csv,
    sample_shadows_from_csv,
    get_model_path_from_metadata,
    get_model_train_indices_from_metadata,
    sample_forget_set_with_sample,
    create_retain_loader_excluding_forget,
)
from unlearned_cache import (
    UnlearnedManifest,
    get_method_key,
    get_manifest_path,
    get_unlearned_model_path,
)
from digit_forget import parse_vru_variant
from pretrain_model import MNIST_HP_CONFIGS, MNIST_DISCOUNT, MNIST_EPOCHS

# Changing the attack protocol or evaluation metric must invalidate cached results.
CACHE_PROTOCOL_VERSION = "paper-audit-v3"


def shadow_manifest_fingerprint(csv_path: str) -> str:
    """Identify the shadow membership table and its checkpoint metadata."""
    csv_file = Path(csv_path)
    metadata_file = csv_file.with_name(
        csv_file.stem.replace("manifest", "metadata") + ".json"
    )
    digest = hashlib.sha256()
    for path in (csv_file, metadata_file):
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def gaussian_log_likelihood_ratio(value, reference_scores, target_scores):
    """Log density ratio of two fitted one-dimensional Gaussians."""
    mu_ref = float(np.mean(reference_scores))
    std_ref = float(np.std(reference_scores)) + 1e-8
    mu_tgt = float(np.mean(target_scores))
    std_tgt = float(np.std(target_scores)) + 1e-8
    log_ref = -np.log(std_ref) - 0.5 * ((value - mu_ref) / std_ref) ** 2
    log_tgt = -np.log(std_tgt) - 0.5 * ((value - mu_tgt) / std_tgt) ** 2
    return float(log_tgt - log_ref), (mu_ref, std_ref, mu_tgt, std_tgt)


def _noise_shadow_for_attack(model, method_params, shadow_r_f, rng, device):
    """Apply the same empirical output-noise rule used for the target model."""
    kappa = float(method_params.get("kappa_dp", 0.0))
    if kappa <= 0:
        return
    timing = method_params.get("timing")
    if timing != "empirical":
        raise NotImplementedError(
            f"Shadow calibration for noise timing {timing!r} is not implemented"
        )
    std = kappa * shadow_r_f / (1.0 - shadow_r_f)
    apply_gaussian_noise_to_model(model, std, rng, device)


def _global_subset_indices(dataset, full_dataset):
    """Map a possibly nested Subset back to the shadow manifest's sample IDs."""
    if dataset is full_dataset:
        return list(range(len(full_dataset)))
    if isinstance(dataset, torch.utils.data.Subset):
        parent_ids = _global_subset_indices(dataset.dataset, full_dataset)
        return [parent_ids[int(i)] for i in dataset.indices]
    raise ValueError("held_out_dataset must be a subset of full_dataset")

def parse_method_with_kappa(method_str: str) -> tuple[str, float]:
    """
    Parse method string with optional kappa_dp value.

    Format: "method:kappa" or "method"
    Examples:
        "vru:0.1" -> ("vru", 0.1)
        "vru-avg-proj:2.0" -> ("vru-avg-proj", 2.0)
        "nft" -> ("nft", 0.0)

    Returns:
        (method_name, kappa_dp)
    """
    if ":" in method_str:
        parts = method_str.rsplit(":", 1)  # Split from right to handle "vru-avg-proj:0.1"
        method_name = parts[0]
        try:
            kappa_dp = float(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid kappa_dp format in '{method_str}'. "
                f"Expected format: 'method:kappa' (e.g., 'vru:0.1')"
            )
        return method_name, kappa_dp
    else:
        # No kappa specified, default to 0.0
        return method_str, 0.0

def _get_pretty_name(method_name: str) -> str:
    m = method_name.lower()
    if m == 'nft':
        return 'Fine-Tune'
    if m == 'vru':
        return 'VRU'
    if m == 'vru-avg-proj':
        return 'VRU-AP'
    if m == 'vru-avg':
        return 'VRU-Avg'
    if m == 'vru-proj':
        return 'VRU-Proj'
    if m == 'neggrad+':
        return 'NegGrad+'
    if m == 'scrub':
        return 'SCRUB'
    return method_name.upper()

def parse_args():
    p = argparse.ArgumentParser(description="Excess Risk vs Privacy Risk grid")
    p.add_argument("--min_rf", type=float, default=1e-3)
    p.add_argument("--max_rf", type=float, default=1e-1)
    p.add_argument("--n_rf", type=int, default=5)
    p.add_argument("--rf_values", type=float, nargs="+", default=None,
                   help="Explicit forget fractions (overrides min_rf/max_rf/n_rf)")
    p.add_argument("--T", type=float, default=10, help="Epoch budget")
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds (runs 0 to seeds-1)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--n_shadows", type=int, default=16, help="Number of shadow models for MIA")
    p.add_argument("--attack_parallel_jobs", type=int, default=1,
                   help="Number of parallel jobs for sample attacks (default: 1 sequential)")
    p.add_argument("--dataset", type=str, default="digits")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--device", type=str, default=None,
                   help="Device to use (e.g., 'cpu', 'cuda', 'mps'). If not specified, uses cuda if available and --no_cuda is not set, else cpu")
    p.add_argument("--save_file", type=str, default="risk_vs_privacy_plot.pdf")
    p.add_argument("--cache_file", type=str, default="risk_vs_privacy_cache.pkl", help="Path to cache file")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN"],
                   help="Model architecture: LR or NN")

    # Methods: list of (name, lr, decay, timing)
    # Method name can include kappa_dp as "method:kappa" (e.g., "vru:0.1")
    p.add_argument("--run", nargs=4, action="append", metavar=("METHOD[:KAPPA]", "LR", "DECAY", "TIMING"),
                   help="Add method. Examples: --run vru:0.1 0.01 0.9 after, --run vru-avg-proj:2.0 0.01 0.9 empirical")
    # Shadow model and display parameters
    p.add_argument("--show_std", action="store_true", help="Show 2D error bars on the risk vs privacy plot")
    p.add_argument("--original_style", action="store_true",
                   help="Use original matplotlib styling instead of improved styling")
    p.add_argument("--average", action="store_true",
                   help="Use Polyak averaging for VRU")
    p.add_argument("--project", action="store_true",
                   help="Project VRU updates onto ball")
    p.add_argument("--cache_dir", type=str, default="shadow_manifests",
                   help="Directory for shadow manifests and unlearned model cache (default: shadow_manifests)")
    p.add_argument("--dummy", action="store_true",
                   help="Run in dummy mode with minimal parameters for quick testing (n_rf=2, seeds=1, n_shadows=2, T=2)")

    return p.parse_args()


def auto_discover_shadow_csv(
    dataset: str,
    model_type: str,
    weight_decay: float,
    n_shadows: int,
    shadow_manifests_dir: str = "shadow_manifests"
) -> str:
    """
    Auto-discover the shadow CSV file with the largest N that matches the given parameters.

    Args:
        dataset: Dataset name (e.g., "digits")
        model_type: Model architecture (e.g., "LR", "NN")
        weight_decay: Weight decay value
        n_shadows: Number of shadows requested by user
        shadow_manifests_dir: Directory containing shadow manifests

    Returns:
        Path to the shadow CSV file with largest N >= n_shadows

    Raises:
        FileNotFoundError: If no matching CSV is found
        ValueError: If largest N found is less than n_shadows
    """
    import glob
    import re
    import os

    # Format weight decay to match filename format
    wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"

    # Pattern: {dataset}_shadow_manifest_N*_{model_type}_wd{wd}.csv
    pattern = os.path.join(
        shadow_manifests_dir,
        f"{dataset}_shadow_manifest_N*_{model_type}_wd{wd_str}.csv"
    )

    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(
            f"No shadow CSV found matching pattern: {pattern}\n"
            f"Generate shadow models first using generate_shadow_csv.py"
        )

    # Extract N from each filename and find the maximum
    n_values = []
    for filepath in matching_files:
        filename = os.path.basename(filepath)
        # Extract N from "N{number}_" pattern
        match = re.search(r'_N(\d+)_', filename)
        if match:
            n_values.append((int(match.group(1)), filepath))

    if not n_values:
        raise ValueError(f"Could not parse N values from matching files: {matching_files}")

    # Find the file with largest N
    n_values.sort(reverse=True)  # Sort by N descending
    max_n, csv_path = n_values[0]

    # Validate that max_n >= n_shadows
    if max_n < n_shadows:
        raise ValueError(
            f"Shadow CSV has N={max_n} base models, but you requested --n_shadows={n_shadows}. "
            f"Generate more shadow models or reduce --n_shadows to {max_n} or less."
        )

    print(f"Auto-discovered shadow CSV: {os.path.basename(csv_path)} (N={max_n})")
    return csv_path


def get_file_hash(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def averaged_empirical_noised_retain_loss(
    model: torch.nn.Module,
    *,
    std: float,
    repetitions: int,
    uloaders: UnlearnLoaders,
    device: torch.device,
    weight_decay: float,
    rng: np.random.Generator,
    eval_loader=None,
) -> float:
    """Compute expected evaluation loss under fixed Gaussian noise."""
    losses = []
    for _ in range(repetitions):
        m = copy.deepcopy(model).to(device=device, dtype=torch.float64)
        apply_gaussian_noise_to_model(m, std, rng, device)
        loss = (_evaluate_loader_loss(m, eval_loader, device, weight_decay)
                if eval_loader is not None else
                _evaluate_retain_loss(m, uloaders, device, weight_decay))
        losses.append(loss)
    return float(np.mean(losses)) if losses else float('inf')

def get_all_data(loader, device):
    inputs, targets = [], []
    for x, y in loader:
        inputs.append(x.to(device))
        targets.append(y.to(device))
    return torch.cat(inputs), torch.cat(targets)


def _train_or_load_base_model(
    base_path: str,
    loaders_for_train,
    device,
    model_type: str,
    weight_decay: float,
    dataset_name: str,
    train_seed: int,
    shadows_dir: str,
    known_config: tuple = None,
):
    """Train a base model if not cached, or load from cache."""
    if os.path.exists(base_path):
        # Load cached model
        model = build_model(model_type, loaders_for_train.n_features, loaders_for_train.n_classes).to(device, dtype=torch.float64)
        try:
            sd = torch.load(base_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(base_path, map_location=device)
        for k, v in list(sd.items()):
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                sd[k] = v.to(dtype=torch.float64)
        model.load_state_dict(sd)
        return model, None

    # Train new model with hyperparameter sweep
    print(f"      Training base model -> {os.path.basename(base_path)}...")
    model = build_model(model_type, loaders_for_train.n_features, loaders_for_train.n_classes).to(device, dtype=torch.float64)

    if dataset_name == "mnist":
        # Grid sweep matching test_vru_retain_loss.py
        for lr_raw, optimizer, algo, scheduler in MNIST_HP_CONFIGS:
            lr_eff = lr_raw / 10 if model_type == "NN" else lr_raw
            model = build_model(model_type, loaders_for_train.n_features, loaders_for_train.n_classes).to(device, dtype=torch.float64)
            _ = train_model(
                model, loaders_for_train, device,
                algo=algo, epochs=MNIST_EPOCHS, lr=lr_eff,
                seed=train_seed, discount_factor=MNIST_DISCOUNT, scheduler=scheduler, momentum=0.0,
                optimizer=optimizer,
                weight_decay=weight_decay, log_interval=0,
                save_folder=shadows_dir,
                model_type=model_type
            )
    else:
        # Original behavior for digits dataset
        epochs_sweep = 3000
        best_config = None

        for pretrain_lr in [0.1, 0.03, 0.01, 0.003]:
            lr_eff = pretrain_lr
            if model_type == "NN":
                lr_eff = pretrain_lr / 10

            for optimizer in ["sgd", "adam"]:
                _ = train_model(
                    model, loaders_for_train, device,
                    algo="sgd", epochs=epochs_sweep, lr=lr_eff,
                    seed=train_seed, discount_factor=0.2, scheduler='step', momentum=0.0,
                    optimizer=optimizer,
                    weight_decay=weight_decay, log_interval=0,
                    save_folder=shadows_dir,
                    model_type=model_type
                )

    # train_model saves "best_model...", we rename it
    wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
    saved_name = f"best_model-seed{train_seed}_model{model_type.upper()}_wd{wd_str}.pt"
    src = os.path.join(shadows_dir, saved_name)
    if os.path.exists(src):
        shutil.move(src, base_path)
        # Reload the best model
        try:
            sd = torch.load(base_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(base_path, map_location=device)
        for k, v in list(sd.items()):
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                sd[k] = v.to(dtype=torch.float64)
        model.load_state_dict(sd)
    else:
        # Fallback: save current state
        torch.save(model.state_dict(), base_path)

    return model, best_config


def _load_model_state(model_path: str, device: torch.device):
    """Load model state dict with float64 casting."""
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    for k, v in state.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            state[k] = v.to(dtype=torch.float64)
    return state


def prep_sample_attack(
    sample_idx: int,
    forget_sample_id: int,
    csv_df,
    metadata: dict,
    n_shadows: int,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    model_type: str,
    loaders: Loaders,
    device: torch.device,
    manifest_path: str,
    forget_size: int,
    full_dataset,
    rng: np.random.Generator,
    shadow_r_f: float,
    weight_decay: float,
    cache_dir: str,
    dataset_name: str,
):
    """
    Prepare attack data for a single sample (cheap, runs in main process).

    Samples IN/OUT models, scores OUT models (reference), scores cached IN models,
    and builds a list of unlearning jobs still needed.

    Returns:
        dict with keys:
            sample_idx, forget_sample_id, ref_scores_array, cached_tgt_scores,
            n_reused, unlearn_jobs, n_shadows
    """
    def get_lira_scores(model_curr, x, y):
        with torch.no_grad():
            logits = model_curr(x)
            true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.view(-1, 1), False)
            other_logits = logits[mask].view(logits.size(0), logits.size(1) - 1)
            other_lse = torch.logsumexp(other_logits, dim=1)
            return (true_logits - other_lse).cpu().numpy()

    # Sample IN and OUT models for this specific sample
    in_model_ids, out_model_ids = sample_shadows_from_csv(
        csv_df, forget_sample_id, n_shadows, rng
    )

    # --- Reference shadows: OUT models (never saw this sample) - use directly ---
    ref_scores = []
    for model_id in out_model_ids:
        model_path = get_model_path_from_metadata(metadata, model_id)
        temp_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
        state = _load_model_state(model_path, device)
        temp_model.load_state_dict(state)
        temp_model.eval()
        scores = get_lira_scores(temp_model, eval_x, eval_y)
        ref_scores.append(scores)

    ref_scores_array = np.array(ref_scores)

    # --- Target shadows: IN models, check cache ---
    cached_tgt_scores = []
    in_model_ids_set = set(int(m) for m in in_model_ids)

    manifest = UnlearnedManifest(manifest_path)
    compatible = manifest.find_compatible_shadows(
        sample_id=forget_sample_id,
        n_needed=n_shadows,
        allowed_base_ids=in_model_ids_set,
        exclude_base_ids=set(),
    )

    n_reused = 0
    used_base_ids = set()
    for entry in compatible:
        temp_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
        state = _load_model_state(entry.path, device)
        expected_keys = set(temp_model.state_dict().keys())
        if set(state.keys()) != expected_keys:
            continue
        temp_model.load_state_dict(state)
        temp_model.eval()
        scores = get_lira_scores(temp_model, eval_x, eval_y)
        cached_tgt_scores.append(scores)
        used_base_ids.add(entry.base_model_id)
        n_reused += 1

    # Build list of unlearning jobs still needed
    n_remaining = n_shadows - n_reused
    available_base_ids = [int(m) for m in in_model_ids if int(m) not in used_base_ids]

    if n_remaining > len(available_base_ids):
        raise RuntimeError(f"Only {len(available_base_ids)} usable IN shadows for sample "
                           f"{forget_sample_id}; need {n_remaining}")

    unlearn_jobs = []
    for i in range(n_remaining):
        base_id = available_base_ids[i]
        model_path = get_model_path_from_metadata(metadata, base_id)
        base_train_indices = get_model_train_indices_from_metadata(metadata, base_id)
        unlearn_jobs.append({
            'sample_idx': sample_idx,
            'forget_sample_id': forget_sample_id,
            'base_id': base_id,
            'model_path': model_path,
            'base_train_indices': base_train_indices,
        })

    return {
        'sample_idx': sample_idx,
        'forget_sample_id': forget_sample_id,
        'ref_scores_array': ref_scores_array,
        'cached_tgt_scores': cached_tgt_scores,
        'n_reused': n_reused,
        'unlearn_jobs': unlearn_jobs,
        'n_shadows': n_shadows,
    }


def run_single_unlearning(
    sample_idx: int,
    forget_sample_id: int,
    base_id: int,
    model_path: str,
    base_train_indices: list,
    rng_seed: int,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    model_type: str,
    loaders: Loaders,
    device: torch.device,
    forget_size: int,
    full_dataset,
    base_algo: str,
    shadow_r_f: float,
    method_params: dict,
    weight_decay: float,
    cache_dir: str,
    dataset_name: str,
    use_average: bool,
    use_project: bool,
    seed_val: int,
):
    """
    Run a single unlearning job (expensive, runs in worker process).

    Returns:
        dict with keys: sample_idx, base_id, scores, entry_data, loss_history
    """
    def get_lira_scores(model_curr, x, y):
        with torch.no_grad():
            logits = model_curr(x)
            true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.view(-1, 1), False)
            other_logits = logits[mask].view(logits.size(0), logits.size(1) - 1)
            other_lse = torch.logsumexp(other_logits, dim=1)
            return (true_logits - other_lse).cpu().numpy()

    # Each worker creates its own RNG from the pre-drawn seed
    job_rng = np.random.default_rng(rng_seed)

    base_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
    state = _load_model_state(model_path, device)
    base_model.load_state_dict(state)

    # Sample a forget set containing this sample
    shadow_forget_indices, shadow_forget_loader = sample_forget_set_with_sample(
        target_sample_idx=forget_sample_id,
        forget_size=forget_size,
        full_dataset=full_dataset,
        rng=job_rng,
        exclude_indices=[],
        candidate_indices=base_train_indices,
        batch_size=loaders.train.batch_size,
    )

    # Create retain loader from base model's training data minus forget set
    shadow_retain_loader, shadow_retain_full = create_retain_loader_excluding_forget(
        full_dataset, shadow_forget_indices, batch_size=loaders.train.batch_size,
        base_train_indices=base_train_indices
    )

    # Create UnlearnLoaders
    shadow_uloaders = UnlearnLoaders(
        retain=shadow_retain_loader,
        retain_full=shadow_retain_full,
        forget=shadow_forget_loader,
        n_retain=len(shadow_retain_loader.dataset),
        n_forget=len(shadow_forget_indices),
    )

    # Unlearn
    unlearn_seed = seed_val + base_id + sample_idx * 1000
    set_seed(unlearn_seed)
    unlearn_epochs = method_params.get('epochs', 30)
    checkpoints, full_loss_history = run_unlearning_with_checkpoints(
        method=base_algo,
        base_model=base_model,
        loaders=loaders,
        uloaders=shadow_uloaders,
        r_f=shadow_r_f,
        device=device,
        epochs=unlearn_epochs,
        lr=method_params.get('lr', 0.01),
        scheduler=method_params.get('scheduler', 'step'),
        scheduler_step_freq=1,
        discount_factor=method_params.get('decay', 0.3),
        weight_decay=weight_decay,
        checkpoint_every=unlearn_epochs + 1,  # Only final checkpoint
        scratch_best_path="dummy_shadow.pt",
        scratch_best_loss=float('inf'),
        alpha=method_params.get('alpha', 0.05),
        verbose=False,
        average=use_average,
        project=use_project,
    )
    if checkpoints:
        base_model.load_state_dict(checkpoints[-1][1])

    _noise_shadow_for_attack(base_model, method_params, shadow_r_f, job_rng, device)

    # Save unlearned model to cache
    unique_id = seed_val * 1000000 + base_id * 1000 + sample_idx
    unlearned_path = get_unlearned_model_path(
        cache_dir, dataset_name, unique_id, base_id, shadow_r_f,
        model_type=model_type,
    )
    torch.save(base_model.state_dict(), unlearned_path)

    # Score
    base_model.eval()
    scores = get_lira_scores(base_model, eval_x, eval_y)

    entry_data = {
        'base_model_id': base_id,
        'forget_indices': shadow_forget_indices,
        'path': unlearned_path,
        'seed_used': unlearn_seed,
    }

    return {
        'sample_idx': sample_idx,
        'base_id': base_id,
        'scores': scores,
        'entry_data': entry_data,
        'loss_history': full_loss_history if full_loss_history else None,
    }


def _unlearn_job_wrapper(args):
    """Wrapper for multiprocessing - unpacks args and calls run_single_unlearning."""
    return run_single_unlearning(**args)


def attack_single_sample(
    sample_idx: int,
    forget_sample_id: int,
    csv_df,
    metadata: dict,
    n_shadows: int,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    model_type: str,
    loaders: Loaders,
    device: torch.device,
    manifest_path: str,
    forget_size: int,
    full_dataset,
    rng: np.random.Generator,
    base_algo: str,
    shadow_r_f: float,
    method_params: dict,
    weight_decay: float,
    cache_dir: str,
    dataset_name: str,
    use_average: bool,
    use_project: bool,
    seed_val: int,
):
    """
    Attack a single sample using shadow models.

    Each worker processes ONE complete sample:
      1. Load 16 OUT models (reference, no unlearning)
      2. Load/unlearn 16 IN models (sequential within this sample)
      3. Score target model
      4. Return results

    NO parallelization within this function - it's sequential per sample.

    Args:
        sample_idx: Index of the sample in the forget set
        forget_sample_id: Actual sample ID in the dataset
        csv_df: Shadow CSV DataFrame
        metadata: Shadow model metadata
        n_shadows: Number of IN/OUT shadow models per sample
        eval_x: Evaluation features
        eval_y: Evaluation labels
        model_type: Model architecture
        loaders: Data loaders
        device: Computation device
        manifest_path: Path to manifest (workers load their own copy)
        forget_size: Size of forget set for shadow unlearning
        full_dataset: Complete training dataset
        rng: Random number generator
        base_algo: Base unlearning algorithm
        shadow_r_f: Forget fraction for shadow models
        method_params: Unlearning hyperparameters
        weight_decay: L2 regularization
        cache_dir: Cache directory
        dataset_name: Dataset name
        use_average: Use averaging in VRU
        use_project: Use projection in VRU
        seed_val: Random seed

    Returns:
        tuple: (sample_idx, ref_scores, tgt_scores, n_reused, n_newly_unlearned, new_entries, target_histories)
        where new_entries is a list of dicts with entry data for manifest
        and target_histories is a list of training loss histories (step, loss) for IN models
    """
    # Define get_lira_scores locally for this function
    def get_lira_scores(model_curr, x, y):
        with torch.no_grad():
            logits = model_curr(x)
            true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.view(-1, 1), False)
            other_logits = logits[mask].view(logits.size(0), logits.size(1) - 1)
            other_lse = torch.logsumexp(other_logits, dim=1)
            return (true_logits - other_lse).cpu().numpy()

    n_reused = 0
    n_newly_unlearned = 0
    new_entries = []  # Collect entry data to return to parent
    target_histories = []  # Collect training histories from IN model unlearning

    # Load manifest (each worker loads its own copy for reading)
    manifest = UnlearnedManifest(manifest_path)

    # Sample IN and OUT models for this specific sample
    in_model_ids, out_model_ids = sample_shadows_from_csv(
        csv_df, forget_sample_id, n_shadows, rng
    )

    # --- Reference shadows: OUT models (never saw this sample) - use directly ---
    ref_scores = []
    for model_id in out_model_ids:
        model_path = get_model_path_from_metadata(metadata, model_id)
        temp_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
        state = _load_model_state(model_path, device)
        temp_model.load_state_dict(state)
        temp_model.eval()
        scores = get_lira_scores(temp_model, eval_x, eval_y)
        ref_scores.append(scores)

    ref_scores_array = np.array(ref_scores)

    # --- Target shadows: IN models, with cache ---
    tgt_scores = []
    in_model_ids_set = set(int(m) for m in in_model_ids)

    # 1. Check cache for compatible unlearned models
    compatible = manifest.find_compatible_shadows(
        sample_id=forget_sample_id,
        n_needed=n_shadows,
        allowed_base_ids=in_model_ids_set,
        exclude_base_ids=set(),
    )

    # 2. Score reused models
    used_base_ids = set()
    for entry in compatible:
        temp_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
        state = _load_model_state(entry.path, device)
        # Validate state dict keys match model architecture (skip stale cache from different model_type)
        expected_keys = set(temp_model.state_dict().keys())
        if set(state.keys()) != expected_keys:
            continue
        temp_model.load_state_dict(state)
        temp_model.eval()
        scores = get_lira_scores(temp_model, eval_x, eval_y)
        tgt_scores.append(scores)
        used_base_ids.add(entry.base_model_id)
        n_reused += 1

    # 3. Generate remaining unlearned models
    n_remaining = n_shadows - n_reused
    available_base_ids = [int(m) for m in in_model_ids if int(m) not in used_base_ids]

    if n_remaining > len(available_base_ids):
        raise RuntimeError(f"Only {len(available_base_ids)} usable IN shadows for sample "
                           f"{forget_sample_id}; need {n_remaining}")

    for i in range(n_remaining):
        base_id = available_base_ids[i]
        model_path = get_model_path_from_metadata(metadata, base_id)
        base_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
        state = _load_model_state(model_path, device)
        base_model.load_state_dict(state)

        # Get base model's training indices from CSV metadata
        base_train_indices = get_model_train_indices_from_metadata(metadata, base_id)

        # Sample a forget set containing this sample
        shadow_forget_indices, shadow_forget_loader = sample_forget_set_with_sample(
            target_sample_idx=forget_sample_id,
            forget_size=forget_size,
            full_dataset=full_dataset,
            rng=rng,
            exclude_indices=[],
            candidate_indices=base_train_indices,
            batch_size=loaders.train.batch_size,
        )

        # Create retain loader from base model's training data minus forget set
        shadow_retain_loader, shadow_retain_full = create_retain_loader_excluding_forget(
            full_dataset, shadow_forget_indices, batch_size=loaders.train.batch_size,
            base_train_indices=base_train_indices
        )

        # Create UnlearnLoaders
        shadow_uloaders = UnlearnLoaders(
            retain=shadow_retain_loader,
            retain_full=shadow_retain_full,
            forget=shadow_forget_loader,
            n_retain=len(shadow_retain_loader.dataset),
            n_forget=len(shadow_forget_indices),
        )

        # Unlearn using run_unlearning_with_checkpoints (supports all methods)
        unlearn_seed = seed_val + base_id + sample_idx * 1000
        set_seed(unlearn_seed)
        unlearn_epochs = method_params.get('epochs', 30)
        checkpoints, full_loss_history = run_unlearning_with_checkpoints(
            method=base_algo,
            base_model=base_model,
            loaders=loaders,
            uloaders=shadow_uloaders,
            r_f=shadow_r_f,
            device=device,
            epochs=unlearn_epochs,
            lr=method_params.get('lr', 0.01),
            scheduler=method_params.get('scheduler', 'step'),
            scheduler_step_freq=1,
            discount_factor=method_params.get('decay', 0.3),
            weight_decay=weight_decay,
            checkpoint_every=unlearn_epochs + 1,  # Only final checkpoint
            scratch_best_path="dummy_shadow.pt",
            scratch_best_loss=float('inf'),
            alpha=method_params.get('alpha', 0.05),
            verbose=False,
            average=use_average,
            project=use_project,
        )
        # Load final unlearned state into base_model
        if checkpoints:
            base_model.load_state_dict(checkpoints[-1][1])

        _noise_shadow_for_attack(base_model, method_params, shadow_r_f, rng, device)

        # Generate unique ID for this unlearned model
        # Use combination of seed, base_id, and sample_idx to ensure uniqueness
        unique_id = seed_val * 1000000 + base_id * 1000 + sample_idx

        # Save unlearned model to cache
        unlearned_path = get_unlearned_model_path(
            cache_dir, dataset_name, unique_id, base_id, shadow_r_f,
            model_type=model_type,
        )
        torch.save(base_model.state_dict(), unlearned_path)

        # Collect entry data for parent to add to manifest later
        entry_data = {
            'base_model_id': base_id,
            'forget_indices': shadow_forget_indices,
            'path': unlearned_path,
            'seed_used': unlearn_seed,
        }
        new_entries.append(entry_data)

        # Store training history for plotting
        if full_loss_history:
            target_histories.append(full_loss_history)

        # Score
        base_model.eval()
        scores = get_lira_scores(base_model, eval_x, eval_y)
        tgt_scores.append(scores)
        n_newly_unlearned += 1

    if len(tgt_scores) != n_shadows:
        raise RuntimeError(f"Expected {n_shadows} target shadows, got {len(tgt_scores)}")

    tgt_scores_array = np.array(tgt_scores)

    return (sample_idx, ref_scores_array, tgt_scores_array, n_reused, n_newly_unlearned, new_entries, target_histories)


def _attack_sample_wrapper(args):
    """Wrapper for multiprocessing - unpacks args and calls attack_single_sample."""
    return attack_single_sample(*args)


def evaluate_privacy_risk_csv(
    model,
    full_dataset,
    forget_indices: list,
    csv_path: str,
    device: torch.device,
    method_name: str = "unknown",
    method_params: dict = None,
    n_shadows: int = 16,
    seed_val: int = 0,
    weight_decay: float = 0.1,
    dataset_name: str = "digits",
    model_type: str = "LR",
    loaders: Loaders = None,
    debug: bool = False,
    rng: np.random.Generator = None,
    cache_dir: str = "shadow_manifests",
    held_out_dataset=None,
    attack_parallel_jobs: int = 1,
):
    """
    Evaluate privacy risk using CSV-based shadow model sampling (U-LiRA).

    For each sample in forget set, samples N IN models (saw sample) and N OUT models
    (never saw sample) from the CSV manifest. Uses a cache of unlearned models to
    avoid redundant unlearning passes: if a cached unlearned model's forget set
    contains the attacked sample, it can be reused.

    Enforces base model diversity: never uses two unlearned models from the same
    base model to attack a single sample.

    Args:
        model: Target model to evaluate (already unlearned)
        full_dataset: Complete training dataset
        forget_indices: List of indices that were forgotten
        csv_path: Path to shadow CSV manifest
        device: Computation device
        method_name: Unlearning method name
        method_params: Dictionary with unlearning hyperparameters
        n_shadows: Number of IN/OUT shadow models per sample
        seed_val: Random seed
        weight_decay: L2 regularization
        dataset_name: Dataset name
        model_type: Model architecture
        loaders: Data loaders (for val/test)
        debug: Enable debug plots
        rng: Random generator
        cache_dir: Directory for unlearned model cache

    Returns:
        Tuple of (privacy_risk, shadow_histories, None)
    """
    if rng is None:
        rng = np.random.default_rng(seed_val * 100000)

    if method_params is None:
        print("Warning: Missing method_params. Returning 0.0")
        return 0.0, {}, None

    print(f"\n{'='*60}")
    print(f"CSV-Based Privacy Risk Evaluation (U-LiRA)")
    print(f"{'='*60}")
    print(f"CSV Manifest: {csv_path}")
    print(f"Forget Set Size: {len(forget_indices)}")
    print(f"N Shadows per Sample: {n_shadows}")

    # Load CSV manifest and metadata
    csv_df, metadata = load_shadow_csv(csv_path)
    n_samples_csv = len(csv_df)
    if n_samples_csv != len(full_dataset):
        raise ValueError("Shadow manifest and target dataset have different sample counts")
    if metadata.get('batch_size') != loaders.train.batch_size:
        raise ValueError(
            "Shadow manifest batch size differs from target batch size or is missing. "
            "Regenerate shadows with generate_shadow_csv.py --batch_size "
            f"{loaders.train.batch_size} into a new output directory."
        )

    # Verify shadow manifest matches requested model type
    shadow_model_type = metadata.get('model_type', 'LR')
    if shadow_model_type != model_type:
        print(f"\n{'='*60}")
        print(f"❌ MODEL TYPE MISMATCH")
        print(f"{'='*60}")
        print(f"Requested model type: {model_type}")
        print(f"Shadow manifest model type: {shadow_model_type}")
        print(f"\nThe shadow CSV at {csv_path}")
        print(f"contains {shadow_model_type} models, but you requested {model_type} models.")
        print(f"\nPlease generate a shadow manifest with {model_type} architecture:")
        print(f"  python generate_shadow_csv.py --dataset {dataset_name} --model {model_type} --N {n_shadows} --weight_decay {weight_decay}")
        print(f"{'='*60}\n")
        raise ValueError(f"Shadow manifest model type ({shadow_model_type}) does not match requested model type ({model_type})")

    print(f"CSV Samples: {n_samples_csv}")
    print(f"CSV Models: {len([c for c in csv_df.columns if c.startswith('model_')])}")

    # --- Set up unlearned model cache ---
    base_algo, use_average, use_project = parse_vru_variant(method_name.lower())
    # Allow method_params to override flags (e.g. from CLI --average/--project)
    if method_params.get('average', False):
        use_average = True
    if method_params.get('project', False):
        use_project = True
    forget_size = len(forget_indices)
    # Shadow base models and the target model both train on half the full pool.
    shadow_r_f = forget_size / len(loaders.train.dataset)

    method_key = get_method_key(
        method=base_algo,
        model_type=model_type,
        lr=method_params.get('lr', 0.01),
        decay=method_params.get('decay', 0.3),
        wd=weight_decay,
        epochs=method_params.get('epochs', 30),
        rf=shadow_r_f,
        average=use_average,
        project=use_project,
        alpha=method_params.get('alpha'),
    )

    method_key += (
        f"_{CACHE_PROTOCOL_VERSION}_seed{seed_val}"
        f"_bs{loaders.train.batch_size}"
        f"_timing{method_params.get('timing')}_k{method_params.get('kappa_dp', 0.0):g}"
        f"_shadow{shadow_manifest_fingerprint(csv_path)[:16]}"
    )
    # Paths must be method-specific; the legacy checkpoint naming scheme could
    # overwrite one method's shadow model with another method's state.
    cache_dir = os.path.join(cache_dir, CACHE_PROTOCOL_VERSION, method_key)

    manifest_path = get_manifest_path(cache_dir, dataset_name, method_key)
    manifest = UnlearnedManifest(manifest_path, base_csv_path=csv_path)
    manifest.method_key = method_key
    manifest.dataset = dataset_name

    # Verify consistency with base CSV
    if not manifest.verify_consistency(csv_path):
        raise ValueError("Base shadow CSV changed; use a fresh cache directory")

    cache_stats = manifest.stats()
    print(f"Unlearned Cache: {cache_stats['n_entries']} entries, "
          f"{cache_stats['n_unique_base_models']} unique base models")

    # Prepare forget set data
    forget_dataset_subset = torch.utils.data.Subset(full_dataset, forget_indices)
    forget_loader = DataLoader(forget_dataset_subset, batch_size=len(forget_indices), shuffle=False)

    for xb, yb in forget_loader:
        forget_x = xb.to(device)
        forget_y = yb.to(device)
        break

    n_forget = len(forget_indices)

    if held_out_dataset is None or len(held_out_dataset) < n_forget:
        raise ValueError("The attack requires at least |Df| held-out target nonmembers")
    held_out_indices = list(range(len(held_out_dataset)))
    rng.shuffle(held_out_indices)
    held_out_indices = held_out_indices[:n_forget]
    held_out_global_ids = _global_subset_indices(held_out_dataset, full_dataset)
    attacked_sample_ids = list(forget_indices) + [held_out_global_ids[i] for i in held_out_indices]
    if len(set(attacked_sample_ids)) != len(attacked_sample_ids):
        raise ValueError("Forget and held-out attack samples must be disjoint")
    test_subset = torch.utils.data.Subset(held_out_dataset, held_out_indices)
    test_loader = DataLoader(test_subset, batch_size=n_forget, shuffle=False)

    for xb, yb in test_loader:
        test_x = xb.to(device)
        test_y = yb.to(device)
        break

    # Combine for evaluation
    eval_x = torch.cat([forget_x, test_x])
    eval_y = torch.cat([forget_y, test_y])
    n_test = test_x.size(0)  # Actual number of test samples (may differ from n_forget)

    print(f"Evaluating {n_forget} forget + {n_test} test samples")

    # LiRA score function
    def get_lira_scores(model_curr, x, y):
        with torch.no_grad():
            logits = model_curr(x)
            true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.view(-1, 1), False)
            other_logits = logits[mask].view(logits.size(0), logits.size(1) - 1)
            other_lse = torch.logsumexp(other_logits, dim=1)
            return (true_logits - other_lse).cpu().numpy()

    # Storage for shadow scores
    all_reference_scores = []  # OUT models (never saw forget samples)
    all_target_scores = []     # IN models (unlearned forget samples)

    # Counters for cache effectiveness
    n_reused = 0
    n_newly_unlearned = 0

    # Fit a separate conditional shadow distribution for every attacked point.
    # Target nonmembers can be IN for some shadow models, so they need their own
    # IN/OUT selection and unlearning, just like target forget members.
    print(f"\nProcessing {len(attacked_sample_ids)} attack samples...")

    # Start timing
    sample_attack_start = time.time()

    # Parallel or sequential execution based on attack_parallel_jobs
    if attack_parallel_jobs > 1:
        print(f"  Using {attack_parallel_jobs} parallel jobs (flattened unlearning)")

        # Phase 1: Prep all samples sequentially (cheap: model loading + cache lookup)
        all_preps = []
        all_unlearn_jobs = []
        for sample_idx, forget_sample_id in enumerate(attacked_sample_ids):
            prep = prep_sample_attack(
                sample_idx=sample_idx,
                forget_sample_id=forget_sample_id,
                csv_df=csv_df,
                metadata=metadata,
                n_shadows=n_shadows,
                eval_x=eval_x,
                eval_y=eval_y,
                model_type=model_type,
                loaders=loaders,
                device=device,
                manifest_path=manifest.manifest_path,
                forget_size=forget_size,
                full_dataset=full_dataset,
                rng=rng,
                shadow_r_f=shadow_r_f,
                weight_decay=weight_decay,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
            )
            # Pre-draw RNG seeds for each unlearning job (deterministic order)
            for job in prep['unlearn_jobs']:
                job['rng_seed'] = int(rng.integers(0, 2**31))
                # Add shared parameters needed by run_single_unlearning
                job['eval_x'] = eval_x
                job['eval_y'] = eval_y
                job['model_type'] = model_type
                job['loaders'] = loaders
                job['device'] = device
                job['forget_size'] = forget_size
                job['full_dataset'] = full_dataset
                job['base_algo'] = base_algo
                job['shadow_r_f'] = shadow_r_f
                job['method_params'] = method_params
                job['weight_decay'] = weight_decay
                job['cache_dir'] = cache_dir
                job['dataset_name'] = dataset_name
                job['use_average'] = use_average
                job['use_project'] = use_project
                job['seed_val'] = seed_val
            all_unlearn_jobs.extend(prep['unlearn_jobs'])
            all_preps.append(prep)

        n_total_jobs = len(all_unlearn_jobs)
        n_cached = sum(p['n_reused'] for p in all_preps)
        print(f"  Prep complete: {n_cached} cached, {n_total_jobs} unlearning jobs to run on {attack_parallel_jobs} workers")

        # Phase 2: Parallel unlearning (flattened — one job per worker task)
        unlearn_results = []
        if n_total_jobs > 0:
            mp_context = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=attack_parallel_jobs, mp_context=mp_context) as executor:
                unlearn_results = list(tqdm(
                    executor.map(_unlearn_job_wrapper, all_unlearn_jobs),
                    total=n_total_jobs,
                    desc="Unlearning (parallel)",
                    disable=IN_LOGFILE,
                    mininterval=0.1
                ))

        # Phase 3: Aggregate results per sample
        # Group unlearn results by sample_idx
        results_by_sample = defaultdict(list)
        for result in unlearn_results:
            results_by_sample[result['sample_idx']].append(result)

        all_new_entries = []
        all_target_histories = []
        for prep in all_preps:
            sidx = prep['sample_idx']
            all_reference_scores.append(prep['ref_scores_array'])
            n_reused += prep['n_reused']

            # Combine cached scores + newly unlearned scores
            tgt_scores = list(prep['cached_tgt_scores'])
            for result in results_by_sample.get(sidx, []):
                tgt_scores.append(result['scores'])
                all_new_entries.append(result['entry_data'])
                if result['loss_history']:
                    all_target_histories.append(result['loss_history'])
                n_newly_unlearned += 1

            if len(tgt_scores) != n_shadows:
                raise RuntimeError(f"Expected {n_shadows} target shadows, got {len(tgt_scores)}")

            all_target_scores.append(np.array(tgt_scores))

        # Add all collected entries to manifest (parent process only)
        for entry_data in all_new_entries:
            manifest.add_entry(**entry_data)
    else:
        print(f"  Using sequential execution (attack_parallel_jobs=1)")

        # Sequential execution
        all_target_histories = []
        for sample_idx, forget_sample_id in enumerate(tqdm(attacked_sample_ids, desc="Attacking samples",
                                                            disable=IN_LOGFILE,
                                                            mininterval=0.1)):
            # Call extracted function
            idx, ref_scores, tgt_scores, n_reused_sample, n_newly_sample, new_entries, tgt_hists = attack_single_sample(
                sample_idx=sample_idx,
                forget_sample_id=forget_sample_id,
                csv_df=csv_df,
                metadata=metadata,
                n_shadows=n_shadows,
                eval_x=eval_x,
                eval_y=eval_y,
                model_type=model_type,
                loaders=loaders,
                device=device,
                manifest_path=manifest.manifest_path,
                forget_size=forget_size,
                full_dataset=full_dataset,
                rng=rng,
                base_algo=base_algo,
                shadow_r_f=shadow_r_f,
                method_params=method_params,
                weight_decay=weight_decay,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                use_average=use_average,
                use_project=use_project,
                seed_val=seed_val,
            )

            all_reference_scores.append(ref_scores)
            all_target_scores.append(tgt_scores)
            n_reused += n_reused_sample
            n_newly_unlearned += n_newly_sample
            all_target_histories.extend(tgt_hists)

            # Add entries to manifest (sequential mode)
            for entry_data in new_entries:
                manifest.add_entry(**entry_data)

    # Save manifest after all unlearning is done
    manifest.save()

    # End timing and print statistics
    sample_attack_elapsed = time.time() - sample_attack_start
    print(f"\nSample Attack Statistics:")
    print(f"  Total time:       {sample_attack_elapsed:8.2f}s ({sample_attack_elapsed/60:.1f} min)")
    print(f"  Avg per sample:   {sample_attack_elapsed / len(attacked_sample_ids):8.2f}s")
    print(f"  Cache reuse rate: {n_reused / max(1, n_reused + n_newly_unlearned) * 100:5.1f}%")
    print(f"  Models reused:    {n_reused}")
    print(f"  Models unlearned: {n_newly_unlearned}")
    print(f"  Manifest entries: {len(manifest.entries)} total")

    # Convert to arrays: shape (n_attack, n_shadows, n_attack)
    scores_reference = np.array(all_reference_scores)
    scores_target = np.array(all_target_scores)

    # Compute target model scores
    target_model_scores = get_lira_scores(model, eval_x, eval_y)

    # Compute per-sample privacy risk using Gaussian LiRA densities.
    privacy_risks = []

    # Collect per-sample attack data for visualization
    per_sample_attack_data = []

    for i in range(len(eval_x)):
        ref_dist = scores_reference[i, :, i]
        tgt_dist = scores_target[i, :, i]

        val = target_model_scores[i]

        log_ratio, (mu_ref, std_ref, mu_tgt, std_tgt) = gaussian_log_likelihood_ratio(
            val, ref_dist, tgt_dist
        )
        privacy_risks.append(log_ratio)

        # Store per-sample data for visualization (forget and test samples)
        per_sample_attack_data.append({
            'target_score': val,
            'mean_ref': mu_ref,
            'std_ref': std_ref,
            'mean_tgt': mu_tgt,
            'std_tgt': std_tgt,
            'likelihood_ratio': float(np.exp(np.clip(log_ratio, -700, 700))),
            'is_forget': i < n_forget,
        })

    privacy_risks = np.array(privacy_risks)

    predictions = privacy_risks > 0.0
    true_labels = np.array([1] * n_forget + [0] * n_test)

    tp = np.sum(predictions[:n_forget])
    fn = np.sum(~predictions[:n_forget])
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    fp = np.sum(predictions[n_forget:])
    tn = np.sum(~predictions[n_forget:])
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Use balanced accuracy as privacy risk metric
    privacy_risk = (tpr + (1 - fpr)) / 2.0

    print(f"\nPrivacy Risk Results:")
    print(f"  TPR (Recall on Forget): {tpr:.3f}")
    print(f"  FPR (False Positive on Test): {fpr:.3f}")
    print(f"  Balanced Accuracy: {privacy_risk:.3f}")

    shadow_histories = {
        'reference': [],
        'target': all_target_histories,
        'tpr': tpr,
        'fpr': fpr,
        'n_reused': n_reused,
        'n_newly_unlearned': n_newly_unlearned,
        'per_sample_data': per_sample_attack_data,
    }

    return privacy_risk, shadow_histories, None


def plot_main_figure(args, runs, all_run_results, r_f_values, figures_dir):
    """Plot the main risk-vs-privacy figure and save it.

    Returns (history_color_map, pretty_color_map) for use in subsequent plots.
    """
    # Set style for fanciness
    if args.original_style:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'lines.linewidth': 3,
            'lines.markersize': 12,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })
    else:
        # Modern, publication-quality styling with ICML latex fonts
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 17,
            'axes.labelweight': 'normal',
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 13,
            'lines.linewidth': 2.0,
            'lines.markersize': 12,
            'axes.linewidth': 0.8,
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#000000',
            'text.color': '#000000',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'grid.linewidth': 0.6,
            'grid.alpha': 0.25,
            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate marker sizes based on r_f (log scale)
    log_rf = np.log10(r_f_values)
    min_log = np.log10(args.min_rf)
    max_log = np.log10(args.max_rf)

    if max_log > min_log:
        norm_rf = (log_rf - min_log) / (max_log - min_log)
    else:
        norm_rf = np.zeros_like(log_rf)

    # Map normalized r_f to size range
    if args.original_style:
        sizes = 200 + 400 * norm_rf
    else:
        sizes = 180 + 350 * norm_rf  # Larger for double-column figures

    unlearn_handles = []
    unlearn_labels = []
    unlearn_colors = []
    unlearn_linestyles = []
    retrain_handles = []
    retrain_labels = []
    retrain_colors = []
    retrain_linestyles = []
    retrain_methods = {"gd", "sgd", "svrg"}

    # Color maps for consistent colours across all pipeline figures
    history_color_map = {}    # maps "method_name (timing)" -> (color, linestyle)
    pretty_color_map = {}     # maps _get_pretty_name(method_name) -> (color, linestyle)

    # Track scatter point coordinates for axis limit computation
    all_x_points = []
    all_y_points = []
    finite_excess = all_run_results[..., 0][np.isfinite(all_run_results[..., 0])]
    signed_risks = bool(finite_excess.size and np.any(finite_excess <= 0))

    # Modern color palette (vibrant, modern colors for visual impact)
    if args.original_style:
        colors = None  # Use default matplotlib colors
        linestyles = ['-'] * 20
    else:
        # Vibrant modern palette - orange colors made more distinct
        colors = ['#FF6B35', '#004E89', '#D4A017', '#06A77D', '#9B59B6',
                  '#E74C3C', '#3498DB', '#2ECC71', '#C70039', '#1ABC9C']
        # Line styles for B&W printing: solid, dashed, dotted, dashdot, etc.
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2)),
                     (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]

    for idx, (method_name, lr, decay, timing) in enumerate(runs):
        # Data for this method: [n_rf, seeds, 2]
        method_data = all_run_results[idx]

        # Signed test-loss differences cannot be displayed on a log axis.
        y_data = method_data[:, :, 0]
        if signed_risks:
            y_mean = np.nanmean(y_data, axis=1)
            y_err_lower = y_err_upper = np.nanstd(y_data, axis=1)
        else:
            with np.errstate(invalid='ignore'):
                log_y = np.log(y_data)
            y_mean = np.exp(np.nanmean(log_y, axis=1))
            y_geo_std = np.exp(np.nanstd(log_y, axis=1))
            y_err_lower = y_mean - (y_mean / y_geo_std)
            y_err_upper = (y_mean * y_geo_std) - y_mean
        y_err = [y_err_lower, y_err_upper]

        # X-axis: Privacy Risk (index 1) -> Arithmetic Mean & Std
        x_data = method_data[:, :, 1]

        # Mirror accuracy below 0.5 to above 0.5
        # An attacker can always flip predictions, so accuracy < 0.5 means
        # the "inverse" attack achieves 1 - accuracy > 0.5
        # Important: Mirror individual data points BEFORE calculating mean/std for correct error bars
        x_data_mirrored = np.where(x_data < 0.5, 1.0 - x_data, x_data)

        x_mean = np.nanmean(x_data_mirrored, axis=1)
        x_std = np.nanstd(x_data_mirrored, axis=1)

        # Mask infinite/NaN values to avoid matplotlib errors
        mask = np.isfinite(y_mean) & np.isfinite(x_mean)
        if not np.any(mask):
            continue

        x_mean_plot = x_mean[mask]
        y_mean_plot = y_mean[mask]
        sizes_plot = sizes[mask]

        label = _get_pretty_name(method_name)
        if 'vru' in method_name.lower():
            if args.show_std:
                label += f" (lr={lr}, $\\gamma$={decay})"
            else:
                label = 'VRU'

        # Select color and linestyle
        if colors is not None:
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
        else:
            color = None
            linestyle = '-'

        # Plot trajectory line with distinctive linestyle for B&W printing
        if args.original_style:
            line, = ax.plot(x_mean_plot, y_mean_plot, linestyle='-', linewidth=2, alpha=0.7, zorder=1, color=color)
            if color is None:
                color = line.get_color()
        else:
            line, = ax.plot(x_mean_plot, y_mean_plot, linestyle=linestyle, linewidth=2.0, alpha=0.7, zorder=1, color=color)

        # Populate color maps for consistent colours across sub-figures
        history_key = f"{method_name} ({timing})"
        history_color_map[history_key] = (color, linestyle)
        pretty_color_map[_get_pretty_name(method_name)] = (color, linestyle)

        # Track point coordinates for axis limits (points only, not error bars)
        all_x_points.extend(x_mean_plot.tolist())
        all_y_points.extend(y_mean_plot.tolist())

        # Plot error bars
        if args.show_std:
            x_std_plot = x_std[mask]
            y_err_lower_plot = y_err_lower[mask]
            y_err_upper_plot = y_err_upper[mask]
            if args.original_style:
                ax.errorbar(x_mean_plot, y_mean_plot, xerr=x_std_plot, yerr=[y_err_lower_plot, y_err_upper_plot], fmt='none', ecolor=color, alpha=0.5, capsize=3, zorder=1)
            else:
                ax.errorbar(x_mean_plot, y_mean_plot, xerr=x_std_plot, yerr=[y_err_lower_plot, y_err_upper_plot], fmt='none', ecolor=color, alpha=0.3, capsize=3.5, elinewidth=1.2, zorder=1)

        # Plot scatter points - no transparency to avoid seeing edges
        if args.original_style:
            sc = ax.scatter(x_mean_plot, y_mean_plot, s=sizes_plot, label=label, color=color, alpha=0.9, edgecolors='white', linewidth=1.5, marker='X', zorder=2)
        else:
            sc = ax.scatter(x_mean_plot, y_mean_plot, s=sizes_plot, label=label, color=color, alpha=1.0, edgecolors='white', linewidth=1.5, marker='o', zorder=2)

        if method_name in retrain_methods:
            retrain_handles.append(sc)
            retrain_labels.append(label)
            retrain_colors.append(color)
            retrain_linestyles.append(linestyle)
        else:
            unlearn_handles.append(sc)
            unlearn_labels.append(label)
            unlearn_colors.append(color)
            unlearn_linestyles.append(linestyle)

    ax.set_xlabel("Empirical Privacy Risk")
    ax.set_ylabel("Excess Risk")
    ax.set_xscale('linear')
    if not signed_risks:
        ax.set_yscale('log')
    # ax.set_title(f"Excess Risk vs Privacy Risk (T={args.T})")

    # Set axis limits based on scatter points only (ignore error bar extent)
    if all_x_points and all_y_points:
        x_min, x_max = min(all_x_points), max(all_x_points)
        y_min, y_max = min(all_y_points), max(all_y_points)
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.01
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        if signed_risks:
            y_pad = (y_max - y_min) * 0.05 if y_max > y_min else max(abs(y_min) * 0.1, 1e-8)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        else:
            log_y_min, log_y_max = np.log10(y_min), np.log10(y_max)
            log_y_pad = (log_y_max - log_y_min) * 0.05 if log_y_max > log_y_min else 0.1
            ax.set_ylim(10 ** (log_y_min - log_y_pad), 10 ** (log_y_max + log_y_pad))

    # Fancy grid and spines
    if args.original_style:
        ax.grid(True, which="major", linestyle='--', alpha=0.5, color='gray')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    else:
        # Modern, clean grid
        ax.grid(True, which="major", linestyle='-', alpha=0.15, color='#CCCCCC', zorder=0)
        ax.grid(True, which="minor", linestyle='-', alpha=0.08, color='#E0E0E0', zorder=0)
        # Complete square border with all four spines
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        # Add subtle offset to spines
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))

    # Legend for Methods - create composite handles showing both line and marker
    legend_handles = []
    legend_labels = []

    if unlearn_handles:
        # Sort to ensure VRU variants come first
        combined = list(zip(unlearn_labels, unlearn_handles, unlearn_colors, unlearn_linestyles))
        combined.sort(key=lambda x: 0 if x[0].startswith('VRU') else 1)
        unlearn_labels, unlearn_handles, unlearn_colors, unlearn_linestyles = zip(*combined)

        # Create composite handles: line + marker for each unlearning method
        for i, (handle, label) in enumerate(zip(unlearn_handles, unlearn_labels)):
            # Create a tuple of (line, marker) for the legend
            line_handle = Line2D([0], [0], color=unlearn_colors[i], linestyle=unlearn_linestyles[i],
                               linewidth=2.0, marker='o', markersize=12, markerfacecolor=unlearn_colors[i],
                               markeredgecolor='white', markeredgewidth=1.5)
            legend_handles.append(line_handle)
            legend_labels.append(label)

    if retrain_handles:
        legend_handles.append(Line2D([0], [0], color='none', marker='', linestyle=''))
        legend_labels.append(r"$\mathit{Retraining}$")
        # Create composite handles: line + marker for each retraining method
        for i, (handle, label) in enumerate(zip(retrain_handles, retrain_labels)):
            line_handle = Line2D([0], [0], color=retrain_colors[i], linestyle=retrain_linestyles[i],
                               linewidth=2.0, marker='o', markersize=12, markerfacecolor=retrain_colors[i],
                               markeredgecolor='white', markeredgewidth=1.5)
            legend_handles.append(line_handle)
            legend_labels.append("  " + label)

    # Common legend styling for consistency
    if args.original_style:
        legend_style = {'frameon': True, 'framealpha': 1.0, 'edgecolor': 'black',
                       'facecolor': 'white', 'fontsize': 'medium'}
        title_props = {'weight': 'semibold', 'size': 'medium'}
    else:
        legend_style = {'frameon': True, 'framealpha': 1.0, 'edgecolor': '#666666',
                       'facecolor': 'white', 'fontsize': 13, 'borderpad': 0.8,
                       'handletextpad': 0.6, 'labelspacing': 0.7, 'handlelength': 3}
        title_props = {'weight': 'bold', 'size': 13}

    # Legend for Sizes (r_f) - positioned on the left side
    legend_elements = []
    # Select min, mid, max indices for legend
    indices = sorted(list(set([0, args.n_rf // 2, args.n_rf - 1])), reverse=True)
    for i in indices:
        rf_val = r_f_values[i]
        s_val = sizes[i]
        base, exponent = f"{rf_val:.0e}".split('e')
        rf_str = f"{base}e{int(exponent)}"
        if args.original_style:
            legend_elements.append(ax.scatter([], [], s=s_val, c='gray', alpha=0.6, edgecolors='none', marker='X', label=f"$r_f$={rf_str}"))
        else:
            # Add white edge contours like the methods legend markers
            legend_elements.append(ax.scatter([], [], s=s_val, c='#666666', alpha=1.0,
                                            edgecolors='white', linewidths=1.5, marker='o', label=f"$r_f$={rf_str}"))

    # Position legends - move outside plot when show_std is enabled for readability
    if args.show_std:
        legend_rf = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 0.5),
                             title=r"Forget Ratio $r_f$", title_fontproperties=title_props,
                             alignment='left', **legend_style)
        ax.add_artist(legend_rf)

        ax.legend(legend_handles, legend_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                 title="Methods", title_fontproperties=title_props,
                 alignment='left', **legend_style)
    else:
        # Shared vertical nudge for both legends (axes fraction, negative = down)
        legends_y_offset = -0.01

        # Pin Methods legend at lower right as a permanent artist
        legend_methods = ax.legend(legend_handles, legend_labels, loc="lower right",
                                   bbox_to_anchor=(1.0, legends_y_offset), bbox_transform=ax.transAxes,
                                   title="Methods", title_fontproperties=title_props,
                                   alignment='left', **legend_style)
        ax.add_artist(legend_methods)

        # Temporarily place r_f legend to measure its rendered size
        ax.legend(handles=legend_elements, loc="lower right",
                  title=r"Forget Ratio $r_f$", title_fontproperties=title_props,
                  alignment='left', **legend_style)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        methods_bb = legend_methods.get_window_extent(renderer)
        rf_bb = ax.get_legend().get_window_extent(renderer)
        ax_bb = ax.get_window_extent(renderer)

        # r_f legend: right edge = methods legend left edge, bottoms aligned
        gap_px = 6
        anchor_x = (methods_bb.x0 - gap_px - rf_bb.width - ax_bb.x0) / ax_bb.width
        anchor_y = (methods_bb.y0 - ax_bb.y0) / ax_bb.height

        # Fine-tune offsets in axes fraction (negative = left / down)
        rf_x_offset = -0.02
        rf_y_offset = -0.02

        # Recreate at the computed position (lower-left corner = anchor point)
        ax.legend(handles=legend_elements, loc="lower left",
                  bbox_to_anchor=(anchor_x + rf_x_offset, anchor_y + rf_y_offset), bbox_transform=ax.transAxes,
                  title=r"Forget Ratio $r_f$", title_fontproperties=title_props,
                  alignment='left', **legend_style)

    # Save to default location (for backwards compatibility)
    plt.savefig(args.save_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved plot to {args.save_file}")

    # Save to organized figures directory with comprehensive hyperparameters
    main_plot_path = os.path.join(figures_dir, "risk_vs_privacy_main.pdf")
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"✓ Saved main plot to: {main_plot_path}")

    return history_color_map, pretty_color_map


def main():
    args = parse_args()

    # Apply dummy mode overrides for quick testing
    if args.dummy:
        print("\n" + "="*60)
        print("DUMMY MODE ENABLED - Using minimal parameters for quick testing")
        print("="*60)
        args.n_rf = 2
        args.seeds = 1
        args.n_shadows = 2
        args.T = 2
        print(f"  n_rf: {args.n_rf} (reduced from default 5)")
        print(f"  seeds: {args.seeds} (reduced from default 3)")
        print(f"  n_shadows: {args.n_shadows} (reduced from default 16)")
        print(f"  T (epoch budget): {args.T} (reduced from default 10)")
        print("="*60 + "\n")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}")
    print(f"Model Architecture: {args.model}")

    # Default runs if none provided
    if args.run is None:
        runs = [
            ("vru:1.0", "1e-1", "0.85", "after"),
            ("nft:1.0", "1e-3", "0.9", "before"),
        ]
    else:
        runs = args.run

    # Grid of r_f
    if args.rf_values is not None:
        r_f_values = np.asarray(args.rf_values, dtype=float)
        if (np.any(r_f_values <= 0) or np.any(r_f_values >= 1)
                or len(set(r_f_values)) != len(r_f_values)
                or np.any(np.diff(r_f_values) <= 0)):
            raise ValueError("--rf_values must be unique, increasing fractions in (0, 1)")
        args.min_rf = float(r_f_values[0])
        args.max_rf = float(r_f_values[-1])
        args.n_rf = len(r_f_values)
    else:
        r_f_values = np.logspace(np.log10(args.min_rf), np.log10(args.max_rf), args.n_rf)

    # Extract kappa values from methods for figure naming
    kappa_values = set()
    for method_str, lr, decay, timing in runs:
        _, kappa = parse_method_with_kappa(method_str)
        base_method, _, _ = parse_vru_variant(method_str.split(":")[0].lower())
        # Only include kappa if method uses DP noise
        if timing not in ["none"] and base_method not in ["neggrad+", "scrub"]:
            kappa_values.add(kappa)

    # Create kappa string for figure naming
    if kappa_values:
        if len(kappa_values) == 1:
            kappa_str = f"k{list(kappa_values)[0]:g}"
        else:
            min_k = min(kappa_values)
            max_k = max(kappa_values)
            kappa_str = f"k{min_k:g}-{max_k:g}"
    else:
        kappa_str = "k0"

    # Create comprehensive hyperparameter string for figure directory and filenames
    hp_str = f"d{args.dataset}_m{args.model}_T{args.T:g}_{kappa_str}_ns{args.n_shadows}_s{args.seeds}_wd{args.weight_decay:g}_bs{args.batch_size}_rf{args.min_rf:.1e}-{args.max_rf:.1e}_n{args.n_rf}"

    # Create temporary output directory for this run's figures (will be renamed at the end
    # with completion timestamp and SLURM job ID for easy identification)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    figures_dir = os.path.join("risk_vs_privacy_figures", f"_in_progress_j{job_id}_{hp_str}")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"FIGURES DIRECTORY (in progress, job {job_id}): {figures_dir}")
    print(f"{'='*80}\n")

    # Store results: [method, r_f, seed, 2] -> last dim: (excess_risk, privacy_risk)
    all_run_results = np.zeros((len(runs), args.n_rf, args.seeds, 2))

    # Indices to plot history for (first, middle, last)
    indices_to_plot = sorted(list(set([0, args.n_rf // 2, args.n_rf - 1])))
    history_results = {idx: {} for idx in indices_to_plot}
    shadow_trajectory_data = {idx: {} for idx in indices_to_plot}
    mia_attack_data = {idx: {} for idx in indices_to_plot}
    scratch_losses_for_plot = {}

    # Load cache
    cache = {}
    if os.path.exists(args.cache_file):
        try:
            with open(args.cache_file, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded cache from {args.cache_file} ({len(cache)} entries)")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")

    for seed_val in range(args.seeds):
        print(f"\n=== Processing Seed {seed_val}/{args.seeds - 1} ===")
        set_seed(seed_val)

        # Auto-discover shadow CSV for U-LiRA evaluation (required)
        shadow_csv_path = auto_discover_shadow_csv(
            dataset=args.dataset,
            model_type=args.model,
            weight_decay=args.weight_decay,
            n_shadows=args.n_shadows,
            shadow_manifests_dir=args.cache_dir
        )
        if shadow_csv_path is None:
            raise RuntimeError(
                f"Shadow CSV not found. Generate shadows first with:\n"
                f"  python generate_shadow_csv.py --dataset {args.dataset} --model {args.model} "
                f"--weight_decay {args.weight_decay} --N {args.n_shadows}"
            )
        print(f"Using shadow CSV: {shadow_csv_path}")
        shadow_fingerprint = shadow_manifest_fingerprint(shadow_csv_path)

        # 1. Data Loaders (Full)
        train_val_test_sets, loaders = get_data_loaders(
            batch_size=args.batch_size,
            val_size=0.0,
            test_size=0.0,
            seed=seed_val,
            standardize=True,
            dataset=args.dataset,
        )
        train_ds, _, _ = train_val_test_sets
        full_train_ds = train_ds  # Keep reference to full train set for CSV-based MIA

        # Subsample train_ds to half to match shadow base model training size
        # The other half becomes held-out data for true non-member sampling in MIA
        n_full = len(train_ds)
        n_half = n_full // 2
        generator = torch.Generator().manual_seed(seed_val + 9999)
        train_ds, held_out_ds = torch.utils.data.random_split(train_ds, [n_half, n_full - n_half], generator=generator)
        # Rebuild loaders with the subsampled training set
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        full_train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        loaders = Loaders(
            train=train_loader,
            full_train=full_train_loader,
            val=loaders.val,
            test=loaders.test,
            n_features=loaders.n_features,
            n_classes=loaders.n_classes,
            n_train=len(train_ds),
        )
        held_out_eval_loader = DataLoader(held_out_ds, batch_size=512, shuffle=False)
        print(f"  Subsampled train set to {n_half} samples (matching shadow base model size)")
        print(f"  Held-out set: {len(held_out_ds)} samples for true non-member MIA evaluation")

        # 2. Ensure Full Model (independent of r_f)
        full_path = _ensure_pretrained_full_model(
            seed=seed_val,
            weight_decay=args.weight_decay,
            model_type=args.model,
            loaders=loaders,
            device=device,
            dataset=args.dataset,
            half_train=True,
        )
        base_model_template = _load_full_base(full_path, loaders, device, model_type=args.model)
        full_hash = get_file_hash(full_path)

        for i, r_f in enumerate(tqdm(r_f_values, desc=f"Seed {seed_val} r_f loop",
                                      disable=IN_LOGFILE,
                                      mininterval=0.1)):
            # a. Split Data
            uloaders = get_retain_forget_loaders(
                train_ds=train_ds,
                r_f=r_f,
                seed=seed_val,
                batch_size=args.batch_size,
            )

            # b. Ensure Scratch Model
            scratch_path = _ensure_scratch_model(
                seed=seed_val,
                weight_decay=args.weight_decay,
                r_f=r_f,
                model_type=args.model,
                loaders=loaders,
                uloaders=uloaders,
                device=device,
                dataset=args.dataset,
                half_train=True,
            )
            scratch_hash = get_file_hash(scratch_path)

            # The paper evaluates final excess risk on the held-out target split.
            scratch_model = build_model(args.model, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
            try:
                scratch_state = torch.load(scratch_path, map_location=device, weights_only=True)
            except TypeError:
                scratch_state = torch.load(scratch_path, map_location=device)
            for k, v in list(scratch_state.items()):
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    scratch_state[k] = v.to(dtype=torch.float64)
            scratch_model.load_state_dict(scratch_state)
            scratch_retain_loss = _evaluate_retain_loss(scratch_model, uloaders, device, args.weight_decay)
            if seed_val == args.seeds - 1 and i in indices_to_plot:
                scratch_losses_for_plot[i] = scratch_retain_loss

            scratch_loss = _evaluate_loader_loss(scratch_model, held_out_eval_loader, device, args.weight_decay)

            # Use the independently retrained scratch model as the fixed risk baseline.
            method_results_buffer = {}

            # Check which methods need running
            methods_to_run = []
            for idx, (m_name_raw, m_lr, m_decay, m_time) in enumerate(runs):
                # Parse method name and kappa
                m_name, m_kappa = parse_method_with_kappa(m_name_raw)

                # Parse VRU variant to get flags for cache key
                _, m_use_avg, m_use_proj = parse_vru_variant(m_name)
                if args.average:
                    m_use_avg = True
                if args.project:
                    m_use_proj = True

                eff_lr = float(m_lr)
                if m_time == "before":
                    eff_lr = float(eff_lr * (r_f / args.min_rf)**0.5)

                # Validate kappa for methods that need DP noise
                base_method, _, _ = parse_vru_variant(m_name.lower())
                eff_kappa = 0.0
                if m_time not in ["none"] and base_method not in ["neggrad+", "scrub"]:
                    if m_kappa == 0.0 and ":" not in m_name_raw:
                        raise ValueError(
                            f"Method '{m_name}' with timing='{m_time}' requires kappa_dp. "
                            f"Use format: '{m_name}:kappa' (e.g., '{m_name}:0.1')"
                        )
                    eff_kappa = m_kappa

                # Build cache key
                key = (m_name, eff_lr, m_decay, m_time, seed_val, r_f,
                       args.weight_decay, args.dataset, args.batch_size, eff_kappa, args.n_shadows, args.T, args.model,
                       scratch_hash, full_hash, m_use_avg, m_use_proj,
                       CACHE_PROTOCOL_VERSION, shadow_fingerprint)

                # Run if not in cache OR if privacy_risk is missing from cache entry
                if key not in cache:
                    methods_to_run.append(idx)
                elif 'privacy_risk' not in cache[key] or 'loss' not in cache[key] or cache[key].get('privacy_risk', 0.0) == 0.0:
                    methods_to_run.append(idx)

            # if not methods_to_run:
            #     # Load from cache
            #     for idx, (m_name, m_lr, m_decay, m_time) in enumerate(runs):
            #         eff_lr = float(m_lr)
            #         if m_time == "before" or m_name in ["neggrad+", "scrub"]:
            #             eff_lr = float(eff_lr * (r_f / args.min_rf)**0.5)

            #         key = (m_name, eff_lr, m_decay, m_time, seed_val, r_f,
            #                args.weight_decay, args.dataset, args.batch_size, args.kappa_dp, args.n_shadows, args.T, args.model, args.shadow_retrain_algo, args.shadow_retrain_lr, args.shadow_retrain_discount, args.shadow_retrain_epochs,
            #                scratch_hash, full_hash)
            #         entry = cache[key]
            #         # We store raw loss in buffer, excess risk computed later
            #         method_results_buffer[idx] = {'loss': entry['loss'], 'privacy_risk': entry.get('privacy_risk', 0.0)}

            #         # Recalculate if 0.0 (legacy cache)
            #         p_risk = entry.get('privacy_risk', 0.0)
            #         if p_risk == 0.0:
            #             p_risk, _, _ = evaluate_privacy_risk(
            #                 None, None, None,
            #                 method_name=m_name,
            #                 r_f=r_f,
            #                 base_model_template=base_model_template,
            #                 loaders=loaders,
            #                 method_params=None, # Can't easily reconstruct without parsing
            #                 n_shadows=args.n_shadows,
            #                 seed_val=seed_val,
            #                 weight_decay=args.weight_decay,
            #                 model_type=args.model,
            #                 shadow_retrain_algo=args.shadow_retrain_algo,
            #                 shadow_retrain_lr=args.shadow_retrain_lr,
            #                 shadow_retrain_discount=args.shadow_retrain_discount,
            #                 shadow_retrain_epochs=args.shadow_retrain_epochs,
            #             )
            #             method_results_buffer[idx]['privacy_risk'] = p_risk

            #         # Load history if needed
            #         if seed_val == args.seeds - 1 and i in indices_to_plot:
            #             s_hists = entry.get('shadow_histories')
            #             method_label = f"{m_name} ({m_time})"
            #             if s_hists:
            #                 if method_label not in shadow_trajectory_data[i]:
            #                     shadow_trajectory_data[i][method_label] = {'retrain': [], 'unlearn': []}
            #                 if s_hists.get('retrain'):
            #                     shadow_trajectory_data[i][method_label]['retrain'].extend(s_hists['retrain'])
            #                 if s_hists.get('unlearn'):
            #                     shadow_trajectory_data[i][method_label]['unlearn'].extend(s_hists['unlearn'])

            #             hist = entry.get('history')
            #             if hist is not None:
            #                 history_results[i][f"{m_name} ({m_time})"] = hist
            #                 history_results[i][f"{m_name} ({m_time})::final_loss"] = entry['loss']
            #     continue

            # c. Run Methods
            for idx, (method_name_raw, lr_str, decay_str, timing) in enumerate(runs):
                # Isolate the randomness of this method run from previous ones (and their MIAs).
                # This ensures that changing parameters of Method A doesn't alter the training trajectory of Method B.
                set_seed(seed_val * 1000 + idx)

                # Parse method name and kappa
                method_name, method_kappa = parse_method_with_kappa(method_name_raw)

                # Parse VRU variant name early to use in cache key and execution
                base_method, use_average, use_project = parse_vru_variant(method_name)
                if args.average:
                    use_average = True
                if args.project:
                    use_project = True

                lr = float(lr_str)
                if timing == "before":
                    lr = float(lr * (r_f / args.min_rf)**0.5)

                # Validate and set kappa for methods that need DP noise
                eff_kappa = 0.0
                if timing not in ["none"] and base_method not in ["neggrad+", "scrub"]:
                    if method_kappa == 0.0 and ":" not in method_name_raw:
                        raise ValueError(
                            f"Method '{method_name}' with timing='{timing}' requires kappa_dp. "
                            f"Use format: '{method_name}:kappa' (e.g., '{method_name}:0.1')"
                        )
                    eff_kappa = method_kappa

                # Build cache key
                key = (method_name, lr, decay_str, timing, seed_val, r_f,
                       args.weight_decay, args.dataset, args.batch_size, eff_kappa, args.n_shadows, args.T, args.model,
                       scratch_hash, full_hash, use_average, use_project,
                       CACHE_PROTOCOL_VERSION, shadow_fingerprint)

                if idx not in methods_to_run:
                    entry = cache[key]
                    method_results_buffer[idx] = {'loss': entry['loss'], 'privacy_risk': entry.get('privacy_risk', 0.0)}

                    # # Recalculate if 0.0 (legacy cache)
                    # p_risk = entry.get('privacy_risk', 0.0)
                    # if p_risk == 0.0 and entry.get('loss', 0.0) != 0.0:
                    #     p_risk, _, _ = evaluate_privacy_risk(
                    #         None, None, None,
                    #         method_name=method_name,
                    #         r_f=r_f,
                    #         base_model_template=base_model_template,
                    #         loaders=loaders,
                    #         method_params=None,
                    #         n_shadows=args.n_shadows,
                    #         seed_val=seed_val,
                    #         weight_decay=args.weight_decay,
                    #         model_type=args.model,
                    #         shadow_retrain_algo=args.shadow_retrain_algo,
                    #         shadow_retrain_lr=args.shadow_retrain_lr,
                    #         shadow_retrain_discount=args.shadow_retrain_discount,
                    #         shadow_retrain_epochs=args.shadow_retrain_epochs,
                    #     )
                    #     method_results_buffer[idx]['privacy_risk'] = p_risk

                    # Load history if needed
                    if seed_val == args.seeds - 1 and i in indices_to_plot:
                        s_hists = entry.get('shadow_histories')
                        method_label = _get_pretty_name(method_name)
                        history_key = f"{method_name_raw} ({timing})"
                        if s_hists:
                            if method_label not in shadow_trajectory_data[i]:
                                shadow_trajectory_data[i][method_label] = {'reference': [], 'target': []}
                            if s_hists.get('reference'):
                                shadow_trajectory_data[i][method_label]['reference'].extend(s_hists['reference'])
                            if s_hists.get('target'):
                                shadow_trajectory_data[i][method_label]['target'].extend(s_hists['target'])

                            # Load MIA attack data from cache
                            if method_label not in mia_attack_data[i]:
                                mia_attack_data[i][method_label] = []
                            per_sample = s_hists.get('per_sample_data', [])
                            if per_sample:
                                mia_attack_data[i][method_label] = per_sample

                        hist = entry.get('history')
                        if hist is not None:
                            history_results[i][history_key] = hist
                            history_results[i][f"{history_key}::final_loss"] = entry['loss']
                    continue

                decay = float(decay_str)

                retrain_methods = {"gd", "sgd", "svrg"}
                is_retrain = base_method in retrain_methods

                # For neggrad+ and scrub, timing is alpha, and noise is not used.
                alpha_val = 0.05  # Default value
                noise_timing = timing
                if base_method in ["neggrad+", "scrub"]:
                    try:
                        alpha_val = float(timing)
                        noise_timing = "none"
                    except ValueError:
                        raise ValueError(
                            f"For '{method_name}', the 'timing' parameter must be a float for alpha, but got '{timing}'"
                        )

                if is_retrain and noise_timing != "none":
                    raise ValueError(f"Retraining method '{method_name}' must be run with timing='none', got '{noise_timing}'.")

                current_uloaders = uloaders
                run_method_name = base_method

                if base_method == "gd":
                    run_method_name = "nft"
                    current_uloaders = UnlearnLoaders(
                        retain=uloaders.retain_full,
                        retain_full=uloaders.retain_full,
                        forget=uloaders.forget,
                        n_retain=uloaders.n_retain,
                        n_forget=uloaders.n_forget,
                    )
                elif base_method == "sgd":
                    run_method_name = "nft"

                if base_method == "gd":
                     cost_per_epoch = uloaders.n_retain
                else:
                     batches_per_epoch = len(uloaders.retain)
                     cost_per_epoch = batches_per_epoch * args.batch_size

                if cost_per_epoch == 0:
                    all_run_results[idx, i, seed_val, :] = np.nan
                    continue

                double_grad_methods = {"vru", "cheat-vru", "svrg", "seqvru", "neggrad+"}

                # SCRUB has cost (1.25 + 1.5*r_f) per iteration
                if run_method_name.lower() == "scrub":
                    grads_per_sample = 1.25 + 1.5 * r_f
                else:
                    grads_per_sample = 2 if run_method_name.lower() in double_grad_methods else 1

                T_effective = args.T
                if run_method_name.lower() == "svrg":
                    T_effective -= 3

                epochs = int(T_effective / grads_per_sample)
                epochs = max(1, epochs)

                if is_retrain:
                    model = build_model(args.model, loaders.n_features, loaders.n_classes).to(device, dtype=torch.float64)
                else:
                    model = copy.deepcopy(base_model_template).to(device, dtype=torch.float64)

                # Noise Before
                if noise_timing == "before":
                    d_init = torch.norm(
                        _param_vector(model) - _param_vector(scratch_model), p=2
                    ).item()
                    std_init = eff_kappa * d_init
                    rng = np.random.default_rng(seed_val + i * 10000)
                    apply_gaussian_noise_to_model(model, std_init, rng, device)

                checkpoints, history = run_unlearning_with_checkpoints(
                    method=run_method_name,
                    base_model=model,
                    loaders=loaders,
                    uloaders=current_uloaders,
                    r_f=r_f,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    scheduler="step",
                    scheduler_step_freq=1,
                    discount_factor=decay,
                    weight_decay=args.weight_decay,
                    checkpoint_every=len(current_uloaders.retain),
                    scratch_best_path=scratch_path if scratch_path else "dummy_path.pt",
                    scratch_best_loss=scratch_loss,
                    alpha=alpha_val,
                    verbose=True,  # Main unlearning run should show progress
                    average=use_average,
                    project=use_project,
                )

                # Store history if this is one of the r_f values to plot (only for last seed)
                excess_hist = None
                if seed_val == args.seeds - 1 and i in indices_to_plot and history is not None:
                    # Handle potential dict return or list
                    vals = history
                    if isinstance(history, dict):
                        vals = history.get('retain_loss', history.get('loss', []))

                    # Convert to float and compute excess risk
                    try:
                        # Handle list of tuples [(step, loss), ...] from fast_grid
                        if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], tuple):
                            clean_vals = [float(v[1]) for v in vals]
                        else:
                            clean_vals = [v.item() if torch.is_tensor(v) else v for v in vals]
                        excess_hist = clean_vals # Store raw loss, convert to excess later
                        history_results[i][f"{method_name_raw} ({timing})"] = excess_hist
                    except Exception as e:
                        print(f"  Warning: Could not process history for {method_name}: {e}")

                if not checkpoints:
                    final_state = model.state_dict()
                else:
                    final_state = checkpoints[-1][1]

                model.load_state_dict(final_state)

                # Save checkpoint for future MIA evaluation
                ckpt_dir = "checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                run_id = hashlib.md5(str(key).encode()).hexdigest()
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_{run_id}.pt")
                torch.save(final_state, ckpt_path)

                # Evaluate Excess Risk
                if noise_timing == "after":
                    loss = averaged_noised_retain_loss(
                        model,
                        best_scratch_model=scratch_model,
                        kappa_dp=eff_kappa,
                        repetitions=10,
                        uloaders=uloaders,
                        device=device,
                        weight_decay=args.weight_decay,
                        rng=np.random.default_rng(seed_val + i * 10000),
                        eval_loader=held_out_eval_loader,
                    )
                elif noise_timing == "empirical":
                    std_empirical = eff_kappa * r_f / (1.0 - r_f)
                    loss = averaged_empirical_noised_retain_loss(
                        model,
                        std=std_empirical,
                        repetitions=10,
                        uloaders=uloaders,
                        device=device,
                        weight_decay=args.weight_decay,
                        rng=np.random.default_rng(seed_val + i * 10000),
                        eval_loader=held_out_eval_loader,
                    )
                else:
                    loss = _evaluate_loader_loss(model, held_out_eval_loader, device, args.weight_decay)

                # Update history with final risk if needed (to show noise jump) - only last seed
                if seed_val == args.seeds - 1 and i in indices_to_plot:
                    history_key = f"{method_name_raw} ({timing})"
                    if history_key in history_results[i]:
                        history_results[i][f"{history_key}::final_loss"] = loss

                # Apply kappa_dp noise to model before privacy evaluation
                # (must match the noise used for excess risk evaluation)
                if noise_timing == "after" and eff_kappa > 0:
                    d = torch.norm(
                        _param_vector(model) - _param_vector(scratch_model), p=2
                    ).item()
                    std_priv = eff_kappa * d
                    apply_gaussian_noise_to_model(
                        model, std_priv,
                        np.random.default_rng(seed_val + i * 10000 + 1), device)
                elif noise_timing == "empirical" and eff_kappa > 0:
                    std_priv = eff_kappa * r_f / (1.0 - r_f)
                    apply_gaussian_noise_to_model(
                        model, std_priv,
                        np.random.default_rng(seed_val + i * 10000 + 1), device)

                # Evaluate Privacy Risk
                m_params = {
                    'lr': lr,
                    'decay': decay,
                    'timing': timing,
                    'epochs': epochs,
                    'kappa_dp': eff_kappa,
                    'scratch_model': scratch_model,
                    'alpha': alpha_val,
                    'method_name': method_name,  # For conditional hash construction
                }

                do_debug = (seed_val == args.seeds - 1) and (i in indices_to_plot)

                # CSV-based U-LiRA with unlearned model cache
                # Extract forget indices from the forget subset
                forget_subset_ds = uloaders.forget.dataset
                if hasattr(forget_subset_ds, 'indices'):
                    forget_indices_local = list(forget_subset_ds.indices)
                else:
                    forget_indices_local = list(range(len(forget_subset_ds)))

                # Map local indices (within train_ds subset) to original dataset indices
                # needed for CSV lookup. train_ds.indices maps position -> original index.
                if hasattr(train_ds, 'indices'):
                    forget_indices_list = [train_ds.indices[idx] for idx in forget_indices_local]
                else:
                    forget_indices_list = forget_indices_local

                privacy_risk, shadow_histories, _ = evaluate_privacy_risk_csv(
                    model,
                    full_dataset=full_train_ds,
                    forget_indices=forget_indices_list,
                    csv_path=shadow_csv_path,
                    device=device,
                    method_name=method_name,
                    method_params=m_params,
                    n_shadows=args.n_shadows,
                    seed_val=seed_val,
                    weight_decay=args.weight_decay,
                    dataset_name=args.dataset,
                    model_type=args.model,
                    loaders=loaders,
                    debug=do_debug,
                    cache_dir=args.cache_dir,
                    held_out_dataset=held_out_ds,
                    attack_parallel_jobs=args.attack_parallel_jobs,
                )

                method_results_buffer[idx] = {'loss': loss, 'privacy_risk': privacy_risk}
                excess_risk_temp = loss - scratch_loss

                if seed_val == args.seeds - 1 and i in indices_to_plot:
                    method_label = _get_pretty_name(method_name)
                    if method_label not in shadow_trajectory_data[i]:
                        shadow_trajectory_data[i][method_label] = {'reference': [], 'target': []}
                    shadow_trajectory_data[i][method_label]['reference'].extend(shadow_histories['reference'])
                    shadow_trajectory_data[i][method_label]['target'].extend(shadow_histories['target'])

                    # Collect MIA attack data for visualization
                    if method_label not in mia_attack_data[i]:
                        mia_attack_data[i][method_label] = []
                    mia_attack_data[i][method_label] = shadow_histories.get('per_sample_data', [])


                print(f"    Excess Risk (vs scratch): {excess_risk_temp:.4e} | Privacy Risk: {privacy_risk:.4e}")

                # Update Cache
                if key not in cache:
                    cache[key] = {}
                cache[key].update({
                    'excess_risk': excess_risk_temp,
                    'loss': np.float64(loss),
                    'privacy_risk': privacy_risk,
                    'history': excess_hist,
                    'ckpt_path': ckpt_path,
                    'shadow_histories': shadow_histories if (seed_val == args.seeds - 1 and i in indices_to_plot) else None,
                })

            # Save cache periodically
            with open(args.cache_file, "wb") as f:
                pickle.dump(cache, f)

            # Finalize excess risk for this r_f/seed against the fixed scratch loss.
            for idx, res in method_results_buffer.items():
                final_excess = res['loss'] - scratch_loss
                all_run_results[idx, i, seed_val, 0] = final_excess
                all_run_results[idx, i, seed_val, 1] = res['privacy_risk']

            # History traces remain on retain loss rather than test loss.
            if seed_val == args.seeds - 1 and i in indices_to_plot:
                scratch_losses_for_plot[i] = scratch_retain_loss

    # 4. Plotting
    history_color_map, pretty_color_map = plot_main_figure(args, runs, all_run_results, r_f_values, figures_dir)

    # Plot shadow histories
    _plot_shadow_histories(
        shadow_trajectory_data,
        r_f_values,
        indices_to_plot,
        scratch_losses_for_plot,
        args.save_file,
        figures_dir,
        pretty_color_map,
    )

    # Plot MIA attack performance
    _plot_mia_attack_performance(
        mia_attack_data,
        r_f_values,
        indices_to_plot,
        args.save_file,
        figures_dir,
    )

    # 5. Plotting Histories
    if len(indices_to_plot) > 0:
        # Create subplots: 1 row, N columns
        fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(5 * len(indices_to_plot), 5), constrained_layout=True)
        if len(indices_to_plot) == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices_to_plot):
            r_f_val = r_f_values[idx]
            ax.set_title(f"$r_f$ = {r_f_val:.4e}")
            ax.set_xlabel("Epochs/Steps")
            ax.set_ylabel("Excess Risk")
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.5)

            for label, hist in history_results[idx].items():
                if "::final" in label:
                    continue

                scratch_loss_final = scratch_losses_for_plot.get(idx, 0.0)
                # Filter positive values for log plot to avoid blank plots if values are <= 0
                valid_points = [(x, (y - scratch_loss_final) + 1e-15) for x, y in enumerate(hist) if (y - scratch_loss_final) + 1e-15 > 0]
                if valid_points:
                    xs, ys = zip(*valid_points)
                    # Use consistent colour from main plot
                    hist_color, hist_ls = history_color_map.get(label, (None, '-'))
                    line, = ax.plot(xs, ys, label=label, color=hist_color, linestyle=hist_ls)

                    # Check for final risk point (to visualize noise jump for 'after' methods)
                    final_key = f"{label}::final_loss"
                    if final_key in history_results[idx]:
                        final_risk = (history_results[idx][final_key] - scratch_loss_final) + 1e-15
                        if final_risk > 0:
                            ax.plot(xs[-1], final_risk, marker='*', markersize=8, color=line.get_color())
                            ax.plot([xs[-1], xs[-1]], [ys[-1], final_risk], linestyle=':', color=line.get_color(), alpha=0.7)
            ax.legend()

        # Save to default location (backwards compatibility)
        base, ext = os.path.splitext(args.save_file)
        history_file = f"{base}_history{ext}"
        plt.savefig(history_file, bbox_inches='tight')
        print(f"Saved history plot to {history_file}")

        # Save to organized figures directory
        history_file_organized = os.path.join(figures_dir, "risk_vs_privacy_history.pdf")
        plt.savefig(history_file_organized, bbox_inches='tight')
        print(f"✓ Saved history plot to: {history_file_organized}")

    # Save underlying figure data as CSVs
    data_dir = os.path.join(figures_dir, "figure_data")
    os.makedirs(data_dir, exist_ok=True)
    print(f"\nSaving figure data to: {data_dir}")
    _save_main_plot_data(all_run_results, runs, r_f_values, args.seeds, data_dir)
    _save_history_data(history_results, scratch_losses_for_plot, r_f_values,
                       indices_to_plot, data_dir)
    _save_shadow_trajectory_data(shadow_trajectory_data, r_f_values,
                                 indices_to_plot, scratch_losses_for_plot, data_dir)
    _save_mia_attack_data(mia_attack_data, r_f_values, indices_to_plot, data_dir)

    # Rename figures directory with completion timestamp and SLURM job ID
    dt_str = datetime.datetime.now().strftime("%y-%m-%d-%H%M")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    final_figures_dir = os.path.join("risk_vs_privacy_figures", f"{dt_str}_j{job_id}_{hp_str}")
    os.rename(figures_dir, final_figures_dir)
    figures_dir = final_figures_dir

    # Final summary of where all figures are saved
    print(f"\n{'='*80}")
    print(f"SUMMARY: All figures and data saved to organized directory:")
    print(f"  {figures_dir}")
    print(f"  {os.path.join(figures_dir, 'figure_data')}/")
    print(f"{'='*80}\n")

    plt.show()

def _save_main_plot_data(
    all_run_results: np.ndarray,
    runs: list,
    r_f_values: np.ndarray,
    n_seeds: int,
    data_dir: str,
):
    """Save underlying data for the main risk_vs_privacy plot to CSV."""
    path = os.path.join(data_dir, "risk_vs_privacy_main.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "lr", "decay", "timing", "r_f", "seed",
                     "excess_risk", "privacy_risk"])
        for idx, (method_name, lr, decay, timing) in enumerate(runs):
            for i, r_f in enumerate(r_f_values):
                for s in range(n_seeds):
                    er = all_run_results[idx, i, s, 0]
                    pr = all_run_results[idx, i, s, 1]
                    w.writerow([method_name, lr, decay, timing, f"{r_f:.6e}",
                                s, f"{er:.10e}", f"{pr:.10e}"])
    print(f"  Saved main plot data to: {path}")


def _save_history_data(
    history_results: Dict,
    scratch_losses_for_plot: Dict,
    r_f_values: np.ndarray,
    indices_to_plot: list,
    data_dir: str,
):
    """Save underlying data for the excess-risk history plot to CSV."""
    path = os.path.join(data_dir, "risk_vs_privacy_history.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_f", "method", "epoch", "excess_risk", "final_excess_risk"])
        for idx in indices_to_plot:
            r_f = r_f_values[idx]
            scratch_loss = scratch_losses_for_plot.get(idx, 0.0)
            for label, hist in history_results[idx].items():
                if "::final" in label:
                    continue
                final_key = f"{label}::final_loss"
                final_loss = history_results[idx].get(final_key, None)
                final_er = ""
                if final_loss is not None:
                    final_er = f"{(final_loss - scratch_loss) + 1e-15:.10e}"
                for epoch, val in enumerate(hist):
                    er = (val - scratch_loss) + 1e-15
                    w.writerow([f"{r_f:.6e}", label, epoch, f"{er:.10e}", final_er])
    print(f"  Saved history data to: {path}")


def _save_shadow_trajectory_data(
    shadow_data: Dict,
    r_f_values: np.ndarray,
    indices_to_plot: list,
    scratch_losses: Dict,
    data_dir: str,
):
    """Save underlying data for the shadow trajectories plot to CSV."""
    path = os.path.join(data_dir, "risk_vs_privacy_shadow_trajectories.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_f", "method", "shadow_type", "shadow_idx", "step", "excess_risk"])
        for r_f_idx in indices_to_plot:
            r_f = r_f_values[r_f_idx]
            scratch_loss = scratch_losses.get(r_f_idx, 0.0)
            method_data = shadow_data.get(r_f_idx, {})
            # Reference shadows (shared across methods — write once from first method)
            ref_written = False
            for method_label, histories in method_data.items():
                if not ref_written:
                    for i, hist in enumerate(histories.get('reference', [])):
                        if not hist:
                            continue
                        if isinstance(hist[0], tuple):
                            for step, loss in hist:
                                w.writerow([f"{r_f:.6e}", "_reference", "reference",
                                            i, step, f"{loss - scratch_loss:.10e}"])
                        else:
                            for step, loss in enumerate(hist):
                                w.writerow([f"{r_f:.6e}", "_reference", "reference",
                                            i, step, f"{loss - scratch_loss:.10e}"])
                    ref_written = True
                for i, hist in enumerate(histories.get('target', [])):
                    if not hist:
                        continue
                    for step, loss in hist:
                        w.writerow([f"{r_f:.6e}", method_label, "target",
                                    i, step, f"{loss - scratch_loss:.10e}"])
    print(f"  Saved shadow trajectory data to: {path}")


def _save_mia_attack_data(
    mia_data: Dict,
    r_f_values: np.ndarray,
    indices_to_plot: list,
    data_dir: str,
):
    """Save underlying data for the MIA attack performance plot to CSV."""
    path = os.path.join(data_dir, "risk_vs_privacy_mia_attack.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_f", "method", "sample_idx", "is_forget",
                     "target_score", "mean_ref", "std_ref", "mean_tgt", "std_tgt",
                     "likelihood_ratio"])
        for r_f_idx in indices_to_plot:
            r_f = r_f_values[r_f_idx]
            for method_label in sorted(mia_data.get(r_f_idx, {}).keys()):
                per_sample = mia_data[r_f_idx][method_label]
                for i, sample in enumerate(per_sample):
                    w.writerow([
                        f"{r_f:.6e}", method_label, i,
                        sample.get('is_forget', True),
                        f"{sample['target_score']:.10e}",
                        f"{sample['mean_ref']:.10e}",
                        f"{sample['std_ref']:.10e}",
                        f"{sample['mean_tgt']:.10e}",
                        f"{sample['std_tgt']:.10e}",
                        f"{sample.get('likelihood_ratio', 1.0):.10e}",
                    ])
    print(f"  Saved MIA attack data to: {path}")


def _plot_shadow_histories(
    shadow_data: Dict,
    r_f_values: np.ndarray,
    indices_to_plot: list,
    scratch_losses: Dict,
    save_file: str,
    figures_dir: str,
    pretty_color_map: Dict = None,
):
    n_plots = len(indices_to_plot)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), constrained_layout=True, sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax_idx, r_f_idx in enumerate(indices_to_plot):
        ax = axes[ax_idx]
        r_f = r_f_values[r_f_idx]
        scratch_loss = scratch_losses.get(r_f_idx, 0.0)
        ax.set_title(f"$r_f$ = {r_f:.4e}")
        ax.set_xlabel("Training Progress (Epochs/Steps)")
        if ax_idx == 0:
            ax.set_ylabel("Excess Risk")
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)

        method_data = shadow_data.get(r_f_idx, {})

        all_ref_hists = [h for histories in method_data.values() for h in histories.get('reference', [])]

        ref_color = None
        for i, hist in enumerate(all_ref_hists):
            if not hist: continue
            if isinstance(hist[0], tuple):
                steps = [t[0] for t in hist]
                losses = [t[1] for t in hist]
            else:
                steps = range(len(hist))
                losses = hist
            excess_risk = [loss - scratch_loss for loss in losses]
            if ref_color is None:
                line, = ax.plot(steps, excess_risk, alpha=0.2, label='Shadow Reference')
                ref_color = line.get_color()
            else:
                ax.plot(steps, excess_risk, color=ref_color, alpha=0.2)

        for method_i, (method_label, histories) in enumerate(method_data.items()):
            # Use consistent colour from main plot when available
            mapped = pretty_color_map.get(method_label) if pretty_color_map else None
            method_color = mapped[0] if mapped else None
            first_line = True
            for i, hist in enumerate(histories.get('target', [])):
                if not hist: continue
                steps = [t[0] for t in hist]
                losses = [t[1] for t in hist]
                excess_risk = [loss - scratch_loss for loss in losses]
                if first_line:
                    line, = ax.plot(steps, excess_risk, alpha=0.2, color=method_color,
                                   label=f'Shadow Target ({method_label})')
                    method_color = line.get_color()  # fallback if mapped was None
                    first_line = False
                else:
                    ax.plot(steps, excess_risk, color=method_color, alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

    # Save to default location (backwards compatibility)
    base, ext = os.path.splitext(save_file)
    history_file = f"{base}_shadow_trajectories{ext}"
    plt.savefig(history_file, bbox_inches='tight')
    print(f"Saved shadow trajectories plot to {history_file}")

    # Save to organized figures directory
    shadow_traj_file = os.path.join(figures_dir, "risk_vs_privacy_shadow_trajectories.pdf")
    plt.savefig(shadow_traj_file, bbox_inches='tight')
    print(f"✓ Saved shadow trajectories plot to: {shadow_traj_file}")

def _plot_mia_attack_performance(
    mia_data: Dict,
    r_f_values: np.ndarray,
    indices_to_plot: list,
    save_file: str,
    figures_dir: str,
):
    """
    Plot MIA attack performance visualization.

    For each (method, r_f) combination, shows how well the unlearning method
    defends against membership inference attacks by visualizing the distributions
    of shadow model scores compared to the target model.

    Args:
        mia_data: Dict mapping r_f_idx -> method_label -> list of per-sample data
        r_f_values: Array of all r_f values
        indices_to_plot: List of r_f indices to plot (first, median, last)
        save_file: Base path for saving plots
        figures_dir: Directory for organized figure output
    """
    if not mia_data or len(indices_to_plot) == 0:
        print("No MIA attack data to plot")
        return

    # Collect all methods across all r_f indices
    all_methods = set()
    for idx in indices_to_plot:
        if idx in mia_data:
            all_methods.update(mia_data[idx].keys())

    if not all_methods:
        print("No methods found in MIA attack data")
        return

    all_methods = sorted(list(all_methods))
    n_methods = len(all_methods)
    n_rf = len(indices_to_plot)

    # Create figure with subplots: rows = methods, cols = r_f values
    fig, axes = plt.subplots(n_methods, n_rf, figsize=(6 * n_rf, 4 * n_methods),
                             constrained_layout=True, squeeze=False)

    for method_idx, method_label in enumerate(all_methods):
        for rf_idx, r_f_plot_idx in enumerate(indices_to_plot):
            ax = axes[method_idx, rf_idx]
            r_f = r_f_values[r_f_plot_idx]

            # Get data for this (method, r_f) combination
            per_sample_data = mia_data.get(r_f_plot_idx, {}).get(method_label, [])

            if not per_sample_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f"{method_label}\n$r_f$ = {r_f:.4e}")
                ax.set_xlabel("LiRA Score (relative to target)")
                if rf_idx == 0:
                    ax.set_ylabel("Sample Index (sorted)")
                continue

            # Extract and sort samples by likelihood ratio (ascending)
            samples_with_ratio = []
            for sample in per_sample_data:
                ratio = sample.get('likelihood_ratio', 1.0)
                samples_with_ratio.append((ratio, sample))

            samples_with_ratio.sort(key=lambda x: x[0])
            sorted_samples = [s[1] for s in samples_with_ratio]
            sorted_ratios = [s[0] for s in samples_with_ratio]

            n_samples = len(sorted_samples)

            # Find decision boundary crossing (where ratio crosses 1)
            decision_boundary_idx = None
            for i, ratio in enumerate(sorted_ratios):
                if ratio >= 1.0:
                    decision_boundary_idx = i
                    break

            # Plot each sample on its own row
            for i, sample in enumerate(sorted_samples):
                target_score = sample['target_score']
                mean_ref = sample['mean_ref']
                std_ref = sample['std_ref']
                mean_tgt = sample['mean_tgt']
                std_tgt = sample['std_tgt']
                is_forget = sample.get('is_forget', True)

                # Colors: forget samples = blue/orange, test samples = red/green
                ref_color = 'blue' if is_forget else 'red'
                tgt_color = 'orange' if is_forget else 'green'

                # Normalize all scores relative to target_score (target -> 0)
                mean_ref_norm = mean_ref - target_score
                mean_tgt_norm = mean_tgt - target_score

                # Reference shadow: vertical line with center dot (|•|)
                ref_left = mean_ref_norm - std_ref
                ref_right = mean_ref_norm + std_ref
                # Vertical lines at endpoints
                ax.plot([ref_left, ref_left], [i-0.2, i+0.2], c=ref_color, alpha=0.6, linewidth=1.5)
                ax.plot([ref_right, ref_right], [i-0.2, i+0.2], c=ref_color, alpha=0.6, linewidth=1.5)
                # Horizontal connecting line (faint)
                ax.plot([ref_left, ref_right], [i, i], c=ref_color, alpha=0.2, linewidth=1)
                # Center dot at mean
                ax.scatter([mean_ref_norm], [i], c=ref_color, alpha=0.8, s=30, marker='o', zorder=3)

                # Target shadow: vertical line with center dot (|•|)
                tgt_left = mean_tgt_norm - std_tgt
                tgt_right = mean_tgt_norm + std_tgt
                # Vertical lines at endpoints
                ax.plot([tgt_left, tgt_left], [i-0.2, i+0.2], c=tgt_color, alpha=0.6, linewidth=1.5)
                ax.plot([tgt_right, tgt_right], [i-0.2, i+0.2], c=tgt_color, alpha=0.6, linewidth=1.5)
                # Horizontal connecting line (faint)
                ax.plot([tgt_left, tgt_right], [i, i], c=tgt_color, alpha=0.2, linewidth=1)
                # Center dot at mean
                ax.scatter([mean_tgt_norm], [i], c=tgt_color, alpha=0.8, s=30, marker='o', zorder=3)

                # Target model score (purple vertical line at x=0)
                ax.axvline(x=0, ymin=i/n_samples, ymax=(i+1)/n_samples,
                          color='purple', alpha=0.3, linewidth=1)

            # Add decision boundary line (where likelihood ratio = 1)
            if decision_boundary_idx is not None:
                ax.axhline(y=decision_boundary_idx - 0.5, color='black', linestyle='--',
                          linewidth=1.5, alpha=0.5, label='Decision Boundary (ratio=1)')

            # Formatting
            ax.set_title(f"{method_label}\n$r_f$ = {r_f:.4e}")
            ax.set_xlabel("LiRA Score (relative to target)")
            if rf_idx == 0:
                ax.set_ylabel("Sample Index (sorted by likelihood ratio)")
            ax.set_ylim(-0.5, n_samples - 0.5)
            ax.grid(True, alpha=0.3, axis='x')

            # Add legend (only on first subplot)
            if method_idx == 0 and rf_idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue',
                          markersize=8, linewidth=2, label=r'Forget: Ref (OUT) $\mu \pm \sigma$', alpha=0.6),
                    Line2D([0], [0], marker='o', color='orange', markerfacecolor='orange',
                          markersize=8, linewidth=2, label=r'Forget: Tgt (IN) $\mu \pm \sigma$', alpha=0.6),
                    Line2D([0], [0], marker='o', color='red', markerfacecolor='red',
                          markersize=8, linewidth=2, label=r'Test: Ref (OUT) $\mu \pm \sigma$', alpha=0.6),
                    Line2D([0], [0], marker='o', color='green', markerfacecolor='green',
                          markersize=8, linewidth=2, label=r'Test: Tgt (IN) $\mu \pm \sigma$', alpha=0.6),
                    Line2D([0], [0], color='purple', linewidth=2, label='Target Model', alpha=0.6),
                    Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                          label='Decision Boundary', alpha=0.5)
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Save to default location (backwards compatibility)
    base, ext = os.path.splitext(save_file)
    mia_plot_file = f"{base}_mia_attack{ext}"
    plt.savefig(mia_plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved MIA attack performance plot to {mia_plot_file}")

    # Save to organized figures directory
    mia_plot_organized = os.path.join(figures_dir, "risk_vs_privacy_mia_attack.pdf")
    plt.savefig(mia_plot_organized, dpi=300, bbox_inches='tight')
    print(f"✓ Saved MIA attack performance plot to: {mia_plot_organized}")

if __name__ == "__main__":
    main()
