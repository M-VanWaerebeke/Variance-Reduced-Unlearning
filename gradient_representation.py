#!/usr/bin/env python3
"""
Gradient representation study for unlearning methods on scikit-learn's Digits dataset.

Goal
-----
Run an unlearning fine-tuning pass for each requested method (NFT, VRU, cheat-VRU).
At every mini-batch step, compute the full-batch gradient on the RETAIN set at the
current parameters, and compare it to the method's per-step gradient estimator.

We track per-step:
  • cosine similarity between the method gradient and the true retain full-batch gradient
  • L2 norm of the method gradient (optionally also the true gradient norm for reference)

We then produce three plots:
  1) cosine similarity between the method gradient and the true retain full-batch gradient
  2) relative gradient norm vs step for each method (|g_est| / |g_true|)
  3) full-batch retain loss vs step
"""
from __future__ import annotations
import argparse
import copy
from typing import Dict, List, Tuple, Optional
import datetime

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Detect if we're in a log file (not a terminal) to reduce tqdm output
IN_LOGFILE = not sys.stdout.isatty()

# Tolerance for strict-better comparisons when saving
TOL_SAVE = 1e-12  # tolerance for strict-better comparisons when saving

# ---- Reuse helpers from the existing codebase ----
from digit_train import (
    set_seed,
    get_data_loaders,
    compute_full_gradient,
    grad_batch,
    Loaders,
    build_model,
    train as train_model,
)
from digit_forget import (
    get_retain_forget_loaders,
    UnlearnLoaders,
    _best_model_path,
)
from pretrain_model import (
    PIPELINE_HP_CONFIGS, PIPELINE_DISCOUNT,
    MNIST_HP_CONFIGS, MNIST_DISCOUNT, MNIST_EPOCHS,
)


# ---------------------- Utilities ----------------------

def _concat_flat(grads: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate and flatten a list of tensors to a single 1D tensor on CPU."""
    return torch.cat([g.detach().reshape(-1).cpu() for g in grads])

def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    an = a.norm().item()
    bn = b.norm().item()
    if an < eps or bn < eps:
        return 0.0
    return float(torch.dot(a, b).item() / (an * bn))


# ---------------------------------------------------------------------------
# Parallel training workers (seed-level parallelism)
# ---------------------------------------------------------------------------

def _init_worker_threads(n_workers):
    """Set per-worker thread count to avoid oversubscription."""
    n_cpus = os.cpu_count() or 1
    torch.set_num_threads(max(1, n_cpus // max(1, n_workers)))


def _train_full_model_worker(args):
    """Worker: train a full model for one seed (all HP configs), save best."""
    seed, weight_decay, model_type, dataset, batch_size, half_train, n_workers = args
    _init_worker_threads(n_workers)

    set_seed(seed)
    train_val_test, loaders = get_data_loaders(
        batch_size=batch_size, val_size=0.2, test_size=0.3,
        seed=seed, standardize=True, dataset=dataset,
    )

    if half_train:
        from torch.utils.data import random_split
        train_ds = train_val_test[0]
        n_full = len(train_ds)
        n_half = n_full // 2
        gen = torch.Generator().manual_seed(seed + 9999)
        train_half, held_out = random_split(train_ds, [n_half, n_full - n_half], generator=gen)
        loaders = Loaders(
            train=DataLoader(train_half, batch_size=batch_size, shuffle=True),
            full_train=DataLoader(train_half, batch_size=n_half, shuffle=False),
            val=loaders.val,
            test=DataLoader(held_out, batch_size=512, shuffle=False),
            n_features=loaders.n_features,
            n_classes=loaders.n_classes,
            n_train=n_half,
        )

    best_path = _best_model_path(seed, weight_decay, None, model_type, dataset,
                                  half_train=half_train, n_train=loaders.n_train)
    if os.path.exists(best_path):
        return best_path

    folder_dataset = "digit" if dataset == "digits" else dataset
    suffix = "_half" if half_train else ""
    save_folder = f"{folder_dataset}_full_models{suffix}"
    device = torch.device("cpu")

    if dataset == "mnist":
        hp_configs, discount, epochs = MNIST_HP_CONFIGS, MNIST_DISCOUNT, MNIST_EPOCHS
    else:
        hp_configs, discount, epochs = PIPELINE_HP_CONFIGS, PIPELINE_DISCOUNT, 1000

    for lr_raw, optimizer, algo, scheduler in hp_configs:
        lr_eff = lr_raw / 10 if model_type == "NN" else lr_raw
        model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
        train_model(
            model, loaders, device,
            algo=algo, seed=seed, epochs=epochs,
            lr=lr_eff, scheduler=scheduler, optimizer=optimizer,
            discount_factor=discount, momentum=0.0,
            weight_decay=weight_decay, log_interval=0,
            save_folder=save_folder, model_type=model_type,
            n_train=loaders.n_train,
        )
    print(f"  Full model done for seed {seed}")
    return best_path


def _train_scratch_model_worker(args):
    """Worker: train a scratch model for one (seed, r_f) pair, save best."""
    seed, r_f, weight_decay, model_type, dataset, batch_size, half_train, n_workers = args
    _init_worker_threads(n_workers)

    set_seed(seed)
    train_val_test, loaders = get_data_loaders(
        batch_size=batch_size, val_size=0.2, test_size=0.3,
        seed=seed, standardize=True, dataset=dataset,
    )
    train_ds = train_val_test[0]

    if half_train:
        from torch.utils.data import random_split
        n_full = len(train_ds)
        n_half = n_full // 2
        gen = torch.Generator().manual_seed(seed + 9999)
        train_half, held_out = random_split(train_ds, [n_half, n_full - n_half], generator=gen)
        train_ds = train_half
        loaders = Loaders(
            train=DataLoader(train_half, batch_size=batch_size, shuffle=True),
            full_train=DataLoader(train_half, batch_size=n_half, shuffle=False),
            val=loaders.val,
            test=DataLoader(held_out, batch_size=512, shuffle=False),
            n_features=loaders.n_features,
            n_classes=loaders.n_classes,
            n_train=n_half,
        )

    uloaders = get_retain_forget_loaders(train_ds, r_f, seed, batch_size)

    best_path = _best_model_path(seed, weight_decay, r_f, model_type, dataset,
                                  half_train=half_train, n_train=uloaders.n_retain)
    if os.path.exists(best_path):
        return best_path

    scratch_loader = Loaders(
        train=uloaders.retain,
        full_train=uloaders.retain_full,
        val=loaders.val,
        test=loaders.test,
        n_features=loaders.n_features,
        n_classes=loaders.n_classes,
        n_train=uloaders.n_retain,
    )
    folder_dataset = "digit" if dataset == "digits" else dataset
    suffix = "_half" if half_train else ""
    save_folder = f"{folder_dataset}_scratch_models{suffix}"
    device = torch.device("cpu")

    if dataset == "mnist":
        hp_configs, discount, epochs = MNIST_HP_CONFIGS, MNIST_DISCOUNT, MNIST_EPOCHS
    else:
        hp_configs, discount, epochs = PIPELINE_HP_CONFIGS, PIPELINE_DISCOUNT, 3001

    for lr_raw, optimizer, algo, scheduler in hp_configs:
        lr_eff = lr_raw if (model_type == "LR" and dataset in ("digits", "breast_cancer")) else lr_raw / 10
        model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
        train_model(
            model, scratch_loader, device,
            algo=algo, seed=seed, epochs=epochs,
            lr=lr_eff, scheduler=scheduler, optimizer=optimizer,
            discount_factor=discount, momentum=0.0,
            weight_decay=weight_decay, log_interval=0,
            save_folder=save_folder, r_f=r_f, model_type=model_type,
            n_train=uloaders.n_retain,
        )
    print(f"  Scratch model done for seed {seed}, r_f={r_f:.4e}")
    return best_path


def ensure_pretrained_models_parallel(
    seeds, weight_decay, model_type, dataset,
    batch_size=64, half_train=False, max_workers=None,
):
    """Pre-train full models for multiple seeds in parallel (one thread per worker)."""
    if max_workers is None:
        max_workers = min(len(seeds), os.cpu_count() or 1)
    tasks = [
        (seed, weight_decay, model_type, dataset, batch_size, half_train, max_workers)
        for seed in seeds
    ]
    if max_workers <= 1:
        return [_train_full_model_worker(t) for t in tasks]
    mp_ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers, mp_context=mp_ctx) as pool:
        return list(pool.map(_train_full_model_worker, tasks))


def ensure_scratch_models_parallel(
    seed_rf_pairs, weight_decay, model_type, dataset,
    batch_size=64, half_train=False, max_workers=None,
):
    """Pre-train scratch models for multiple (seed, r_f) pairs in parallel."""
    if max_workers is None:
        max_workers = min(len(seed_rf_pairs), os.cpu_count() or 1)
    tasks = [
        (seed, r_f, weight_decay, model_type, dataset, batch_size, half_train, max_workers)
        for seed, r_f in seed_rf_pairs
    ]
    if max_workers <= 1:
        return [_train_scratch_model_worker(t) for t in tasks]
    mp_ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers, mp_context=mp_ctx) as pool:
        return list(pool.map(_train_scratch_model_worker, tasks))


def _ensure_pretrained_full_model(
    seed: int,
    weight_decay: float,
    model_type: str,
    loaders,
    device: torch.device,
    dataset: str = "digits",
    half_train: bool = False,
) -> str:
    """Ensure a best full model checkpoint exists; train if needed.
    Returns the checkpoint path.
    """
    model_type = model_type.upper()
    wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
    folder_dataset = "digit" if dataset == "digits" else dataset
    suffix = "_half" if half_train else ""
    best_path = f"{folder_dataset}_full_models{suffix}/best_model-seed{seed}_model{model_type}_wd{wd_str}_n{loaders.n_train}.pt"

    save_folder = f"{folder_dataset}_full_models{suffix}"
    if not os.path.exists(best_path):
        # Check for legacy path without n_train suffix and rename if found
        legacy_path = best_path.replace(f"_n{loaders.n_train}", "")
        if os.path.exists(legacy_path):
            print(f"Found legacy pretrained model: {legacy_path}. Renaming to {best_path}")
            os.rename(legacy_path, best_path)
            return best_path

        print(f"No pretrained found at {best_path} — training base model ({loaders.n_train} samples)...")

        if dataset == "mnist":
            hp_configs, discount, epochs = MNIST_HP_CONFIGS, MNIST_DISCOUNT, MNIST_EPOCHS
        else:
            hp_configs, discount, epochs = PIPELINE_HP_CONFIGS, PIPELINE_DISCOUNT, 1000

        for lr_raw, optimizer, algo, scheduler in hp_configs:
            lr_eff = lr_raw / 10 if model_type == "NN" else lr_raw
            base = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
            _ = train_model(
                base,
                loaders,
                device,
                algo=algo,
                seed=seed,
                epochs=epochs,
                lr=lr_eff,
                scheduler=scheduler,
                optimizer=optimizer,
                discount_factor=discount,
                momentum=0.0,
                weight_decay=weight_decay,
                log_interval=0,
                save_folder=save_folder,
                model_type=model_type,
                n_train=loaders.n_train,
            )
        print("Pretraining complete.")
    else:
        print(f"Found pretrained model: {best_path}")

    return best_path


# --- New helper: ensure scratch (retain-only) model and return checkpoint path ---
def _ensure_scratch_model(
    seed: int,
    weight_decay: float,
    r_f: float,
    model_type: str,
    loaders,
    uloaders: UnlearnLoaders,
    device: torch.device,
    dataset: str = "digits",
    half_train: bool = False,
) -> str:
    """Ensure a best *scratch* model (trained on RETAIN only) exists; train if needed.
    Returns the checkpoint path.
    """
    model_type = model_type.upper()
    best_path = _best_model_path(seed, weight_decay, r_f, model_type, dataset, half_train=half_train, n_train=uloaders.n_retain)

    if not os.path.exists(best_path):
        # Check for legacy path without n_train suffix and rename if found
        legacy_path = best_path.replace(f"_n{uloaders.n_retain}", "")
        if os.path.exists(legacy_path):
            print(f"Found legacy scratch model: {legacy_path}. Renaming to {best_path}")
            os.rename(legacy_path, best_path)
            return best_path

        print(f"No scratch model found at {best_path} — training on RETAIN set...")
        folder_dataset = "digit" if dataset == "digits" else dataset
        suffix = "_half" if half_train else ""
        scratch_save_folder = f"{folder_dataset}_scratch_models{suffix}"
        # Build a loaders object that uses only the RETAIN split for training
        scratch_loader = Loaders(
            train=uloaders.retain,
            full_train=uloaders.retain_full,
            val=loaders.val,
            test=loaders.test,
            n_features=loaders.n_features,
            n_classes=loaders.n_classes,
            n_train=uloaders.n_retain,
        )
        if dataset == "mnist":
            hp_configs, discount, epochs = MNIST_HP_CONFIGS, MNIST_DISCOUNT, MNIST_EPOCHS
        else:
            hp_configs, discount, epochs = PIPELINE_HP_CONFIGS, PIPELINE_DISCOUNT, 3001

        for lr_raw, optimizer, algo, scheduler in hp_configs:
            lr_eff = lr_raw if (model_type == "LR" and dataset in ("digits", "breast_cancer")) else lr_raw / 10
            base = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
            _ = train_model(
                base,
                scratch_loader,
                device,
                algo=algo,
                seed=seed,
                epochs=epochs,
                lr=lr_eff,
                scheduler=scheduler,
                optimizer=optimizer,
                discount_factor=discount,
                momentum=0.0,
                weight_decay=weight_decay,
                log_interval=0,
                save_folder=scratch_save_folder,
                r_f=r_f,
                model_type=model_type,
                n_train=uloaders.n_retain,
            )
        print("Scratch training complete.")
    else:
        print(f"Found scratch model: {best_path}")

    return best_path


# ---------------------- Core runner ----------------------

def run_unlearning_gradient_trace(
    method: str,
    base_model: nn.Module,
    loaders,
    uloaders: UnlearnLoaders,
    r_f: float,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    scheduler: str,
    discount_factor: float,
    weight_decay: float,
    maybe_save_better: bool = False,
    scratch_best_path: Optional[str] = None,
    scratch_best_loss: Optional[float] = None,
) -> Dict[str, List[float]]:
    """Run one unlearning method and record per-step gradient stats.

    `maybe_save_better`, `scratch_best_path`, and `scratch_best_loss` are retained
    for API compatibility. An unlearning trajectory never changes a retrained
    baseline checkpoint or the reference loss used for comparison.

    Returns a dict with keys:
      - cos: list[float] cosine similarity (estimator vs retain full-batch grad)
      - est_norm: list[float] L2 norm of estimator per step
      - true_norm: list[float] L2 norm of true retain full-batch grad per step
      - true_loss: list[float] retain full-batch objective (loss) per step
      - steps_per_epoch: list[int] cumulative steps after each epoch (for plotting epoch ticks)
    """
    method = method.lower()
    assert method in {"nft", "vru", "cheat-vru", "svrg", "seqvru"}, f"Unsupported method: {method}"

    # Working copy of the base model (pretrained on FULL train set)
    model = copy.deepcopy(base_model).to(device=device, dtype=torch.float64)
    model.train()
    # Track the best retain full-batch loss and corresponding parameters during the run
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    # For VRU variants, SVRG, and SeqVRU, prepare snapshot and retain gradient reference once at start
    full_model = None
    retain_grad_full = None
    if method in {"vru", "cheat-vru", "svrg", "seqvru"}:
        # Snapshot model; for SVRG/SeqVRU we'll refresh this snapshot at certain epochs or every epoch
        full_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
        full_model.train()
        if method in {"vru", "seqvru"}:
            # Use the forget-set full gradient and transform it to a retain proxy via the VRU scaling
            forget_grad_full, _ = compute_full_gradient(
                full_model, uloaders.forget, device, weight_decay, no_grads=False
            )
            retain_grad_full = [-(r_f / (1.0 - r_f)) * g for g in forget_grad_full]
        else:  # cheat-VRU or SVRG
            retain_grad_full, _ = compute_full_gradient(
                full_model, uloaders.retain_full, device, weight_decay, no_grads=False
            )

    current_lr = lr
    thirds = [epochs // 3, (2 * epochs) // 3]

    cos_hist: List[float] = []
    est_norm_hist: List[float] = []
    true_norm_hist: List[float] = []
    true_loss_hist: List[float] = []
    steps_per_epoch: List[int] = []

    step_count = 0
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"{method.upper()} epochs", leave=True,
                     mininterval=float('inf') if IN_LOGFILE else 0.1)
    for epoch in epoch_bar:
        if epoch in thirds and scheduler == "step":
            current_lr *= discount_factor
        # Refresh policy:
        # - SVRG: refresh at 'thirds' (retain full gradient)
        # - SeqVRU: refresh at 'thirds' (forget full gradient scaled to retain proxy)
        if method == "svrg" and epoch in thirds:
            full_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
            full_model.train()
            retain_grad_full, _ = compute_full_gradient(
                full_model, uloaders.retain_full, device, weight_decay, no_grads=False
            )
        elif method == "seqvru" and epoch in thirds:
            full_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
            full_model.train()
            forget_grad_full, _ = compute_full_gradient(
                full_model, uloaders.forget, device, weight_decay, no_grads=False
            )
            retain_grad_full = [-(r_f / (1.0 - r_f)) * g for g in forget_grad_full]

        for xb, yb in uloaders.retain:
            xb = xb.to(device=device, dtype=torch.float64)
            yb = yb.to(device)

            # --- Method gradient estimator ---
            if method == "nft":
                g_est = grad_batch(model, xb, yb, weight_decay)
            else:
                assert full_model is not None and retain_grad_full is not None
                g_cur = grad_batch(model, xb, yb, weight_decay)
                g_snap = grad_batch(full_model, xb, yb, weight_decay)
                g_est = [gc - gs + mu for gc, gs, mu in zip(g_cur, g_snap, retain_grad_full)]

            # --- True retain full-batch gradient at current parameters ---
            g_true, fb_retain_loss = compute_full_gradient(
                model, uloaders.retain_full, device, weight_decay, no_grads=False
            )
            # Recompute the retain full-batch loss in float64 for higher precision (no gradients needed)
            _, fb_retain_loss64 = compute_full_gradient(
                model, uloaders.retain_full, device, weight_decay, no_grads=True, as_double=True
            )
            # Track best retain FB loss and state (before the update is applied)
            if fb_retain_loss < best_loss:
                best_loss = float(fb_retain_loss)
                best_state = copy.deepcopy(model.state_dict())

            # --- Metrics ---
            est_vec = _concat_flat(g_est)
            true_vec = _concat_flat(g_true)
            cos_hist.append(_cosine(est_vec, true_vec))
            est_norm_hist.append(float(est_vec.norm().item()))
            true_norm_hist.append(float(true_vec.norm().item()))
            true_loss_hist.append(float(fb_retain_loss64))

            # --- Parameter update ---
            with torch.no_grad():
                if method == "seqvru":
                    # Diagnostic: invert update sign for SeqVRU to check cosine behavior
                    for p, ge in zip(model.parameters(), g_est):
                        p.add_(current_lr * ge)
                else:
                    for p, ge in zip(model.parameters(), g_est):
                        p.add_(-current_lr * ge)

            step_count += 1
        steps_per_epoch.append(step_count)
    return {
        "cos": cos_hist,
        "est_norm": est_norm_hist,
        "true_norm": true_norm_hist,
        "true_loss": true_loss_hist,
        "steps_per_epoch": steps_per_epoch,
        "best_retain_loss": best_loss,
    }


# ---------------------- Plotting ----------------------


def make_plots(
    traces: Dict[str, Dict[str, List[float]]],
    *,
    scratch_loss: float,
    save_prefix: str,
    show: bool,
) -> Tuple[str, str]:
    """Create plots and save them to disk.

    The third subplot shows **excess retain loss** = (current retain full-batch loss) − (scratch retain full-batch loss),
    using a log y-scale.
    Returns (cos_path, norms_path) for compatibility (single combined figure).
    """
    # Helper: rolling mean over a fixed window (pads the first w-1 values with NaNs)
    def _rolling_mean(vals, window: int = 5):
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return arr
        w = min(window, arr.size)
        out = np.full_like(arr, np.nan, dtype=float)
        # NaN-aware rolling mean
        for i in range(w - 1, arr.size):
            out[i] = np.nanmean(arr[i - w + 1 : i + 1])
        return out

    # Single figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Cosine similarity subplot (rolling average as main curve, raw points transparent)
    ax = axes[0]
    for name, out in traces.items():
        y = out["cos"]
        x = np.arange(len(y))
        y_rm = _rolling_mean(y, window=10)
        # Subsample for clarity: plot/scatter every 10th point
        ax.plot(x[::10], y_rm[::10], label=name.upper())
        ax.scatter(x[::10], np.array(y)[::10], s=10, alpha=0.2)
    ax.set_title("Cosine similarity to retain full-batch gradient")
    ax.set_xlabel("Step")
    ax.set_ylabel("cos(g_est, g_true)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    # 2) Relative gradient norm subplot (|g_est| / |g_true|)
    ax = axes[1]
    for name, out in traces.items():
        y_est = np.asarray(out["est_norm"], dtype=float)
        y_true = np.asarray(out["true_norm"], dtype=float)
        ratio = y_est / np.maximum(y_true, 1e-12)
        x = np.arange(len(ratio))
        ratio_rm = _rolling_mean(ratio, window=10)
        # Subsample for clarity: plot/scatter every 10th point
        ax.plot(x[::10], ratio_rm[::10], label=f"{name.upper()} |g_est|/|g_true|")
        ax.scatter(x[::10], ratio[::10], s=10, alpha=0.2)
    ax.set_title("Relative gradient norm per step")
    ax.set_xlabel("Step")
    ax.set_ylabel("|g_est| / |g_true|")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    # 3) Excess retain loss = current retain FB loss − scratch retain FB loss (log scale)
    ax = axes[2]
    for name    , out in traces.items():
        y_loss = np.asarray(out.get("true_loss", []), dtype=float)
        x_loss = np.arange(len(y_loss))
        y_excess = y_loss - float(scratch_loss)
        # Mask nonpositive values so log scale is valid without artificial eps
        # y_excess = np.where(y_excess > 0, y_excess, np.nan)
        y_excess_rm = _rolling_mean(y_excess, window=10)
        # Subsample for clarity: plot/scatter every 10th point
        ax.plot(x_loss[::10], y_excess_rm[::10], label=f"{name.upper()} excess loss")
        ax.scatter(x_loss[::10], y_excess[::10], s=10, alpha=0.2)
    ax.set_title("Excess retain loss vs. scratch")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss(current) − Loss(scratch)")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    plt.tight_layout()

    combined_path = f"{save_prefix}_combined.pdf"
    fig.savefig(combined_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    return combined_path, combined_path


# ---------------------- CLI ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradient representation for unlearning methods (Digits)")
    # Data / model
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN"])

    # Unlearning setup
    p.add_argument("--r_f", type=float, default=0.1, help="Fraction of forget data (0,1)")
    p.add_argument("--methods", nargs="+", type=str, default=["nft", "vru", "cheat-vru", "svrg"],
                   choices=["nft", "vru", "cheat-vru", "svrg", "seqvru"], help="Methods to run")

    # Optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lrs", nargs="+", type=float, default=[0.1, 0.01, 0.001],
                   help="One or more learning rates to sweep; best final loss per method is kept")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "step"])
    p.add_argument("--discount-factor", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=0.1)

    # Plotting
    p.add_argument("--save-prefix", type=str, default="gradient_repr")
    p.add_argument("--show", action="store_true")
    p.add_argument("--dataset", type=str, default="digits", choices=["digits", "mnist", "breast_cancer"],
                   help="Dataset to use")

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}")

    # Create comprehensive hyperparameter string for figure directory and filenames
    dt_str = datetime.datetime.now().strftime("%y-%m-%d-%H%M")
    hp_str = f"d{args.dataset}_m{args.model}_rf{args.r_f:g}_wd{args.weight_decay:g}_bs{args.batch_size}_s{args.seed}_ep{args.epochs}"

    # Create output directory for this run's figures
    figures_dir = os.path.join("gradient_representation_figures", f"{dt_str}_{hp_str}")
    os.makedirs(figures_dir, exist_ok=True)

    # Update save_prefix to point to the figures directory
    original_save_prefix = args.save_prefix
    args.save_prefix = os.path.join(figures_dir, os.path.basename(args.save_prefix))

    print(f"\n{'='*80}")
    print(f"FIGURES DIRECTORY: {figures_dir}")
    print(f"{'='*80}\n")

    # Data
    train_val_test_sets, loaders = get_data_loaders(
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        standardize=not args.no_standardize,
        dataset=args.dataset,
    )
    train_ds, _, _ = train_val_test_sets

    # Retain/Forget split from the training set
    uloaders = get_retain_forget_loaders(
        train_ds=train_ds,
        r_f=args.r_f,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"Retain size: {uloaders.n_retain} | Forget size: {uloaders.n_forget}")

    # Ensure and load base model (pretrained on FULL train set)
    best_path = _ensure_pretrained_full_model(
        seed=args.seed,
        weight_decay=args.weight_decay,
        model_type=args.model,
        loaders=loaders,
        device=device,
        dataset=args.dataset,
    )
    base = build_model(args.model, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=device)
    for k, v in list(state.items()):
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            state[k] = v.to(dtype=torch.float64)
    base.load_state_dict(state)

    # Ensure and load the best scratch model (RETAIN-only) and compute its retain full-batch loss
    scratch_best_path = _ensure_scratch_model(
        seed=args.seed,
        weight_decay=args.weight_decay,
        r_f=args.r_f,
        model_type=args.model,
        loaders=loaders,
        uloaders=uloaders,
        device=device,
        dataset=args.dataset,
    )
    scratch_model = build_model(args.model, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
    try:
        scratch_state = torch.load(scratch_best_path, map_location=device, weights_only=True)
    except TypeError:
        scratch_state = torch.load(scratch_best_path, map_location=device)
    for k, v in list(scratch_state.items()):
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            scratch_state[k] = v.to(dtype=torch.float64)
    scratch_model.load_state_dict(scratch_state)
    _, scratch_loss = compute_full_gradient(
        scratch_model, uloaders.retain_full, device, args.weight_decay, no_grads=True, as_double=True
    )

    # Run each method and collect traces (sweep several learning rates and keep the best final loss)
    traces: Dict[str, Dict[str, List[float]]] = {}

    # Minimal change approach:
    #  - Ensure CHEAT-VRU runs before SVRG by sorting methods
    #  - Compare on lowercase method names
    #  - When CHEAT-VRU finishes, lock SVRG to the same best LR
    methods_sorted = sorted(list(args.methods), key=lambda x: 0 if x.lower() == "cheat-vru" else 1)
    SVRG_lr: Optional[float] = None

    for m in methods_sorted:
        m_lower = m.lower()
        best_out: Optional[Dict[str, List[float]]] = None
        best_final_loss: float = float("inf")
        best_lr: Optional[float] = None

        for lr_i in args.lrs:
            # If we already found the best LR with CHEAT-VRU, run SVRG only at that LR
            if m_lower == "svrg" and SVRG_lr is not None and lr_i != SVRG_lr:
                continue

            out = run_unlearning_gradient_trace(
                method=m_lower,
                base_model=base,
                loaders=loaders,
                uloaders=uloaders,
                r_f=args.r_f,
                device=device,
                epochs=args.epochs,
                lr=lr_i,
                scheduler=args.scheduler,
                discount_factor=args.discount_factor,
                weight_decay=args.weight_decay,
                maybe_save_better=False,
                scratch_best_path=scratch_best_path,
                scratch_best_loss=scratch_loss,
            )
            # Determine final loss of this run (use last point of true_loss if available)
            final_loss = float(out["true_loss"][-1]) if out.get("true_loss") else float("inf")
            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_out = out
                best_lr = lr_i

        # If CHEAT-VRU just finished, record its best LR to lock SVRG
        if m_lower == "cheat-vru" and best_lr is not None:
            SVRG_lr = best_lr
            print(f"Locking SVRG learning rate to CHEAT-VRU best: {SVRG_lr}")

        # Store only the best trace for this method, labeled with the chosen LR
        label = m_lower if best_lr is None else f"{m_lower} (lr={best_lr})"
        traces[label] = best_out if best_out is not None else {}

    # Make the requested plots
    cos_path, norms_path = make_plots(traces, scratch_loss=scratch_loss, save_prefix=args.save_prefix, show=args.show)
    print(f"Saved plots to:\n  - {cos_path}\n  - {norms_path}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: All figures saved to:")
    print(f"  {figures_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()