#!/usr/bin/env python3
"""
Compare unlearning methods on scikit-learn's Digits dataset using a pretrained
logistic regression (PyTorch).

Two unlearning methods are compared:
  1) Fine-tuning with SGD on the retain set
  2) Fine-tuning with SVRG on the retain set

We reuse utilities from `digit_train.py` (model, objective, full-gradient, etc.)
to keep behavior consistent with the training script.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm.auto import trange
from collections import OrderedDict

# Detect if we're in a log file (not a terminal) to reduce tqdm output
IN_LOGFILE = not sys.stdout.isatty()

# ---- Reuse your helpers from the training file ----
# These names are expected to exist in your digit_train.py (per your provided code).

from digit_train import (
    LogisticRegression,
    Loaders,
    set_seed,
    get_data_loaders,
    objective_loss,
    accuracy as evaluate_model,
    compute_full_gradient,
    grad_batch,
    train as train_model,
    build_model,
)


# ---- Filename helpers (make formatting consistent) ----

def _fmt_float(val: float, ndigits: int) -> str:
    """Format floats with a fixed number of decimals for filenames."""
    return f"{val:.{ndigits}f}"


def _best_model_path(seed: int, weight_decay: float, r_f: float = None, model_type: str = "LR", dataset: str = "digits", half_train: bool = False, n_train: int = None) -> str:
    """Build the checkpoint path consistent with train_model's saving scheme.

    Format rules:
      - model_type uppercased in the filename
      - weight_decay: scientific notation with 1 decimal if <1e-3 else 3 decimals
      - r_f (if provided): "rf" + sci with 1 decimal if <1e-2 else 3 decimals
      - n_train (if provided): included to distinguish models trained on different data splits
      - scratch models (r_f provided) go under 'digit_scratch_models/'
      - full models (r_f is None) go under 'digit_full_models/'
      - filename layout (scratch): best_model-seed{seed}_model{MODELTYPE}_rf{RF}_wd{WD}_n{N}.pt
      - filename layout (full):    best_model-seed{seed}_model{MODELTYPE}_wd{WD}_n{N}.pt
      - half_train=True appends '_half' to the folder name (for shadow-matched training)
    """
    tag = (model_type or "LR").upper()

    folder_dataset = "digit" if dataset == "digits" else dataset
    suffix = "_half" if half_train else ""
    # weight decay formatting
    wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
    # n_train suffix (only if provided)
    n_str = f"_n{n_train}" if n_train is not None else ""

    if r_f is not None:
        # scratch model path (trained on RETAIN only)
        rf_str = f"{r_f:.1e}" if r_f < 1e-2 else f"{r_f:.3f}"
        save_folder = f"{folder_dataset}_scratch_models{suffix}"
        return (
            f"{save_folder}/best_model-seed{seed}_model{tag}_rf{rf_str}_wd{wd_str}{n_str}.pt"
        )
    else:
        # full model path (trained on FULL train set)
        save_folder = f"{folder_dataset}_full_models{suffix}"
        return (
            f"{save_folder}/best_model-seed{seed}_model{tag}_wd{wd_str}{n_str}.pt"
        )


# ---------------------- Unlearning Data Helpers ----------------------

@dataclass
class UnlearnLoaders:
    """Holds retain/forget loaders derived from the original train set."""
    retain: DataLoader
    retain_full: DataLoader  # for full gradient computation (SVRG, metrics)
    forget: DataLoader       # for cheap full gradient computation at step 1
    n_retain: int
    n_forget: int



def get_retain_forget_loaders(
    train_ds: torch.utils.data.Dataset,
    r_f: float,
    seed: int,
    batch_size: int,
) -> UnlearnLoaders:
    """Split the training dataset into retain/forget subsets without stratification.

    Uses a reproducible random split driven by `seed`.
    """
    assert 0.0 < r_f < 1.0, "forget_ratio must be in (0, 1)"
    n = len(train_ds)
    n_forget = int(round(r_f * n))
    n_retain = n - n_forget
    if min(n_retain, n_forget) < 1:
        raise ValueError(f"Empty dataset for forget_ratio={r_f:.4f} on train set of size {n}")

    generator = torch.Generator().manual_seed(seed)
    retain_subset, forget_subset = torch.utils.data.random_split(
        train_ds, [n_retain, n_forget], generator=generator
    )

    retain = DataLoader(retain_subset, batch_size=batch_size, shuffle=True)
    forget = DataLoader(forget_subset, batch_size=batch_size, shuffle=True)
    retain_full = DataLoader(retain_subset, batch_size=len(retain_subset), shuffle=False)


    return UnlearnLoaders(
        retain=retain,
        retain_full=retain_full,
        forget=forget,
        n_retain=len(retain_subset),
        n_forget=len(forget_subset),
    )


def load_best_scratch_state(seed: int, weight_decay: float, r_f: float, model_type: str, dataset: str, n_retain: int = None) -> "OrderedDict[str, torch.Tensor]":
    """Load the best scratch model for reference in plots.
    This is the best objective achieved on the RETAIN set
    (for the given forget fraction r_f) for the specified model type.
    """
    path = _best_model_path(seed, weight_decay, r_f, model_type, dataset, n_train=n_retain)
    try:
        sd = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=torch.device("cpu"))
    for k, v in list(sd.items()):
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            sd[k] = v.to(dtype=torch.float64)
    return sd



# ---------------------- Unlearning Loops ----------------------

def forget(
    base_model: nn.Module,
    loaders: Loaders,
    uloaders: UnlearnLoaders,
    r_f: float,
    device: torch.device,
    *,
    seed: int,
    algo: str,                 # "sgd" or "svrg"
    epochs: int,
    lr: float,
    scheduler: str = "none",   # "none" or "step"
    discount_factor: float = 0.1, # for "step" scheduler
    momentum: float = 0.0,     # used for SGD only
    weight_decay: float = 0.0, # L2 via objective_loss
    log_interval: int = 0,
    model_type: str = "LR",
    dataset: str = "digits",
    average: bool = False,  # Use Polyak averaging for VRU
    project: bool = False,  # Project updates onto ball for VRU
    skip_scratch_reference: bool = False,  # Skip loading scratch model (for shadow unlearning)
    batch_loss_interval: int = 1,  # Record full-batch retain loss every N steps (0 = disable)
) -> dict:
    """
    Unlearn (fine-tune from the given pretrained model) using only the RETAIN set.
    - algo='sgd': standard SGD on retain set
    - algo='svrg': SVRG updates on retain set

    Tracks retain objective, validation metrics, and forget-set accuracy as signals.
    """
    algo = algo.lower()
    if algo not in {"sgd", "svrg", "vru", "cheat-vru", "semi-cheat-vru", "seqvru"}:
        raise ValueError(f"Unsupported algo '{algo}'. Use 'sgd', 'svrg', 'vru', 'cheat-vru', 'semi-cheat-vru', or 'seqvru'.")

    set_seed(seed)
    model = base_model.to(device=device, dtype=torch.float64)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # Optimizer for SGD branch (weight_decay handled in loss for parity with SVRG)
    current_lr = lr
    optimizer = None
    if algo == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=current_lr, momentum=momentum, weight_decay=0.0
        )



    thirds = [epochs // 3, (2 * epochs) // 3]

    hist_retain_obj = []     # full-batch retain objective per epoch
    hist_grad_norm = []      # grad norm between current and perfect model
    hist_param_diff = []      # loss difference between current and perfect model
    hist_batch_loss = []     # per-batch retain loss (after each gradient step)
    retain_grad_cos_sim = None  # cosine sim between estimated and true retain gradient

    best_state = None
    best_model = None
    best_loss = float("inf")
    best_loss_this_run = float("inf")
    if not skip_scratch_reference:
        # Load the best model for this setup
        best_model = copy.deepcopy(model).to(device)
        best_state = load_best_scratch_state(seed, weight_decay, r_f, model_type, dataset, n_retain=uloaders.n_retain)
        best_model.load_state_dict(best_state)
        _, best_loss = compute_full_gradient(  # Best loss prior to this unlearning run
            best_model, uloaders.retain_full, device, weight_decay, no_grads=False
        )

    if algo in {"vru", "cheat-vru", "semi-cheat-vru"}:
        # Compute the initial full gradient on the retain set throught the forget set
        full_model = copy.deepcopy(model).to(device)
        full_model.train()
        forget_grad_full, _ = compute_full_gradient(
            full_model, uloaders.forget, device, weight_decay,
        )
        # Count the actual number of samples in each loader to compute the effective r_f value
        real_r_f = len(uloaders.forget.dataset) / (len(uloaders.retain.dataset) + len(uloaders.forget.dataset))
        retain_grad_full = [-real_r_f/(1-real_r_f)*g.detach() for g in forget_grad_full]

        retain_grad_full_sanity, _ = compute_full_gradient(
            full_model, uloaders.retain_full, device, weight_decay,
        )

        print('Norm difference between gradients computed on forget and retain sets:',
        np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full]) -
                        np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full_sanity])))
        retain_grad_cos_sim = float(
                np.dot(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full]),
                         np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full_sanity])) /
                (np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full])) *
                 np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full_sanity])) + 1e-10)
                )
        print('Cosine similarity between the two:', retain_grad_cos_sim)
        print('Respective norms of the two retain grads:,',
                np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full])),
                np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full_sanity])))
        # Compute the gradient of the full model on the full dataset for logging purposes
        full_model = copy.deepcopy(model).to(device)
        full_batch_grads, _ = compute_full_gradient(
            full_model, loaders.full_train, device, weight_decay, no_grads=False
        )
        print('grad norm of the full model on the full dataset',
              np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in full_batch_grads])))

        if algo == "cheat-vru":
            retain_grad_full = retain_grad_full_sanity
            print("Using the full retain dataset gradient in cheat-VRU")
        elif algo == "semi-cheat-vru":
            retain_norm = np.linalg.norm(
                np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full_sanity])
            )
            proxy_norm = np.linalg.norm(
                np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full])
            )
            scale = retain_norm / (proxy_norm + 1e-12)
            retain_grad_full = [g * scale for g in retain_grad_full]
            print(f"Rescaling VRU retain proxy gradient by factor {scale:.4f} to match retain norm")

        # Initialize projection if requested
        if project:
            G_f = np.linalg.norm(
                np.concatenate([g.detach().cpu().flatten() for g in forget_grad_full])
            )
            mu = weight_decay
            projection_radius = (G_f / mu) * (real_r_f / (1 - real_r_f))
            init_params = [p.clone().detach() for p in model.parameters()]
            print(f"Projection enabled: radius={projection_radius:.6f} (G_f={G_f:.6f}, mu={mu}, r_f={real_r_f:.4f})")

        # Initialize averaging if requested
        if average:
            avg_params = [p.clone().detach() for p in model.parameters()]
            num_updates = 0
            print("Averaging enabled: Polyak averaging of all iterates")

    pbar = trange(1, epochs + 1, desc=f"Unlearn:{algo.upper()}", leave=True,
                  mininterval=float('inf') if IN_LOGFILE else 0.1)
    for epoch in pbar:
        model.train()

        if epoch in thirds and scheduler == "step":
            current_lr *= discount_factor
            if optimizer is not None:
                for g in optimizer.param_groups:
                    g["lr"] = current_lr
            pbar.set_postfix(lr=current_lr)


        # Objective computation on full retain set (for logging/model selection)
        full_batch_grads, retain_fb_loss = compute_full_gradient(
        model, uloaders.retain_full, device, weight_decay, no_grads=False
        )
        for g in full_batch_grads:
            g.detach_()
        with torch.no_grad():
            grad_norm = np.linalg.norm(
                np.concatenate([g.detach().cpu().flatten() for g in full_batch_grads])
            )
            param_diff = 0.0
            if best_model is not None:
                for p, pb in zip(model.parameters(), best_model.parameters()):
                    param_diff += float(torch.sum((p - pb) ** 2))
                param_diff = np.sqrt(param_diff)

        hist_retain_obj.append(retain_fb_loss)
        hist_grad_norm.append(grad_norm)
        hist_param_diff.append(param_diff)

        if algo == "sgd":
            for step, (xb, yb) in enumerate(uloaders.retain, start=1):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                loss = objective_loss(model, xb, yb, criterion, weight_decay)
                loss.backward()
                optimizer.step()
                if batch_loss_interval and (step == 1 or step % batch_loss_interval == 0):
                    _, step_fb_loss = compute_full_gradient(model, uloaders.retain_full, device, weight_decay, no_grads=True)
                    hist_batch_loss.append(step_fb_loss)

                # running += float(loss) * xb.size(0)
                # seen += xb.size(0)
                # if log_interval and step % log_interval == 0:
                #     pbar.set_postfix(retain_obj=_fmt_float(running / max(1, seen), 4), lr=_fmt_float(current_lr, 4))

            # Compute full-batch retain objective for logging/model selection


        elif algo == "svrg":
            # Snapshot and full gradient on retain set snapshot
            snapshot = copy.deepcopy(model).to(device)
            snapshot.train()
            # full_batch_grads, _ = compute_full_gradient(
            #     snapshot, uloaders.retain_full, device, weight_decay
            # )

            # running = 0.0
            # seen = 0
            for step, (xb, yb) in enumerate(uloaders.retain, start=1):
                xb, yb = xb.to(device), yb.to(device)

                g_cur = grad_batch(model, xb, yb, weight_decay)
                g_snap = grad_batch(snapshot, xb, yb, weight_decay)

                with torch.no_grad():
                    for p, gc, gs, mu in zip(model.parameters(), g_cur, g_snap, full_batch_grads):
                        p.add_(-current_lr * (gc - gs + mu))
                if batch_loss_interval and (step == 1 or step % batch_loss_interval == 0):
                    _, step_fb_loss = compute_full_gradient(model, uloaders.retain_full, device, weight_decay, no_grads=True)
                    hist_batch_loss.append(step_fb_loss)

        elif algo == "seqvru":
            # SVRG-style updates but the full-batch gradient (mu) is computed on the FORGET set
            snapshot = copy.deepcopy(model).to(device)
            snapshot.train()
            forget_full_grads, _ = compute_full_gradient(
                snapshot, uloaders.forget, device, weight_decay
            )
            # Convert forget full gradient into a retain proxy via VRU scaling
            mu = [-(real_r_f / (1.0 - real_r_f)) * g for g in forget_full_grads]

            for step, (xb, yb) in enumerate(uloaders.retain, start=1):
                xb, yb = xb.to(device), yb.to(device)

                g_cur = grad_batch(model, xb, yb, weight_decay)
                g_snap = grad_batch(snapshot, xb, yb, weight_decay)

                with torch.no_grad():
                    for p, gc, gs, mu_i in zip(model.parameters(), g_cur, g_snap, mu):
                        p.add_(-current_lr * (gc - gs + mu_i))
                if batch_loss_interval and (step == 1 or step % batch_loss_interval == 0):
                    _, step_fb_loss = compute_full_gradient(model, uloaders.retain_full, device, weight_decay, no_grads=True)
                    hist_batch_loss.append(step_fb_loss)

        elif algo in {"vru", "cheat-vru", "semi-cheat-vru"}:
            # running = 0.0
            # seen = 0
            for step, (xb, yb) in enumerate(uloaders.retain, start=1):
                xb, yb = xb.to(device), yb.to(device)

                g_cur = grad_batch(model, xb, yb, weight_decay)
                g_full = grad_batch(full_model, xb, yb, weight_decay)

                with torch.no_grad():
                    for p, gc, gs, mu in zip(model.parameters(), g_cur, g_full, retain_grad_full):
                        p.add_(-current_lr * (gc - gs + mu))

                    # Apply projection if requested
                    if project:
                        param_diff_sq = sum(torch.sum((p - p0)**2)
                                           for p, p0 in zip(model.parameters(), init_params))
                        current_dist = torch.sqrt(param_diff_sq)
                        if current_dist > projection_radius:
                            scale = projection_radius / current_dist
                            for p, p0 in zip(model.parameters(), init_params):
                                p.copy_(p0 + scale * (p - p0))

                    # Record full-batch retain loss after this step
                    if batch_loss_interval and (step == 1 or step % batch_loss_interval == 0):
                        _, step_fb_loss = compute_full_gradient(model, uloaders.retain_full, device, weight_decay, no_grads=True)
                        hist_batch_loss.append(step_fb_loss)

                    # Update averaging if requested
                    if average:
                        num_updates += 1
                        alpha_avg = 1.0 / num_updates
                        for avg_p, p in zip(avg_params, model.parameters()):
                            avg_p.mul_(1 - alpha_avg).add_(p, alpha=alpha_avg)

                        # Print the norms of gc and of the difference between gs and mu
                    # print('Norm of g cur, g full, retain_full_grad:', np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in g_cur])),
                    #       np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in g_full])),
                    #       np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full])))
                    # print('Cosine g full and retain_full_grad:',
                    #       np.dot(np.concatenate([g.detach().cpu().flatten() for g in g_full]),
                    #              np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full])) /
                    #       (np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in g_full])) *
                    #        np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in retain_grad_full])) + 1e-10)
                    #       )

                # Approximate instantaneous retain objective for logging
                # running += float(objective_loss(model, xb, yb, criterion, weight_decay)) * xb.size(0)
                # seen += xb.size(0)
                # if log_interval and step % log_interval == 0:
                #     pbar.set_postfix(retain_obj=running / max(1, seen), lr=current_lr)




        # Track best by retain full-batch objective
        if retain_fb_loss < best_loss_this_run:
            best_loss_this_run = retain_fb_loss
            best_state = copy.deepcopy(model.state_dict())

        # # Validation metrics on original val set
        # val_loss, val_acc = evaluate_model(model, loaders.val, weight_decay, device)
        # # Retain/Forget signals (on the *train* split subsets)
        # retain_loss_e, retain_acc_e = evaluate_model(model, uloaders.retain_full, weight_decay, device)
        # forget_loss_e, forget_acc_e = evaluate_model(model, uloaders.forget, weight_decay, device)

        if best_loss < float("inf"):
            pbar.set_postfix(retain_excess = retain_fb_loss - best_loss, lr=_fmt_float(current_lr, 3))
        else:
            pbar.set_postfix(retain_loss = _fmt_float(retain_fb_loss, 6), lr=_fmt_float(current_lr, 3))


    # Use averaged model if requested (for VRU algorithms)
    if average and algo in {"vru", "cheat-vru", "semi-cheat-vru"}:
        with torch.no_grad():
            for p, avg_p in zip(model.parameters(), avg_params):
                p.copy_(avg_p)
        print("Loaded averaged model for final output")
    else:
        # Restore best (by retain objective), then test
        if best_state is not None:
            model.load_state_dict(best_state)

    # Unlearning outputs must never replace an independently retrained baseline.

    test_loss, test_acc = evaluate_model(model, loaders.test, weight_decay, device)

    out = {
        "retain_objective": hist_retain_obj,
        "grad_norm": hist_grad_norm,
        "param_diff": hist_param_diff,
        "batch_loss": hist_batch_loss,
        # "val_loss": hist_val_loss,
        # "val_acc": hist_val_acc,
        # "retain_acc": hist_retain_acc,
        # "forget_acc": hist_forget_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "retain_grad_cos_sim": retain_grad_cos_sim,
    }
    return out


# ---------------------- Comparison & Plotting ----------------------

def parse_vru_variant(algo: str) -> tuple[str, bool, bool]:
    """
    Parse VRU algorithm variant name into (base_algo, average, project).

    Supported variants:
    - "vru" -> ("vru", False, False)
    - "vru-avg" -> ("vru", True, False)
    - "vru-proj" -> ("vru", False, True)
    - "vru-avg-proj" -> ("vru", True, True)
    - Other algos -> (algo, False, False)

    Returns:
        (base_algo, average, project)
    """
    if algo == "vru-avg":
        return ("vru", True, False)
    elif algo == "vru-proj":
        return ("vru", False, True)
    elif algo == "vru-avg-proj":
        return ("vru", True, True)
    else:
        # For "vru", "cheat-vru", "sgd", "svrg", etc., no modifications
        return (algo, False, False)


def compare_unlearning_algorithms(
    methods: List[str],
    full_model: nn.Module,
    loaders: Loaders,
    uloaders: UnlearnLoaders,
    r_f: float,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    scheduler: str,
    discount_factor: float,
    momentum: float,
    weight_decay: float,
    log_interval: int,
    seed: int,
    model_type: str = "LR",
    save_path: Optional[str] = None,
    show: bool = True,
    dataset: str = "digits",
    global_average: bool = False,  # Global --average flag override
    global_project: bool = False,  # Global --project flag override
) -> None:
    """
    Fine-tune (unlearn) from the same pretrained base model using each method
    provided in `methods` (e.g., ["sgd", "svrg"]) and compare curves.
    """
    # Ensure method names are lowercase and unique while preserving order
    seen = set()
    methods = [m.lower() for m in methods if not (m.lower() in seen or seen.add(m.lower()))]

    # Best model for reference (trained from scratch on the RETAIN set)
    best_path = _best_model_path(seed, weight_decay, r_f, model_type, dataset, n_train=uloaders.n_retain)
    if not os.path.exists(best_path):
        print(f"No existing best scratch model found at: {best_path}, training one now...")
        folder_dataset = "digit" if dataset == "digits" else dataset
        # Modify the "loaders" object to only contain the retain set for training
        scratch_loader = Loaders(
            train=uloaders.retain,
            full_train=uloaders.retain_full,
            val=loaders.val,
            test=loaders.test,
            n_features=loaders.n_features,
            n_classes=loaders.n_classes,
            n_train=uloaders.n_retain,
        )
        train_model(
            copy.deepcopy(full_model).to(device),
            scratch_loader,
            device,
            algo="sgd",
            seed=seed,
            epochs=3001 if dataset in ("digits", "breast_cancer") else 1001,
            lr= 0.1 if (model_type == "LR" and dataset in ("digits", "breast_cancer")) else 0.01,
            scheduler="step",
            discount_factor=0.2,
            momentum=0.0,
            weight_decay=weight_decay,
            log_interval=0,
            save_folder=f"{folder_dataset}_scratch_models",
            r_f=r_f,
            model_type=model_type,
            n_train=uloaders.n_retain,
        )
    else:
        # print(f"Found existing best scratch model at: {best_path}")
        pass


    # Sanity check: what is the norm of the base model's gradient on the full dataset ?
    full_model = copy.deepcopy(full_model).to(device=device, dtype=torch.float64)
    base_grad, base_performance = compute_full_gradient(
        full_model, loaders.full_train, device, weight_decay, no_grads=False
    )
    grad_norm = np.linalg.norm(np.concatenate([g.detach().cpu().flatten() for g in base_grad]))
    print(model_type)
    if base_performance > (1 + weight_decay + (model_type=="NN")) or grad_norm > 0.02:
        print('\n')
        print(f"Base model full dataset gradient norm: {grad_norm}, performance {base_performance:.4f}")
        print('\n')
        raise ValueError("This is fishy, stopping here")

    # --- Run unlearning for each requested method ---
    histories = {}
    test_accs = {}
    best_lrs = {}

    for algo in methods:
        # Parse VRU variant (e.g., "vru-avg" -> ("vru", True, False))
        base_algo, use_average, use_project = parse_vru_variant(algo)

        # Global flags override variant-based settings (for backward compatibility)
        if global_average:
            use_average = True
        if global_project:
            use_project = True

        # Validate supported methods (forget() will also validate)
        if base_algo not in {"sgd", "svrg", "vru", "cheat-vru", "semi-cheat-vru", "seqvru"}:
            raise ValueError(f"Unsupported method '{algo}'. Supported: 'sgd', 'svrg', 'vru' (and variants), 'cheat-vru', 'semi-cheat-vru', 'seqvru'.")

        if type(lr) is float:
            lr = [lr]
        best_perf = float("inf")
        for lr_i in lr:
            model_i = copy.deepcopy(full_model).to(device)
            h = forget(
                model_i,
                loaders,
                uloaders,
                r_f,
                device,
                seed=seed,
                algo=base_algo,  # Use base algorithm name
                epochs=epochs,
                lr=lr_i,
                scheduler=scheduler,
                discount_factor=discount_factor,
                momentum=momentum,  # unused for SVRG
                weight_decay=weight_decay,
                log_interval=log_interval,
                model_type=model_type,
                dataset=dataset,
                average=use_average,  # NEW
                project=use_project,  # NEW
            )

            current_perf = min(h["retain_objective"]) + max(h["retain_objective"]) # The curve should not go above initial value as this indicated instability
            if current_perf < best_perf: # Only keep the performanvs of the best lr tried
                best_perf = current_perf
                histories[algo] = h
                test_accs[algo] = h["test_acc"]
                best_lrs[algo] = lr_i


    # Re-compute best performance in case it was improved
    best_state = load_best_scratch_state(seed, weight_decay, r_f, model_type, dataset)
    best_model = copy.deepcopy(full_model).to(device=device, dtype=torch.float64)
    best_model.load_state_dict(best_state)
    _, best_performance = compute_full_gradient(
        best_model, uloaders.retain_full, device, weight_decay, no_grads=False
    )

    # --- Plot: retain objective (excess), parameter distance, gradient distance ---
    epochs_axis = list(range(1, epochs + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    # (1) Retain objective (excess over best)
    ax = axes[0]
    for algo, h in histories.items():
        ax.plot(
            epochs_axis,
            [v - best_performance + 1e-10 for v in h["retain_objective"]],
            label=f"{algo.upper()} retain obj (lr={best_lrs.get(algo)})"
        )
        if min(h["retain_objective"]) + 1e-7 < best_performance:
            print('\n')
            print(f"WARNING: {algo.upper()} achieved a retain objective {min(h['retain_objective'])} better than the best scratch model {best_performance}. This should not happen if the code is ran properly.\n")


    ax.set_title("Retain objective (excess)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    # (2) Parameter distance history (L2 distance to best model)
    ax = axes[1]
    for algo, h in histories.items():
        ax.plot(
            epochs_axis,
            h["param_diff"],
            label=f"{algo.upper()} param diff (lr={best_lrs.get(algo)})"
        )
    ax.set_title("Parameter distance to best model")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 distance")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    # (3) Gradient difference history (L2 diff of full gradients)
    ax = axes[2]
    for algo, h in histories.items():
        ax.plot(
            epochs_axis,
            h["grad_norm"],
            label=f"{algo.upper()} grad diff (lr={best_lrs.get(algo)})"
        )
    # Get cosine similarity from any VRU history that has it
    cos_sim = next((h["retain_grad_cos_sim"] for h in histories.values() if h.get("retain_grad_cos_sim") is not None), None)
    grad_title = "Gradient norm" if cos_sim is None else f"Gradient norm (retain grad cos sim: {cos_sim:.3e})"
    ax.set_title(grad_title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    fig.suptitle(f"Unlearning on Digits: Fine-tuning on Retain (with {model_type})")

    # Compact display of r_f and weight_decay (use scientific notation when very small)
    wd_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"
    rf_str = f"{r_f:.1e}" if r_f < 1e-2 else f"{r_f:.3f}"

    param_str = (
        f"r_f={rf_str} | weight_decay={wd_str} | "
        f"lr={lr} | scheduler={scheduler} | discount_factor={discount_factor} | "
        f"epochs={epochs} | batch_size={uloaders.retain.batch_size} | "
    )
    fig.text(0.5, 0.01, param_str, ha="center", fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved unlearning comparison figure to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    # Print final summary
    summary = ", ".join([f"{algo.upper()}: test acc {test_accs[algo]*100:.2f}%" for algo in methods])
    print(f"FINAL — {summary}")


# ---------------------- Main ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare unlearning (SGD vs SVRG) by fine-tuning on retain set")
    # Data / model
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for retain fine-tuning")
    p.add_argument("--val-size", type=float, default=0.2, help="Validation fraction (of whole dataset)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction (of whole dataset)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-standardize", action="store_true", help="Disable feature standardization")
    p.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN"],
                   help="Model architecture: LR (logistic regression) or NN (3-layer MLP)")

    # Unlearning setup
    p.add_argument("--r_f", type=float, default=0.1, help="Fraction of forget data")
    p.add_argument("--algo", nargs='+', type=str, default=["sgd"],
                   choices=["sgd", "svrg", "vru", "vru-avg", "vru-proj", "vru-avg-proj",
                            "cheat-vru", "semi-cheat-vru", "seqvru"],
                   help="Algorithm(s) to run. VRU variants: vru-avg (with averaging), vru-proj (with projection), vru-avg-proj (both)")
    p.add_argument("--compare", action="store_true", help="Compare both unlearning algorithms side-by-side")

    # Optimization
    p.add_argument("--epochs", type=int, default=30, help="Unlearning epochs (fine-tuning steps)")
    p.add_argument("--lr", nargs="+", type=float, default=0.05, help="Learning rate for unlearning, multiple values can be used for comparison")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "step"], help="LR schedule")
    p.add_argument("--discount-factor", type=float, default=0.3, help="LR discount factor (step scheduler)")
    p.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (SGD only)")
    p.add_argument("--weight-decay", type=float, default=0.1, help="L2 regularization strength in objective")
    p.add_argument("--log-interval", type=int, default=100, help="Steps between logs within an epoch (0 to disable)")

    # Plotting
    p.add_argument("--save-plot", type=str, default="unlearn_compare.png", help="Where to save the comparison plot (when --compare)")
    p.add_argument("--show-plot", action="store_true", help="Display the plot window (when --compare)")
    p.add_argument("--dataset", type=str, default="digits", choices=["digits", "mnist", "breast_cancer"],
                   help="Dataset to use")

    # VRU enhancements
    p.add_argument("--average", action="store_true",
                   help="Use Polyak averaging of all iterates for final model (VRU only)")
    p.add_argument("--project", action="store_true",
                   help="Project updates onto ball around initialization (VRU only)")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_val_test_sets, loaders = get_data_loaders(
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        standardize=not args.no_standardize,
        dataset=args.dataset,
    )
    train_ds, _, _ = train_val_test_sets

    # Build retain/forget from the *training* set
    uloaders = get_retain_forget_loaders(
        train_ds=train_ds,
        r_f=args.r_f,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"Retain size: {uloaders.n_retain} | Forget size: {uloaders.n_forget}")

    # Check if a best full model already exists for this setup
    base_model = build_model(args.model, loaders.n_features, loaders.n_classes).to(device)
    best_path = _best_model_path(args.seed, args.weight_decay, None, args.model, args.dataset, n_train=loaders.n_train)
    if not os.path.exists(best_path):
        print("No pretrained provided — pretraining base model on FULL train set with SGD...")
        folder_dataset = "digit" if args.dataset == "digits" else args.dataset
        for pretrain_lr in [0.1, 0.03, 0.01, 0.003]:
            if args.model == "NN":
                pretrain_lr = pretrain_lr / 10
            for optimizer in ["sgd", "adam"]:
                _ = train_model(
                    base_model,
                    loaders,
                    device,
                    algo="sgd",
                    seed=args.seed,
                    epochs=3000 if args.dataset in ("digits", "breast_cancer") else 1000,
                    lr = pretrain_lr,
                    scheduler="step",
                    optimizer=optimizer,
                    discount_factor=0.2,
                    momentum=0.0,
                    weight_decay=args.weight_decay,
                    log_interval=args.log_interval,
                    save_folder=f"{folder_dataset}_full_models",
                    model_type=args.model,
                    n_train=loaders.n_train,
                )
        print("Pretraining complete.")

    try:
        base_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    except TypeError:
        base_model.load_state_dict(torch.load(best_path, map_location=device))

    # Compare both or run a single algo
    if args.compare:
        compare_unlearning_algorithms(
            methods=args.algo,
            full_model=base_model,
            loaders=loaders,
            uloaders=uloaders,
            r_f=args.r_f,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            scheduler=args.scheduler,
            discount_factor=args.discount_factor,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            log_interval=args.log_interval,
            seed=args.seed,
            model_type=args.model,
            save_path=args.save_plot,
            show=args.show_plot,
            dataset=args.dataset,
            global_average=args.average,  # Pass global flags
            global_project=args.project,
        )
    else:
        # Single algorithm mode - take first algo from list
        algo_str = args.algo[0] if isinstance(args.algo, list) else args.algo

        # Parse VRU variant
        base_algo, use_average, use_project = parse_vru_variant(algo_str)

        # Global flags override variant-based settings
        if args.average:
            use_average = True
        if args.project:
            use_project = True

        model = copy.deepcopy(base_model).to(device)
        print(f"Unlearning algorithm: {algo_str.upper()}")
        stats = forget(
            model,
            loaders,
            uloaders,
            args.r_f,
            device,
            seed=args.seed,
            algo=base_algo,  # Use base algorithm
            epochs=args.epochs,
            lr=args.lr,
            scheduler=args.scheduler,
            discount_factor=args.discount_factor,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            log_interval=args.log_interval,
            model_type=args.model,
            dataset=args.dataset,
            average=use_average,  # NEW
            project=use_project,  # NEW
        )
        # print(
        #     f"Done. Test loss: {stats['test_loss']:.4f} | Test acc: {stats['test_acc']*100:.2f}% "
        #     f"| Best retain objective: {stats['best_retain_objective']:.4f}"
        # )


if __name__ == "__main__":
    main()
