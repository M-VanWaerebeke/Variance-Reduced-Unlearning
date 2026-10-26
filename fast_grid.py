
#!/usr/bin/env python3
"""
fast_grid.py — a lightweight, fast grid over (error, kappa_dp) using a
single unlearning run with noise_after=True simulated post-hoc.

High-level flow
----------------
1) Ensure scratch (retrained) baseline exists for the RETAIN set.
2) Ensure a full-model pretrained checkpoint exists.
3) Run ONE unlearning pass for the requested method, checkpointing the model
   every N steps (default: 10). The scratch model remains an independent
   retraining baseline.
4) Run ONE retraining (scratch) pass on RETAIN only, checkpointing every N steps.
5) For a grid of kappa_dp values, simulate "noise_after" by adding Gaussian
   noise to each saved UNLEARN checkpoint and evaluate the RETAIN full-batch
   loss. For the SCRATCH baseline, evaluate each saved SCRATCH checkpoint
   (kappa_dp=None, i.e., no noise added).
6) Convert the loss-vs-step traces into a time-to-threshold grid for a set of
   error targets; then reproduce the 2×2 plot like grid.py:
     • ratio (unlearn/scratch),
     • unlearn times,
     • scratch times,
     • count grid.

Notes
-----
• Dataset is fixed to scikit-learn Digits via the existing digit_* helpers.
• noise_after is conceptually enforced by post-hoc noise on saved checkpoints.
• We keep the code self-contained (no ExperimentRecord / replay infra).
• If your repository already provides a precise DP noise calibration for
  noise_after, replace `apply_noise_after()` with your implementation.

This script borrows utilities from iterations/gradient_representation.py.
"""
from __future__ import annotations
import argparse
import copy
import os
import sys
from typing import Dict, List, Optional, Tuple
import datetime

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.figure import Figure
from tqdm import tqdm

# Detect if we're in a log file (not a terminal) to reduce tqdm output
IN_LOGFILE = not sys.stdout.isatty()

# ---- Reuse helpers from the existing codebase ----
from gradient_representation import (
    set_seed,
    get_data_loaders,
    compute_full_gradient,
    grad_batch,
    Loaders,
    build_model,
)
from gradient_representation import (
    get_retain_forget_loaders,
    UnlearnLoaders,
)

# from digit_train import train as train_model

# (Private helpers in gradient_representation.py — we can import them)
from gradient_representation import (
    _ensure_pretrained_full_model,  # returns full-model path
    _ensure_scratch_model,          # returns scratch path (RETAIN-only model)
)


# ---------------------- Noise Application (Centralized) ----------------------

def apply_gaussian_noise_to_model(
    model: nn.Module,
    std: float,
    rng: np.random.Generator,
    device: torch.device = None,
) -> None:
    """
    Apply Gaussian noise to all floating-point parameters of a model in-place.

    This is the centralized noise application function used throughout the codebase
    for privacy-preserving unlearning. The noise is sampled from N(0, std^2).

    Args:
        model: The model to apply noise to (modified in-place)
        std: Standard deviation of the Gaussian noise
        rng: NumPy random generator for reproducibility
        device: Optional device override (uses parameter device by default)

    Note:
        Uses numpy RNG to avoid PyTorch RNG state entanglement across calls.
        The noise is converted to the parameter's dtype and device automatically.
    """
    if std <= 0:
        return  # No noise to apply

    with torch.no_grad():
        for p in model.parameters():
            if p.dtype.is_floating_point:
                target_device = device if device is not None else p.device
                noise = torch.from_numpy(
                    rng.normal(loc=0.0, scale=std, size=p.shape)
                ).to(target_device, dtype=p.dtype)
                p.add_(noise)


# ---------------------- Args ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast grid via single unlearning + noise-after simulation (Digits)")

    # Core grid
    p.add_argument("--divides", type=int, default=24, help="Number of grid divisions on each axis")
    p.add_argument("--min_error", type=float, default=1e-4, help="Min error/target loss for x-axis")
    p.add_argument("--max_error", type=float, default=1.0, help="Max error/target loss for x-axis")
    p.add_argument("--min_kappa_dp", type=float, default=1e-3, help="Min kappa_dp for y-axis")
    p.add_argument("--max_kappa_dp", type=float, default=1e1, help="Max kappa_dp for y-axis")
    p.add_argument("--scale", nargs=2, default=["log", "log"], choices=[["log", "log"], ["log", "lin"], ["lin", "log"], ["lin", "lin"]],
                   help="Scaling for (errors, kappa_dp)")
    p.add_argument("--level_lines", nargs="+", type=float, default=None, help="Optional levels for ratio contours")

    # Unlearning config
    p.add_argument("--method", type=str, default="cheat-vru",
                   choices=["nft", "vru", "cheat-vru", "svrg", "seqvru", "neggrad+", "scrub"],
                   help="Unlearning method to run once")
    p.add_argument("--compare_unlearning", action="store_true",
                   help="If True, compare two unlearning methods (method vs method2) instead of unlearning vs scratch.")
    p.add_argument("--method2", type=str, default="nft",
                   choices=["nft", "vru", "cheat-vru", "svrg", "seqvru", "neggrad+", "scrub"],
                   help="Second unlearning method when --compare_unlearning is set.")
    p.add_argument("--r_f", type=float, default=0.1, help="Fraction of forget data (0,1)")
    p.add_argument("--alpha", type=float, default=0.1, help="Balancing coefficient for NegGrad+")

    # Optimization
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (fixed, no sweep)")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "step"])
    p.add_argument("--scheduler_step_freq", type=int, default=100, help="Step frequency for scheduler (epochs)")
    p.add_argument("--discount_factor", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)

    # Checkpointing / evaluation
    p.add_argument("--checkpoint_every", type=str, default="10",
                   help='Checkpointing policy: an integer N to save every N steps, or "adaptive" to save only when the current best retain full-batch loss is beaten.')
    p.add_argument("--noise_sigma", type=float, default=1e-5,
                   help="Base noise sigma; actual std = noise_sigma / max(kappa_dp, 1e-12)")
    p.add_argument("--noising_repetitions", type=int, default=10, help="How many noise samples to average for noise_after (expected loss)")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save_prefix", type=str, default="fast_grid_digits")
    p.add_argument("--dataset", type=str, default="digits", choices=["digits", "mnist", "breast_cancer"],
                   help="Dataset to use")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN"],
                   help="Model architecture: LR (logistic regression) or NN (neural network)")
    p.add_argument("--average", action="store_true",
                   help="Use Polyak averaging of all iterates for final model (VRU only)")
    p.add_argument("--project", action="store_true",
                   help="Project updates onto ball around initialization (VRU only)")

    return p.parse_args()


# ---------------------- Noise-after simulation ----------------------

def _param_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).to(dtype=torch.float64).cpu() for p in model.parameters() if p.dtype.is_floating_point])

def averaged_noised_retain_loss(
    model: nn.Module,
    *,
    best_scratch_model: nn.Module,
    kappa_dp: float,
    repetitions: int,
    uloaders: UnlearnLoaders,
    device: torch.device,
    weight_decay: float,
    rng: np.random.Generator,
    eval_loader: Optional[DataLoader] = None,
) -> float:
    """Compute expected loss on `eval_loader` (retain by default) over noised copies.
    Noise std is kappa_dp * ||theta - theta*||_2 where theta* is the scratch optimum.
    """
    with torch.no_grad():
        d = torch.norm(
            _param_vector(model).to(torch.float64) - _param_vector(best_scratch_model).to(torch.float64),
            p=2,
        ).item()
        std = float(kappa_dp) * float(d)
    losses = []
    for _ in range(repetitions):
        m = copy.deepcopy(model).to(device=device, dtype=torch.float64)
        apply_gaussian_noise_to_model(m, std, rng, device)
        loader = eval_loader if eval_loader is not None else uloaders.retain_full
        loss = _evaluate_loader_loss(m, loader, device, weight_decay)
        losses.append(loss)
    return float(np.mean(losses)) if losses else float('inf')


# ---------------------- Runs with checkpointing ----------------------

def _evaluate_loader_loss(model: nn.Module, eval_loader: DataLoader, device: torch.device, weight_decay: float) -> float:
    """Evaluate the regularized objective on an explicit data loader."""
    _, fb_loss = compute_full_gradient(model, eval_loader, device, weight_decay, no_grads=True, as_double=True)
    return float(fb_loss)


def _evaluate_retain_loss(model: nn.Module, uloaders: UnlearnLoaders, device: torch.device, weight_decay: float) -> float:
    return _evaluate_loader_loss(model, uloaders.retain_full, device, weight_decay)


def _ensure_models(args, loaders, uloaders, device) -> Tuple[str, str, float, nn.Module]:
    """Ensure full-model and scratch-model exist. Return (full_path, scratch_path, scratch_loss, scratch_model)."""
    model_type = getattr(args, 'model', 'LR').upper()
    full_path = _ensure_pretrained_full_model(
        seed=args.seed,
        weight_decay=args.weight_decay,
        model_type=model_type,
        loaders=loaders,
        device=device,
        dataset=args.dataset,
    )
    scratch_path = _ensure_scratch_model(
        seed=args.seed,
        weight_decay=args.weight_decay,
        r_f=args.r_f,
        model_type=model_type,
        loaders=loaders,
        uloaders=uloaders,
        device=device,
        dataset=args.dataset,
    )
    scratch_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
    try:
        scratch_state = torch.load(scratch_path, map_location=device, weights_only=True)
    except TypeError:
        scratch_state = torch.load(scratch_path, map_location=device)
    # ensure float64
    for k, v in list(scratch_state.items()):
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            scratch_state[k] = v.to(dtype=torch.float64)
    scratch_model.load_state_dict(scratch_state)
    scratch_loss = _evaluate_retain_loss(scratch_model, uloaders, device, args.weight_decay)
    return full_path, scratch_path, scratch_loss, scratch_model


def _load_full_base(full_path: str, loaders, device, model_type: str = "LR") -> nn.Module:
    base = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
    try:
        state = torch.load(full_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(full_path, map_location=device)
    for k, v in list(state.items()):
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            state[k] = v.to(dtype=torch.float64)
    base.load_state_dict(state)
    return base


def run_unlearning_with_checkpoints(
    *,
    method: str,
    base_model: nn.Module,
    loaders,
    uloaders: UnlearnLoaders,
    r_f: float,
    device: torch.device,
    epochs: int,
    lr: float,
    scheduler: str,
    scheduler_step_freq: int,
    discount_factor: float,
    weight_decay: float,
    checkpoint_every: int,
    scratch_best_path: str,
    scratch_best_loss: float,
    alpha: float = 0.05,
    verbose: bool = True,
    average: bool = False,
    project: bool = False,
) -> Tuple[List[Tuple[int, Dict[str, torch.Tensor], float]], List[Tuple[int, float]]]:
    """Run an unlearning method and checkpoint model every `checkpoint_every` steps.

    Returns (checkpoints, full_loss_history), where checkpoints is a list of
    (global_step, state_dict, retain_loss) tuples and full_loss_history stores
    the retain full-batch loss at every optimization step.

    Args:
        average: If True, use Polyak averaging of all iterates for final model (VRU only)
        project: If True, project updates onto ball around initialization (VRU only)
        scratch_best_path, scratch_best_loss: Kept for caller compatibility; the
            scratch baseline is never updated from an unlearning trajectory.
    """
    # Normalize checkpointing policy: integer frequency or "adaptive"
    adaptive_ckpt = isinstance(checkpoint_every, str) and checkpoint_every.lower() == "adaptive"
    if not adaptive_ckpt:
        try:
            checkpoint_every_int = int(checkpoint_every)  # allow passing str digits from argparse
            if checkpoint_every_int <= 0:
                checkpoint_every_int = 1
        except (TypeError, ValueError):
            checkpoint_every_int = 10
    # Track best loss observed at the moment of checkpointing (for adaptive policy)
    last_ckpt_best = float("inf")

    method = method.lower()
    assert method in {"nft", "vru", "cheat-vru", "svrg", "seqvru", "neggrad+", "scrub"}
    model = copy.deepcopy(base_model).to(device=device, dtype=torch.float64)
    model.train()

    # Snapshot & retain-grad reference for VRU variants / SVRG
    full_model = None
    retain_grad_full = None
    if method in {"vru", "cheat-vru", "svrg", "seqvru", "scrub"}:
        full_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
        full_model.train()
        if method == "scrub":
            full_model.eval()

        if method in {"vru", "seqvru"}:
            forget_grad_full, _ = compute_full_gradient(
                full_model, uloaders.forget, device, weight_decay, no_grads=False
            )
            # Compute the actual ratio of forget data to total data, that might differ from r_f because of the finite amount of data
            real_r_f = len(uloaders.forget.dataset) / (len(uloaders.retain.dataset) + len(uloaders.forget.dataset))

            retain_grad_full = [-(real_r_f / (1.0 - real_r_f)) * g for g in forget_grad_full]

            # Initialize projection if requested
            if project:
                G_f = np.linalg.norm(
                    np.concatenate([g.detach().cpu().flatten().numpy() for g in forget_grad_full])
                )
                mu = weight_decay
                projection_radius = (G_f / mu) * (real_r_f / (1 - real_r_f))
                init_params = [p.clone().detach() for p in model.parameters()]
                if verbose:
                    print(f"Projection enabled: radius={projection_radius:.6f} (G_f={G_f:.6f}, mu={mu}, r_f={real_r_f:.4f})")

            # Initialize averaging if requested
            if average:
                avg_params = [p.clone().detach() for p in model.parameters()]
                num_updates = 0
                if verbose:
                    print("Averaging enabled: Polyak averaging of all iterates")
        elif method in {"cheat-vru", "svrg"}:  # cheat-VRU or SVRG
            retain_grad_full, _ = compute_full_gradient(
                full_model, uloaders.retain_full, device, weight_decay, no_grads=False
            )

            # Initialize projection for cheat-vru (use retain gradient norm instead)
            if project and method == "cheat-vru":
                G_f = np.linalg.norm(
                    np.concatenate([g.detach().cpu().flatten().numpy() for g in retain_grad_full])
                )
                # For cheat-vru, we don't have real_r_f from forget set, use the passed r_f
                real_r_f = r_f
                mu = weight_decay
                projection_radius = (G_f / mu) * (real_r_f / (1 - real_r_f))
                init_params = [p.clone().detach() for p in model.parameters()]
                if verbose:
                    print(f"Projection enabled: radius={projection_radius:.6f} (G_f={G_f:.6f}, mu={mu}, r_f={real_r_f:.4f})")

            # Initialize averaging for cheat-vru
            if average and method == "cheat-vru":
                avg_params = [p.clone().detach() for p in model.parameters()]
                num_updates = 0
                if verbose:
                    print("Averaging enabled: Polyak averaging of all iterates")

    step_count = 0
    checkpoints: List[Tuple[int, Dict[str, torch.Tensor], float]] = []
    full_loss_history: List[Tuple[int, float]] = []

    # Save initialization (Step 0)
    loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
    checkpoints.append((step_count, copy.deepcopy(model.state_dict()), loss64))
    last_ckpt_best = loss64

    # fourths = [epochs // 4, (2 * epochs) // 4, (3 * epochs) // 4]
    thirds = [epochs // 3, (2 * epochs) // 3]
    current_lr = lr
    forget_iter = iter(uloaders.forget)

    # Progress bar for unlearning
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"{method.upper()}", disable=not verbose, leave=False,
                     mininterval=float('inf') if IN_LOGFILE else 0.1)
    for epoch in epoch_bar:
        # if epoch in thirds and scheduler == "step":
        if method == "svrg" and epoch in thirds: # (epoch % 100 == 0):
            full_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
            full_model.train()
            retain_grad_full, _ = compute_full_gradient(
                full_model, uloaders.retain_full, device, weight_decay, no_grads=False
            )
        elif method == "seqvru" and (epoch % 100 == 0): # epoch in thirds:
            full_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
            full_model.train()
            forget_grad_full, _ = compute_full_gradient(
                full_model, uloaders.forget, device, weight_decay, no_grads=False
            )
            retain_grad_full = [-(real_r_f / (1.0 - real_r_f)) * g for g in forget_grad_full]

        epoch_loader = uloaders.retain
        is_scrub_max = False
        if method == "scrub":
            # Alternating schedule: Maximize (Forget) then Minimize (Retain).
            # Constraint: The last epoch must be Minimize.
            # If total epochs and current epoch have different parity, it's a Max step.
            is_scrub_max = (epochs % 2 != epoch % 2)
            if is_scrub_max:
                epoch_loader = uloaders.forget

        for xb, yb in epoch_loader:
            xb = xb.to(device=device, dtype=torch.float64)
            yb = yb.to(device)

            # estimator
            if method == "nft":
                g_est = grad_batch(model, xb, yb, weight_decay)
            elif method == "neggrad+":
                try:
                    xb_f, yb_f = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(uloaders.forget)
                    xb_f, yb_f = next(forget_iter)
                xb_f = xb_f.to(device=device, dtype=torch.float64)
                yb_f = yb_f.to(device)
                g_retain = grad_batch(model, xb, yb, weight_decay)
                g_forget = grad_batch(model, xb_f, yb_f, weight_decay)
                g_est = [gr - alpha * gf for gr, gf in zip(g_retain, g_forget)]
            elif method == "scrub":
                if is_scrub_max:
                    # Maximization Step (Forget Set): Maximize KL(Teacher||Student)
                    # Equivalent to minimizing -KL.
                    # Update: theta = theta + lr * alpha * grad(KL)
                    # g_est should be -alpha * grad(KL) so that p.add_(-lr * g_est) works.
                    with torch.no_grad():
                        teacher_logits = full_model(xb)
                        teacher_probs = torch.softmax(teacher_logits, dim=1)
                    student_logits = model(xb)
                    student_logprobs = torch.log_softmax(student_logits, dim=1)
                    kl_loss = nn.KLDivLoss(reduction="batchmean")(student_logprobs, teacher_probs)
                    g_kl = torch.autograd.grad(kl_loss, list(model.parameters()), create_graph=False)
                    g_est = [-alpha * g.detach() for g in g_kl]
                else:
                    # Minimization Step (Retain Set): Minimize CE + KL(Teacher||Student)
                    g_ce = grad_batch(model, xb, yb, weight_decay)
                    with torch.no_grad():
                        teacher_logits = full_model(xb)
                        teacher_probs = torch.softmax(teacher_logits, dim=1)
                    student_logits = model(xb)
                    student_logprobs = torch.log_softmax(student_logits, dim=1)
                    kl_loss = nn.KLDivLoss(reduction="batchmean")(student_logprobs, teacher_probs)
                    g_kl = torch.autograd.grad(kl_loss, list(model.parameters()), create_graph=False)
                    # Combine gradients
                    g_est = [gc + gk.detach() for gc, gk in zip(g_ce, g_kl)]
            else: # VRU
                g_cur = grad_batch(model, xb, yb, weight_decay)
                g_snap = grad_batch(full_model, xb, yb, weight_decay)
                g_est = [gc - gs + mu for gc, gs, mu in zip(g_cur, g_snap, retain_grad_full)]

            # Retain loss for diagnostics; no full-set gradient is needed here.
            _, fb_loss = compute_full_gradient(
                model, uloaders.retain_full, device, weight_decay, no_grads=True
            )
            # Store the noiseless retain loss for every step (full history)
            full_loss_history.append((step_count, float(fb_loss)))

            # If using adaptive checkpointing, save a checkpoint whenever we beat the previous best
            if adaptive_ckpt and (fb_loss < last_ckpt_best):
                # print('Imroved best loss at step ', step_count)
                # Recompute high-precision retain loss for the checkpoint payload
                loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
                checkpoints.append((step_count, copy.deepcopy(model.state_dict()), loss64))
                last_ckpt_best = float(fb_loss)

            # step (update)
            with torch.no_grad():
                if method == "seqvru":
                    for p, ge in zip(model.parameters(), g_est):
                        p.add_(current_lr * ge)
                else:
                    for p, ge in zip(model.parameters(), g_est):
                        p.add_(-current_lr * ge)

                # Apply projection for VRU/cheat-vru if requested
                if project and method in {"vru", "cheat-vru", "seqvru"}:
                    param_diff_sq = sum(torch.sum((p - p0)**2)
                                       for p, p0 in zip(model.parameters(), init_params))
                    current_dist = torch.sqrt(param_diff_sq)
                    if current_dist > projection_radius:
                        scale = projection_radius / current_dist
                        for p, p0 in zip(model.parameters(), init_params):
                            p.copy_(p0 + scale * (p - p0))

                # Update averaging for VRU/cheat-vru if requested
                if average and method in {"vru", "cheat-vru", "seqvru"}:
                    num_updates += 1
                    alpha_avg = 1.0 / num_updates
                    for avg_p, p in zip(avg_params, model.parameters()):
                        avg_p.mul_(1 - alpha_avg).add_(p, alpha=alpha_avg)

            step_count += 1

            # Periodic checkpointing if using an integer frequency (save AFTER update)
            if (not adaptive_ckpt) and (step_count % checkpoint_every_int == 0):
                loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
                checkpoints.append((step_count, copy.deepcopy(model.state_dict()), loss64))

        if (epoch % scheduler_step_freq == 0) and scheduler == "step":
            current_lr *= discount_factor

        # Update progress bar
        if verbose and epoch % 10 == 0:
            epoch_bar.set_postfix({"loss": f"{float(fb_loss):.4f}", "lr": f"{current_lr:.2e}"})

    # Load averaged model for final output if requested
    averaged_output = average and method in {"vru", "cheat-vru", "seqvru"}
    if averaged_output:
        with torch.no_grad():
            for p, avg_p in zip(model.parameters(), avg_params):
                p.copy_(avg_p)
        if verbose:
            print("Loaded averaged model for final output")

    # The periodic checkpoint at step_count was saved before loading the
    # Polyak average. Replace it so callers receive the actual final output.
    if averaged_output:
        loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
        final_checkpoint = (step_count, copy.deepcopy(model.state_dict()), loss64)
        if checkpoints and checkpoints[-1][0] == step_count:
            checkpoints[-1] = final_checkpoint
        else:
            checkpoints.append(final_checkpoint)
    # Ensure the final step is logged if using periodic checkpointing
    elif (not adaptive_ckpt) and (step_count > 0):
        last_saved = checkpoints[-1][0] if checkpoints else -1
        if last_saved != step_count:
            loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
            checkpoints.append((step_count, copy.deepcopy(model.state_dict()), loss64))

    # Ensure the final step is added to the full loss history
    if full_loss_history and full_loss_history[-1][0] != step_count:
        final_loss = _evaluate_retain_loss(model, uloaders, device, weight_decay)
        full_loss_history.append((step_count, final_loss))

    return checkpoints, full_loss_history



def run_scratch_training_loss_trace(
    *,
    base_model: nn.Module,
    loaders,
    uloaders: UnlearnLoaders,
    device: torch.device,
    epochs: int,
    lr: float,
    scheduler: str,
    discount_factor: float,
    weight_decay: float,
    log_every: int,
    model_type: str = "LR",
) -> Tuple[List[Tuple[int, float]], float, Optional[Dict[str, torch.Tensor]]]:
    """Retrain on RETAIN only from a fresh random initialization (not the full model); return a trace of (global_step, retain_loss) sampled every `log_every` steps."""
    # Start from a fresh, randomly initialized model (true "scratch")
    model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
    model.train()

    # thirds = [epochs // 3, (2 * epochs) // 3]
    fourths = [epochs // 4, (2 * epochs) // 4, (3 * epochs) // 4]
    current_lr = lr
    step_count = 0
    trace: List[Tuple[int, float]] = []

    # Normalize logging policy: integer frequency or "adaptive"
    adaptive_log = isinstance(log_every, str) and log_every.lower() == "adaptive"
    if not adaptive_log:
        try:
            log_every_int = int(log_every)
            if log_every_int <= 0:
                log_every_int = 1
        except (TypeError, ValueError):
            log_every_int = 10
    last_logged_best = float("inf")
    TOL_LOG = 1e-12

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        # if epoch in thirds and scheduler == "step"
        if (epoch in fourths) and scheduler == "step":
            current_lr *= discount_factor
        for xb, yb in uloaders.retain:
            xb = xb.to(device=device, dtype=torch.float64)
            yb = yb.to(device)
            g_est = grad_batch(model, xb, yb, weight_decay)
            with torch.no_grad():
                for p, ge in zip(model.parameters(), g_est):
                    p.add_(-current_lr * ge)
            step_count += 1

            if adaptive_log:
                loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
                if loss64 < last_logged_best - TOL_LOG:
                    trace.append((step_count, loss64))
                    last_logged_best = loss64
                if loss64 < best_loss - TOL_LOG:
                    best_loss = float(loss64)
                    best_state = copy.deepcopy(model.state_dict())
            else:
                if (step_count % log_every_int) == 0:
                    loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
                    trace.append((step_count, loss64))
                    if loss64 < best_loss - TOL_LOG:
                        best_loss = float(loss64)
                        best_state = copy.deepcopy(model.state_dict())

    # Ensure the final step is logged
    if not trace or trace[-1][0] != step_count:
        loss64 = _evaluate_retain_loss(model, uloaders, device, weight_decay)
        trace.append((step_count, loss64))
        if loss64 < best_loss - TOL_LOG:
            best_loss = float(loss64)
            best_state = copy.deepcopy(model.state_dict())

    return trace, best_loss, best_state


# ---------------------- Grid assembly ----------------------

def _make_axis(min_v: float, max_v: float, n: int, kind: str) -> np.ndarray:
    if kind == "log":
        return np.logspace(np.log10(min_v), np.log10(max_v), n)
    return np.linspace(min_v, max_v, n)


def _axis_centers(axis: np.ndarray, kind: str) -> np.ndarray:
    """Return bin centers for an axis built via `_make_axis`."""
    left = axis[:-1]
    right = axis[1:]
    if kind == "log":
        return np.sqrt(left * right)
    return 0.5 * (left + right)


def _finalize_curve_data(points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert list of (kappa, loss) tuples into aligned numpy arrays."""
    if not points:
        return np.array([]), np.array([])
    arr = np.array(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.array([]), np.array([])
    return arr[:, 0], arr[:, 1]


def _time_to_thresholds(
    checkpoints: List[Tuple[int, Dict[str, torch.Tensor], float]],
    *,
    model_template: nn.Module,
    best_scratch_model: nn.Module,
    best_scratch_loss: float,
    uloaders: UnlearnLoaders,
    device: torch.device,
    weight_decay: float,
    errors: np.ndarray,
    kappa_dp: Optional[float],
    repetitions: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    e_targets = 0.5 * (errors[:-1] + errors[1:])
    times = np.full((len(e_targets),), np.inf, dtype=float)
    history_steps: List[int] = []
    history_losses: List[float] = []
    for step_idx, state, _ in checkpoints:
        model = copy.deepcopy(model_template).to(device=device, dtype=torch.float64)
        model.load_state_dict(state)
        if kappa_dp is not None:
            loss = averaged_noised_retain_loss(
                model,
                best_scratch_model=best_scratch_model,
                kappa_dp=float(kappa_dp),
                repetitions=repetitions,
                uloaders=uloaders,
                device=device,
                weight_decay=weight_decay,
                rng=rng,
            )
        else:
            loss = _evaluate_retain_loss(model, uloaders, device, weight_decay)
        # Convert to EXCESS retain loss relative to the best scratch model
        loss_excess = float(loss - best_scratch_loss)
        if kappa_dp is not None:
            history_steps.append(int(step_idx))
            history_losses.append(float(loss_excess))
        for j, thr in enumerate(e_targets):
            if loss_excess <= thr and np.isinf(times[j]):
                times[j] = float(step_idx)
        if not np.isinf(times).any():
            break
    history = None
    if kappa_dp is not None and history_steps:
        history = (np.array(history_steps, dtype=float), np.array(history_losses, dtype=float))
    return times, history


def _select_best_checkpoint(
    checkpoints: List[Tuple[int, Dict[str, torch.Tensor], float]]
) -> Optional[Tuple[int, Dict[str, torch.Tensor], float]]:
    """Pick the checkpoint with the lowest recorded retain loss."""
    if not checkpoints:
        return None
    return min(checkpoints, key=lambda t: t[2])


# --- Helper: times from loss trace ---
def _times_from_loss_trace(trace: List[Tuple[int, float]], *, errors: np.ndarray, best_scratch_loss: float) -> np.ndarray:
    e_targets = 0.5 * (errors[:-1] + errors[1:])
    times = np.full((len(e_targets),), np.inf, dtype=float)
    for step_idx, loss in trace:
        loss_excess = float(loss - best_scratch_loss)
        for j, thr in enumerate(e_targets):
            if loss_excess <= thr and np.isinf(times[j]):
                times[j] = float(step_idx)
        if not np.isinf(times).any():
            break
    return times


# ---------------------- Plotting ----------------------

def _finalize_figure(fig: Figure, show: bool, figure_collector: Optional[List[Figure]]) -> None:
    """Keep figures alive for deferred display or close them immediately."""
    if show:
        if figure_collector is not None:
            figure_collector.append(fig)
        else:
            plt.show()
    else:
        plt.close(fig)

def _plot_grids(
    *,
    ratio_grid: np.ndarray,
    unlearn_grid: np.ndarray,
    second_grid: np.ndarray,
    count_grid: np.ndarray,
    errors: np.ndarray,
    kappas: np.ndarray,
    level_lines: Optional[List[float]],
    save_prefix: str,
    show: bool,
    first_label: str,
    second_label: str,
    figure_collector: Optional[List[Figure]] = None,
) -> None:
    cmap = 'jet'
    alpha = 0.85

    # trim last row/col like grid.py to avoid edge artifacts
    rg = ratio_grid[:-1, :-1]
    ug = unlearn_grid[:-1, :-1]
    sg = second_grid[:-1, :-1]
    cg = count_grid[:-1, :-1]
    e = errors[:-1]
    k = kappas[:-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    # Ratio heatmap bounds (log-scale, positive finite only)
    mask_ratio = (rg > 0) & np.isfinite(rg)
    if np.any(mask_ratio):
        ratio_vmin = float(np.nanmin(rg[mask_ratio]))
        ratio_vmax = float(np.nanmax(rg[mask_ratio]))
        if ratio_vmin == ratio_vmax:
            ratio_vmin, ratio_vmax = max(1e-4, ratio_vmin), ratio_vmax * 10.0 + 1e-4
    else:
        ratio_vmin, ratio_vmax = 1e-4, 1.0

    im_ratio = axes[0, 0].imshow(
        rg,
        cmap=cmap,
        interpolation='nearest',
        norm=LogNorm(vmin=ratio_vmin, vmax=ratio_vmax),
        alpha=alpha,
    )
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title(f'Ratio: {first_label} / {second_label} Time Grid')
    axes[0, 0].set_xlabel('Error target')
    axes[0, 0].set_ylabel('kappa_dp')

    # Contours for the ratio plot
    if level_lines:
        levels = level_lines
    else:
        try:
            levels = np.logspace(np.log10(ratio_vmin), np.log10(ratio_vmax), 10)
        except ValueError:
            levels = None
    if levels is not None and np.isfinite(ratio_vmin) and np.isfinite(ratio_vmax) and ratio_vmin > 0:
        try:
            contours = axes[0, 0].contour(
                rg,
                levels=levels,
                colors='white',
                linewidths=1.2,
                norm=LogNorm(vmin=ratio_vmin, vmax=ratio_vmax),
            )
            axes[0, 0].clabel(contours, inline=True, fontsize=10, fmt="%.2f")
        except ValueError:
            pass

    fig.colorbar(im_ratio, ax=axes[0, 0], orientation='vertical', alpha=alpha)

    # Compute shared bounds for time grids (positive finite only)
    mask_u = np.isfinite(ug) & (ug > 0)
    mask_s = np.isfinite(sg) & (sg > 0)
    if np.any(mask_u) or np.any(mask_s):
        vals = []
        if np.any(mask_u):
            vals.append(ug[mask_u])
        if np.any(mask_s):
            vals.append(sg[mask_s])
        t_vmin = float(min(v.min() for v in vals))
        t_vmax = float(max(v.max() for v in vals))
    else:
        t_vmin, t_vmax = 1.0, 1e5

    # Unlearn grid (Method A)
    im_u = axes[0, 1].imshow(
        ug,
        cmap=cmap,
        interpolation='nearest',
        norm=LogNorm(vmin=t_vmin, vmax=t_vmax),
        alpha=alpha,
    )
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title(f'{first_label} Time Grid')
    axes[0, 1].set_xlabel('Error target')
    fig.colorbar(im_u, ax=axes[0, 1], orientation='vertical', alpha=alpha)

    # Second grid (Method B or Scratch)
    im_s = axes[1, 0].imshow(
        sg,
        cmap=cmap,
        interpolation='nearest',
        norm=LogNorm(vmin=t_vmin, vmax=t_vmax),
        alpha=alpha,
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title(second_label + " Time Grid")
    axes[1, 0].set_xlabel('Error target')
    fig.colorbar(im_s, ax=axes[1, 0], orientation='vertical', alpha=alpha)

    # Count grid
    im_c = axes[1, 1].imshow(cg, cmap='plasma', interpolation='nearest', alpha=alpha)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title('Count Grid')
    axes[1, 1].set_xlabel('Error target')
    fig.colorbar(im_c, ax=axes[1, 1], orientation='vertical', alpha=alpha)

    # A few ticks: 4 evenly spaced in index space, labeled by numeric values
    def _ticks(vals: np.ndarray, n: int = 4):
        if len(vals) < 2:
            return [0], [f"{vals[0]:.2g}"]
        idx = np.linspace(0, len(vals) - 1, n, dtype=int)
        labels = [f"{vals[i]:.2g}" for i in idx]
        return idx, labels

    x_idx, x_lbl = _ticks(e)
    y_idx, y_lbl = _ticks(k)
    for ax in axes.flat:
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_lbl)
        ax.set_yticks(y_idx)
        ax.set_yticklabels(y_lbl)

    out_path = f"{save_prefix}_grid.pdf"
    plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    _finalize_figure(fig, show, figure_collector)
    print(f"Saved figure to {out_path}")


def _plot_excess_vs_kappa(
    method_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    save_prefix: str,
    show: bool,
    kappa_scale: str,
    figure_collector: Optional[List[Figure]] = None,
) -> Optional[str]:
    """Plot excess retain loss averaged over noised checkpoints vs kappa_dp."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    plotted = False
    positive_only = True

    for label, (k_vals, losses) in method_curves.items():
        if k_vals.size == 0 or losses.size == 0:
            continue
        mask = np.isfinite(k_vals) & np.isfinite(losses)
        if not np.any(mask):
            continue
        x = k_vals[mask]
        y = losses[mask]
        if x.size == 0 or y.size == 0:
            continue
        plotted = True
        if np.any(y <= 0):
            positive_only = False
        ax.plot(x, y, marker="o", label=label)

    if not plotted:
        plt.close(fig)
        return None

    if kappa_scale == "log":
        ax.set_xscale("log")
    if positive_only:
        ax.set_yscale("log")

    ax.set_xlabel("kappa_dp")
    ax.set_ylabel("Excess retain loss (noise-averaged)")
    ax.set_title("Excess retain loss vs kappa_dp")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    out_path = f"{save_prefix}_excess_vs_kappa.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    _finalize_figure(fig, show, figure_collector)
    print(f"Saved figure to {out_path}")
    return out_path

def _plot_noiseless_histories(
    unlearn_loss_history: List[Tuple[int, float]],
    scratch_trace: list,
    *,
    best_scratch_loss: float,
    save_prefix: str,
    show: bool,
    figure_collector: Optional[List[Figure]] = None,
) -> str:
    """
    Plot the retain full-batch (NOISELESS) **EXCESS** loss vs. step for:
      • the unlearning trajectory (full per-step history)
      • the scratch retraining trajectory

    Excess loss = retain_loss − best_scratch_loss (masked at
positive values for log scale).

    Parameters
    ----------
    unlearn_loss_history : List[Tuple[int, float]]
        Tuples of (global_step, retain_loss) returned by
        `run_unlearning_with_checkpoints` (full history, every step).
    scratch_trace : List[Tuple[int, float]]
        Tuples of (global_step, retain_loss) returned by
        `run_scratch_training_loss_trace`.
    best_scratch_loss : float
        The best (lowest) scratch retraining loss (used as baseline for excess loss).
    save_prefix : str
        File prefix used elsewhere in this script; we append `_excess_histories.pdf`.
    show : bool
        Whether to call plt.show() after saving.

    Returns
    -------
    path : str
        The path of the saved PDF figure.
    """

    # Unlearning (steps, EXCESS losses) from full per-step history
    steps_u = [int(t[0]) for t in unlearn_loss_history]
    raw_unlearn_losses = [float(t[1]) for t in unlearn_loss_history]
    losses_u = [lu - float(best_scratch_loss) for lu in raw_unlearn_losses]

    # Scratch retraining (steps, EXCESS losses) from trace
    steps_s = [int(t[0]) for t in scratch_trace]
    losses_s = [float(t[1]) - float(best_scratch_loss) for t in scratch_trace]

    # Mask nonpositive values for log scale
    losses_u = [lu if lu > 0 else float('nan') for lu in losses_u]
    losses_s = [ls if ls > 0 else float('nan') for ls in losses_s]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    if steps_u and any(np.isfinite(losses_u)):
        ax.plot(steps_u, losses_u, label="Unlearning (excess retain loss)")
    if steps_s and any(np.isfinite(losses_s)):
        ax.plot(steps_s, losses_s, label="Scratch retraining (excess retain loss)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Excess retain loss (vs. best scratch)")
    ax.set_yscale("log")
    ax.set_title("Excess retain loss histories (noiseless)")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    best_unlearn_loss = min(raw_unlearn_losses) if raw_unlearn_losses else float("nan")
    best_scratch_trace = min([float(t[1]) for t in scratch_trace]) if scratch_trace else float("nan")
    print(
        "In this plot, the best losses for unlearning, retrain and best are",
        best_unlearn_loss,
        best_scratch_trace,
        best_scratch_loss,
    )

    path = f"{save_prefix}_excess_histories.pdf"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    _finalize_figure(fig, show, figure_collector)
    return path

def _plot_noiseless_histories_compare(
    unlearn_loss_history_a: List[Tuple[int, float]],
    unlearn_loss_history_b: List[Tuple[int, float]],
    *,
    best_scratch_loss: float,
    labels: Tuple[str, str],
    save_prefix: str,
    show: bool,
    figure_collector: Optional[List[Figure]] = None,
) -> str:
    """
    Plot the retain full-batch (NOISELESS) **EXCESS** loss vs. step for two unlearning
    methods (A and B) using their full per-step histories, when comparing methods
    (i.e., --compare_unlearning=True).

    Excess loss = retain_loss − best_scratch_loss (masked at positive values for log scale).
    """
    # Steps and EXCESS losses for method A
    steps_a = [int(t[0]) for t in unlearn_loss_history_a]
    losses_a = [float(t[1]) - float(best_scratch_loss) for t in unlearn_loss_history_a]

    # Steps and EXCESS losses for method B
    steps_b = [int(t[0]) for t in unlearn_loss_history_b]
    losses_b = [float(t[1]) - float(best_scratch_loss) for t in unlearn_loss_history_b]

    # Mask nonpositive values for log scale
    losses_a = [la if la > 0 else float('nan') for la in losses_a]
    losses_b = [lb if lb > 0 else float('nan') for lb in losses_b]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    if steps_a and any(np.isfinite(losses_a)):
        ax.plot(steps_a, losses_a, label=str(labels[0]))
    if steps_b and any(np.isfinite(losses_b)):
        ax.plot(steps_b, losses_b, label=str(labels[1]))

    ax.set_xlabel("Step")
    ax.set_ylabel("Excess retain loss (vs. best scratch)")
    ax.set_yscale("log")
    ax.set_title("Excess retain loss histories (noiseless): method comparison")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()

    path = f"{save_prefix}_excess_histories_compare.pdf"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    _finalize_figure(fig, show, figure_collector)
    return path

# ---------------------- Main ----------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    figure_collector: Optional[List[Figure]] = [] if args.show else None

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}")

    # Create comprehensive hyperparameter string for figure directory and filenames
    dt_str = datetime.datetime.now().strftime("%y-%m-%d-%H%M")
    hp_str = f"d{args.dataset}_m{args.method}_rf{args.r_f:g}_wd{args.weight_decay:g}_bs{args.batch_size}_s{args.seed}_ep{args.epochs}_lr{args.lr:g}"

    # Create output directory for this run's figures
    figures_dir = os.path.join("fast_grid_figures", f"{dt_str}_{hp_str}")
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
        val_size=0.0,
        test_size=0.0,
        seed=args.seed,
        standardize=True,
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

    # Ensure models
    full_path, scratch_path, scratch_best_loss, scratch_best_model = _ensure_models(args, loaders, uloaders, device)

    # Load full base model
    base_model = _load_full_base(full_path, loaders, device)

    # Prepare error/kappa_dp axes
    errors = _make_axis(args.min_error, args.max_error, args.divides, args.scale[0])
    kappas = _make_axis(args.min_kappa_dp, args.max_kappa_dp, args.divides, args.scale[1])
    kappa_centers = _axis_centers(kappas, args.scale[1])
    rng = np.random.default_rng(args.seed)

    # Unlearning run (checkpointing)
    unlearn_checkpoints, unlearn_loss_history = run_unlearning_with_checkpoints(
        method=args.method,
        base_model=base_model,
        loaders=loaders,
        uloaders=uloaders,
        r_f=args.r_f,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        scheduler=args.scheduler,
        scheduler_step_freq=args.scheduler_step_freq,
        discount_factor=args.discount_factor,
        weight_decay=args.weight_decay,
        checkpoint_every=args.checkpoint_every,
        scratch_best_path=scratch_path,
        scratch_best_loss=scratch_best_loss,
        alpha=args.alpha,
        average=args.average,
        project=args.project,
    )

    method_curve_points: Dict[str, List[Tuple[float, float]]] = {args.method: []}

    if args.compare_unlearning:
        method_curve_points[args.method2] = []

    if not args.compare_unlearning:
        # Scratch retraining run (loss trace)
        scratch_trace, scratch_trace_best_loss, scratch_trace_best_state = run_scratch_training_loss_trace(
            base_model=base_model,
            loaders=loaders,
            uloaders=uloaders,
            device=device,
            epochs=args.epochs*10, #retraining usually takes longer
            lr=args.lr,
            scheduler=args.scheduler,
            discount_factor=args.discount_factor,
            weight_decay=args.weight_decay,
            log_every=args.checkpoint_every,
            model_type=args.model,
        )
        if scratch_trace_best_state is not None and scratch_trace_best_loss < scratch_best_loss:
            try:
                prev_scratch_loss = scratch_best_loss
                torch.save(scratch_trace_best_state, scratch_path)
                scratch_best_model.load_state_dict(scratch_trace_best_state)
                scratch_best_loss = float(scratch_trace_best_loss)
                print(
                    f"New best SCRATCH model saved to {scratch_path} (loss {scratch_trace_best_loss:.12g} < {prev_scratch_loss:.12g})"
                )
            except Exception as e:
                print(f"WARNING: failed to save improved scratch model: {e}")
        # errors represent EXCESS losses; subtract the best scratch retain loss
        scratch_times = _times_from_loss_trace(scratch_trace, errors=errors, best_scratch_loss=scratch_best_loss)

        # For each (error, kappa_dp), compute time to threshold for unlearn (with noise_after) and scratch (no noise)
        best_unlearn_ckpt = _select_best_checkpoint(unlearn_checkpoints)
        shape = (len(kappas)-1, len(errors)-1)
        unlearn_grid = np.full(shape, np.inf, dtype=float)
        second_grid = np.tile(scratch_times, (len(kappas)-1, 1))
        count_grid = np.zeros(shape, dtype=int)
        ratio_grid = np.full(shape, np.nan, dtype=float)

        for i, kappa in tqdm(enumerate(kappas[:-1]), total=len(kappas)-1,
                             mininterval=float('inf') if IN_LOGFILE else 0.1):
            unlearn_times, history = _time_to_thresholds(
                unlearn_checkpoints,
                model_template=base_model,
                best_scratch_model=scratch_best_model,
                best_scratch_loss=scratch_best_loss,
                uloaders=uloaders,
                device=device,
                weight_decay=args.weight_decay,
                errors=errors,
                kappa_dp=kappa,
                repetitions=args.noising_repetitions,
                rng=rng,
            )
            unlearn_grid[i, :] = unlearn_times
            count_grid[i, :] = np.isfinite(unlearn_times).astype(int)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_grid[i, :] = np.where(scratch_times > 0, unlearn_times / scratch_times, np.nan)

        # Excess vs kappa: use only the single best checkpoint (lowest retain loss) and progressively add noise
        if best_unlearn_ckpt is not None:
            _, best_state, _ = best_unlearn_ckpt
            best_model = copy.deepcopy(base_model).to(device=device, dtype=torch.float64)
            best_model.load_state_dict(best_state)
            for kappa in kappa_centers:
                loss = averaged_noised_retain_loss(
                    best_model,
                    best_scratch_model=scratch_best_model,
                    kappa_dp=float(kappa),
                    repetitions=args.noising_repetitions,
                    uloaders=uloaders,
                    device=device,
                    weight_decay=args.weight_decay,
                    rng=rng,
                )
                method_curve_points[args.method].append((float(kappa), float(loss - scratch_best_loss)))

        # Plot (second grid is Scratch)
        _plot_grids(
            ratio_grid=ratio_grid,
            unlearn_grid=unlearn_grid,
            second_grid=second_grid,
            count_grid=count_grid,
            errors=errors,
            kappas=kappas,
            level_lines=args.level_lines,
            save_prefix=args.save_prefix,
            show=args.show,
            first_label=args.method,
            second_label="Scratch",
            figure_collector=figure_collector,
        )

        # Also plot noiseless histories for the first method vs scratch
        _plot_noiseless_histories(
            unlearn_loss_history,
            scratch_trace,
            best_scratch_loss=scratch_best_loss,
            save_prefix=args.save_prefix,
            show=args.show,
            figure_collector=figure_collector,
        )

        method_curves = {
            args.method: _finalize_curve_data(method_curve_points[args.method]),
        }
        _plot_excess_vs_kappa(
            method_curves,
            save_prefix=args.save_prefix,
            show=args.show,
            kappa_scale=args.scale[1],
            figure_collector=figure_collector,
        )
    else:
        # Compare two unlearning methods (method vs method2)
        method2 = args.method2
        print(f"Comparing unlearning methods: {args.method} (A) vs {method2} (B)")
        unlearn_checkpoints_b, unlearn_loss_history_b = run_unlearning_with_checkpoints(
            method=method2,
            base_model=base_model,
            loaders=loaders,
            uloaders=uloaders,
            r_f=args.r_f,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            scheduler=args.scheduler,
            scheduler_step_freq=args.scheduler_step_freq,
            discount_factor=args.discount_factor,
            weight_decay=args.weight_decay,
            checkpoint_every=args.checkpoint_every,
            scratch_best_path=scratch_path,
            scratch_best_loss=scratch_best_loss,
            alpha=args.alpha,
            average=args.average,
            project=args.project,
        )

        best_unlearn_ckpt_a = _select_best_checkpoint(unlearn_checkpoints)
        best_unlearn_ckpt_b = _select_best_checkpoint(unlearn_checkpoints_b)

        shape = (len(kappas)-1, len(errors)-1)
        unlearn_grid = np.full(shape, np.inf, dtype=float)   # Method A
        second_grid = np.full(shape, np.inf, dtype=float)    # Method B
        count_grid = np.zeros(shape, dtype=int)
        ratio_grid = np.full(shape, np.nan, dtype=float)

        for i, kappa in tqdm(enumerate(kappas[:-1]), total=len(kappas)-1,
                             mininterval=float('inf') if IN_LOGFILE else 0.1):
            times_a, history_a = _time_to_thresholds(
                unlearn_checkpoints,
                model_template=base_model,
                best_scratch_model=scratch_best_model,
                best_scratch_loss=scratch_best_loss,
                uloaders=uloaders,
                device=device,
                weight_decay=args.weight_decay,
                errors=errors,
                kappa_dp=kappa,
                repetitions=args.noising_repetitions,
                rng=rng,
            )
            times_b, history_b = _time_to_thresholds(
                unlearn_checkpoints_b,
                model_template=base_model,
                best_scratch_model=scratch_best_model,
                best_scratch_loss=scratch_best_loss,
                uloaders=uloaders,
                device=device,
                weight_decay=args.weight_decay,
                errors=errors,
                kappa_dp=kappa,
                repetitions=args.noising_repetitions,
                rng=rng,
            )
            unlearn_grid[i, :] = times_a
            second_grid[i, :] = times_b
            count_grid[i, :] = (np.isfinite(times_a) & np.isfinite(times_b)).astype(int)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_grid[i, :] = np.where(times_b > 0, times_a / times_b, np.nan)

        # Excess vs kappa for each method: progressively noise only its single best checkpoint
        if best_unlearn_ckpt_a is not None:
            _, best_state_a, _ = best_unlearn_ckpt_a
            best_model_a = copy.deepcopy(base_model).to(device=device, dtype=torch.float64)
            best_model_a.load_state_dict(best_state_a)
            for kappa in kappa_centers:
                loss_a = averaged_noised_retain_loss(
                    best_model_a,
                    best_scratch_model=scratch_best_model,
                    kappa_dp=float(kappa),
                    repetitions=args.noising_repetitions,
                    uloaders=uloaders,
                    device=device,
                    weight_decay=args.weight_decay,
                    rng=rng,
                )
                method_curve_points[args.method].append((float(kappa), float(loss_a - scratch_best_loss)))
        if best_unlearn_ckpt_b is not None:
            _, best_state_b, _ = best_unlearn_ckpt_b
            best_model_b = copy.deepcopy(base_model).to(device=device, dtype=torch.float64)
            best_model_b.load_state_dict(best_state_b)
            for kappa in kappa_centers:
                loss_b = averaged_noised_retain_loss(
                    best_model_b,
                    best_scratch_model=scratch_best_model,
                    kappa_dp=float(kappa),
                    repetitions=args.noising_repetitions,
                    uloaders=uloaders,
                    device=device,
                    weight_decay=args.weight_decay,
                    rng=rng,
                )
                method_curve_points[method2].append((float(kappa), float(loss_b - scratch_best_loss)))

        # Plot (second grid is Method B)
        _plot_grids(
            ratio_grid=ratio_grid,
            unlearn_grid=unlearn_grid,
            second_grid=second_grid,
            count_grid=count_grid,
            errors=errors,
            kappas=kappas,
            level_lines=args.level_lines,
            save_prefix=args.save_prefix,
            show=args.show,
            first_label=args.method,
            second_label=args.method2,
            figure_collector=figure_collector,
        )

        _plot_noiseless_histories_compare(
            unlearn_loss_history,
            unlearn_loss_history_b,
            best_scratch_loss=scratch_best_loss,
            labels=(args.method, method2),
            save_prefix=args.save_prefix,
            show=args.show,
            figure_collector=figure_collector,
        )

        method_curves = {
            label: _finalize_curve_data(points)
            for label, points in method_curve_points.items()
        }
        _plot_excess_vs_kappa(
            method_curves,
            save_prefix=args.save_prefix,
            show=args.show,
            kappa_scale=args.scale[1],
            figure_collector=figure_collector,
        )

    # Final summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: All figures saved to:")
    print(f"  {figures_dir}")
    print(f"{'='*80}\n")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
