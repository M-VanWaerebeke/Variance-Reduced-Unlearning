#!/usr/bin/env python3
"""
Train a multinomial logistic regression or NN on scikit-learn's Digits dataset using PyTorch.
"""
from __future__ import annotations
import argparse
import copy
import math
import os
import random
import sys
import itertools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits, load_breast_cancer
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Detect if we're in a log file (not a terminal) to reduce tqdm output
IN_LOGFILE = not sys.stdout.isatty()

# ---------------------- Utilities ----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class Loaders:
    train: DataLoader
    full_train: DataLoader  # for full gradient computation in SVRG
    val: DataLoader
    test: DataLoader
    n_features: int
    n_classes: int
    n_train: int


def get_data_loaders(
    batch_size: int,
    val_size: float,
    test_size: float,
    seed: int,
    standardize: bool = True,
    dataset: str = "digits",
) -> Loaders:
    if dataset == "digits":
        digits = load_digits()
        X = digits.data.astype(np.float64)  # shape (n_samples, 64)
        y = digits.target.astype(np.int64)
    elif dataset == "breast_cancer":
        bc = load_breast_cancer()
        X = bc.data.astype(np.float64)   # shape (569, 30)
        y = bc.target.astype(np.int64)
    elif dataset == "mnist":
        # Download MNIST, flatten, and merge train/test for custom splitting
        # We use a temporary folder for download
        raw_train = datasets.MNIST('./data', train=True, download=True)
        raw_test = datasets.MNIST('./data', train=False, download=True)
        X = torch.cat([raw_train.data, raw_test.data]).float().numpy()
        X = X.reshape(len(X), -1).astype(np.float64) # Flatten 28x28 -> 784
        y = torch.cat([raw_train.targets, raw_test.targets]).long().numpy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_classes = len(np.unique(y))

    # Handle data splitting based on test_size and val_size
    # If sizes are 0, use all data for training
    if test_size > 0:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
    else:
        # No test split, use all data
        X_trainval, y_trainval = X, y
        X_test, y_test = np.empty((0, X.shape[1]), dtype=np.float64), np.empty(0, dtype=np.int64)

    if val_size > 0 and len(X_trainval) > 0:
        val_rel = val_size / (1.0 - test_size) if test_size > 0 else val_size
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=seed
        )
    else:
        # No val split, use all trainval as train
        X_train, y_train = X_trainval, y_trainval
        X_val, y_val = np.empty((0, X.shape[1]), dtype=np.float64), np.empty(0, dtype=np.int64)

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if len(X_val) > 0:
            X_val = scaler.transform(X_val)
        if len(X_test) > 0:
            X_test = scaler.transform(X_test)

    # Convert to tensors, using float64 for all data
    X_train_t = torch.from_numpy(X_train).to(torch.float64)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val).to(torch.float64)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test).to(torch.float64)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # Use a slightly larger batch for full gradient pass to be efficient
    full_bs = len(train_ds)
    full_train_loader = DataLoader(train_ds, batch_size=full_bs, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False) if len(test_ds) > 0 else None

    return [train_ds, val_ds, test_ds], Loaders(
        train=train_loader,
        full_train=full_train_loader,
        val=val_loader,
        test=test_loader,
        n_features=X.shape[1],
        n_classes=n_classes,
        n_train=len(train_ds),
    )


# ---------------------- Model ----------------------


class LogisticRegression(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


# ---------------------- Neural Network Model ----------------------

class NeuralNet(nn.Module):
    def __init__(self, in_features: int, n_classes: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------- CNN Model ----------------------

class SimpleCNN(nn.Module):
    """Tiny CNN for MNIST (28x28 images) - optimized for fast CPU training.

    Architecture: ~27K parameters (vs 106K before)
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        # Input: 1x28x28
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x is shape (batch, 784) - need to reshape to (batch, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# ---------------------- Model Factory ----------------------

def build_model(model_type: str, in_features: int, n_classes: int) -> nn.Module:
    model_type = model_type.upper()
    if model_type == "LR":
        return LogisticRegression(in_features, n_classes)
    elif model_type == "NN":
        if in_features == 30:
            return NeuralNet(in_features, n_classes, hidden1=32, hidden2=16)
        elif in_features == 64:
            return NeuralNet(in_features, n_classes, hidden1=64, hidden2=32)
        elif in_features == 784:
            # Smaller NN for faster training: ~109K params (vs 536K before)
            return NeuralNet(in_features, n_classes, hidden1=128, hidden2=64)
        else:
            raise ValueError("Hidden dimensions profile were not set for this input size.")
    elif model_type == "CNN":
        if in_features == 784:
            return SimpleCNN(n_classes)
        else:
            raise ValueError(f"CNN only supports 784 features (28x28 images), got {in_features}")
    else:
        raise ValueError(f"Unknown model '{model_type}' (expected 'LR', 'NN', or 'CNN').")


# ---------------------- Loss & Metrics ----------------------

def objective_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    weight_decay: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy + (lambda/2)*||w||^2 for all parameters.

    Adding L2 here ensures custom optimizers (SVRG) include regularization.
    """
    logits = model(x)
    loss = criterion(logits, y)
    if weight_decay > 0:
        l2 = torch.tensor(0.0, device=x.device)
        for p in model.parameters():
            l2 = l2 + p.pow(2).sum()
        loss = loss + 0.5 * weight_decay * l2
    return loss


def accuracy(model: nn.Module, loader: DataLoader, weight_decay: float ,device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss(reduction="mean")
    model.eval()
    n = 0
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device=device, dtype=torch.float64), yb.to(device)
            loss = objective_loss(model, xb, yb, criterion, weight_decay)
            total_loss += float(loss) * xb.size(0)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += xb.size(0)
    return total_loss / max(1, n), correct / max(1, n)


# ---------------------- SVRG Helpers ----------------------

def grad_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    weight_decay: float,
) -> List[torch.Tensor]:
    criterion = nn.CrossEntropyLoss(reduction="mean")
    x = x.to(dtype=torch.float64)
    loss = objective_loss(model, x, y, criterion, weight_decay)
    # Start by clearning existing gradients
    model.zero_grad(set_to_none=True)
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)
    return [g.detach() for g in grads]


def compute_full_gradient(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weight_decay: float,
    no_grads: bool = False,
    as_double: bool = True,
) -> List[torch.Tensor]:
    """Compute the full gradient of the objective over the entire training set.

    This averages per-sample loss correctly by weighting each batch by (B/N).
    """
    model.train()
    # Optionally evaluate in float64 to reduce cancellation/rounding error
    # We deep-copy the model so the caller's dtype/params remain unchanged.
    if as_double:
        model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
        params = list(model.parameters())
    else:
        params = list(model.parameters())
    # Initialize accumulators with zeros on device (dtype follows params)
    accum = [torch.zeros_like(p, device=device) for p in params]
    N = 0
    criterion = nn.CrossEntropyLoss(reduction="mean")
    total_loss = 0.0

    for xb, yb in loader:
        if as_double:
            xb = xb.to(device=device, dtype=torch.float64)
            yb = yb.to(device)
        else:
            xb = xb.to(device=device, dtype=torch.float64)
            yb = yb.to(device)
        B = xb.size(0)
        N += B
        if no_grads:
            with torch.no_grad():
                loss = objective_loss(model, xb, yb, criterion, weight_decay)
        else:
            loss = objective_loss(model, xb, yb, criterion, weight_decay)
            grads = torch.autograd.grad(loss, params, create_graph=False)
            for i, g in enumerate(grads):
                accum[i] = accum[i] + g.detach() * B  # accumulate as sum of grads (not mean)
        total_loss += float(loss.detach()) * B

        # Since 'loss' is a mean over the batch (plus L2), this already gives the gradient
        # of the mean. To average over full dataset, weight by (B/N_total) when finalizing.
    # Convert sums to mean over all N samples
    if N == 0:
        raise ValueError("No samples found in loader for full gradient computation.")
    if no_grads:
        return None, total_loss / N
    return [a / N for a in accum], total_loss / N


# ---------------------- Training Loops ----------------------

def train(
    model: nn.Module,
    loaders: Loaders,
    device: torch.device,
    *,
    seed: int,
    algo: str,
    optimizer: str = "sgd",
    epochs: int,
    lr: float,
    discount_factor: float,
    scheduler: str = "none",
    momentum: float,
    weight_decay: float,
    log_interval: int,
    save_folder: str = "digit_full_models",
    r_f: float = None,  # Only for path specification of scratch models
    model_type: str = "LR",
    svrg_m: int = 0,  # number of inner SVRG updates per epoch (0 = one full pass)
    svrg_double: bool = False,  # compute snapshot full gradient in high precision
    svrg_interval: int = 1, # update snapshot every N epochs
    n_train: int = None,  # training set size for filename (optional)
    epoch_callback=None,  # optional callback(epoch, model) called after each epoch's metrics
) -> dict:
    """Unified training loop supporting 'sgd', 'gd' (full-batch gradient descent), and 'svrg'.

    - For SGD, uses torch.optim.SGD (momentum & weight_decay supported).
    - For SVRG, performs per-epoch snapshot + full-gradient computation and
      variance-reduced updates: w <- w - lr * (grad_f_i(w) - grad_f_i(w~) + mu).
    - For GD (full-batch), performs one exact full-gradient step per epoch using the full training set.
      - Optimizer (when algo='sgd'): 'sgd', 'adam' (default: 'sgd'), or 'lbfgs'.
      - Scheduler: 'none', 'step', 'cyclic', or 'step+cyclic'.
        * 'cyclic' uses a triangular policy between lr*discount_factor (min) and lr (max) over a fixed cycle length (10 epochs).
        * 'step+cyclic' combines step decay of the bounds at 1/4, 1/2, 3/4 of training with the cyclic oscillation within each epoch window.
    """
    model.to(device=device, dtype=torch.float64)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    best_state = None

    hist_train_metric = []  # per-epoch average (loss for SGD, objective approx for SVRG)
    hist_val_loss = []
    hist_val_acc = []
    hist_grad_norm = []

    algo = algo.lower()
    if algo not in {"sgd", "gd", "svrg"}:
        raise ValueError(f"Unsupported algo '{algo}'. Use 'sgd', 'gd', or 'svrg'.")
    opt_name = optimizer.lower()
    if opt_name not in {"sgd", "adam", "lbfgs"}:
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Use 'sgd', 'adam', or 'lbfgs'.")

    # Optimizer only for SGD branch
    current_lr = lr  # track for step scheduler
    opt = None
    best_loss = float('inf')  # track best training loss for best model saving
    use_full_batch_for_optimizer = False
    if algo == "sgd":
        if opt_name == "sgd":
            opt = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=0  # L2 via loss
            )
        elif opt_name == "adam":
            opt = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=0  # L2 via loss
            )
        elif opt_name == "lbfgs":
            # LBFGS works best with full-batch updates
            opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=50, history_size=50, line_search_fn="strong_wolfe")
            use_full_batch_for_optimizer = True

    # Create 10 evenly spaced decay points for more gradual learning rate reduction
    decay_points = [int(epochs * (i / 10)) for i in range(1, 10)]  # at 10%, 20%, ..., 90%
    step_multiplier = 1.0
    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", leave=True,
                      mininterval=float('inf') if IN_LOGFILE else 0.1)
    for epoch in epoch_bar:
        model.train()
        n_seen = 0
        running_metric = 0.0
        use_step = scheduler in {"step", "step+cyclic"}
        use_cyclic = scheduler in {"cyclic", "step+cyclic"}

        # Apply step decay to the LR bounds at 10%, 20%, ..., 90% of training
        if use_step and epoch in decay_points:
            step_multiplier *= discount_factor

        # Determine LR for this epoch
        max_lr = lr * step_multiplier
        min_lr = lr * discount_factor * step_multiplier if use_cyclic else max_lr

        if scheduler == "cosine":
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        elif use_cyclic:
            cycle_len = epochs // 4
            phase = (epoch - 1) % cycle_len
            half = cycle_len // 2
            if phase < half or cycle_len == 1:
                frac = phase / max(1, half)
            else:
                frac = 1.0 - (phase - half) / max(1, cycle_len - half)
            current_lr = min_lr + (max_lr - min_lr) * frac
        else:
            current_lr = max_lr

        if opt is not None:
            for param_group in opt.param_groups:
                param_group["lr"] = current_lr

        if algo == "sgd":
            batch_iter = loaders.full_train if use_full_batch_for_optimizer else loaders.train
            for step, (xb, yb) in enumerate(batch_iter, start=1):
                xb, yb = xb.to(device), yb.to(device)
                assert opt is not None
                if opt_name == "lbfgs":
                    def closure():
                        opt.zero_grad(set_to_none=True)
                        loss = objective_loss(model, xb, yb, criterion, weight_decay)
                        loss.backward()
                        return loss
                    loss = opt.step(closure)
                else:
                    opt.zero_grad(set_to_none=True)
                    loss = objective_loss(model, xb, yb, criterion, weight_decay)
                    loss.backward()
                    opt.step()

                running_metric += float(loss.detach()) * xb.size(0)
                n_seen += xb.size(0)
            full_batch_grads, full_batch_loss = compute_full_gradient(model, loaders.full_train, device, weight_decay, no_grads=False)
            if full_batch_loss < best_loss:
                best_loss = full_batch_loss
                best_state = copy.deepcopy(model.state_dict())
        elif algo == "gd":
            # Full-batch gradient descent: one exact full-gradient step per epoch
            full_batch_grads, full_batch_loss = compute_full_gradient(model, loaders.full_train, device, weight_decay)
            if full_batch_loss < best_loss:
                best_loss = full_batch_loss
                best_state = copy.deepcopy(model.state_dict())
            with torch.no_grad():
                for p, g in zip(model.parameters(), full_batch_grads):
                    p.add_(-current_lr * g)
            running_metric += float(full_batch_loss) * loaders.n_train
            n_seen += loaders.n_train
        else:  # SVRG
            # Snapshot & full gradient at snapshot (update every svrg_interval epochs)
            if (epoch - 1) % svrg_interval == 0:
                snapshot = copy.deepcopy(model).to(device)
                snapshot.train()
                full_batch_grads, full_batch_loss = compute_full_gradient(
                    snapshot, loaders.full_train, device, weight_decay, as_double=svrg_double
                )
            if full_batch_loss < best_loss:
                best_loss = full_batch_loss
                best_state = copy.deepcopy(model.state_dict())

            if svrg_m and svrg_m > 0:
                # Perform exactly svrg_m variance-reduced updates per epoch
                batch_iter = itertools.cycle(loaders.train)
                for step in range(1, svrg_m + 1):
                    xb, yb = next(batch_iter)
                    xb, yb = xb.to(device), yb.to(device)

                    g_cur = grad_batch(model, xb, yb, weight_decay)
                    g_snap = grad_batch(snapshot, xb, yb, weight_decay)

                    with torch.no_grad():
                        for p, gc, gs, mu in zip(model.parameters(), g_cur, g_snap, full_batch_grads):
                            p.add_(-current_lr * (gc - gs + mu))
                    running_metric += float(objective_loss(model, xb, yb, criterion, weight_decay)) * xb.size(0)
                    n_seen += xb.size(0)
            else:
                # Default: one full pass over the training loader
                for step, (xb, yb) in enumerate(loaders.train, start=1):
                    xb, yb = xb.to(device), yb.to(device)

                    g_cur = grad_batch(model, xb, yb, weight_decay)
                    g_snap = grad_batch(snapshot, xb, yb, weight_decay)

                    with torch.no_grad():
                        for p, gc, gs, mu in zip(model.parameters(), g_cur, g_snap, full_batch_grads):
                            p.add_(-current_lr * (gc - gs + mu))
                    running_metric += float(objective_loss(model, xb, yb, criterion, weight_decay)) * xb.size(0)
                    n_seen += xb.size(0)

        # Validation metrics (if any), otherwise use train accuracy
        if loaders.val is not None:
            val_loss, val_acc = accuracy(model, loaders.val, weight_decay, device)
        else:
            # Show train accuracy when no validation set
            train_loss, train_acc = accuracy(model, loaders.train, weight_decay, device)
            val_loss, val_acc = train_loss, train_acc

        # Epoch metrics
        hist_train_metric.append(full_batch_loss)
        # Gradient norm: L2 over concatenated full-batch gradient
        grad_norm = math.sqrt(sum(g.detach().pow(2).sum().item() for g in full_batch_grads))
        hist_grad_norm.append(grad_norm)
        if epoch_callback is not None:
            epoch_callback(epoch, model)
        hist_val_loss.append(val_loss)
        hist_val_acc.append(val_acc)

        # Update progress bar with latest stats
        epoch_bar.set_postfix({
            "train": f"{full_batch_loss:.4f}",
            "val": f"{val_loss:.4f}",
            "acc": f"{val_acc*100:.2f}%",
            "|g|": f"{grad_norm:.3e}",
            "lr": f"{current_lr:.3g}",
        })

    if best_state is not None:
        model.load_state_dict(best_state)

    if loaders.test is not None:
        test_loss, test_acc = accuracy(model, loaders.test, weight_decay, device)
    else:
        test_loss, test_acc = float('nan'), float('nan')

    # Check if the trained model beats the best train loss in existing trainings
    model_type = model_type.upper()
    weight_decay_str = f"{weight_decay:.1e}" if weight_decay < 1e-3 else f"{weight_decay:.3f}"

    best_train_loss = float('inf')
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        n_str = f"_n{n_train}" if n_train is not None else ""
        if r_f is not None:
            r_f_string = f"rf{r_f:.1e}" if r_f < 1e-2 else f"rf{r_f:.3f}"
            best_model_path = (
                f"{save_folder}/best_model-seed{seed}_model{model_type}_{r_f_string}_wd{weight_decay_str}{n_str}.pt"
            )
        else:
            best_model_path = f"{save_folder}/best_model-seed{seed}_model{model_type}_wd{weight_decay_str}{n_str}.pt"
        current_train_loss, _ = accuracy(model, loaders.train, weight_decay, device)
        # Checking that accuracy and full batch loss are consistent
        _, current_train_loss_full = compute_full_gradient(model, loaders.full_train, device, weight_decay, no_grads=True)
        grad_diff = abs(current_train_loss - current_train_loss_full)
        if grad_diff > 1e-6:
            raise ValueError(f"Train loss mismatch: {current_train_loss} vs {current_train_loss_full} (diff {grad_diff})")

        try:
            best_model = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
            #future-proof loading
            try:
                sd = torch.load(best_model_path, map_location=device, weights_only=True)
            except TypeError:
                sd = torch.load(best_model_path, map_location=device)
            # Cast floating tensors to float64 for compatibility
            for k, v in list(sd.items()):
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    sd[k] = v.to(dtype=torch.float64)
            best_model.load_state_dict(sd)
            best_train_loss, _ = accuracy(best_model, loaders.train, weight_decay, device)
            if current_train_loss < best_train_loss:
                torch.save(model.state_dict(), best_model_path)
                print('\n')
                print(f"New best model saved to {best_model_path} with loss {current_train_loss} (old was {best_train_loss})")
                gradient = compute_full_gradient(model, loaders.full_train, device, weight_decay, no_grads=False)[0]
                print("Gradient norm:", np.linalg.norm(np.concatenate([g.cpu().numpy().ravel() for g in gradient])))
                print('\n')
                best_train_loss = current_train_loss
            else:
                print(f"Got a loss of {current_train_loss}, which did not beat existing best model's train loss of {best_train_loss}, with best path: {best_model_path}")
        except FileNotFoundError:
            torch.save(model.state_dict(), best_model_path)
            best_train_loss = current_train_loss
            print(f"No existing best model. Saved current model to {best_model_path}")
    else:
        best_train_loss, _ = accuracy(model, loaders.train, weight_decay, device)

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_metric": hist_train_metric,
        "val_loss": hist_val_loss,
        "val_acc": hist_val_acc,
        "best_loss_on_record": best_train_loss,
        "grad_norm": hist_grad_norm,
    }


# ---------------------- Comparison & Plotting ----------------------

def compare_algorithms(
    loaders: Loaders,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    optimizer: List[str],
    discount: float,
    scheduler: str,
    momentum: float,
    weight_decay: float,
    log_interval: int,
    seed: int,
    model_type: str = "LR",
    save_path: str | None = None,
    show: bool = True,
    svrg_m: int = 0, svrg_double: bool = False,
) -> None:
    """Train SGD, GD (full-batch), and SVRG with identical hyper-params and compare.

    Produces a side-by-side figure with (left) epoch excess loss and (right) validation loss.
    If `save_path` is provided, saves the figure there.
    If 'optimizer' is provided, it is used for the SGD run (ignored by SVRG).
    Supports scheduler options: 'none', 'step', 'cyclic', and 'step+cyclic'.
    """
    # --- Loop over optimizers: run SGD and GD for each ---
    histories: List[tuple[str, dict]] = []

    # Normalize optimizers into a list
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    optimizers = [opt.lower() for opt in optimizers]

    for opt_name in optimizers:
        # SGD run for this optimizer
        set_seed(seed)
        model_sgd = build_model(model_type, loaders.n_features, loaders.n_classes).to(device)
        h_sgd = train(
            model_sgd,
            loaders,
            device,
            algo="sgd",
            optimizer=opt_name,
            seed=seed,
            epochs=epochs,
            lr=lr,
            discount_factor=discount,
            scheduler=scheduler,
            momentum=momentum,
            weight_decay=weight_decay,
            log_interval=log_interval,
            model_type=model_type,
        )
        histories.append((f"SGD ({opt_name})", h_sgd))

        # GD run (optimizer argument ignored by GD)
        set_seed(seed)
        model_gd = build_model(model_type, loaders.n_features, loaders.n_classes).to(device)
        h_gd = train(
            model_gd,
            loaders,
            device,
            algo="gd",
            optimizer=opt_name,
            seed=seed,
            epochs=epochs,
            lr=lr,
            discount_factor=discount,
            scheduler=scheduler,
            momentum=momentum,
            weight_decay=weight_decay,
            log_interval=log_interval,
            model_type=model_type,
        )
        histories.append((f"GD ({opt_name})", h_gd))

        # SVRG run using the same lr/scheduler settings
        set_seed(seed)
        model_svrg = build_model(model_type, loaders.n_features, loaders.n_classes).to(device)
        h_svrg = train(
            model_svrg,
            loaders,
            device,
            algo="svrg",
            optimizer=opt_name,  # ignored by SVRG
            seed=seed,
            epochs=epochs,
            lr=lr,
            discount_factor=discount,
            scheduler=scheduler,
            momentum=momentum,
            weight_decay=weight_decay,
            log_interval=log_interval,
            model_type=model_type,
            svrg_m=svrg_m,
            svrg_double=svrg_double,
        )
        histories.append((f"SVRG ({opt_name})", h_svrg))

    # Global best training loss across all runs for excess-risk normalization
    best_train_loss = min(min(h[1]["train_metric"]) for h in histories)

    # --- Plot side by side ---
    epochs_axis = list(range(1, epochs + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

    # Left: Excess training loss per epoch
    ax = axes[0]
    for label, hist in histories:
        ax.plot(epochs_axis, [value - best_train_loss + 1e-7 for value in hist["train_metric"]], label=f"{label} train loss")
    ax.set_title("Excess risk per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Objective")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)

    # Right: Validation loss per epoch
    ax = axes[1]
    for label, hist in histories:
        if "val_loss" in hist:
            ax.plot(epochs_axis, hist["val_loss"], label=f"{label} val loss")
    ax.set_title("Validation loss per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)

    # Middle-right: Gradient norm per epoch
    ax = axes[2]
    for label, hist in histories:
        if "grad_norm" in hist:
            ax.plot(epochs_axis, hist["grad_norm"], label=f"{label} |g|")
    ax.set_title("Gradient norm per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||\u2207f||_2")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.suptitle(f"GD & SGD across optimizers on Digits (Model: {model_type})")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved comparison figure to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------- Main ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training a LR or NN on Digits with SGD, GD, or SVRGD (schedulers: none, step, cyclic, step+cyclic)")
    p.add_argument("--algo", type=str, default="sgd", choices=["sgd", "gd", "svrg"],
                   help="Training algorithm: sgd (standard), gd (full-batch), or svrg (SVRGD)")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN", "CNN"],
                   help="Model architecture: LR (logistic regression), NN (3-layer MLP), or CNN (ConvNet)")
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--optimizer", type=str.lower, nargs="+", default=["sgd"], choices=["sgd", "adam", "lbfgs"],
                   help="One or more optimizers to use for SGD runs (e.g., --optimizer sgd adam). Ignored by GD.")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cyclic", "step+cyclic", "cosine"],)
    p.add_argument("--discount_factor", type=float, default=0.3, help="lr discount factor (step scheduler only)")
    p.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (SGD only)")
    p.add_argument("--weight-decay", type=float, default=0.1, help="L2 regularization strength")
    p.add_argument("--val-size", type=float, default=0.0, help="Validation fraction (of whole dataset)")
    p.add_argument("--test-size", type=float, default=0.0, help="Test fraction (of whole dataset)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for model initialization")
    p.add_argument("--no-standardize", action="store_true", help="Disable feature standardization")
    p.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--log-interval", type=int, default=100, help="Steps between log prints during an epoch (0 to disable)")
    p.add_argument("--compare", action="store_true", help="Train both algorithms and plot losses side by side")
    p.add_argument("--save-plot", type=str, default="compare_losses.png", help="Where to save the comparison plot (when --compare)")
    p.add_argument("--show-plot", action="store_true", help="Display the plot window (when --compare)")
    p.add_argument("--noise-after", action="store_true", default=False,
                   help="Add privacy noise after fine-tuning instead of before")
    p.add_argument("--svrg-m", type=int, default=0,
                   help="Number of SVRG inner updates per epoch (0 = one full pass over the training loader)")
    p.add_argument("--svrg-double-fullgrad", action="store_true",
                   help="Compute the full gradient at the snapshot in higher precision for extra stability")
    p.add_argument("--dataset", type=str, default="digits", choices=["digits", "mnist", "breast_cancer"],
                   help="Dataset to use")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #else "mps" if torch.backends.mps.is_available()
    print(f"Using device: {device}")

    _, loaders = get_data_loaders(
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        standardize=not args.no_standardize,
        dataset=args.dataset,
    )

    if args.compare:
        compare_algorithms(
            loaders,
            device,
            epochs=args.epochs,
            lr=args.lr,
            optimizer=args.optimizer,
            discount=args.discount_factor,
            scheduler=args.scheduler,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            log_interval=args.log_interval,
            seed=args.seed,
            model_type=args.model,
            save_path=args.save_plot,
            show=args.show_plot,
            svrg_m=args.svrg_m, svrg_double=args.svrg_double_fullgrad,
        )
    else:
        model = build_model(args.model, loaders.n_features, loaders.n_classes)
        opt_single = args.optimizer[0] if isinstance(args.optimizer, list) else args.optimizer
        print(f"Model: {args.model} | Algorithm: {args.algo.upper()} | Optimizer: {opt_single.upper()}")
        folder_dataset = "digit" if args.dataset == "digits" else args.dataset
        train(
            model,
            loaders,
            device,
            algo=args.algo,
            optimizer=opt_single,
            epochs=args.epochs,
            lr=args.lr,
            discount_factor=args.discount_factor,
            scheduler=args.scheduler,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            log_interval=args.log_interval,
            model_type=args.model,
            seed=args.seed,
            svrg_m=args.svrg_m, svrg_double=args.svrg_double_fullgrad,
            save_folder=f"{folder_dataset}_full_models",
        )


if __name__ == "__main__":
    main()
