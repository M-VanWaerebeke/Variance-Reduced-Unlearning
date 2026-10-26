#!/usr/bin/env python3
"""
Standalone script to train a single full or scratch model.
Used for parallelized pre-training before running risk_vs_privacy.py.

Usage:
    # Train a full model (trained on entire training set)
    python pretrain_model.py --type full --seed 0 --weight_decay 0.1 --model NN --dataset mnist

    # Train a half-trained full model (50% data, matching shadow models for MIA)
    python pretrain_model.py --type full --seed 0 --weight_decay 0.1 --model NN --dataset mnist --half_train

    # Train a scratch model (trained on retain set only)
    python pretrain_model.py --type scratch --seed 0 --weight_decay 0.1 --model NN --dataset mnist --r_f 0.03

    # Train a half-trained scratch model (50% data, matching shadow models for MIA)
    python pretrain_model.py --type scratch --seed 0 --weight_decay 0.1 --model NN --dataset mnist --r_f 0.03 --half_train
"""
import argparse
import os
import sys
import torch

torch.set_default_dtype(torch.float64)

from digit_train import (
    set_seed,
    get_data_loaders,
    build_model,
    train as train_model,
    Loaders,
)
from digit_forget import (
    get_retain_forget_loaders,
    _best_model_path,
)

# HP grid matching test_vru strategy (4 configs total) — used for digits
# Each entry: (lr_raw, optimizer, algo, scheduler)
PIPELINE_HP_CONFIGS = [
    (0.01,  "adam",  "sgd", "step"),
    (1.0,   "lbfgs", "sgd", "none"),
    (0.3,   "sgd",   "gd",  "step"),
    (0.1,   "sgd",   "sgd", "cosine"),
]
PIPELINE_DISCOUNT = 0.5

# MNIST HP grid — matches test_vru_retain_loss.py HP_CONFIGS exactly
MNIST_HP_CONFIGS = [
    (0.01,  "adam",  "sgd", "step"),
    (1.0,   "lbfgs", "sgd", "none"),
    (3.0,   "sgd",   "gd",  "step"),
    (1.0,   "sgd",   "gd",  "step"),
    (0.3,   "sgd",   "gd",  "step"),
    (0.3,   "sgd",   "sgd", "cosine"),
    (0.1,   "sgd",   "sgd", "cosine"),
    (0.03,  "sgd",   "sgd", "cosine"),
]
MNIST_DISCOUNT = 0.5
MNIST_EPOCHS = 300


def parse_args():
    p = argparse.ArgumentParser(description="Train a single full or scratch model")
    p.add_argument("--type", type=str, required=True, choices=["full", "scratch"],
                   help="Model type: 'full' (entire train set) or 'scratch' (retain only)")
    p.add_argument("--seed", type=int, required=True, help="Random seed")
    p.add_argument("--weight_decay", type=float, default=0.1, help="L2 regularization")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN"],
                   help="Model architecture: LR or NN")
    p.add_argument("--dataset", type=str, default="mnist", choices=["digits", "mnist", "breast_cancer"],
                   help="Dataset to use")
    p.add_argument("--r_f", type=float, default=None,
                   help="Forget ratio (required for scratch models)")
    p.add_argument("--half_train", action="store_true",
                   help="Train on 50%% of data (to match shadow model training for MIA)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    p.add_argument("--device", type=str, default=None,
                   help="Device to use (e.g., 'cpu', 'cuda', 'mps')")
    p.add_argument("--force", action="store_true",
                   help="Force retraining even if model exists")
    return p.parse_args()


def get_model_path(args) -> str:
    """Get the expected model path based on arguments."""
    model_type = args.model.upper()
    wd_str = f"{args.weight_decay:.1e}" if args.weight_decay < 1e-3 else f"{args.weight_decay:.3f}"
    folder_dataset = "digit" if args.dataset == "digits" else args.dataset
    suffix = "_half" if args.half_train else ""

    if args.type == "full":
        return f"{folder_dataset}_full_models{suffix}/best_model-seed{args.seed}_model{model_type}_wd{wd_str}.pt"
    else:
        # scratch model
        rf_str = f"{args.r_f:.1e}" if args.r_f < 1e-2 else f"{args.r_f:.3f}"
        return f"{folder_dataset}_scratch_models{suffix}/best_model-seed{args.seed}_model{model_type}_rf{rf_str}_wd{wd_str}.pt"


def train_full_model(args, loaders, device):
    """Train a model on the full training set."""
    model_type = args.model.upper()
    folder_dataset = "digit" if args.dataset == "digits" else args.dataset
    suffix = "_half" if args.half_train else ""
    save_folder = f"{folder_dataset}_full_models{suffix}"

    if args.dataset == "mnist":
        # Grid sweep matching test_vru_retain_loss.py
        for lr_raw, optimizer, algo, scheduler in MNIST_HP_CONFIGS:
            lr_eff = lr_raw / 10 if model_type == "NN" else lr_raw
            base = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
            _ = train_model(
                base,
                loaders,
                device,
                algo=algo,
                seed=args.seed,
                epochs=MNIST_EPOCHS,
                lr=lr_eff,
                scheduler=scheduler,
                optimizer=optimizer,
                discount_factor=MNIST_DISCOUNT,
                momentum=0.0,
                weight_decay=args.weight_decay,
                log_interval=0,
                save_folder=save_folder,
                model_type=model_type,
            )
    else:
        # Digits dataset: iterate over HP grid (L-BFGS, GD, cosine, Adam)
        for lr_raw, optimizer, algo, scheduler in PIPELINE_HP_CONFIGS:
            lr_eff = lr_raw / 10 if model_type == "NN" else lr_raw
            base = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
            _ = train_model(
                base,
                loaders,
                device,
                algo=algo,
                seed=args.seed,
                epochs=1000,
                lr=lr_eff,
                scheduler=scheduler,
                optimizer=optimizer,
                discount_factor=PIPELINE_DISCOUNT,
                momentum=0.0,
                weight_decay=args.weight_decay,
                log_interval=0,
                save_folder=save_folder,
                model_type=model_type,
            )
    print("Training complete.")


def train_scratch_model(args, loaders, device):
    """Train a model on the retain set only."""
    model_type = args.model.upper()
    folder_dataset = "digit" if args.dataset == "digits" else args.dataset
    suffix = "_half" if args.half_train else ""
    save_folder = f"{folder_dataset}_scratch_models{suffix}"

    # Get retain/forget split
    uloaders = get_retain_forget_loaders(
        loaders.train.dataset,
        args.r_f,
        args.seed,
        args.batch_size,
    )

    # Build scratch loaders using only retain data
    scratch_loader = Loaders(
        train=uloaders.retain,
        full_train=uloaders.retain_full,
        val=loaders.val,
        test=loaders.test,
        n_features=loaders.n_features,
        n_classes=loaders.n_classes,
        n_train=uloaders.n_retain,
    )

    if args.dataset == "mnist":
        # Grid sweep matching test_vru_retain_loss.py
        for lr_raw, optimizer, algo, scheduler in MNIST_HP_CONFIGS:
            lr_eff = lr_raw / 10 if model_type == "NN" else lr_raw
            base = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
            _ = train_model(
                base,
                scratch_loader,
                device,
                algo=algo,
                seed=args.seed,
                epochs=MNIST_EPOCHS,
                lr=lr_eff,
                scheduler=scheduler,
                optimizer=optimizer,
                discount_factor=MNIST_DISCOUNT,
                momentum=0.0,
                weight_decay=args.weight_decay,
                log_interval=0,
                save_folder=save_folder,
                r_f=args.r_f,
                model_type=model_type,
            )
    else:
        # Digits dataset: iterate over HP grid (L-BFGS, GD, cosine, Adam)
        for lr_raw, optimizer, algo, scheduler in PIPELINE_HP_CONFIGS:
            lr_eff = lr_raw if (model_type == "LR" and args.dataset in ("digits", "breast_cancer")) else lr_raw / 10
            base = build_model(model_type, loaders.n_features, loaders.n_classes).to(device=device, dtype=torch.float64)
            _ = train_model(
                base,
                scratch_loader,
                device,
                algo=algo,
                seed=args.seed,
                epochs=3001,
                lr=lr_eff,
                scheduler=scheduler,
                optimizer=optimizer,
                discount_factor=PIPELINE_DISCOUNT,
                momentum=0.0,
                weight_decay=args.weight_decay,
                log_interval=0,
                save_folder=save_folder,
                r_f=args.r_f,
                model_type=model_type,
            )
    print("Training complete.")


def main():
    args = parse_args()

    # Validate arguments
    if args.type == "scratch" and args.r_f is None:
        print("Error: --r_f is required for scratch models", file=sys.stderr)
        sys.exit(1)

    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Check if model already exists
    model_path = get_model_path(args)
    if os.path.exists(model_path) and not args.force:
        print(f"Model already exists: {model_path}")
        print("Use --force to retrain.")
        sys.exit(0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Training {args.type} model:")
    print(f"  seed={args.seed}, weight_decay={args.weight_decay}, model={args.model}")
    print(f"  dataset={args.dataset}, device={device}")
    if args.type == "scratch":
        print(f"  r_f={args.r_f}")
    if args.half_train:
        print(f"  half_train=True (training on 50% of data to match shadow models)")
    print(f"  Output: {model_path}")

    # Set seed
    set_seed(args.seed)

    # Load data
    _, loaders = get_data_loaders(
        batch_size=args.batch_size,
        val_size=0.0,
        test_size=0.0,
        seed=args.seed,
        standardize=True,
        dataset=args.dataset,
    )

    # If half_train, split dataset to 50% for training
    if args.half_train:
        from torch.utils.data import DataLoader, random_split

        train_ds = loaders.train.dataset
        n_full = len(train_ds)
        n_half = n_full // 2

        # Use a deterministic split based on seed
        generator = torch.Generator().manual_seed(args.seed + 9999)
        train_ds_half, _ = random_split(train_ds, [n_half, n_full - n_half], generator=generator)

        print(f"  Dataset split: using {n_half}/{n_full} samples (50%) for training")

        # Rebuild loaders with half the data
        loaders = Loaders(
            train=DataLoader(train_ds_half, batch_size=args.batch_size, shuffle=True),
            full_train=DataLoader(train_ds_half, batch_size=len(train_ds_half), shuffle=False),
            val=loaders.val,
            test=loaders.test,
            n_features=loaders.n_features,
            n_classes=loaders.n_classes,
            n_train=n_half,
        )

    # Train
    if args.type == "full":
        train_full_model(args, loaders, device)
    else:
        train_scratch_model(args, loaders, device)

    print(f"Done! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
