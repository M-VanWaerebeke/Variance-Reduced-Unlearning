#!/usr/bin/env python3
"""
risk_vs_rf.py — Excess risk vs. Forget Set Size (r_f) for fixed compute budget.
"""
import argparse
import copy
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import os
import pickle
import hashlib

# Ensure float64
torch.set_default_dtype(torch.float64)

# Detect if we're in a log file (not a terminal) to reduce tqdm output
IN_LOGFILE = not sys.stdout.isatty()

from fast_grid import (
    run_unlearning_with_checkpoints,
    averaged_noised_retain_loss,
    apply_gaussian_noise_to_model,
    _evaluate_loader_loss,
    _evaluate_retain_loss,
    _param_vector,
    _load_full_base,
)
from gradient_representation import (
    set_seed,
    get_data_loaders,
    get_retain_forget_loaders,
    UnlearnLoaders,
    build_model,
    _ensure_pretrained_full_model,
    _ensure_scratch_model,
    ensure_pretrained_models_parallel,
    ensure_scratch_models_parallel,
)
from digit_forget import parse_vru_variant

# A cache entry is valid only for this test-set metric and output-noise protocol.
CACHE_SCHEMA = "test_objective_fixed_lr_empirical_v2"
OUTPUT_NOISE_REPETITIONS = 10

def parse_method_with_kappa(method_str: str) -> tuple[str, float]:
    """Parse method string with optional kappa_dp value.

    Args:
        method_str: Method string in format "method" or "method:kappa"
                   Examples: "vru", "vru:0.1", "vru-avg-proj:2.0"

    Returns:
        Tuple of (method_name, kappa_dp)

    Raises:
        ValueError: If kappa_dp format is invalid
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
        return method_str, 0.0


def _effective_kappa(method_str: str, noise_timing: str) -> float:
    """Validate and return the noise scale encoded in a method name."""
    if noise_timing not in {"before", "after", "empirical"}:
        return 0.0
    _, kappa = parse_method_with_kappa(method_str)
    if ":" not in method_str or (noise_timing == "empirical" and
                              (not np.isfinite(kappa) or kappa <= 0)):
        raise ValueError(
            f"Method {method_str!r} with timing={noise_timing!r} requires "
            "METHOD:KAPPA; empirical KAPPA must be positive and finite."
        )
    return kappa


def _averaged_empirical_noised_loss(
    model: torch.nn.Module,
    *,
    std: float,
    repetitions: int,
    eval_loader,
    device: torch.device,
    weight_decay: float,
    rng: np.random.Generator,
) -> float:
    """Average test or retain objective over independently noised outputs."""
    losses = []
    for _ in range(repetitions):
        noised_model = copy.deepcopy(model).to(device=device, dtype=torch.float64)
        apply_gaussian_noise_to_model(noised_model, std, rng, device)
        losses.append(_evaluate_loader_loss(noised_model, eval_loader, device, weight_decay))
    return float(np.mean(losses)) if losses else float("inf")

def _get_pretty_name(method_name: str) -> str:
    m = method_name.lower()
    if m == 'nft':
        return 'NFT'
    if m == 'vru':
        return 'VRU'
    if m == 'sgd':
        return 'SGD'
    if m == 'gd':
        return 'GD'
    if m == 'svrg':
        return 'SVRG'
    if m == 'neggrad+':
        return 'NegGrad+'
    if m == 'scrub':
        return 'SCRUB'
    return method_name.upper()

def parse_args():
    p = argparse.ArgumentParser(description="Excess Risk vs r_f grid")
    p.add_argument("--min_rf", type=float, default=1e-3)
    p.add_argument("--max_rf", type=float, default=5e-1)
    p.add_argument("--n_rf", type=int, default=5)
    p.add_argument("--T", type=float, default=10, help="Epoch budget")
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds (runs 0 to seeds-1)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--dataset", type=str, default="digits")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--save_file", type=str, default="rf_grid_plot.pdf")
    p.add_argument("--cache_file", type=str, default="rf_grid_cache.pkl", help="Path to cache file")
    p.add_argument("--model", type=str.upper, default="LR", choices=["LR", "NN"],
                   help="Model architecture: LR or NN")

    # Methods: list of (name, lr, decay, timing)
    p.add_argument("--run", nargs=4, action="append", metavar=("METHOD[:KAPPA]", "LR", "DECAY", "TIMING"),
                   help="Add method with optional kappa_dp. Examples: --run vru:0.1 0.01 0.9 after, --run vru:2.0 1 0.6 empirical")
    p.add_argument("--original_style", action="store_true",
                   help="Use original matplotlib styling instead of improved styling")
    p.add_argument("--average", action="store_true",
                   help="Use Polyak averaging for VRU")
    p.add_argument("--project", action="store_true",
                   help="Project VRU updates onto ball")

    return p.parse_args()


def get_file_hash(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    args = parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}")
    print(f"Model Architecture: {args.model}")

    # Default two-method subset of the Figure 1 Table 1 settings.
    if args.run is None:
        runs = [
            ("vru:1.0", "1.1", "0.55", "after"),
            ("nft:1.0", "0.3", "0.8", "before"),
        ]
    else:
        runs = args.run

    # 3. Grid of r_f
    r_f_values = np.logspace(np.log10(args.min_rf), np.log10(args.max_rf), args.n_rf)

    # Store results: [method, r_f, seed]
    all_run_results = np.zeros((len(runs), args.n_rf, args.seeds))

    # Indices to plot history for (first, middle, last)
    indices_to_plot = sorted(list(set([0, args.n_rf // 2, args.n_rf - 1])))
    history_results = {idx: {} for idx in indices_to_plot}
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

    # Pre-train all full and scratch models in parallel across seeds
    seeds = list(range(args.seeds))
    print(f"\n=== Pre-training full models ({len(seeds)} seeds, parallel) ===")
    ensure_pretrained_models_parallel(
        seeds, args.weight_decay, args.model, args.dataset,
        batch_size=args.batch_size,
    )
    seed_rf_pairs = [(s, rf) for s in seeds for rf in r_f_values]
    print(f"\n=== Pre-training scratch models ({len(seed_rf_pairs)} tasks, parallel) ===")
    ensure_scratch_models_parallel(
        seed_rf_pairs, args.weight_decay, args.model, args.dataset,
        batch_size=args.batch_size,
    )

    for seed_val in range(args.seeds):
        print(f"\n=== Processing Seed {seed_val}/{args.seeds - 1} ===")
        set_seed(seed_val)

        # 1. Data Loaders (Full)
        train_val_test_sets, loaders = get_data_loaders(
            batch_size=args.batch_size,
            val_size=0.2,
            test_size=0.3,
            seed=seed_val,
            standardize=True,
            dataset=args.dataset,
        )
        train_ds, _, _ = train_val_test_sets

        # 2. Ensure Full Model (independent of r_f)
        full_path = _ensure_pretrained_full_model(
            seed=seed_val,
            weight_decay=args.weight_decay,
            model_type=args.model,
            loaders=loaders,
            device=device,
            dataset=args.dataset,
        )
        base_model_template = _load_full_base(full_path, loaders, device, model_type=args.model)
        full_hash = get_file_hash(full_path)

        for i, r_f in enumerate(tqdm(r_f_values, desc=f"Seed {seed_val} r_f loop",
                                      mininterval=float('inf') if IN_LOGFILE else 0.1)):
            print(f"\n=== Processing r_f = {r_f:.4e} (Seed {seed_val}) ===")

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
            )
            scratch_hash = get_file_hash(scratch_path)

            # Load scratch model and compute loss (Moved up)
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
            # Apply the same CE+L2 objective to test data; App. C.1 calls this
            # "test loss" without specifying whether its L2 term is included.
            scratch_test_loss = _evaluate_loader_loss(scratch_model, loaders.test, device, args.weight_decay)

            # Training history remains on the retain objective.
            if seed_val == args.seeds - 1 and i in indices_to_plot:
                scratch_losses_for_plot[i] = scratch_retain_loss

            # Check which methods need running
            methods_to_run = []
            for idx, (m_name_raw, m_lr, m_decay, m_time) in enumerate(runs):
                # Parse method name and kappa
                m_name, _ = parse_method_with_kappa(m_name_raw)

                eff_lr = float(m_lr)

                # Determine effective noise timing for kappa calculation
                # For neggrad+/scrub, timing encodes alpha, so noise_timing is "none"
                base_method, _, _ = parse_vru_variant(m_name.lower())
                m_noise_timing = m_time
                if base_method in ["neggrad+", "scrub"]:
                    m_noise_timing = "none"

                eff_kappa = _effective_kappa(m_name_raw, m_noise_timing)

                # Parse method variant to get actual use_average and use_project values
                _, m_use_average, m_use_project = parse_vru_variant(m_name)
                if args.average:
                    m_use_average = True
                if args.project:
                    m_use_project = True

                # Include global args and model hashes in key to ensure validity
                key = (CACHE_SCHEMA, m_name, eff_lr, m_decay, m_time, seed_val, r_f,
                       args.weight_decay, args.dataset, args.batch_size, eff_kappa, args.T, args.model,
                       scratch_hash, full_hash, m_use_average, m_use_project)

                if key not in cache:
                    methods_to_run.append(idx)
                elif 'test_loss' not in cache[key]:
                    methods_to_run.append(idx)
                elif seed_val == args.seeds - 1 and i in indices_to_plot:
                    # If we need history for plotting but don't have it in cache
                    if cache[key].get('history') is None:
                        methods_to_run.append(idx)

            if not methods_to_run:
                print("  All methods cached. Skipping computation.")
                for idx, (m_name_raw, m_lr, m_decay, m_time) in enumerate(runs):
                    # Parse method name and kappa
                    m_name, _ = parse_method_with_kappa(m_name_raw)

                    eff_lr = float(m_lr)

                    base_method, _, _ = parse_vru_variant(m_name.lower())
                    m_noise_timing = m_time
                    if base_method in ["neggrad+", "scrub"]:
                        m_noise_timing = "none"

                    eff_kappa = _effective_kappa(m_name_raw, m_noise_timing)

                    # Parse method variant to get actual use_average and use_project values
                    _, m_use_average, m_use_project = parse_vru_variant(m_name)
                    if args.average:
                        m_use_average = True
                    if args.project:
                        m_use_project = True

                    key = (CACHE_SCHEMA, m_name, eff_lr, m_decay, m_time, seed_val, r_f,
                           args.weight_decay, args.dataset, args.batch_size, eff_kappa, args.T, args.model,
                           scratch_hash, full_hash, m_use_average, m_use_project)
                    entry = cache[key]
                    all_run_results[idx, i, seed_val] = entry['test_loss'] - scratch_test_loss

                    if seed_val == args.seeds - 1 and i in indices_to_plot:
                        hist = entry.get('history')
                        if hist is not None:
                            label = _get_pretty_name(m_name)
                            history_results[i][label] = hist
                            if entry.get('final_retain_loss') is not None:
                                history_results[i][f"{label}::final"] = entry['final_retain_loss']
                continue

            # c. Run Methods
            for idx, (method_name_raw, lr_str, decay_str, timing) in enumerate(runs):
                # Parse method name and kappa
                method_name, _ = parse_method_with_kappa(method_name_raw)

                lr = float(lr_str)
                decay = float(decay_str)

                # Parse VRU variant (e.g., "vru-avg" -> ("vru", True, False))
                base_method, use_average, use_project = parse_vru_variant(method_name)

                # Global flags override variant-based settings
                if args.average:
                    use_average = True
                if args.project:
                    use_project = True

                # Determine if retraining
                retrain_methods = {"gd", "sgd", "svrg"}
                is_retrain = base_method in retrain_methods

                # For neggrad+ and scrub, timing is alpha, and noise is not used.
                alpha_val = 0.05  # Default value from fast_grid.py
                noise_timing = timing
                if base_method in ["neggrad+", "scrub"]:
                    try:
                        alpha_val = float(timing)
                        noise_timing = "none"
                    except ValueError:
                        raise ValueError(
                            f"For '{method_name}', the 'timing' parameter must be a float for alpha, but got '{timing}'"
                        )

                eff_kappa = _effective_kappa(method_name_raw, noise_timing)

                key = (CACHE_SCHEMA, method_name, lr, decay_str, timing, seed_val, r_f,
                       args.weight_decay, args.dataset, args.batch_size, eff_kappa, args.T, args.model,
                       scratch_hash, full_hash, use_average, use_project)

                # Check if we can skip this specific method
                if idx not in methods_to_run:
                    entry = cache[key]
                    all_run_results[idx, i, seed_val] = entry['test_loss'] - scratch_test_loss
                    if seed_val == args.seeds - 1 and i in indices_to_plot:
                        hist = entry.get('history')
                        if hist is not None:
                            label = _get_pretty_name(method_name)
                            history_results[i][label] = hist
                            if entry.get('final_retain_loss') is not None:
                                history_results[i][f"{label}::final"] = entry['final_retain_loss']
                    print(f"  Loaded {method_name} ({timing}) from cache.")
                    continue

                # Isolate the randomness of this method run from previous ones.
                # This ensures that changing parameters of Method A doesn't alter the training trajectory of Method B.
                set_seed(seed_val * 1000 + idx)

                if is_retrain and noise_timing != "none":
                    raise ValueError(f"Retraining method '{method_name}' must be run with timing='none', got '{noise_timing}'.")

                current_uloaders = uloaders
                run_method_name = base_method  # Use base method (without variant suffix)

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

                # Calculate Epochs
                if base_method == "gd":
                     cost_per_epoch = uloaders.n_retain
                else:
                     batches_per_epoch = len(uloaders.retain)
                     cost_per_epoch = batches_per_epoch * args.batch_size

                if cost_per_epoch == 0:
                    print("  Warning: Cost per epoch is 0. Skipping.")
                    all_run_results[idx, i, seed_val] = np.nan
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

                print(f"  Running {method_name} ({timing}): {epochs} epochs (Cost/Ep={cost_per_epoch}, G/S={grads_per_sample})")

                # Prepare Model
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
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.dtype.is_floating_point:
                                noise = torch.from_numpy(rng.normal(loc=0.0, scale=std_init, size=p.shape)).to(p.device, dtype=torch.float64)
                                p.add_(noise)

                # Run Unlearning
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
                    scratch_best_path=scratch_path,
                    scratch_best_loss=scratch_retain_loss,
                    alpha=alpha_val,
                    average=use_average,  # Use parsed variant settings
                    project=use_project,  # Use parsed variant settings
                )

                # Store history if this is one of the r_f values to plot (only for last seed)
                # Store raw loss values - excess risk is computed at plot time using final scratch_loss
                raw_hist = None
                if seed_val == args.seeds - 1 and i in indices_to_plot and history is not None:
                    # Handle potential dict return or list
                    vals = history
                    if isinstance(history, dict):
                        vals = history.get('retain_loss', history.get('loss', []))

                    # Convert to float (keep raw loss, not excess risk)
                    try:
                        # Handle list of tuples [(step, loss), ...] from fast_grid
                        if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], tuple):
                            clean_vals = [float(v[1]) for v in vals]
                        else:
                            clean_vals = [v.item() if torch.is_tensor(v) else v for v in vals]
                        raw_hist = clean_vals  # Store raw loss, convert to excess at plot time
                        label = _get_pretty_name(method_name)
                        history_results[i][label] = raw_hist
                    except Exception as e:
                        print(f"  Warning: Could not process history for {method_name}: {e}")

                # Get final checkpoint
                if not checkpoints:
                    final_state = model.state_dict()
                else:
                    final_state = checkpoints[-1][1]

                model.load_state_dict(final_state)

                # Save checkpoint
                ckpt_dir = "checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                run_id = hashlib.md5(str(key).encode()).hexdigest()
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_{run_id}.pt")
                torch.save(final_state, ckpt_path)

                # Final risk uses the held-out test objective. The saved history
                # above is still the retain objective used during training.
                final_retain_loss = None
                if noise_timing == "after":
                    loss = averaged_noised_retain_loss(
                        model,
                        best_scratch_model=scratch_model,
                        kappa_dp=eff_kappa,
                        repetitions=OUTPUT_NOISE_REPETITIONS,
                        uloaders=uloaders,
                        device=device,
                        weight_decay=args.weight_decay,
                        rng=np.random.default_rng(seed_val + i * 10000),
                        eval_loader=loaders.test,
                    )
                    if seed_val == args.seeds - 1 and i in indices_to_plot:
                        # Reuse the same noise draws for the retain diagnostic.
                        final_retain_loss = averaged_noised_retain_loss(
                            model,
                            best_scratch_model=scratch_model,
                            kappa_dp=eff_kappa,
                            repetitions=OUTPUT_NOISE_REPETITIONS,
                            uloaders=uloaders,
                            device=device,
                            weight_decay=args.weight_decay,
                            rng=np.random.default_rng(seed_val + i * 10000),
                        )
                elif noise_timing == "empirical":
                    if not 0.0 < r_f < 1.0:
                        raise ValueError("Empirical output noise requires 0 < r_f < 1")
                    std_empirical = eff_kappa * r_f / (1.0 - r_f)
                    loss = _averaged_empirical_noised_loss(
                        model,
                        std=std_empirical,
                        repetitions=OUTPUT_NOISE_REPETITIONS,
                        eval_loader=loaders.test,
                        device=device,
                        weight_decay=args.weight_decay,
                        rng=np.random.default_rng(seed_val + i * 10000),
                    )
                    if seed_val == args.seeds - 1 and i in indices_to_plot:
                        final_retain_loss = _averaged_empirical_noised_loss(
                            model,
                            std=std_empirical,
                            repetitions=OUTPUT_NOISE_REPETITIONS,
                            eval_loader=uloaders.retain_full,
                            device=device,
                            weight_decay=args.weight_decay,
                            rng=np.random.default_rng(seed_val + i * 10000),
                        )
                else:
                    loss = _evaluate_loader_loss(model, loaders.test, device, args.weight_decay)
                    if seed_val == args.seeds - 1 and i in indices_to_plot:
                        final_retain_loss = _evaluate_retain_loss(model, uloaders, device, args.weight_decay)

                excess_risk = loss - scratch_test_loss
                all_run_results[idx, i, seed_val] = excess_risk
                print(f"    Test Excess Risk: {excess_risk:.4e}")

                # Update history with final loss if needed (to show noise jump) - only last seed
                # Store raw loss - excess risk is computed at plot time
                if seed_val == args.seeds - 1 and i in indices_to_plot:
                    label = _get_pretty_name(method_name)
                    if label in history_results[i] and final_retain_loss is not None:
                        history_results[i][f"{label}::final"] = final_retain_loss

                # Update Cache
                if key not in cache:
                    cache[key] = {}
                cache[key].update({
                    'excess_risk': excess_risk,
                    'test_loss': np.float64(loss),
                    'final_retain_loss': final_retain_loss,
                    'history': raw_hist,
                    'ckpt_path': ckpt_path
                })

            # Save cache periodically
            with open(args.cache_file, "wb") as f:
                pickle.dump(cache, f)

            # Diagnostic: Print best method for this r_f/seed
            current_risks = all_run_results[:, i, seed_val]
            valid_indices = np.where(~np.isnan(current_risks))[0]
            if len(valid_indices) > 0:
                best_local_idx = np.argmin(current_risks[valid_indices])
                best_idx = valid_indices[best_local_idx]
                best_val = current_risks[best_idx]
                b_name, _, _, b_time = runs[best_idx]
                if best_val < 0:
                    print(f"  Negative test excess risk (Seed {seed_val}, r_f={r_f:.4e}): {b_name} ({b_time}) = {best_val:.4e}")


    # 4. Plotting
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
        # Modern, publication-quality styling (use LaTeX if available)
        import shutil
        _has_latex = shutil.which("latex") is not None
        plt.rcParams.update({
            'text.usetex': _has_latex,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'] if _has_latex else ['DejaVu Serif'],
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

    # A test-set difference can legitimately be zero or negative. Keep the
    # geometric/log presentation when possible, and show signed values on a
    # linear axis rather than silently dropping them.
    finite_risks = all_run_results[np.isfinite(all_run_results)]
    signed_risks = bool(finite_risks.size and np.any(finite_risks <= 0))
    if signed_risks:
        plot_means = np.nanmean(all_run_results, axis=2)
        plot_stds = np.nanstd(all_run_results, axis=2)
    else:
        log_data = np.log(all_run_results + 1e-15)
        plot_means = np.exp(np.nanmean(log_data, axis=2))
        plot_stds = np.exp(np.nanstd(log_data, axis=2))

    # Use a distinct colormap
    if args.original_style:
        if len(runs) <= 10:
            colors = [plt.cm.tab10(i) for i in range(len(runs))]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(runs)))
        linestyles = ['-'] * 20
    else:
        # Vibrant modern palette - orange colors made more distinct
        colors = ['#FF6B35', '#004E89', '#D4A017', '#06A77D', '#9B59B6',
                  '#E74C3C', '#3498DB', '#2ECC71', '#C70039', '#1ABC9C']
        # Line styles for B&W printing: solid, dashed, dotted, dashdot, etc.
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2)),
                     (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]

    unlearn_handles = []
    unlearn_labels = []
    unlearn_colors = []
    unlearn_linestyles = []
    retrain_handles = []
    retrain_labels = []
    retrain_colors = []
    retrain_linestyles = []
    retrain_methods = {"gd", "sgd", "svrg"}

    for idx, (method_name, lr, decay, timing) in enumerate(runs):
        label = _get_pretty_name(method_name)
        y = plot_means[idx]
        if signed_risks:
            yerr_lower = plot_stds[idx]
            yerr_upper = plot_stds[idx]
        else:
            factor = plot_stds[idx]
            yerr_lower = y - (y / factor)
            yerr_upper = (y * factor) - y

        if args.original_style:
            color = colors[idx]
            linestyle = '-'
            container = ax.errorbar(
                r_f_values, y, yerr=[yerr_lower, yerr_upper],
                marker='o', label=label, capsize=6, elinewidth=2,
                color=color, alpha=0.9, linestyle=linestyle
            )
        else:
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            container = ax.errorbar(
                r_f_values, y, yerr=[yerr_lower, yerr_upper],
                marker='o', label=label, capsize=4, elinewidth=1.5,
                color=color, alpha=1.0, markersize=12, markeredgewidth=1.5,
                markeredgecolor='white', linewidth=2.0, linestyle=linestyle
            )

        if method_name in retrain_methods:
            retrain_handles.append(container)
            retrain_labels.append(label)
            retrain_colors.append(color)
            retrain_linestyles.append(linestyle)
        else:
            unlearn_handles.append(container)
            unlearn_labels.append(label)
            unlearn_colors.append(color)
            unlearn_linestyles.append(linestyle)

    ax.set_xscale('log')
    if not signed_risks:
        ax.set_yscale('log')
    ax.set_xlabel(r"Forget Set Relative Size ($r_f$)")
    ax.set_ylabel("Test Excess Risk")

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

    # Legend construction - create composite handles showing both line and marker
    legend_handles = []
    legend_labels = []

    if unlearn_handles:
        legend_handles.append(Line2D([0], [0], color='none', marker='', linestyle=''))
        legend_labels.append(r"$\mathit{Unlearning}$")
        # Create composite handles: line + marker for each unlearning method
        for i, (handle, label) in enumerate(zip(unlearn_handles, unlearn_labels)):
            line_handle = Line2D([0], [0], color=unlearn_colors[i], linestyle=unlearn_linestyles[i],
                               linewidth=2.0, marker='o', markersize=12, markerfacecolor=unlearn_colors[i],
                               markeredgecolor='white', markeredgewidth=1.5)
            legend_handles.append(line_handle)
            legend_labels.append("  " + label)

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
                       'facecolor': 'white', 'fontsize': 'small'}
        title_props = {'weight': 'semibold', 'size': 'small'}
    else:
        legend_style = {'frameon': True, 'framealpha': 1.0, 'edgecolor': '#666666',
                       'facecolor': 'white', 'fontsize': 13, 'borderpad': 0.8,
                       'handletextpad': 0.6, 'labelspacing': 0.7, 'handlelength': 3}
        title_props = {'weight': 'bold', 'size': 13}

    ax.legend(legend_handles, legend_labels, loc='best', title="Methods",
             title_fontproperties=title_props, alignment='left', **legend_style)

    plt.savefig(args.save_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved plot to {args.save_file}")

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
            ax.set_ylabel("Retain Objective Gap")
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.5)

            scratch_loss_final = scratch_losses_for_plot.get(idx, 0.0)

            for label, hist in history_results[idx].items():
                if "::final" in label:
                    continue

                # Convert raw loss to excess risk and filter positive values for log plot
                valid_points = [(x, (y - scratch_loss_final) + 1e-15) for x, y in enumerate(hist) if (y - scratch_loss_final) + 1e-15 > 0]
                if valid_points:
                    xs, ys = zip(*valid_points)
                    line, = ax.plot(xs, ys, label=label)

                    # Check for final risk point (to visualize noise jump for 'after' methods)
                    final_key = f"{label}::final"
                    if final_key in history_results[idx]:
                        final_risk = (history_results[idx][final_key] - scratch_loss_final) + 1e-15
                        if final_risk > 0:
                            ax.plot(xs[-1], final_risk, marker='*', markersize=12, color=line.get_color())
                            ax.plot([xs[-1], xs[-1]], [ys[-1], final_risk], linestyle=':', color=line.get_color(), alpha=0.7)
            ax.legend()

        base, ext = os.path.splitext(args.save_file)
        history_file = f"{base}_history{ext}"
        plt.savefig(history_file, bbox_inches='tight')
        print(f"Saved history plot to {history_file}")

    # 6. Summary of Best Results
    print("\n" + "="*110)
    print(f"{'SUMMARY: Lowest Excess Risk per r_f (across all seeds)':^110}")
    print("="*110)
    print(f"{'r_f':<12} | {'Min Excess Risk':<18} | {'Min (Mean-Std)':<18} | {'Method':<30} | {'Seed':<5}")
    print("-" * 110)
    for i, r_f in enumerate(r_f_values):
        data_rf = all_run_results[:, i, :]
        if np.all(np.isnan(data_rf)):
            print(f"{r_f:<12.4e} | {'No valid data':<18} | {'-':<18} | {'-':<30} | {'-':<5}")
            continue
        min_flat_idx = np.nanargmin(data_rf)
        m_idx, s_idx = np.unravel_index(min_flat_idx, data_rf.shape)
        min_val = data_rf[m_idx, s_idx]
        m_name, _, _, m_time = runs[m_idx]

        with np.errstate(all='ignore'):
            means = np.nanmean(data_rf + 1e-15, axis=1)
            stds = np.nanstd(data_rf + 1e-15, axis=1)
        lower_bounds = means - stds
        min_lower_bound = np.nanmin(lower_bounds)

        print(f"{r_f:<12.4e} | {min_val:<18.6e} | {min_lower_bound:<18.6e} | {_get_pretty_name(m_name):<30} | {s_idx:<5}")
    print("="*110 + "\n")

    plt.show()

if __name__ == "__main__":
    main()
