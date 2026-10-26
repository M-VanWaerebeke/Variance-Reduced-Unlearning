# Variance-Reduced Unlearning (VRU)

Research code for *Variance-Reduced (ε, δ)-Unlearning Using Forget Set Gradients*, by Martin Van Waerebeke, Marco Lorenzi, Kevin Scaman, El Mahdi El Mhamdi, and Giovanni Neglia.

[Paper](https://arxiv.org/abs/2602.14938) · [ICML 2026](https://icml.cc/virtual/2026/poster/66141)

VRU uses forget-set gradients in its unlearning updates. This repository contains the implementation, training utilities, and scripts for utility, empirical privacy, and projection-ablation experiments.

## Installation

Use a virtual environment with Python 3.9 or 3.11. The pinned CPU dependencies are in `requirements-cpu.txt`; `requirements.txt` lists broader dependency ranges.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

On Linux, CPU-only PyTorch wheels can be installed before the remaining dependencies:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

## Code structure

| File | Purpose |
| --- | --- |
| `fast_grid.py` | Unlearning updates and trajectory checkpoints for VRU and comparison methods |
| `digit_train.py` | Dataset loading, logistic-regression and neural-network models, and training |
| `digit_forget.py` | Retain/forget splits and standalone unlearning routines |
| `pretrain_model.py` | Pretraining configurations and checkpoint preparation |
| `risk_vs_rf.py` | Utility evaluation across forget fractions |
| `generate_shadow_csv.py` | Shadow-model training and membership-manifest generation |
| `shadow_utils.py` | Shadow-model selection and membership utilities |
| `unlearned_cache.py` | Unlearned shadow-model cache handling |
| `risk_vs_privacy.py` | Utility and membership-inference evaluation |
| `gradient_representation.py` | Gradient-estimator diagnostics and supporting training helpers |

The dataset loaders support scikit-learn Digits and Breast Cancer, plus MNIST through torchvision. Digits and Breast Cancer are bundled with scikit-learn; MNIST is downloaded to `data/` when requested. Model options include logistic regression (`LR`) and a feedforward neural network (`NN`).

## Experiment launchers

Run the shell scripts from a checkout of this repository. Each script specifies its dataset, model, learning rates, noise settings, forget fractions, and number of seeds. The table describes the configurations included in the scripts.

| Launcher | Experiment configuration |
| --- | --- |
| `scripts/figure1.sh` | Digits, LR: utility versus forget fraction; VRU and comparison methods |
| `scripts/figure2.sh LR` | Digits, LR: utility versus membership-inference risk, with five shadows |
| `scripts/figure2.sh NN` | Digits, NN: utility versus membership-inference risk, with five shadows |
| `scripts/figure2b_archived.sh` | Digits, NN: an additional privacy/utility configuration with 100 shadows and batch size 64 |
| `scripts/figure3.sh` | Digits, LR: projected versus unprojected VRU |
| `scripts/figure4.sh` | Breast Cancer, LR: privacy/utility evaluation with seven shadows |

For example:

```bash
bash scripts/figure1.sh
bash scripts/figure2.sh LR
bash scripts/figure3.sh
```

These launchers run full experiment grids and can require substantial compute, particularly for shadow-model training. They describe the included configurations; they do not establish an exact reproduction of the paper figures.

`PYTHON=/path/to/python` selects the interpreter used by a launcher. Privacy launchers generate shadow manifests before running the evaluation. `--parallel_jobs` in `generate_shadow_csv.py` controls shadow-training workers, and `--attack_parallel_jobs` in `risk_vs_privacy.py` controls attack-evaluation workers.

Inspect the individual command-line options with:

```bash
python risk_vs_rf.py --help
python generate_shadow_csv.py --help
python risk_vs_privacy.py --help
```

## Outputs

- `outputs/`: generated plots.
- `cache/`: saved experiment results.
- `shadow_manifests/`: shadow membership tables and metadata.
- Dataset-specific model directories: training and retraining checkpoints.

The launchers create their output and cache directories. Existing models and results can be reused by the runners. Generated data, checkpoints, caches, and plots are excluded from Git.

## Tests

```bash
python -m unittest discover -s tests -v
```

The CPU tests exercise update routines, checkpoint handling, and evaluation utilities. The GitHub Actions workflow runs these tests on Linux with Python 3.11.

## Citation and license

Please cite the associated paper when using this research code. Citation metadata is in [CITATION.cff](CITATION.cff). The code is available under the [MIT license](LICENSE).
