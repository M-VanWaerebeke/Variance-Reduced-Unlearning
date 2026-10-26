"""Regression checks for Figure 1's held-out metric and fixed learning rate."""

import argparse
import copy
import hashlib
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import risk_vs_rf
from fast_grid import (
    UnlearnLoaders,
    _evaluate_loader_loss,
    averaged_noised_retain_loss,
    build_model,
)


def _loader(features):
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.zeros(len(features), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=len(features))


class FigureOneEvaluationTests(unittest.TestCase):
    def test_noised_loss_uses_explicit_test_loader(self):
        model = build_model("LR", 2, 2).to(dtype=torch.float64)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
            model.linear.bias.zero_()
        retain = _loader([[1.0, 0.0]])
        test = _loader([[0.0, 1.0]])
        uloaders = UnlearnLoaders(retain, retain, retain, 1, 1)
        kwargs = dict(
            best_scratch_model=model,
            kappa_dp=0.0,
            repetitions=2,
            uloaders=uloaders,
            device=torch.device("cpu"),
            weight_decay=0.0,
            rng=np.random.default_rng(0),
        )

        retain_loss = averaged_noised_retain_loss(model, **kwargs)
        test_loss = averaged_noised_retain_loss(model, eval_loader=test, **kwargs)

        self.assertAlmostEqual(retain_loss, _evaluate_loader_loss(model, retain, torch.device("cpu"), 0.0))
        self.assertAlmostEqual(test_loss, _evaluate_loader_loss(model, test, torch.device("cpu"), 0.0))
        self.assertGreater(test_loss, retain_loss)

    def test_runner_uses_test_baseline_fixed_lr_new_cache_and_empirical_noise(self):
        retain = _loader([[1.0, 0.0]])
        test = _loader([[0.0, 1.0]])
        uloaders = UnlearnLoaders(retain, retain, retain, 1, 1)
        loaders = SimpleNamespace(test=test, n_features=2, n_classes=2)
        scratch = build_model("LR", 2, 2).to(dtype=torch.float64)
        final = build_model("LR", 2, 2).to(dtype=torch.float64)
        with torch.no_grad():
            scratch.linear.weight.zero_()
            scratch.linear.bias.zero_()
            final.linear.weight.copy_(torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))
            final.linear.bias.zero_()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path = root / "full.pt"
            scratch_path = root / "scratch.pt"
            cache_path = root / "cache.pkl"
            torch.save(scratch.state_dict(), full_path)
            torch.save(scratch.state_dict(), scratch_path)
            scratch_hash = hashlib.md5(scratch_path.read_bytes()).hexdigest()
            full_hash = hashlib.md5(full_path.read_bytes()).hexdigest()
            old_key = (
                "nft", 0.3, "0.8", "before", 0, 1e-3,
                0.0, "digits", 1, 0.1, 1.0, "LR", scratch_hash, full_hash,
                False, False,
            )
            with cache_path.open("wb") as handle:
                pickle.dump({old_key: {"loss": 0.0, "history": [0.0]}}, handle)

            args = argparse.Namespace(
                min_rf=1e-3, max_rf=1e-1, n_rf=2, T=1.0, seeds=1,
                batch_size=1, weight_decay=0.0, dataset="digits", no_cuda=True,
                save_file=str(root / "figure.pdf"), cache_file=str(cache_path),
                model="LR", run=[("nft:0.1", "0.3", "0.8", "before")],
                original_style=True, average=False, project=False,
            )
            observed_lrs = []

            def fake_unlearn(**kwargs):
                observed_lrs.append(kwargs["lr"])
                return [(1, copy.deepcopy(final.state_dict()), 1.0)], [(0, 1.0)]

            old_cwd = Path.cwd()
            noise_fn = risk_vs_rf.apply_gaussian_noise_to_model
            try:
                os.chdir(root)
                with patch.object(risk_vs_rf, "parse_args", return_value=args), \
                     patch.object(risk_vs_rf, "set_seed"), \
                     patch.object(risk_vs_rf, "ensure_pretrained_models_parallel"), \
                     patch.object(risk_vs_rf, "ensure_scratch_models_parallel"), \
                     patch.object(risk_vs_rf, "get_data_loaders", return_value=((retain.dataset, None, test.dataset), loaders)), \
                     patch.object(risk_vs_rf, "get_retain_forget_loaders", return_value=uloaders), \
                     patch.object(risk_vs_rf, "_ensure_pretrained_full_model", return_value=str(full_path)), \
                     patch.object(risk_vs_rf, "_ensure_scratch_model", return_value=str(scratch_path)), \
                     patch.object(risk_vs_rf, "_load_full_base", return_value=scratch), \
                     patch.object(risk_vs_rf, "run_unlearning_with_checkpoints", side_effect=fake_unlearn), \
                     patch.object(risk_vs_rf, "apply_gaussian_noise_to_model", wraps=noise_fn) as noise_spy, \
                     patch.object(risk_vs_rf.plt, "savefig"), \
                     patch.object(risk_vs_rf.plt, "show"):
                    risk_vs_rf.main()
                    args.run = [("nft:0.2", "0.3", "0.8", "empirical")]
                    risk_vs_rf.main()
                    noise_calls = list(noise_spy.call_args_list)
            finally:
                os.chdir(old_cwd)
                risk_vs_rf.plt.close("all")

            self.assertEqual(observed_lrs, [0.3, 0.3, 0.3, 0.3])
            self.assertEqual(len(noise_calls), 2 * 2 * risk_vs_rf.OUTPUT_NOISE_REPETITIONS)
            observed_stds = [call.args[1] for call in noise_calls]
            for r_f in (1e-3, 1e-1):
                expected_std = 0.2 * r_f / (1.0 - r_f)
                self.assertEqual(sum(np.isclose(value, expected_std) for value in observed_stds),
                                 2 * risk_vs_rf.OUTPUT_NOISE_REPETITIONS)
            with cache_path.open("rb") as handle:
                cache = pickle.load(handle)
            new_entries = {key: entry for key, entry in cache.items()
                           if key[0] == risk_vs_rf.CACHE_SCHEMA and key[4] == "before"}
            self.assertEqual(len(new_entries), 2)
            empirical_entries = {key: entry for key, entry in cache.items()
                                 if key[0] == risk_vs_rf.CACHE_SCHEMA and key[4] == "empirical"}
            self.assertEqual(len(empirical_entries), 2)
            self.assertTrue(all(key[10] == 0.2 for key in empirical_entries))
            expected_test = _evaluate_loader_loss(final, test, torch.device("cpu"), 0.0)
            scratch_test = _evaluate_loader_loss(scratch, test, torch.device("cpu"), 0.0)
            self.assertLess(expected_test - scratch_test, 0.0)
            for entry in new_entries.values():
                self.assertAlmostEqual(entry["test_loss"], expected_test)
                self.assertAlmostEqual(entry["excess_risk"], expected_test - scratch_test)
                self.assertNotAlmostEqual(entry["test_loss"], _evaluate_loader_loss(final, retain, torch.device("cpu"), 0.0))
            largest_rf_entry = max(empirical_entries.items(), key=lambda item: item[0][6])[1]
            self.assertGreater(abs(largest_rf_entry["test_loss"] - expected_test), 1e-6)

    def test_empirical_kappa_must_be_positive_and_explicit(self):
        for method in ("vru", "vru:0", "vru:-1", "vru:nan", "vru:inf"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                risk_vs_rf._effective_kappa(method, "empirical")
        self.assertEqual(risk_vs_rf._effective_kappa("vru:0.2", "empirical"), 0.2)


if __name__ == "__main__":
    unittest.main()
