"""Checks for the corrected shadow attack's key statistical assumptions."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from risk_vs_privacy import (
    _global_subset_indices,
    _noise_shadow_for_attack,
    gaussian_log_likelihood_ratio,
    shadow_manifest_fingerprint,
)
from shadow_utils import sample_forget_set_with_sample


class AttackProtocolTests(unittest.TestCase):
    def test_gaussian_density_includes_variance_normalizer(self):
        log_ratio, _ = gaussian_log_likelihood_ratio(
            0.0, np.array([-1.0, 1.0]), np.array([-2.0, 2.0])
        )
        self.assertAlmostEqual(log_ratio, -np.log(2.0), places=6)

    def test_shadow_forget_items_were_in_base_training(self):
        dataset = TensorDataset(torch.arange(12), torch.zeros(12))
        base_train_indices = [1, 3, 4, 8, 9]
        chosen, loader = sample_forget_set_with_sample(
            target_sample_idx=3,
            forget_size=3,
            full_dataset=dataset,
            rng=np.random.default_rng(7),
            candidate_indices=base_train_indices,
            batch_size=2,
        )
        self.assertEqual(len(chosen), 3)
        self.assertTrue(set(chosen).issubset(base_train_indices))
        self.assertEqual(loader.batch_size, 2)
        with self.assertRaises(ValueError):
            sample_forget_set_with_sample(
                2, 2, dataset, np.random.default_rng(7),
                candidate_indices=base_train_indices,
            )

    def test_held_out_ids_map_to_manifest_coordinates(self):
        dataset = TensorDataset(torch.arange(10), torch.zeros(10))
        half = Subset(dataset, [9, 2, 5, 7])
        selected = Subset(half, [2, 0])
        self.assertEqual(_global_subset_indices(selected, dataset), [5, 9])

    def test_empirical_shadow_receives_same_noise_scale(self):
        model = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.float64)
        with torch.no_grad():
            model.weight.zero_()
        seed = 11
        _noise_shadow_for_attack(
            model, {"timing": "empirical", "kappa_dp": 0.1}, 0.2,
            np.random.default_rng(seed), torch.device("cpu"),
        )
        expected = np.random.default_rng(seed).normal(scale=0.1 * 0.2 / 0.8)
        self.assertAlmostEqual(model.weight.item(), expected)

    def test_shadow_cache_identity_changes_with_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv = Path(temp_dir) / "digits_shadow_manifest_N2_LR_wd0.100.csv"
            metadata = Path(temp_dir) / "digits_shadow_metadata_N2_LR_wd0.100.json"
            csv.write_text("sample_id,model_0\n0,1\n")
            metadata.write_text('{"models": ["a.pt"]}')
            original = shadow_manifest_fingerprint(str(csv))
            csv.write_text("sample_id,model_0\n0,0\n")
            changed_membership = shadow_manifest_fingerprint(str(csv))
            metadata.write_text('{"models": ["b.pt"]}')
            changed_metadata = shadow_manifest_fingerprint(str(csv))
            self.assertEqual(len({original, changed_membership, changed_metadata}), 3)


if __name__ == "__main__":
    unittest.main()
