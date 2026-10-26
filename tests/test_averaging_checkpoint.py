"""The final VRU checkpoint must contain the Polyak-averaged output."""

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_grid import UnlearnLoaders, build_model, run_unlearning_with_checkpoints


class AveragingCheckpointTests(unittest.TestCase):
    def test_final_periodic_checkpoint_is_replaced_by_average(self):
        device = torch.device("cpu")
        retain_data = TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
            torch.tensor([0, 1]),
        )
        forget_data = TensorDataset(
            torch.tensor([[1.0, 1.0]], dtype=torch.float64),
            torch.tensor([0]),
        )
        uloaders = UnlearnLoaders(
            retain=DataLoader(retain_data, batch_size=1, shuffle=False),
            retain_full=DataLoader(retain_data, batch_size=2, shuffle=False),
            forget=DataLoader(forget_data, batch_size=1, shuffle=False),
            n_retain=2,
            n_forget=1,
        )
        torch.manual_seed(7)
        base_model = build_model("LR", 2, 2).to(dtype=torch.float64)
        kwargs = dict(
            method="vru",
            base_model=base_model,
            loaders=None,
            uloaders=uloaders,
            r_f=1.0 / 3.0,
            device=device,
            epochs=1,
            lr=0.3,
            scheduler="none",
            scheduler_step_freq=1,
            discount_factor=1.0,
            weight_decay=0.1,
            checkpoint_every=1,
            scratch_best_path="unused.pt",
            scratch_best_loss=0.0,
            project=False,
            verbose=False,
        )

        raw_checkpoints, _ = run_unlearning_with_checkpoints(**kwargs, average=False)
        averaged_checkpoints, _ = run_unlearning_with_checkpoints(**kwargs, average=True)

        self.assertEqual([step for step, _, _ in raw_checkpoints], [0, 1, 2])
        self.assertEqual([step for step, _, _ in averaged_checkpoints], [0, 1, 2])
        for name, averaged_tensor in averaged_checkpoints[-1][1].items():
            expected = (raw_checkpoints[1][1][name] + raw_checkpoints[2][1][name]) / 2
            torch.testing.assert_close(averaged_tensor, expected)
        self.assertTrue(any(
            not torch.allclose(averaged_checkpoints[-1][1][name], raw_checkpoints[-1][1][name])
            for name in averaged_checkpoints[-1][1]
        ))


if __name__ == "__main__":
    unittest.main()
