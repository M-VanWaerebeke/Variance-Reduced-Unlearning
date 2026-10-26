"""The retraining baseline must survive an unlearning run unchanged."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_grid import UnlearnLoaders, run_unlearning_with_checkpoints


class ScratchBaselineTests(unittest.TestCase):
    def test_unlearning_cannot_replace_scratch_checkpoint(self):
        device = torch.device("cpu")
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        dataset = TensorDataset(features, labels)
        retain = DataLoader(dataset, batch_size=2)
        uloaders = UnlearnLoaders(
            retain=retain,
            retain_full=retain,
            forget=retain,
            n_retain=2,
            n_forget=2,
        )
        base_model = nn.Linear(2, 2).to(dtype=torch.float64)
        scratch_model = nn.Linear(2, 2).to(dtype=torch.float64)
        with torch.no_grad():
            scratch_model.weight.fill_(7.0)
            scratch_model.bias.fill_(-3.0)

        with tempfile.TemporaryDirectory() as directory:
            scratch_path = Path(directory) / "scratch.pt"
            torch.save(scratch_model.state_dict(), scratch_path)
            original_checkpoint = scratch_path.read_bytes()

            _, history = run_unlearning_with_checkpoints(
                method="nft",
                base_model=base_model,
                loaders=None,
                uloaders=uloaders,
                r_f=0.5,
                device=device,
                epochs=1,
                lr=0.1,
                scheduler="none",
                scheduler_step_freq=1,
                discount_factor=1.0,
                weight_decay=0.0,
                checkpoint_every=1,
                scratch_best_path=str(scratch_path),
                scratch_best_loss=float("inf"),
                verbose=False,
            )

            self.assertTrue(history)
            self.assertLess(min(loss for _, loss in history), float("inf"))
            self.assertEqual(scratch_path.read_bytes(), original_checkpoint)


if __name__ == "__main__":
    unittest.main()
