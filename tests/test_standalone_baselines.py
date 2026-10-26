"""Standalone diagnostics must also preserve independent scratch checkpoints."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from digit_forget import UnlearnLoaders, forget
from digit_train import Loaders
from gradient_representation import run_unlearning_gradient_trace


class StandaloneBaselineTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        data = TensorDataset(
            torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64), torch.tensor([0, 1]),
        )
        retain = DataLoader(data, batch_size=2)
        self.uloaders = UnlearnLoaders(
            retain=retain, retain_full=retain, forget=retain, n_retain=2, n_forget=2,
        )
        self.loaders = Loaders(retain, retain, None, retain, 2, 2, 2)
        self.model = nn.Linear(2, 2).to(dtype=torch.float64)
        self.scratch = nn.Linear(2, 2).to(dtype=torch.float64)
        with torch.no_grad():
            self.scratch.weight.copy_(torch.tensor([[-20., 20.], [20., -20.]]))
            self.scratch.bias.zero_()

    def test_gradient_diagnostic_never_replaces_scratch_even_with_legacy_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scratch.pt"
            torch.save(self.scratch.state_dict(), path)
            original = path.read_bytes()
            result = run_unlearning_gradient_trace(
                "nft", self.model, self.loaders, self.uloaders, 0.5, self.device,
                epochs=1, lr=0.1, scheduler="none", discount_factor=1., weight_decay=0.,
                maybe_save_better=True, scratch_best_path=str(path), scratch_best_loss=float("inf"),
            )
            self.assertTrue(result["true_loss"])
            self.assertEqual(path.read_bytes(), original)

    def test_forget_never_replaces_independent_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scratch.pt"
            torch.save(self.scratch.state_dict(), path)
            original = path.read_bytes()
            with patch("digit_forget.load_best_scratch_state", return_value=self.scratch.state_dict()), \
                 patch("digit_forget._best_model_path", return_value=str(path)):
                result = forget(
                    self.model, self.loaders, self.uloaders, 0.5, self.device,
                    seed=0, algo="sgd", epochs=1, lr=0.1,
                )
            self.assertTrue(torch.isfinite(torch.tensor(result["test_loss"])))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
