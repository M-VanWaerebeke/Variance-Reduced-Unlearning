#!/usr/bin/env python3
"""
unlearned_cache.py - Cache and reuse unlearned shadow models across U-LiRA attacks.

Manages a manifest of unlearned models indexed by:
- Base model ID (from shadow CSV)
- Forget set (which samples were unlearned)
- Method + hyperparameters

Enables reuse: if model M unlearns forget set {A, B, C}, it can serve as a
target shadow for attacking any of A, B, or C.

Enforces base model diversity: never uses two unlearned models derived from
the same base model to attack a single sample.
"""
from __future__ import annotations
import json
import hashlib
import os
import tempfile
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class UnlearnedEntry:
    """A single cached unlearned model."""
    unlearned_id: int
    base_model_id: int
    forget_indices: List[int]
    path: str
    seed_used: int


def get_method_key(
    method: str,
    model_type: str,
    lr: float,
    decay: float,
    wd: float,
    epochs: int,
    rf: float,
    average: bool = False,
    project: bool = False,
    alpha: float = None,
) -> str:
    """
    Generate a deterministic key from unlearning hyperparameters.

    This key uniquely identifies the unlearning configuration so that
    models produced with the same setup can be reused.

    Args:
        method: Unlearning algorithm name (e.g., "vru", "sgd")
        model_type: Model architecture ("LR" or "NN")
        lr: Learning rate
        decay: Discount factor / decay
        wd: Weight decay
        epochs: Number of unlearning epochs
        rf: Forget fraction (|D_f| / |D|)
        average: Whether Polyak averaging is used
        project: Whether projection is used
        alpha: Alpha parameter for neggrad+ and scrub methods

    Returns:
        String key for manifest filename
    """
    wd_str = f"{wd:.1e}" if wd < 1e-3 else f"{wd:.3f}"
    rf_str = f"{rf:.1e}" if rf < 1e-2 else f"{rf:.3f}"
    lr_str = f"{lr:.1e}" if lr < 1e-2 else f"{lr:.3f}"
    decay_str = f"{decay:.2f}"

    key = f"{method}_{model_type}_lr{lr_str}_decay{decay_str}_wd{wd_str}_epochs{epochs}_rf{rf_str}"
    if average:
        key += "_avg"
    if project:
        key += "_proj"
    if alpha is not None:
        alpha_str = f"{alpha:.3f}" if alpha >= 1e-2 else f"{alpha:.1e}"
        key += f"_alpha{alpha_str}"
    return key


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for consistency checking."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class UnlearnedManifest:
    """
    Manages a collection of cached unlearned models for a specific
    (method, hyperparams) configuration.

    The manifest is stored as a JSON file and tracks:
    - Which base model each unlearned model came from
    - Which forget set was used
    - The path to the saved checkpoint

    Supports efficient lookup: given a sample ID, find all cached
    unlearned models that contain that sample in their forget set.
    """

    def __init__(self, manifest_path: str, base_csv_path: str = None):
        """
        Load or create an unlearned model manifest.

        Args:
            manifest_path: Path to the manifest JSON file
            base_csv_path: Path to the base model CSV (for consistency check)
        """
        self.manifest_path = manifest_path
        self.entries: List[UnlearnedEntry] = []
        self._sample_index: dict = {}  # sample_id -> list of entry indices
        self._next_id: int = 0
        self.base_csv_path: Optional[str] = base_csv_path
        self.base_csv_hash: Optional[str] = None
        self.method_key: Optional[str] = None
        self.dataset: Optional[str] = None

        if os.path.exists(manifest_path):
            self._load()
        elif base_csv_path:
            self.base_csv_hash = compute_file_hash(base_csv_path)

    def _load(self):
        """Load manifest from disk."""
        with open(self.manifest_path, 'r') as f:
            content = f.read()
        if not content.strip():
            # File exists but is empty (race condition during parallel writes)
            return
        data = json.loads(content)

        self.method_key = data.get('method_key')
        self.dataset = data.get('dataset')
        self.base_csv_path = data.get('base_csv')
        self.base_csv_hash = data.get('base_csv_hash')

        for entry_dict in data.get('entries', []):
            entry = UnlearnedEntry(
                unlearned_id=entry_dict['unlearned_id'],
                base_model_id=entry_dict['base_model_id'],
                forget_indices=entry_dict['forget_indices'],
                path=entry_dict['path'],
                seed_used=entry_dict['seed_used'],
            )
            self.entries.append(entry)

        # Rebuild index
        self._rebuild_index()

        # Set next ID
        if self.entries:
            self._next_id = max(e.unlearned_id for e in self.entries) + 1

    def _rebuild_index(self):
        """Rebuild the sample -> entries index for fast lookup."""
        self._sample_index = {}
        for idx, entry in enumerate(self.entries):
            for sample_id in entry.forget_indices:
                if sample_id not in self._sample_index:
                    self._sample_index[sample_id] = []
                self._sample_index[sample_id].append(idx)

    def find_compatible_shadows(
        self,
        sample_id: int,
        n_needed: int,
        allowed_base_ids: Set[int],
        exclude_base_ids: Optional[Set[int]] = None,
    ) -> List[UnlearnedEntry]:
        """
        Find existing unlearned models usable as target shadows for sample_id.

        Returns entries where:
        - sample_id is in entry.forget_indices
        - entry.base_model_id is in allowed_base_ids (IN models for this sample)
        - entry.base_model_id not in exclude_base_ids
        - All returned entries have distinct base_model_ids

        Args:
            sample_id: The sample being attacked
            n_needed: Maximum number of entries to return
            allowed_base_ids: Set of base model IDs that are IN for this sample
            exclude_base_ids: Set of base model IDs to skip (already used)

        Returns:
            List of compatible UnlearnedEntry objects (up to n_needed)
        """
        if exclude_base_ids is None:
            exclude_base_ids = set()

        # Get entry indices for this sample
        entry_indices = self._sample_index.get(sample_id, [])

        result = []
        used_base_ids = set(exclude_base_ids)

        for idx in entry_indices:
            entry = self.entries[idx]

            # Must be an IN model for this sample
            if entry.base_model_id not in allowed_base_ids:
                continue

            # Must not reuse same base model
            if entry.base_model_id in used_base_ids:
                continue

            # Check that the checkpoint still exists
            if not os.path.exists(entry.path):
                continue

            result.append(entry)
            used_base_ids.add(entry.base_model_id)

            if len(result) >= n_needed:
                break

        return result

    def add_entry(
        self,
        base_model_id: int,
        forget_indices: List[int],
        path: str,
        seed_used: int,
    ) -> UnlearnedEntry:
        """
        Register a new unlearned model in the manifest.

        Args:
            base_model_id: ID of the base model from the CSV
            forget_indices: List of sample indices that were unlearned
            path: Path to the saved unlearned checkpoint
            seed_used: Random seed used for this unlearning run

        Returns:
            The created UnlearnedEntry
        """
        entry = UnlearnedEntry(
            unlearned_id=self._next_id,
            base_model_id=base_model_id,
            forget_indices=sorted(forget_indices),
            path=path,
            seed_used=seed_used,
        )

        self.entries.append(entry)
        entry_idx = len(self.entries) - 1

        # Update index
        for sample_id in forget_indices:
            if sample_id not in self._sample_index:
                self._sample_index[sample_id] = []
            self._sample_index[sample_id].append(entry_idx)

        self._next_id += 1
        return entry

    def save(self):
        """Persist manifest to disk atomically to avoid race conditions."""
        manifest_dir = os.path.dirname(self.manifest_path)
        os.makedirs(manifest_dir, exist_ok=True)

        data = {
            'method_key': self.method_key,
            'dataset': self.dataset,
            'base_csv': self.base_csv_path,
            'base_csv_hash': self.base_csv_hash,
            'n_entries': len(self.entries),
            'entries': [asdict(e) for e in self.entries],
        }

        # Write to a temp file then atomically rename to avoid readers
        # seeing a truncated/empty file during concurrent access.
        fd, tmp_path = tempfile.mkstemp(dir=manifest_dir, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.manifest_path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    def verify_consistency(self, base_csv_path: str) -> bool:
        """
        Verify that the manifest is consistent with the current base CSV.

        If the base CSV has been regenerated (different hash), the cached
        unlearned models may be invalid.

        Args:
            base_csv_path: Path to the current base model CSV

        Returns:
            True if consistent, False if the base CSV has changed
        """
        if self.base_csv_hash is None:
            return True

        current_hash = compute_file_hash(base_csv_path)
        return current_hash == self.base_csv_hash

    def stats(self) -> dict:
        """Return summary statistics about the manifest."""
        if not self.entries:
            return {
                'n_entries': 0,
                'n_unique_base_models': 0,
                'n_indexed_samples': 0,
            }

        base_ids = set(e.base_model_id for e in self.entries)
        return {
            'n_entries': len(self.entries),
            'n_unique_base_models': len(base_ids),
            'n_indexed_samples': len(self._sample_index),
            'avg_forget_size': sum(len(e.forget_indices) for e in self.entries) / len(self.entries),
        }


def get_manifest_path(
    output_dir: str,
    dataset: str,
    method_key: str,
) -> str:
    """
    Construct the manifest file path from components.

    Args:
        output_dir: Base directory for shadow manifests
        dataset: Dataset name
        method_key: Method key from get_method_key()

    Returns:
        Full path to the manifest JSON file
    """
    filename = f"{dataset}_unlearned_manifest_{method_key}.json"
    return os.path.join(output_dir, filename)


def get_unlearned_model_path(
    output_dir: str,
    dataset: str,
    unlearned_id: int,
    base_model_id: int,
    rf: float,
    model_type: str = None,
) -> str:
    """
    Construct the path for saving an unlearned model checkpoint.

    Args:
        output_dir: Base directory for shadow manifests
        dataset: Dataset name
        unlearned_id: Unique ID for this unlearned model
        base_model_id: ID of the base model
        rf: Forget fraction
        model_type: Model architecture (e.g., "LR", "NN"). Included in filename
                    to prevent collisions between different architectures.

    Returns:
        Full path for the unlearned model checkpoint
    """
    rf_str = f"{rf:.1e}" if rf < 1e-2 else f"{rf:.3f}"
    models_dir = os.path.join(output_dir, f"{dataset}_unlearned_models")
    os.makedirs(models_dir, exist_ok=True)
    mt_str = f"_{model_type}" if model_type else ""
    filename = f"unlearned_{unlearned_id:05d}_base{base_model_id:03d}{mt_str}_rf{rf_str}.pt"
    return os.path.join(models_dir, filename)
