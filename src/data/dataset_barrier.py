# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Gabriele Tiboni
# MSc Thesis — Computer Engineering, University of Padua (UniPD)
#
# Rights and licensing:
# - This source file is released for academic/research dissemination as part of a
#   Master’s thesis project.
# - If this repository provides a LICENSE file, this file is distributed under
#   those terms. Otherwise, all rights are reserved by the author.
#
# Notes on originality:
# - The Dataset/DataLoader pattern implemented in this module is idiomatic PyTorch
#   and widely used across public examples and repositories. This code does not
#   show distinctive signatures of direct copying from a specific upstream source.
# -----------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader


class BarrierOptionDataset(Dataset):
    """
    Dataset for barrier options with normalized scalar features.

    inputs:  (N, C_in, 1, 1)  - already min-max normalized by scripts/build_barrier_dataset.py
    targets: (N, C_out, 1, 1) - [price, delta, vega]
    """
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        super().__init__()
        assert inputs.shape[0] == targets.shape[0], "inputs and targets must share the same N"
        self.inputs = inputs.float()
        self.targets = targets.float()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x, y


def load_barrier_option_dataloaders(
    data_path: str,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    shuffle: bool = True,
):
    """
    Load the dataset produced by scripts/build_barrier_dataset.py and return
    train/validation DataLoaders plus metadata (including x_min/x_max for inference).

    The .pt file referenced by data_path is expected to contain:
        - "inputs":  Tensor [N, C_in, 1, 1]
        - "targets": Tensor [N, C_out, 1, 1]
        - "x_min":   Tensor [C_in]
        - "x_max":   Tensor [C_in]
        - "S_grid":  Tensor [H] (optional placeholder)
        - "t_grid":  Tensor [W] (optional placeholder)
        - "meta":    dict with additional information
    """
    data = torch.load(data_path, map_location="cpu")

    inputs = data["inputs"]
    targets = data["targets"]
    x_min = data.get("x_min", None)
    x_max = data.get("x_max", None)
    S_grid = data.get("S_grid", None)
    t_grid = data.get("t_grid", None)
    meta_raw = data.get("meta", {})

    N = inputs.shape[0]
    n_val = int(val_ratio * N)

    # Random but reproducible train/validation split (assuming set_seed() is called upstream).
    indices = torch.randperm(N)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]

    train_dataset = BarrierOptionDataset(train_inputs, train_targets)
    val_dataset = BarrierOptionDataset(val_inputs, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Metadata propagated to scripts (inference, analysis, benchmarking, etc.).
    meta = dict(meta_raw)  # shallow copy
    meta["x_min"] = x_min
    meta["x_max"] = x_max
    meta["S_grid"] = S_grid
    meta["t_grid"] = t_grid
    meta["C_in"] = inputs.shape[1]
    meta["C_out"] = targets.shape[1]
    meta["N_total"] = N
    meta["N_train"] = train_inputs.shape[0]
    meta["N_val"] = val_inputs.shape[0]

    return train_loader, val_loader, meta
