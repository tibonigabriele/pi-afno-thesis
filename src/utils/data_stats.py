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
# Notes on originality and references:
# - Computing dataset-wide mean/std statistics is a standard preprocessing step
#   in machine learning pipelines (widely used across public repositories).
# - This implementation is straightforward and not derived from a unique public
#   codebase; no specific attribution beyond the general convention is required.
# -----------------------------------------------------------------------------

import torch


def compute_target_stats(data_path: str) -> dict:
    """
    Load a .pt dataset and compute per-target mean and standard deviation for:
        [price, delta, vega].

    Returns
    -------
    dict
        {
            "mean": torch.Tensor of shape [3],
            "std":  torch.Tensor of shape [3],
        }
    """
    data = torch.load(data_path, map_location="cpu")

    y = data["targets"]  # [N, 3, 1, 1]
    # Collapse singleton spatial dimensions.
    y = y.view(y.shape[0], y.shape[1])  # [N, 3]

    mean = y.mean(dim=0)
    std = y.std(dim=0)

    print("\n[compute_target_stats] Target statistics computed from the dataset:")
    print(f"  price: mean={mean[0]: .6f}, std={std[0]: .6f}")
    print(f"  delta: mean={mean[1]: .6f}, std={std[1]: .6f}")
    print(f"   vega: mean={mean[2]: .6f}, std={std[2]: .6f}")

    return {"mean": mean, "std": std}