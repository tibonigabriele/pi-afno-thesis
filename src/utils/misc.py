# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Gabriele Tiboni
# MSc Thesis — Computer Engineering, University of Padua (UniPD)
#
# Rights and licensing:
# - This file is provided as part of an academic thesis codebase.
# - If a repository-wide LICENSE file is present, this file is distributed under
#   those terms; otherwise, all rights are reserved by the author.
#
# Notes on originality and references:
# - The utilities below (seeding RNGs, selecting a torch device, ensuring output
#   directories) follow common conventions broadly used in PyTorch-based code.
# - Similar patterns appear across public examples and official documentation;
#   this implementation is generic and does not reproduce a distinctive third-
#   party codebase that would require explicit attribution.
# -----------------------------------------------------------------------------

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set RNG seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu=True):
    """Return the preferred torch.device, selecting CUDA when available."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist."""
    if path is None:
        return
    os.makedirs(path, exist_ok=True)