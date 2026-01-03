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
# - This module implements a lightweight, augmentation-oriented variant inspired by
#   the general idea of padding-aware frequency-domain filtering. It is NOT intended
#   as a line-by-line reproduction of any specific "P-FTD" denoising algorithm.
# - The implementation uses standard PyTorch primitives (padding + FFT + masking),
#   which are common in many public codebases. No distinctive structure suggests
#   derivation from a single upstream repository.
# - Conceptual pointers (for attribution in a research context):
#   * Fourier-domain operator learning and spectral layers: FNO literature (Li et al., 2021).
#   * Padding strategies for spectral models are discussed in multiple works on
#     Fourier/spectral neural operators (e.g., variants addressing boundary effects).
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.fft as fft


def apply_spatial_padding(x, pad_mode: str = "mirror", pad_size: int = 4):
    """
    Apply a simple spatial padding strategy to mitigate boundary effects.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (C, H, W).
    pad_mode : str, optional
        Padding mode. Supported:
        - "mirror": reflection padding via PyTorch 'reflect'
        - "zero": constant-zero padding
        Unrecognized modes disable padding and return the input unchanged.
    pad_size : int, optional
        Requested padding size (symmetric on all spatial borders).

    Notes
    -----
    This function handles small spatial dimensions safely:
    - For 'reflect' padding, the effective padding per dimension must be < size.
    - If a consistent padding cannot be applied, the input is returned unchanged.
    """
    if pad_size <= 0:
        return x

    C, H, W = x.shape

    # For reflection padding, each pad must be strictly smaller than the
    # corresponding spatial dimension.
    max_pad_h = max(H - 1, 0)
    max_pad_w = max(W - 1, 0)
    effective_pad = min(pad_size, max_pad_h, max_pad_w)

    # If padding is not feasible, return the original tensor.
    if effective_pad <= 0:
        return x

    pad = (effective_pad, effective_pad, effective_pad, effective_pad)  # (left, right, top, bottom)

    if pad_mode == "mirror":
        x_padded = F.pad(x, pad, mode="reflect")
    elif pad_mode == "zero":
        x_padded = F.pad(x, pad, mode="constant", value=0.0)
    else:
        # Disable padding for unsupported modes.
        return x

    return x_padded


def pftd_augment(
    x: torch.Tensor,
    keep_fraction: float = 0.5,
    noise_std: float = 0.0,
    pad_size: int = 0,
    pad_mode: str = "mirror",
):
    """
    Frequency-domain augmentation via padding-aware low-pass filtering (simplified).

    This routine provides a lightweight augmentation mechanism inspired by the
    general concept of padding-based Fourier filtering. It is designed for data
    augmentation (not for faithful replication of a specific denoising pipeline).

    Parameters
    ----------
    x : torch.Tensor
        Single-sample input tensor with shape (C, H, W).
    keep_fraction : float, optional
        Fraction of low-frequency coefficients retained along each frequency axis.
        At least one coefficient per axis is always preserved.
    noise_std : float, optional
        Standard deviation of additive noise in the frequency domain (optional).
    pad_size : int, optional
        Optional spatial padding size applied before the FFT.
    pad_mode : str, optional
        Padding mode passed to `apply_spatial_padding`.

    Returns
    -------
    torch.Tensor
        Augmented tensor with the same spatial shape as the (possibly padded) input.

    Procedure
    ---------
    1) Optional spatial padding.
    2) 2D real FFT.
    3) Low-pass masking, retaining a fixed fraction of coefficients.
    4) Optional additive perturbation in the frequency domain.
    5) Inverse real FFT.
    """
    # Optional spatial padding (safe for small H/W).
    if pad_size > 0:
        x = apply_spatial_padding(x, pad_mode=pad_mode, pad_size=pad_size)

    C, H, W = x.shape
    x_ft = fft.rfft2(x, norm="ortho")

    # Frequency-domain sizes.
    Hf = x_ft.size(-2)
    Wf = x_ft.size(-1)

    # Compute retained frequency extents, ensuring at least one coefficient per axis.
    m1 = int(keep_fraction * Hf)
    m2 = int(keep_fraction * Wf)

    m1 = max(1, m1)
    m2 = max(1, m2)

    m1 = min(m1, Hf)
    m2 = min(m2, Wf)

    # Low-pass mask.
    mask = torch.zeros_like(x_ft, dtype=torch.bool)
    mask[..., :m1, :m2] = True

    x_ft_filtered = x_ft.clone()
    x_ft_filtered[~mask] = 0.0

    # Optional frequency-domain noise (added to the real component).
    # Note: this keeps the implementation intentionally lightweight.
    if noise_std > 0.0:
        noise = noise_std * torch.randn_like(x_ft_filtered.real)
        x_ft_filtered = x_ft_filtered + noise.to(x_ft_filtered.device)

    x_denoised = fft.irfft2(x_ft_filtered, s=(H, W), norm="ortho")
    return x_denoised