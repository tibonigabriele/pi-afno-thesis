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
# - This module implements a lightweight, augmentation-oriented routine inspired
#   by the general idea of frequency-domain perturbations (often discussed in the
#   context of Fourier-based denoising/augmentation). It is NOT a full, faithful
#   reimplementation of any specific "P-FTD" denoising method.
# - The use of FFT, phase perturbation, and amplitude scaling is a common pattern
#   in signal-augmentation contexts, and does not appear uniquely attributable to
#   a single public repository.
# - Conceptual pointers (research context):
#   * Fourier-domain manipulation is widely used in spectral ML and operator
#     learning settings (e.g., FNO-related literature).
# -----------------------------------------------------------------------------

import math
import torch


def fourier_augment_batch(
    x: torch.Tensor,
    prob: float = 0.5,
    max_scale: float = 0.15,
) -> torch.Tensor:
    """
    P-FTD-like frequency augmentation for scalar feature vectors.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape [B, C_in, 1, 1], typically min-max normalized in [0, 1].
        The feature dimension C_in is interpreted as a 1D signal.
    prob : float, optional
        Probability of applying the augmentation independently to each sample.
        If prob <= 0, the input is returned unchanged.
    max_scale : float, optional
        Maximum relative scaling applied per frequency component, sampled as
        U(1-max_scale, 1+max_scale).

    Returns
    -------
    torch.Tensor
        Augmented tensor with the same shape [B, C_in, 1, 1].

    Method
    ------
    For each selected sample:
      1) rFFT over the feature vector (length C_in).
      2) Apply random phase shifts in U(-pi, pi).
      3) Apply multiplicative amplitude scaling in U(1-max_scale, 1+max_scale).
      4) Inverse rFFT to return to the feature domain.
      5) Clamp to [0, 1] to limit extreme outliers under min-max normalization.
    """
    if prob <= 0.0:
        return x

    B, C, H, W = x.shape
    assert H == 1 and W == 1, "fourier_augment_batch assumes input shaped as [B, C, 1, 1]."

    x_flat = x[:, :, 0, 0]  # [B, C]
    x_aug = x_flat.clone()

    for i in range(B):
        if torch.rand(1).item() > prob:
            continue

        sig = x_flat[i]  # [C]

        # rFFT over the feature vector.
        freq = torch.fft.rfft(sig)

        # Phase ~ U(-pi, pi), scaling ~ U(1-max_scale, 1+max_scale).
        phases = torch.rand_like(freq.real) * 2.0 * math.pi - math.pi
        scales = 1.0 + (torch.rand_like(freq.real) * 2.0 - 1.0) * max_scale

        freq_aug = freq * torch.exp(1j * phases) * scales

        sig_aug = torch.fft.irfft(freq_aug, n=C)
        x_aug[i] = sig_aug.real

    # Optional: project back to [0, 1] to limit excessively aggressive perturbations.
    x_aug = torch.clamp(x_aug, 0.0, 1.0)
    return x_aug.view(B, C, 1, 1)
