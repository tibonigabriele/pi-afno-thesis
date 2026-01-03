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
# - The Fourier layer below follows the standard “spectral convolution” pattern
#   popularized by Fourier Neural Operators (FNO). Similar reference
#   implementations are publicly available in the FNO literature/code ecosystem.
#   This is a clean-room implementation tailored to this thesis codebase and to
#   small spatial grids (often 1×1 in this project’s dataset representation).
# - The AFNO-style components implement a minimal mode-wise gating mechanism in
#   the Fourier domain. This is an intentionally lightweight variant inspired by
#   the AFNO idea, and it is NOT a full reproduction of the complete AFNO
#   architecture (e.g., full mixing blocks, token/channel mixing design choices).
# - Conceptual pointers (research context):
#   * FNO: “Fourier Neural Operator” (operator learning via truncated Fourier modes).
#   * AFNO: “Adaptive Fourier Neural Operator” (learned adaptivity in Fourier space).
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.fft as fft

# Note: AFNOBarrier / AFNOBarrierPINO implement a lightweight Fourier-mode gating
# mechanism inspired by AFNO. This is not a 1:1 reimplementation of the full AFNO
# architecture; only the minimal “learned gate over retained modes” is used here
# for ablation and thesis-focused experimentation.


class SpectralConv2d(nn.Module):
    """
    Basic 2D spectral convolution layer (FNO-style): retains only low-frequency modes.

    Input
    -----
    x : torch.Tensor
        Shape (B, C_in, H, W)

    Output
    ------
    torch.Tensor
        Shape (B, C_out, H, W)
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Complex weights for retained (low-frequency) modes.
        # Stored as real/imag parts in the last dimension for convenience.
        self.scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        # (B, C_in, Hf, Wf) x (C_in, C_out, m1, m2) -> (B, C_out, m1, m2)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bchw,cohw->bohw", input, cweights)

    def forward(self, x):
        batchsize, channels, height, width = x.shape
        # Forward FFT (real-input 2D FFT).
        x_ft = fft.rfft2(x, norm="ortho")

        # Retain a bounded number of modes (robust to very small H/W).
        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.size(-2),
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )

        weight = self.weight[..., :m1, :m2, :]
        out_ft[..., :m1, :m2] = self.compl_mul2d(x_ft[..., :m1, :m2], weight)

        # Inverse FFT back to the spatial domain.
        x = fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x


class AFNOSpectralConv2d(nn.Module):
    """
    Adaptive spectral convolution: applies a learned gate over Fourier modes.

    This module is intentionally minimal: it uses a sigmoid gate on the retained
    low-frequency coefficients to modulate their contribution.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )
        # Gate logits are initialized slightly negative so that the initial gate is < 0.5
        # (sigmoid(-0.5) ≈ 0.38), providing a conservative starting point.
        self.gate_logits = nn.Parameter(
            -0.5 * torch.ones(1, 1, modes1, modes2)
        )

    def compl_mul2d(self, input, weights, gate):
        cweights = torch.view_as_complex(weights)
        # gate: (1, 1, m1, m2), broadcast over (C_in, C_out, m1, m2)
        gated_weights = cweights * gate
        return torch.einsum("bchw,cohw->bohw", input, gated_weights)

    def forward(self, x):
        batchsize, channels, height, width = x.shape
        x_ft = fft.rfft2(x, norm="ortho")

        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.size(-2),
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )

        weight = self.weight[..., :m1, :m2, :]
        gate = torch.sigmoid(self.gate_logits[..., :m1, :m2])
        out_ft[..., :m1, :m2] = self.compl_mul2d(x_ft[..., :m1, :m2], weight, gate)

        x = fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x


class FNOBlock2d(nn.Module):
    """
    Single FNO-style block: spectral convolution + 1x1 convolution + nonlinearity.
    """
    def __init__(self, width, modes1, modes2, adaptive=False):
        super().__init__()
        Conv = AFNOSpectralConv2d if adaptive else SpectralConv2d
        self.conv = Conv(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.act(x)
        return x


class FNOBarrierBase(nn.Module):
    """
    Base class for FNO-like models operating on a (S, t) grid representation.

    Maps:
        (B, C_in, H, W) -> (B, C_out, H, W)

    In this thesis codebase, H and W may be very small (often 1×1) depending on
    the dataset representation used for scalar market features.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=64,
        modes1=16,
        modes2=16,
        n_layers=4,
        adaptive=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2, adaptive=adaptive) for _ in range(n_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C_in, H, W).
        """
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.output_proj(x)
        return x


class FNOBarrier(FNOBarrierBase):
    """
    Vanilla FNO baseline model for barrier-option learning tasks.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=64,
        modes1=16,
        modes2=16,
        n_layers=4,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            n_layers=n_layers,
            adaptive=False,
        )


class FNOBarrierPINO(FNOBarrier):
    """
    Architecturally identical to FNOBarrier.

    The “physics-informed” behavior is introduced externally through the training
    objective (i.e., a physics loss term), not through model-architecture changes.
    """
    pass


class AFNOBarrier(FNOBarrierBase):
    """
    AFNO-like variant: FNO backbone with a lightweight learned Fourier-mode gate.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=64,
        modes1=16,
        modes2=16,
        n_layers=4,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            n_layers=n_layers,
            adaptive=True,
        )


class AFNOBarrierPINO(AFNOBarrier):
    """
    AFNO-like model with an external physics-informed loss term (thesis main model).
    """
    pass
