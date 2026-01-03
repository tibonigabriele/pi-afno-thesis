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
# - The construction of a physics-informed loss via automatic differentiation
#   (computing first/second derivatives w.r.t. selected normalized inputs) is a
#   standard PINN/PINO technique and appears broadly across the literature and
#   public implementations.
# - This specific implementation is tailored to the thesis setting: (i) scalar
#   feature vectors stored as [B, C_in, 1, 1], (ii) min–max de-normalization to
#   recover raw parameters, (iii) a Black–Scholes residual expressed in (x, T)
#   coordinates with x = log(S/K), and (iv) a barrier consistency penalty
#   approximating V(S=B) ≈ 0 by evaluating the model at the barrier state.
# - No verbatim copying from a single public repository is intended; similarities
#   with common PINN/PINO patterns are expected due to the conventional nature of
#   autodiff-based PDE residual enforcement.
# -----------------------------------------------------------------------------

import torch
from typing import Dict, Any, Tuple, Callable, Union


BatchType = Union[tuple, list, dict]


def _unpack_batch(batch: BatchType) -> torch.Tensor:
    """
    Extract the input tensor from a batch.

    Supported batch formats:
      - (inputs, targets)
      - {"inputs": ..., "targets": ...}

    Returns
    -------
    torch.Tensor
        The input tensor only.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        inputs, _ = batch
    elif isinstance(batch, dict):
        inputs = batch["inputs"]
    else:
        raise ValueError(
            "Unrecognized batch format in physics_loss: expected (x, y) or a dict "
            "with keys 'inputs'/'targets'."
        )
    return inputs


def make_black_scholes_physics_loss(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    lambda_barrier: float = 1.0,
) -> Callable:
    """
    Build a Black–Scholes physics-informed loss with a barrier consistency term,
    expressed in (x, T) coordinates.

    Parameters
    ----------
    x_min, x_max : torch.Tensor
        Min–max normalization statistics of shape [C_in], as stored in the dataset.
        Assumed RAW feature ordering (pre-normalization):
          0: log_moneyness = log(S/K)
          1: sigma_imp
          2: r
          3: T (time to maturity)
          4: B_over_S = B/S
          5: is_call
    lambda_barrier : float
        Weight of the barrier penalty term.

    Returns
    -------
    Callable
        A function with signature:
            physics_loss_fn(model, batch, model_outputs) -> (loss_tensor, info_dict)

        Notes:
          - batch is the same object returned by the DataLoader (tuple or dict),
            already moved to the target device by the training loop.
          - model_outputs is not used here (the forward pass is recomputed with
            gradients enabled).
    """
    # Freeze normalization parameters as constants.
    x_min = x_min.clone().detach()
    x_max = x_max.clone().detach()
    x_range = (x_max - x_min).clamp_min(1e-8)

    def physics_loss_fn(
        model,
        batch: BatchType,
        model_outputs: torch.Tensor,  # unused (kept for interface compatibility)
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # Extract inputs from the batch.
        inputs = _unpack_batch(batch)
        device = inputs.device

        B, C_in, _, _ = inputs.shape
        assert C_in >= 6, "This physics loss assumes at least 6 input features."

        # Ensure gradients are enabled even if the caller uses torch.no_grad().
        with torch.enable_grad():
            # Clone inputs with gradient tracking enabled.
            z = inputs.clone().detach().to(device)
            z.requires_grad_(True)

            # Forward pass: only the price channel (index 0) is used for the PDE residual.
            y = model(z)              # [B, C_out, 1, 1]
            V = y[:, 0, 0, 0]         # [B]

            # De-normalize input features to raw space.
            x_min_d = x_min.to(device)
            x_range_d = x_range.to(device)

            raw = x_min_d.view(1, C_in, 1, 1) + z * x_range_d.view(1, C_in, 1, 1)
            raw_flat = raw[:, :, 0, 0]     # [B, C_in]

            x_logm = raw_flat[:, 0]        # log(S/K)
            sigma = raw_flat[:, 1]
            r = raw_flat[:, 2]
            T = raw_flat[:, 3]
            B_over_S = raw_flat[:, 4]

            # First derivatives w.r.t. x (via z0) and T (via z3).
            grads = torch.autograd.grad(
                V.sum(), z, create_graph=True, retain_graph=True
            )[0]  # [B, C_in, 1, 1]

            dV_dz0 = grads[:, 0, 0, 0]
            dV_dz3 = grads[:, 3, 0, 0]

            dx = x_range_d[0]
            dT = x_range_d[3]
            dV_dx = dV_dz0 / dx
            dV_dT = dV_dz3 / dT

            # Second derivative w.r.t. x.
            grads2 = torch.autograd.grad(
                dV_dz0.sum(), z, create_graph=True, retain_graph=True
            )[0]  # [B, C_in, 1, 1]

            d2V_dz0_2 = grads2[:, 0, 0, 0]
            d2V_dx2 = d2V_dz0_2 / (dx * dx)

            # Black–Scholes PDE in (x, T):
            #   ∂V/∂T = 0.5 σ^2 ∂²V/∂x² + (r - 0.5 σ²) ∂V/∂x - r V
            pde_rhs = (
                0.5 * sigma**2 * d2V_dx2
                + (r - 0.5 * sigma**2) * dV_dx
                - r * V
            )
            residual = dV_dT - pde_rhs

            pde_loss = (residual**2).mean()

            # --- Barrier consistency: enforce V(S = B) ≈ 0 (up-and-out constraint). ---

            # Barrier log-moneyness: x_B = x + log(B/S).
            x_B = x_logm + torch.log(B_over_S.clamp_min(1e-8))

            # Raw feature vector at the barrier.
            raw_B = raw_flat.clone()
            raw_B[:, 0] = x_B
            raw_B[:, 4] = torch.ones_like(B_over_S)  # At the barrier, B/S_B = 1.

            # Re-normalize back to the network input space.
            z_B = (raw_B - x_min_d) / x_range_d
            z_B = z_B.view(B, C_in, 1, 1).to(device)

            V_B = model(z_B)[:, 0, 0, 0]
            barrier_loss = (V_B**2).mean()

            total_phys = pde_loss + lambda_barrier * barrier_loss

        info = {
            "pde_loss": float(pde_loss.detach().cpu()),
            "barrier_loss": float(barrier_loss.detach().cpu()),
        }
        return total_phys, info

    return physics_loss_fn
