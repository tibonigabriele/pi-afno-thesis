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
# - The Black–Scholes PDE residual formulation and the use of finite-difference
#   approximations for derivatives are standard techniques in physics-informed
#   learning and computational finance. This implementation is a lightweight,
#   self-contained utility tailored to this thesis codebase (tensor shapes,
#   barrier penalty, and logging format).
# - The barrier enforcement via a squared-violation penalty corresponds to a
#   common “soft constraint” approach used in PINN/PINO-style training; it is not
#   a direct copy of any single public repository, but rather an application of
#   widely known methodology to the present setting.
# -----------------------------------------------------------------------------

import torch


def bs_residual_fd(u, S_grid, t_grid, r, sigma):
    """
    Black–Scholes PDE residual computed via finite differences on a grid.

    Parameters
    ----------
    u : torch.Tensor
        Tensor of shape (B, 1, H, W) or (B, C, H, W). The first channel (index 0)
        is interpreted as the option price surface.
    S_grid : torch.Tensor
        Spatial grid of shape (H,).
    t_grid : torch.Tensor
        Time grid of shape (W,).

    Returns
    -------
    torch.Tensor
        Residual tensor defined on the interior grid points, with shape
        (B, H-2, W-2) after consistent cropping.
    """
    # Use the price channel only (channel 0).
    if u.dim() != 4:
        raise ValueError("u must be (B, C, H, W)")
    u_price = u[:, 0]  # (B, H, W)

    B, H, W = u_price.shape
    device = u_price.device

    S = S_grid.to(device).view(1, H, 1).expand(B, H, W)
    t = t_grid.to(device).view(1, 1, W).expand(B, H, W)

    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]

    # Interior finite differences.
    # Time derivative (central difference over t).
    u_t = (u_price[:, :, 2:] - u_price[:, :, :-2]) / (2 * dt)

    # Spatial derivatives (central difference over S).
    u_S = (u_price[:, 2:, :] - u_price[:, :-2, :]) / (2 * dS)
    u_SS = (u_price[:, 2:, :] - 2 * u_price[:, 1:-1, :] + u_price[:, :-2, :]) / (dS**2)

    # Crop tensors to align dimensions.
    S_mid = S[:, 1:-1, 1:-1]
    u_t_mid = u_t[:, 1:-1, :]
    u_S_mid = u_S[:, :, 1:-1]
    u_SS_mid = u_SS[:, :, 1:-1]

    # PDE: u_t + 0.5 sigma^2 S^2 u_SS + r S u_S - r u = 0
    u_mid = u_price[:, 1:-1, 1:-1]
    residual = (
        u_t_mid
        + 0.5 * sigma**2 * S_mid**2 * u_SS_mid
        + r * S_mid * u_S_mid
        - r * u_mid
    )

    return residual


def barrier_mask_up_and_out(S_grid, B):
    """
    Up-and-out barrier mask on the spatial grid.

    Returns
    -------
    torch.Tensor
        Mask of shape (H,) equal to 1 for points strictly below the barrier
        (S < B) and 0 for points at/above the barrier (S >= B).
    """
    return (S_grid < B).float()


def physics_loss_bs_fd(
    u_pred,
    S_grid,
    t_grid,
    params,
    B=None,
    lambda_res=1.0,
    lambda_bc=1.0,
):
    """
    Physics-informed loss: Black–Scholes PDE residual + barrier constraint penalty.

    Parameters
    ----------
    u_pred : torch.Tensor
        Predicted fields of shape (B, C, H, W). Channel 0 is interpreted as the
        option price surface.
    S_grid : torch.Tensor
        Spatial grid of shape (H,).
    t_grid : torch.Tensor
        Time grid of shape (W,).
    params : dict
        Dictionary containing at least {"r": float, "sigma": float}.
    B : float or None
        Barrier level for an up-and-out knock-out condition. If None, only the
        PDE residual term is used.
    lambda_res : float
        Weight for the PDE residual term.
    lambda_bc : float
        Weight for the barrier penalty term.

    Returns
    -------
    tuple
        (loss_total, metrics_dict) where metrics_dict contains scalar logging
        entries for residual and barrier components.
    """
    r = params["r"]
    sigma = params["sigma"]

    # PDE residual on the interior domain.
    residual = bs_residual_fd(u_pred, S_grid, t_grid, r=r, sigma=sigma)
    loss_res = (residual**2).mean()

    # Soft enforcement of the knock-out boundary (up-and-out): u(S >= B, t) ≈ 0.
    if B is not None:
        device = u_pred.device
        S = S_grid.to(device)
        mask = (S >= B).float()  # knock-out region
        u_price = u_pred[:, 0]  # (B, H, W)
        # Broadcast mask (H,) -> (B, H, W).
        mask_b = mask.view(1, -1, 1).expand_as(u_price)
        bc_violation = (u_price * mask_b) ** 2
        loss_bc = bc_violation.mean()
    else:
        loss_bc = torch.tensor(0.0, device=u_pred.device)

    loss_total = lambda_res * loss_res + lambda_bc * loss_bc
    return loss_total, {"residual": loss_res.item(), "barrier": loss_bc.item()}
