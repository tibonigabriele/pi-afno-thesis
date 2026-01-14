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
# - The training loop structure implemented here (epoch-based optimization,
#   validation pass, checkpointing on best validation loss, optional gradient
#   clipping, and optional learning-rate scheduling) follows widely used patterns
#   in the PyTorch ecosystem and common academic codebases.
# - The ReduceLROnPlateau handling, MAE/RMSE reporting, and best-checkpoint
#   serialization are standard practices; this file is tailored to the thesis
#   setting (multi-target regression for [price, delta, vega], optional
#   physics-informed term, and optional data augmentation).
# - No portion of this code is intended as a verbatim copy of a specific public
#   repository; similarities with common boilerplates are expected due to the
#   conventional nature of training loops.
# -----------------------------------------------------------------------------

import os
from typing import Optional, Callable, Any, Tuple, Dict

import torch
import torch.nn as nn


def _move_batch_to_device(batch, device):
    """
    Move a batch to the target device.

    Supported batch formats:
      - (inputs, targets)
      - {"inputs": ..., "targets": ...}

    Returns
    -------
    tuple
        (inputs, targets, batch_on_device) where batch_on_device preserves the
        original container format (tuple/list or dict).
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        inputs, targets = batch
    elif isinstance(batch, dict):
        inputs = batch["inputs"]
        targets = batch["targets"]
    else:
        raise ValueError(
            "Unrecognized batch format: expected (x, y) or a dict with keys "
            "'inputs' and 'targets'."
        )

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Reconstruct batch_on_device in the original container format.
    if isinstance(batch, (tuple, list)):
        batch_on_device = (inputs, targets)
    else:
        batch_on_device = {"inputs": inputs, "targets": targets}

    return inputs, targets, batch_on_device


def _compute_data_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    target_stats: Optional[dict],
    normalize_loss: bool,
):
    """
    Compute the supervised data loss.

    If normalize_loss=True and target_stats is provided, the criterion is applied
    to standardized targets:
        y_norm = (y - mean) / std

    If normalize_loss=False or target_stats is None, the criterion is applied in
    the original (raw) target scale.
    """
    if target_stats is None or not normalize_loss:
        # Loss in raw scale (price/delta/vega as provided).
        return criterion(outputs, targets)

    mean = target_stats["mean"].to(outputs.device).view(1, -1, 1, 1)
    std = target_stats["std"].to(outputs.device).view(1, -1, 1, 1)

    targets_norm = (targets - mean) / std
    outputs_norm = (outputs - mean) / std

    return criterion(outputs_norm, targets_norm)


def _compute_mae_rmse(outputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute per-channel MAE and RMSE.

    Parameters
    ----------
    outputs, targets : torch.Tensor
        Tensors shaped as [B, C, ...] with an arbitrary number of trailing
        dimensions.

    Returns
    -------
    tuple
        (mae, rmse) where each is a tensor of shape [C].
    """
    diff = outputs - targets

    if diff.ndim == 2:
        reduce_dims = (0,)
    elif diff.ndim == 3:
        reduce_dims = (0, 2)
    elif diff.ndim == 4:
        reduce_dims = (0, 2, 3)
    else:
        # Generic fallback: reduce over all dimensions except the channel dim (1).
        reduce_dims = tuple(d for d in range(diff.ndim) if d != 1)

    mae = diff.abs().mean(dim=reduce_dims)
    rmse = torch.sqrt((diff**2).mean(dim=reduce_dims))
    return mae, rmse


def _save_best_model(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_dir: Optional[str],
    best_val_loss: float,
):
    """
    Save the current best checkpoint to:
        best_model.pt

    Checkpoint format
    -----------------
    {
        "model_state_dict": ...,
        "optimizer_state_dict": ... (may be None),
        "best_val_loss": float,
    }
    """
    if checkpoint_dir is None:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "best_model.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint updated: {path} (best val_loss = {best_val_loss:.4e})")


def _call_physics_loss(
    physics_loss_fn: Optional[Callable[..., Any]],
    model: nn.Module,
    batch_on_device: Any,
    outputs: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Robust wrapper for calling an optional physics loss.

    The physics_loss_fn may return:
      - a single tensor (loss), or
      - (loss, info_dict)

    Returns
    -------
    tuple
        (loss_tensor, info_dict)
    """
    if physics_loss_fn is None:
        return torch.tensor(0.0, device=outputs.device), {}

    out = physics_loss_fn(model, batch_on_device, outputs)

    if isinstance(out, tuple):
        phys_loss, info = out
    else:
        phys_loss, info = out, {}

    if not torch.is_tensor(phys_loss):
        phys_loss = torch.tensor(float(phys_loss), device=outputs.device)

    return phys_loss, info or {}


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    physics_loss_fn: Optional[Callable[[nn.Module, Any, torch.Tensor], Any]] = None,
    lambda_phys: float = 0.0,
    checkpoint_dir: Optional[str] = None,
    print_every: int = 1,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None,
    augment_fn: Optional[Callable[[torch.Tensor, torch.Tensor], tuple]] = None,
    target_stats: Optional[dict] = None,
    normalize_loss: bool = True,  # default True to preserve baseline behavior
):
    """
    Generic training loop for supervised (and optionally physics-informed) learning.

    Key arguments
    -------------
    model : nn.Module
        Neural network (e.g., FNOBarrier).
    train_loader, val_loader :
        PyTorch DataLoaders.
    optimizer : torch.optim.Optimizer
        Optimization algorithm.
    device : torch.device
        Training device.
    num_epochs : int
        Number of epochs.
    physics_loss_fn : callable, optional
        Optional physics loss, called as physics_loss_fn(model, batch, outputs).
    lambda_phys : float
        Weight for the physics loss term.
    checkpoint_dir : str, optional
        Directory where the best checkpoint (best_model.pt) is saved.
    scheduler : lr_scheduler, optional
        Learning-rate scheduler (e.g., ReduceLROnPlateau).
    grad_clip : float, optional
        Gradient norm clipping threshold.
    augment_fn : callable, optional
        Data augmentation function: (inputs, targets) -> (inputs, targets).
    target_stats : dict, optional
        {"mean": tensor[C_out], "std": tensor[C_out]} used for loss normalization.
    normalize_loss : bool
        If True and target_stats is provided, computes the loss in standardized
        target space; otherwise computes the loss in raw target scale.
    """
    model.to(device)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    # Debug: print batch-level loss decomposition for the first epoch and first few batches.
    # (This helps detect "physics loss not applied" or scale/logging issues.)
    debug_first_epoch_batches = 2  # print for batch_idx < 2 in epoch 1

    for epoch in range(1, num_epochs + 1):
        # ==================== TRAIN ====================
        model.train()
        running_loss = 0.0
        n_train_batches = 0

        # Track physics-loss mean across the epoch (if enabled).
        train_phys_loss_sum = 0.0
        n_train_phys_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets, batch_on_device = _move_batch_to_device(batch, device)

            if augment_fn is not None:
                inputs, targets = augment_fn(inputs, targets)

                # If augmentation is applied, update batch_on_device accordingly.
                if isinstance(batch_on_device, (tuple, list)):
                    batch_on_device = (inputs, targets)
                else:
                    batch_on_device = {"inputs": inputs, "targets": targets}

            optimizer.zero_grad()

            outputs = model(inputs)

            # Supervised data loss (optionally normalized).
            data_loss = _compute_data_loss(
                outputs,
                targets,
                criterion,
                target_stats,
                normalize_loss,
            )

            # Optional physics loss term.
            if physics_loss_fn is not None and lambda_phys > 0.0:
                phys_loss, phys_info = _call_physics_loss(
                    physics_loss_fn, model, batch_on_device, outputs
                )
                loss = data_loss + lambda_phys * phys_loss

                train_phys_loss_sum += phys_loss.item()
                n_train_phys_batches += 1

                # --- DEBUG PRINT (first epoch, first 2 batches) ---
                if epoch == 1 and batch_idx < debug_first_epoch_batches:
                    pde_loss = phys_info.get("pde_loss", None)
                    barrier_loss = phys_info.get("barrier_loss", None)

                    msg = (
                        f"[dbg][train] epoch={epoch} batch={batch_idx} "
                        f"data_loss={data_loss.item():.4e} "
                        f"phys_loss={phys_loss.item():.4e} "
                        f"lambda={lambda_phys:.3e} "
                        f"total={loss.item():.4e}"
                    )
                    if pde_loss is not None or barrier_loss is not None:
                        msg += (
                            f" | pde_loss={pde_loss if pde_loss is not None else 'NA'} "
                            f"barrier_loss={barrier_loss if barrier_loss is not None else 'NA'}"
                        )
                    print(msg)
            else:
                loss = data_loss

                # --- DEBUG PRINT (first epoch, first 2 batches) ---
                if epoch == 1 and batch_idx < debug_first_epoch_batches:
                    print(
                        f"[dbg][train] epoch={epoch} batch={batch_idx} "
                        f"data_loss={data_loss.item():.4e} "
                        f"phys_loss=NA lambda={lambda_phys:.3e} "
                        f"total={loss.item():.4e}"
                    )

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += loss.item()
            n_train_batches += 1

        train_loss = running_loss / max(1, n_train_batches)

        if n_train_phys_batches > 0:
            train_phys_loss_epoch = train_phys_loss_sum / n_train_phys_batches
        else:
            train_phys_loss_epoch = 0.0

        # ==================== VALIDATION ====================
        model.eval()
        val_running_loss = 0.0
        n_val_batches = 0

        val_phys_loss_sum = 0.0
        n_val_phys_batches = 0

        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs, targets, batch_on_device = _move_batch_to_device(batch, device)
                outputs = model(inputs)

                data_loss = _compute_data_loss(
                    outputs,
                    targets,
                    criterion,
                    target_stats,
                    normalize_loss,
                )

                if physics_loss_fn is not None and lambda_phys > 0.0:
                    phys_loss, phys_info = _call_physics_loss(
                        physics_loss_fn, model, batch_on_device, outputs
                    )
                    loss = data_loss + lambda_phys * phys_loss

                    val_phys_loss_sum += phys_loss.item()
                    n_val_phys_batches += 1

                    # --- DEBUG PRINT (first epoch, first 2 batches) ---
                    if epoch == 1 and batch_idx < debug_first_epoch_batches:
                        pde_loss = phys_info.get("pde_loss", None)
                        barrier_loss = phys_info.get("barrier_loss", None)

                        msg = (
                            f"[dbg][val]   epoch={epoch} batch={batch_idx} "
                            f"data_loss={data_loss.item():.4e} "
                            f"phys_loss={phys_loss.item():.4e} "
                            f"lambda={lambda_phys:.3e} "
                            f"total={loss.item():.4e}"
                        )
                        if pde_loss is not None or barrier_loss is not None:
                            msg += (
                                f" | pde_loss={pde_loss if pde_loss is not None else 'NA'} "
                                f"barrier_loss={barrier_loss if barrier_loss is not None else 'NA'}"
                            )
                        print(msg)
                else:
                    loss = data_loss

                    # --- DEBUG PRINT (first epoch, first 2 batches) ---
                    if epoch == 1 and batch_idx < debug_first_epoch_batches:
                        print(
                            f"[dbg][val]   epoch={epoch} batch={batch_idx} "
                            f"data_loss={data_loss.item():.4e} "
                            f"phys_loss=NA lambda={lambda_phys:.3e} "
                            f"total={loss.item():.4e}"
                        )

                val_running_loss += loss.item()
                n_val_batches += 1

                all_outputs.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

        val_loss = val_running_loss / max(1, n_val_batches)

        if n_val_phys_batches > 0:
            val_phys_loss_epoch = val_phys_loss_sum / n_val_phys_batches
        else:
            val_phys_loss_epoch = 0.0

        # Concatenate to compute per-channel metrics.
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        mae, rmse = _compute_mae_rmse(all_outputs, all_targets)

        # Scheduler step (e.g., ReduceLROnPlateau).
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ==================== LOGGING ====================
        if (epoch % print_every) == 0:
            # Convention: C_out = 3 -> [price, delta, vega].
            mae_price, mae_delta, mae_vega = mae
            rmse_price, rmse_delta, rmse_vega = rmse

            print(
                f"[Epoch {epoch}/{num_epochs}] "
                f"train_loss={train_loss:.4e}  val_loss={val_loss:.4e}"
            )

            # If physics loss is enabled, report epoch means.
            if physics_loss_fn is not None and lambda_phys > 0.0:
                print(
                    f"  physics_loss: train={train_phys_loss_epoch:.4e}, "
                    f"val={val_phys_loss_epoch:.4e}"
                )

            print(
                "  MAE:  "
                f"price={mae_price:.4e}, delta={mae_delta:.4e}, vega={mae_vega:.4e}"
            )
            print(
                "  RMSE: "
                f"price={rmse_price:.4e}, delta={rmse_delta:.4e}, vega={rmse_vega:.4e}"
            )

            # σ-MAE (MAE normalized by target standard deviation), if available.
            if target_stats is not None:
                std = target_stats["std"].cpu()
                sigma_mae = mae / std
                sigma_price, sigma_delta, sigma_vega = sigma_mae
                print(
                    "  σ-MAE: "
                    f"price={sigma_price:.3f}, delta={sigma_delta:.3f}, vega={sigma_vega:.3f}"
                )

        # ==================== CHECKPOINT ====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_best_model(model, optimizer, checkpoint_dir, best_val_loss)
