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
# - The script follows a standard research training layout (data loaders, model
#   instantiation, optimizer/scheduler, training loop, checkpointing). This is a
#   conventional pattern and does not indicate direct reuse of a specific public
#   repository implementation.
# - Conceptual references for context:
#   * FNO: Li et al., 2021 (Fourier Neural Operator).
#   * PINO / physics-informed operator learning: PDE-residual regularization for
#     operator learning under governing equations (e.g., Black–Scholes).
# -----------------------------------------------------------------------------

import torch
import torch.optim as optim

from src.data.dataset_barrier import load_barrier_option_dataloaders
from src.models.fno_family import FNOBarrierPINO
from src.training.loop import train_model
from src.training.physics_loss import make_black_scholes_physics_loss
from src.utils.misc import set_seed, get_device, ensure_dir
from src.utils.data_stats import compute_target_stats


def main():
    set_seed(42)
    device = get_device()

    data_path = "data/barrier_dataset.pt"

    # A smaller batch size is used relative to the purely supervised baseline
    # due to the additional autograd overhead required by the physics loss.
    batch_size = 256

    # Match the training budget of the FNO baseline.
    num_epochs = 200

    lr = 3e-4
    weight_decay = 1e-4
    lambda_phys = 0.1  # physics-loss weight

    checkpoint_dir = "checkpoints/fno_pino_normloss"
    ensure_dir(checkpoint_dir)

    # === Data ===
    train_loader, val_loader, meta = load_barrier_option_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=0.1,
        shuffle=True,
    )

    C_in = meta["C_in"]
    C_out = meta["C_out"]
    x_min = meta["x_min"]
    x_max = meta["x_max"]

    print(f"Dataset loaded from: {data_path}")
    print(f"  N_total: {meta['N_total']} (train={meta['N_train']}, val={meta['N_val']})")
    print(f"  C_in = {C_in}, C_out = {C_out}")

    # === Target statistics (aligned with the FNO baseline) ===
    target_stats = compute_target_stats(data_path)

    # === Model (architecture matched to the FNO baseline) ===
    model = FNOBarrierPINO(
        in_channels=C_in,
        out_channels=C_out,
        width=96,
        modes1=6,
        modes2=6,
        n_layers=4,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler aligned with the baseline configuration.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        verbose=True,
    )

    # === Black–Scholes physics loss (log-moneyness and T coordinates) ===
    physics_loss_fn = make_black_scholes_physics_loss(
        x_min=x_min,
        x_max=x_max,
        lambda_barrier=1.0,
    )

    # === Training loop ===
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        physics_loss_fn=physics_loss_fn,
        lambda_phys=lambda_phys,
        checkpoint_dir=checkpoint_dir,
        print_every=1,
        scheduler=scheduler,
        grad_clip=1.0,
        augment_fn=None,            # no frequency augmentation: physics term only
        target_stats=target_stats,  # used for σ-MAE reporting in logs
        normalize_loss=True,       # normalize losses by target stddev
    )


if __name__ == "__main__":
    main()
