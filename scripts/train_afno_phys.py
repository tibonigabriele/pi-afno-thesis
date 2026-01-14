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
# - The overall structure (dataset loaders + PyTorch training loop + checkpointing
#   + LR scheduling) follows standard research engineering practice and does not
#   indicate direct reuse from a specific public repository.
# - Conceptual references for context:
#   * FNO family: Li et al., 2021 (Fourier Neural Operator).
#   * AFNO-style spectral mixing: adaptive spectral filtering literature.
#   * PINO: physics-informed operator learning via PDE residual regularization.
# - The "frequency augmentation" used here is a project-internal implementation
#   (src/data/frequency_augmentation.py) inspired by frequency-domain perturbation
#   strategies commonly used in signal processing and recent ML augmentation work.
# -----------------------------------------------------------------------------

import torch
import torch.optim as optim

from src.data.dataset_barrier import load_barrier_option_dataloaders
from src.data.frequency_augmentation import fourier_augment_batch
from src.models.fno_family import AFNOBarrierPINO
from src.training.loop import train_model
from src.training.physics_loss import make_black_scholes_physics_loss
from src.utils.misc import set_seed, get_device, ensure_dir
from src.utils.data_stats import compute_target_stats


def main():
    # Reproducibility and execution device selection.
    set_seed(42)
    device = get_device()

    # Paths and training hyperparameters.
    data_path = "data/barrier_dataset.pt"
    batch_size = 256
    num_epochs = 200
    lr = 3e-4
    weight_decay = 1e-4
    lambda_phys = 0.1
    checkpoint_dir = "checkpoints/afno_phys_full"
    ensure_dir(checkpoint_dir)

    # Target statistics (consistent with the other training scripts).
    target_stats = compute_target_stats(data_path)

    # Data loaders.
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

    # AFNO model with physics-informed regularization (PINO-style training).
    model = AFNOBarrierPINO(
        in_channels=C_in,
        out_channels=C_out,
        width=96,
        modes1=6,
        modes2=6,
        n_layers=4,
    )

    # Optimizer (Adam) with explicit weight decay.
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Validation-driven learning-rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        verbose=True,
    )

    # Black–Scholes physics loss including barrier-related constraints.
    physics_loss_fn = make_black_scholes_physics_loss(
        x_min=x_min,
        x_max=x_max,
        lambda_barrier=1.0,
        # Keep physics loss numerically comparable with a normalized supervised loss.
        # (Scale residual and barrier penalty by the dataset price std.)
        price_scale=target_stats["std"][0],
    )

    # Frequency-domain augmentation (P-FTD-inspired): applied to inputs only.
    def augment_fn(batch_x: torch.Tensor, batch_y: torch.Tensor):
        batch_x_aug = fourier_augment_batch(
            batch_x,
            prob=0.05,
            max_scale=0.02,
        )
        return batch_x_aug, batch_y

    # Shared training loop.
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
        augment_fn=None,
        target_stats=target_stats,   # used for σ-MAE reporting in logs
        normalize_loss=True,        # normalize data loss by target stddev
    )


if __name__ == "__main__":
    main()
