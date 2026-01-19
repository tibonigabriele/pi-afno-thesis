# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Gabriele Tiboni
# MSc Thesis — Computer Engineering, University of Padua (UniPD)
#
# Rights and licensing:
# - This source file is intended for academic/research dissemination as part of a
#   Master’s thesis project.
# - If this repository includes a LICENSE file, this file is distributed under
#   those terms. Otherwise, all rights are reserved by the author.
#
# Notes on originality and references:
# - The overall training-script structure follows standard PyTorch patterns.
# - Neural Operators references:
#   * FNO: Li et al., 2021.
#   * AFNO-like: adaptive/gated spectral variants used in recent literature.
# - Optional input augmentation inspired by padding-based Fourier denoising /
#   masking strategies (P-FTD-like), applied to inputs only.
# -----------------------------------------------------------------------------

import torch
import torch.optim as optim

from src.data.augment_pftd import pftd_augment
from src.data.dataset_barrier import load_barrier_option_dataloaders
from src.models.fno_family import FNOBarrier, FNOBarrierPINO, AFNOBarrierPINO
from src.training.loop import train_model
from src.training.physics_loss import make_black_scholes_physics_loss
from src.utils.data_stats import compute_target_stats
from src.utils.misc import ensure_dir, get_device, set_seed


def make_pftd_augment_fn(
    keep_fraction: float = 0.6,
    noise_std: float = 0.01,
    pad_size: int = 4,
    pad_mode: str = "mirror",
):
    # signature (x, y) -> (x_aug, y)
    def augment_fn(batch_x: torch.Tensor, batch_y: torch.Tensor):
        B = batch_x.shape[0]
        augmented = []
        for b in range(B):
            augmented.append(
                pftd_augment(
                    batch_x[b],  # (C, H, W)
                    keep_fraction=keep_fraction,
                    noise_std=noise_std,
                    pad_size=pad_size,
                    pad_mode=pad_mode,
                )
            )
        return torch.stack(augmented, dim=0), batch_y

    return augment_fn


def run_one(
    exp_name: str,
    model_class,
    use_pftd: bool,
    use_physics: bool,
    *,
    data_path: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    lambda_phys: float,
):
    set_seed(42)
    device = get_device()

    ckpt_dir = f"checkpoints/ablation_{exp_name}"
    ensure_dir(ckpt_dir)

    train_loader, val_loader, meta = load_barrier_option_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=0.1,
        shuffle=True,
    )

    C_in = meta["C_in"]
    C_out = meta["C_out"]
    x_min = meta.get("x_min")
    x_max = meta.get("x_max")

    model = model_class(
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

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
    )

    augment_fn = make_pftd_augment_fn() if use_pftd else None

    target_stats = compute_target_stats(data_path)
    physics_loss_fn = None
    lam = 0.0

    if use_physics:
        if x_min is None or x_max is None:
            raise RuntimeError("meta['x_min'/'x_max'] missing: required for physics loss.")
        physics_loss_fn = make_black_scholes_physics_loss(
            x_min=x_min,
            x_max=x_max,
            lambda_barrier=1.0,
            price_scale=target_stats["std"][0],
        )
        lam = float(lambda_phys)

    print(
        f"\n=== {exp_name} | model={model_class.__name__} | pftd={use_pftd} | "
        f"physics={use_physics} (lambda={lam}) ==="
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        physics_loss_fn=physics_loss_fn,
        lambda_phys=lam,
        checkpoint_dir=ckpt_dir,
        print_every=1,
        scheduler=scheduler,
        grad_clip=1.0,
        augment_fn=augment_fn,
        target_stats=target_stats,
        normalize_loss=True,
    )


def main():
    data_path = "data/barrier_dataset.pt"

    # Keep consistent with main training unless you intentionally want faster ablations
    batch_size = 256
    num_epochs = 200
    lr = 3e-4
    weight_decay = 1e-4

    # 8 base configs (then repeated with PFTD)
    base_experiments = [
        # --- FNO supervised ---
        dict(exp_name="fno_plain", model_class=FNOBarrier, use_physics=False, lambda_phys=0.0),
        # --- FNO PINO lambdas ---
        dict(exp_name="fno_pino_lam0", model_class=FNOBarrierPINO, use_physics=True, lambda_phys=0.01),
        dict(exp_name="fno_pino", model_class=FNOBarrierPINO, use_physics=True, lambda_phys=0.1),
        dict(exp_name="fno_pino_lam1", model_class=FNOBarrierPINO, use_physics=True, lambda_phys=1.0),
        # --- AFNO supervised (no-physics) ---
        dict(exp_name="afno_no_phys", model_class=AFNOBarrierPINO, use_physics=False, lambda_phys=0.0),
        # --- AFNO PINO lambdas ---
        dict(exp_name="afno_phys_lam0", model_class=AFNOBarrierPINO, use_physics=True, lambda_phys=0.01),
        dict(exp_name="afno_phys", model_class=AFNOBarrierPINO, use_physics=True, lambda_phys=0.1),
        dict(exp_name="afno_phys_lam1", model_class=AFNOBarrierPINO, use_physics=True, lambda_phys=1.0),
    ]

    # --- 8 WITHOUT PFTD ---
    for cfg in base_experiments:
        run_one(
            exp_name=cfg["exp_name"],
            model_class=cfg["model_class"],
            use_pftd=False,
            use_physics=cfg["use_physics"],
            data_path=data_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            lambda_phys=cfg["lambda_phys"],
        )

    # --- 8 WITH PFTD ---
    for cfg in base_experiments:
        run_one(
            exp_name=f'{cfg["exp_name"]}_pftd',
            model_class=cfg["model_class"],
            use_pftd=True,
            use_physics=cfg["use_physics"],
            data_path=data_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            lambda_phys=cfg["lambda_phys"],
        )


if __name__ == "__main__":
    main()
