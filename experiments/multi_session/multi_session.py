"""
Full Runner Script for a Multi-Session Experiment
With Training of LeJEPA Model and Distillation into
a Spiking Neural Network Model
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import snntorch
import torch
from snntorch import surrogate
from torch.utils.data import DataLoader

from jepsyn.losses import lejepa_loss
from jepsyn.losses.distillation import DistillationLoss
from jepsyn.models import NeuralEncoder, NeuralPredictor
from jepsyn.utils import (
    apply_unit_dropout,
    create_context_mask,
    evaluate_model,
    identify_units,
    load_and_prepare_data,
    run_linear_probe,
    save_results,
    update_ema,
    verify_config,
)


def train_lejepa(
    config: Dict[str, Any],
    train_data: DataLoader,
    val_data: DataLoader,
    unit_maps: Dict[int, Dict[int, int]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train the LeJEPA teacher model on multi-session neural data.

    Architecture:
        context_encoder  (online, gradients flow) — sees masked spike events
        target_encoder   (EMA copy, no gradients) — sees all spike events
        predictor        (narrow Transformer)     — maps Z_ctx → Z_pred ≈ Z_tgt

    Loss per batch:
        pred_loss = MSE(h_pred, h_tgt)             on mean-pooled [B, D] representations
        reg_loss  = SIGReg(h_ctx) + SIGReg(h_tgt) on mean-pooled [B, D] representations
        total     = (1 - lambd) * pred_loss + lambd * reg_loss

    The optimizer updates only context_encoder + predictor.
    The target encoder is updated via EMA after every step.

    Args:
        config     : Validated config dict (from verify_config).
        train_data : Training DataLoader.
        val_data   : Validation DataLoader.
        unit_maps  : {session_id: {raw_unit_id: 1-indexed idx}} from load_and_prepare_data.

    Returns:
        (model_bundle, metrics_df)
        model_bundle: {"context_encoder", "target_encoder", "predictor"}
        metrics_df columns: epoch, train_loss, train_pred_loss, train_reg_loss, val_loss
    """
    model_cfg = config.get("model_config", {})
    train_cfg = config.get("training_config", {})

    # Encoder hyperparameters
    d_model = model_cfg.get("d_model", 256)
    n_latents = model_cfg.get("n_latents", 64)
    window_size_s = model_cfg.get("window_size_s", 0.4)
    encoder_type = model_cfg.get("encoder_type", "perceiver")
    encoder_kwargs = {
        k: model_cfg[k]
        for k in (
            "n_cross_attn_heads",
            "n_self_attn_layers",
            "n_self_attn_heads",
            "dim_feedforward",
            "dropout",
            "rope_t_min",
            "rope_t_max",
            "use_delimiter_tokens",
        )
        if k in model_cfg
    }

    # Predictor hyperparameters
    predictor_type = model_cfg.get("predictor_type", "transformer")
    predictor_kwargs: Dict[str, Any] = {}
    if "predictor_n_layers" in model_cfg:
        predictor_kwargs["n_layers"] = model_cfg["predictor_n_layers"]
    if "predictor_n_heads" in model_cfg:
        predictor_kwargs["n_heads"] = model_cfg["predictor_n_heads"]
    if "predictor_dim_feedforward" in model_cfg:
        predictor_kwargs["dim_feedforward"] = model_cfg["predictor_dim_feedforward"]

    # Training hyperparameters
    n_epochs = train_cfg.get("epochs", 100)
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.05)
    ema_momentum = train_cfg.get("ema_momentum", 0.996)
    mask_ratio = train_cfg.get("mask_ratio", 0.5)
    lambd = train_cfg.get("lambd", 0.05)
    num_slices = train_cfg.get("num_slices", 256)
    reg_type = train_cfg.get("reg_type", "sigreg").lower()
    vic_sim = train_cfg.get("vic_sim", 25.0)
    vic_std = train_cfg.get("vic_std", 25.0)
    vic_cov = train_cfg.get("vic_cov", 1.0)
    unit_dropout = train_cfg.get("unit_dropout", 0.0)
    results_path = config.get("results_out_path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Context encoder: online network, gradients flow through this.
    context_encoder = NeuralEncoder(
        session_unit_maps=unit_maps,
        d_model=d_model,
        n_latents=n_latents,
        window_size_s=window_size_s,
        encoder_type=encoder_type,
        **encoder_kwargs,
    ).to(device)

    # Target encoder: EMA copy, never updated by the optimizer.
    target_encoder = copy.deepcopy(context_encoder)
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    # Predictor: narrow Transformer mapping context latents → predicted target latents.
    predictor = NeuralPredictor(
        d_model=d_model,
        predictor_type=predictor_type,
        **predictor_kwargs,
    ).to(device)

    # Optimizer: context encoder + predictor only.
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    all_metrics = []
    global_step = 0

    for epoch in range(n_epochs):
        # ---- Training ----
        context_encoder.train()
        predictor.train()

        train_loss_sum = train_pred_sum = train_reg_sum = 0.0
        n_train = 0

        for batch in train_data:
            session_ids = batch["session_ids"].to(device)
            unit_ids = batch["unit_ids"].to(device)
            time_ids = batch["time_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            # Hide mask_ratio of real tokens from the context encoder.
            ctx_mask = create_context_mask(attn_mask, mask_ratio)

            # Unit dropout: randomly drop a fraction of units per sample so the
            # model learns to be robust to missing neurons (POYO augmentation).
            if unit_dropout > 0.0:
                ctx_mask = apply_unit_dropout(
                    unit_ids, ctx_mask, dropout_ratio=unit_dropout
                )

            # Context encoder: sees only unmasked events, gradients flow.
            Z_ctx, h_ctx = context_encoder(session_ids, unit_ids, time_ids, ctx_mask)

            # Target encoder: sees all events, EMA weights, no gradients.
            with torch.no_grad():
                _, h_tgt = target_encoder(session_ids, unit_ids, time_ids, attn_mask)

            # Predictor: Z_ctx [B, L, D] → Z_pred [B, L, D].
            Z_pred = predictor(Z_ctx)
            h_pred = Z_pred.mean(dim=1)  # [B, D]

            loss, pred_loss, reg_loss = lejepa_loss(
                h_ctx,
                h_tgt,
                h_pred,
                global_step,
                reg_type,
                lambd,
                num_slices,
                vic_sim,
                vic_std,
                vic_cov,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA: momentum-weighted update of target encoder from context encoder.
            update_ema(context_encoder, target_encoder, ema_momentum)

            train_loss_sum += loss.item()
            train_pred_sum += pred_loss.item()
            train_reg_sum += reg_loss.item()
            n_train += 1
            global_step += 1

        # ---- Validation ----
        context_encoder.eval()
        predictor.eval()

        val_loss_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_data:
                session_ids = batch["session_ids"].to(device)
                unit_ids = batch["unit_ids"].to(device)
                time_ids = batch["time_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)

                ctx_mask = create_context_mask(attn_mask, mask_ratio)
                Z_ctx, h_ctx = context_encoder(
                    session_ids, unit_ids, time_ids, ctx_mask
                )
                _, h_tgt = target_encoder(session_ids, unit_ids, time_ids, attn_mask)
                Z_pred = predictor(Z_ctx)
                h_pred = Z_pred.mean(dim=1)  # [B, D]

                loss, _, _ = lejepa_loss(
                    h_ctx,
                    h_tgt,
                    h_pred,
                    global_step,
                    reg_type,
                    lambd,
                    num_slices,
                    vic_sim,
                    vic_std,
                    vic_cov,
                )

                val_loss_sum += loss.item()
                n_val += 1

        train_loss_avg = train_loss_sum / max(n_train, 1)
        val_loss_avg = val_loss_sum / max(n_val, 1)

        all_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "train_pred_loss": train_pred_sum / max(n_train, 1),
                "train_reg_loss": train_reg_sum / max(n_train, 1),
                "val_loss": val_loss_avg,
            }
        )

        avg_pred = train_pred_sum / max(n_train, 1)
        avg_reg = train_reg_sum / max(n_train, 1)
        w_pred = (1 - lambd) * avg_pred
        w_reg = lambd * avg_reg
        pred_pct = 100 * w_pred / max(train_loss_avg, 1e-9)
        reg_pct = 100 * w_reg / max(train_loss_avg, 1e-9)
        print(
            f"Epoch {epoch:3d} | train={train_loss_avg:.4f} | val={val_loss_avg:.4f}"
            f" | pred={avg_pred:.4f} ({pred_pct:.1f}%) | reg={avg_reg:.4f} ({reg_pct:.1f}%)"
        )

    # ---- Checkpoint ----
    if results_path:
        ckpt_dir = Path(results_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "lejepa_checkpoint.pt"
        torch.save(
            {
                "context_encoder": context_encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "config": config,
                "unit_maps": unit_maps,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")

    return (
        {
            "context_encoder": context_encoder,
            "target_encoder": target_encoder,
            "predictor": predictor,
        },
        pd.DataFrame(all_metrics),
    )


def train_mae(
    config: Dict[str, Any],
    train_data: DataLoader,
    val_data: DataLoader,
    unit_maps: Dict[int, Dict[int, int]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train a MAE (Masked Autoencoder) model on multi-session neural data.

    Architecture:
        encoder  (NeuralEncoder, same as LeJEPA context encoder)
        decoder  (MAEDecoder: cross-attends over latents to reconstruct masked spike embeddings)

    Loss:
        MSE reconstruction loss at masked spike token positions only.

    Unlike LeJEPA, there is no EMA target encoder or latent-space predictor.
    The model learns by reconstructing original spike embeddings at masked positions.

    Args:
        config     : Validated config dict (from verify_config).
        train_data : Training DataLoader.
        val_data   : Validation DataLoader.
        unit_maps  : {session_id: {raw_unit_id: 1-indexed idx}} from load_and_prepare_data.

    Returns:
        (model_bundle, metrics_df)
        model_bundle: {"encoder", "decoder"}
        metrics_df columns: epoch, train_loss, val_loss
    """
    from jepsyn.models import MAEDecoder

    model_cfg = config.get("model_config", {})
    train_cfg = config.get("training_config", {})

    # Encoder hyperparameters (same as LeJEPA)
    d_model = model_cfg.get("d_model", 256)
    n_latents = model_cfg.get("n_latents", 64)
    window_size_s = model_cfg.get("window_size_s", 0.4)
    encoder_type = model_cfg.get("encoder_type", "perceiver")
    encoder_kwargs = {
        k: model_cfg[k]
        for k in (
            "n_cross_attn_heads",
            "n_self_attn_layers",
            "n_self_attn_heads",
            "dim_feedforward",
            "dropout",
            "rope_t_min",
            "rope_t_max",
            "use_delimiter_tokens",
        )
        if k in model_cfg
    }

    # MAE decoder hyperparameters
    mae_n_layers = model_cfg.get("mae_decoder_n_layers", 2)
    mae_n_heads = model_cfg.get("mae_decoder_n_heads", 4)
    mae_dim_ff = model_cfg.get("mae_decoder_dim_feedforward", 512)
    mae_dropout = model_cfg.get("dropout", 0.1)

    # Training hyperparameters
    n_epochs = train_cfg.get("epochs", 100)
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.05)
    mask_ratio = train_cfg.get("mask_ratio", 0.75)
    unit_dropout = train_cfg.get("unit_dropout", 0.0)
    results_path = config.get("results_out_path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MAE on {device}")

    # Encoder
    encoder = NeuralEncoder(
        session_unit_maps=unit_maps,
        d_model=d_model,
        n_latents=n_latents,
        window_size_s=window_size_s,
        encoder_type=encoder_type,
        **encoder_kwargs,
    ).to(device)

    print(encoder)

    # MAE Decoder
    decoder = MAEDecoder(
        d_model=d_model,
        n_heads=mae_n_heads,
        n_layers=mae_n_layers,
        dim_feedforward=mae_dim_ff,
        dropout=mae_dropout,
    ).to(device)

    # Optimizer: both encoder and decoder are trained end-to-end
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    all_metrics = []

    for epoch in range(n_epochs):
        # ---- Training ----
        encoder.train()
        decoder.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_data:
            session_ids = batch["session_ids"].to(device)
            unit_ids = batch["unit_ids"].to(device)
            time_ids = batch["time_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            # Context mask: True = visible to encoder, False = masked
            ctx_mask = create_context_mask(attn_mask, mask_ratio)

            if unit_dropout > 0.0:
                ctx_mask = apply_unit_dropout(
                    unit_ids, ctx_mask, dropout_ratio=unit_dropout
                )

            # Encoder sees only visible (unmasked) tokens
            Z, _ = encoder(session_ids, unit_ids, time_ids, ctx_mask)

            # Build reconstruction targets: original unit embeddings before masking.
            # We need the raw unit embeddings — extract them from the encoder directly.
            with torch.no_grad():
                x_target = torch.zeros(
                    session_ids.shape[0],
                    unit_ids.shape[1],
                    d_model,
                    device=device,
                    dtype=torch.float32,
                )
                for sid_val in session_ids.unique():
                    sid_str = str(sid_val.item())
                    mask = session_ids == sid_val
                    x_target[mask] = encoder.encoder.unit_embeds[sid_str](
                        unit_ids[mask]
                    )

            # Decoder reconstructs masked positions from latents Z
            _, loss = decoder(Z, x_target, ctx_mask, attn_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_train += 1

        # ---- Validation ----
        encoder.eval()
        decoder.eval()
        val_loss_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_data:
                session_ids = batch["session_ids"].to(device)
                unit_ids = batch["unit_ids"].to(device)
                time_ids = batch["time_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)

                ctx_mask = create_context_mask(attn_mask, mask_ratio)
                Z, _ = encoder(session_ids, unit_ids, time_ids, ctx_mask)

                x_target = torch.zeros(
                    session_ids.shape[0],
                    unit_ids.shape[1],
                    d_model,
                    device=device,
                    dtype=torch.float32,
                )
                for sid_val in session_ids.unique():
                    sid_str = str(sid_val.item())
                    mask = session_ids == sid_val
                    x_target[mask] = encoder.encoder.unit_embeds[sid_str](
                        unit_ids[mask]
                    )

                _, loss = decoder(Z, x_target, ctx_mask, attn_mask)
                val_loss_sum += loss.item()
                n_val += 1

        train_loss_avg = train_loss_sum / max(n_train, 1)
        val_loss_avg = val_loss_sum / max(n_val, 1)

        all_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
            }
        )

        print(f"Epoch {epoch:3d} | train={train_loss_avg:.4f} | val={val_loss_avg:.4f}")

    # ---- Checkpoint ----
    if results_path:
        ckpt_dir = Path(results_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "mae_checkpoint.pt"
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "config": config,
                "unit_maps": unit_maps,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")

    return (
        {"encoder": encoder, "decoder": decoder},
        pd.DataFrame(all_metrics),
    )


def load_checkpoint(
    ckpt_path: Path,
    unit_maps: Dict[int, Dict[int, int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[int, Dict[int, int]]]:
    """
    Load a saved LeJEPA or MAE checkpoint and reconstruct the model bundle.

    The checkpoint must have been saved by train_lejepa() or train_mae().
    unit_maps are saved inside the checkpoint (for checkpoints saved after this
    change); for older checkpoints without unit_maps, pass them explicitly.

    Args:
        ckpt_path:  Path to the .pt checkpoint file.
        unit_maps:  {session_id: {raw_unit_id: 1-indexed idx}}.
                    Only needed for checkpoints that pre-date unit_maps saving.
                    If the checkpoint contains unit_maps this argument is ignored.

    Returns:
        (model_bundle, config, unit_maps)
        model_bundle: JEPA → {"context_encoder", "target_encoder", "predictor"}
                      MAE  → {"encoder", "decoder"}
    """
    from jepsyn.models import MAEDecoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    config = ckpt["config"]
    unit_maps = ckpt.get("unit_maps", unit_maps)
    if unit_maps is None:
        raise ValueError(
            "unit_maps not found in checkpoint. "
            "Pass them explicitly via load_checkpoint(ckpt_path, unit_maps=...)."
        )

    model_cfg = config.get("model_config", {})

    d_model = model_cfg.get("d_model", 256)
    n_latents = model_cfg.get("n_latents", 64)
    window_size_s = model_cfg.get("window_size_s", 0.4)
    encoder_type = model_cfg.get("encoder_type", "perceiver")
    encoder_kwargs = {
        k: model_cfg[k]
        for k in (
            "n_cross_attn_heads",
            "n_self_attn_layers",
            "n_self_attn_heads",
            "dim_feedforward",
            "dropout",
            "rope_t_min",
            "rope_t_max",
            "use_delimiter_tokens",
        )
        if k in model_cfg
    }

    is_mae = "encoder" in ckpt and "context_encoder" not in ckpt

    if is_mae:
        encoder = NeuralEncoder(
            session_unit_maps=unit_maps,
            d_model=d_model,
            n_latents=n_latents,
            window_size_s=window_size_s,
            encoder_type=encoder_type,
            **encoder_kwargs,
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        decoder = MAEDecoder(
            d_model=d_model,
            n_heads=model_cfg.get("mae_decoder_n_heads", 4),
            n_layers=model_cfg.get("mae_decoder_n_layers", 2),
            dim_feedforward=model_cfg.get("mae_decoder_dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)
        decoder.load_state_dict(ckpt["decoder"])
        decoder.eval()

        model_bundle = {"encoder": encoder, "decoder": decoder}
        print(f"Loaded MAE checkpoint from {ckpt_path}")

    else:
        predictor_type = model_cfg.get("predictor_type", "transformer")
        predictor_kwargs: Dict[str, Any] = {}
        if "predictor_n_layers" in model_cfg:
            predictor_kwargs["n_layers"] = model_cfg["predictor_n_layers"]
        if "predictor_n_heads" in model_cfg:
            predictor_kwargs["n_heads"] = model_cfg["predictor_n_heads"]
        if "predictor_dim_feedforward" in model_cfg:
            predictor_kwargs["dim_feedforward"] = model_cfg["predictor_dim_feedforward"]

        context_encoder = NeuralEncoder(
            session_unit_maps=unit_maps,
            d_model=d_model,
            n_latents=n_latents,
            window_size_s=window_size_s,
            encoder_type=encoder_type,
            **encoder_kwargs,
        ).to(device)
        context_encoder.load_state_dict(ckpt["context_encoder"])
        context_encoder.eval()

        target_encoder = copy.deepcopy(context_encoder)
        target_encoder.load_state_dict(ckpt["target_encoder"])
        target_encoder.eval()
        for p in target_encoder.parameters():
            p.requires_grad_(False)

        predictor = NeuralPredictor(
            d_model=d_model,
            predictor_type=predictor_type,
            **predictor_kwargs,
        ).to(device)
        predictor.load_state_dict(ckpt["predictor"])
        predictor.eval()

        model_bundle = {
            "context_encoder": context_encoder,
            "target_encoder": target_encoder,
            "predictor": predictor,
        }
        print(f"Loaded LeJEPA checkpoint from {ckpt_path}")

    return model_bundle, config, unit_maps


def distill_snn(
    config: Dict[str, Any], teacher_model: Any, train_data: Any, val_data: Any
) -> Tuple[Any, pd.DataFrame]:
    """
    Distill LeJEPA teacher into spiking neural network student.
    Includes validation during distillation.

    Args:
        config: Configuration dictionary
        teacher_model: Trained LeJEPA model bundle
        train_data: Training DataLoader
        val_data: Validation DataLoader

    Returns:
        Tuple of (trained_snn, distillation_metrics_df)
    """
    # TODO: Implement SNN distillation
    # - Initialize SNN from config
    # - Distillation training loop with validation
    # - Save checkpoints
    # - Return trained SNN and metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = config["model_config"]["d_model"]

    # 1. Initialize SNN Student
    # We use the same d_model to ensure the latent dimensions match the Teacher
    net = torch.nn.Sequential(
        torch.nn.Linear(d_model, d_model * 2),
        snntorch.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True),
        torch.nn.Linear(d_model * 2, d_model),
        snntorch.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True, output=True
        ),
    ).to(device)

    # 2. Setup Specialized Distillation Loss
    dist_loss_fn = DistillationLoss(
        latent_dim=d_model,
        cca_weight=1.0,  # High weight on manifold alignment
        homeostatic_weight=0.01,  # Keep firing rates in biological range
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    teacher_model["context_encoder"].eval()

    metrics_log = []

    # 3. Distillation Loop (Short 5-10 epoch run for the deadline)
    for epoch in range(10):
        total_cca_sim = 0
        for batch in train_data:
            with torch.no_grad():
                z_teacher, _ = teacher_model["context_encoder"](
                    batch["session_ids"].to(device),
                    batch["unit_ids"].to(device),
                    batch["time_ids"].to(device),
                    batch["attention_mask"].to(device),
                )

            # SNN Forward Pass
            spk, mem = net(z_teacher)

            # Flatten for CCA: (Batch, Time, Dim) -> (Batch*Time, Dim)
            loss, metrics = dist_loss_fn(
                mem.view(-1, d_model), z_teacher.view(-1, d_model)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cca_sim += metrics["cca_similarity"]

        avg_sim = total_cca_sim / len(train_data)
        print(f"Epoch {epoch} | CCA Similarity: {avg_sim:.4f}")
        metrics_log.append({"epoch": epoch, "cca_similarity": avg_sim})

    return net, pd.DataFrame(metrics_log)


def main(config_path: Path) -> None:
    """
    Main experiment runner that orchestrates the full pipeline.

    Args:
        config_path: Path to the configuration YAML file
    """
    print("=" * 60)
    print("Verifying Configuration")
    config = verify_config(config_path)
    print(f"Configuration Verified. Using: {config_path}")

    print("\n" + "=" * 60)
    print("Loading and Preparing Data")
    train_data, val_data, test_data, unit_maps, test_session_ids = (
        load_and_prepare_data(config)
    )
    print("Data loaded successfully")

    # (for MAE ablation) Confirm which model to run based on reg_type in config
    reg_type = config.get("training_config", {}).get("reg_type", "sigreg").lower()

    if reg_type == "none":
        print("\n" + "=" * 60)
        print("Training MAE Ablation Model")
        mae_model, mae_train_metrics = train_mae(
            config, train_data, val_data, unit_maps
        )
        save_results(
            stage="MAE", phase="training", metrics=mae_train_metrics, config=config
        )
        print("MAE training complete")

        print("\n" + "=" * 60)
        print("Unit Identification (adapting test-session unit embeddings)")
        identify_units(mae_model, test_data, test_session_ids, config)

        print("\n" + "=" * 60)
        print("Evaluating MAE on Test Set")
        _mask_ratio = config.get("training_config", {}).get("mask_ratio", 0.75)
        mae_test_metrics = evaluate_model(
            mae_model, test_data, stage="MAE", mask_ratio=_mask_ratio
        )
        save_results(stage="MAE", phase="test", metrics=mae_test_metrics, config=config)
        print("MAE evaluation complete")

        print("\n" + "=" * 60)
        print("Running Linear Probes (MAE)")
        run_linear_probe(mae_model, test_data, stage="MAE")
        print("Linear probing complete")

    else:
        if reg_type == "vicreg":
            stage_name = "VICReg"
        elif reg_type == "no_reg":
            stage_name = "LeJEPA-NoReg"
        else:
            stage_name = "LeJEPA"

        print("\n" + "=" * 60)
        # print("Training LeJEPA Teacher Model")
        print(f"Training {stage_name} Model")
        jepa_model, jepa_train_metrics = train_lejepa(
            config, train_data, val_data, unit_maps
        )
        # save_results(stage="LeJEPA", phase="training", metrics=jepa_train_metrics, config=config)
        save_results(
            stage=stage_name,
            phase="training",
            metrics=jepa_train_metrics,
            config=config,
        )
        # print("LeJEPA training complete")
        print(f"{stage_name} training complete")

        print("\n" + "=" * 60)
        print("Unit Identification (adapting test-session unit embeddings)")
        identify_units(jepa_model, test_data, test_session_ids, config)

        print("\n" + "=" * 60)
        # print("Evaluating LeJEPA on Test Set")
        print(f"Evaluating {stage_name} on Test Set")
        _mask_ratio = config.get("training_config", {}).get("mask_ratio", 0.5)
        jepa_test_metrics = evaluate_model(
            jepa_model, test_data, stage=stage_name, mask_ratio=_mask_ratio
        )
        # save_results(stage="LeJEPA", phase="test", metrics=jepa_test_metrics, config=config)
        save_results(
            stage=stage_name, phase="test", metrics=jepa_test_metrics, config=config
        )
        print(f"{stage_name} evaluation complete")

        print("\n" + "=" * 60)
        print(f"Running Linear Probes ({stage_name})")
        run_linear_probe(jepa_model, test_data, stage=stage_name)
        print("Linear probing complete")

        print("\n" + "=" * 60)
        print("Distilling into Spiking Neural Network")
        snn_result = distill_snn(config, jepa_model, train_data, val_data)
        if snn_result is not None:
            snn_model, snn_train_metrics = snn_result
            save_results(
                stage="SNN",
                phase="distillation",
                metrics=snn_train_metrics,
                config=config,
            )
            print("SNN distillation complete")

            print("\n" + "=" * 60)
            print("Evaluating Distilled SNN on Test Set")
            snn_test_metrics = evaluate_model(snn_model, test_data, stage="SNN")
            save_results(
                stage="SNN", phase="test", metrics=snn_test_metrics, config=config
            )
            print("SNN evaluation complete")
        else:
            print("SNN distillation not yet implemented; skipping.")

    print("\n" + "=" * 60)
    print("Multi-Session Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-session neural experiment with LeJEPA and SNN distillation."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to configuration YAML file for experiment settings",
    )
    args = parser.parse_args()

    print("Starting Multi-Session Experiment")
    print("=" * 60)
    main(config_path=args.config_path)
