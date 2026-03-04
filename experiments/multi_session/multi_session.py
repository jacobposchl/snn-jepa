"""
Full Runner Script for a Multi-Session Experiment
With Training of LeJEPA Model and Distillation into
a Spiking Neural Network Model
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from jepsyn.data import REQUIRED_COLUMNS, SpikeWindowDataset, spike_collate_fn
from jepsyn.losses import lejepa_loss
from jepsyn.models import NeuralEncoder, NeuralPredictor
from jepsyn.utils import (apply_unit_dropout, create_context_mask, update_ema,
                          verify_config)


def load_and_prepare_data(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, Dict[int, int]], List[int]]:
    """
    Load the spike-window parquet, validate it, split by session, and return DataLoaders.

    Each DataLoader yields batches from spike_collate_fn:
        session_ids    LongTensor [B]
        unit_ids       LongTensor [B, max_E]  — 1-indexed contiguous unit idx (0 = PAD)
        time_ids       LongTensor [B, max_E]  — floor(ms offset), clipped to window
        attention_mask BoolTensor [B, max_E]  — True = real token, False = padding
        labels         list[dict]             — test loader only, flattened stimulus fields

    Args:
        config: Validated configuration dict (from verify_config).

    Returns:
        (train_loader, val_loader, test_loader, session_unit_maps, test_session_ids)
        session_unit_maps: {session_id: {raw_unit_id: 1-indexed contiguous idx}}
            Needed by the training function to size per-session embedding tables.
    """
    data_path = config.get("data_path")
    if not data_path:
        raise ValueError("data_path not found in configuration")

    dataset = pd.read_parquet(data_path, engine="pyarrow")

    # Column validation
    missing = [c for c in REQUIRED_COLUMNS if c not in dataset.columns]
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")

    print("Validating dataset integrity...")

    # No duplicate window_ids with conflicting timestamps
    dup = dataset.groupby("window_id").agg(
        {"window_start_ms": "nunique", "window_end_ms": "nunique"}
    )
    conflicts = dup[(dup["window_start_ms"] > 1) | (dup["window_end_ms"] > 1)]
    if not conflicts.empty:
        raise ValueError(
            f"Found {len(conflicts)} window_ids with conflicting timestamps: "
            f"{conflicts.index.tolist()[:5]}"
        )

    # events_units and events_times_ms must have equal length per row
    mismatches = dataset[
        dataset["events_units"].apply(len) != dataset["events_times_ms"].apply(len)
    ]
    if not mismatches.empty:
        raise ValueError(
            f"Found {len(mismatches)} rows where events_units and events_times_ms "
            f"have different lengths (window_ids: {mismatches['window_id'].tolist()[:5]})"
        )

    print("Passed basic validation checks.")

    # Build per-session unit maps from the full dataset before splitting.
    # Maps raw AllenSDK unit IDs → 1-indexed contiguous indices; 0 is reserved for PAD.
    session_unit_maps: Dict[int, Dict[int, int]] = {}
    for sid, grp in dataset.groupby("session_id"):
        all_units = sorted({int(u) for arr in grp["events_units"] for u in arr})
        session_unit_maps[int(sid)] = {
            raw: idx + 1 for idx, raw in enumerate(all_units)
        }
    print(
        f"Built unit maps for {len(session_unit_maps)} sessions "
        f"(sizes: {[len(m) for m in session_unit_maps.values()]})"
    )

    # Session-level train / val / test split
    data_cfg = config.get("data", {})
    train_size = data_cfg.get("train_split", 0.7)
    val_size = data_cfg.get("val_split", 0.15)
    test_size = data_cfg.get("test_split", 0.15)
    random_state = data_cfg.get("random_state", 42)

    unique_sessions = dataset["session_id"].unique()
    train_val_sessions, test_sessions = train_test_split(
        unique_sessions, test_size=test_size, random_state=random_state
    )
    train_sessions, val_sessions = train_test_split(
        train_val_sessions,
        test_size=val_size / (train_size + val_size),
        random_state=random_state,
    )

    train_df = dataset[dataset["session_id"].isin(train_sessions)]
    val_df = dataset[dataset["session_id"].isin(val_sessions)]
    test_df = dataset[dataset["session_id"].isin(test_sessions)]

    print(f"Train: {len(train_df)} windows ({len(train_sessions)} sessions)")
    print(f"Val:   {len(val_df)} windows ({len(val_sessions)} sessions)")
    print(f"Test:  {len(test_df)} windows ({len(test_sessions)} sessions)")

    batch_size = config.get("training_config", {}).get("batch_size", 32)
    has_stimulus = "stimulus" in dataset.columns

    train_loader = DataLoader(
        SpikeWindowDataset(train_df, session_unit_maps),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=spike_collate_fn,
    )
    val_loader = DataLoader(
        SpikeWindowDataset(val_df, session_unit_maps),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=spike_collate_fn,
    )
    test_loader = DataLoader(
        SpikeWindowDataset(test_df, session_unit_maps, include_labels=has_stimulus),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=spike_collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        session_unit_maps,
        sorted(test_sessions.tolist()),
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
    mae_n_layers      = model_cfg.get("mae_decoder_n_layers", 2)
    mae_n_heads       = model_cfg.get("mae_decoder_n_heads", 4)
    mae_dim_ff        = model_cfg.get("mae_decoder_dim_feedforward", 512)
    mae_dropout       = model_cfg.get("dropout", 0.1)

    # Training hyperparameters
    n_epochs     = train_cfg.get("epochs", 100)
    lr           = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.05)
    mask_ratio   = train_cfg.get("mask_ratio", 0.75)
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
            unit_ids    = batch["unit_ids"].to(device)
            time_ids    = batch["time_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)

            # Context mask: True = visible to encoder, False = masked
            ctx_mask = create_context_mask(attn_mask, mask_ratio)

            if unit_dropout > 0.0:
                ctx_mask = apply_unit_dropout(unit_ids, ctx_mask, dropout_ratio=unit_dropout)

            # Encoder sees only visible (unmasked) tokens
            Z, _ = encoder(session_ids, unit_ids, time_ids, ctx_mask)

            # Build reconstruction targets: original unit embeddings before masking.
            # We need the raw unit embeddings — extract them from the encoder directly.
            with torch.no_grad():
                x_target = torch.zeros(
                    session_ids.shape[0], unit_ids.shape[1], d_model,
                    device=device, dtype=torch.float32,
                )
                for sid_val in session_ids.unique():
                    sid_str = str(sid_val.item())
                    mask    = (session_ids == sid_val)
                    x_target[mask] = encoder.encoder.unit_embeds[sid_str](unit_ids[mask])

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
                unit_ids    = batch["unit_ids"].to(device)
                time_ids    = batch["time_ids"].to(device)
                attn_mask   = batch["attention_mask"].to(device)

                ctx_mask = create_context_mask(attn_mask, mask_ratio)
                Z, _     = encoder(session_ids, unit_ids, time_ids, ctx_mask)

                x_target = torch.zeros(
                    session_ids.shape[0], unit_ids.shape[1], d_model,
                    device=device, dtype=torch.float32,
                )
                for sid_val in session_ids.unique():
                    sid_str = str(sid_val.item())
                    mask    = (session_ids == sid_val)
                    x_target[mask] = encoder.encoder.unit_embeds[sid_str](unit_ids[mask])

                _, loss = decoder(Z, x_target, ctx_mask, attn_mask)
                val_loss_sum += loss.item()
                n_val += 1

        train_loss_avg = train_loss_sum / max(n_train, 1)
        val_loss_avg   = val_loss_sum   / max(n_val,   1)

        all_metrics.append({
            "epoch":      epoch,
            "train_loss": train_loss_avg,
            "val_loss":   val_loss_avg,
        })

        print(f"Epoch {epoch:3d} | train={train_loss_avg:.4f} | val={val_loss_avg:.4f}")

    # ---- Checkpoint ----
    if results_path:
        ckpt_dir  = Path(results_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "mae_checkpoint.pt"
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "config":  config,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")

    return (
        {"encoder": encoder, "decoder": decoder},
        pd.DataFrame(all_metrics),
    )


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
    pass


def identify_units(
    model: Dict[str, Any],
    test_data: DataLoader,
    test_session_ids: List[int],
    config: Dict[str, Any],
) -> None:
    """
    Unit identification (POYO-style): adapt test-session unit embeddings using the
    self-supervised LeJEPA objective, with all shared weights frozen.

    After LeJEPA pretraining the shared encoder weights have learned a general
    "language" for neural population dynamics, but the unit embedding tables for
    test sessions were never updated (they stayed at random init). This function
    finds the right embedding vectors for test-session units by running the JEPA
    loss on test data with only those embedding tables trainable.

    No labels are needed — the self-supervised prediction objective is used as-is.

    Modifies context_encoder and target_encoder unit embeddings in-place.
    Call this between train_lejepa() and evaluate_model().

    Args:
        model            : {"context_encoder", "target_encoder", "predictor"}
        test_data        : Test DataLoader (same one passed to evaluate_model)
        test_session_ids : Session IDs that appear only in the test split
        config           : Config dict; reads training_config.{unit_id_steps,
                           unit_id_lr, mask_ratio, lambd, num_slices}
    """
    train_cfg = config.get("training_config", {})
    n_steps = train_cfg.get("unit_id_steps", 200)
    lr = train_cfg.get("unit_id_lr", 1e-3)
    mask_ratio = train_cfg.get("mask_ratio", 0.5)
    reg_type = train_cfg.get("reg_type", "sigreg").lower()
    lambd = train_cfg.get("lambd", 0.05)
    num_slices = train_cfg.get("num_slices", 256)
    vic_sim = train_cfg.get("vic_sim", 25.0)
    vic_std = train_cfg.get("vic_std", 25.0)
    vic_cov = train_cfg.get("vic_cov", 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_encoder = model["context_encoder"].to(device)
    target_encoder = model["target_encoder"].to(device)
    predictor = model["predictor"].to(device)

    # --- Freeze every parameter in all three modules ---
    for m in (context_encoder, target_encoder, predictor):
        for p in m.parameters():
            p.requires_grad_(False)

    # --- Unfreeze test-session unit embedding tables in context_encoder ---
    params_to_optimize: List[torch.Tensor] = []
    for sid in test_session_ids:
        embed_table = context_encoder.encoder.unit_embeds[str(sid)]
        for p in embed_table.parameters():
            p.requires_grad_(True)
            params_to_optimize.append(p)

    if not params_to_optimize:
        print("[Unit ID] No test-session unit embeddings found; skipping.")
        return

    # --- Also adapt session embedding rows for test sessions ---
    # Use a gradient hook to zero out updates for training-session rows,
    # so only the test-session rows are modified.
    sess_embed_weight = context_encoder.encoder.session_embed.weight
    test_sess_indices = [
        context_encoder.encoder.session_id_to_idx[sid] for sid in test_session_ids
    ]

    def _mask_sess_grad(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        for idx in test_sess_indices:
            masked[idx] = grad[idx]
        return masked

    sess_embed_weight.requires_grad_(True)
    hook_handle = sess_embed_weight.register_hook(_mask_sess_grad)
    params_to_optimize.append(sess_embed_weight)

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    context_encoder.train()
    target_encoder.eval()
    predictor.eval()

    unit_params = sum(p.numel() for p in params_to_optimize[:-1])
    sess_params = len(test_sess_indices) * context_encoder.encoder.d_model
    total_params = unit_params + sess_params
    print(
        f"\n[Unit ID] Adapting {len(test_session_ids)} test-session unit + session tables "
        f"({total_params:,} params) for {n_steps} steps  lr={lr}"
    )

    step = 0
    global_step = 0

    while step < n_steps:
        for batch in test_data:
            if step >= n_steps:
                break

            session_ids_t = batch["session_ids"].to(device)
            unit_ids_t = batch["unit_ids"].to(device)
            time_ids_t = batch["time_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            ctx_mask = create_context_mask(attn_mask, mask_ratio)

            # Target encoder: always frozen, no gradient needed
            with torch.no_grad():
                _, h_tgt = target_encoder(
                    session_ids_t, unit_ids_t, time_ids_t, attn_mask
                )

            # Context encoder: only unit embed params are trainable;
            # gradients flow through frozen cross-attn/self-attn weights back to them.
            Z_ctx, h_ctx = context_encoder(
                session_ids_t, unit_ids_t, time_ids_t, ctx_mask
            )
            Z_pred = predictor(Z_ctx)
            h_pred = Z_pred.mean(dim=1)

            total_loss, pred_loss, _ = lejepa_loss(
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
            total_loss.backward()
            optimizer.step()

            step += 1
            global_step += 1

            if step % 50 == 0 or step == n_steps:
                print(
                    f"  step {step:3d}/{n_steps} | "
                    f"loss={total_loss.item():.4f} | pred={pred_loss.item():.4f}"
                )

    # Remove gradient hook and re-freeze session embedding weight.
    hook_handle.remove()
    sess_embed_weight.requires_grad_(False)

    # Sync adapted unit embeddings and session embedding → target_encoder.
    print("  Syncing adapted unit embeds + session embed → target encoder...")
    with torch.no_grad():
        for sid in test_session_ids:
            sid_str = str(sid)
            src = context_encoder.encoder.unit_embeds[sid_str]
            tgt = target_encoder.encoder.unit_embeds[sid_str]
            for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
                p_tgt.copy_(p_src)
        # Sync the full session embedding weight (only test rows changed due to the hook).
        target_encoder.encoder.session_embed.weight.copy_(
            context_encoder.encoder.session_embed.weight
        )

    context_encoder.eval()
    print("[Unit ID] Done.\n")


def evaluate_model(
    model: Any, test_data: Any, stage: str, mask_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model bundle
        test_data: Test DataLoader
        stage: Model stage name ("LeJEPA" or "SNN")
        mask_ratio: Fraction of real tokens to mask, matching training conditions for pred_loss eval

    Returns:
        DataFrame containing per-batch evaluation metrics.
        Includes h_ctx, session_ids, is_change, stim_block arrays for downstream plotting/probing.
    """
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if stage == "LeJEPA":
        context_encoder = model["context_encoder"].to(device).eval()
        target_encoder = model["target_encoder"].to(device).eval()
        predictor = model["predictor"].to(device).eval()
    all_metrics = []

    with torch.no_grad():
        for batch in test_data:
            session_ids = batch["session_ids"].to(device)
            unit_ids = batch["unit_ids"].to(device)
            time_ids = batch["time_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            if stage == "LeJEPA":
                # Apply masking to match training conditions — gives an honest pred_loss
                ctx_mask = create_context_mask(attn_mask, mask_ratio)
                Z_ctx, h_ctx = context_encoder(
                    session_ids, unit_ids, time_ids, ctx_mask
                )
                _, h_tgt = target_encoder(session_ids, unit_ids, time_ids, attn_mask)
                Z_pred = predictor(Z_ctx)
                h_pred = Z_pred.mean(dim=1)  # [B, D]

                pred_loss = torch.nn.functional.mse_loss(h_pred, h_tgt)
                cos_similarity = torch.nn.functional.cosine_similarity(
                    h_ctx, h_tgt
                ).mean()

                # Stimulus labels (present only when test_data was built with include_labels=True)
                labels = batch.get("labels", [])
                B = session_ids.size(0)
                is_change = (
                    np.array([lb["is_change"] for lb in labels], dtype=bool)
                    if labels
                    else np.zeros(B, dtype=bool)
                )
                stim_block = (
                    np.array([lb["stimulus_block"] for lb in labels], dtype=int)
                    if labels
                    else np.full(B, -1, dtype=int)
                )

                all_metrics.append(
                    {
                        "stage": stage,
                        "pred_loss": pred_loss.item(),
                        "cos_similarity": cos_similarity.item(),
                        "h_ctx": h_ctx.cpu().numpy(),
                        "session_ids": session_ids.cpu().numpy(),
                        "is_change": is_change,
                        "stim_block": stim_block,
                    }
                )

            # if stage == "SNN": ...

    results_df = pd.DataFrame(all_metrics)
    print(f"\n[{stage}] Evaluation Metrics:")
    print(results_df.mean(numeric_only=True).to_string())

    # Linear probe: can stimulus/session structure be decoded linearly from frozen representations?
    if stage == "LeJEPA" and "is_change" in results_df.columns:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        all_h = np.vstack(results_df["h_ctx"].values)  # [N, D]
        all_change = np.concatenate(results_df["is_change"].values)  # [N] bool
        all_block = np.concatenate(results_df["stim_block"].values)  # [N] int
        all_sids = np.concatenate(results_df["session_ids"].values)  # [N] int

        # Diagnostics: always print so we can see what's in the test set
        valid = all_block >= 0
        print(f"\n[{stage}] Label diagnostics (test set):")
        print(f"  Total windows       : {len(all_change)}")
        print(
            f"  stim_block >= 0     : {int(valid.sum())}  (windows with a stimulus event)"
        )
        print(
            f"  stim_block == -1    : {int((all_block == -1).sum())}  (baseline / no stimulus event)"
        )
        print(f"  is_change=True      : {int(all_change.sum())}  (among all windows)")
        print(f"  Unique sessions     : {len(np.unique(all_sids))}")

        ran_probe = False

        # --- Probe 1: is_change (requires both True/False among stimulus windows) ---
        if valid.sum() >= 10 and len(np.unique(all_change[valid])) >= 2:
            X = StandardScaler().fit_transform(all_h[valid])
            y = all_change[valid].astype(int)
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            scores = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy")
            print(
                f"\n[{stage}] Probe 1 — is_change  |  5-fold balanced acc: "
                f"{scores.mean():.3f} ± {scores.std():.3f}  (chance = 0.500)"
            )
            ran_probe = True

        # --- Probe 2: change-event window vs baseline (requires both stim_block classes) ---
        y_stim = (all_block >= 0).astype(int)
        if not ran_probe and len(np.unique(y_stim)) >= 2 and y_stim.sum() >= 10:
            X = StandardScaler().fit_transform(all_h)
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            scores = cross_val_score(clf, X, y_stim, cv=5, scoring="balanced_accuracy")
            print(
                f"\n[{stage}] Probe 2 — change-event vs baseline  |  5-fold balanced acc: "
                f"{scores.mean():.3f} ± {scores.std():.3f}  (chance = 0.500)\n"
                f"  (is_change had only 1 class — dataset only labels change events)"
            )
            ran_probe = True

        # --- Probe 3: session identity — always available, confirms session geometry ---
        # This is the guaranteed fallback: UMAP shows clear session clusters, so a well-
        # trained encoder should decode session from the representation far above chance.
        n_sessions = len(np.unique(all_sids))
        if n_sessions >= 2:
            X = StandardScaler().fit_transform(all_h)
            y_sid = LabelEncoder().fit_transform(all_sids)
            clf = LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs"
            )
            n_folds = min(
                5, int(np.bincount(y_sid).min())
            )  # can't have more folds than min class size
            n_folds = max(n_folds, 2)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y_sid, cv=cv, scoring="balanced_accuracy")
            chance = 1.0 / n_sessions
            print(
                f"\n[{stage}] Probe 3 — session identity  |  {n_folds}-fold balanced acc: "
                f"{scores.mean():.3f} ± {scores.std():.3f}  (chance = {chance:.3f})"
            )
            if not ran_probe:
                print(
                    f"  (Probes 1 & 2 skipped — test sessions lack mixed stimulus labels.\n"
                    f"   Fix: ensure non-change trial windows have stimulus metadata in the parquet.)"
                )

    return results_df
    # when to try implementing RoPE?


def save_results(
    stage: str, phase: str, metrics: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """
    Save results, metrics, and generate plots.

    Args:
        stage: Experiment stage ("LeJEPA" or "SNN")
        phase: Training phase ("training", "validation", "test")
        metrics: DataFrame containing metrics
        config: Configuration dictionary
    """
    # TODO: Implement results output
    # - Save metrics to CSV
    # - Generate plots (training curves, latent space, etc.)
    # - Save figures to output directory

    # pull output dir from config; skip if not set
    results_path = config.get("results_out_path")
    if not results_path:
        print("No results_out_path in config; skip saving metrics.")
        return

    # create folder (results / LeJEPA /training)
    out_dir = Path(results_path) / stage / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    # Saving metrics to CSV (drop array columns used only for latent space plot)
    csv_path = out_dir / "metrics.csv"
    metrics.drop(
        columns=["h_ctx", "session_ids", "is_change", "stim_block"], errors="ignore"
    ).to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    # Generate plots (training curves, latent space, etc.)
    import matplotlib.pyplot as plt
    import numpy as np

    if phase == "training":
        # training curves
        # two subplots (total loss, pred/reg loss)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{stage} - Training Curves")
        # train_loss here
        # plot train vs val total loss across epochs
        if "train_loss" in metrics.columns:
            axes[0].plot(metrics["epoch"], metrics["train_loss"], label="train")
            axes[0].plot(metrics["epoch"], metrics["val_loss"], label="val")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Total Loss")
            axes[0].legend()
        # train_pred_loss here
        # plot pred loss vs reg loss across epochs
        if "train_pred_loss" in metrics.columns:
            axes[1].plot(
                metrics["epoch"], metrics["train_pred_loss"], label="pred loss"
            )
            axes[1].plot(metrics["epoch"], metrics["train_reg_loss"], label="reg loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Pred vs Reg Loss")
            axes[1].legend()

        # Save figures to output directory
        fig_path = out_dir / "training_curves.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved training curves to {fig_path}")

    elif phase == "test":
        # test metrics here
        # avg each metric across all test batches for single summary val
        fig, ax = plt.subplots(figsize=(6, 4))
        mean_metrics = metrics[["pred_loss", "cos_similarity"]].mean()
        ax.bar(mean_metrics.index, mean_metrics.values)
        ax.set_ylabel("Value")
        ax.set_title(f"{stage} - Mean Test Metrics")
        plt.tight_layout()
        plt.savefig(out_dir / "test_metrics.png")
        plt.close()
        print(f"Saved test metrics to {out_dir / 'test_metrics.png'}")

        # latent space plot
        # with h_ctx from evaluate_model()
        if "h_ctx" in metrics.columns:
            try:
                import umap

                # stacks all batches into 1 array [N, D]
                latent_vectors = np.vstack(metrics["h_ctx"].values)
                session_labels = np.concatenate(metrics["session_ids"].values)
                # reduce to 2D for visualization
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings2d = reducer.fit_transform(latent_vectors)
                # scatter plot colored by session id to reveal the learned structure
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(
                    embeddings2d[:, 0],
                    embeddings2d[:, 1],
                    c=session_labels,
                    cmap="tab10",
                    alpha=0.5,
                    s=10,
                )
                plt.colorbar(scatter, ax=ax, label="Session ID")
                ax.set_title(f"{stage} - Latent Space (UMAP)")
                ax.set_xlabel("DIM 1")
                ax.set_ylabel("DIM 2")
                plt.tight_layout()
                plt.savefig(out_dir / "latent_space.png")
                plt.close()
                print(f"Saved latent space plot to {out_dir / 'latent_space.png'}")

                # Second UMAP colored by is_change (stimulus windows only)
                if "is_change" in metrics.columns and "stim_block" in metrics.columns:
                    all_change = np.concatenate(metrics["is_change"].values).astype(int)
                    all_block = np.concatenate(metrics["stim_block"].values)
                    valid = all_block >= 0
                    if valid.sum() >= 10:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(
                            embeddings2d[~valid, 0],
                            embeddings2d[~valid, 1],
                            c="lightgray",
                            alpha=0.2,
                            s=8,
                            label="no stimulus",
                        )
                        for val, label, color in [
                            (0, "no change", "steelblue"),
                            (1, "change", "tomato"),
                        ]:
                            mask = valid & (all_change == val)
                            ax.scatter(
                                embeddings2d[mask, 0],
                                embeddings2d[mask, 1],
                                c=color,
                                alpha=0.6,
                                s=10,
                                label=label,
                            )
                        ax.legend()
                        ax.set_title(
                            f"{stage} - Latent Space by Change Detection (UMAP)"
                        )
                        ax.set_xlabel("DIM 1")
                        ax.set_ylabel("DIM 2")
                        plt.tight_layout()
                        plt.savefig(out_dir / "latent_space_change.png")
                        plt.close()
                        print(
                            f"Saved change UMAP to {out_dir / 'latent_space_change.png'}"
                        )

            except ImportError:
                # skip if umap-learn not installed (need to still)
                print("umap-learn not installed; skipping latent space plot.")

    elif phase == "distillation":
        # distillation training curves (total loss, pred loss vs homeostatic penalty)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{stage} - Distillation Curves")
        if "train_loss" in metrics.columns:
            axes[0].plot(metrics["epoch"], metrics["train_loss"], label="train")
            if "val_loss" in metrics.columns:
                axes[0].plot(metrics["epoch"], metrics["val_loss"], label="val")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Total Loss")
            axes[0].legend()
        if "distill_loss" in metrics.columns:
            axes[1].plot(
                metrics["epoch"], metrics["distill_loss"], label="distill loss"
            )
            if "homeo_loss" in metrics.columns:
                axes[1].plot(
                    metrics["epoch"], metrics["homeo_loss"], label="homeostatic loss"
                )
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Distill vs Homeostatic Loss")
            axes[1].legend()
        fig_path = out_dir / "distillation_curves.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved distillation curves to {fig_path}")


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
            mae_model, test_data, stage="LeJEPA", mask_ratio=_mask_ratio
        )
        save_results(stage="MAE", phase="test", metrics=mae_test_metrics, config=config)
        print("MAE evaluation complete")

    else:
        stage_name = "VICReg" if reg_type == "vicreg" else "LeJEPA"
    
        print("\n" + "=" * 60)
        # print("Training LeJEPA Teacher Model")
        print(f"Training {stage_name} Model")
        jepa_model, jepa_train_metrics = train_lejepa(
            config, train_data, val_data, unit_maps
        )
        # save_results(stage="LeJEPA", phase="training", metrics=jepa_train_metrics, config=config)
        save_results(stage=stage_name, phase="training", metrics=jepa_train_metrics, config=config)
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
            jepa_model, test_data, stage="LeJEPA", mask_ratio=_mask_ratio
        )
        # save_results(stage="LeJEPA", phase="test", metrics=jepa_test_metrics, config=config)
        save_results(stage=stage_name, phase="test", metrics=jepa_test_metrics, config=config)
        # print("LeJEPA evaluation complete")
        print(f"{stage_name} evaluation complete")

        print("\n" + "=" * 60)
        print("Distilling into Spiking Neural Network")
        snn_result = distill_snn(config, jepa_model, train_data, val_data)
        if snn_result is not None:
            snn_model, snn_train_metrics = snn_result
            save_results(
                stage="SNN", phase="distillation", metrics=snn_train_metrics, config=config
            )
            print("SNN distillation complete")

            print("\n" + "=" * 60)
            print("Evaluating Distilled SNN on Test Set")
            snn_test_metrics = evaluate_model(snn_model, test_data, stage="SNN")
            save_results(stage="SNN", phase="test", metrics=snn_test_metrics, config=config)
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
