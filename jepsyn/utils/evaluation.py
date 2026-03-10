"""
Model evaluation utilities: test-time unit identification, inference loop, and linear probing.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from jepsyn.losses import lejepa_loss
from .training import create_context_mask
from jepsyn.models.mae_ssl import MAEDecoder


def evaluate_model(
    model: Any, test_data: Any, stage: str, mask_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Evaluate a trained model on test data.

    Args:
        model:      Trained model bundle.
        test_data:  Test DataLoader.
        stage:      Model stage name ("LeJEPA" or "SNN").
        mask_ratio: Fraction of real tokens to mask, matching training conditions for pred_loss eval.

    Returns:
        DataFrame containing per-batch evaluation metrics.
        Includes h_ctx, session_ids, is_change, stim_block arrays for downstream plotting/probing.
    """
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

        all_h = np.vstack(results_df["h_ctx"].values)       # [N, D]
        all_change = np.concatenate(results_df["is_change"].values)   # [N] bool
        all_block = np.concatenate(results_df["stim_block"].values)   # [N] int
        all_sids = np.concatenate(results_df["session_ids"].values)   # [N] int

        # Diagnostics: always print so we can see what's in the test set
        valid = all_block >= 0
        print(f"\n[{stage}] Label diagnostics (test set):")
        print(f"  Total windows       : {len(all_change)}")
        print(f"  stim_block >= 0     : {int(valid.sum())}  (windows with a stimulus event)")
        print(f"  stim_block == -1    : {int((all_block == -1).sum())}  (baseline / no stimulus event)")
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
            n_folds = min(5, int(np.bincount(y_sid).min()))  # can't have more folds than min class size
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
    if "encoder" in model and "context_encoder" not in model:
        import copy
        _decoder = model["decoder"]
        model = {
            "context_encoder": model["encoder"],
            "target_encoder":  copy.deepcopy(model["encoder"]),
            "predictor":       _decoder,
        }

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
            # Z_pred = predictor(Z_ctx)
            # h_pred = Z_pred.mean(dim=1)
            if isinstance(predictor, MAEDecoder):
                Z_pred, _ = predictor(Z_ctx, Z_ctx, ctx_mask, attn_mask)
            else:
                Z_pred = predictor(Z_ctx)
            h_pred = Z_pred.mean(dim=1)

            total_loss, pred_loss, _ = lejepa_loss(
                h_ctx, h_tgt, h_pred, global_step,
                reg_type, lambd, num_slices, vic_sim, vic_std, vic_cov,
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
