"""
Model evaluation utilities: test-time unit identification, inference loop, and linear probing.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from jepsyn.losses import lejepa_loss
from .training import create_context_mask


def _run_probes(
    all_h: np.ndarray,
    all_is_change: np.ndarray,
    all_image_name: np.ndarray,
    all_session_ids: np.ndarray,
    stage: str,
) -> Dict[str, Any]:
    """
    Fit linear probes on pre-extracted representations.

    Probes (all: 5-fold stratified CV, StandardScaler, LogisticRegression):
        1. is_change detection    — binary,     balanced acc + AUROC  (chance = 0.500)
        2. image_name identity    — multi-class, balanced acc + macro AUROC
        3. session identity       — multi-class, balanced acc          (chance = 1/n_sessions)

    Args:
        all_h:           [N, D] representation matrix.
        all_is_change:   [N] int array (1 = change, 0 = no change).
        all_image_name:  [N] object array of image name strings (None where unavailable).
        all_session_ids: [N] int array of session IDs.
        stage:           Stage label for printing.

    Returns:
        Dict with probe results keyed by probe name.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print(f"\n[{stage}] Linear Probe  (representation dim={all_h.shape[1]}, N={all_h.shape[0]})")
    results: Dict[str, Any] = {}

    # --- Probe 1: is_change (binary) ---
    y_change = all_is_change
    if len(np.unique(y_change)) >= 2 and int(y_change.sum()) >= 10:
        X       = StandardScaler().fit_transform(all_h)
        clf     = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        bal_acc = cross_val_score(clf, X, y_change, cv=5, scoring="balanced_accuracy")
        proba   = cross_val_predict(clf, X, y_change, cv=5, method="predict_proba")[:, 1]
        auroc   = roc_auc_score(y_change, proba)
        results["is_change"] = {
            "balanced_acc":     float(bal_acc.mean()),
            "balanced_acc_std": float(bal_acc.std()),
            "auroc":            float(auroc),
            "chance":           0.5,
            "n":                int(len(y_change)),
        }
        print(
            f"  Probe 1 — is_change        | "
            f"balanced acc: {bal_acc.mean():.3f} ± {bal_acc.std():.3f} | "
            f"AUROC: {auroc:.3f}  (chance=0.500)"
        )
    else:
        print(f"  Probe 1 — is_change        | skipped (insufficient data or single class)")

    # --- Probe 2: image_name identity (multi-class) ---
    has_image = np.array([x is not None for x in all_image_name])
    y_img_raw = all_image_name[has_image]
    X_img     = all_h[has_image]

    if len(np.unique(y_img_raw)) >= 2 and len(y_img_raw) >= 20:
        le      = LabelEncoder()
        y_img   = le.fit_transform(y_img_raw)
        n_cls   = len(le.classes_)
        X       = StandardScaler().fit_transform(X_img)
        clf     = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs",
        )
        n_folds = max(2, min(5, int(np.bincount(y_img).min())))
        cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        proba   = cross_val_predict(clf, X, y_img, cv=cv, method="predict_proba")
        from sklearn.metrics import balanced_accuracy_score
        bal_acc = np.array([
            balanced_accuracy_score(y_img[val_idx], proba[val_idx].argmax(axis=1))
            for _, val_idx in cv.split(X, y_img)
        ])
        auroc   = roc_auc_score(y_img, proba, multi_class="ovr", average="macro")
        chance  = 1.0 / n_cls
        results["image_name"] = {
            "balanced_acc":     float(bal_acc.mean()),
            "balanced_acc_std": float(bal_acc.std()),
            "auroc":            float(auroc),
            "chance":           chance,
            "n_classes":        n_cls,
            "n":                int(len(y_img)),
        }
        print(
            f"  Probe 2 — image identity   | {n_cls} classes | "
            f"balanced acc: {bal_acc.mean():.3f} ± {bal_acc.std():.3f} | "
            f"AUROC: {auroc:.3f}  (chance={chance:.3f})"
        )
    else:
        print(f"  Probe 2 — image identity   | skipped (insufficient data or classes)")

    # --- Probe 3: session identity ---
    n_sessions = len(np.unique(all_session_ids))
    if n_sessions >= 2:
        le_sid  = LabelEncoder()
        y_sid   = le_sid.fit_transform(all_session_ids)
        X       = StandardScaler().fit_transform(all_h)
        clf     = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        n_folds = max(2, min(5, int(np.bincount(y_sid).min())))
        cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        bal_acc = cross_val_score(clf, X, y_sid, cv=cv, scoring="balanced_accuracy")
        chance  = 1.0 / n_sessions
        results["session_id"] = {
            "balanced_acc":     float(bal_acc.mean()),
            "balanced_acc_std": float(bal_acc.std()),
            "chance":           chance,
            "n_sessions":       n_sessions,
            "n":                int(len(y_sid)),
        }
        print(
            f"  Probe 3 — session identity | {n_sessions} sessions | "
            f"balanced acc: {bal_acc.mean():.3f} ± {bal_acc.std():.3f}  "
            f"(chance={chance:.3f})"
        )

    return results


def evaluate_model(
    model: Any,
    test_data: Any,
    stage: str,
    mask_ratio: float = 0.5,
    test_session_ids: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    teacher_model: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Evaluate a trained model on test data.

    JEPA mode (teacher_model=None):
        When test_session_ids and config are provided, runs the full test pipeline:
            1. identify_units — adapt test-session unit embeddings (no labels)
            2. JEPA metrics   — pred_loss, cos_similarity, R², CKA (debug output)
            3. probes         — downstream decoding of stimulus content (primary result)
        When test_session_ids is None, only JEPA metrics are computed.

    SNN mode (teacher_model provided):
        Uses teacher_model's context_encoder to extract latents, passes them through
        model (the SNN), and uses mem.mean(dim=1) as the representation. Runs probes.
        Returns a metrics DataFrame with the same h_tgt/session_ids/is_change/image_name/
        stim_block columns as JEPA mode so the UMAP cell works identically.

    Args:
        model:            JEPA bundle {"context_encoder", "target_encoder", "predictor"},
                          or an SNN nn.Module when teacher_model is provided.
        test_data:        Test DataLoader.
        stage:            Stage name ("LeJEPA", "VICReg", "SNN", etc.).
        mask_ratio:       Fraction of real tokens to mask for pred_loss eval (JEPA only).
        test_session_ids: Session IDs held out from training (JEPA only).
        config:           Config dict (required when test_session_ids is provided).
        teacher_model:    JEPA bundle used to extract latents for SNN evaluation.
                          When provided, triggers SNN evaluation mode.

    Returns:
        (metrics_df, probe_results)
        metrics_df:    Per-batch DataFrame with h_tgt, session_ids, is_change,
                       image_name, stim_block columns (plus JEPA-specific metrics
                       in JEPA mode). Always returned; used by UMAP cells.
        probe_results: Dict of linear probe results, or None if probes were not run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── SNN mode ──────────────────────────────────────────────────────────────
    if teacher_model is not None:
        import snntorch

        teacher_enc = teacher_model["context_encoder"].to(device).eval()
        snn = model.to(device).eval()

        all_batches: List[Dict] = []

        with torch.no_grad():
            for batch in test_data:
                session_ids = batch["session_ids"].to(device)
                unit_ids    = batch["unit_ids"].to(device)
                time_ids    = batch["time_ids"].to(device)
                attn_mask   = batch["attention_mask"].to(device)

                z_teacher, _ = teacher_enc(session_ids, unit_ids, time_ids, attn_mask)
                snntorch.utils.reset(snn)
                spk, mem = snn(z_teacher)
                h = mem.mean(dim=1)  # [B, D]

                labels     = batch.get("labels", [])
                B          = session_ids.size(0)
                is_change  = (
                    np.array([lb["is_change"] for lb in labels], dtype=bool)
                    if labels else np.zeros(B, dtype=bool)
                )
                stim_block = (
                    np.array([lb["stimulus_block"] for lb in labels], dtype=int)
                    if labels else np.full(B, -1, dtype=int)
                )
                image_name = (
                    [lb.get("image_name") for lb in labels]
                    if labels else [None] * B
                )

                all_batches.append({
                    "h_tgt":       h.cpu().numpy(),
                    "session_ids": session_ids.cpu().numpy(),
                    "is_change":   is_change,
                    "image_name":  image_name,
                    "stim_block":  stim_block,
                })

        metrics_df = pd.DataFrame(all_batches)

        all_h           = np.vstack(metrics_df["h_tgt"].values)
        all_is_change   = np.concatenate(metrics_df["is_change"].values).astype(int)
        all_session_ids = np.concatenate(metrics_df["session_ids"].values)
        all_image_name  = np.array(
            [x for row in metrics_df["image_name"].values for x in row], dtype=object
        )

        probe_results = _run_probes(all_h, all_is_change, all_image_name, all_session_ids, stage)
        return metrics_df, probe_results

    # ── JEPA mode ─────────────────────────────────────────────────────────────

    # Step 1: adapt test-session unit embeddings.
    if test_session_ids is not None:
        if config is None:
            raise ValueError("config must be provided when test_session_ids is given")
        identify_units(model, test_data, test_session_ids, config)

    context_encoder = model["context_encoder"].to(device).eval()
    target_encoder  = model["target_encoder"].to(device).eval()
    predictor       = model["predictor"].to(device).eval()

    all_metrics = []

    def _cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Linear CKA between two [B, D] representation matrices."""
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        XXT = X @ X.T
        YYT = Y @ Y.T
        numerator   = (XXT * YYT).sum()
        denominator = torch.sqrt((XXT * XXT).sum() * (YYT * YYT).sum()) + 1e-8
        return (numerator / denominator).item()

    with torch.no_grad():
        for batch in test_data:
            session_ids = batch["session_ids"].to(device)
            unit_ids    = batch["unit_ids"].to(device)
            time_ids    = batch["time_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)

            labels = batch.get("labels", [])
            B = session_ids.size(0)
            is_change = (
                np.array([lb["is_change"] for lb in labels], dtype=bool)
                if labels else np.zeros(B, dtype=bool)
            )
            stim_block = (
                np.array([lb["stimulus_block"] for lb in labels], dtype=int)
                if labels else np.full(B, -1, dtype=int)
            )
            image_name = (
                [lb.get("image_name") for lb in labels]
                if labels else [None] * B
            )

            ctx_mask = create_context_mask(attn_mask, mask_ratio)
            Z_ctx, h_ctx = context_encoder(session_ids, unit_ids, time_ids, ctx_mask)
            _, h_tgt     = target_encoder(session_ids, unit_ids, time_ids, attn_mask)

            Z_pred = predictor(Z_ctx)
            h_pred = Z_pred.mean(dim=1)

            pred_loss      = torch.nn.functional.mse_loss(h_pred, h_tgt)
            cos_similarity = torch.nn.functional.cosine_similarity(h_ctx, h_tgt).mean()

            ss_res = ((h_pred - h_tgt) ** 2).sum()
            ss_tot = ((h_tgt - h_tgt.mean(0)) ** 2).sum()
            r2     = (1 - ss_res / (ss_tot + 1e-8)).item()

            cka_score = _cka(h_ctx, h_tgt)

            all_metrics.append(
                {
                    "stage":          stage,
                    "pred_loss":      pred_loss.item(),
                    "cos_similarity": cos_similarity.item(),
                    "r2":             r2,
                    "cka":            cka_score,
                    "h_tgt":          h_tgt.cpu().numpy(),
                    "session_ids":    session_ids.cpu().numpy(),
                    "is_change":      is_change,
                    "image_name":     image_name,
                    "stim_block":     stim_block,
                }
            )

    results_df = pd.DataFrame(all_metrics)
    print(f"\n[{stage}] JEPA Metrics (debug):")
    print(results_df.mean(numeric_only=True).to_string())

    # Step 3: linear probes.
    probe_results = None
    if test_session_ids is not None:
        probe_results = run_linear_probe(model, test_data, stage)

    return results_df, probe_results


def run_linear_probe(
    model: Dict[str, Any],
    test_data: DataLoader,
    stage: str,
) -> Dict[str, Any]:
    """
    Extract frozen target representations and run linear probes.

    Convenience wrapper: runs inference with the target encoder to collect
    representations, then delegates probe fitting to _run_probes.

    Args:
        model:     Model bundle {"context_encoder", "target_encoder", "predictor"}.
        test_data: Test DataLoader (must have include_labels=True).
        stage:     Stage label for printing.

    Returns:
        Dict with probe results keyed by probe name.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model["target_encoder"].to(device).eval()

    all_h, all_is_change, all_image_name, all_session_ids = [], [], [], []

    with torch.no_grad():
        for batch in test_data:
            session_ids = batch["session_ids"].to(device)
            unit_ids    = batch["unit_ids"].to(device)
            time_ids    = batch["time_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)

            _, h = encoder(session_ids, unit_ids, time_ids, attn_mask)

            labels = batch.get("labels", [])
            B = session_ids.size(0)
            is_change = (
                np.array([lb["is_change"] for lb in labels], dtype=bool)
                if labels else np.zeros(B, dtype=bool)
            )
            image_name = (
                [lb.get("image_name") for lb in labels]
                if labels else [None] * B
            )

            all_h.append(h.cpu().numpy())
            all_is_change.append(is_change)
            all_image_name.extend(image_name)
            all_session_ids.append(session_ids.cpu().numpy())

    all_h           = np.vstack(all_h)
    all_is_change   = np.concatenate(all_is_change).astype(int)
    all_session_ids = np.concatenate(all_session_ids)
    all_image_name  = np.array(all_image_name, dtype=object)

    return _run_probes(all_h, all_is_change, all_image_name, all_session_ids, stage)


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
    train_cfg  = config.get("training_config", {})
    n_steps    = train_cfg.get("unit_id_steps", 200)
    lr         = train_cfg.get("unit_id_lr", 1e-3)
    mask_ratio = train_cfg.get("mask_ratio", 0.5)
    reg_type   = train_cfg.get("reg_type", "sigreg").lower()
    lambd      = train_cfg.get("lambd", 0.05)
    num_slices = train_cfg.get("num_slices", 256)
    vic_sim    = train_cfg.get("vic_sim", 25.0)
    vic_std    = train_cfg.get("vic_std", 25.0)
    vic_cov    = train_cfg.get("vic_cov", 1.0)

    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_encoder = model["context_encoder"].to(device)
    target_encoder  = model["target_encoder"].to(device)
    predictor       = model["predictor"].to(device)

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

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    context_encoder.train()
    target_encoder.eval()
    predictor.eval()

    total_params = sum(p.numel() for p in params_to_optimize)
    print(
        f"\n[Unit ID] Adapting {len(test_session_ids)} test-session unit tables "
        f"({total_params:,} params) for {n_steps} steps  lr={lr}"
    )

    step = 0
    global_step = 0

    while step < n_steps:
        for batch in test_data:
            if step >= n_steps:
                break

            session_ids_t = batch["session_ids"].to(device)
            unit_ids_t    = batch["unit_ids"].to(device)
            time_ids_t    = batch["time_ids"].to(device)
            attn_mask     = batch["attention_mask"].to(device)

            ctx_mask = create_context_mask(attn_mask, mask_ratio)

            with torch.no_grad():
                _, h_tgt = target_encoder(session_ids_t, unit_ids_t, time_ids_t, attn_mask)

            Z_ctx, h_ctx = context_encoder(session_ids_t, unit_ids_t, time_ids_t, ctx_mask)
            Z_pred = predictor(Z_ctx)
            h_pred = Z_pred.mean(dim=1)

            total_loss, pred_loss, _ = lejepa_loss(
                h_ctx, h_tgt, h_pred, global_step,
                reg_type, lambd, num_slices, vic_sim, vic_std, vic_cov,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            step        += 1
            global_step += 1

            if step % 50 == 0 or step == n_steps:
                print(
                    f"  step {step:3d}/{n_steps} | "
                    f"loss={total_loss.item():.4f} | pred={pred_loss.item():.4f}"
                )

    # Sync adapted unit embeddings → target_encoder.
    print("  Syncing adapted unit embeds → target encoder...")
    with torch.no_grad():
        for sid in test_session_ids:
            sid_str = str(sid)
            src = context_encoder.encoder.unit_embeds[sid_str]
            tgt = target_encoder.encoder.unit_embeds[sid_str]
            for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
                p_tgt.copy_(p_src)

    context_encoder.eval()
    print("[Unit ID] Done.\n")
