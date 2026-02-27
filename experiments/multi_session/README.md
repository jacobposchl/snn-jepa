# Multi-session experiment pipeline (step 2)

This is the **experiment pipeline**: it loads a **pre-built windowed dataset** (Parquet from the dataset pipeline), splits by session into train/val/test, then runs LeJEPA training followed by SNN distillation. It does **not** extract sessions or process raw Allen data — that is done once in the [dataset pipeline](../data/README.md).

## Prerequisite

Run the **dataset pipeline** first so that a windowed Parquet exists:

```bash
python -m experiments.data.create_dataset path/to/dataset_config.yaml
```

Optionally verify the Parquet before training:

```bash
python -m experiments.data.verify_dataset path/to/dataset.parquet
```

Your experiment config's `data_path` must point to that Parquet (e.g. `./datasets/visual_cortex_windows.parquet`).

## Running the experiment

From the repo root:

```bash
python -m experiments.multi_session.multi_session path/to/experiment_config.yaml
```

The script will:

1. Verify the config (`verify_config`).
2. Load the Parquet from `data_path` and validate schema and integrity.
3. Split windows by `session_id` into train / val / test sets.
4. Train the **LeJEPA teacher** (context encoder + target encoder via EMA + predictor).
5. Save the LeJEPA checkpoint to `<results_out_path>/lejepa_checkpoint.pt`.
6. Save training curves and evaluate LeJEPA on the test set (pred loss, cosine similarity, UMAP latent space plot).
7. *(Planned)* Distill into a **Spiking Neural Network** student and evaluate. Currently skipped — `distill_snn` is not yet implemented.

Results are written under `<results_out_path>/<stage>/<phase>/` (e.g. `results/LeJEPA/training/`, `results/LeJEPA/test/`).

## Config fields

Use `configs/lejepa_lif_visual_cortex.yaml` as your starting template. The fields are:

| Field | Required | Description |
|---|---|---|
| `data_path` | Yes | Path to Parquet from dataset pipeline |
| `results_out_path` | No | Directory for checkpoints, CSVs, and plots |
| `data.train_split` | No | Fraction of sessions for training (default 0.7) |
| `data.val_split` | No | Fraction of sessions for validation (default 0.15) |
| `data.test_split` | No | Fraction of sessions for test (default 0.15) |
| `data.random_state` | No | RNG seed for the session split (default 42) |
| `model_config.*` | Yes | Encoder and predictor hyperparameters (see below) |
| `training_config.*` | Yes | Training hyperparameters (see below) |

### Model config

```yaml
model_config:
  # Encoder (PerceiverEncoder)
  encoder_type: perceiver         # only 'perceiver' currently supported
  d_model: 256                    # embedding and latent dimension
  n_latents: 64                   # number of latent slots (L)
  max_time_ms: 400                # time embedding vocab size — must be >= window_size_ms
  n_cross_attn_heads: 4
  n_self_attn_layers: 4
  n_self_attn_heads: 8
  dim_feedforward: 1024
  dropout: 0.1

  # Predictor (intentionally narrower than encoder)
  predictor_type: transformer     # only 'transformer' currently supported
  predictor_n_layers: 2           # encoder has 4
  predictor_n_heads: 4            # encoder has 8
  predictor_dim_feedforward: 512  # encoder uses 1024
```

> **Important**: `max_time_ms` must be **at least as large as `window_size_ms`** used when building the dataset. The encoder uses it as a time-embedding vocabulary size; a mismatch causes an index error at runtime.

### Training config

```yaml
training_config:
  epochs: 100
  batch_size: 32
  lr: 1.0e-4
  weight_decay: 0.05
  ema_momentum: 0.996   # target encoder EMA decay; higher = slower update
  mask_ratio: 0.5       # fraction of real tokens hidden from the context encoder
  lambd: 0.05           # weight of SIGReg regularization relative to prediction loss
  num_slices: 256       # number of random projections for SIGReg
```

## LeJEPA checkpoint

After training, the checkpoint is saved to:

```
<results_out_path>/lejepa_checkpoint.pt
```

It contains:

```python
{
    "context_encoder": context_encoder.state_dict(),
    "target_encoder":  target_encoder.state_dict(),
    "predictor":       predictor.state_dict(),
    "config":          config,   # full experiment config dict
}
```

To load it later for SNN distillation or analysis:

```python
ckpt = torch.load("results/lejepa_checkpoint.pt", map_location="cpu")
context_encoder.load_state_dict(ckpt["context_encoder"])
```

## Output structure

```
<results_out_path>/
  lejepa_checkpoint.pt
  LeJEPA/
    training/
      metrics.csv              # epoch, train_loss, train_pred_loss, train_reg_loss, val_loss
      training_curves.png      # total loss + pred vs reg loss across epochs
    test/
      metrics.csv              # pred_loss, cos_similarity per batch
      test_metrics.png         # bar chart of mean test metrics
      latent_space.png         # UMAP of context latents colored by session (if umap-learn installed)
  SNN/                         # populated once distill_snn is implemented
    distillation/
      metrics.csv
      distillation_curves.png
    test/
      metrics.csv
      test_metrics.png
```

## Config templates

| File | When to use |
|---|---|
| `configs/lejepa_lif_visual_cortex.yaml` | Full template with all encoder/predictor/training fields |
| `configs/experiment_from_dataset.yaml` | Minimal placeholder if you just want to verify the script runs |

## Future: torch_brain

[torch_brain](https://github.com/neuro-galaxy/torch_brain) will be integrated for multi-recording training, optimized data loading, and models (e.g. POYO). The experiment pipeline will then consume the same pre-built Parquet dataset format.
