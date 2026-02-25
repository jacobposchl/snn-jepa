# Multi-session experiment pipeline (step 2)

This is the **experiment pipeline**: it loads a **pre-built windowed dataset** (Parquet from the dataset pipeline), splits by session into train/val/test, then runs LeJEPA training and SNN distillation. It does **not** extract sessions or process raw Allen data—that is done once in the [dataset pipeline](../data/README.md).

## Prerequisite

Run the **dataset pipeline** first so that a windowed Parquet exists:

```bash
python -m experiments.data.create_dataset path/to/dataset_config.yaml
```

Your experiment config’s `data_path` must point to that Parquet (e.g. `./datasets/visual_cortex_windows.parquet`).

## Config: experiment-only

The experiment config only needs:

- **`data_path`**: path to the Parquet produced by the dataset pipeline (no `dataset_config` needed).
- **`data`**: train/val/test split sizes and `random_state`.
- **Model and training** fields used by `multi_session.py` (see config template below).

Example: [`configs/experiment_from_dataset.yaml`](configs/experiment_from_dataset.yaml) — use this when you already have a dataset Parquet.

## Running the experiment

From the repo root:

```bash
python -m experiments.multi_session.multi_session path/to/experiment_config.yaml
```

The script will:

1. Verify the config.
2. Load the dataset from `data_path` and validate integrity.
3. Split by `session_id` into train/val/test.
4. Train LeJEPA teacher, evaluate on test.
5. Distill into SNN, evaluate on test.
6. Save results and metrics (TODO: full implementation).

## Config template

See `configs/lejepa_lif_visual_cortex.yaml` for a full template. For an experiment that only points at an existing dataset, see `configs/experiment_from_dataset.yaml`.

## Future: torch_brain

[torch_brain](https://github.com/neuro-galaxy/torch_brain) will be integrated for multi-recording training, optimized data loading, and models (e.g. POYO). The experiment pipeline will then consume the same pre-built Parquet dataset format.
