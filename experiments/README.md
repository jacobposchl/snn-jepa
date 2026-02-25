# Experiments: Dataset pipeline and experiment pipeline

The multi-session workflow is split into **two pipelines** so you can build datasets once and run many experiments without re-extracting sessions or re-processing raw data.

---

## 1. Dataset pipeline (run once)

**Purpose:** Extract sessions from the Allen cache, preprocess units, and write a single windowed CSV.

- **Entry point:** [`experiments/data/create_dataset.py`](data/create_dataset.py)
- **Input:** A YAML config with `data_path` (output CSV) and `dataset_config` (cache path, session IDs, brain areas, quality, windowing).
- **Output:** One CSV at `data_path` with one row per temporal window (session_id, window_id, events_units, events_times_ms, stimulus, behavior, etc.).

You run this whenever you want to **change sessions, regions, or windowing**. The result is a reusable dataset file.

**Details:** [experiments/data/README.md](data/README.md)

---

## 2. Experiment pipeline (run many times)

**Purpose:** Load the pre-built CSV, split by session into train/val/test, then train LeJEPA and distill into an SNN.

- **Entry point:** [`experiments/multi_session/multi_session.py`](multi_session/multi_session.py)
- **Input:** A YAML config with `data_path` pointing to the **existing** CSV produced by the dataset pipeline, plus data splits, model, and training settings.
- **No session extraction:** The experiment pipeline does not touch the Allen cache or session IDs; it only reads the CSV.

You run this for each experiment (e.g. different hyperparameters or random seeds). No need to re-extract or re-create the dataset.

**Details:** [experiments/multi_session/README.md](multi_session/README.md)

---

## Quick reference

| Step | Command | Config contains |
|------|---------|-----------------|
| **1. Create dataset** | `python -m experiments.data.create_dataset <dataset_config.yaml>` | `data_path`, `dataset_config` (cache_dir, session_ids, brain_areas, quality, windowing) |
| **2. Run experiment** | `python -m experiments.multi_session.multi_session <experiment_config.yaml>` | `data_path` (path to CSV from step 1), data splits, model, training |

---

## Future integration

- **[torch_brain](https://github.com/neuro-galaxy/torch_brain)** will be integrated for multi-recording training, data loaders, and models (e.g. POYO).
- **[temporaldata](https://github.com/neuro-galaxy/temporaldata)** can be used to abstract spike–stimulus mapping and interval extraction when moving from Allen SDK into the dataset pipeline.
