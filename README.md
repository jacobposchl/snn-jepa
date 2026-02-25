## SNN-JEPA: Spiking Neural Networks - Joint-Embedding Predictive Architecture

SNN-JEPA is a self-supervised learning framework for learning compact, informative representations of high-dimensional neural population activity. It combines **Spiking Neural Networks (SNNs)** with a **Joint-Embedding Predictive Architecture (JEPA)** to model the temporal dynamics of brain activity recorded via Neuropixels.

### Key components

- **`jepsyn/models`**: `NeuralEncoder` (Transformer-based encoder for binned spikes), `NeuralPredictor` (MLP/RNN predictor in latent space), and `SNNEncoder` (snntorch-based spiking encoder).
- **`jepsyn/losses`**: `lejepa_loss` (prediction + SIGReg regularization) and `DistillationLoss` (CCA-based distillation + homeostatic penalty).
- **`jepsyn/data`**: `VBNDataHandler` for the Allen Visual Behavior Neuropixels dataset and `NeuropixelsPreprocessor` for cleaning, filtering, and binning spikes.
- **`jepsyn/plots`**: Utilities for loss curves, latent-space metrics, spike count distributions, and prediction diagnostics.

### Project structure

- **`jepsyn/`**: Core library code (models, losses, data, plotting, utils).
- **`experiments/single_session/`**: Proof-of-concept training + distillation on a single ecephys session.
- **`experiments/multi_session/`**: Scaffold for multi-session training driven by YAML configs.
- **`visual_behavior_neuropixels_data/`**: Local cache for the Allen Neuropixels project (created by `allensdk`).

### Single-session proof-of-concept

This is the main working pipeline right now: it trains a JEPA **teacher** (Transformer encoder + predictor) and then distills it into a spiking **student**.

- **Entry point**: `experiments/single_session/single_session.py`
- **What it does**:
  - Downloads/loads one Visual Behavior Neuropixels session.
  - Preprocesses and bins spikes around image-change events.
  - Trains a JEPA teacher on context → future prediction.
  - Distills the teacher into an SNN using CCA + homeostatic regularization.
  - Saves plots and summaries under `runs/proof_of_concept/`.

**Example (local GPU or remote GPU with VS Code tunnel):**

```bash
export MPLBACKEND=Agg  # headless plotting
python -m experiments.single_session.single_session <SESSION_ID> \
  --dataset-dir /path/to/preprocessed/session_<SESSION_ID>.pkl
```

If `--dataset-dir` is omitted, the script will create `preprocessed/session_<SESSION_ID>.pkl` using the Allen cache under `visual_behavior_neuropixels_data/`.

### Multi-session experiment (WIP)

The multi-session workflow is split into **two pipelines** so you don’t re-extract sessions on every run:

1. **Dataset pipeline** (run once): `experiments/data/create_dataset.py` — extracts sessions from the Allen cache and writes a single windowed Parquet file. Config: `data_path` + `dataset_config` (cache_dir, session_ids, brain_areas, quality, windowing).
2. **Experiment pipeline** (run many times): `experiments/multi_session/multi_session.py` — loads that Parquet via `data_path`, splits by session into train/val/test, then trains LeJEPA and distills into an SNN. No session extraction; only the pre-built Parquet is used.

See **[experiments/README.md](experiments/README.md)** for the full two-pipeline overview and config reference.

- **Entry point (experiment)**: `experiments/multi_session/multi_session.py`
- **Configs**: `experiments/multi_session/configs/lejepa_lif_visual_cortex.yaml` (template), `experiments/multi_session/configs/experiment_from_dataset.yaml` (experiment-only, points at existing Parquet).

The experiment script validates the config, loads the dataset from `data_path`, and defines placeholders for `train_lejepa`, `distill_snn`, `evaluate_model`, and `save_results` (TODO for full implementation). **torch_brain** will be integrated for multi-recording training and data loaders.

### Dependencies

Install dependencies (CPU or GPU build of PyTorch as appropriate):

```bash
pip install -r requirements.txt
```

Key libraries include:

- `torch`, `torchvision`, `snntorch`
- `allensdk` (Visual Behavior Neuropixels project cache)
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`
- `pyarrow`, `temporaldata`

### Outputs

The single-session experiment writes to `runs/proof_of_concept/run_*`:

- Training and validation loss curves for JEPA.
- Latent prediction vs target diagnostics.
- Spike count distributions.
- C CA similarity and homeostatic penalty traces for SNN distillation.
