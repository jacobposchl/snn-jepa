## SNN-JEPA: Spiking Neural Networks - Joint-Embedding Predictive Architecture

SNN-JEPA is a self-supervised learning framework for learning compact, informative representations of high-dimensional neural population activity. It combines **Spiking Neural Networks (SNNs)** with a **Joint-Embedding Predictive Architecture (JEPA)** to model the temporal dynamics of brain activity recorded via Neuropixels.

### Key components

- **`jepsyn/models`**: `NeuralEncoder` (Transformer-based encoder for binned spikes), `NeuralPredictor` (MLP/RNN predictor in latent space), and `SNNEncoder` (snntorch-based spiking encoder).
- **`jepsyn/losses`**: `lejepa_loss` (prediction + SIGReg regularization) and `DistillationLoss` (CCA-based distillation + homeostatic penalty).
- **`jepsyn/data`**: `VBNDataHandler` for the Allen Visual Behavior Neuropixels dataset and `NeuropixelsPreprocessor` for cleaning, filtering, and binning spikes.
- **`jepsyn/plots`**: Utilities for loss curves, latent-space metrics, spike count distributions, and prediction diagnostics.

### 📁 Project structure

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

- **Entry point**: `experiments/multi_session/multi_session.py`
- **Config template**: `experiments/multi_session/configs/lejepa_lif_visual_cortex.yaml`

This script is a scaffold for training on many sessions at once. It:

- Validates a YAML config via `jepsyn.utils.verify_config`.
- Expects a precomputed windowed CSV dataset and splits it into train/val/test by `session_id`.
- Defines placeholders for:
  - `train_lejepa` (multi-session JEPA training),
  - `distill_snn` (multi-session SNN distillation),
  - `evaluate_model` and `save_results`.

These functions are marked TODO and intended as starting points for a full-scale experiment.

### Dependencies

Install dependencies (CPU or GPU build of PyTorch as appropriate):

```bash
pip install -r requirements.txt
```

Key libraries include:

- `torch`, `torchvision`, `snntorch`
- `allensdk` (Visual Behavior Neuropixels project cache)
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`

### Outputs

The single-session experiment writes to `runs/proof_of_concept/run_*`:

- Training and validation loss curves for JEPA.
- Latent prediction vs target diagnostics.
- Spike count distributions.
- CCA similarity and homeostatic penalty traces for SNN distillation.
