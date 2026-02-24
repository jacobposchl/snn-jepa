### Single-session proof-of-concept

**Goal**: Train a JEPA teacher (Transformer encoder + latent predictor) on one Visual Behavior Neuropixels session and distill it into a spiking SNN student, then visualize performance.

### What this script does

- Loads a single ecephys session from the Allen Visual Behavior Neuropixels dataset.
- Preprocesses and filters units, then bins spikes around image-change events (0–400 ms, 10 ms bins).
- Splits each trial into:
  - **Context**: first 200 ms (bins 0–20)
  - **Target**: next 200 ms (bins 20–40)
- **Phase 1**: trains a JEPA teacher (`NeuralEncoder` + `NeuralPredictor`) with prediction + SIGReg loss.
- **Phase 2**: trains an `SNNEncoder` to match teacher latents using CCA + homeostatic penalty.
- Saves plots and a text summary under `runs/proof_of_concept/run_*`.

### Requirements

- Allen Visual Behavior Neuropixels cache available locally (or mount via remote machine/Colab):
  - `visual_behavior_neuropixels_data/` directory at repo root (created by `allensdk`).
- Python environment with dependencies from the repo `requirements.txt` installed.
- For headless runs (e.g., VS Code tunnel, Colab), set `MPLBACKEND=Agg` so matplotlib can save figures without a display.

### How to run

From the repo root:

```bash
export MPLBACKEND=Agg  # for headless environments
python -m experiments.single_session.single_session <SESSION_ID> \
  --dataset-dir /path/to/preprocessed/session_<SESSION_ID>.pkl
```

- If `--dataset-dir` points to an existing `session_<SESSION_ID>.pkl`, it will be loaded directly.
- If `--dataset-dir` is omitted, the script will:
  - Build `visual_behavior_neuropixels_data/` (if needed) via `allensdk`.
  - Create `preprocessed/session_<SESSION_ID>.pkl` with filtered units and trial-aligned binned data.

### Outputs

After a successful run, check:

- `runs/proof_of_concept/run_*/phase1_teacher.png` – JEPA losses, prediction quality, spike stats.
- `runs/proof_of_concept/run_*/phase2_distillation.png` – CCA similarity and homeostatic penalty across epochs.
