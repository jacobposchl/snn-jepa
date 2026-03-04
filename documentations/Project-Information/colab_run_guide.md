# Running SNN-JEPA on Google Colab

This guide explains how to get a training run hosted on **Google Colab** (single-session proof-of-concept or multi-session LeJEPA). The project already mentions Colab in the [project overview](project_overview.md) and provides two runner notebooks; this doc ties everything together and clarifies data/GPU requirements.

---

## Repository state (quick reference)

| Component | Status | Notes |
|-----------|--------|--------|
| **Single-session PoC** | Working | `experiments/single_session/single_session.py` — JEPA teacher + SNN distillation on one session |
| **Multi-session dataset pipeline** | Working | `experiments/data/create_dataset.py` — builds windowed Parquet from Allen cache |
| **Multi-session experiment pipeline** | Working (LeJEPA only) | `experiments/multi_session/multi_session.py` — trains LeJEPA on Parquet; SNN distillation is TODO |
| **Colab notebooks** | Present, need setup fixes | `single_session_runner.ipynb`, `multi_session_runner.ipynb` — see below |

---

## Option A: Single-session run on Colab

**Goal:** Train JEPA teacher + distill into SNN on **one** Visual Behavior Neuropixels session.

### Data choices

1. **Preprocessed pickle on Google Drive**  
   If you already have `session_<SESSION_ID>.pkl` (from running the pipeline locally or a previous Colab run), put it (or a zip containing it) on Drive and point the notebook at it. Fastest for repeated runs.

2. **Download + preprocess in Colab**  
   Omit `--dataset-dir`; the script will use `allensdk` to build the Allen cache under `visual_behavior_neuropixels_data/` and write `preprocessed/session_<SESSION_ID>.pkl`. First run will be slow and needs sufficient disk (Allen data is large). Good for one-off or when you don’t have a preprocessed file.

### Steps (notebook or script)

1. **Clone this repo** (e.g. into `/content/snn-jepa`).
2. **Install dependencies**  
   From repo root: `pip install -r requirements.txt` (use a GPU runtime if you want faster training). For headless plotting, the script sets `MPLBACKEND=Agg` internally.
3. **Run the single-session entry point**  
   - With preprocessed pickle (e.g. unzipped on Drive and mounted):
     ```bash
     python -m experiments.single_session.single_session <SESSION_ID> --dataset-dir /path/to/session_<SESSION_ID>.pkl
     ```
   - Without (script will create the pickle from Allen cache):
     ```bash
     python -m experiments.single_session.single_session <SESSION_ID>
     ```
4. **Outputs**  
   Plots and summaries under `runs/proof_of_concept/run_*`. Zip and download or copy to Drive if you need them after the runtime ends.

### Notebook

Use **`experiments/single_session/single_session_runner.ipynb`**. It should:

- Clone **this** repo (e.g. `https://github.com/<your-org>/snn-jepa`).
- Install from `requirements.txt` (or the minimal subset: `snntorch`, `allensdk`, etc.).
- Either mount Drive and set `--dataset-dir` to your preprocessed pkl, or run without it to build the dataset in Colab.
- Invoke `python -m experiments.single_session.single_session <SESSION_ID> [--dataset-dir ...]` from the **repo root**.

Fix any references to `proof_of_concept_edit.py` → use the module invocation above.

---

## Option B: Multi-session LeJEPA run on Colab

**Goal:** Train the LeJEPA teacher (and later SNN) on **multiple sessions** stored in a single windowed Parquet.

### Prerequisite: Parquet dataset

The multi-session **experiment** pipeline does **not** download or extract raw Allen data. It expects a Parquet produced by the **dataset pipeline**:

```bash
python -m experiments.data.create_dataset path/to/dataset_config.yaml
```

So you either:

1. **Build the Parquet in Colab**  
   Run `create_dataset.py` in Colab with a small `dataset_config` (e.g. 2–3 sessions, one brain region). You’ll need to point `cache_dir` at where `allensdk` downloads the Allen data (and have enough disk/time for that download).

2. **Use a Parquet you already built**  
   Upload the Parquet to Drive (or another URL Colab can read), mount Drive, and set your experiment config’s `data_path` to that file. Easiest for “just run training” in Colab.

### Steps (notebook or script)

1. **Clone this repo** and install dependencies from repo root (e.g. `pip install -r requirements.txt`).
2. **Get the Parquet** at a path Colab can see (e.g. `/content/drive/MyDrive/.../visp_visl_windows.parquet`).
3. **Point the experiment config** at that path: set `data_path` (and optionally `results_out_path`) in a YAML (e.g. `experiments/multi_session/configs/lejepa_lif_visual_cortex.yaml` or a Colab-specific copy).
4. **Run the experiment**  
   From repo root:
   ```bash
   python -m experiments.multi_session.multi_session path/to/experiment_config.yaml
   ```
   Or use **`experiments/multi_session/multi_session_runner.ipynb`**, which imports `multi_session` and calls `load_and_prepare_data`, `train_lejepa`, `evaluate_model`, `save_results` (see that notebook for the exact flow).

### Notebook

Use **`experiments/multi_session/multi_session_runner.ipynb`**. It assumes the repo is at a fixed path (e.g. `/content/snn-jepa` after cloning). Ensure:

- The first cells clone **this** repo into that path (e.g. `git clone https://github.com/<your-org>/snn-jepa` → `/content/snn-jepa`), then `%cd snn-jepa`.
- `CONFIG_PATH` is set to your experiment YAML; that YAML’s `data_path` must point to the Parquet (e.g. on Drive after mount).
- `results_out_path` in the config is set so checkpoints and plots are saved somewhere you can download or copy to Drive.

No need for conda on Colab if you use `pip install -r requirements.txt` and a standard Colab runtime (Python 3.10+).

---

## GPU and runtime

- **Single-session:** Runs on CPU but is much faster with a GPU. In Colab: Runtime → Change runtime type → GPU.
- **Multi-session:** LeJEPA training benefits from GPU; set a GPU runtime if available.

---

## Summary

| Run type | Entry point | Data need | Colab notebook |
|----------|-------------|-----------|----------------|
| Single-session | `python -m experiments.single_session.single_session <SESSION_ID> [--dataset-dir ...]` | One session: preprocessed pkl (optional) or let script build from Allen cache | `experiments/single_session/single_session_runner.ipynb` |
| Multi-session | `python -m experiments.multi_session.multi_session <config.yaml>` | Parquet from `experiments/data/create_dataset.py` (build in Colab or upload) | `experiments/multi_session/multi_session_runner.ipynb` |

Fixing the two runner notebooks to use this repo’s clone URL, correct commands, and paths will give you a single-session or multi-session run hosted on Google Colab as described above.
