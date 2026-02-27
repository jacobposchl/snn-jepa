### Dataset pipeline (step 1)

This folder is the **dataset pipeline**: extract sessions from the Allen cache and write a single windowed Parquet. Run this **once** (or when you change sessions/regions/windowing); the experiment pipeline then uses the saved Parquet and does not re-extract sessions.

Entry points:

- **`data_analysis.py`**: interactive exploration of sessions and metadata (to choose sessions/regions for the dataset).
- **`create_dataset.py`**: batch creation of a preprocessed Parquet dataset consumed by the **experiment pipeline** (`experiments/multi_session/multi_session.py`).
- **`verify_dataset.py`**: sanity-check a Parquet before running experiments — validates schema, checks for session-count and integrity issues, and generates diagnostic plots.

---

### Using `data_analysis.py` (session and metadata search)

**Purpose**: Quickly inspect which sessions are available, what brain areas are recorded, and basic quality/behavioral metrics before deciding what to include in a dataset.

You can run it directly from the repo root:

```bash
python -m experiments.data.data_analysis --help
```

Key modes:

- **Overview of entire dataset** (metadata only, fast):

  ```bash
  python -m experiments.data.data_analysis --overview \
    --cache-dir ./visual_behavior_neuropixels_data
  ```

  This prints global counts: sessions, units, brain areas, genotypes, etc.

- **Detailed summary for one session** (loads full session from Allen cache):

  ```bash
  python -m experiments.data.data_analysis --session 1064415305 \
    --cache-dir ./visual_behavior_neuropixels_data
  ```

  Prints:
  - Session duration and number of units.
  - Units per brain area.
  - Stimulus blocks (active vs passive) with durations and counts.
  - Behavioral performance (hit/miss/FA/CR, hit rate, FA rate, d′).

- **All sessions (long, detailed)**:

  ```bash
  python -m experiments.data.data_analysis --all-sessions \
    --cache-dir ./visual_behavior_neuropixels_data
  ```

  Iterates through all sessions and prints the same detailed summary for each. This is slow but useful once if you want a full picture.

- **Filter sessions by metadata only** (fast, no session downloads):

  ```bash
  python -m experiments.data.data_analysis --filter \
    --cache-dir ./visual_behavior_neuropixels_data \
    --animals all \
    --regions VISp VISl \
    --units-required 300 300 \
    --phase 1
  ```

  This:
  - Uses only the Allen metadata tables (no session objects).
  - Filters by:
    - `--animals`: specific `mouse_id`s or `all`.
    - `--regions`: brain regions (e.g. `VISp VISl`).
    - `--units-required`: minimum number of units per corresponding region.
    - `--phase`: session number (1 or 2).
  - Prints a table of matching sessions with mouse, session number, image set, experience level, and session type.

This tool is mainly for **choosing which sessions and regions to include** when you write the `dataset_config` for `create_dataset.py`.

---

### Using `create_dataset.py` (build flat windowed Parquet)

**Purpose**: Generate a single Parquet with **one row per temporal window** across many sessions. This is the **dataset pipeline** output; the **experiment pipeline** (`experiments/multi_session/multi_session.py`) reads this Parquet via `data_path` and does not perform any session extraction.

Each row in the Parquet has:

- **`session_id`**: ecephys session ID.
- **`window_id`**: unique integer ID for the window.
- **`window_start_ms` / `window_end_ms`**: window bounds in milliseconds (session time).
- **`events_units`**: native NumPy arrays (np.int32) natively serialized in Parquet.
- **`events_times_ms`**: native NumPy arrays (np.float32) natively serialized in Parquet — ms offsets **relative to `window_start_ms`**, so values are in `[0, window_size_ms)`.
- **`stimulus`**: native Python lists of dictionaries (times relative to window start).
- **`behavior`**: native Python lists of dictionaries (times relative to window start).

This is exactly the structure expected by `load_and_prepare_data` in `experiments/multi_session/multi_session.py`.

#### Config file

`create_dataset.py` reads a YAML config that specifies:

- **`data_path`**: output Parquet path. Use this same path as `data_path` in your **experiment config** when running `multi_session.py`.
- **`dataset_config`**:
  - **`cache_dir`**: path to `visual_behavior_neuropixels_data` (Allen cache). **(Note: The Allen Visual Behavior Neuropixels dataset is hundreds of gigabytes. The initial session extraction via allensdk will take significant time and disk space.)**
  - **`session_ids`**: explicit list of ecephys session IDs to include.
  - **`session_filter`**: (planned) criteria to select sessions (e.g. genotype, experience level, brain areas, min units).
    Currently, the implementation expects `session_ids` to be provided explicitly.
  - **`brain_areas`**: list of areas to keep (e.g. `["VISp", "VISl"]`).
  - **`quality`**: SNR / firing rate / ISI thresholds.
  - **`windowing`**: bin size, window length, and stride (in ms).

A minimal example:

```yaml
data_path: ./datasets/visual_cortex_windows.parquet

dataset_config:
  cache_dir: ./visual_behavior_neuropixels_data
  session_ids: [1064415305, 1064644573]
  brain_areas: ["VISp", "VISl"]
  quality:
    min_snr: 1.0
    min_firing_rate: 0.1
    max_isi_violations: 1.0
  windowing:
    bin_size_ms: 10        # bin size used for spike binning
    window_size_ms: 400    # window length (per event-aligned window)
    stride_ms: 200         # planned: step between window starts (currently unused)
```

#### Running the script

From the repo root:

```bash
python -m experiments.data.create_dataset path/to/config.yaml
```

The script will:

- Validate the config.
- For each selected session:
  - Load and preprocess data via `VBNDataHandler` and `NeuropixelsPreprocessor`.
  - Extract exact sub-millisecond spike times within intervals using temporaldata.IrregularTimeSeries without manual binning.
  - Slide a window of length `window_size_ms` with stride `stride_ms`.
  - Build rows with `events_units`, `events_times_ms`, `stimulus`, and `behavior`.
- Concatenate all windows into a single `pandas.DataFrame` and write it to `data_path`.

Then run the **experiment pipeline** (step 2) with a config whose `data_path` points to this Parquet. See [experiments/README.md](../README.md) for the two-pipeline overview.

---

### Using `verify_dataset.py` (sanity-check before training)

**Purpose**: Validate a Parquet produced by `create_dataset.py` before handing it to the experiment pipeline. Run this after building a new dataset to catch issues early.

```bash
python -m experiments.data.verify_dataset path/to/dataset.parquet
```

Checks performed:

- **Schema**: all required columns (`session_id`, `window_id`, `window_start_ms`, `window_end_ms`, `events_units`, `events_times_ms`, `stimulus`, `behavior`) are present.
- **Session count**: at least 3 unique sessions exist (required for train/val/test split in `multi_session.py`).
- **Window integrity**: no `window_id` has conflicting timestamps (mirrors the check in `load_and_prepare_data`).
- **Array alignment**: `events_units` and `events_times_ms` have equal length in every row.
- **Spike statistics**: total spike count, average spikes per window, approximate population firing rate, and global time bounds.

Generated plots (saved next to the Parquet as `dataset_verification_advanced.png`):

- **Raster**: spike times vs unit IDs for one sample window.
- **Spike distribution**: histogram of spike counts per window.
- **Global PSTH**: population spike histogram across all windows.
- **Per-session PSTH**: normalized per-session spike densities overlaid for comparison.
