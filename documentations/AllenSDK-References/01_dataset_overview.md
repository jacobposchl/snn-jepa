# AllenSDK VBN Reference - Part 1: Dataset Overview

**Topics**: Data access, metadata tables, transgenic lines, training pipeline, brain regions

---

## Data Access Methods

### Get Cache

```python
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
from pathlib import Path

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=Path(output_dir))
```

### Common Cache Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_unit_table()` | DataFrame | All 319,013 units across all sessions with quality metrics |
| `get_ecephys_session_table()` | DataFrame | Metadata for 153 recording sessions |
| `get_behavior_session_table()` | DataFrame | Full training history across all mice |
| `get_probe_table()` | DataFrame | Probe metadata (typically 6 per session) |
| `get_channel_table()` | DataFrame | Channel information (384 per probe) |
| `get_ecephys_session(session_id)` | Session object | Load full session: units, spikes, stimuli, behavior |

---

## Metadata Tables Structure

### Table Relationships

```
ecephys_sessions (rows = recording sessions)
        ├→ units (many units per session)
        ├→ probes (typically 6 probes per session)
        └→ channels (384 channels per probe × 6)

behavior_sessions (rows = behavior training sessions)
        └→ links to ecephys_sessions via behavior_session_id
    
units ←→ channels (via ecephys_channel_id)
units ←→ probes (via ecephys_probe_id)
```

### 1. Ecephys Sessions Table

**Key columns**:
- `mouse_id` - Unique mouse identifier
- `session_number` - 1st or 2nd recording day
- `genotype` - Transgenic line (wt, Sst, Vip)
- `sex` - Male/Female
- `age_in_days` - Age at recording
- `image_set` - 'G' or 'H' (stimulus set)
- `experience_level` - 'Familiar' or 'Novel'
- `structure_acronyms` - List of brain areas recorded
- `probe_count` - Number of probes successfully inserted (typically 6)
- `abnormal_histology` - List of areas with tissue damage (null if none)
- `abnormal_activity` - List of times with epileptiform activity (null if none)

**Important**: `get_ecephys_session_table()` filters out abnormal sessions by default. Use `filter_abnormalities=False` to include them.

### 2. Behavior Sessions Table

**Key columns**:
- `mouse_id` - Unique identifier
- `session_type` - Training stage (see Training Pipeline section)
- `equipment_name` - Location ('BEH.*' = behavior facility; 'NP.*' = Neuropixels rig)
- `date_of_acquisition` - Session date
- `sex`, `age_in_days`, `genotype` - Mouse metadata
- `prior_exposures_to_image_set` - Count of prior training sessions with this image set
- `prior_exposures_to_omissions` - Whether mouse previously encountered stimulus omissions

**Note**: Most mice have many behavior sessions (training) leading up to 2 ephys (EPHYS) sessions.

### 3. Units Table

**Linkage columns**:
- `ecephys_session_id` - Links to ecephys_sessions
- `ecephys_probe_id` - Probe identifier
- `ecephys_channel_id` - Peak channel for unit waveform

**Spatial location**:
- `probe_channel_number` - Position on probe (0-383)
- `probe_horizontal_position` - Horizontal offset (perpendicular to shank) in μm
- `probe_vertical_position` - Vertical position (along shank) in μm
- `anterior_posterior_ccf_coordinate`, `dorsal_ventral_ccf_coordinate`, `left_right_ccf_coordinate` - CCF brain coordinates
- `structure_acronym` - Brain area label

**Contains**: All 8 quality metrics, waveform metrics, and metadata (see 02_quality_metrics.md for details)

### 4. Probes Table

**Columns**:
- `ecephys_session_id` - Session identifier
- `structure_acronyms` - Brain areas targeted by this probe (list)
- `unit_count` - Number of units on this probe
- `sampling_rate` - Recording sampling rate (Hz)
- Probe type info

### 5. Channels Table

**Columns**:
- `probe_channel_number` - Position on probe (0-383)
- `structure_acronym` - Brain area at this channel location
- CCF coordinates for each channel
- Relative position info

**Note**: 384 channels per Neuropixels probe (standard)

---

## Transgenic Lines & Genotypes

Three genotypes used in dataset:

| Genotype | Label | Purpose | Optotagging? | Cell Type |
|----------|-------|---------|--------------|-----------|
| C57Bl6J | wt | Wild-type control | ❌ No | All neurons |
| Sst-IRES-Cre;Ai32 | Sst | Label somatostatin interneurons | ✅ Yes (ChR2) | SST+ inhibitory |
| Vip-IRES-Cre;Ai32 | Vip | Label VIP interneurons | ✅ Yes (ChR2) | VIP+ inhibitory |

**SST & VIP**: Two major inhibitory cell classes in cortex; can be identified by photostimulation during recording (see 03_analysis_workflows.md#pattern-3-optotagging)

**Access via**:
```python
sst_sessions = ecephys_sessions[ecephys_sessions['genotype'].str.contains('Sst')]
vip_sessions = ecephys_sessions[ecephys_sessions['genotype'].str.contains('Vip')]
wt_sessions = ecephys_sessions[ecephys_sessions['genotype'].str.contains('wt')]
```

---

## Training Pipeline & Session Types

Mice progress through a **shaping pipeline** to reach Neuropixels recording stage.

### Training Progression

| Stage | Session Type Pattern | Task | Reward | Duration |
|-------|----------------------|------|--------|----------|
| 0 | `TRAINING_0_gratings_autorewards_15min` | Static gratings (0° ↔ 90°); **non-contingent** rewards | 10 μL | 1 day |
| 1 | `TRAINING_1_gratings` | Gratings; **lick-contingent** rewards | 10 μL | ~2-5 days |
| 2 | `TRAINING_2_gratings_flashed` | Flashed gratings (500 ms ISI); lick-contingent | 10 μL | ~2-5 days |
| 3 | `TRAINING_3_images_X` | Natural images (X = G or H); lick-contingent | 10 μL | ~1-3 days |
| 4 | `TRAINING_4_images_X` | Images; reward reduced | 7 μL | ~1-3 days |
| 5a | `TRAINING_5_images_X_epilogue_5uL_reward` | Expose to RF mapping stimulus (Neuropixels paradigm) | 5 μL | ~1 day |
| 5b | `TRAINING_5_images_X_handoff_ready_5uL_reward` | Mouse reached criterion; ready for transfer | 5 μL | Ready for ephys |
| 5b alt | `TRAINING_5_images_X_handoff_lapsed_5uL_reward` | Performance dropped; re-training needed | 5 μL | Re-training |
| Ephys | `EPHYS_1_images_X_3uL_reward` | **First recording session** with flashing images + **~5% omissions** | 3 μL | Full session |
| Ephys | `EPHYS_2_images_Y_3uL_reward` | **Second recording session**; different image set | 3 μL | Full session |

**Key transitions**:
1. Gratings → Images (natural scene stimuli)
2. Behavior facility (BEH) → Neuropixels rig (NP) at HABITUATION
3. Single image set → Two sets across two recording days

**Automatic advancement**: Mice auto-advance between stages based on behavioral performance.

---

## Image Sets & Experience Levels

### Sets G & H

Two stimulus sets with partial overlap:
- **Set G**: One set of 8 images
- **Set H**: 2 images from G + 6 novel images

### Training & Recording Scheme

Each mouse trained with **one image set**, recorded in two sessions:

| Training | Session 1 | Session 2 | Count |
|----------|-----------|-----------|-------|
| Train on G | Familiar (G) | Novel (H) | 38 |
| Train on G | Novel (H) | Familiar (G) | 13 |
| Train on H | Familiar (H) | Novel (G) | 10 |

**Columns in tables**:
- `image_set`: Which set was used ('G' or 'H')
- `experience_level`: 'Familiar' = trained with this set; 'Novel' = first exposure

**Prior exposures**: By first recording session, mice had typically seen familiar set in 15-25 prior behavior training sessions.

**Omissions**: First time mice encounter **stimulus omissions** (5% of flashes) is during EPHYS recording. Allows study of temporal expectation signals.

---

## Brain Regions & Structure Acronyms

### Major Regions

```python
region_dict = {
    'cortex': ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal', 'VISmma', 'VISmmp', 'VISli'],
    'thalamus': ['LGd', 'LD', 'LP', 'VPM', 'TH', 'MGm', 'MGv', 'MGd', 'PO', 'LGv', 'VL', 'VPL', 'POL', 'Eth', 'PoT', 'PP', 'PIL', 'IntG', 'IGL', 'SGN', 'PF', 'RT'],
    'hippocampus': ['CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POST', 'PRE', 'ProS', 'HPF'],
    'midbrain': ['MB', 'SCig', 'SCiw', 'SCsg', 'SCzo', 'PPT', 'APN', 'NOT', 'MRN', 'OP', 'LT', 'RPF', 'CP']
}
```

### Region Characteristics

| Region | Firing Rate | ISI Violations | Key Feature |
|--------|------------|-----------------|------------|
| **Cortex** | Medium | Low-Medium | Visual processing areas |
| **Thalamus** | ~8 Hz (log scale) | Medium | Sensory relay; high SNR |
| **Hippocampus** | Low-Medium | **High** | Complex spike patterns; CA1, CA3 |
| **Midbrain** | **Very High (>20 Hz)** | Low-Medium | Superior colliculus, etc. |

---

## Session Loading & Attributes

### Load Single Session

```python
session = cache.get_ecephys_session(ecephys_session_id=1064644573)
```

### Session Attributes

```python
# Metadata
session.metadata  # Dict: session_id, genotype, experience_level, etc.

# Units and channels
units = session.get_units()
channels = session.get_channels()
unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)

# Spike data
spike_times = session.spike_times  # Dict: unit_id -> spike times (seconds)

# Stimuli
stimulus_presentations = session.stimulus_presentations  # DataFrame
session.mean_waveforms  # Dict: unit_id -> 2D waveform array

# Behavioral data
session.trials              # Trial-by-trial information
session.running_speed       # Running speed timeseries
session.licks               # Lick times
session.rewards             # Reward times
session.pupil_data          # Pupil tracking (if available)

# Optotagging (SST/VIP mice)
opto_table = session.optotagging_table  # Laser parameters
```

### Quick Session Summary

```python
units = session.get_units()
area_counts = units.value_counts('structure_acronym')

print(f"Total units: {len(units)}")
print(f"Units per brain area:\n{area_counts}")
```

---

## Key Statistics

- **153** ecephys sessions total
- **319,013** units across all sessions
- **6** probes per session (typical, some failures)
- **384** channels per probe
- **2** recording sessions per mouse (consecutive days)
- **100 μm** probe movement between session 1 and 2 (cannot track cross-session)

---

## Important Notes

### Probe Movement
Probes moved ~100 μm between session 1 and session 2, making it **impossible to track the same neurons across recording days** (unless analyzing within single session).

### Acute Recordings
Probes are **retracted after each session** (not chronic). All recordings are independent.

### Default Filtering
By default:
- Abnormal sessions are excluded (tissue damage, epileptiform activity)
- Invalid waveforms pre-filtered (noise templates removed)
- **All returned units validated as physiological**

Use `filter_abnormalities=False` to include flagged sessions.

---

## Next Steps

- **Understand quality metrics?** → See [02_quality_metrics.md](02_quality_metrics.md)
- **Implement analysis?** → See [03_analysis_workflows.md](03_analysis_workflows.md)
- **Learn waveforms?** → See [04_waveform_physiology.md](04_waveform_physiology.md)
