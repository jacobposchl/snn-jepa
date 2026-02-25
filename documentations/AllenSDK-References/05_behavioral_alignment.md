# AllenSDK VBN Reference - Part 5: Behavioral Alignment

**Topics**: Stimulus/trials table structure, response latency, aligning running/licking/pupil to events

---

## Session Structure Overview

Every recording session consists of **5 stimulus blocks** presented in this order:

| Block | Epoch | Duration | Purpose |
|-------|-------|----------|---------|
| **0** | Change Detection Task | ~30-60 min | Active behavior: mouse licks for image changes |
| **1** | Gray Screen | Brief | Blank period |
| **2** | Receptive Field Mapping | ~10 min | Gabor stimuli at grid positions to map visual fields |
| **3** | Gray Screen | Brief | Blank period |
| **4** | Full-Field Flashes | ~5 min | Black/white flashes at 80% contrast (4 trials × 3 power levels) |
| **5** | Passive Replay | ~30-60 min | Frame-for-frame replay of block 0 (no lickspout, no rewards) |

**Key note**: Blocks 0 and 5 are frame-matched — every stimulus presentation in passive replay corresponds exactly to active behavior.

---

## Stimulus Presentations Table: Detailed Columns

**Access via**: `session.stimulus_presentations`

### Core Timing Columns

| Column | Type | Description |
|--------|------|-------------|
| `start_time` | float | Stimulus onset (seconds, corrected for display lag) |
| `end_time` | float | Stimulus offset (seconds, corrected for display lag) |
| `duration` | float | `end_time - start_time` |
| `start_frame` | int | Stimulus frame index (matches behavior frame timing) |
| `end_frame` | int | Ending frame index |
| `stimulus_block` | int | Block 0-5 identifier |

**Critical detail**: `start_time` and `end_time` are **display lag corrected**. This is when the stimulus actually appeared on the mouse's screen (not when the computer commanded it). Use these for behavioral alignment.

### General Task Columns (Blocks 0, 5)

| Column | Type | Description |
|--------|------|-------------|
| `stimulus_name` | str | Stimulus category ('images', 'gabors', 'flashes') |
| `active` | bool | `True` only during block 0 (active behavior with lickspout) |
| `image_name` | str | Which natural image presented (e.g., 'im062') |
| `is_change` | bool | `True` if image identity changed from previous |
| `flashes_since_change` | int | How many flashes of same image since last change |
| `omitted` | bool | `True` if 5% probability "omission" occurred (gray screen) |
| `rewarded` | bool | `True` if reward issued; in block 5, indicates reward on corresponding block 0 trial |

### Change Detection Task Special Cases

**Omission rules**:
- 5% baseline probability per flash
- Two omissions cannot occur consecutively
- Omission cannot directly precede a change
- Result: ~3-4% actual omission rate

**Timing**: Each image flash = 250 ms presentation + 500 ms inter-flash interval

### Receptive Field Mapping (Block 2)

| Column | Type | Description |
|--------|------|-------------|
| `orientation` | float | Gabor orientation (degrees) |
| `position_x` | float | Azimuth (degrees; negative = nasal, positive = temporal) |
| `position_y` | float | Elevation (degrees; negative = lower visual field) |
| `spatial_frequency` | float | Cycles per degree |
| `temporal_frequency` | float | Flicker frequency (Hz) |
| `contrast` | float | Stimulus contrast (0-1) |

**Grid structure**: Typically 8×8 or 10×10 positions tested with multiple contrasts.

### Full-Field Flashes (Block 4)

| Column | Type | Description |
|--------|------|-------------|
| `color` | int | Flash color: `1` = white, `-1` = black |
| `contrast` | float | Always 0.8 for this block |

**Power levels**: 3 laser power levels × 2 waveforms (short/long pulse) = 6 conditions total

---

## Behavior Trials Table: Detailed Columns

**Access via**: `session.trials`

Each row = one trial of the change detection task (block 0 active behavior only).

### Trial Timing

| Column | Type | Description |
|--------|------|-------------|
| `start_time` | float | Trial start (seconds) |
| `end_time` | float | Trial end (seconds) |
| `trial_length` | float | Duration in seconds |
| `change_time_no_display_delay` | float | When task computer commanded change (not display corrected!) |
| `change_frame` | int | Frame index when change occurred (link to stimulus_presentations) |

**⚠️ Important**: `change_time_no_display_delay` is **not** corrected for display lag. For accurate behavioral latency, use the display-corrected time from stimulus_presentations table (see alignment section below).

### Stimulus Identity

| Column | Type | Description |
|--------|------|-------------|
| `initial_image_name` | str | Image before change |
| `change_image_name` | str | Image scheduled for change |
| `stimulus_change` | bool | Whether change actually occurred |

### Trial Type Classification

| Column | Type | Description |
|--------|------|-------------|
| `go` | bool | Go trial: stimulus changed AND trial not autorewarded |
| `catch` | bool | Catch trial: sham change (no image change), mouse tests false alarm rate |
| `auto_rewarded` | bool | Reward automatically delivered regardless of lick (early trials to engage mouse) |

### Behavioral Outcome

| Column | Type | Description |
|--------|------|-------------|
| `response_time` | float | First lick time (not fully reliable — use licks table for precise timing) |
| `hit` | bool | Go trial + change occurred + lick in 150-750 ms window |
| `miss` | bool | Go trial + change occurred + NO lick in response window |
| `false_alarm` | bool | Catch trial + sham change + lick in reward window |
| `correct_reject` | bool | Catch trial + sham change + NO lick |
| `aborted` | bool | Lick before scheduled change/sham change (trial discarded) |

### Reward

| Column | Type | Description |
|--------|------|-------------|
| `reward_time` | float | When reward command triggered (hit trials only) |
| `reward_volume` | float | Water volume dispensed (µL) |
| `lick_times` | list | Lick times during trial (from behavioral control software) |

---

## Calculating Response Latency

Response latency = time from stimulus change to first lick. **Use display-corrected times for accuracy.**

### Step 1: Get Corrected Change Times

```python
def get_change_time_from_stim_table(row):
    """Map trials change frame to display-corrected time"""
    change_frame = row['change_frame']
    if np.isnan(change_frame) or change_frame < 0:
        return np.nan
    
    change_time = stimulus_presentations[
        stimulus_presentations['start_frame'] == change_frame
    ]['start_time'].values[0]
    
    return change_time

# Add corrected times to trials table
trials['change_time_corrected'] = trials.apply(
    get_change_time_from_stim_table, axis=1
)
```

### Step 2: Extract Hit Trial Latencies

```python
licks = session.licks  # Access precise lick sensor data

# Get hit trials only
hit_trials = trials[trials['hit']]

# Find first lick after each change
lick_indices = np.searchsorted(
    licks['timestamps'],
    hit_trials['change_time_corrected']
)
first_lick_times = licks['timestamps'].values[lick_indices]

# Calculate latency
response_latencies = first_lick_times - hit_trials['change_time_corrected']

# Plot
plt.hist(response_latencies, bins=50)
plt.xlabel('Response latency (s)')
plt.ylabel('Trials')
plt.axvline(0.15, color='r', label='Response window start')
plt.axvline(0.75, color='r', label='Response window end')
plt.show()
```

**Expected distribution**: Typically bimodal with peaks at ~0.2 s and ~0.5 s (reflecting two response strategies).

---

## Aligning Running, Licking, and Pupil Data to Events

**Available behavioral data**:
- `session.running_speed`: Encoder voltage sampled at ~60 Hz
- `session.licks`: Lick sensor timestamps (one per detected lick)
- `session.eye_tracking`: Pupil center, size, and corneal reflection (video frame rate ~30-60 Hz)

### Filter Eye Tracking for Blink Artifacts

```python
eye_tracking = session.eye_tracking

# Remove frames with likely blinks
eye_tracking_clean = eye_tracking[~eye_tracking['likely_blink']]

print(f"Removed {len(eye_tracking) - len(eye_tracking_clean)} blink frames")
```

### Align Behavioral Data to Event

```python
stimulus_presentations = session.stimulus_presentations
running_speed = session.running_speed
licks = session.licks
eye_tracking_clean = eye_tracking[~eye_tracking['likely_blink']]

# Define event window
event_time = session.rewards.iloc[0]['timestamps']  # or any other event
time_before = 3.0  # seconds
time_after = 3.0   # seconds

# Extract data in window
running_aligned = running_speed.query(
    f'timestamps >= {event_time - time_before} and '
    f'timestamps <= {event_time + time_after}'
)

licks_aligned = licks.query(
    f'timestamps >= {event_time - time_before} and '
    f'timestamps <= {event_time + time_after}'
)

pupil_aligned = eye_tracking_clean.query(
    f'timestamps >= {event_time - time_before} and '
    f'timestamps <= {event_time + time_after}'
)

# Get stimulus presentations in same window
behavior_stim = stimulus_presentations[
    stimulus_presentations['active'] & 
    ~stimulus_presentations['omitted']
]
stim_aligned = behavior_stim.query(
    f'end_time >= {event_time - time_before} and '
    f'start_time <= {event_time + time_after}'
)
```

### Plot Aligned Data

```python
fig, ax_run = plt.subplots(figsize=(14, 6))

# Running (left y-axis)
ax_run.plot(running_aligned['timestamps'] - event_time, 
            running_aligned['speed'], 'k-', linewidth=2)
ax_run.set_ylabel('Running speed (cm/s)')
ax_run.set_xlabel('Time from event (s)')

# Pupil (right y-axis)
ax_pupil = ax_run.twinx()
ax_pupil.plot(pupil_aligned['timestamps'] - event_time,
              pupil_aligned['pupil_area'], 'g-', alpha=0.7)
ax_pupil.set_ylabel('Pupil area (pixels²)', color='g')
ax_pupil.tick_params(axis='y', labelcolor='g')

# Mark event
ax_run.axvline(0, color='r', linestyle='--', linewidth=2, label='Event')

# Mark licks
ax_run.scatter(licks_aligned['timestamps'] - event_time,
               np.zeros(len(licks_aligned)), 
               marker='v', s=100, color='m', label='Licks')

# Shade stimulus presentations
for idx, stim in stim_aligned.iterrows():
    ax_run.axvspan(stim['start_time'] - event_time,
                   stim['end_time'] - event_time,
                   alpha=0.2, color='gray')

ax_run.legend()
plt.tight_layout()
plt.show()
```

### Compute Event-Aligned Averages

```python
# Get all reward times
reward_times = session.rewards['timestamps'].values

# Pre-allocate aligned arrays
n_events = len(reward_times)
window_size = int((time_before + time_after) * 60)  # assuming 60 Hz sampling
running_aligned_all = np.zeros([n_events, window_size])
pupil_aligned_all = np.zeros([n_events, window_size])

# Align each reward event
time_axis = np.arange(-time_before, time_after, 1/60.0)

for idx, event_time in enumerate(reward_times):
    running_mask = (running_speed['timestamps'] >= event_time - time_before) & \
                   (running_speed['timestamps'] <= event_time + time_after)
    pupil_mask = (eye_tracking_clean['timestamps'] >= event_time - time_before) & \
                 (eye_tracking_clean['timestamps'] <= event_time + time_after)
    
    running_aligned_all[idx, :len(running_speed[running_mask])] = \
        running_speed[running_mask]['speed'].values
    pupil_aligned_all[idx, :len(eye_tracking_clean[pupil_mask])] = \
        eye_tracking_clean[pupil_mask]['pupil_area'].values

# Plot population average
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_axis, np.mean(running_aligned_all, axis=0), 'k-', linewidth=2)
ax.fill_between(time_axis,
                np.mean(running_aligned_all, axis=0) - np.std(running_aligned_all, axis=0),
                np.mean(running_aligned_all, axis=0) + np.std(running_aligned_all, axis=0),
                alpha=0.3)
ax.axvline(0, color='r', linestyle='--', label='Reward')
ax.set_xlabel('Time from reward (s)')
ax.set_ylabel('Running speed (cm/s)')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Key Alignment Principles

1. **Use display-corrected times for stimulus**: `stimulus_presentations.start_time` accounts for monitor lag
2. **Use `start_frame` for frame-level precision**: Link stimulus_presentations ↔ behavior at frame granularity
3. **Licks table > trials table**: Trials table `response_time` is less accurate; use `session.licks` for precise timing
4. **Account for omissions**: 3-4% of presentations are omitted flashes; exclude these when aligning
5. **Clean eye tracking data**: Remove `likely_blink` frames before analysis
6. **Behavioral sampling rates**:
   - Running speed: ~60 Hz
   - Eye tracking: 30-60 Hz (video frame rate)
   - Licks: Variable (event-based)
   - Stimuli frames: 60 Hz (VBN stimulus refresh)

---

## Next Steps

- **Quality filtering?** → See [02_quality_metrics.md](02_quality_metrics.md)
- **Analysis patterns?** → See [03_analysis_workflows.md](03_analysis_workflows.md)
- **Dataset structure?** → See [01_dataset_overview.md](01_dataset_overview.md)
