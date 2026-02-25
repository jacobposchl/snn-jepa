# AllenSDK VBN Reference - Part 3: Analysis Workflows

**Topics**: PSTH, receptive field mapping, optotagging with copy-paste code examples

---

## Stimulus Presentations Table

**Key columns in `session.stimulus_presentations`**:

| Column | Type | Description |
|--------|------|-------------|
| `stimulus_name` | str | Type of stimulus ('natural_movie', 'gabor', etc.) |
| `start_time` | float | Stimulus onset (seconds) |
| `stop_time` | float | Stimulus offset (seconds) |
| `duration` | float | stop_time - start_time |
| `active` | bool | Whether presented during active task block |
| `is_change` | bool | Whether this was a change event |
| `position_x` | float | Azimuth for gabor stimuli (degrees, visual field) |
| `position_y` | float | Elevation for gabor stimuli (degrees, visual field) |

### Common Stimulus Access Patterns

```python
stimulus = session.stimulus_presentations

# Image change events during active behavior
change_times = stimulus[stimulus['active'] & stimulus['is_change']]['start_time'].values

# Receptive field mapping (gabor stimuli)
rf_stim = stimulus[stimulus['stimulus_name'].str.contains('gabor')]
xs = np.sort(rf_stim.position_x.unique())  # azimuth positions
ys = np.sort(rf_stim.position_y.unique())  # elevation positions

# Optotagging laser pulses (SST/VIP mice only)
opto_times = session.optotagging_table['start_time'].values
```

---

## Pattern 1: PSTH (Peri-Stimulus Time Histogram)

Computing firing rate around stimulus events. **Most common analysis pattern.**

### Helper Function

```python
def makePSTH(spikes, event_times, window_duration, bin_size=0.001):
    """
    Compute PSTH aligned to events.
    
    Args:
        spikes: Array of spike times (seconds)
        event_times: Array of event times to align to
        window_duration: Duration to extract after event (seconds)
        bin_size: Histogram bin size (seconds)
    
    Returns:
        counts: Firing rate in each bin (Hz)
        bins: Bin edges (seconds)
    """
    bins = np.arange(0, window_duration + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)
    
    for start in event_times:
        start_idx = np.searchsorted(spikes, start)
        end_idx = np.searchsorted(spikes, start + window_duration)
        counts += np.histogram(spikes[start_idx:end_idx] - start, bins)[0]
    
    counts = counts / len(event_times)  # Average across events
    counts = counts / bin_size           # Convert to Hz
    return counts, bins
```

### Basic Usage

```python
# Get image change times
stimulus = session.stimulus_presentations
change_times = stimulus[stimulus['is_change']]['start_time'].values

# Compute response to changes for all units
spike_times = session.spike_times
units = session.get_units()

responses = {}
for unit_id in units.index:
    psth, bins = makePSTH(spike_times[unit_id], change_times,
                         window_duration=1.0,    # 1 second total
                         bin_size=0.01)          # 10 ms bins
    responses[unit_id] = psth
```

### Visualize as Heatmap + Population

```python
import matplotlib.pyplot as plt

# Filter for good units
good_units = units[(units['snr'] > 1) & 
                   (units['isi_violations'] < 1)]

# Compute PSTHs
area_responses = []
for unit_id in good_units.index:
    psth, _ = makePSTH(spike_times[unit_id], change_times,
                      window_duration=1.0, bin_size=0.01)
    area_responses.append(psth)

area_responses = np.array(area_responses)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Heatmap
clims = [np.percentile(area_responses, p) for p in (0.1, 99.9)]
im = axes[0].imshow(area_responses, cmap='viridis', aspect='auto')
im.set_clim(clims)
axes[0].set_title(f'Image Change Responses (n={len(area_responses)} units)')
axes[0].set_xlabel('Time from change (100ms bins)')
axes[0].set_ylabel('Unit (sorted by depth)')

# Population average
axes[1].plot(np.mean(area_responses, axis=0), 'k-', linewidth=2)
axes[1].fill_between(range(len(np.mean(area_responses, axis=0))),
                     np.mean(area_responses, axis=0) - np.std(area_responses, axis=0)/np.sqrt(len(area_responses)),
                     np.mean(area_responses, axis=0) + np.std(area_responses, axis=0)/np.sqrt(len(area_responses)),
                     alpha=0.3)
axes[1].set_title(f'Population Average')
axes[1].set_xlabel('Time from change (100ms bins)')
axes[1].set_ylabel('Firing Rate (Hz)')

plt.tight_layout()
plt.show()
```

### Key Parameters

- **Time window**: Capture baseline + response
  - Pre-event: 0.5-1.0 sec (baseline)
  - Post-event: 0.5-2.0 sec (response)
  - Total: 1.0-2.5 sec typical
  
- **Bin size**: Trade-off between smoothness and temporal resolution
  - 1 ms (0.001): High temporal resolution, noisy
  - 10 ms (0.01): Good balance
  - 50 ms (0.05): Very smooth, low resolution

---

## Pattern 2: Receptive Field Mapping

Mapping visual field organization of neurons using gabor stimuli.

### Helper Function

```python
def find_rf(spike_times, stimulus_table, xs, ys, response_window=(0.01, 0.2)):
    """
    Map receptive field at each stimulus position.
    
    Args:
        spike_times: Array of spike times for one unit
        stimulus_table: Filtered stimulus_presentations (gabor only)
        xs: Array of x positions (azimuth, degrees)
        ys: Array of y positions (elevation, degrees)
        response_window: (pre_time, post_time) in seconds
    
    Returns:
        unit_rf: 2D array [elevation x azimuth] with firing rates
    """
    unit_rf = np.zeros([len(ys), len(xs)])
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            # Get stimulus presentations at this position
            stim_times = stimulus_table[
                (stimulus_table.position_x == x) & 
                (stimulus_table.position_y == y)
            ]['start_time'].values
            
            if len(stim_times) == 0:
                continue
            
            # Compute PSTH for this position
            psth, _ = makePSTH(spike_times, stim_times,
                             window_duration=response_window[1],
                             bin_size=0.001)
            
            # Average response in post-stimulus window
            response_start = int(response_window[0] / 0.001)
            response_end = int(response_window[1] / 0.001)
            unit_rf[iy, ix] = psth[response_start:response_end].mean()
    
    return unit_rf
```

### Basic Usage

```python
# Get receptive field stimulus
stimulus = session.stimulus_presentations
rf_stim = stimulus[stimulus['stimulus_name'].str.contains('gabor')]
xs = np.sort(rf_stim.position_x.unique())
ys = np.sort(rf_stim.position_y.unique())

# Map RF for all good units in visual cortex
spike_times = session.spike_times
units = session.get_units()
cortex_units = units[units['structure_acronym'].isin(['VISp', 'VISl'])]

rfs = {}
for unit_id in cortex_units.index:
    rf = find_rf(spike_times[unit_id], rf_stim, xs, ys)
    rfs[unit_id] = rf
```

### Visualize RFs

```python
# Plot grid of RFs
n_units = len(rfs)
n_cols = 10
n_rows = int(np.ceil(n_units / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*1.2))
axes = axes.flatten()

for idx, (unit_id, rf) in enumerate(rfs.items()):
    ax = axes[idx]
    im = ax.imshow(rf, cmap='hot', origin='lower', aspect='auto')
    ax.set_title(f'Unit {unit_id}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### Interpretation

- **Rows**: Elevation (top = up in visual field, bottom = down in visual field)
- **Columns**: Azimuth (left = left in visual field, right = right in visual field)
- **Brighter pixels**: Stronger response at that visual field location
- **RF center**: Preferred visual position for this unit
- **RF size**: Spatial summation / receptive field size

---

## Pattern 3: Optotagging (SST/VIP Identification)

Identifying optogenetically tagged interneurons via photostimulation. **SST and VIP mice only.**

### Laser Waveforms

Session uses **2 laser waveforms** at **3 power levels** = 6 conditions:
- **Short pulse**: 10 ms square pulse
- **Long pulse**: 1 second half-period cosine

### Get Optotagging Stimulus

```python
opto_table = session.optotagging_table
print(opto_table.columns)
# Columns: start_time, duration, level (power)

# Get high-power short pulses (most reliable for tagging)
duration_short = opto_table[opto_table['duration'] == opto_table['duration'].min()]
level_high = opto_table['level'].max()

opto_times = opto_table[
    (opto_table['duration'] < 0.05) &  # Short pulse
    (opto_table['level'] == level_high)  # High power
]['start_time'].values
```

### Compute Opto Response

```python
spike_times = session.spike_times
units = session.get_units()

# Get cortical units (most responsive in cortex)
cortex_units = units[units['structure_acronym'].isin(['VISp', 'VISl', 'VISrl'])]

# Compute PSTH during optotagging
time_before_laser = 0.01  # 10 ms pre-stimulus baseline
psth_window = 0.03        # 30 ms total window
bin_size = 0.001          # 1 ms bins

opto_responses = []
unit_list = []

for unit_id in cortex_units.index:
    opto_psth, bins = makePSTH(spike_times[unit_id],
                               opto_times - time_before_laser,
                               window_duration=psth_window,
                               bin_size=bin_size)
    
    # Calculate response magnitude (exclude laser artifacts)
    baseline_window = slice(0, 10)        # First 10 ms = pre-laser
    response_window = slice(10, 20)       # Next 10 ms = laser on (10-20 ms)
    # Skip first & last 1-2 ms (laser artifacts at onset/offset)
    
    baseline_rate = opto_psth[baseline_window].mean()
    response_rate = opto_psth[response_window].mean()
    response_magnitude = response_rate - baseline_rate
    
    opto_responses.append(opto_psth)
    unit_list.append((unit_id, response_magnitude))

opto_responses = np.array(opto_responses)
```

### Identify Tagged Units

```python
# Sort by response magnitude
unit_list.sort(key=lambda x: x[1], reverse=True)

# Threshold for calling a unit "tagged"
response_magnitudes = np.array([x[1] for x in unit_list])
threshold = np.percentile(response_magnitudes, 90)  # Top 10%

tagged_units = [uid for uid, mag in unit_list if mag > threshold]
print(f"Identified {len(tagged_units)} tagged units (top 10%)")
```

### Visualize Opto Responses

```python
# Heatmap of all responses
fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(opto_responses, origin='lower', aspect='auto', cmap='hot')

# Mark laser onset/offset
laser_onset = int(0.01 / 0.001)      # 10 ms
laser_offset = int(0.02 / 0.001)     # 20 ms
ax.axvline(laser_onset, color='cyan', linestyle='--', linewidth=2, label='Laser on')
ax.axvline(laser_offset, color='cyan', linestyle='--', linewidth=2)

ax.set_xlabel('Time from laser onset (ms)')
ax.set_ylabel('Unit')
ax.set_title('Opto-evoked responses (SST/VIP cells)')
ax.set_xticks(range(0, len(bins)-1, 5))
ax.set_xticklabels(np.round((bins[:-1:5] - time_before_laser) * 1000, 1))
plt.colorbar(im, label='Firing Rate (Hz)')
plt.legend()
plt.tight_layout()
plt.show()

# Population average
fig, ax = plt.subplots(figsize=(8, 4))
mean_response = opto_responses.mean(axis=0)
std_response = opto_responses.std(axis=0) / np.sqrt(len(opto_responses))

ax.plot(bins[:-1] - time_before_laser, mean_response, 'k-', linewidth=2)
ax.fill_between(bins[:-1] - time_before_laser,
                mean_response - std_response,
                mean_response + std_response,
                alpha=0.3, color='k')
ax.axvline(0.01, color='red', linestyle='--', alpha=0.5, label='Laser')
ax.axvline(0.02, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Time from laser onset (s)')
ax.set_ylabel('Firing Rate (Hz)')
ax.set_title(f'Population opto response (n={len(opto_responses)} cortical units)')
plt.legend()
plt.tight_layout()
plt.show()
```

### Key Points

- **Laser artifact**: Exclude data at onset (first ~1 ms) and offset (last ~1 ms)
- **Response window**: 10-20 ms after laser onset most reliable
- **Short pulse > long pulse**: Short pulses more reliable for identification
- **Response magnitude**: Positive response indicates ChR2 expression
- **Latency**: Short-latency spikes (<5 ms) most reliable for tagging

---

## Next Steps

- **Filtering first?** → See [02_quality_metrics.md](02_quality_metrics.md)
- **Waveform analysis?** → See [04_waveform_physiology.md](04_waveform_physiology.md)
- **Dataset structure?** → See [01_dataset_overview.md](01_dataset_overview.md)
