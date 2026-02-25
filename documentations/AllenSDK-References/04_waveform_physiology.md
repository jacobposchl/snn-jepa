# AllenSDK VBN Reference - Part 4: Waveform Physiology

**Topics**: Waveform metrics, cell-type classification, regional morphology patterns

---

## Waveform Metrics Overview

Neural waveforms contain structural information about spike morphology. **Computed on spike templates** (averaged waveforms from spike sorter).

### 5 Key Waveform Metrics

| Metric | Range | Computation | Interpretation |
|--------|-------|-----------|-----------------|
| `waveform_duration` | ~0.15-0.5 ms | Peak time to trough time | Spike duration (FS cells = narrow, RS = broad) |
| `velocity_above` | ~0.05-1.0 mm/ms | (PT height)/(PT time) | Pre-trough rising slope |
| `velocity_below` | Usually **negative** | (T height)/(TRep time) | Post-trough undershoot slope (typically >0 after spike ends) |
| `repolarization_slope` | ~0.1-0.5 | Post-trough recovery rate | Speed of return to baseline |
| `pt_ratio` | ~0.5-1.0 | Peak height / Trough depth ratio | Symmetry of spike (1.0 = symmetric, <1.0 = peak dominant) |

### Technical Note on Velocity_Below

The **sign of `velocity_below` depends on probe geometry**:
- **Negative in output**: Indicates the spike waveform goes below baseline and then recovers
- **Physical meaning**: All spikes have post-spike AHP (afterhyperpolarization); sign is convention
- **Use |velocity_below|**: When comparing across regions/probes, use absolute value

---

## Cell Type Classification: RS vs FS

**Regular Spiking (RS) vs Fast Spiking (FS)** distinguish excitatory pyramidal cells from inhibitory parvalbumin+ interneurons.

### Classification via Waveform Duration

```python
units = session.get_units()

# Standard threshold: 0.4 ms
RS_threshold = 0.4  # milliseconds

units['cell_type'] = units['waveform_duration'].apply(
    lambda x: 'FS' if x < RS_threshold else 'RS'
)

# Counts
print("Cell type distribution:")
print(units['cell_type'].value_counts())
# Output typically:
#   RS   ~18,000 units (75-80%)
#   FS   ~6,000 units  (20-25%)
```

### Why This Works

| Property | RS (Pyramidal) | FS (Parvalbumin) |
|----------|----------------|------------------|
| **Waveform Duration** | 0.4-0.5 ms | 0.15-0.3 ms |
| **Repolarization** | Slow (broad AHP) | Fast (narrow AHP) |
| **Firing Pattern** | Regular, adaptive | Rapid, persistent |
| **Peak-Trough Ratio** | Lower | Higher (sharper spike) |
| **Velocity_Above** | Slower rise | Faster rise |
| **Brain Location** | Distributed | Dense in L4 (cortex) |

### Classification with Velocity

```python
# More strict: combine duration + velocity for high confidence
units['putative_cell_type'] = 'uncertain'

rs_mask = (units['waveform_duration'] > 0.4) & (units['velocity_above'] < 0.4)
fs_mask = (units['waveform_duration'] < 0.4) & (units['velocity_above'] > 0.3)

units.loc[rs_mask, 'putative_cell_type'] = 'RS'
units.loc[fs_mask, 'putative_cell_type'] = 'FS'

print(f"High-confidence RS: {(units['putative_cell_type'] == 'RS').sum()}")
print(f"High-confidence FS: {(units['putative_cell_type'] == 'FS').sum()}")
print(f"Uncertain: {(units['putative_cell_type'] == 'uncertain').sum()}")
```

---

## Regional Morphology Patterns

Waveform properties vary **substantially by brain region** due to recording geometry and cell-type composition.

### Cortex (Excitatory-Dominant)

**Typical units: 70% RS, 30% FS**

- `waveform_duration`: Mean ~0.45 ms (broader than subcortical)
- `velocity_above`: ~0.20 mm/ms (slower than thalamus)
- `repolarization_slope`: ~0.15 (gradual recovery)
- **Structure**: Layer IV dense in FS interneurons; Layers II/III mixed
- **Morphology pattern**: Bimodal distribution with distinct RS and FS peaks

**Example filtering**:
```python
cortex_areas = ['VISp', 'VISl', 'VISrl', 'VISpm', 'VISam', 'VISa']
cortex_units = units[units['structure_acronym'].isin(cortex_areas)]

rs_cortex = cortex_units[cortex_units['waveform_duration'] > 0.4]
fs_cortex = cortex_units[cortex_units['waveform_duration'] < 0.4]

print(f"Cortex: {len(rs_cortex)} RS, {len(fs_cortex)} FS")
# Typical: "Cortex: 12000 RS, 5000 FS"
```

### Hippocampus

**Typical units: 50% RS, 40% FS, 10% unclear**

- `waveform_duration`: Mean ~0.32 ms (narrower than cortex!)
- `velocity_above`: ~0.25 mm/ms (faster rise)
- `repolarization_slope`: ~0.22 (fast recovery, brief AHP)
- **Structure**: CA1 pyramidal cells have faster kinetics; interneurons mixed
- **Morphology pattern**: Broader distribution; less clear RS/FS separation

**Why different?**: Hippocampal pyramidal cells have faster membrane time constants than cortical pyramids

```python
hc_units = units[units['structure_acronym'].isin(['CA1', 'CA3', 'DG'])]

# Less clear separation - use velocity too
hc_rs = hc_units[(hc_units['waveform_duration'] > 0.3) & 
                 (hc_units['velocity_above'] < 0.3)]
hc_fs = hc_units[(hc_units['waveform_duration'] < 0.3) & 
                 (hc_units['velocity_above'] > 0.25)]

print(f"HC: {len(hc_rs)} putative RS, {len(hc_fs)} putative FS")
# Typical: "HC: 1200 putative RS, 800 putative FS" (less bimodal)
```

### Thalamus (LGN)

**Typical units: 85% RS, 15% FS** (few interneurons)

- `waveform_duration`: Mean ~0.38 ms (moderate)
- `velocity_above`: ~0.35 mm/ms (faster than cortex, different cell types)
- `repolarization_slope`: ~0.18 (moderate recovery)
- **Structure**: Mostly relay neurons (thalamocortical cells); few local interneurons
- **Morphology pattern**: Right-skewed; dominated by single morphology

```python
thal_units = units[units['structure_acronym'].isin(['dLGN'])]
thal_rs = thal_units[thal_units['waveform_duration'] > 0.35]
print(f"Thalamus: {len(thal_rs)} units (mostly RS-like)")
```

### Midbrain/Other (Diverse)

**Heterogeneous structures with variable morphology**

- `waveform_duration`: 0.2-0.5 ms (broad range)
- `velocity_above`: Highly variable
- **Structure**: Mix of dopamine neurons, GABAergic, cholinergic populations
- **Morphology pattern**: Multimodal (multiple distinct peaks)

```python
midbrain_areas = ['SC', 'IC', 'MB', 'SN', 'VTA']
midbrain_units = units[units['structure_acronym'].isin(midbrain_areas)]

# Plot distribution to identify clusters
plt.hist(midbrain_units['waveform_duration'], bins=50)
plt.xlabel('Waveform Duration (ms)')
plt.ylabel('Count')
plt.title(f'Midbrain morphology (n={len(midbrain_units)})')
plt.show()
```

---

## Waveform-Based Filtering Strategies

### Strategy 1: Simple Duration Threshold

```python
# Extract narrow-spiking neurons (interneurons)
narrow_spiking = units[units['waveform_duration'] < 0.35]

# Extract broad-spiking neurons (excitatory)
broad_spiking = units[units['waveform_duration'] > 0.45]
```

**Pros**: Single parameter, robust
**Cons**: ~10% overlap, misses FS cells in hippocampus

### Strategy 2: Duration + Velocity (Bivariate)

```python
# Strict cell-type assignment
units['morphology_type'] = 'ambiguous'

# FS criterion: narrow + fast
units.loc[(units['waveform_duration'] < 0.35) & 
          (units['velocity_above'] > 0.3), 'morphology_type'] = 'FS'

# RS criterion: broad + slow  
units.loc[(units['waveform_duration'] > 0.45) & 
          (units['velocity_above'] < 0.25), 'morphology_type'] = 'RS'

print(units['morphology_type'].value_counts())
```

**Pros**: Higher specificity
**Cons**: More parameters, region-dependent thresholds

### Strategy 3: Regional Adaptive Thresholds

```python
def classify_by_region(units_df):
    """Apply region-specific waveform thresholds"""
    classified = units_df.copy()
    classified['morphology_class'] = 'unclassified'
    
    # Cortex
    cortex_mask = classified['structure_acronym'].isin(['VISp', 'VISl', 'VISrl'])
    classified.loc[cortex_mask & (classified['waveform_duration'] < 0.38), 'morphology_class'] = 'FS'
    classified.loc[cortex_mask & (classified['waveform_duration'] > 0.42), 'morphology_class'] = 'RS'
    
    # Hippocampus (broader, faster)
    hc_mask = classified['structure_acronym'].isin(['CA1', 'CA3', 'DG'])
    classified.loc[hc_mask & (classified['waveform_duration'] < 0.30), 'morphology_class'] = 'FS'
    classified.loc[hc_mask & (classified['waveform_duration'] > 0.35), 'morphology_class'] = 'RS'
    
    # Thalamus
    thal_mask = classified['structure_acronym'].isin(['dLGN'])
    classified.loc[thal_mask & (classified['waveform_duration'] < 0.35), 'morphology_class'] = 'FS'
    classified.loc[thal_mask & (classified['waveform_duration'] > 0.40), 'morphology_class'] = 'RS'
    
    return classified

units = classify_by_region(units)
```

**Pros**: Compensates for regional variation
**Cons**: Requires knowledge of region-specific distributions

---

## Practical Recipes

### Plot Morphology Distribution by Region

```python
import matplotlib.pyplot as plt
regions = units['structure_acronym'].unique()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, region in enumerate(regions[:4]):
    region_units = units[units['structure_acronym'] == region]
    axes[idx].hist(region_units['waveform_duration'], bins=30, alpha=0.7, color='blue')
    axes[idx].axvline(0.4, color='red', linestyle='--', label='RS/FS threshold')
    axes[idx].set_title(f'{region} (n={len(region_units)})')
    axes[idx].set_xlabel('Waveform Duration (ms)')
    axes[idx].set_ylabel('Count')
    axes[idx].legend()

plt.tight_layout()
plt.show()
```

### Combined Waveform + Spike Sort Quality Filter

```python
# Good morphology + good spike sorting
high_quality = units[
    (units['waveform_duration'] > 0.15) &  # Not artifact
    (units['waveform_duration'] < 0.60) &  # Not multi-unit
    (units['snr'] > 1.0) &                 # Good SNR
    (units['isi_violations'] < 1.0) &      # Clean spikes
    (units['isolation_distance'] > 15)     # Isolated
]

print(f"High quality units: {len(high_quality)} ({100*len(high_quality)/len(units):.1f}%)")
```

### Extract Units by Morphology + Region + Genotype

```python
# ChR2+ cortical RS cells (for behavior correlations)
cortex = ['VISp', 'VISl']
chर2_cortex_rs = session.get_units()
chर2_cortex_rs = chر2_cortex_rs[
    (chर2_cortex_rs['ecephys_structure_acronym'].isin(cortex)) &
    (chर2_cortex_rs['waveform_duration'] > 0.42) &
    (chर2_cortex_rs['snr'] > 1.5)
]

print(f"Found {len(chр2_cortex_rs)} ChR2+ cortical RS cells")
```

---

## Next Steps

- **Struggling with filtering?** → See [02_quality_metrics.md](02_quality_metrics.md)
- **Need specific analysis patterns?** → See [03_analysis_workflows.md](03_analysis_workflows.md)
- **Understanding the dataset?** → See [01_dataset_overview.md](01_dataset_overview.md)
