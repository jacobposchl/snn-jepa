# AllenSDK VBN Reference - Part 2: Quality Metrics

**Topics**: 8 unit quality metrics, how to interpret them, filtering strategies, code examples

---

## Overview: Why Quality Metrics Matter

**All units returned by default** (unlike Visual Coding dataset). You decide which to keep.

**Key principle**: Quality metrics form a **gradient**, not binary categories. No perfect threshold—choose based on your analysis tolerance for contamination.

**Sorting errors occur** due to:
1. **Over-merging**: Multiple neurons in one cluster → `isi_violations`, `nn_hit_rate` help detect
2. **Under-amplitude**: Spikes below threshold → `amplitude_cutoff`, low `firing_rate` detect
3. **Electrode drift**: Waveform changes → `presence_ratio`, `amplitude_cutoff` detect

---

## Metrics Quick Reference

| Metric | Range | What It Measures | When to Use |
|--------|-------|-----------------|------------|
| `firing_rate` | Continuous (Hz) | Spike count / session duration | Filter out units with too few spikes |
| `presence_ratio` | 0-0.99 | Fraction of time unit spiking | Detect drift across entire session |
| `amplitude_cutoff` | 0-0.5 | Est. fraction of missing spikes | High-precision spike timing analyses |
| `isi_violations` | 0-∞ | Relative contamination rate | **Most direct measure of contamination** |
| `snr` | Continuous | Peak waveform / background noise | Literature comparison only (not standalone filter) |
| `isolation_distance` | Continuous | Mahalanobis distance in PC space | Use with `isi_violations` for complementary assessment |
| `d_prime` | 0-∞ | Linear discriminant separability | Research in progress; use cautiously |
| `nn_hit_rate` | 0-1 | Nearest neighbor cluster membership | **Best overall quality proxy; easy to compare** |

---

## Detailed Metric Descriptions

### 1. firing_rate

**Calculation**: Total spikes / recording duration (seconds)

**Distribution**: Approximately lognormal (more at tail than normal distribution)

**Biases**:
- Over-estimated if unit poorly isolated (contaminating spikes included)
- Under-estimated if amplitude near threshold, drifted out, or data gaps
- Thalamus ~8 Hz; midbrain >20 Hz (regional variation)

**How to use**:
- Filter out units with too few spikes (e.g., `firing_rate > 0.1`)
- For local analyses, use interval-specific rate (some units drift out)

**Don't use for**: Isolation quality assessment alone

### 2. presence_ratio

**Range**: 0-0.99 (off-by-one error prevents reaching 1.0)

**What it measures**: Fraction of time during session unit was actively spiking

**Threshold**: >0.9 means unit present for ≥90% of recording

**Biases**:
- High ratio doesn't guarantee immunity to drift (amplitude can drift toward threshold)
- Low ratio can result from selective spiking (e.g., running-only neurons)

**How to use**:
- Detect drift across entire session
- Exclude units for session-wide comparisons
- If only analyzing short segment, relax this threshold to maximize units

**Pro tip**: Plot spike amplitudes over time to distinguish drift from selective firing

### 3. amplitude_cutoff

**Calculation**: Degree of truncation in spike amplitude distribution

**Range**: 0-0.5 (0.5 = cannot estimate, peak at minimum)

**Interpretation**: ~0.1 means ~10% of spikes estimated below threshold

**Biases**:
- Assumes symmetric amplitude distribution (invalid if waveform drifts)
- Weakly correlated with other quality metrics
- 44% of dataset has amplitude_cutoff = 0.5 (cannot estimate)

**How to use**:
- For high-precision spike timing, use low threshold (0.01 or lower)
- Complementary to `presence_ratio` for detecting incomplete units

### 4. isi_violations ⭐ (Most Direct Contamination Measure)

**Basis**: Neurons have refractory period (~1.5 ms). Spikes <1.5 ms apart = different neurons.

**Interpretation**: ISI violations = relative firing rate of contaminating neurons
- 0.5 = contaminating spikes at ~50% rate of "true" spikes
- Can exceed 1.0 for highly contaminated units

**Calculation**: Hill et al. (2011) J Neurosci 31: 8699-8705

**Biases**:
- Not stable over time (varies across session)
- Can miss merged units with non-overlapping firing periods
- Hippocampus highest (dense cells); midbrain lowest

**How to use**:
- **Conservative**: ISI < 0.5 (permissive but removes some contamination)
- **Strict**: ISI < 0.1 (removes most contaminated units)
- **For individual unit comparisons**: Use low threshold (≤0.1)
- **For population comparisons**: May tolerate higher contamination

**% units with ISI = 0**: ~37%

### 5. snr (Signal-to-Noise Ratio)

**Calculation**: Max amplitude of mean waveform / std dev of background noise (peak channel only)

**Why NOT recommended for Neuropixels**:
1. Only considers peak channel (waveforms span 12+ channels)
2. Drops dramatically with drift despite stable isolation

**How to use**:
- Literature comparison only
- Should NOT be primary isolation filter

**Regional pattern**: Hippocampus < Cortex < Thalamus/Midbrain

### 6. isolation_distance

**Basis**: Principal component (PC) based metric. Calculates Mahalanobis distance sphere.

**What it measures**: Separation of unit's PC cluster from neighbors' clusters in 96-D space

**Biases**:
- Sensitive to drift (waveform change reduces distance)
- Value depends on PC count (hard to compare cross-dataset)

**How to use**:
- Correlates with cluster quality; use with `isi_violations`
- Not a direct measure of contamination
- Typical range: 0-125

### 7. d_prime

**Basis**: Linear discriminant analysis on PCs. Calculates separability.

**Biases**:
- Not drift-tolerant
- Session-wide value is lower bound on instantaneous value
- Still being validated in literature

**How to use**: Research in progress; use cautiously

**Typical range**: 0-10

### 8. nn_hit_rate ⭐ (Recommended for QC)

**Basis**: Adapted from Chung, Magland et al. (2017)

**What it measures**: Fraction of nearest neighbors in PC space that belong to same cluster

**Interpretation**: High hit rate = well-isolated; low hit rate = contaminated

**Advantages**:
- Normalized to 0-1 scale (easy interpretation, cross-dataset comparison)
- Good overall quality proxy

**Biases**:
- Sensitive to drift
- Should be used with contamination measures like `isi_violations`

**Threshold**: >0.9 recommended

---

## Common Filtering Strategies

### Strategy 1: Visual Coding Default (Conservative)

Used in [Visual Coding Neuropixels](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels) dataset:

```python
units_filtered = units[
    (units.isi_violations < 0.5) &
    (units.amplitude_cutoff < 0.1) &
    (units.presence_ratio > 0.9)
]
# Result: ~37.7% pass (120,139 / 319,013 units)
```

**Removes**: Contaminated, incomplete, and drifted units

**Best for**: Session-wide analyses, single-unit comparisons

### Strategy 2: Lenient (SNR + Firing Rate)

```python
units_filtered = units[
    (units.snr > 1) &
    (units.firing_rate > 0.2)
]
# Result: ~81.1% pass (258,764 / 319,013 units)
```

**Best for**: Population-level analyses where some contamination acceptable

### Strategy 3: High Confidence (PC-based)

```python
units_filtered = units[
    (units.nn_hit_rate > 0.9) &
    (units.isi_violations < 0.5) &
    (units.amplitude_cutoff < 0.1)
]
# Result: Much smaller subset
```

**Best for**: Critical analyses, publications

### Strategy 4: Region-Specific

```python
# Adjust thresholds by recording area
cortex_units = units[
    (units.structure_acronym.isin(region_dict['cortex'])) &
    (units.firing_rate > 0.5) &
    (units.nn_hit_rate > 0.85)
]
```

**Rationale**: Different regions have different firing rates and waveform properties

### For Your Proof-of-Concept

Suggested starting point:

```python
good_units = units[
    (units['snr'] > 1) &
    (units['isi_violations'] < 1) &
    (units['firing_rate'] > 0.1)
]
# Moderate: ~70-75% of units
# Can tighten later if needed
```

**Reasoning**:
- SNR > 1: Reasonable baseline
- ISI violations < 1: Moderate contamination tolerance
- Firing rate > 0.1 Hz: Excludes silent/nearly-silent units

---

## Decision Tree: Which Metrics to Use

**Q: Comparing firing rates of individual units across areas?**
→ Use conservative filters (ISI < 0.1, amplitude_cutoff < 0.1, presence_ratio > 0.9)

**Q: Computing population firing rates (comparing areas)?**
→ May tolerate higher contamination; use moderate filters (ISI < 0.5)

**Q: High-precision spike timing analysis?**
→ Use `amplitude_cutoff < 0.01` to ensure spike recovery

**Q: Analyzing short segment only (not full session)?**
→ Relax `presence_ratio` threshold

**Q: Identifying cell types from waveforms?**
→ Less critical on isolation (morphology is primary feature)

---

## Key Caveats

**No ground truth**: Without simultaneous intracellular recordings or known spike times, we cannot validate sorting accuracy perfectly.

**Metrics aren't independent**: Firing rate biased by incomplete units; amplitude_cutoff weak with others.

**Time-varying quality**: Metrics calculated over full session; instantaneous quality may vary.

**Drift affects PC metrics**: `snr`, `isolation_distance`, `d_prime`, `nn_hit_rate` all degrade with electrode movement.

---

## Resources

- Quality metrics calculation: https://github.com/AllenInstitute/ecephys_spike_sorting
- SpikeInterface package: https://github.com/SpikeInterface/spikemetrics
- Hill et al. ISI violations: J Neurosci 31: 8699-8705 (2011)
- Chung et al. nn_hit_rate: Neuron 95, 1341–1356 (2017)

---

## Next Steps

- **Ready to filter?** Use Strategy 1, 2, or 4 above
- **Need analysis code?** → See [03_analysis_workflows.md](03_analysis_workflows.md)
- **Waveform confusion?** → See [04_waveform_physiology.md](04_waveform_physiology.md)
