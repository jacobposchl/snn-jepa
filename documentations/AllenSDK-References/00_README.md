# Allen SDK Visual Behavior Neuropixels - Reference Guide

Quick reference documentation extracted from AllenSDK notebooks for easy lookup during development.

## Files Overview

| File | Purpose | Quick Link |
|------|---------|-----------|
| [01_dataset_overview.md](01_dataset_overview.md) | Dataset structure, metadata tables, training pipeline, genotypes | Start here for understanding data organization |
| [02_quality_metrics.md](02_quality_metrics.md) | Unit quality metrics explained (8 metrics), filtering strategies, code examples | Use for deciding which filters to apply |
| [03_analysis_workflows.md](03_analysis_workflows.md) | Common analysis patterns: PSTH, receptive fields, optotagging with copy-paste code | Use when implementing analyses |
| [04_waveform_physiology.md](04_waveform_physiology.md) | Waveform metrics, cell-type identification, region-specific morphology | Use for understanding waveform features |
| [05_behavioral_alignment.md](05_behavioral_alignment.md) | Stimulus/trials table structure, behavioral alignment, response latency calculations | Use for aligning running/pupil/licks to events |

## Quick Start

**Just want to load data?**
→ See [01_dataset_overview.md](01_dataset_overview.md#data-access-methods)

**Need to filter units?**
→ See [02_quality_metrics.md](02_quality_metrics.md#common-filtering-strategies)

**Building an analysis?**
→ See [03_analysis_workflows.md](03_analysis_workflows.md)

**Confused about a waveform metric?**
→ See [04_waveform_physiology.md](04_waveform_physiology.md)

**Need to align behavior (running/pupil/licks) to stimulus events?**
→ See [05_behavioral_alignment.md](05_behavioral_alignment.md)

---

## Dataset At a Glance

- **153** ecephys (Neuropixels) recording sessions
- **319,013** total units across all sessions
- **3** genotypes: wt, SST, VIP
- **2** image sets (G, H) with 2 recording days per mouse
- **4** major brain regions: cortex, thalamus, hippocampus, midbrain
- **Spike sorting**: Kilosort2 + automatic noise filtering
- **All units returned by default** (no pre-filtering by Allen Institute)

**Key**: Probes moved ~100 μm between session 1 and 2 → cannot track neurons across days

---

## Common Tasks & Where to Find Info

### Data Access
- Load all sessions metadata: [01_dataset_overview.md#data-access-methods](01_dataset_overview.md#data-access-methods)
- Load single session: [01_dataset_overview.md#session-loading--attributes](01_dataset_overview.md#session-loading--attributes)
- Understand table relationships: [01_dataset_overview.md#metadata-tables-structure](01_dataset_overview.md#metadata-tables-structure)

### Unit Filtering
- Understand metrics: [02_quality_metrics.md#overview-table](02_quality_metrics.md#overview-table)
- Quick filtering code: [02_quality_metrics.md#common-filtering-strategies](02_quality_metrics.md#common-filtering-strategies)
- Visual Coding defaults: [02_quality_metrics.md#example-1-visual-coding-default](02_quality_metrics.md#example-1-visual-coding-default)

### Analysis Implementation
- PSTH (peri-stimulus time histogram): [03_analysis_workflows.md#pattern-1-psth](03_analysis_workflows.md#pattern-1-psth)
- Receptive field mapping: [03_analysis_workflows.md#pattern-2-receptive-field-mapping](03_analysis_workflows.md#pattern-2-receptive-field-mapping)
- Optotagging (SST/VIP identification): [03_analysis_workflows.md#pattern-3-optotagging](03_analysis_workflows.md#pattern-3-optotagging)

### Cell Type & Morphology
- Cell-type identification from waveforms: [04_waveform_physiology.md#waveform-based-cell-classification](04_waveform_physiology.md#waveform-based-cell-classification)
- Regional differences: [04_waveform_physiology.md#physiological-interpretation](04_waveform_physiology.md#physiological-interpretation)
- Waveform metric definitions: [04_waveform_physiology.md#waveform-metrics-reference](04_waveform_physiology.md#waveform-metrics-reference)

---

## Key Concepts

**Quality Metrics**: 8 metrics assess unit isolation quality. No single "good" threshold—choose based on your analysis tolerance for contamination.

**Spike Sorting**: Kilosort2-sorted, validated against noise artifacts. Errors still occur (over-merging, under-amplitude, drift). Quality metrics estimate severity.

**Genotypes**:
- **wt**: Wild-type, no optotagging
- **SST**: ChR2-expressing somatostatin interneurons, optotag during recording
- **VIP**: ChR2-expressing VIP interneurons, optotag during recording

**Training**: Mice trained on change detection task with visual stimuli. Two recording sessions per mouse: one with familiar image set, one with novel.

**Brain Areas**: Cortex (visual), thalamus (LGd, etc.), hippocampus (CA1, etc.), midbrain (superior colliculus, etc.)

---

## Reference Sources

All information extracted from Allen Institute notebooks:
- `visual_behavior_neuropixels_data_access.ipynb` - Dataset structure
- `visual_behavior_neuropixels_dataset_manifest.ipynb` - Metadata and training pipeline
- `visual_behavior_neuropixels_quickstart.ipynb` - Analysis patterns
- `visual_behavior_neuropixels_quality_metrics.ipynb` - Quality metrics deep dive

Full SDK documentation: https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html
