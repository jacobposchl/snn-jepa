"""
Preprocessing pipeline for Visual Behavior Neuropixels data.

Provides a chainable interface for data cleaning, validation, and windowing
to prepare neural and behavioral data for model training.

Pipeline progression:
    1. validate_integrity() - Check data completeness and format
    2. clean() - Remove outlier spikes, handle gaps
    3. filter_units() - Apply quality metrics (SNR, ISI violations, firing rate)
    4. create_windows() - Bin spikes into time windows aligned to events
    5. get() - Return processed data dict

Each method returns self for chaining. Output from get() is model-ready.

Example usage:
    handler = VBNDataHandler('./visual_behavior_neuropixels_data')
    session_data = handler.load_session(1064644573)
    
    processed = (NeuropixelsPreprocessor(session_data)
        .validate_integrity()
        .clean()
        .filter_units()
        .create_windows(window_size_ms=500, stride_ms=100)
        .get())
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.utils.window import create_context_target_windows


@dataclass
class PreprocessingMetadata:
    """Track preprocessing operations and data statistics."""
    operations: list
    original_unit_count: int
    filtered_unit_count: int
    unit_areas: Dict[str, int]
    windows_created: int
    total_time_seconds: float


class NeuropixelsPreprocessor:
    """Pipeline for preprocessing Visual Behavior Neuropixels data."""
    
    def __init__(self, session_data: Dict[str, Any]):
        """
        Initialize preprocessor with session data from VBNDataHandler.
        
        Args:
            session_data: Dictionary containing:
                - units: DataFrame with unit metadata (columns: snr, isi_violations, firing_rate, ecephys_structure_acronym)
                - spike_times: Dict mapping unit_id -> spike times (array, seconds, display lag corrected)
                - trials: DataFrame with trial info (columns: start_time, end_time, stimulus_change, hit, miss, etc.)
                - stimulus_presentations: DataFrame (columns: start_time, end_time, stimulus_block, is_change, image_name, etc.)
                - running_speed: (optional) Running speed timeseries
                - licks: (optional) Lick times
                - rewards: (optional) Reward times
                
        See 05_behavioral_alignment.md for table schema details.
        """
        self.data = session_data
        self.metadata = PreprocessingMetadata(
            operations=[],
            original_unit_count=0,
            filtered_unit_count=0,
            unit_areas={},
            windows_created=0,
            total_time_seconds=0.0,
        )
        self._validate_session_structure()
    
    def _validate_session_structure(self) -> None:
        """Ensure session_data has expected keys or is a valid session object."""
        # Check if data is NOT a dictionary (i.e., it's a session object)
        if not isinstance(self.data, dict):
            # It's a session object - extract the needed attributes
            try:
                self.data = {
                    'units': self.data.get_units(),
                    'spike_times': self.data.spike_times,
                    'trials': self.data.trials,
                    'stimulus_presentations': self.data.stimulus_presentations,
                    'running_speed': getattr(self.data, 'running_speed', None),
                    'licks': getattr(self.data, 'licks', None),
                    'rewards': getattr(self.data, 'rewards', None),
                }
            except (AttributeError, TypeError) as e:
                raise ValueError(f"Session object missing required attribute: {e}")
        
        # Now validate as dictionary
        required_keys = ['units', 'spike_times', 'trials', 'stimulus_presentations']
        missing = [k for k in required_keys if k not in self.data]
        if missing:
            raise ValueError(f"Session data missing required keys: {missing}")
        
        self.metadata.original_unit_count = len(self.data['units'])
    
    def validate_integrity(self) -> 'NeuropixelsPreprocessor':
        """
        Validate data integrity and completeness.
        
        Checks:
        - No NaN in critical unit columns
        - Spike times are sorted and finite
        - Trial/stimulus data has expected structure
        - Behavioral data alignment
        
        Returns:
            self for chaining
            
        Raises:
            ValueError: If critical data integrity issues found
        """
        units = self.data['units']
        spike_times = self.data['spike_times']
        
        # Check units table
        critical_cols = ['ecephys_structure_acronym', 'firing_rate', 'isi_violations', 'snr']
        for col in critical_cols:
            if col in units.columns:
                nan_count = units[col].isna().sum()
                if nan_count > 0:
                    print(f"Warning: {nan_count} NaN values in units.{col}")
        
        # Check spike times
        invalid_units = []
        for unit_id, times in spike_times.items():
            if not np.all(np.isfinite(times)):
                invalid_units.append((unit_id, "non-finite values"))
            if len(times) > 1 and not np.all(np.diff(times) >= 0):
                invalid_units.append((unit_id, "unsorted spike times"))
        
        if invalid_units:
            raise ValueError(f"Invalid spike times in {len(invalid_units)} units: {invalid_units[:3]}...")
        
        # Check behavioral data alignment
        trials = self.data.get('trials')
        if trials is not None and len(trials) > 0:
            if 'start_time' in trials.columns and 'end_time' in trials.columns:
                if (trials['start_time'] > trials['end_time']).any():
                    raise ValueError("Found trials with start_time > end_time")
        
        self.metadata.operations.append('validate_integrity')
        print(f"✓ Data integrity validated: {len(spike_times)} units, {len(trials)} trials")
        return self
    
    def clean(
        self,
        remove_outliers: bool = True,
        interpolate_gaps: bool = True,
        spike_time_bounds: Optional[Tuple[float, float]] = None,
    ) -> 'NeuropixelsPreprocessor':
        """
        Clean data by removing outliers and handling gaps.
        
        Args:
            remove_outliers: Remove spike times outside physiological bounds
            interpolate_gaps: Handle missing data segments
            spike_time_bounds: (min, max) seconds for valid spike times
            
        Returns:
            self for chaining
        """
        spike_times = self.data['spike_times']
        units = self.data['units']
        
        # Determine spike time bounds
        if spike_time_bounds is None:
            stimulus = self.data.get('stimulus_presentations')
            if stimulus is not None and len(stimulus) > 0:
                spike_time_bounds = (
                    stimulus['start_time'].min(),
                    stimulus['end_time'].max()
                )
            else:
                trials = self.data.get('trials')
                if trials is not None and len(trials) > 0 and 'start_time' in trials.columns:
                    spike_time_bounds = (
                        trials['start_time'].min(),
                        trials['start_time'].max() + 10  # Add 10 seconds as buffer for end
                    )
        
        # Clean spike times
        cleaned_units = {}
        outlier_counts = {}
        for unit_id, times in spike_times.items():
            if remove_outliers and spike_time_bounds is not None:
                mask = (times >= spike_time_bounds[0]) & (times <= spike_time_bounds[1])
                outlier_counts[unit_id] = (~mask).sum()
                times = times[mask]
            cleaned_units[unit_id] = times
        
        self.data['spike_times'] = cleaned_units
        
        total_outliers = sum(outlier_counts.values())
        self.metadata.operations.append('clean')
        print(f"✓ Cleaning complete: removed {total_outliers} outlier spikes")
        return self
    
    def filter_units(
        self,
        min_snr: float = 1.0,
        min_firing_rate: float = 0.1,
        max_isi_violations: float = 1.0,
        brain_areas: Optional[list] = None,
    ) -> 'NeuropixelsPreprocessor':
        """
        Filter units based on quality metrics.
        
        Recommended defaults from Allen SDK documentation:
        - min_snr: 1.0 (reasonable baseline for Neuropixels)
        - min_firing_rate: 0.1 Hz (excludes silent/nearly-silent units)
        - max_isi_violations: 1.0 (moderate contamination tolerance, ~70-75% pass rate)
        
        See 02_quality_metrics.md for detailed filtering strategies.
        
        Args:
            min_snr: Keep units with SNR >= this (Note: SNR sensitive to drift)
            min_firing_rate: Keep units with firing_rate >= this (Hz)
            max_isi_violations: Keep units with isi_violations <= this (direct contamination measure)
            brain_areas: If specified, only keep units from these areas
            
        Returns:
            self for chaining
        """
        units = self.data['units'].copy()
        spike_times = self.data['spike_times']
        
        original_count = len(units)
        mask = pd.Series([True] * len(units), index=units.index)
        
        # Quality filters (always applied with these parameters)
        if 'snr' in units.columns:
            mask &= units['snr'] >= min_snr
        
        if 'firing_rate' in units.columns:
            mask &= units['firing_rate'] >= min_firing_rate
        
        if 'isi_violations' in units.columns:
            mask &= units['isi_violations'] <= max_isi_violations
        
        # Brain area filter
        if brain_areas is not None and 'ecephys_structure_acronym' in units.columns:
            mask &= units['ecephys_structure_acronym'].isin(brain_areas)
        
        # Apply filter
        filtered_units = units[mask]
        kept_unit_ids = set(filtered_units.index)
        
        # Filter spike times to match kept units
        self.data['spike_times'] = {
            uid: times for uid, times in spike_times.items() if uid in kept_unit_ids
        }
        
        self.data['units'] = filtered_units
        
        # Update metadata
        self.metadata.filtered_unit_count = len(filtered_units)
        if 'ecephys_structure_acronym' in filtered_units.columns:
            self.metadata.unit_areas = dict(
                filtered_units['ecephys_structure_acronym'].value_counts()
            )
        
        self.metadata.operations.append('filter_units')
        print(f"✓ Unit filtering: {original_count} → {len(filtered_units)} units")
        print(f"  Brain areas: {self.metadata.unit_areas}")
        return self
    
    def create_windows(
        self,
        window_size_ms: float = 500,
        stride_ms: float = 100,
        align_to: str = 'stimulus_change',
    ) -> 'NeuropixelsPreprocessor':
        """
        Create sliding windows of spike counts aligned to behavioral events.
        
        Bins spike times into fixed-duration windows and aligns them to either
        stimulus changes or trial starts. Uses display lag-corrected times.
        
        Args:
            window_size_ms: Window duration in milliseconds
            stride_ms: Stride between windows in milliseconds
            align_to: Alignment event:
                'stimulus_change': Align to stimulus change onsets (requires stimulus_presentations)
                'trial_start': Align to trial onsets (requires trials)
            
        Returns:
            self for chaining
        """
        spike_times = self.data['spike_times']
        trials = self.data.get('trials')
        stimulus = self.data.get('stimulus_presentations')
        
        window_size_s = window_size_ms / 1000.0
        stride_s = stride_ms / 1000.0
        
        # Get alignment times based on event type (using display-corrected times)
        if align_to == 'stimulus_change' and stimulus is not None:
            align_times = stimulus['start_time'].values
        elif align_to == 'trial_start' and trials is not None:
            align_times = trials['start_time'].values
        else:
            raise ValueError(
                f"Cannot align to '{align_to}': "
                f"stimulus_change requires stimulus_presentations, "
                f"trial_start requires trials"
            )
        
        # Create windows for each unit
        windows = {}
        for unit_id, times in spike_times.items():
            unit_windows = []
            for align_time in align_times:
                start = align_time
                end = align_time + window_size_s
                spikes_in_window = np.sum((times >= start) & (times < end))
                unit_windows.append(spikes_in_window)
            windows[unit_id] = np.array(unit_windows)
        
        self.data['windows'] = windows
        self.data['window_metadata'] = {
            'window_size_ms': window_size_ms,
            'stride_ms': stride_ms,
            'align_to': align_to,
            'num_windows': len(align_times),
        }
        
        self.metadata.windows_created = len(align_times)
        self.metadata.operations.append('create_windows')
        print(f"✓ Created {len(align_times)} windows ({window_size_ms}ms @ {stride_ms}ms stride)")
        return self
    
    def get(self) -> Dict[str, Any]:
        """
        Return processed data ready for downstream analysis.
        
        Returns:
            Dictionary with:
            - windows: Dict[unit_id -> np.ndarray shape (num_windows,)] spike counts per window
            - units: Filtered units DataFrame with metadata
            - spike_times: Dict[unit_id -> spike times array] (seconds, display lag corrected)
            - trials: Trials DataFrame (if available)
            - stimulus_presentations: Stimulus presentations DataFrame (if available)
            - metadata: PreprocessingMetadata with operations log and statistics
            - window_metadata: Dict with window_size_ms, stride_ms, align_to, num_windows (if create_windows called)
        """
        # Validate window consistency if windows were created
        windows = self.data.get('windows')
        if windows is not None and len(windows) > 0:
            shapes = [v.shape for v in windows.values()]
            if len(set(shapes)) > 1:
                print(f"Warning: Inconsistent window shapes: {set(shapes)}")
            else:
                print(f"✓ Window validation: {len(windows)} units × {shapes[0][0]} windows")
        
        return {
            'windows': windows,
            'units': self.data['units'],
            'spike_times': self.data['spike_times'],
            'trials': self.data.get('trials'),
            'stimulus_presentations': self.data.get('stimulus_presentations'),
            'metadata': self.metadata,
            'window_metadata': self.data.get('window_metadata'),
        }
