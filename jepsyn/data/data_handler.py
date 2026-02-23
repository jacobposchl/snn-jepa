"""
Data handler for Visual Behavior Neuropixels dataset.

Provides clean access to downloaded session data with filtering and preprocessing.
Returns dataset components ready for analysis without additional preprocessing.

Example usage:
    handler = VBNDataHandler('./visual_behavior_neuropixels_data')
    
    # Get quality-filtered units from V1
    units = handler.get_units_by_area(1064644573, ['VISp'], apply_quality_filter=True)
    
    # Get image change trials (hits, misses, etc.)
    trials = handler.get_image_change_trials(1064644573, trial_types=['hit', 'miss'])
    
    # Get stimulus presentations from active behavior
    stim = handler.get_stimulus_presentations(1064644573, stimulus_blocks=[0], active_only=True)
"""

import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)


class VBNDataHandler:
    """Handler for accessing Visual Behavior Neuropixels data."""
    
    def __init__(self, cache_dir: Union[str, Path], metadata_only: bool = False):
        """
        Initialize data handler with cache directory.
        
        Args:
            cache_dir: Path to the visual behavior neuropixels cache directory
            metadata_only: If True, only use metadata tables and block session downloads
        """

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_only = metadata_only
        
        self.cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=self.cache_dir
        )
        self._sessions_table = None
        self._behavior_sessions_table = None
        self._units_table = None
        self._probes_table = None
        self._channels_table = None
    
    @property
    def sessions_table(self) -> pd.DataFrame:
        """Lazy-load and cache the sessions table."""
        if self._sessions_table is None:
            self._sessions_table = self.cache.get_ecephys_session_table()
        return self._sessions_table
    
    @property
    def behavior_sessions_table(self) -> pd.DataFrame:
        """Lazy-load and cache the behavior sessions table."""
        if self._behavior_sessions_table is None:
            self._behavior_sessions_table = self.cache.get_behavior_session_table()
        return self._behavior_sessions_table
    
    @property
    def units_table(self) -> pd.DataFrame:
        """Lazy-load and cache the units table."""
        if self._units_table is None:
            self._units_table = self.cache.get_unit_table()
        return self._units_table
    
    @property
    def probes_table(self) -> pd.DataFrame:
        """Lazy-load and cache the probes table."""
        if self._probes_table is None:
            self._probes_table = self.cache.get_probe_table()
        return self._probes_table
    
    @property
    def channels_table(self) -> pd.DataFrame:
        """Lazy-load and cache the channels table."""
        if self._channels_table is None:
            self._channels_table = self.cache.get_channel_table()
        return self._channels_table
    
    # Session Loading
    def load_session(self, session_id: int):
        """
        Load a single session with all data streams.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Session object with all data streams available
        """
        if self.metadata_only:
            raise RuntimeError(
                "Session data loading is disabled (metadata_only=True). "
                "Use metadata tables or initialize without metadata_only to load sessions."
            )
        return self.cache.get_ecephys_session(ecephys_session_id=session_id)
    
    def load_sessions(self, session_ids: List[int]) -> Dict[int, Any]:
        """
        Load multiple sessions.
        
        Args:
            session_ids: List of session IDs to load
            
        Returns:
            Dictionary mapping session_id -> session object
        """
        sessions = {}
        for session_id in session_ids:
            try:
                sessions[session_id] = self.load_session(session_id)
            except Exception as e:
                print(f"Failed to load session {session_id}: {e}")
        return sessions
    
    # Quality-Filtered Unit Access    
    def get_good_units(
        self,
        session_id: int,
        snr_threshold: float,
        isi_violation_threshold: float,
        min_firing_rate: float,
    ) -> pd.DataFrame:
        """
        Get quality-filtered units for a session.
        
        Applies standard quality filters used in Allen Institute analyses:
        - SNR > threshold (signal-to-noise ratio)
        - ISI violations < threshold (inter-spike interval violations)
        - Firing rate > minimum threshold
        
        Args:
            session_id: Session ID
            snr_threshold: Minimum signal-to-noise ratio (default: 1.0)
            isi_violation_threshold: Maximum ISI violations (default: 1.0)
            min_firing_rate: Minimum mean firing rate in Hz (default: 0.1)
            
        Returns:
            DataFrame of units passing quality filters, indexed by unit_id
        """
        session = self.load_session(session_id)
        units = session.get_units()
        
        quality_filter = (
            (units['snr'] > snr_threshold) &
            (units['isi_violations'] < isi_violation_threshold) &
            (units['firing_rate'] > min_firing_rate)
        )
        
        return units[quality_filter]
    
    def get_units_by_area(
        self,
        session_id: int,
        brain_areas: Union[str, List[str]],
        apply_quality_filter: bool = True,
        **quality_kwargs,
    ) -> pd.DataFrame:
        """
        Get units from specific brain areas.
        
        Args:
            session_id: Session ID
            brain_areas: Brain area(s) to filter by (e.g., 'VISp' or ['VISp', 'VISl'])
            apply_quality_filter: Whether to apply quality filters (default: True)
            **quality_kwargs: Additional arguments for get_good_units()
            
        Returns:
            DataFrame of units from specified brain areas, indexed by unit_id
        """
        if isinstance(brain_areas, str):
            brain_areas = [brain_areas]
        
        if apply_quality_filter:
            units = self.get_good_units(session_id, **quality_kwargs)
        else:
            session = self.load_session(session_id)
            units = session.get_units()
        
        area_filter = units['ecephys_structure_acronym'].isin(brain_areas)
        return units[area_filter]
    

    # Stimulus-Aligned Data Extraction
    def get_image_change_trials(
        self,
        session_id: int,
        trial_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get trials for image change detection task.
        
        Trial types:
        - 'hit': Mouse correctly detected image change
        - 'miss': Mouse failed to detect image change
        - 'false_alarm': Mouse licked when no change occurred (catch trial)
        - 'correct_reject': Mouse correctly withheld lick on catch trial
        - 'aborted': Trial was aborted
        - 'auto_rewarded': Reward was given automatically
        
        Args:
            session_id: Session ID
            trial_types: List of trial types to include (default: all types)
            
        Returns:
            DataFrame of trials, indexed by trial number
        """
        session = self.load_session(session_id)
        trials = session.trials.copy()
        
        if trial_types is not None:
            # Filter by trial types
            type_filter = pd.Series(False, index=trials.index)
            for trial_type in trial_types:
                if trial_type in trials.columns:
                    type_filter |= trials[trial_type]
            trials = trials[type_filter]
        
        return trials
    
    def get_stimulus_presentations(
        self,
        session_id: int,
        stimulus_blocks: Optional[List[int]] = None,
        active_only: bool = False,
    ) -> pd.DataFrame:
        """
        Get stimulus presentation table with optional filtering.
        
        Stimulus blocks:
        - Block 0: Active change detection task
        - Block 1: Brief gray screen
        - Block 2: Receptive field mapping (Gabors)
        - Block 3: Longer gray screen
        - Block 4: Full-field flashes
        - Block 5: Passive replay (same stimuli as block 0, no lickspout)
        
        Args:
            session_id: Session ID
            stimulus_blocks: List of stimulus blocks to include (default: all)
            active_only: If True, only return active behavior block (block 0)
            
        Returns:
            DataFrame of stimulus presentations
        """
        session = self.load_session(session_id)
        stim = session.stimulus_presentations.copy()
        
        if active_only:
            stim = stim[stim['active'] == True]
        elif stimulus_blocks is not None:
            stim = stim[stim['stimulus_block'].isin(stimulus_blocks)]
        
        return stim
    
    def get_receptive_field_presentations(self, session_id: int) -> pd.DataFrame:
        """
        Get Gabor presentations for receptive field mapping (block 2).
        
        Args:
            session_id: Session ID
            
        Returns:
            DataFrame of Gabor stimulus presentations with orientation, position, etc.
        """
        stim = self.get_stimulus_presentations(session_id, stimulus_blocks=[2])
        return stim
    


    # Behavioral Data Access
    def get_running_speed(
        self,
        session_id: int,
        time_range: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Get running speed data.
        
        Args:
            session_id: Session ID
            time_range: Optional (start_time, end_time) in seconds to filter data
            
        Returns:
            DataFrame with columns ['timestamps', 'speed']
        """
        session = self.load_session(session_id)
        running = session.running_speed.copy()
        
        if time_range is not None:
            start_time, end_time = time_range
            time_filter = (
                (running['timestamps'] >= start_time) &
                (running['timestamps'] <= end_time)
            )
            running = running[time_filter]
        
        return running
    
    def get_licks(
        self,
        session_id: int,
        time_range: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Get lick times.
        
        Args:
            session_id: Session ID
            time_range: Optional (start_time, end_time) in seconds to filter data
            
        Returns:
            DataFrame with lick timestamps
        """
        session = self.load_session(session_id)
        licks = session.licks.copy()
        
        if time_range is not None:
            start_time, end_time = time_range
            time_filter = (
                (licks['timestamps'] >= start_time) &
                (licks['timestamps'] <= end_time)
            )
            licks = licks[time_filter]
        
        return licks
    
    def get_pupil_data(
        self,
        session_id: int,
        remove_blinks: bool = True,
    ) -> pd.DataFrame:
        """
        Get eye tracking data.
        
        Args:
            session_id: Session ID
            remove_blinks: If True, remove frames marked as likely blinks
            
        Returns:
            DataFrame with pupil area, eye/pupil position, etc.
        """
        session = self.load_session(session_id)
        eye_tracking = session.eye_tracking.copy()
        
        if remove_blinks and 'likely_blink' in eye_tracking.columns:
            eye_tracking = eye_tracking[~eye_tracking['likely_blink']]
        
        return eye_tracking
    
    def get_rewards(self, session_id: int) -> pd.DataFrame:
        """
        Get reward delivery times and volumes.
        
        Args:
            session_id: Session ID
            
        Returns:
            DataFrame with reward timestamps and volumes
        """
        session = self.load_session(session_id)
        return session.rewards.copy()
    


    # Session Metadata & Filtering
    def filter_sessions(
        self,
        genotype: Optional[str] = None,
        experience_level: Optional[str] = None,
        brain_areas: Optional[List[str]] = None,
        min_units: Optional[int] = None,
    ) -> List[int]:
        """
        Filter available sessions by criteria.
        
        Args:
            genotype: Filter by genotype (partial match, e.g., 'Sst', 'Vip')
            experience_level: Filter by experience level ('Familiar' or 'Novel')
            brain_areas: Filter sessions recording from these areas
            min_units: Minimum number of units required
            
        Returns:
            List of session IDs matching all criteria
        """
        filtered = self.sessions_table.copy()
        
        if genotype:
            filtered = filtered[
                filtered['genotype'].str.contains(genotype, case=False, na=False)
            ]
        
        if experience_level:
            filtered = filtered[filtered['experience_level'] == experience_level]
        
        if brain_areas:
            mask = filtered['structure_acronyms'].apply(
                lambda areas: any(area in areas for area in brain_areas)
                if isinstance(areas, list) else False
            )
            filtered = filtered[mask]
        
        if min_units is not None:
            # Need to load sessions to count units - this could be slow
            valid_ids = []
            for session_id in filtered.index:
                try:
                    session = self.load_session(session_id)
                    if len(session.units) >= min_units:
                        valid_ids.append(session_id)
                except Exception:
                    continue
            return valid_ids
        
        return list(filtered.index)
    
    def get_data_availability(self, session_id: int) -> Dict[str, bool]:
        """
        Report which data streams are available for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary mapping data stream name -> availability boolean
        """
        session = self.load_session(session_id)
        availability = {}
        
        data_streams = [
            'units', 'spike_times', 'stimulus_presentations', 'trials',
            'running_speed', 'licks', 'rewards', 'eye_tracking'
        ]
        
        for stream in data_streams:
            try:
                data = getattr(session, stream)
                availability[stream] = data is not None and len(data) > 0
            except (AttributeError, Exception):
                availability[stream] = False
        
        return availability
    

    # Time Alignment Helpers
    def align_to_change_times(
        self,
        session_id: int,
        data_stream: str,
        window_before: float,
        window_after: float,
    ) -> List[pd.DataFrame]:
        """
        Align any data stream to image change times.
        
        Args:
            session_id: Session ID
            data_stream: Name of data stream ('running_speed', 'licks', 'eye_tracking')
            window_before: Time before change event in seconds
            window_after: Time after change event in seconds
            
        Returns:
            List of DataFrames, one per change event, with time relative to change
        """
        trials = self.get_image_change_trials(session_id)
        change_times = trials['change_time_no_display_delay'].dropna().values
        
        if data_stream == 'running_speed':
            data = self.get_running_speed(session_id)
        elif data_stream == 'licks':
            data = self.get_licks(session_id)
        elif data_stream == 'eye_tracking':
            data = self.get_pupil_data(session_id)
        else:
            raise ValueError(f"Unsupported data stream: {data_stream}")
        
        aligned_data = []
        for change_time in change_times:
            window_data = data[
                (data['timestamps'] >= change_time - window_before) &
                (data['timestamps'] <= change_time + window_after)
            ].copy()
            
            # Make timestamps relative to change time
            window_data['timestamps'] = window_data['timestamps'] - change_time
            aligned_data.append(window_data)
        
        return aligned_data
    
    def align_to_rewards(
        self,
        session_id: int,
        data_stream: str,
        window_before: float,
        window_after: float,
    ) -> List[pd.DataFrame]:
        """
        Align any data stream to reward times.
        
        Args:
            session_id: Session ID
            data_stream: Name of data stream ('running_speed', 'licks', 'eye_tracking')
            window_before: Time before reward in seconds
            window_after: Time after reward in seconds
            
        Returns:
            List of DataFrames, one per reward, with time relative to reward
        """
        rewards = self.get_rewards(session_id)
        reward_times = rewards['timestamps'].values
        
        if data_stream == 'running_speed':
            data = self.get_running_speed(session_id)
        elif data_stream == 'licks':
            data = self.get_licks(session_id)
        elif data_stream == 'eye_tracking':
            data = self.get_pupil_data(session_id)
        else:
            raise ValueError(f"Unsupported data stream: {data_stream}")
        
        aligned_data = []
        for reward_time in reward_times:
            window_data = data[
                (data['timestamps'] >= reward_time - window_before) &
                (data['timestamps'] <= reward_time + window_after)
            ].copy()
            
            # Make timestamps relative to reward time
            window_data['timestamps'] = window_data['timestamps'] - reward_time
            aligned_data.append(window_data)
        
        return aligned_data
    

    # Optotagging Utilities 
    def get_optotagging_table(self, session_id: int) -> pd.DataFrame:
        """
        Get optogenetic stimulation events.
        
        Useful for SST and VIP genotype experiments where cells are optogenetically
        tagged to identify cell types.
        
        Args:
            session_id: Session ID
            
        Returns:
            DataFrame with opto stimulation start times, durations, and conditions
        """
        session = self.load_session(session_id)
        return session.optogenetic_stimulation.copy()
    
    def identify_optotagged_units(
        self,
        session_id: int,
        response_threshold: float,
        baseline_window: tuple,
        response_window: tuple,
    ) -> List[int]:
        """
        Identify putative optotagged cells based on response to optogenetic stimulation.
        
        Compares firing rate during response window to baseline window.
        Units with response_threshold-fold increase are considered optotagged.
        
        Args:
            session_id: Session ID
            response_threshold: Fold-change threshold for considering unit optotagged
            baseline_window: Time window for baseline firing rate (relative to stim)
            response_window: Time window for response (relative to stim onset)
            
        Returns:
            List of unit IDs that are putatively optotagged
        """
        session = self.load_session(session_id)
        opto_table = self.get_optotagging_table(session_id)
        
        if len(opto_table) == 0:
            return []
        
        spike_times = session.spike_times
        units = session.units
        
        optotagged_units = []
        
        for unit_id in units.index:
            unit_spikes = spike_times[unit_id]
            
            # Calculate baseline and response firing rates
            baseline_rates = []
            response_rates = []
            
            for _, stim in opto_table.iterrows():
                stim_time = stim['start_time']
                
                # Baseline spikes
                baseline_spikes = np.sum(
                    (unit_spikes >= stim_time + baseline_window[0]) &
                    (unit_spikes < stim_time + baseline_window[1])
                )
                baseline_rate = baseline_spikes / (baseline_window[1] - baseline_window[0])
                baseline_rates.append(baseline_rate)
                
                # Response spikes
                response_spikes = np.sum(
                    (unit_spikes >= stim_time + response_window[0]) &
                    (unit_spikes < stim_time + response_window[1])
                )
                response_rate = response_spikes / (response_window[1] - response_window[0])
                response_rates.append(response_rate)
            
            # Check if mean response exceeds threshold
            mean_baseline = np.mean(baseline_rates)
            mean_response = np.mean(response_rates)
            
            if mean_baseline > 0 and (mean_response / mean_baseline) >= response_threshold:
                optotagged_units.append(unit_id)
        
        return optotagged_units
    

    # Batch Processing

    
    def iter_sessions(
        self,
        session_ids: List[int],
        data_streams: Optional[List[str]] = None,
    ):
        """
        Iterator for processing multiple sessions efficiently.
        
        Yields session data one at a time to avoid memory issues.
        
        Args:
            session_ids: List of session IDs to iterate over
            data_streams: List of data streams to load (None = load session object only)
            
        Yields:
            Tuple of (session_id, session_data)
            If data_streams is None, session_data is the full session object
            Otherwise, session_data is a dict mapping stream_name -> data
        """
        for session_id in session_ids:
            try:
                session = self.load_session(session_id)
                
                if data_streams is None:
                    yield session_id, session
                else:
                    data = {}
                    for stream in data_streams:
                        try:
                            data[stream] = getattr(session, stream)
                        except AttributeError:
                            data[stream] = None
                    yield session_id, data
                    
            except Exception as e:
                print(f"Failed to load session {session_id}: {e}")
                continue
    
    # Dataset Overview
    def print_dataset_overview(self):
        """Print comprehensive overview of the entire dataset."""
        
        print("\n" + "="*80)
        print("VISUAL BEHAVIOR NEUROPIXELS DATASET OVERVIEW")
        print("="*80)
        
        # Load all metadata tables
        sessions_table = self.sessions_table
        behavior_table = self.behavior_sessions_table
        units_table = self.units_table
        probes_table = self.probes_table
        channels_table = self.channels_table
        
        # ========== RECORDING SESSIONS ==========
        print("\n--- RECORDING SESSIONS ---")
        print(f"Total ecephys sessions: {len(sessions_table)}")
        print(f"Total unique mice: {sessions_table['mouse_id'].nunique()}")
        
        # Check for filtered sessions
        try:
            all_sessions = self.cache.get_ecephys_session_table(filter_abnormalities=False)
            if len(all_sessions) > len(sessions_table):
                filtered_count = len(all_sessions) - len(sessions_table)
                print(f"\nNote: {filtered_count} additional sessions exist but were filtered for quality issues")
                print(f"      (abnormal histology or electrical activity)")
                print(f"      Total recorded sessions: {len(all_sessions)}")
        except:
            pass
        
        print("\nSessions by image set:")
        print(sessions_table['image_set'].value_counts())
        
        print("\nSessions by experience level:")
        print(sessions_table['experience_level'].value_counts())
        
        print("\nSessions by session number (1st vs 2nd recording day):")
        print(sessions_table['session_number'].value_counts())
        
        print("\nSessions by genotype:")
        print(sessions_table['genotype'].value_counts())
        
        print("\nSessions by sex:")
        print(sessions_table['sex'].value_counts())
        
        # ========== PROBES ==========
        print("\n--- PROBE DATA ---")
        print(f"Total probe insertions: {len(probes_table)}")
        
        # Check unique sessions in probes table
        unique_sessions_with_probes = probes_table['ecephys_session_id'].nunique()
        filtered_sessions = unique_sessions_with_probes - len(sessions_table)
        
        print(f"\nSession Quality Control:")
        print(f"  Total sessions attempted (with probe data): {unique_sessions_with_probes}")
        print(f"  Quality-controlled sessions (in ecephys_sessions_table): {len(sessions_table)}")
        print(f"  Filtered out sessions (failed QC): {filtered_sessions}")
        
        # Verify this by checking if probe sessions exist in ecephys_sessions_table
        probe_session_ids = set(probes_table['ecephys_session_id'].unique())
        ecephys_session_ids = set(sessions_table.index)
        filtered_session_ids = probe_session_ids - ecephys_session_ids
        
        print(f"\nVerification:")
        print(f"  Sessions in probes_table not in ecephys_sessions_table: {len(filtered_session_ids)}")
        if len(filtered_session_ids) > 0:
            print(f"  (These {len(filtered_session_ids)} sessions were filtered for quality issues)")
        
        print(f"\nProbe insertion statistics:")
        print(f"  Average probes per session: {len(probes_table) / unique_sessions_with_probes:.1f}")
        
        probes_per_session = probes_table.groupby('ecephys_session_id').size()
        sessions_with_6_probes = (probes_per_session == 6).sum()
        sessions_with_5_or_more = (probes_per_session >= 5).sum()
        
        print(f"  Sessions with exactly 6 probes: {sessions_with_6_probes}")
        print(f"  Sessions with 5+ probes: {sessions_with_5_or_more}")
        print(f"  Min probes in a session: {probes_per_session.min()}")
        print(f"  Max probes in a session: {probes_per_session.max()}")
        
        # Check for probe_type column
        if 'probe_type' in probes_table.columns:
            print("\nProbe types:")
            print(probes_table['probe_type'].value_counts())
        elif 'model_name' in probes_table.columns:
            print("\nProbe models:")
            print(probes_table['model_name'].value_counts())
        
        # ========== CHANNELS ==========
        print("\n--- CHANNELS ---")
        print(f"Total channels: {len(channels_table)}")
        if len(probes_table) > 0:
            print(f"Average channels per probe: {len(channels_table) / len(probes_table):.0f}")
        
        # ========== UNITS ==========
        print("\n--- UNITS (NEURONS) ---")
        print(f"Total units: {len(units_table)}")
        print(f"Average units per session: {len(units_table) / len(sessions_table):.0f}")
        
        print("\nUnit quality distribution:")
        print(units_table['quality'].value_counts())
        quality_pct = (units_table['quality'].value_counts() / len(units_table) * 100).round(1)
        print(quality_pct)
        
        # Find the correct column name for brain areas
        area_col = None
        if 'ecephys_structure_acronym' in units_table.columns:
            area_col = 'ecephys_structure_acronym'
        elif 'structure_acronym' in units_table.columns:
            area_col = 'structure_acronym'
        
        if area_col:
            print("\nTop 15 brain areas by unit count:")
            top_areas = units_table[area_col].value_counts().head(15)
            for area, count in top_areas.items():
                pct = (count / len(units_table) * 100)
                print(f"  {area:10s}: {count:5d} units ({pct:5.1f}%)")
            
            print(f"\nTotal unique brain areas: {units_table[area_col].nunique()}")
        else:
            print("\nBrain area information not available")
        
        # ========== BEHAVIORAL DATA ==========
        print("\n--- BEHAVIORAL TRAINING ---")
        print(f"Total behavior sessions: {len(behavior_table)}")
        
        print("\nBehavior sessions by equipment:")
        print(behavior_table['equipment_name'].value_counts())
        
        print("\nTop training stages:")
        top_stages = behavior_table['session_type'].value_counts().head(10)
        for stage, count in top_stages.items():
            print(f"  {stage}: {count}")
        
        # ========== SUMMARY STATISTICS ==========
        print("\n--- SUMMARY STATISTICS ---")
        print(f"\nUnit quality metrics (good units only):")
        good_units = units_table[units_table['quality'] == 'good']
        print(f"  Good units: {len(good_units)} ({len(good_units)/len(units_table)*100:.1f}%)")
        
        # Safely access optional columns
        metrics = {
            'snr': ('SNR', '.2f'),
            'firing_rate': ('Firing rate', '.2f Hz'),
            'isolation_distance': ('Isolation distance', '.2f'),
            'amplitude': ('Amplitude', '.2f µV'),
        }
        
        for col, (label, fmt) in metrics.items():
            if col in good_units.columns:
                val = good_units[col].mean()
                if fmt.endswith('Hz'):
                    print(f"  Avg {label}: {val:.2f} Hz")
                elif fmt.endswith('µV'):
                    print(f"  Avg {label}: {val:.2f} µV")
                else:
                    print(f"  Avg {label}: {val:.2f}")
        
        print(f"\nWaveform metrics (good units):")
        waveform_metrics = {
            'waveform_duration': ('waveform duration', 'ms'),
            'amplitude_cutoff': ('amplitude cutoff', '%'),
            'isi_violations': ('ISI violations', ''),
        }
        
        for col, (label, unit) in waveform_metrics.items():
            if col in good_units.columns:
                val = good_units[col].mean()
                if unit == '%':
                    print(f"  Avg {label}: {val:.2%}")
                elif unit == 'ms':
                    print(f"  Avg {label}: {val:.2f} {unit}")
                else:
                    print(f"  Avg {label}: {val:.4f}")
        
        # ========== BRAIN AREA COVERAGE ==========
        print("\n--- BRAIN AREA COVERAGE ---")
        unique_areas_per_session = sessions_table['structure_acronyms'].apply(
            lambda x: len(eval(x)) if isinstance(x, str) else 0
        )
        print(f"Average brain areas per session: {unique_areas_per_session.mean():.1f}")
        print(f"Max brain areas in single session: {unique_areas_per_session.max()}")
        
        print("\n" + "="*80 + "\n")

