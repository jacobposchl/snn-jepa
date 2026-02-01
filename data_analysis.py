"""
Data analysis utilities for exploring Visual Behavior Neuropixels dataset structure.

Provides high-level summaries of session content, active vs passive blocks,
trial statistics, and unit inventory.

Usage:
    python data_analysis.py --overview
    python data_analysis.py --session <session_id>
    python data_analysis.py --all-sessions
    python data_analysis.py --filter --animals <ids|all> --regions <areas> --units-required <counts> --phase <1|2>
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.data.data_handler import VBNDataHandler


# Reference: AllenSDK Visual Behavior Neuropixels metadata tables
# See documentations/AllenSDK/visual_behavior_neuropixels_dataset_manifest.ipynb


def analyze_session_structure(handler: VBNDataHandler, session_id: int) -> Dict:
    """
    Analyze the complete structure and content of a session.
    
    Returns:
        Dictionary with stimulus blocks, trial counts, unit inventory, and duration.
    """
    session = handler.load_session(session_id)
    stim = session.stimulus_presentations
    trials = session.trials
    
    # Get units from the units table for this session
    units = handler.units_table[handler.units_table['ecephys_session_id'] == session_id]
    
    # Group by stimulus block
    block_info = {}
    for block_id in sorted(stim['stimulus_block'].unique()):
        block_data = stim[stim['stimulus_block'] == block_id]
        
        block_info[block_id] = {
            'duration_sec': float(block_data['end_time'].max() - block_data['start_time'].min()),
            'num_presentations': len(block_data),
            'num_unique_images': block_data['image_name'].nunique() if 'image_name' in block_data.columns else 0,
        }
    
    # Trial counts (active block only)
    trial_counts = {}
    if len(trials) > 0:
        trial_counts = {
            'total': len(trials),
            'hit': int(trials['hit'].sum()) if 'hit' in trials.columns else 0,
            'miss': int(trials['miss'].sum()) if 'miss' in trials.columns else 0,
            'false_alarm': int(trials['false_alarm'].sum()) if 'false_alarm' in trials.columns else 0,
            'correct_reject': int(trials['correct_reject'].sum()) if 'correct_reject' in trials.columns else 0,
        }
    
    # Unit counts by area - check for correct column name
    area_col = 'ecephys_structure_acronym' if 'ecephys_structure_acronym' in units.columns else 'structure_acronym'
    units_by_area = units[area_col].value_counts().to_dict() if len(units) > 0 and area_col in units.columns else {}
    
    return {
        'session_id': session_id,
        'total_duration_sec': float(stim['end_time'].max()),
        'stimulus_blocks': block_info,
        'trial_counts': trial_counts,
        'total_units': len(units),
        'units_by_area': units_by_area,
    }


def compare_active_vs_passive(handler: VBNDataHandler, session_id: int) -> Dict:
    """
    Compare active behavior block (0) vs passive replay block (usually last block).
    
    Returns:
        Dictionary with stimulus and trial statistics for each block type.
    """
    stim = handler.get_stimulus_presentations(session_id)
    
    # Identify active block (stimulus_block == 0) and passive block (highest stimulus_block)
    active = stim[stim['stimulus_block'] == 0]
    passive_block_ids = stim['stimulus_block'].unique()
    passive_block_id = max([b for b in passive_block_ids if b != 0]) if len(passive_block_ids) > 1 else None
    passive = stim[stim['stimulus_block'] == passive_block_id] if passive_block_id else pd.DataFrame()
    
    def summarize_block(block_stim):
        if len(block_stim) == 0:
            return {'num_stimuli': 0, 'duration_sec': 0, 'unique_images': 0}
        return {
            'num_stimuli': len(block_stim),
            'duration_sec': float(block_stim['end_time'].max() - block_stim['start_time'].min()),
            'unique_images': int(block_stim['image_name'].nunique()) if 'image_name' in block_stim.columns else 0,
        }
    
    return {
        'active_block': {**summarize_block(active), 'block_id': 0},
        'passive_block': {**summarize_block(passive), 'block_id': passive_block_id},
    }


def _parse_animals(animals: Optional[List[str]]) -> Optional[List[str]]:
    if not animals:
        return None
    if len(animals) == 1 and animals[0].lower() == "all":
        return None
    return [str(a) for a in animals]


def filter_sessions_metadata(
    handler: VBNDataHandler,
    animals: Optional[List[str]],
    regions: Optional[List[str]],
    units_required: Optional[List[int]],
    phase: Optional[int],
) -> pd.DataFrame:
    """
    Filter sessions using metadata tables only.

    Args:
        animals: Mouse IDs to include (strings). None means all.
        regions: Brain regions to include (e.g., ['VISp', 'CA1']).
        units_required: Minimum units per region (same length as regions).
        phase: Session number (1 or 2).

    Returns:
        Filtered sessions table.
    """
    sessions = handler.sessions_table.copy()

    # Filter by session phase
    if phase is not None:
        sessions = sessions[sessions['session_number'] == phase]

    # Filter by animals (mouse_id)
    animal_ids = _parse_animals(animals)
    if animal_ids is not None:
        sessions = sessions[sessions['mouse_id'].astype(str).isin(animal_ids)]

    # Filter by regions + units required using units table
    if regions:
        if not units_required or len(units_required) != len(regions):
            raise ValueError("--units-required must be provided with the same length as --regions")

        units = handler.units_table
        area_col = 'ecephys_structure_acronym' if 'ecephys_structure_acronym' in units.columns else 'structure_acronym'

        # Build counts per session and region
        region_counts = (
            units[units[area_col].isin(regions)]
            .groupby(['ecephys_session_id', area_col])
            .size()
            .unstack(fill_value=0)
        )

        # Enforce minimum unit counts per region
        for region, min_units in zip(regions, units_required):
            if region not in region_counts.columns:
                region_counts[region] = 0
            region_counts = region_counts[region_counts[region] >= min_units]

        eligible_session_ids = set(region_counts.index)
        sessions = sessions[sessions.index.isin(eligible_session_ids)]

    return sessions


def summarize_all_sessions(
    handler: VBNDataHandler, 
    session_ids: List[int],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create summary dataframe across all sessions.
    
    Returns:
        DataFrame with one row per session, key metrics as columns.
    """
    summaries = []
    
    for session_id in session_ids:
        try:
            info = analyze_session_structure(handler, session_id)
            comparison = compare_active_vs_passive(handler, session_id)
            
            summaries.append({
                'session_id': session_id,
                'total_duration_sec': info['total_duration_sec'],
                'total_units': info['total_units'],
                'num_brain_areas': len(info['units_by_area']),
                'active_stimuli': comparison['active_block']['num_stimuli'],
                'active_duration_sec': comparison['active_block']['duration_sec'],
                'passive_stimuli': comparison['passive_block']['num_stimuli'],
                'passive_duration_sec': comparison['passive_block']['duration_sec'],
                'total_trials': info['trial_counts'].get('total', 0),
                'hit_trials': info['trial_counts'].get('hit', 0),
            })
        except Exception as e:
            if verbose:
                print(f"Error analyzing session {session_id}: {e}")
            continue
    
    return pd.DataFrame(summaries)


def print_full_summary(handler: VBNDataHandler, session_ids: List[int]):
    """
    Print comprehensive summary of all sessions.
    """
    print("=" * 80)
    print("DATA ANALYSIS SUMMARY")
    print("=" * 80)
    
    for session_id in session_ids:
        try:
            info = analyze_session_structure(handler, session_id)
            comparison = compare_active_vs_passive(handler, session_id)
            
            print(f"\n{'='*80}")
            print(f"SESSION {session_id}")
            print(f"{'='*80}")
            
            print(f"\nSession Duration: {info['total_duration_sec']:.1f} seconds ({info['total_duration_sec']/60:.1f} min)")
            print(f"Total Units: {info['total_units']}")
            print(f"Number of Brain Areas: {len(info['units_by_area'])}")
            
            print(f"\nUnits by Brain Area:")
            for area, count in sorted(info['units_by_area'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {area}: {count}")
            
            print(f"\n{'─'*80}")
            print(f"STIMULUS BLOCKS BREAKDOWN ({len(info['stimulus_blocks'])} blocks)")
            print(f"{'─'*80}")
            
            for block_id in sorted(info['stimulus_blocks'].keys()):
                block = info['stimulus_blocks'][block_id]
                block_type = "Active" if block_id == 0 else "Passive/Replay"
                
                print(f"\nBlock {block_id} ({block_type}):")
                print(f"  Duration: {block['duration_sec']:.1f} seconds ({block['duration_sec']/60:.2f} min)")
                print(f"  Stimulus Presentations: {block['num_presentations']}")
                print(f"  Unique Images: {block['num_unique_images']}")
                if block['num_presentations'] > 0:
                    avg_freq = block['num_presentations'] / block['duration_sec']
                    print(f"  Average Frequency: {avg_freq:.2f} presentations/sec")
            
            print(f"\n{'─'*80}")
            print(f"ACTIVE VS PASSIVE COMPARISON")
            print(f"{'─'*80}")
            active = comparison['active_block']
            passive = comparison['passive_block']
            
            print(f"\nActive Block (Block 0):")
            print(f"  Duration: {active['duration_sec']:.1f}s")
            print(f"  Stimuli: {active['num_stimuli']}")
            print(f"  Unique Images: {active['unique_images']}")
            
            print(f"\nPassive Block (Block {passive['block_id']}):")
            print(f"  Duration: {passive['duration_sec']:.1f}s")
            print(f"  Stimuli: {passive['num_stimuli']}")
            print(f"  Unique Images: {passive['unique_images']}")
            
            if info['trial_counts']:
                print(f"\n{'─'*80}")
                print(f"BEHAVIORAL PERFORMANCE (Active Block Only)")
                print(f"{'─'*80}")
                trials = info['trial_counts']
                print(f"\nTotal Trials: {trials['total']}")
                print(f"  Hits: {trials['hit']}")
                print(f"  Misses: {trials['miss']}")
                print(f"  False Alarms: {trials['false_alarm']}")
                print(f"  Correct Rejects: {trials['correct_reject']}")
                
                if trials['total'] > 0:
                    hit_rate = trials['hit'] / (trials['hit'] + trials['miss']) if (trials['hit'] + trials['miss']) > 0 else 0
                    fa_rate = trials['false_alarm'] / (trials['false_alarm'] + trials['correct_reject']) if (trials['false_alarm'] + trials['correct_reject']) > 0 else 0
                    print(f"\nPerformance Metrics:")
                    print(f"  Hit Rate: {hit_rate:.2%}")
                    print(f"  False Alarm Rate: {fa_rate:.2%}")
                    dprime = 2.66 * (hit_rate - fa_rate) if (hit_rate > 0 and fa_rate < 1) else 0
                    print(f"  d': {dprime:.2f}")
        
        except Exception as e:
            print(f"\nError analyzing session {session_id}: {e}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Visual Behavior Neuropixels dataset')
    parser.add_argument('--overview', action='store_true', help='Print overview of entire dataset')
    parser.add_argument('--session', type=int, help='Analyze specific session by ID')
    parser.add_argument('--all-sessions', action='store_true', help='Analyze all sessions')
    parser.add_argument('--filter', action='store_true', help='List sessions matching filters (metadata only)')
    parser.add_argument('--animals', nargs='+', help='Mouse IDs to include, or "all"')
    parser.add_argument('--regions', nargs='+', help='Brain regions to include (e.g., VISp CA1)')
    parser.add_argument('--units-required', nargs='+', type=int,
                        help='Minimum units per region (same length as --regions)')
    parser.add_argument('--phase', type=int, choices=[1, 2],
                        help='Session phase: 1 (first) or 2 (second)')
    parser.add_argument('--cache-dir', type=str, default='./visual_behavior_neuropixels_data', 
                        help='Path to data cache directory')
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)

    needs_session_data = bool(args.session or args.all_sessions)
    handler = VBNDataHandler(cache_dir, metadata_only=not needs_session_data)

    if args.overview:
        handler.print_dataset_overview()
    elif args.session:
        print_full_summary(handler, [args.session])
    elif args.all_sessions:
        all_sessions = handler.sessions_table.index.tolist()
        print_full_summary(handler, all_sessions)
    elif args.filter:
        filtered = filter_sessions_metadata(
            handler,
            animals=args.animals,
            regions=args.regions,
            units_required=args.units_required,
            phase=args.phase,
        )
        print("\nFILTERED SESSIONS")
        print("=" * 80)
        print(f"Total matching sessions: {len(filtered)}")
        if len(filtered) > 0:
            cols = ['mouse_id', 'session_number', 'image_set', 'experience_level', 'session_type']
            available_cols = [c for c in cols if c in filtered.columns]
            print(filtered[available_cols].sort_values(by=available_cols).to_string())
    else:
        # Default: print overview
        handler.print_dataset_overview()
