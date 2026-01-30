"""
Download Visual Behavior Neuropixels dataset from the Allen Institute.

"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)
from tqdm import tqdm


def get_cache(output_dir: str) -> VisualBehaviorNeuropixelsProjectCache:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_path)
    return cache


def filter_sessions(
    sessions_table,
    genotype: Optional[str] = None,
    experience_level: Optional[str] = None,
    session_type: Optional[str] = None,
    brain_areas: Optional[List[str]] = None,
    mouse_id: Optional[int] = None,
) -> List[int]:
    """Filter sessions table based on provided criteria.
    
    Args:
        sessions_table: DataFrame with session metadata
        genotype: Filter by genotype
        experience_level: Filter by experience level ('Familiar' or 'Novel')
        session_type: Filter by exact session type
        brain_areas: Filter sessions that recorded from these brain areas
        mouse_id: Filter by specific mouse ID
    
    Returns:
        List of session IDs matching all criteria
    """
    filtered = sessions_table.copy()
    
    if genotype:
        filtered = filtered[filtered['full_genotype'].str.contains(genotype, case=False, na=False)]
    
    if experience_level:
        filtered = filtered[filtered['experience_level'] == experience_level]
    
    if session_type:
        filtered = filtered[filtered['session_type'] == session_type]
    
    if brain_areas:
        # Filter sessions that have any of the specified brain areas
        mask = filtered['structure_acronyms'].apply(
            lambda areas: any(area in areas for area in brain_areas) if isinstance(areas, list) else False
        )
        filtered = filtered[mask]
    
    if mouse_id is not None:
        filtered = filtered[filtered['mouse_id'] == str(mouse_id)]
    
    return list(filtered.index)


def show_session_info(cache: VisualBehaviorNeuropixelsProjectCache, session_ids: List[int]) -> None:
    """Display detailed information about specified sessions.
    
    Args:
        cache: The project cache
        session_ids: List of session IDs to display info for
    """
    sessions_table = cache.get_ecephys_session_table()
    
    for session_id in session_ids:
        if session_id not in sessions_table.index:
            print(f"\nSession {session_id}: NOT FOUND")
            continue
        
        row = sessions_table.loc[session_id]
        
        print(f"\n{'='*80}")
        print(f"Session ID: {session_id}")
        print(f"{'='*80}")
        print(f"Mouse ID: {row.get('mouse_id', 'N/A')}")
        print(f"Session Type: {row.get('session_type', 'N/A')}")
        print(f"Genotype: {row.get('full_genotype', 'N/A')}")
        print(f"Experience Level: {row.get('experience_level', 'N/A')}")
        print(f"Sex: {row.get('sex', 'N/A')}")
        print(f"Image Set: {row.get('image_set', 'N/A')}")
        
        # Brain areas
        areas = row.get('structure_acronyms', [])
        if isinstance(areas, list) and areas:
            print(f"Brain Areas Recorded: {', '.join(areas)}")
        else:
            print(f"Brain Areas Recorded: N/A")
        
        # Try to load session to get unit counts
        try:
            print("\nLoading session data...")
            session = cache.get_ecephys_session(ecephys_session_id=session_id)
            units = session.units
            probes = session.probes
            
            print(f"Total Units: {len(units)}")
            print(f"Total Probes: {len(probes)}")
            
            # Units per brain area
            if hasattr(units, 'ecephys_structure_acronym'):
                area_counts = units['ecephys_structure_acronym'].value_counts()
                print("\nUnits per brain area:")
                for area, count in area_counts.items():
                    print(f"  {area}: {count}")
        except Exception as e:
            print(f"Could not load session data: {e}")


def load_download_tracking(cache_dir: Path) -> dict:
    """Load the download tracking file.
    
    Args:
        cache_dir: Cache directory path
    
    Returns:
        Dictionary mapping session_id -> download timestamp
    """
    tracking_file = cache_dir / '.downloaded_sessions.json'
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            return json.load(f)
    return {}


def save_download_tracking(cache_dir: Path, tracking: dict) -> None:
    """Save the download tracking file.
    
    Args:
        cache_dir: Cache directory path
        tracking: Dictionary mapping session_id -> download timestamp
    """
    tracking_file = cache_dir / '.downloaded_sessions.json'
    with open(tracking_file, 'w') as f:
        json.dump(tracking, f, indent=2)


def list_available_sessions(cache: VisualBehaviorNeuropixelsProjectCache) -> None:

    sessions_table = cache.get_ecephys_session_table()
    print(f"\nAvailable sessions ({len(sessions_table)} total):")
    
    for session_id in sessions_table.index:
        row = sessions_table.loc[session_id]
        mouse_id = row.get('mouse_id', 'N/A')
        session_type = row.get('session_type', 'N/A')
        print(f"  {session_id} | Mouse: {mouse_id} | Type: {session_type}")
    


def download_sessions(
    cache: VisualBehaviorNeuropixelsProjectCache,
    session_ids: list,
    download_all: bool = False,
    force_redownload: bool = False,
    dry_run: bool = False,
) -> None:

    sessions_table = cache.get_ecephys_session_table()
    available_ids = set(sessions_table.index)
    
    if download_all:
        session_ids = list(available_ids)
    else:
        # Validate provided session IDs
        invalid_ids = set(session_ids) - available_ids
        if invalid_ids:
            print(f"Warning: The following session IDs are not available: {invalid_ids}")
            session_ids = [sid for sid in session_ids if sid in available_ids]
        
        if not session_ids:
            print("Error: No valid session IDs to download.")
            return
    
    # Load download tracking
    cache_dir = Path(cache.fetch_api.cache._manifest._cache_dir)
    tracking = load_download_tracking(cache_dir)
    
    # Filter out already downloaded sessions unless force_redownload is True
    if not force_redownload:
        already_downloaded = [sid for sid in session_ids if str(sid) in tracking]
        if already_downloaded:
            print(f"\nSkipping {len(already_downloaded)} already downloaded sessions.")
            print("Use --force-redownload to re-download them.")
            session_ids = [sid for sid in session_ids if str(sid) not in tracking]
    
    if not session_ids:
        print("\nNo new sessions to download.")
        return
    
    print(f"\n{'='*80}")
    print(f"Sessions to download: {len(session_ids)}")
    print(f"{'='*80}")
    
    if dry_run:
        print("\nDRY RUN - Would download the following sessions:")
        for session_id in session_ids:
            row = sessions_table.loc[session_id]
            print(f"  {session_id} | Mouse: {row.get('mouse_id', 'N/A')} | Type: {row.get('session_type', 'N/A')}")
        return
    
    successful = 0
    failed = 0
    failed_ids = []
    
    pbar = tqdm(session_ids, desc="Downloading sessions", unit="session")
    for session_id in pbar:
        pbar.set_postfix(session=session_id, status="downloading...")
        try:
            session = cache.get_ecephys_session(ecephys_session_id=session_id)
            # Try to get unit/probe counts if available
            try:
                units = session.units
                probes = session.probes
                pbar.set_postfix(session=session_id, units=len(units), probes=len(probes))
            except AttributeError:
                pbar.set_postfix(session=session_id, status="done")
            
            # Mark as successfully downloaded
            tracking[str(session_id)] = datetime.now().isoformat()
            save_download_tracking(cache_dir, tracking)
            successful += 1
            
        except Exception as e:
            pbar.set_postfix(session=session_id, status=f"FAILED: {e}")
            failed += 1
            failed_ids.append(session_id)
            
            # Remove from tracking if it was there (failed download should be retryable)
            if str(session_id) in tracking:
                del tracking[str(session_id)]
                save_download_tracking(cache_dir, tracking)
    
    print(f"\n{'='*80}")
    print(f"Download complete! Success: {successful}, Failed: {failed}")
    if failed_ids:
        print(f"Failed session IDs: {failed_ids}")
    print(f"Data saved to: {cache_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Visual Behavior Neuropixels dataset from the Allen Institute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help=(
            "Session IDs to download. Use 'all' to download all sessions, "
            "or provide a space-separated list of session IDs."
        ),
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./visual_behavior_neuropixels_data",
        help="Output directory for downloaded data (default: ./visual_behavior_neuropixels_data)",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sessions",
        help="List all available session IDs without downloading.",
    )
    
    # Filtering options
    parser.add_argument(
        "--genotype",
        type=str,
        help="Filter sessions by genotype (partial match, e.g., 'Sst', 'Vip', 'C57BL6J')",
    )
    
    parser.add_argument(
        "--experience-level",
        choices=["Familiar", "Novel"],
        help="Filter sessions by experience level",
    )
    
    parser.add_argument(
        "--session-type",
        type=str,
        help="Filter sessions by specific session type",
    )
    
    parser.add_argument(
        "--brain-areas",
        nargs="+",
        help="Filter sessions that recorded from these brain areas (e.g., VISp VISl)",
    )
    
    parser.add_argument(
        "--mouse-id",
        type=int,
        help="Download all sessions for a specific mouse ID",
    )
    
    # Info and utility options
    parser.add_argument(
        "--info",
        nargs="+",
        type=int,
        help="Show detailed information about specific session IDs",
    )
    
    parser.add_argument(
        "--export-metadata",
        type=str,
        help="Export filtered session metadata to CSV file",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download of sessions even if already downloaded",
    )
    
    args = parser.parse_args()
    
    # Initialize cache
    print(f"Initializing cache at: {args.output}")
    cache = get_cache(args.output)
    
    # Handle --info flag
    if args.info:
        show_session_info(cache, args.info)
        return
    
    # Handle --list flag
    if args.list_sessions:
        list_available_sessions(cache)
        return
    
    # Handle --sessions argument with filtering
    if args.sessions is None and args.mouse_id is None:
        # Check if any filters are specified
        has_filters = any([
            args.genotype,
            args.experience_level,
            args.session_type,
            args.brain_areas,
        ])
        
        if not has_filters:
            parser.print_help()
            print("\nError: Please specify --sessions, --mouse-id, or use filters with --sessions all")
            return
    
    # Get sessions table for filtering
    sessions_table = cache.get_ecephys_session_table()
    
    # Apply filters if any are specified
    filter_applied = any([
        args.genotype,
        args.experience_level,
        args.session_type,
        args.brain_areas,
        args.mouse_id,
    ])
    
    if filter_applied:
        print("\nApplying filters...")
        filtered_ids = filter_sessions(
            sessions_table,
            genotype=args.genotype,
            experience_level=args.experience_level,
            session_type=args.session_type,
            brain_areas=args.brain_areas,
            mouse_id=args.mouse_id,
        )
        print(f"Filtered sessions: {len(filtered_ids)} matching criteria")
        
        # Export metadata if requested
        if args.export_metadata:
            filtered_table = sessions_table.loc[filtered_ids]
            filtered_table.to_csv(args.export_metadata)
            print(f"Metadata exported to: {args.export_metadata}")
        
        # If --sessions is specified, use the intersection of filtered and specified
        if args.sessions and args.sessions[0].lower() != "all":
            try:
                specified_ids = [int(sid) for sid in args.sessions]
                session_ids = [sid for sid in specified_ids if sid in filtered_ids]
                print(f"Intersection with specified sessions: {len(session_ids)} sessions")
            except ValueError as e:
                print(f"Error: Invalid session ID format. Session IDs must be integers. {e}")
                return
        else:
            session_ids = filtered_ids
        
        download_sessions(
            cache,
            session_ids,
            download_all=False,
            force_redownload=args.force_redownload,
            dry_run=args.dry_run,
        )
    else:
        # No filters, handle --sessions normally
        if len(args.sessions) == 1 and args.sessions[0].lower() == "all":
            download_sessions(
                cache,
                [],
                download_all=True,
                force_redownload=args.force_redownload,
                dry_run=args.dry_run,
            )
        else:
            # Convert session IDs to integers
            try:
                session_ids = [int(sid) for sid in args.sessions]
            except ValueError as e:
                print(f"Error: Invalid session ID format. Session IDs must be integers. {e}")
                return
            
            download_sessions(
                cache,
                session_ids,
                download_all=False,
                force_redownload=args.force_redownload,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()

