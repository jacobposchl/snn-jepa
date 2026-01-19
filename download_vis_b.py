"""
Download Visual Behavior Neuropixels dataset from the Allen Institute.

Usage:
    python download_vis_b.py --sessions all                    # Download all sessions
    python download_vis_b.py --sessions 1064644573 1065437523  # Download specific sessions
    python download_vis_b.py --list                            # List available sessions
"""

import argparse
from pathlib import Path

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)
from tqdm import tqdm


def get_cache(output_dir: str) -> VisualBehaviorNeuropixelsProjectCache:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_path)
    return cache


def list_available_sessions(cache: VisualBehaviorNeuropixelsProjectCache) -> None:

    sessions_table = cache.get_ecephys_session_table()
    print(f"\nAvailable sessions ({len(sessions_table)} total):")
    
    for session_id in sessions_table.index:
        row = sessions_table.loc[session_id]
        mouse_id = row.get('mouse_id', 'N/A')
        session_type = row.get('session_type', 'N/A')
        print(f"  {session_id} | Mouse: {mouse_id} | Type: {session_type}")
    


def download_sessions(cache: VisualBehaviorNeuropixelsProjectCache, session_ids: list, download_all: bool = False, ) -> None:

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
    
    successful = 0
    failed = 0
    
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
            successful += 1
        except Exception as e:
            pbar.set_postfix(session=session_id, status=f"FAILED: {e}")
            failed += 1
    
    print(f"\nDownload complete! Success: {successful}, Failed: {failed}")
    print(f"Data saved to: {cache.fetch_api.cache._manifest._cache_dir}")


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
    
    args = parser.parse_args()
    
    # Initialize cache
    print(f"Initializing cache at: {args.output}")
    cache = get_cache(args.output)
    
    # Handle --list flag
    if args.list_sessions:
        list_available_sessions(cache)
        return
    
    # Handle --sessions argument
    if args.sessions is None:
        parser.print_help()
        print("\nError: Please specify --sessions or --list")
        return
    
    if len(args.sessions) == 1 and args.sessions[0].lower() == "all":
        download_sessions(cache, [], download_all=True)
    else:
        # Convert session IDs to integers
        try:
            session_ids = [int(sid) for sid in args.sessions]
        except ValueError as e:
            print(f"Error: Invalid session ID format. Session IDs must be integers. {e}")
            return
        
        download_sessions(cache, session_ids, download_all=False)


if __name__ == "__main__":
    main()

