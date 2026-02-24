"""
Script to create a preprocessed dataset given a configuration
"""

# Imports
import argparse
from pathlib import Path
import yaml 

from jepsyn.utils import verify_config

def main(config_path : Path):

    print("Verifying config...")
    config = verify_config(config_path=config_path)
    print("Verified. Preprocessing")

    # TODO: Implement the preprocessing pipeline
    # refer to multi_session.py load_and_prepare_data comment
    # for details on expected structure
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a preprocessed dataset for experiments"
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to configuration YAML file for experiment settings"
    )
    args = parser.parse_args()
    
    print("Creating Dataset")
    print("=" * 60)
    main(config_path=args.config_path)