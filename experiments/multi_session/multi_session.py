"""
Full Runner Script for a Multi-Session Experiment
With Training of LeJEPA Model and Distillation into
a Spiking Neural Network Model
"""

# Imports
import argparse
import pandas as pd
import yaml
from Lib import Path 


# Validate Configuration
def verify_config(config_path : Path) -> dict:
    """
    Should check configuration yaml file for inconsistincies
    Should verify it contains the necessary information
    """

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except:
        return ValueError, "Error Opening Configuration File, check path..."
    
    # Check configuration file data

    return config

# Import and Validate Dataset

# Train Teacher on Train Set
def train_lejepa(config: yaml):
    pass

# Test Teacher on Test Set
def test_lejepa(config: yaml):
    pass

# Train SNN on Train Set
def train_snn(config: yaml):
    pass

# Test SNN on Test Set
def test_snn(config: yaml):
    pass

# Output Results Dataset
def output_results(stage: str, train = bool, data = pd.DataFrame):
    """
    Should print results summary and save figures of data
    """
    pass


# Run Experiment
def main(config_path : Path):

    print("Verifying Configuration")
    config = verify_config(config_path)
    print(f"Configuration Verified. Using: {config_path}")


    print("Training LeJEPA Model")
    jepa_training_data = train_lejepa(config, )
    output_results(stage = "JEPA", train = True, data = jepa_training_data)

    print("Testing LeJEPA")
    jepa_testing_data = test_lejepa(config, )
    output_results(stage = "JEPA", train = False, data = jepa_testing_data)

    print("Distilling into SNN")
    snn_training_data = train_snn(config, )
    output_results(stage = "SNN", train = True, data = snn_training_data)

    print("Testing Distilled SNN")
    snn_testing_data = test_snn(config, )
    output_results(stage = "SNN", train = False, data = snn_testing_data)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Arguments for running a multi-session experiment.")
    parser.add_argument("config_path", help = "Input Configuration File Path for Experiment Settings")
    args = parser.parse_args()

    config_path = args.config_path
    
    print("Starting Multi Session Experiment")
    main(config_path=config_path)


