# Imports
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml 


# Validate Configuration
def verify_config(config_path: Path) -> Dict[str, Any]:
    """
    Validate configuration YAML file for inconsistencies.
    Verifies it contains the necessary information for the experiment.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary containing validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or missing required fields
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error opening configuration file: {e}")
    
    # Validate required fields
    required_fields = ["data_path", "model_config", "training_config"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(f"Missing required fields in configuration: {', '.join(missing_fields)}")

    # Resolve relative paths relative to the config file's directory so that
    # the config works correctly regardless of the working directory.
    config_dir = config_path.resolve().parent
    for key in ("data_path", "results_out_path", "plots_out_path"):
        if config.get(key):
            config[key] = str((config_dir / config[key]).resolve())

    return config