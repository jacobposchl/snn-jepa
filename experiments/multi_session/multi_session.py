"""
Full Runner Script for a Multi-Session Experiment
With Training of LeJEPA Model and Distillation into
a Spiking Neural Network Model
"""

# Imports
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from jepsyn.models import NeuralEncoder
from jepsyn.utils import verify_config


def load_and_prepare_data(config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """
    Load and prepare multi-session neural data for training and testing.

    Args:
        config: Configuration dictionary containing data_path and split parameters

    Returns:
        Tuple of (train_data, val_data, test_data)

    Expected Dataset Format:
        Columns:
            - session_id (int): Unique identifier for each recording session
            - window_id (int): Unique identifier for each temporal window
            - window_start_ms (int): Window start time in milliseconds
            - window_end_ms (int): Window end time in milliseconds
            - events_units (np.array): Array of unit indices for each spike event
            - events_times_ms (np.array): Spike times relative to window start (ms)
            - stimulus: Lists of dictionaries (stimulus events with timestamps relative to window start)
            - behavior: Lists of dictionaries (behavioral events with timestamps relative to window start)

        Each row corresponds to a single temporal window of neural activity.

    Validation Checks:
        - No duplicate window_ids with conflicting window times
        - Equal length arrays for events_units and events_times_ms
    """
    # Load dataset from Parquet instead of CSV for better performance
    data_path = config.get("data_path")
    if not data_path:
        raise ValueError("data_path not found in configuration")

    dataset = pd.read_parquet(data_path, engine="pyarrow")

    # Validate data integrity
    print("Validating dataset integrity...")

    # Check for duplicate window_ids with conflicting timestamps
    duplicate_check = dataset.groupby("window_id").agg(
        {"window_start_ms": "nunique", "window_end_ms": "nunique"}
    )
    conflicts = duplicate_check[
        (duplicate_check["window_start_ms"] > 1)
        | (duplicate_check["window_end_ms"] > 1)
    ]
    if not conflicts.empty:
        raise ValueError(
            f"Found {len(conflicts)} window_ids with conflicting timestamps: "
            f"{conflicts.index.tolist()[:5]}..."
        )

    # Verify events_units and events_times_ms have matching lengths
    length_mismatches = dataset[
        dataset["events_units"].apply(len) != dataset["events_times_ms"].apply(len)
    ]
    if not length_mismatches.empty:
        raise ValueError(
            f"Found {len(length_mismatches)} rows where events_units and events_times_ms "
            f"have different lengths (window_ids: {length_mismatches['window_id'].tolist()[:5]})"
        )

    print("Passed Basic Validation Checks, no duplicates or length mismatches found.")

    # Extract split configuration
    train_size = config.get("data", {}).get("train_split")
    val_size = config.get("data", {}).get("val_split")
    test_size = config.get("data", {}).get("test_split")
    random_state = config.get("data", {}).get("random_state")

    # Split data by session_id to prevent data leakage across sessions
    unique_sessions = dataset["session_id"].unique()

    # First split: separate test set
    train_val_sessions, test_sessions = train_test_split(
        unique_sessions, test_size=test_size, random_state=random_state
    )

    # Second split: separate train and validation sets
    train_sessions, val_sessions = train_test_split(
        train_val_sessions,
        test_size=val_size / (train_size + val_size),
        random_state=random_state,
    )

    # Create dataset splits
    train_data = dataset[dataset["session_id"].isin(train_sessions)]
    val_data = dataset[dataset["session_id"].isin(val_sessions)]
    test_data = dataset[dataset["session_id"].isin(test_sessions)]

    print(f"Train: {len(train_data)} windows ({len(train_sessions)} sessions)")
    print(f"Val:   {len(val_data)} windows ({len(val_sessions)} sessions)")
    print(f"Test:  {len(test_data)} windows ({len(test_sessions)} sessions)")

    return train_data, val_data, test_data


def train_lejepa(
    config: Dict[str, Any], train_data: Any, val_data: Any
) -> Tuple[Any, pd.DataFrame]:
    """
    Train LeJEPA teacher model on multi-session neural data.
    Includes validation during training.

    Args:
        config: Configuration dictionary
        train_data: Training dataset
        val_data: Validation dataset

    Returns:
        Tuple of (trained_model, training_metrics_df)
    """
    # TODO: Implement LeJEPA training

    # - Initialize model from config (support nested model_config or flat config)
    model_cfg = config.get("model_config") or config
    enc_model = model_cfg.get("encoder_type")
    n_units = model_cfg.get("n_units")
    latent_dim = model_cfg.get("latent_dim")

    # Inputs probably need fixing
    encoder = NeuralEncoder(
        n_units=n_units, latent_dim=latent_dim, encoder_type=enc_model
    )

    # - Training loop with validation

    # Might need adjusting
    # 1. Tokenize data (unit_id, time)
    # 2. Encode tokenized data
    # 3. Evaluate model
    # val_results = evaluate_model(model = , test_data = val_data, stage = "LeJEPA")

    # - Save checkpoints
    # - Return trained model and metrics
    pass


def distill_snn(
    config: Dict[str, Any], teacher_model: Any, train_data: Any, val_data: Any
) -> Tuple[Any, pd.DataFrame]:
    """
    Distill LeJEPA teacher into spiking neural network student.
    Includes validation during distillation.

    Args:
        config: Configuration dictionary
        teacher_model: Trained LeJEPA model
        train_data: Training dataset
        val_data: Validation dataset

    Returns:
        Tuple of (trained_snn, distillation_metrics_df)
    """
    # TODO: Implement SNN distillation
    # - Initialize SNN from config
    # - Distillation training loop with validation
    # - Save checkpoints
    # - Return trained SNN and metrics
    pass


def evaluate_model(model: Any, test_data: Any, stage: str) -> pd.DataFrame:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model (LeJEPA or SNN)
        test_data: Test dataset
        stage: Model stage name ("LeJEPA" or "SNN")

    Returns:
        DataFrame containing evaluation metrics
    """
    # TODO: Implement evaluation
    # - Run inference on test data
    # - Compute metrics (reconstruction, prediction, etc.)
    # - Return results dataframe
    pass


def save_results(
    stage: str, phase: str, metrics: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """
    Save results, metrics, and generate plots.

    Args:
        stage: Experiment stage ("LeJEPA" or "SNN")
        phase: Training phase ("training", "validation", "test")
        metrics: DataFrame containing metrics
        config: Configuration dictionary
    """
    # TODO: Implement results output
    # - Save metrics to CSV
    # - Generate plots (training curves, latent space, etc.)
    # - Save figures to output directory
    pass


# Run Experiment
def main(config_path: Path) -> None:
    """
    Main experiment runner that orchestrates the full pipeline.

    Args:
        config_path: Path to the configuration YAML file
    """
    # Verify configuration
    print("=" * 60)
    print("Verifying Configuration")
    config = verify_config(config_path)
    print(f"Configuration Verified. Using: {config_path}")

    # Load and prepare data
    print("\n" + "=" * 60)
    print("Loading and Preparing Data")
    train_data, val_data, test_data = load_and_prepare_data(config)
    print("Data loaded successfully")

    # Train LeJEPA teacher model
    print("\n" + "=" * 60)
    print("Training LeJEPA Teacher Model")
    jepa_model, jepa_train_metrics = train_lejepa(config, train_data, val_data)
    save_results(
        stage="LeJEPA", phase="training", metrics=jepa_train_metrics, config=config
    )
    print("LeJEPA training complete")

    # Evaluate LeJEPA on test set
    print("\n" + "=" * 60)
    print("Evaluating LeJEPA on Test Set")
    jepa_test_metrics = evaluate_model(jepa_model, test_data, stage="LeJEPA")
    save_results(stage="LeJEPA", phase="test", metrics=jepa_test_metrics, config=config)
    print("LeJEPA evaluation complete")

    # Distill into SNN student model
    print("\n" + "=" * 60)
    print("Distilling into Spiking Neural Network")
    snn_model, snn_train_metrics = distill_snn(config, jepa_model, train_data, val_data)
    save_results(
        stage="SNN", phase="distillation", metrics=snn_train_metrics, config=config
    )
    print("SNN distillation complete")

    # Evaluate SNN on test set
    print("\n" + "=" * 60)
    print("Evaluating Distilled SNN on Test Set")
    snn_test_metrics = evaluate_model(snn_model, test_data, stage="SNN")
    save_results(stage="SNN", phase="test", metrics=snn_test_metrics, config=config)
    print("SNN evaluation complete")

    print("\n" + "=" * 60)
    print("Multi-Session Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-session neural experiment with LeJEPA and SNN distillation."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to configuration YAML file for experiment settings",
    )
    args = parser.parse_args()

    print("Starting Multi-Session Experiment")
    print("=" * 60)
    main(config_path=args.config_path)
