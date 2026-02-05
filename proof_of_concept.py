"""

For this proof of concept, we're aiming to test on a singular session,
if prediction loss & SIGReg loss decreases


"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Imports
from src.data.data_handler import VBNDataHandler
from src.data.preprocess import NeuropixelsPreprocessor, slice_trial_windows
from src.models.encoder import NeuralEncoder
from src.models.predictor import NeuralPredictor
from src.losses.lejepa import lejepa_loss
from src.plots.training import plot_loss_curves
from src.plots.latent_space import plot_prediction_accuracy
from src.plots.neural_activity import plot_spike_count_distribution
from src.plots.model_performance import plot_prediction_vs_actual
from src.utils.binning import bin_trial_aligned


def get_next_run_number(base_dir: Path) -> int:
    """Get the next run number in the runs directory."""
    runs = [d for d in base_dir.glob('run_*') if d.is_dir()]
    if not runs:
        return 1
    max_run = max([int(d.name.split('_')[1]) for d in runs])
    return max_run + 1


def get_or_create_dataset(
    dataset_path: Optional[Path],
    session_id: int,
    preprocessed_dir: Path,
) -> dict:
    """
    Get existing dataset or create a new one.
    
    Args:
        dataset_path: Optional path to existing preprocessed dataset
        session_id: Session ID to process
        preprocessed_dir: Directory where preprocessed datasets are stored
        
    Returns:
        Dictionary containing processed data
    """
    # If dataset path provided, load it
    if dataset_path is not None:
        if dataset_path.exists():
            print(f"Loading dataset from {dataset_path}...")
            with open(dataset_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset_file = preprocessed_dir / f"session_{session_id}.pkl"
    
    # Create new dataset
    print(f"Creating new dataset for session {session_id}...")
    data_dir = Path(__file__).parent / "visual_behavior_neuropixels_data"
    handler = VBNDataHandler(str(data_dir))
    session_data = handler.load_session(session_id)
    
    # Get filtered units
    preprocessor = (NeuropixelsPreprocessor(session_data)
        .validate_integrity()
        .clean()
        .filter_units(min_snr=1.0, max_isi_violations=1.0, min_firing_rate=0.1))
    
    processed = preprocessor.get()
    
    # Get stimulus change times
    trials = session_data.trials
    change_times = trials['change_time_no_display_delay'].dropna().values
    
    # Create trial-aligned binned data
    good_units = processed['units'].index.tolist()
    trial_aligned = bin_trial_aligned(
        spike_times_dict=session_data.spike_times,
        event_times=change_times,
        unit_ids=good_units,
        bin_size_ms=10,  # 10ms bins for fine temporal resolution
        pre_time_ms=0,    # Start at stimulus change
        post_time_ms=400,  # Up to end of decision/motor prep phase
    )
    
    # Store trial-aligned data in processed dict
    processed['trial_aligned'] = trial_aligned
    
    # Save dataset to preprocessed directory
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {dataset_file}...")
    with open(dataset_file, 'wb') as f:
        pickle.dump(processed, f)
    
    return processed


def main(
    session_id: int,
    dataset_dir: Optional[str] = None,
):
    
    # Setup runs directory
    runs_dir = Path(__file__).parent / "runs" / "proof_of_concept"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_num = get_next_run_number(runs_dir)
    run_dir = runs_dir / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup preprocessed data directory
    preprocessed_dir = Path(__file__).parent / "preprocessed"
    
    # Get or create dataset
    dataset_path = Path(dataset_dir) if dataset_dir else None
    processed = get_or_create_dataset(
        dataset_path,
        session_id,
        preprocessed_dir,
    )
    
    print("Dataset ready")

    print("\n========== Dataset Debug Info ==========")
    trial_aligned = processed['trial_aligned']
    units = processed['units']
    num_units = len(units)
    trial_data = trial_aligned['binned']  # (n_trials, n_units, n_time_bins)
    n_trials, _, n_time_bins = trial_data.shape
    
    print(f"Trial-aligned data shape: {trial_data.shape}")
    print(f"  Trials: {n_trials} | Units: {num_units} | Time bins: {n_time_bins}")
    print(f"  Bin size: {trial_aligned['bin_size_ms']:.1f}ms")
    print(f"  Min: {trial_data.min():.6f} | Max: {trial_data.max():.6f}")
    print(f"  Mean: {trial_data.mean():.6f} | Std: {trial_data.std():.6f}")
    print(f"  % zeros: {(trial_data == 0).sum() / trial_data.size * 100:.1f}%")
    
    print("\nSplitting into train/test sets...")
    n_train = int(0.8 * n_trials)
    train_trials = trial_data[:n_train]
    test_trials = trial_data[n_train:]
    print(f"Train trials: {n_train} | Test trials: {n_trials - n_train}")
    
    print("\nCreating context-target windows...")
    
    # With pre_time=0ms, post_time=400ms, bin_size=10ms:
    # - Context bins: 0-19 (0-200ms)
    # - Target bins: 20-39 (200-400ms)
    context_start, context_end = 0, 20
    target_start, target_end = 20, 40
    context_size = context_end - context_start
    target_size = target_end - target_start
    
    train_context, train_target = slice_trial_windows(
        trial_data=train_trials,
        context_start=context_start,
        context_end=context_end,
        target_start=target_start,
        target_end=target_end,
    )
    test_context, test_target = slice_trial_windows(
        trial_data=test_trials,
        context_start=context_start,
        context_end=context_end,
        target_start=target_start,
        target_end=target_end,
    )
    
    print(f"Train context shape: {train_context.shape}")
    print(f"Train target shape: {train_target.shape}")
    print(f"Test context shape: {test_context.shape}")
    print(f"Test target shape: {test_target.shape}")
    
    # Move to device and convert to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_context = torch.FloatTensor(train_context).to(device)
    train_target = torch.FloatTensor(train_target).to(device)
    test_context = torch.FloatTensor(test_context).to(device)
    test_target = torch.FloatTensor(test_target).to(device)
    
    # Flatten to (n_windows * time_bins, n_units) for normalization
    train_context_flat = train_context.permute(0, 2, 1).reshape(-1, num_units)
    train_target_flat = train_target.permute(0, 2, 1).reshape(-1, num_units)
    train_all_flat = torch.cat([train_context_flat, train_target_flat], dim=0)
    
    spike_mean = train_all_flat.mean(dim=0, keepdim=True)
    spike_std = train_all_flat.std(dim=0, keepdim=True)
    
    # Apply same normalization to both train and test
    train_context = (train_context - spike_mean.unsqueeze(0)) / (spike_std.unsqueeze(0) + 1e-8)
    train_target = (train_target - spike_mean.unsqueeze(0)) / (spike_std.unsqueeze(0) + 1e-8)
    test_context = (test_context - spike_mean.unsqueeze(0)) / (spike_std.unsqueeze(0) + 1e-8)
    test_target = (test_target - spike_mean.unsqueeze(0)) / (spike_std.unsqueeze(0) + 1e-8)
    
    print(f"\nNormalized train context shape: {train_context.shape}")
    print(f"  Min: {train_context.min().item():.6f} | Max: {train_context.max().item():.6f}")
    print(f"  Mean: {train_context.mean().item():.6f} | Std: {train_context.std().item():.6f}")
    print(f"Normalized test context shape: {test_context.shape}")
    print(f"  Min: {test_context.min().item():.6f} | Max: {test_context.max().item():.6f}")
    print(f"  Mean: {test_context.mean().item():.6f} | Std: {test_context.std().item():.6f}")
    print("="*40)
    
    # Model training pipeline
    print("\nTraining model...")
    
    latent_dim = 128
    context_len = context_size  # 15 time bins
    horizon = target_size  # 20 time bins to predict
    
    encoder = NeuralEncoder(num_units, latent_dim, encoder_type="transformer").to(device)
    predictor = NeuralPredictor(latent_dim, horizon, context_len, predictor_type="rnn", hidden_dim=256, num_layers=2).to(device)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3
    )
    
    total_losses = []
    pred_losses = []
    sigreg_losses = []
    
    num_epochs = 100
    batch_size = 32
    lambd = 0.2
    num_slices = 1024
    
    n_train_windows = train_context.shape[0]
    n_test_windows = test_context.shape[0]
    
    for epoch in range(num_epochs):
        encoder.train()
        predictor.train()
        
        total_loss_epoch = 0
        pred_loss_epoch = 0
        sigreg_loss_epoch = 0
        num_batches = 0
        
        # Shuffle indices for each epoch
        indices = torch.randperm(n_train_windows)
        
        for i in range(0, n_train_windows, batch_size):
            batch_indices = indices[i:min(i + batch_size, n_train_windows)]
            
            context_batch = train_context[batch_indices].permute(0, 2, 1)  # (batch, context_len, n_units)
            target_batch = train_target[batch_indices].permute(0, 2, 1)    # (batch, horizon, n_units)
            
            # Encode full sequences
            z_context = encoder(context_batch)  # (batch, context_len, latent_dim)
            z_target = encoder(target_batch)    # (batch, horizon, latent_dim)
            
            # Predict full target sequence
            z_predicted = predictor(z_context)  # (batch, horizon, latent_dim)
            
            # Compare FULL sequences by flattening temporal dimension
            z_predicted_flat = z_predicted.reshape(z_predicted.shape[0], -1)
            z_target_flat = z_target.reshape(z_target.shape[0], -1)
            z_context_flat = z_context[:, -1, :]  # Last context timestep for SIGReg
            
            loss, pred_loss, sigreg_loss = lejepa_loss(
                z_context_flat, z_target_flat, z_predicted_flat,
                epoch, lambd, num_slices
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            pred_loss_epoch += pred_loss.item()
            sigreg_loss_epoch += sigreg_loss.item()
            num_batches += 1
        
        total_losses.append(total_loss_epoch / num_batches)
        pred_losses.append(pred_loss_epoch / num_batches)
        sigreg_losses.append(sigreg_loss_epoch / num_batches)
        
        # Evaluate on test set
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            test_context_enc = encoder(test_context.permute(0, 2, 1))
            test_target_enc = encoder(test_target.permute(0, 2, 1))
            test_pred = predictor(test_context_enc)
            
            test_pred_flat = test_pred.reshape(test_pred.shape[0], -1)
            test_target_flat = test_target_enc.reshape(test_target_enc.shape[0], -1)
            test_context_flat = test_context_enc[:, -1, :]
            
            test_loss, test_pred_loss, test_sigreg = lejepa_loss(
                test_context_flat, test_target_flat, test_pred_flat,
                epoch, lambd, num_slices
            )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train - Loss={total_loss_epoch/num_batches:.4f}, "
                  f"Pred={pred_loss_epoch/num_batches:.4f}, "
                  f"SIGReg={sigreg_loss_epoch/num_batches:.4f}")
            print(f"  Test  - Loss={test_loss.item():.4f}, "
                  f"Pred={test_pred_loss.item():.4f}, "
                  f"SIGReg={test_sigreg.item():.4f}")
    
    print("training complete")

    # Plots & Summary
    print("generating plots...")
    
    encoder.eval()
    predictor.eval()
    
    with torch.no_grad():
        # Evaluate on full test set
        z_context_test = encoder(test_context.permute(0, 2, 1))
        z_target_test = encoder(test_target.permute(0, 2, 1))
        z_pred_test = predictor(z_context_test)
        
        # Flatten for comparison
        z_target_flat = z_target_test.reshape(z_target_test.shape[0], -1)
        z_pred_flat = z_pred_test.reshape(z_pred_test.shape[0], -1)
        
        # Get all embeddings for visualization (from training data)
        batch_contexts = train_context[::10].permute(0, 2, 1)  # Subsample for memory
        all_embeddings = encoder(batch_contexts)[:, -1, :]  # Use last timestep
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    plot_loss_curves(total_losses, pred_losses, sigreg_losses, ax=axes[0, 0])
    plot_prediction_accuracy(z_pred_flat, z_target_flat, ax=axes[0, 1])
    plot_spike_count_distribution(test_context.cpu().numpy().reshape(-1, num_units), ax=axes[1, 0])
    plot_prediction_vs_actual(z_pred_flat, z_target_flat, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('poc_results.png', dpi=150, bbox_inches='tight')
    
    # Compute final test performance
    with torch.no_grad():
        final_test_loss, final_test_pred, final_test_sigreg = lejepa_loss(
            z_context_test[:, -1, :], z_target_flat, z_pred_flat,
            num_epochs, lambd, num_slices
        )
    
    print("\n=========== summary: ==========")
    print(f"Session ID: {session_id}")
    print(f"Units: {num_units} | Trials: {n_trials} (train: {n_train}, test: {n_trials-n_train})")
    print(f"Windows: {n_train_windows} train, {n_test_windows} test")
    print(f"Architecture: {num_units}→{latent_dim} latent (RNN predictor)")
    print(f"  Context: {context_len} bins (0-200ms, full early sensory+integration)")
    print(f"  Target: {horizon} bins (200-400ms, decision/motor prep)")
    print(f"  Bin size: {trial_aligned['bin_size_ms']:.1f}ms")
    print(f"\nTraining:")
    print(f"  Epochs: {num_epochs} | Batch size: {batch_size} | λ: {lambd}")
    print(f"  Train Final Loss: {total_losses[-1]:.4f}")
    print(f"    Prediction: {pred_losses[-1]:.4f} | SIGReg: {sigreg_losses[-1]:.4f}")
    print(f"  Train Improvement: {(total_losses[0] - total_losses[-1]) / total_losses[0] * 100:.1f}%")
    print(f"\nTest Performance:")
    print(f"  Test Loss: {final_test_loss.item():.4f}")
    print(f"    Prediction: {final_test_pred.item():.4f} | SIGReg: {final_test_sigreg.item():.4f}")
    print(f"\nResults: PoC {'successful' if total_losses[-1] < total_losses[0] else 'inconclusive'}")
    print("="*60)    

    # Plots & Summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LE-JEPA model on single Visual Behavior Neuropixels session")
    parser.add_argument("session_id", type=int, help="Ecephys session ID to train on")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to existing preprocessed dataset (if omitted, a new one is created in preprocessed/)"
    )
    args = parser.parse_args()
    
    main(args.session_id, args.dataset_dir)
    

