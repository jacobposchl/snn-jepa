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
from src.data.preprocess import NeuropixelsPreprocessor, slice_trial_windows, get_or_create_dataset
from src.models.encoder import NeuralEncoder
from src.models.predictor import NeuralPredictor
from src.losses.lejepa import lejepa_loss
from src.plots.training import plot_loss_curves
from src.plots.latent_space import plot_prediction_accuracy
from src.plots.neural_activity import plot_spike_count_distribution
from src.plots.model_performance import plot_prediction_vs_actual
from src.utils.binning import bin_trial_aligned


def main(
    session_id: int,
    dataset_dir: Optional[str] = None,
):
    
    # Setup runs directory
    runs_dir = Path(__file__).parent / "runs" / "proof_of_concept"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Get next run number
    existing_runs = [d for d in runs_dir.glob('run_*') if d.is_dir()]
    run_num = 1 if not existing_runs else max([int(d.name.split('_')[1]) for d in existing_runs]) + 1

    run_dir = runs_dir / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup preprocessed data directory
    preprocessed_dir = Path(__file__).parent / "preprocessed"
    
    # Get or create dataset
    dataset_path = Path(dataset_dir) if dataset_dir else None
    processed = get_or_create_dataset(
        session_id=session_id,
        preprocessed_dir=preprocessed_dir,
        dataset_path=dataset_path,
        brain_areas=['VISp', 'VISl'],
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
    
    spike_mean = train_all_flat.mean(dim=0, keepdim=True)  # (1, n_units)
    spike_std = train_all_flat.std(dim=0, keepdim=True)    # (1, n_units)
    
    # Reshape to (1, n_units, 1) for broadcasting with (batch, n_units, time)
    spike_mean = spike_mean.unsqueeze(2)
    spike_std = spike_std.unsqueeze(2)
    
    # Clip standard deviation to prevent division by near-zero (silent units)
    spike_std = torch.clamp(spike_std, min=0.1)
    
    # Apply same normalization to both train and test
    train_context = (train_context - spike_mean) / spike_std
    train_target = (train_target - spike_mean) / spike_std
    test_context = (test_context - spike_mean) / spike_std
    test_target = (test_target - spike_mean) / spike_std
    
    # Clip extreme outliers after normalization to prevent instability
    train_context = torch.clamp(train_context, -10, 10)
    train_target = torch.clamp(train_target, -10, 10)
    test_context = torch.clamp(test_context, -10, 10)
    test_target = torch.clamp(test_target, -10, 10)
    
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
            
            # Compare full sequences by flattening temporal dimension
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
        batch_contexts = train_context[::10].permute(0, 2, 1)
        all_embeddings = encoder(batch_contexts)[:, -1, :]  # Use last timestep
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    plot_loss_curves(total_losses, pred_losses, sigreg_losses, ax=axes[0, 0])
    plot_prediction_accuracy(z_pred_flat, z_target_flat, ax=axes[0, 1])
    plot_spike_count_distribution(test_context.cpu().numpy().reshape(-1, num_units), ax=axes[1, 0])
    plot_prediction_vs_actual(z_pred_flat, z_target_flat, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(run_dir / 'poc_results.png', dpi=150, bbox_inches='tight')
    
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
    parser.add_argument("session_id", type=int, nargs='?', help="Ecephys session ID to train on (optional if --dataset-dir provided)")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to existing preprocessed dataset (if omitted, a new one is created in preprocessed/)"
    )
    args = parser.parse_args()
    
    if args.session_id is None:
        if args.dataset_dir is None:
            parser.error("Either session_id or --dataset-dir must be provided")
        import re
        match = re.search(r'session_(\d+)', args.dataset_dir)
        if match:
            args.session_id = int(match.group(1))
        else:
            parser.error("Could not extract session_id from dataset path. Please provide session_id explicitly.")
    
    main(args.session_id, args.dataset_dir)
