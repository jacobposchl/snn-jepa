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
from src.data.preprocess import NeuropixelsPreprocessor
from src.models.encoder import NeuralEncoder
from src.models.predictor import NeuralPredictor
from src.losses.lejepa import lejepa_loss
from src.plots.training import plot_loss_curves
from src.plots.latent_space import plot_prediction_accuracy
from src.plots.neural_activity import plot_spike_count_distribution
from src.plots.model_performance import plot_prediction_vs_actual


def get_next_run_number(base_dir: Path) -> int:
    """Get the next run number in the runs directory."""
    runs = [d for d in base_dir.glob('run_*') if d.is_dir()]
    if not runs:
        return 1
    max_run = max([int(d.name.split('_')[1]) for d in runs])
    return max_run + 1


def find_existing_dataset(base_dir: Path, session_id: int) -> Optional[Path]:
    """
    Look for an existing dataset for this session in any previous run.
    Returns the path to processed.pkl if found, None otherwise.
    """
    runs = sorted([d for d in base_dir.glob('run_*') if d.is_dir()], 
                  key=lambda x: int(x.name.split('_')[1]), reverse=True)
    
    for run_dir in runs:
        dataset_file = run_dir / "dataset" / "processed.pkl"
        metadata_file = run_dir / "dataset" / "session_id.txt"
        
        # Check if dataset exists
        if dataset_file.exists():
            # If metadata exists, verify session ID matches
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    stored_session_id = int(f.read().strip())
                    if stored_session_id == session_id:
                        return dataset_file
            else:
                # No metadata, assume it's for this session and return it
                # (create metadata for future runs)
                return dataset_file
    
    return None


def main(session_id: int):
    
    # Setup runs directory
    runs_dir = Path(__file__).parent / "runs" / "proof_of_concept"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists for this session in any previous run
    existing_dataset = find_existing_dataset(runs_dir, session_id)
    if existing_dataset:
        print(f"Loading dataset from {existing_dataset}...")
        with open(existing_dataset, 'rb') as f:
            processed = pickle.load(f)
    else:
        # Create new run directory
        run_num = get_next_run_number(runs_dir)
        run_dir = runs_dir / f"run_{run_num}"
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = run_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset
        print("Creating dataset...")
        data_dir = Path(__file__).parent / "visual_behavior_neuropixels_data"
        handler = VBNDataHandler(str(data_dir))
        session_data = handler.load_session(session_id)
        
        processed = (NeuropixelsPreprocessor(session_data)
            .validate_integrity()
            .clean()
            .filter_units(min_snr=1.0, max_isi_violations=1.0, min_firing_rate=0.1)
            .create_windows(window_size_ms=500, stride_ms=100, align_to='stimulus_change')
            .get())
        
        # Save dataset
        dataset_file = dataset_dir / "processed.pkl"
        print(f"Saving dataset to {dataset_file}...")
        with open(dataset_file, 'wb') as f:
            pickle.dump(processed, f)
        
        # Save session ID for future reference
        metadata_file = dataset_dir / "session_id.txt"
        with open(metadata_file, 'w') as f:
            f.write(str(session_id))
    
    print("Dataset ready")

    # Dataset debugging
    print("\n========== Dataset Debug Info ==========")
    windows = processed['windows']
    units = processed['units']
    num_units = len(units)
    num_windows = processed['window_metadata']['num_windows']
    
    spike_counts = np.array([windows[uid] for uid in units.index])
    print(f"Raw spike counts shape: {spike_counts.shape}")
    print(f"  Units: {num_units} | Windows: {num_windows}")
    print(f"  Min: {spike_counts.min():.6f} | Max: {spike_counts.max():.6f}")
    print(f"  Mean: {spike_counts.mean():.6f} | Std: {spike_counts.std():.6f}")
    print(f"  % zeros: {(spike_counts == 0).sum() / spike_counts.size * 100:.1f}%")
    
    # Move to device and convert to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spike_counts = torch.FloatTensor(spike_counts).T.to(device)
    
    # Standardize each unit (z-score normalization)
    spike_mean = spike_counts.mean(dim=0, keepdim=True)
    spike_std = spike_counts.std(dim=0, keepdim=True)
    spike_counts = (spike_counts - spike_mean) / (spike_std + 1e-8)
    
    print(f"\nNormalized spike counts shape: {spike_counts.shape}")
    print(f"  Min: {spike_counts.min().item():.6f} | Max: {spike_counts.max().item():.6f}")
    print(f"  Mean: {spike_counts.mean().item():.6f} | Std: {spike_counts.std().item():.6f}")
    print("="*40)
    
    # Model training pipeline
    print("\nTraining model...")
    
    latent_dim = 128
    context_len = 5
    horizon = 5  # Predict 5 steps ahead to capture temporal patterns
    
    encoder = NeuralEncoder(num_units, latent_dim, encoder_type="mlp").to(device)
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
    
    for epoch in range(num_epochs):
        total_loss_epoch = 0
        pred_loss_epoch = 0
        sigreg_loss_epoch = 0
        num_batches = 0
        
        for i in range(0, num_windows - context_len - horizon, batch_size):
            batch_end = min(i + batch_size, num_windows - context_len - horizon)
            
            context_data = []
            target_data = []
            
            for j in range(i, batch_end):
                context_data.append(spike_counts[j:j+context_len])
                target_data.append(spike_counts[j+context_len:j+context_len+horizon])
            
            context_batch = torch.stack(context_data)
            target_batch = torch.stack(target_data)
            
            z_context = encoder(context_batch)
            z_target = encoder(target_batch)
            
            z_context_flat = z_context[:, -1, :]
            z_predicted = predictor(z_context)[:, -1, :]
            z_target_flat = z_target[:, -1, :]
            
            loss, pred_loss, sigreg_loss = lejepa_loss(
                z_context_flat, z_target_flat, z_predicted,
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
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss_epoch/num_batches:.4f}, "
                  f"Pred={pred_loss_epoch/num_batches:.4f}, "
                  f"SIGReg={sigreg_loss_epoch/num_batches:.4f}")
    
    print("training complete")

    # Plots & Summary
    print("generating plots...")
    
    encoder.eval()
    predictor.eval()
    
    with torch.no_grad():
        test_idx = num_windows // 2
        context_test = spike_counts[test_idx:test_idx+context_len].unsqueeze(0)
        target_test = spike_counts[test_idx+context_len:test_idx+context_len+horizon].unsqueeze(0)
        z_target_test = encoder(target_test)[:, -1, :]
        z_pred_test = predictor(encoder(context_test))[:, -1, :]
        
        all_embeddings = encoder(spike_counts.unsqueeze(0))[:, :, :].squeeze(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    plot_loss_curves(total_losses, pred_losses, sigreg_losses, ax=axes[0, 0])
    plot_prediction_accuracy(z_pred_test, z_target_test, ax=axes[0, 1])
    plot_spike_count_distribution(spike_counts.cpu().numpy(), ax=axes[1, 0])
    plot_prediction_vs_actual(z_pred_test, z_target_test, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('poc_results.png', dpi=150, bbox_inches='tight')
    
    print("=========== summary: ==========")
    print(f"Session ID: {session_id}")
    print(f"Units: {num_units} | Windows: {num_windows}")
    print(f"Architecture: {num_units}→{latent_dim} latent (RNN predictor)")
    print(f"  Context: {context_len} timesteps | Horizon: {horizon} timesteps")
    print(f"\nTraining:")
    print(f"  Epochs: {num_epochs} | Batch size: {batch_size} | λ: {lambd}")
    print(f"  Final Loss: {total_losses[-1]:.4f}")
    print(f"  Prediction: {pred_losses[-1]:.4f} | SIGReg: {sigreg_losses[-1]:.4f}")
    print(f"  Improvement: {(total_losses[0] - total_losses[-1]) / total_losses[0] * 100:.1f}%")
    print(f"\nResults: PoC {'successful' if total_losses[-1] < total_losses[0] else 'inconclusive'}")
    print("="*60)    

    # Plots & Summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LE-JEPA model on single Visual Behavior Neuropixels session")
    parser.add_argument("session_id", type=int, help="Ecephys session ID to train on")
    args = parser.parse_args()
    
    main(args.session_id)
    

