"""
Test SNN distillation on a single session using trained JEPA teacher.
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.optim as optim

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.preprocess import get_or_create_dataset, slice_trial_windows
from src.models.encoder import NeuralEncoder  # JEPA encoder
from src.models.snn import SNNEncoder  # Your SNN
from src.losses.distillation import DistillationLoss  # Your distillation loss


def main(session_id: int, dataset_dir: str = None):
    
    # Setup directories
    preprocessed_dir = Path(__file__).parent / "preprocessed"
    
    # Get dataset (same as proof_of_concept.py)
    dataset_path = Path(dataset_dir) if dataset_dir else None
    processed = get_or_create_dataset(
        session_id=session_id,
        preprocessed_dir=preprocessed_dir,
        dataset_path=dataset_path,
        brain_areas=['VISp', 'VISl'],
    )
    
    print("Dataset loaded")
    
    # Extract data
    trial_aligned = processed['trial_aligned']
    units = processed['units']
    num_units = len(units)
    trial_data = trial_aligned['binned']
    n_trials, _, n_time_bins = trial_data.shape
    
    print(f"Data shape: {trial_data.shape}")
    print(f"Units: {num_units}, Trials: {n_trials}, Time bins: {n_time_bins}")
    
    # Split train/test
    n_train = int(0.8 * n_trials)
    train_trials = trial_data[:n_train]
    
    # Create windows (same as proof_of_concept.py)
    context_start, context_end = 0, 20
    target_start, target_end = 20, 40
    
    train_context, train_target = slice_trial_windows(
        trial_data=train_trials,
        context_start=context_start,
        context_end=context_end,
        target_start=target_start,
        target_end=target_end,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to tensors
    train_context = torch.FloatTensor(train_context).to(device)  # (n_windows, n_units, context_len)
    
    # Simple normalization
    train_context = train_context / (train_context.std() + 1e-6)
    train_context = torch.clamp(train_context, -10, 10)
    
    print(f"Train context shape: {train_context.shape}")
    
    # --- Initialize models ---
    latent_dim = 64
    
    # JEPA teacher (frozen - pretend it's already trained)
    jepa_encoder = NeuralEncoder(num_units, latent_dim, encoder_type="transformer").to(device)
    jepa_encoder.eval()  # Freeze it
    for param in jepa_encoder.parameters():
        param.requires_grad = False
    
    # SNN student
    snn_encoder = SNNEncoder(
        input_dim=num_units,
        hidden_dims=[128, 64],
        latent_dim=latent_dim,
        timesteps=20,  # context_len
        beta=0.5,
        threshold=1.0,
        homeostatic_target=0.01
    ).to(device)
    
    # Distillation loss
    distill_loss = DistillationLoss(
        latent_dim=latent_dim,
        cca_weight=1.0,
        homeostatic_weight=0.1
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(snn_encoder.parameters(), lr=1e-3)
    
    print("\n--- Starting SNN Distillation ---")
    
    num_epochs = 50
    batch_size = 32
    
    for epoch in range(num_epochs):
        snn_encoder.train()
        
        epoch_loss = 0
        epoch_cca = 0
        epoch_homeo = 0
        num_batches = 0
        
        # Mini-batch training
        indices = torch.randperm(train_context.shape[0])
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:min(i + batch_size, len(indices))]
            context_batch = train_context[batch_idx].permute(0, 2, 1)  # (batch, time, units)
            
            # Get JEPA latents (frozen teacher)
            with torch.no_grad():
                jepa_latent = jepa_encoder(context_batch)  # (batch, time, latent_dim)
                jepa_latent = jepa_latent[:, -1, :]  # Take last timestep
            
            # Get SNN latents (student)
            snn_latent, metrics = snn_encoder(context_batch)  # (batch, latent_dim)
            
            # Compute homeostatic penalty
            homeostatic_penalty = snn_encoder.compute_homeostatic_penalty(metrics)
            
            # Distillation loss
            total_loss, loss_metrics = distill_loss(
                snn_latent, jepa_latent, homeostatic_penalty
            )
            
            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += loss_metrics['total_loss']
            epoch_cca += loss_metrics['cca_similarity']
            epoch_homeo += loss_metrics['homeostatic_penalty']
            num_batches += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Loss: {epoch_loss/num_batches:.4f}")
            print(f"  CCA Similarity: {epoch_cca/num_batches:.4f}")
            print(f"  Homeostatic: {epoch_homeo/num_batches:.4f}")
    
    print("\n✓ SNN distillation test complete!")
    print(f"Final CCA similarity: {epoch_cca/num_batches:.4f}")
    print(f"Final homeostatic penalty: {epoch_homeo/num_batches:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id", type=int, help="Session ID to test on")
    parser.add_argument("--dataset-dir", type=str, default=None, 
                        help="Path to existing preprocessed dataset")
    args = parser.parse_args()
    
    main(args.session_id, args.dataset_dir)