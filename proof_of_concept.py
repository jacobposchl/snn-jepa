"""
Basic proof of concept for JEPA on a singular session.

Uses binning to input into the encoder model, getting the latent representation of the neural data.

Then, using the predictor model, predicts the future latent representations. 

Goal is to see if such predictions are stable, and loss decreases.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

from models.encoder import NeuralEncoder
from models.predictor import NeuralPredictor
from utils.binning import bin_session


def create_context_target_pairs(latents, context_len, horizon, stride=1):
    """
    Create context-target pairs from latent sequence.
    
    Args:
        latents: (batch, n_time_bins, latent_dim)
        context_len: Number of past timesteps to use as context
        horizon: Number of future timesteps to predict
        stride: Step size between pairs
    
    Returns:
        contexts: (n_pairs, context_len, latent_dim)
        targets: (n_pairs, horizon, latent_dim)
    """
    batch_size, n_time_bins, latent_dim = latents.shape
    
    contexts = []
    targets = []
    
    for i in range(0, n_time_bins - context_len - horizon + 1, stride):
        context = latents[:, i:i+context_len, :]  # (batch, context_len, latent_dim)
        target = latents[:, i+context_len:i+context_len+horizon, :]  # (batch, horizon, latent_dim)
        
        contexts.append(context)
        targets.append(target)
    
    if len(contexts) == 0:
        return None, None
    
    contexts = torch.cat(contexts, dim=0)  # (n_pairs * batch, context_len, latent_dim)
    targets = torch.cat(targets, dim=0)  # (n_pairs * batch, horizon, latent_dim)
    
    return contexts, targets


def train_predictor(predictor, contexts, targets, n_epochs=50, batch_size=32, lr=0.001):
    """Train the predictor model."""
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    
    n_pairs = contexts.shape[0]
    losses = []
    
    print(f"  Training for {n_epochs} epochs...")
    
    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        epoch_losses = []
        
        # Mini-batch training
        indices = torch.randperm(n_pairs)
        n_batches = (n_pairs + batch_size - 1) // batch_size
        
        batch_pbar = tqdm(
            range(0, n_pairs, batch_size),
            desc=f"  Epoch {epoch+1}/{n_epochs}",
            unit="batch",
            leave=False,
        )
        for i in batch_pbar:
            batch_indices = indices[i:i+batch_size]
            batch_contexts = contexts[batch_indices]
            batch_targets = targets[batch_indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, loss = predictor.compute_prediction_loss(
                batch_contexts, batch_targets, criterion
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            batch_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        epoch_pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
    
    return losses


def main():
    print("=" * 60)
    print("JEPA Proof of Concept")
    print("=" * 60)
    
    # Configuration
    data_dir = Path(__file__).parent / "visual_behavior_neuropixels_data"
    session_id = 1047969464
    output_dir = Path(__file__).parent / "out" / "proof_of_concept"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    bin_size_ms = 10.0
    latent_dim = 128
    context_len = 20  # 200ms of context
    horizon = 5       # 50ms prediction horizon
    n_epochs = 50
    
    print(f"\nSession: {session_id}")
    print(f"Latent dim: {latent_dim}, Context: {context_len}, Horizon: {horizon}")
    
    # Load and bin data
    print("\nLoading session...")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=data_dir)
    session = cache.get_ecephys_session(ecephys_session_id=session_id)
    
    print("Binning neural data...")
    result = bin_session(
        session,
        bin_size_ms=bin_size_ms,
        start_time=100.0,
        end_time=600.0,  # 500 seconds of data
        quality_filter=None,
        as_torch=True,
    )
    
    binned = result["binned"]  # (n_units, n_time_bins)
    n_units, n_time_bins = binned.shape
    
    print(f"  Binned shape: {binned.shape}")
    print(f"  Time bins: {n_time_bins} ({n_time_bins * bin_size_ms / 1000:.1f}s)")
    
    # Create models
    print("\nCreating models...")
    encoder = NeuralEncoder(n_units=n_units, latent_dim=latent_dim)
    predictor = NeuralPredictor(
        latent_dim=latent_dim,
        horizon=horizon,
        context_len=context_len,
    )
    
    # Encode the data
    print("Encoding neural data...")
    with torch.no_grad():
        binned_batch = binned.unsqueeze(0).transpose(1, 2)  # (1, n_time_bins, n_units)
        latents = encoder(binned_batch)  # (1, n_time_bins, latent_dim)
    
    print(f"  Latents shape: {latents.shape}")
    print(f"  Compression: {n_units} units -> {latent_dim} latent dim")
    
    # Create training pairs
    print("\nCreating context-target pairs...")
    contexts, targets = create_context_target_pairs(
        latents, context_len=context_len, horizon=horizon, stride=1
    )
    
    if contexts is None:
        print("  ERROR: Not enough data for context-target pairs")
        return
    
    n_pairs = contexts.shape[0]
    print(f"  Created {n_pairs} context-target pairs")
    
    # Split into train/val
    val_split = int(0.9 * n_pairs)
    train_contexts = contexts[:val_split]
    train_targets = targets[:val_split]
    val_contexts = contexts[val_split:]
    val_targets = targets[val_split:]
    
    print(f"  Train: {len(train_contexts)}, Val: {len(val_contexts)}")
    
    # Evaluate before training
    print("\nEvaluating before training...")
    with torch.no_grad():
        _, initial_loss = predictor.compute_prediction_loss(
            val_contexts[:100], val_targets[:100]  # Sample for speed
        )
    print(f"  Initial validation loss: {initial_loss.item():.6f}")
    
    # Train predictor
    print("\nTraining predictor...")
    train_losses = train_predictor(
        predictor, train_contexts, train_targets, n_epochs=n_epochs
    )
    
    # Evaluate after training
    print("\nEvaluating after training...")
    with torch.no_grad():
        val_predictions, final_loss = predictor.compute_prediction_loss(
            val_contexts[:100], val_targets[:100]
        )
    print(f"  Final validation loss: {final_loss.item():.6f}")
    print(f"  Improvement: {((initial_loss - final_loss) / initial_loss * 100).item():.1f}%")
    
    # Save training curve
    print("\nSaving outputs...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: training_curve.png")
    
    # Plot prediction examples
    n_examples = min(5, len(val_contexts))
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 2*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i in range(n_examples):
            ax = axes[i]
            ctx = val_contexts[i:i+1]
            tgt = val_targets[i:i+1]
            pred, _ = predictor.compute_prediction_loss(ctx, tgt)
            
            # Plot first 3 latent dimensions
            time_steps = np.arange(horizon)
            for dim in range(min(3, latent_dim)):
                ax.plot(
                    time_steps, tgt[0, :, dim].numpy(),
                    'o-', alpha=0.7, label=f'Actual Dim {dim}'
                )
                ax.plot(
                    time_steps, pred[0, :, dim].numpy(),
                    's--', alpha=0.7, label=f'Predicted Dim {dim}'
                )
            
            ax.set_title(f"Example {i+1}: Predicted vs Actual")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Latent value")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: prediction_examples.png")
    
    # Save metrics
    metrics = {
        "initial_loss": float(initial_loss.item()),
        "final_loss": float(final_loss.item()),
        "improvement_percent": float((initial_loss - final_loss) / initial_loss * 100),
        "n_pairs": int(n_pairs),
        "train_pairs": int(len(train_contexts)),
        "val_pairs": int(len(val_contexts)),
        "latent_dim": int(latent_dim),
        "context_len": int(context_len),
        "horizon": int(horizon),
    }
    
    import json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: metrics.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Initial loss: {initial_loss.item():.6f}")
    print(f"Final loss:   {final_loss.item():.6f}")
    print(f"Improvement:  {metrics['improvement_percent']:.1f}%")
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
