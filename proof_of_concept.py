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
from src.models.snn import SNNEncoder
from src.losses.distillation import DistillationLoss


def setup_data(session_id: int, dataset_dir: Optional[str] = None):
    """
    Load and preprocess data for training
    
    Returns dict with:
        - device, train/test context/target tensors
        - metadata (num_units, context_size, etc.)
    """
    print("\n========== Data Setup ==========")
    
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
    
    return {
        'device': device,
        'train_context': train_context,
        'train_target': train_target,
        'test_context': test_context,
        'test_target': test_target,
        'num_units': num_units,
        'context_size': context_size,
        'target_size': target_size,
        'latent_dim': 128,
        'trial_aligned': trial_aligned,
        'n_trials': n_trials,
        'n_train': n_train,
    }


def phase_1(data_dict):
    """
    Phase 1: Train JEPA teacher (encoder + predictor)
    
    Returns dict with:
        - encoder, predictor models
        - training metrics
    """
    print("\n========== Phase 1: JEPA Teacher Training ==========")
    
    device = data_dict['device']
    train_context = data_dict['train_context']
    train_target = data_dict['train_target']
    test_context = data_dict['test_context']
    test_target = data_dict['test_target']
    num_units = data_dict['num_units']
    latent_dim = data_dict['latent_dim']
    context_size = data_dict['context_size']
    target_size = data_dict['target_size']
    
    encoder = NeuralEncoder(num_units, latent_dim, encoder_type="transformer").to(device)
    predictor = NeuralPredictor(latent_dim, target_size, context_size, predictor_type="rnn", hidden_dim=256, num_layers=2).to(device)
    
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
    
    # Compute final test performance
    with torch.no_grad():
        final_test_loss, final_test_pred, final_test_sigreg = lejepa_loss(
            test_context_flat, test_target_flat, test_pred_flat,
            num_epochs, lambd, num_slices
        )
    
    print("Phase 1 training complete")
    
    return {
        'encoder': encoder,
        'predictor': predictor,
        'total_losses': total_losses,
        'pred_losses': pred_losses,
        'sigreg_losses': sigreg_losses,
        'final_test_loss': final_test_loss,
        'final_test_pred': final_test_pred,
        'final_test_sigreg': final_test_sigreg,
    }


def phase_2(encoder, data_dict):
    """
    Phase 2: SNN distillation from frozen teacher
    
    Returns dict with:
        - snn_encoder model
        - distillation metrics
    """
    print("\n========== Phase 2: SNN Distillation ==========")
    
    device = data_dict['device']
    train_context = data_dict['train_context']
    test_context = data_dict['test_context']
    num_units = data_dict['num_units']
    latent_dim = data_dict['latent_dim']
    context_size = data_dict['context_size']
    
    # Freeze teacher
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    snn_encoder = SNNEncoder(
        input_dim=num_units,
        hidden_dims=[256, 128],
        latent_dim=latent_dim,
        timesteps=context_size,
        beta=0.5,
        threshold=1.0,
        homeostatic_target=0.05,
    ).to(device)

    distillation_loss_fn = DistillationLoss(
        latent_dim=latent_dim,
        cca_weight=1.0,
        homeostatic_weight=0.1,
    )

    optimizer = optim.Adam(snn_encoder.parameters(), lr=1e-3)

    cca_losses = []
    cca_sims = []
    homeo_penalties = []
    test_cca_sims = []

    num_epochs = 100
    batch_size = 16  # batch*time = 16*20 = 320 > latent_dim=128

    n_train = train_context.shape[0]

    for epoch in range(num_epochs):
        snn_encoder.train()
        
        epoch_cca = 0
        epoch_sim = 0
        epoch_homeo = 0
        num_batches = 0
        
        indices = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:min(i + batch_size, n_train)]
            context_batch = train_context[batch_idx]  # (batch, n_units, time)
            
            # Teacher latents (frozen)
            with torch.no_grad():
                teacher_latent = encoder(context_batch.permute(0, 2, 1))  # (batch, time, latent)
            
            # SNN latents
            snn_latent, snn_metrics = snn_encoder(context_batch)  # (batch, time, latent)
            
            # Flatten time for CCA: (batch, time, latent) -> (batch*time, latent)
            teacher_flat = teacher_latent.reshape(-1, latent_dim)
            snn_flat = snn_latent.reshape(-1, latent_dim)
            
            homeo_penalty = snn_encoder.compute_homeostatic_penalty(snn_metrics)
            
            loss, metrics = distillation_loss_fn(snn_flat, teacher_flat, homeo_penalty)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_cca += metrics['cca_loss']
            epoch_sim += metrics['cca_similarity']
            epoch_homeo += metrics['homeostatic_penalty']
            num_batches += 1
        
        cca_losses.append(epoch_cca / num_batches)
        cca_sims.append(epoch_sim / num_batches)
        homeo_penalties.append(epoch_homeo / num_batches)
        
        # Evaluate on test set
        snn_encoder.eval()
        with torch.no_grad():
            test_teacher_latent = encoder(test_context.permute(0, 2, 1))
            test_snn_latent, test_snn_metrics = snn_encoder(test_context)
            
            test_teacher_flat = test_teacher_latent.reshape(-1, latent_dim)
            test_snn_flat = test_snn_latent.reshape(-1, latent_dim)
            test_homeo = snn_encoder.compute_homeostatic_penalty(test_snn_metrics)
            
            _, test_metrics = distillation_loss_fn(test_snn_flat, test_teacher_flat, test_homeo)
            test_cca_sims.append(test_metrics['cca_similarity'])
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train - CCA Loss={cca_losses[-1]:.4f}, Sim={cca_sims[-1]:.4f}, Homeo={homeo_penalties[-1]:.4f}")
            print(f"  Test  - CCA Sim={test_cca_sims[-1]:.4f}")
    
    print("Phase 2 distillation complete")
    
    return {
        'snn_encoder': snn_encoder,
        'cca_losses': cca_losses,
        'cca_sims': cca_sims,
        'homeo_penalties': homeo_penalties,
        'test_cca_sims': test_cca_sims,
    }


def log_output(run_dir, data_dict, phase1_results, phase2_results):
    """
    Generate plots and print summary statistics
    """
    print("\n========== Generating Plots & Summary ==========")
    
    device = data_dict['device']
    encoder = phase1_results['encoder']
    predictor = phase1_results['predictor']
    snn_encoder = phase2_results['snn_encoder']
    
    train_context = data_dict['train_context']
    test_context = data_dict['test_context']
    test_target = data_dict['test_target']
    num_units = data_dict['num_units']
    
    encoder.eval()
    predictor.eval()
    snn_encoder.eval()
    
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
    
    # Phase 1 plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    
    plot_loss_curves(
        phase1_results['total_losses'], 
        phase1_results['pred_losses'], 
        phase1_results['sigreg_losses'], 
        ax=axes1[0, 0]
    )
    plot_prediction_accuracy(z_pred_flat, z_target_flat, ax=axes1[0, 1])
    plot_spike_count_distribution(test_context.cpu().numpy().reshape(-1, num_units), ax=axes1[1, 0])
    plot_prediction_vs_actual(z_pred_flat, z_target_flat, ax=axes1[1, 1])
    
    plt.tight_layout()
    plt.savefig(run_dir / 'phase1_teacher.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Phase 2 plots
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    
    axes2[0].plot(phase2_results['cca_losses'])
    axes2[0].set_xlabel('Epoch')
    axes2[0].set_ylabel('CCA Loss')
    axes2[0].set_title('CCA Loss (Train)')
    axes2[0].grid(True)
    
    axes2[1].plot(phase2_results['cca_sims'], label='Train')
    axes2[1].plot(phase2_results['test_cca_sims'], label='Test')
    axes2[1].set_xlabel('Epoch')
    axes2[1].set_ylabel('CCA Similarity')
    axes2[1].set_title('CCA Similarity')
    axes2[1].legend()
    axes2[1].grid(True)
    
    axes2[2].plot(phase2_results['homeo_penalties'])
    axes2[2].set_xlabel('Epoch')
    axes2[2].set_ylabel('Homeostatic Penalty')
    axes2[2].set_title('Homeostatic Penalty')
    axes2[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(run_dir / 'phase2_distillation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    session_id = data_dict.get('session_id', 'N/A')
    n_trials = data_dict['n_trials']
    n_train = data_dict['n_train']
    context_size = data_dict['context_size']
    target_size = data_dict['target_size']
    latent_dim = data_dict['latent_dim']
    
    print("\n" + "="*60)
    print("SUMMARY: PROOF OF CONCEPT")
    print("="*60)
    print(f"Session ID: {session_id}")
    print(f"Units: {num_units} | Trials: {n_trials} (train: {n_train}, test: {n_trials-n_train})")
    print(f"Architecture: {num_units}→{latent_dim} latent")
    print(f"  Context: {context_size} bins (0-200ms)")
    print(f"  Target: {target_size} bins (200-400ms)")
    print(f"  Bin size: {data_dict['trial_aligned']['bin_size_ms']:.1f}ms")
    
    print(f"\n--- Phase 1: JEPA Teacher Training ---")
    print(f"  Initial Loss: {phase1_results['total_losses'][0]:.4f}")
    print(f"  Final Loss: {phase1_results['total_losses'][-1]:.4f}")
    print(f"    Prediction: {phase1_results['pred_losses'][-1]:.4f}")
    print(f"    SIGReg: {phase1_results['sigreg_losses'][-1]:.4f}")
    improvement = (phase1_results['total_losses'][0] - phase1_results['total_losses'][-1]) / phase1_results['total_losses'][0] * 100
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Test Loss: {phase1_results['final_test_loss'].item():.4f}")
    
    print(f"\n--- Phase 2: SNN Distillation ---")
    print(f"  Initial CCA Similarity: {phase2_results['cca_sims'][0]:.4f}")
    print(f"  Final CCA Similarity (Train): {phase2_results['cca_sims'][-1]:.4f}")
    print(f"  Final CCA Similarity (Test): {phase2_results['test_cca_sims'][-1]:.4f}")
    print(f"  Final Homeostatic Penalty: {phase2_results['homeo_penalties'][-1]:.6f}")
    
    phase1_success = phase1_results['total_losses'][-1] < phase1_results['total_losses'][0]
    phase2_success = phase2_results['cca_sims'][-1] > 0.5  # Threshold for meaningful alignment
    
    print(f"\n--- Results ---")
    print(f"  Phase 1: {'✓ Success' if phase1_success else '✗ Needs tuning'}")
    print(f"  Phase 2: {'✓ Success' if phase2_success else '✗ Needs tuning'}")
    print(f"  Overall: {'✓ PoC Successful' if (phase1_success and phase2_success) else '○ Partial success / needs tuning'}")
    print("="*60)
    
    print(f"\nPlots saved to {run_dir}")
    print(f"  - phase1_teacher.png")
    print(f"  - phase2_distillation.png")



def main(
    session_id: int,
    dataset_dir: Optional[str] = None,
):
    """
    Main orchestrator for proof of concept:
    1. Setup data
    2. Train JEPA teacher (Phase 1)
    3. Distill to SNN (Phase 2)
    4. Log results and plots
    """
    print("="*60)
    print("PROOF OF CONCEPT: LE-JEPA + SNN DISTILLATION")
    print("="*60)
    
    # Setup runs directory
    runs_dir = Path(__file__).parent / "runs" / "proof_of_concept"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Get next run number
    existing_runs = [d for d in runs_dir.glob('run_*') if d.is_dir()]
    run_num = 1 if not existing_runs else max([int(d.name.split('_')[1]) for d in existing_runs]) + 1

    run_dir = runs_dir / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    
    # Phase 0: Setup data
    data_dict = setup_data(session_id, dataset_dir)
    data_dict['session_id'] = session_id  # Add for logging
    
    # Phase 1: Train JEPA teacher
    phase1_results = phase_1(data_dict)
    
    # Phase 2: SNN distillation
    phase2_results = phase_2(phase1_results['encoder'], data_dict)
    
    # Log outputs
    log_output(run_dir, data_dict, phase1_results, phase2_results)
    
    print(f"\n{'='*60}")
    print("PROOF OF CONCEPT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LE-JEPA model on single Visual Behavior Neuropixels session")
    parser.add_argument("session", type=int, nargs='?', help="Ecephys session ID to train on (optional if --dataset-dir provided)")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to existing preprocessed dataset (if omitted, a new one is created in preprocessed/)"
    )
    args = parser.parse_args()
    
    if args.session is None:
        if args.dataset_dir is None:
            parser.error("Either session or --dataset-dir must be provided")
        import re
        match = re.search(r'session_(\d+)', args.dataset_dir)
        if match:
            args.session = int(match.group(1))
        else:
            parser.error("Could not extract session from dataset path. Please provide session explicitly.")
    
    main(args.session, args.dataset_dir)
