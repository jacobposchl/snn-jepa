'''
Contains the loss functions for LeJEPA
'''

import torch
import torch.distributed as dist
import torch.nn.functional as F

def sigreg(embeddings, global_step, num_slices):
    """
    Sketched Isotropic Gaussian Regularization using Epps-Pulley test.
    
    Args:
        embeddings: Tensor of shape (N, D) where N=batch size, D=embedding dim
        global_step: Current training step (for synchronized random sampling)
        num_slices: Number of random projection directions
    
    Returns:
        Scalar loss value
    """


    dev = dict(device = embeddings.device)
    g = torch.Generator(**dev)
    g.manual_seed(global_step)

    # Sample random projection distributions
    # (rows = # of embedding dims , cols = # of slices we want)
    proj_shape = (embeddings.size(1), num_slices)
    A = torch.randn(proj_shape, generator = g, **dev)
    # Scales vectors to have a magnitude of 1.0
    A /= A.norm(p = 2, dim = 0)

    # Epps-Pulley statistic
    t = torch.linspace( -5, 5, 17, **dev)
    exp_f = torch.exp(-0.5 * t**2)

    # Empirical characteristic function
    x_t = (embeddings @ A).unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)

    if dist.is_initialized():
        dist.all_reduce(ecf, op = dist.ReduceOp.AVG)

    # Weighted L2 distance
    err = (ecf - exp_f).abs().square().mul(exp_f)

    N = embeddings.size(0) * (dist.get_world_size() if dist.is_initialized() else 1)
    T = torch.trapz(err, t, dim = 1) * N

    return T.mean()

def lejepa_loss(z_context, z_target, z_predicted, global_step, lambd, num_slices):
    """
    LeJEPA loss for temporal prediction on neural data.
    
    Args:
        z_context: Context embeddings at time t, shape (bs, D)
        z_target: Target embeddings at time t+Δt, shape (bs, D)
        z_predicted: Predicted future embeddings from predictor(z_context), shape (bs, D)
        global_step: Current training step
        lambd: Trade-off between prediction and SIGReg (default 0.05)
        num_slices: Number of projections for SIGReg (default 1024)
    
    Returns:
        Tuple of (total_loss, prediction_loss, sigreg_loss) for logging
    """

    # Prediction loss between predicted and actual future
    prediction_loss = F.mse_loss(z_predicted, z_target)

    # Regularize embeddings to be isotropic Gaussian 
    sigreg_context = sigreg(z_context, global_step, num_slices)
    sigreg_target = sigreg(z_target, global_step, num_slices)
    sigreg_loss = (sigreg_context + sigreg_target) / 2

    total_loss = (1 - lambd) * prediction_loss + lambd * sigreg_loss

    return total_loss, prediction_loss, sigreg_loss