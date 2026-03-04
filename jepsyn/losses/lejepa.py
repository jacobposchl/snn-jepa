"""
Contains the loss functions for LeJEPA
"""

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

    dev = dict(device=embeddings.device)
    g = torch.Generator(**dev)
    g.manual_seed(global_step)

    # Sample random projection distributions
    # (rows = # of embedding dims , cols = # of slices we want)
    proj_shape = (embeddings.size(1), num_slices)
    A = torch.randn(proj_shape, generator=g, **dev)
    # Scales vectors to have a magnitude of 1.0
    A /= A.norm(p=2, dim=0)

    # Epps-Pulley statistic
    t = torch.linspace(-5, 5, 17, **dev)
    exp_f = torch.exp(-0.5 * t**2)

    # Empirical characteristic function
    x_t = (embeddings @ A).unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)

    if dist.is_initialized():
        dist.all_reduce(ecf, op=dist.ReduceOp.AVG)

    # Weighted L2 distance
    err = (ecf - exp_f).abs().square().mul(exp_f)

    T = torch.trapezoid(err, t, dim=1)

    return T.mean()


def _variance_loss(x):
    """
    Hinge Loss to maintain a standard deviation > 1
    """
    std = torch.sqrt(x.var(dim=0) + 1e-04)
    return torch.mean(F.relu(1.0 - std))


def _covariance_loss(x):
    """
    Penalizes off-diagonal covariance to decouple features
    """
    n, d = x.size()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (n - 1)
    off_diag = cov.pow(2).sum() - torch.diag(cov).pow(2).sum()
    return off_diag / d


def lejepa_loss(
    z_context,
    z_target,
    z_predicted,
    global_step,
    reg_type="sigreg",
    lambd=0.05,
    num_slices=1024,
    vic_sim=25.0,
    vic_std=25.0,
    vic_cov=1.0,
):
    """
    LeJEPA loss for masking JEPA on neural data. supports both VICReg and SIGReg regularization.

    Args:
        z_context: Mean-pooled context embeddings (masked input), shape (bs, D)
        z_target: Mean-pooled target embeddings (full input, EMA encoder), shape (bs, D)
        z_predicted: Mean-pooled predicted embeddings from predictor(Z_ctx), shape (bs, D)
        global_step: Current training step
        reg_type: Type of regularization to apply ('sigreg' or 'vicreg')
        lambd: Trade-off between prediction and SIGReg (default 0.05)
        num_slices: Number of projections for SIGReg (default 1024)
        vic_sim: Weight for VICReg similarity loss (default 25.0)
        vic_std: Weight for VICReg variance loss (default 25.0)
        vic_cov: Weight for VICReg covariance loss (default 1.0)

    Returns:
        Tuple of (total_loss, prediction_loss, sigreg_loss) for logging
    """

    # Prediction loss between predicted and actual future
    prediction_loss = F.mse_loss(z_predicted, z_target)

    if reg_type == "sigreg":
        # Regularize embeddings to be isotropic Gaussian
        sigreg_context = sigreg(z_context, global_step, num_slices)
        sigreg_target = sigreg(z_target, global_step, num_slices)
        reg_loss = (sigreg_context + sigreg_target) / 2
        total_loss = (1 - lambd) * prediction_loss + lambd * reg_loss

    elif reg_type == "vicreg":

        std_loss = (_variance_loss(z_context) + _variance_loss(z_target)) / 2
        cov_loss = (_covariance_loss(z_context) + _covariance_loss(z_target)) / 2

        reg_loss = (vic_std * std_loss) + (vic_cov * cov_loss)
        total_loss = (vic_sim * prediction_loss) + reg_loss

    else:
        raise ValueError(
            f"Unknown regularization type: {reg_type}\n"
            + "expected 'sigreg' or 'vicreg'"
        )

    return total_loss, prediction_loss, reg_loss

