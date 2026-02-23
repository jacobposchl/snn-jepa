"""
Script for distilling from trained JEPA teacher model to SNN model
"""

import torch
import torch.nn as nn


class CCALoss(nn.Module):
    """
    Calculate the rotationally-invariant CCA (Canonical Crorrelation Analysis)
    loss for distillation.

    This loss meansures the correlation between two high dimensionality representation of data.
    It is invariant to linear transformations making it perfect for comparing latent spaces
    where goemetry not transformations are important.

    Minimizes -1 * sum(correlations)
    """

    def __init__(self, out_dim: int, eps: float = 1e-4):
        """
        out_dim: Dimensionality of the latent space (should match SNN and JEPA)
        eps: Small regularization term to ensure numerical stability when inverting covariance matrices
        """
        super().__init__()
        self.out_dim = out_dim
        self.eps = eps

    def forward(self, H1: torch.Tensor, H2: torch.Tensor):
        """
        args:
            H1, H2: (Batch, Hidden_Dim) - The latent representations from SNN and JEPA

        return:
            cca_loss (scalar), cca_similarity (mean correlation for logging)
        """
        # 1. Center the variables (remove mean)
        H1_c = H1 - H1.mean(dim=0)
        H2_c = H2 - H2.mean(dim=0)

        # 2. Compute Covariance Matrices (1/(N-1) * X.T @ X)
        N = H1.shape[0]
        scale = 1.0 / (N - 1)
        # Transpose to shape (Hidden, Batch) for matrix mult
        # cross-covariance between H1 and H2
        sigma_hat12 = scale * torch.matmul(H1_c.T, H2_c)

        # add add eps to the diagonal for numerical stability when inverting
        eye = self.eps * torch.eye(self.out_dim, device=H1.device)

        # auto-covariance for H1 and H2
        sigma_hat11 = scale * torch.matmul(H1_c.T, H1_c) + eye
        sigma_hat22 = scale * torch.matmul(H2_c.T, H2_c) + eye

        # 3. Compute CCA via SVD
        # We need to compute: T = Sigma_11^(-1/2) * Sigma_12 * Sigma_22^(-1/2)

        # Using Cholesky decomposition to find an L such that Sigma = L @ L.T
        # this is effecitvely computing the square root of the co-variance Matrices
        # and is numerically stable
        L1 = torch.linalg.cholesky(sigma_hat11)
        L2 = torch.linalg.cholesky(sigma_hat22)

        # Invert L1 and L2 (since they are lower triangular, this is fast/stable)
        L1_inv = torch.inverse(L1)
        L2_inv = torch.inverse(L2)

        # T = L1_inv.T @ Sigma_12 @ L2_inv
        # T represents the cross-covariance between the whitened versions of H1 and H2
        # whitening means we can remove auto-correlation and focus
        # on the shared structure between H1 and H2
        T = torch.matmul(torch.matmul(L1_inv.T, sigma_hat12), L2_inv)

        # 4. Singular Value Decomposition
        # The singular values of T are the canonical correlations
        # Use simple svd_vals if you don't need eigenvectors
        # this set of singular values represents the strength of the correlation
        # between the two sets of variables in the latent space where
        # 0 means no correlation and 1 means perfect correlation
        singular_values = torch.linalg.svdvals(T)

        # 5. Loss = Negative Sum of correlations (maximize correlation)
        cca_loss = -1.0 * torch.sum(singular_values)
        cca_similarity = torch.mean(singular_values)  # For logging

        return cca_loss, cca_similarity


class DistillationLoss(nn.Module):
    """

    Calculate the combined loss for SNN distillation

    Should output the cca loss , cca similarity , homeostatic penalty , total loss

    Something like this:

    cca_loss, cca_sim = self.cca_loss(snn_latent, jepa_latent)
    total_loss = self.cca_weight * cca_loss + self.homeostatic_weight * homeostatic_penalty

    metrics = {
        'cca_loss': cca_loss.item(),
        'cca_similarity': cca_sim,
        'homeostatic_penalty': homeostatic_penalty.item(),
        'total_loss': total_loss.item()
    }

    return metrics


    Combined loss for SNN distillation: CCA + Homeostatic Penalty
    """

    def __init__(
        self, latent_dim: int, cca_weight: float = 1.0, homeostatic_weight: float = 0.1
    ):
        super().__init__()
        self.cca_loss_fn = CCALoss(out_dim=latent_dim)
        self.cca_weight = cca_weight
        self.homeostatic_weight = homeostatic_weight

    def forward(
        self,
        snn_latent: torch.Tensor,
        jepa_latent: torch.Tensor,
        homeostatic_penalty: torch.Tensor,
    ):
        """
        snn_latent: (Batch, Dim)
        jepa_latent: (Batch, Dim)
        homeostatic_penalty: scalar tensor from SNNEncoder
        """

        # compute CCA loss
        cca_loss, cca_sim = self.cca_loss_fn(snn_latent, jepa_latent)

        # perform a weighted sum of the losses
        total_loss = (self.cca_weight * cca_loss) + (
            self.homeostatic_weight * homeostatic_penalty
        )

        # return metrics for logging
        metrics = {
            "cca_loss": cca_loss.item(),
            "cca_similarity": cca_sim.item(),
            "homeostatic_penalty": homeostatic_penalty.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics
