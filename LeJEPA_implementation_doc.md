# LeJEPA Implementation Guide

This guide explains how to implement **LeJEPA (Latent-Euclidean Joint-Embedding Predictive Architecture)** from the paper "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics" by Balestriero & LeCun.

## Overview

LeJEPA is a self-supervised learning method that combines:
1. **Prediction Loss**: Forces embeddings of different views to be similar
2. **SIGReg (Sketched Isotropic Gaussian Regularization)**: Ensures embeddings follow an isotropic Gaussian distribution

The method is remarkably simple (~50 lines) and works across datasets, architectures, and scales.

## Core Implementation

### 1. SIGReg Loss (Algorithm 1 from paper)

```python
import torch
import torch.distributed as dist

def SIGReg(x, global_step, num_slices=256):
    """
    Sketched Isotropic Gaussian Regularization using Epps-Pulley test.
    
    Args:
        x: Embeddings tensor of shape (N, K) where N=batch size, K=embedding dim
        global_step: Current training step (for synchronized random sampling)
        num_slices: Number of random projection directions (|A| in paper)
    
    Returns:
        Scalar loss value
    """
    # Slice sampling -- synced across devices
    dev = dict(device=x.device)
    g = torch.Generator(**dev)
    g.manual_seed(global_step)
    
    # Sample random projection directions A = {a1, ..., a_M}
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, generator=g, **dev)
    A /= A.norm(p=2, dim=0)  # Normalize to unit norm
    
    # --- Epps-Pulley statistic (see Sec 4.2.3) ---
    # Integration points for characteristic function
    t = torch.linspace(-5, 5, 17, **dev)
    
    # Theoretical CF for N(0,1) with Gaussian window
    exp_f = torch.exp(-0.5 * t**2)
    
    # Empirical CF -- gathered across devices if using DDP
    x_t = (x @ A).unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)  # (M, T)
    
    # If distributed training, average across all devices
    if dist.is_initialized():
        dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
    
    # Weighted L2 distance between empirical and theoretical CF
    err = (ecf - exp_f).abs().square().mul(exp_f)
    
    # Numerical integration using trapezoidal rule
    N = x.size(0) * (dist.get_world_size() if dist.is_initialized() else 1)
    T = torch.trapz(err, t, dim=1) * N
    
    return T.mean()
```

### 2. LeJEPA Training Loss (Algorithm 2 from paper)

```python
def LeJEPA_loss(global_views, all_views, lambd=0.05):
    """
    Complete LeJEPA training objective.
    
    Args:
        global_views: List of tensors, each of shape (bs, K)
                     These are embeddings from large/global crops
        all_views: List of tensors, each of shape (bs, K)
                  All views including both global and local crops
        lambd: Trade-off parameter between prediction and SIGReg (λ in paper)
    
    Returns:
        Total loss value
    """
    bs = global_views[0].size(0)
    K = global_views[0].size(1)
    
    # Compute centers (mean of global views)
    g_emb = torch.cat(global_views)  # (V_g * bs, K)
    centers = g_emb.view(-1, bs, K).mean(0)  # (bs, K)
    
    # Prediction loss: all views predict the centers
    a_emb = torch.cat(all_views)  # (V * bs, K)
    a_emb = a_emb.view(-1, bs, K)  # (V, bs, K)
    
    # Squared L2 distance
    sim = (centers - a_emb).square().mean()
    
    # SIGReg loss: average over all views
    sigreg = torch.stack([
        SIGReg(emb, global_step=0)  # Pass actual global_step in practice
        for emb in a_emb
    ]).mean()
    
    # Combined loss
    return (1 - lambd) * sim + lambd * sigreg
```

## Complete Training Pipeline

### 3. Full Training Example

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class LeJEPATrainer:
    def __init__(
        self,
        backbone,
        lambd=0.05,
        num_slices=1024,
        lr=5e-4,
        weight_decay=1e-2,
        num_global_views=2,
        num_local_views=6
    ):
        """
        Args:
            backbone: Encoder network (e.g., ResNet, ViT)
            lambd: Trade-off hyperparameter (recommended: 0.05)
            num_slices: Number of projection directions (recommended: 1024)
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            num_global_views: Number of global crops (default: 2)
            num_local_views: Number of local crops (default: 6)
        """
        self.backbone = backbone
        self.lambd = lambd
        self.num_slices = num_slices
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            backbone.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100  # Total epochs
        )
        
        self.global_step = 0
    
    def create_views(self, images):
        """
        Create multiple augmented views of images.
        
        Returns:
            global_views: List of global crop tensors
            all_views: List of all crop tensors (global + local)
        """
        # Define augmentations
        global_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        global_views = [global_transform(images) for _ in range(self.num_global_views)]
        local_views = [local_transform(images) for _ in range(self.num_local_views)]
        
        return global_views, global_views + local_views
    
    def train_step(self, images):
        """Single training step."""
        # Create views
        global_views, all_views = self.create_views(images)
        
        # Forward pass through backbone
        global_embs = [self.backbone(view) for view in global_views]
        all_embs = [self.backbone(view) for view in all_views]
        
        # Compute loss
        loss = LeJEPA_loss(global_embs, all_embs, self.lambd)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.backbone.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.cuda()
            loss = self.train_step(images)
            total_loss += loss
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss:.4f}")
        
        self.scheduler.step()
        return total_loss / len(dataloader)
```

### 4. Usage Example

```python
import torchvision.models as models
from torchvision.datasets import ImageFolder

# Create backbone (any architecture works)
backbone = models.resnet50(pretrained=False)
backbone.fc = nn.Identity()  # Remove classification head

# Initialize trainer
trainer = LeJEPATrainer(
    backbone=backbone,
    lambd=0.05,  # Recommended default
    num_slices=1024,
    lr=5e-4,
    weight_decay=1e-2
)

# Prepare dataset
dataset = ImageFolder('path/to/imagenet', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    avg_loss = trainer.train_epoch(dataloader)
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(backbone.state_dict(), f'lejepa_epoch_{epoch+1}.pth')
```

## Key Hyperparameters

According to the paper's extensive ablations:

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `λ` (lambda) | 0.05 | Trade-off between prediction and SIGReg |
| `num_slices` | 1024 - 4096 | More slices = better but slower |
| `batch_size` | ≥ 128 | Works with small batches |
| `num_global_views` | 2 | Standard DINO setup |
| `num_local_views` | 6-8 | Standard DINO setup |
| Learning rate | 5e-4 or 5e-3 | Grid search recommended |
| Weight decay | 1e-2, 1e-5 | Grid search recommended |

## Architecture Support

LeJEPA works out-of-the-box with:
- **ResNets**: Use all views as global (no local crops needed)
- **Vision Transformers (ViTs)**: Use global + local crops
- **ConvNeXt**: Works with default settings
- **Any encoder**: Just ensure output is (batch_size, embedding_dim)

## Advanced Features

### Optional: Stochastic Weight Averaging (SWA)

```python
# For ViTs, applying SWA on the encoder producing μ can help
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model = AveragedModel(backbone)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)

# Update SWA model every epoch after warmup
if epoch > swa_start_epoch:
    swa_model.update_parameters(backbone)
    swa_scheduler.step()
```

### Optional: Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
backbone = DDP(backbone, device_ids=[local_rank])

# SIGReg already handles DDP synchronization via all_reduce
```

## Evaluation

After pretraining, evaluate using linear probing:

```python
# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False

# Add linear classifier
classifier = nn.Linear(backbone_output_dim, num_classes)

# Train only the classifier
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)

# Standard supervised training loop
for images, labels in dataloader:
    features = backbone(images)
    logits = classifier(features)
    loss = F.cross_entropy(logits, labels)
    # ... backward pass
```

## Key Advantages

1. **No heuristics**: No stop-gradient, no momentum encoder, no temperature
2. **Single hyperparameter**: Just tune λ
3. **Stable**: Works across architectures and datasets without tuning
4. **Efficient**: Linear time/memory complexity
5. **Simple**: ~50 lines of core code

## Citation

```bibtex
@article{balestriero2025lejepa,
  title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author={Balestriero, Randall and LeCun, Yann},
  journal={arXiv preprint arXiv:2511.08544},
  year={2025}
}
```