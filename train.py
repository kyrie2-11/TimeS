"""
Training script for the Fuzzy Shapelet AutoEncoder (FSAE).

Implements the three-phase training workflow:
    Phase 1 – Pre-training:   Train Encoder + Decoder as a standard AutoEncoder
                               (Fuzzy Shapelet Layer is bypassed).
    Phase 2 – Shapelet Init:  Collect latent features Z from the pre-trained encoder,
                               run K-Means clustering, and use the resulting cluster
                               centres to initialise the Fuzzy Shapelet Layer parameters.
    Phase 3 – Fine-tuning:    Activate the Fuzzy Shapelet Layer and train end-to-end
                               with Reconstruction Loss + Compactness Loss.

A simple dummy multivariate time series dataset (sinusoidal waves + noise) is included
so the script can be run out-of-the-box without any external data files.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans

from model import FSAE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset / model hyper-parameters
NUM_SAMPLES = 500       # number of training windows
SEQ_LEN = 64            # window length L
NUM_CHANNELS = 3        # multivariate dimension M
LATENT_DIM = 32         # latent space dimensionality D
NUM_SHAPELETS = 8       # number of shapelet prototypes K
ALPHA = 1.0             # Student-t degrees of freedom

# Training hyper-parameters
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
LAMBDA_COMPACT = 0.5    # weight for the compactness loss in Phase 3


# ---------------------------------------------------------------------------
# Dummy dataset: sinusoidal waves with additive Gaussian noise
# ---------------------------------------------------------------------------

def make_dummy_dataset(
    num_samples: int = NUM_SAMPLES,
    seq_len: int = SEQ_LEN,
    num_channels: int = NUM_CHANNELS,
    noise_std: float = 0.1,
) -> TensorDataset:
    """
    Generate a synthetic multivariate time series dataset.

    Each sample is a superposition of sinusoids at different frequencies (one per
    channel), plus small Gaussian noise.  All samples represent *normal* behaviour
    and are used exclusively for training the anomaly detector.

    Args:
        num_samples:  Number of time series windows to generate.
        seq_len:      Length of each window L.
        num_channels: Number of channels (variables) M.
        noise_std:    Standard deviation of additive Gaussian noise.

    Returns:
        A TensorDataset of shape [num_samples, num_channels, seq_len].
    """
    t = torch.linspace(0, 2 * np.pi, seq_len)  # [L]
    data = []
    for _ in range(num_samples):
        channels = []
        for c in range(num_channels):
            # Each channel has a random frequency and phase offset
            freq = 1.0 + c + torch.rand(1).item()
            phase = torch.rand(1).item() * 2 * np.pi
            wave = torch.sin(freq * t + phase)
            channels.append(wave)
        sample = torch.stack(channels, dim=0)  # [M, L]
        noise = torch.randn_like(sample) * noise_std
        data.append(sample + noise)

    x = torch.stack(data, dim=0)  # [N, M, L]
    return TensorDataset(x)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error between input and reconstruction."""
    return nn.functional.mse_loss(x_hat, x)


def compactness_loss(
    memberships: torch.Tensor, distances: torch.Tensor
) -> torch.Tensor:
    """
    Fuzzy compactness loss: encourages Z to be close to the matched shapelets.

        L_compact = mean_over_batch( sum_k( U_k * d(Z, S_k)^2 ) )

    Args:
        memberships: [B, K] fuzzy membership matrix.
        distances:   [B, K] squared Euclidean distances from Z to each shapelet.
    """
    return torch.mean(torch.sum(memberships * distances, dim=1))


# ---------------------------------------------------------------------------
# Phase 1: Pre-training (standard AutoEncoder, fuzzy layer bypassed)
# ---------------------------------------------------------------------------

def pretrain(
    model: FSAE,
    loader: DataLoader,
    epochs: int = PRETRAIN_EPOCHS,
    lr: float = LR,
) -> None:
    """Train the Encoder and Decoder as a plain AutoEncoder."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("=" * 60)
    print("Phase 1: Pre-training (AutoEncoder)")
    print("=" * 60)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            # use_fuzzy=False: Z is passed directly to the decoder
            x_hat, _z, _z_hat, _mem, _dist = model(x, use_fuzzy=False)
            loss = reconstruction_loss(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:>3d}/{epochs}]  Recon Loss: {avg_loss:.6f}")


# ---------------------------------------------------------------------------
# Phase 2: Shapelet initialisation via K-Means in the latent space
# ---------------------------------------------------------------------------

def init_shapelets_kmeans(
    model: FSAE,
    loader: DataLoader,
    num_shapelets: int = NUM_SHAPELETS,
) -> None:
    """
    Collect all latent features Z from the pre-trained encoder, run K-Means,
    and use the cluster centres to initialise the Fuzzy Shapelet Layer.
    """
    model.eval()
    print("=" * 60)
    print("Phase 2: Shapelet Initialisation (K-Means on latent Z)")
    print("=" * 60)

    all_z = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            z = model.encode(x)  # [B, D]
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy()  # [N, D]

    print(f"  Running K-Means with K={num_shapelets} on {all_z.shape[0]} samples ...")
    kmeans = KMeans(n_clusters=num_shapelets, n_init=10, random_state=42)
    kmeans.fit(all_z)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [K, D]

    model.init_shapelets(centers.to(DEVICE))
    print(f"  Shapelet parameters initialised from K-Means cluster centres.")


# ---------------------------------------------------------------------------
# Phase 3: Fine-tuning (Fuzzy Shapelet Layer activated, joint loss)
# ---------------------------------------------------------------------------

def finetune(
    model: FSAE,
    loader: DataLoader,
    epochs: int = FINETUNE_EPOCHS,
    lr: float = LR,
    lambda_compact: float = LAMBDA_COMPACT,
) -> None:
    """End-to-end fine-tuning with Reconstruction Loss + Compactness Loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("=" * 60)
    print("Phase 3: Fine-tuning (FSAE, fuzzy layer active)")
    print("=" * 60)
    for epoch in range(1, epochs + 1):
        total_recon = 0.0
        total_compact = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            # use_fuzzy=True: full FSAE forward pass
            x_hat, z, z_hat, memberships, distances = model(x, use_fuzzy=True)
            l_recon = reconstruction_loss(x_hat, x)
            l_compact = compactness_loss(memberships, distances)
            loss = l_recon + lambda_compact * l_compact
            loss.backward()
            optimizer.step()
            total_recon += l_recon.item() * x.size(0)
            total_compact += l_compact.item() * x.size(0)
        n = len(loader.dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch [{epoch:>3d}/{epochs}]  "
                f"Recon: {total_recon/n:.6f}  "
                f"Compact: {total_compact/n:.6f}  "
                f"Total: {(total_recon + lambda_compact * total_compact)/n:.6f}"
            )


# ---------------------------------------------------------------------------
# Anomaly scoring (inference)
# ---------------------------------------------------------------------------

def compute_anomaly_scores(
    model: FSAE, loader: DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample reconstruction error (anomaly score) on the given data.

    Args:
        model:  Trained FSAE model.
        loader: DataLoader for the dataset to evaluate.

    Returns:
        scores: [N] tensor of per-sample MSE anomaly scores.
        all_memberships: [N, K] tensor of fuzzy memberships.
    """
    model.eval()
    scores = []
    all_memberships = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            x_hat, _z, _z_hat, memberships, _dist = model(x, use_fuzzy=True)
            # Per-sample MSE (mean over channels and time)
            err = ((x - x_hat) ** 2).mean(dim=[1, 2])  # [B]
            scores.append(err.cpu())
            all_memberships.append(memberships.cpu())
    return torch.cat(scores, dim=0), torch.cat(all_memberships, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Build dataset ----
    print("Generating dummy multivariate time series dataset ...")
    dataset = make_dummy_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"  Dataset: {len(dataset)} samples, shape [M={NUM_CHANNELS}, L={SEQ_LEN}]")

    # ---- Build model ----
    model = FSAE(
        in_channels=NUM_CHANNELS,
        seq_len=SEQ_LEN,
        latent_dim=LATENT_DIM,
        num_shapelets=NUM_SHAPELETS,
        alpha=ALPHA,
    ).to(DEVICE)
    print(f"\nModel: FSAE  |  Device: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- Phase 1: Pre-train ----
    pretrain(model, loader)

    # ---- Phase 2: Shapelet init ----
    init_shapelets_kmeans(model, loader, num_shapelets=NUM_SHAPELETS)

    # ---- Phase 3: Fine-tune ----
    finetune(model, loader)

    # ---- Anomaly scoring demo ----
    print("=" * 60)
    print("Anomaly Scoring (training data, for demonstration)")
    print("=" * 60)
    scores, memberships = compute_anomaly_scores(model, loader)
    threshold = scores.mean() + 3 * scores.std()
    num_anomalies = (scores > threshold).sum().item()
    print(f"  Anomaly score: mean={scores.mean():.6f}  std={scores.std():.6f}")
    print(f"  Threshold (mean + 3*std): {threshold:.6f}")
    print(f"  Samples flagged as anomalous: {num_anomalies} / {len(scores)}")
    print(f"  Mean membership per shapelet: {memberships.mean(0).numpy().round(4)}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
