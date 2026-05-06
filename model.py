"""
Fuzzy Shapelet AutoEncoder (FSAE) for Multivariate Time Series Anomaly Detection.

Architecture:
    - Encoder: 1D-CNN network mapping [Batch, Channels, SeqLen] -> latent Z
    - Fuzzy Shapelet Matching Layer: K learnable prototype vectors (Shapelets) in
      latent space; computes Student-t fuzzy membership U and fuses shapelets into Z_hat
    - Decoder: 1D-Transpose-CNN network reconstructing the time series from Z_hat (or Z)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    1D-CNN encoder that maps input time series [B, C, L] to a latent vector Z [B, D].
    """

    def __init__(self, in_channels: int, latent_dim: int, seq_len: int):
        """
        Args:
            in_channels: Number of input channels (multivariate dimension M).
            latent_dim:  Dimensionality of the latent space D.
            seq_len:     Length of the input time series window L.
        """
        super().__init__()
        self.seq_len = seq_len

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L -> L/2
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/2 -> L/4
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # Global average pooling collapses the temporal dimension to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            z: [B, D]
        """
        out = self.conv_layers(x)       # [B, 128, L/4]
        out = self.global_pool(out)     # [B, 128, 1]
        out = out.squeeze(-1)           # [B, 128]
        z = self.fc(out)                # [B, D]
        return z


class FuzzyShapeletLayer(nn.Module):
    """
    Fuzzy Shapelet Matching Layer.

    Stores K learnable prototype vectors (shapelets) in the latent space.
    Given a latent vector Z, it:
        1. Computes the squared Euclidean distance from Z to each shapelet.
        2. Converts distances to fuzzy membership values U using the Student-t
           distribution kernel (as in Deep Embedded Clustering, DEC).
        3. Produces a fused representation Z_hat as the membership-weighted sum
           of the K shapelets.
    """

    def __init__(self, num_shapelets: int, latent_dim: int, alpha: float = 1.0):
        """
        Args:
            num_shapelets: Number of prototype shapelets K.
            latent_dim:    Dimensionality of the latent space D.
            alpha:         Degrees of freedom for the Student-t kernel (default 1).
        """
        super().__init__()
        self.alpha = alpha
        # Learnable shapelet prototypes: [K, D]
        self.shapelets = nn.Parameter(torch.randn(num_shapelets, latent_dim))

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D] latent representation from the encoder.
        Returns:
            z_hat:       [B, D] membership-weighted combination of shapelets.
            memberships: [B, K] fuzzy membership matrix U.
            distances:   [B, K] squared Euclidean distances from z to each shapelet.
        """
        # Squared Euclidean distances: [B, K]
        z_exp = z.unsqueeze(1)                          # [B, 1, D]
        s_exp = self.shapelets.unsqueeze(0)             # [1, K, D]
        distances = torch.sum((z_exp - s_exp) ** 2, dim=2)  # [B, K]

        # Student-t membership (DEC soft assignment)
        # q_k = (1 + d_k^2 / alpha)^(-(alpha+1)/2)
        q = (1.0 + distances / self.alpha) ** (-((self.alpha + 1.0) / 2.0))  # [B, K]
        memberships = q / q.sum(dim=1, keepdim=True)    # [B, K], sums to 1

        # Weighted fusion of shapelets: Z_hat = sum_k( U_k * S_k )
        z_hat = torch.matmul(memberships, self.shapelets)  # [B, D]

        return z_hat, memberships, distances


class Decoder(nn.Module):
    """
    1D-CNN decoder that maps a latent vector [B, D] back to a time series [B, C, L].
    """

    def __init__(self, out_channels: int, latent_dim: int, seq_len: int):
        """
        Args:
            out_channels: Number of output channels (multivariate dimension M).
            latent_dim:   Dimensionality of the latent space D.
            seq_len:      Target output sequence length L.
        """
        super().__init__()
        self.seq_len = seq_len
        self.out_channels = out_channels

        # Project latent vector to a feature map that can be up-sampled
        self._base_len = seq_len // 4  # matches the encoder's down-sampling factor
        self.fc = nn.Linear(latent_dim, 128 * self._base_len)

        self.deconv_layers = nn.Sequential(
            # Block 1: up-sample x2
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Block 2: up-sample x2 -> back to original length
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Final projection to output channels
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, D]
        Returns:
            x_hat: [B, C, L]
        """
        out = self.fc(z)                                       # [B, 128 * base_len]
        out = out.view(out.size(0), 128, self._base_len)       # [B, 128, base_len]
        out = self.deconv_layers(out)                          # [B, C, L'] (approx L)
        # Ensure exact output length via adaptive interpolation
        if out.size(-1) != self.seq_len:
            out = F.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out


class FSAE(nn.Module):
    """
    Fuzzy Shapelet AutoEncoder (FSAE).

    Combines a deep CNN encoder and decoder with a Fuzzy Shapelet Matching Layer
    for interpretable multivariate time series anomaly detection.

    Training is performed in two modes, controlled by ``use_fuzzy``:
        - ``use_fuzzy=False`` (pre-training): Z is passed directly to the decoder
          (standard AutoEncoder).
        - ``use_fuzzy=True`` (fine-tuning): Z is first passed through the Fuzzy
          Shapelet Layer to produce Z_hat, which is then decoded.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        latent_dim: int = 64,
        num_shapelets: int = 10,
        alpha: float = 1.0,
    ):
        """
        Args:
            in_channels:    Number of input/output channels (multivariate dimension M).
            seq_len:        Length of each time series window L.
            latent_dim:     Dimensionality of the latent space D.
            num_shapelets:  Number of learnable shapelet prototypes K.
            alpha:          Student-t degrees of freedom for fuzzy membership.
        """
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, seq_len)
        self.fuzzy_shapelet = FuzzyShapeletLayer(num_shapelets, latent_dim, alpha)
        self.decoder = Decoder(in_channels, latent_dim, seq_len)

    def forward(
        self, x: torch.Tensor, use_fuzzy: bool = True
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Args:
            x:         [B, C, L] input time series.
            use_fuzzy: If True, route through the Fuzzy Shapelet Layer (fine-tuning /
                       inference). If False, bypass it (pre-training).
        Returns:
            x_hat:       [B, C, L] reconstructed time series.
            z:           [B, D]    latent representation from the encoder.
            z_hat:       [B, D]    fused latent (equals z when use_fuzzy=False).
            memberships: [B, K]    fuzzy membership matrix (None when use_fuzzy=False).
            distances:   [B, K]    distances to each shapelet (None when use_fuzzy=False).
        """
        z = self.encoder(x)

        if use_fuzzy:
            z_hat, memberships, distances = self.fuzzy_shapelet(z)
        else:
            z_hat = z
            memberships = None
            distances = None

        x_hat = self.decoder(z_hat)
        return x_hat, z, z_hat, memberships, distances

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: encode only, returns Z."""
        return self.encoder(x)

    def init_shapelets(self, centers: torch.Tensor) -> None:
        """
        Initialise shapelet parameters from pre-computed cluster centres
        (e.g., K-Means centres in the latent space).

        Args:
            centers: [K, D] tensor of cluster centres.
        """
        with torch.no_grad():
            self.fuzzy_shapelet.shapelets.copy_(centers)
