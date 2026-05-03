"""
Target Normalization Utility for DAVIS Affinity Data
Ensures consistent normalization across all phases
"""

import numpy as np
from typing import List, Dict, Tuple


class AffinityNormalizer:
    """Normalize affinity values for stable training"""

    def __init__(self, mean: float = None, std: float = None):
        """
        Initialize with known statistics or compute from data.
        Default: DAVIS dataset statistics
        """
        self.mean = mean if mean is not None else 5.4515
        self.std = std if std is not None else 0.8947

    @staticmethod
    def compute_from_data(data: List[Dict]) -> Tuple[float, float]:
        """Calculate mean/std from dataset"""
        affinities = [d['affinity'] for d in data]
        return float(np.mean(affinities)), float(np.std(affinities))

    def normalize(self, affinity: float) -> float:
        """Normalize single value to ~[-1, 1]"""
        return (affinity - self.mean) / (self.std + 1e-8)

    def denormalize(self, normalized: float) -> float:
        """Denormalize back to original scale"""
        return normalized * self.std + self.mean

    def normalize_batch(self, affinities: np.ndarray) -> np.ndarray:
        """Normalize batch of values"""
        return (affinities - self.mean) / (self.std + 1e-8)

    def denormalize_batch(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize batch"""
        return normalized * self.std + self.mean

    def get_scale_factor(self) -> float:
        """Return scaling factor for MSE/loss calibration"""
        return self.std ** 2

    def info(self) -> str:
        """Print normalizer info"""
        return f"AffinityNormalizer(mean={self.mean:.4f}, std={self.std:.4f})"
