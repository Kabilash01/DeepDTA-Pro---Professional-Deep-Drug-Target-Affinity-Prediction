"""
Affinity Normalizer - normalizes/denormalizes binding affinity values.
Recreated to restore pipeline compatibility.
"""

import numpy as np


class AffinityNormalizer:
    """Normalizes binding affinity values using Z-score normalization."""

    def __init__(self, mean: float = 6.0, std: float = 1.5):
        self.mean = mean
        self.std = std if std > 0 else 1.0

    def normalize(self, value: float) -> float:
        return (value - self.mean) / self.std

    def denormalize(self, value: float) -> float:
        return value * self.std + self.mean

    def normalize_array(self, values) -> np.ndarray:
        arr = np.array(values, dtype=np.float32)
        return (arr - self.mean) / self.std

    def denormalize_array(self, values) -> np.ndarray:
        arr = np.array(values, dtype=np.float32)
        return arr * self.std + self.mean

    def info(self) -> str:
        return f"AffinityNormalizer(mean={self.mean:.4f}, std={self.std:.4f})"

    def __repr__(self):
        return self.info()
