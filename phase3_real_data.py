"""
Phase 3 Real Data Loader - DAVIS dataset loader for pipeline phases 3-7.
Loads from data/davis_all.csv (30,056 real drug-target pairs).
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)


class DAVISDatasetLoader:
    """Dataset loader for DAVIS binding affinity data used across pipeline phases."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_davis(self) -> List[Dict]:
        """
        Load real DAVIS dataset from data/davis_all.csv (30,056 samples).
        Falls back to synthetic data only if the file is missing.

        Returns:
            List of dicts with keys: drug_smiles, protein_sequence, affinity
        """
        samples = self._load_real_csv()
        if samples:
            logger.info(f"Loaded {len(samples)} real DAVIS samples")
            return samples

        logger.warning("data/davis_all.csv not found – generating synthetic fallback.")
        return self._generate_synthetic(n_samples=2000)

    def _load_real_csv(self) -> List[Dict]:
        """Load data/davis_all.csv with columns: compound_iso_smiles, target_sequence, affinity."""
        import pandas as pd

        # Primary location
        candidates = [
            self.data_dir / "davis_all.csv",
            self.data_dir / "davis" / "drug_protein_affinity.csv",
            self.data_dir / "davis.csv",
        ]

        # Column name aliases (different files use different names)
        smiles_cols  = ["compound_iso_smiles", "drug_smiles", "smiles", "SMILES"]
        protein_cols = ["target_sequence", "protein_sequence", "sequence"]
        affinity_cols = ["affinity", "label", "pKd", "score"]

        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                smiles_col   = next((c for c in smiles_cols  if c in df.columns), None)
                protein_col  = next((c for c in protein_cols if c in df.columns), None)
                affinity_col = next((c for c in affinity_cols if c in df.columns), None)

                if not all([smiles_col, protein_col, affinity_col]):
                    logger.warning(f"{path}: cannot find required columns. Found: {df.columns.tolist()}")
                    continue

                df = df[[smiles_col, protein_col, affinity_col]].dropna()
                df.columns = ["drug_smiles", "protein_sequence", "affinity"]
                df["affinity"] = df["affinity"].astype(float)

                records = df.to_dict("records")
                logger.info(f"Read {len(records)} rows from {path}")
                return records

            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")

        return []

    def _generate_synthetic(self, n_samples: int = 2000) -> List[Dict]:
        """Fallback: generate synthetic data when real files are absent."""
        _smiles = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC1=CC=C(C=C1)C(=O)O",
            "C1=CC=C(C=C1)O",
            "CC(=O)Nc1ccc(O)cc1",
        ]
        _proteins = [
            "MKKFFDSRREQGGSGLGSGSSGGGGSGGGYGNQDQSGGG",
            "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIED",
            "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMV",
        ]
        np.random.seed(42)
        return [
            {
                "drug_smiles": _smiles[i % len(_smiles)],
                "protein_sequence": _proteins[i % len(_proteins)],
                "affinity": float(np.clip(np.random.normal(6.5, 1.5), 2.0, 12.0)),
            }
            for i in range(n_samples)
        ]

    def create_splits(
        self,
        data: List[Dict],
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train / val / test sets."""
        rng = np.random.RandomState(seed)
        idx = np.arange(len(data))
        rng.shuffle(idx)

        n_train = int(len(data) * train_frac)
        n_val = int(len(data) * val_frac)

        train = [data[i] for i in idx[:n_train]]
        val = [data[i] for i in idx[n_train: n_train + n_val]]
        test = [data[i] for i in idx[n_train + n_val:]]

        logger.info(f"Splits – train: {len(train)}, val: {len(val)}, test: {len(test)}")
        return train, val, test

    def get_statistics(self, data: List[Dict]) -> Dict[str, float]:
        """Return basic affinity statistics for the dataset."""
        affinities = np.array([s["affinity"] for s in data], dtype=np.float32)
        return {
            "affinity_mean": float(affinities.mean()),
            "affinity_std": float(affinities.std()) if affinities.std() > 0 else 1.0,
            "affinity_min": float(affinities.min()),
            "affinity_max": float(affinities.max()),
            "n_samples": len(data),
        }


class KIBADatasetLoader(DAVISDatasetLoader):
    """
    KIBA dataset loader. Inherits DAVIS behavior but reads kiba_all.csv.
    KIBA has ~118k samples — 4x bigger than DAVIS, used for pretraining.
    """

    def load_kiba(self) -> List[Dict]:
        """Load real KIBA dataset from data/kiba_all.csv (~118k samples)."""
        samples = self._load_kiba_csv()
        if samples:
            logger.info(f"Loaded {len(samples)} real KIBA samples")
            return samples
        logger.warning("data/kiba_all.csv not found - synthetic fallback")
        return self._generate_synthetic(n_samples=4000)

    def _load_kiba_csv(self) -> List[Dict]:
        import pandas as pd

        candidates = [
            self.data_dir / "kiba_all.csv",
            self.data_dir / "kiba" / "drug_protein_affinity.csv",
            self.data_dir / "kiba.csv",
        ]
        smiles_cols   = ["compound_iso_smiles", "drug_smiles", "smiles", "SMILES"]
        protein_cols  = ["target_sequence", "protein_sequence", "sequence"]
        affinity_cols = ["affinity", "label", "score", "kiba_score"]

        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                smiles_col   = next((c for c in smiles_cols   if c in df.columns), None)
                protein_col  = next((c for c in protein_cols  if c in df.columns), None)
                affinity_col = next((c for c in affinity_cols if c in df.columns), None)
                if not all([smiles_col, protein_col, affinity_col]):
                    logger.warning(f"{path}: missing columns. Found: {df.columns.tolist()}")
                    continue
                df = df[[smiles_col, protein_col, affinity_col]].dropna()
                df.columns = ["drug_smiles", "protein_sequence", "affinity"]
                df["affinity"] = df["affinity"].astype(float)
                records = df.to_dict("records")
                logger.info(f"Read {len(records)} rows from {path}")
                return records
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
        return []


def load_real_dataset(
    source: str = "davis",
    data_dir: str = "data",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Convenience function used by phase2_advanced_training.

    Args:
        source: 'davis' or 'kiba'

    Returns:
        (train_data, val_data, test_data)
    """
    if source.lower() == "kiba":
        loader = KIBADatasetLoader(data_dir=data_dir)
        data = loader.load_kiba()
    else:
        loader = DAVISDatasetLoader(data_dir=data_dir)
        data = loader.load_davis()
    return loader.create_splits(data, train_frac=train_frac, val_frac=val_frac)
