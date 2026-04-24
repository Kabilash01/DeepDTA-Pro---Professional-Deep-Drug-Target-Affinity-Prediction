"""
Phase 3 Real Data Integration: DAVIS & KIBA Dataset Loader
Processes actual drug-target affinity data for GNN training
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DAVISDatasetLoader:
    """Load and process DAVIS drug-target binding affinity dataset"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_data = None
        self.processed_data = None

    def load_davis(self) -> List[Dict]:
        """
        Load DAVIS dataset from CSV

        Expected columns:
        - drug_name, drug_smiles
        - target_name, target_sequence
        - binding_affinity (pKd or pIC50)
        """
        davis_path = self.data_dir / "davis_all.csv"

        if not davis_path.exists():
            logger.warning(f"DAVIS dataset not found at {davis_path}")
            return []

        try:
            df = pd.read_csv(davis_path)
            logger.info(f"✅ Loaded DAVIS dataset: {len(df)} samples")
            logger.info(f"   Columns: {df.columns.tolist()}")
            logger.info(f"   Shape: {df.shape}")

            # Map actual column names to standard names
            actual_cols = df.columns.tolist()
            col_map = {}

            # Handle SMILES column
            smiles_cols = ['compound_iso_smiles', 'smiles', 'SMILES', 'Smiles']
            for col in smiles_cols:
                if col in actual_cols:
                    col_map['drug_smiles'] = col
                    break

            # Handle sequence column
            seq_cols = ['target_sequence', 'sequence', 'Sequence', 'protein_sequence']
            for col in seq_cols:
                if col in actual_cols:
                    col_map['protein_sequence'] = col
                    break

            # Handle affinity column
            aff_cols = ['affinity', 'binding_affinity', 'pKd', 'pIC50', 'pKi']
            for col in aff_cols:
                if col in actual_cols:
                    col_map['affinity'] = col
                    break

            logger.info(f"   Column mapping: {col_map}")

            # Extract valid samples
            valid_samples = []
            for idx, row in df.iterrows():
                try:
                    sample = {
                        'drug_smiles': str(row[col_map.get('drug_smiles', '')]).strip() if col_map.get('drug_smiles') else '',
                        'protein_sequence': str(row[col_map.get('protein_sequence', '')]).strip() if col_map.get('protein_sequence') else '',
                        'affinity': float(row[col_map.get('affinity', 0)]) if col_map.get('affinity') else 0.0
                    }

                    # Basic validation
                    if (sample['drug_smiles'] and
                        sample['protein_sequence'] and
                        len(sample['drug_smiles']) > 2 and
                        len(sample['protein_sequence']) > 3):
                        valid_samples.append(sample)

                except (ValueError, TypeError, KeyError):
                    continue

            logger.info(f"   Valid samples: {len(valid_samples)}/{len(df)}")
            return valid_samples

        except Exception as e:
            logger.error(f"Error loading DAVIS: {e}")
            return []

    def load_kiba(self) -> List[Dict]:
        """
        Load KIBA drug-target binding affinity dataset from CSV

        Similar structure to DAVIS
        """
        kiba_path = self.data_dir / "kiba_all.csv"

        if not kiba_path.exists():
            logger.warning(f"KIBA dataset not found at {kiba_path}")
            return []

        try:
            df = pd.read_csv(kiba_path)
            logger.info(f"✅ Loaded KIBA dataset: {len(df)} samples")
            logger.info(f"   Columns: {df.columns.tolist()}")

            # Map actual column names to standard names
            actual_cols = df.columns.tolist()
            col_map = {}

            # Handle SMILES column
            smiles_cols = ['compound_iso_smiles', 'smiles', 'SMILES', 'Smiles']
            for col in smiles_cols:
                if col in actual_cols:
                    col_map['drug_smiles'] = col
                    break

            # Handle sequence column
            seq_cols = ['target_sequence', 'sequence', 'Sequence', 'protein_sequence']
            for col in seq_cols:
                if col in actual_cols:
                    col_map['protein_sequence'] = col
                    break

            # Handle affinity column
            aff_cols = ['affinity', 'binding_affinity', 'pKd', 'pIC50', 'pKi']
            for col in aff_cols:
                if col in actual_cols:
                    col_map['affinity'] = col
                    break

            valid_samples = []
            for idx, row in df.iterrows():
                try:
                    sample = {
                        'drug_smiles': str(row[col_map.get('drug_smiles', '')]).strip() if col_map.get('drug_smiles') else '',
                        'protein_sequence': str(row[col_map.get('protein_sequence', '')]).strip() if col_map.get('protein_sequence') else '',
                        'affinity': float(row[col_map.get('affinity', 0)]) if col_map.get('affinity') else 0.0
                    }

                    if (sample['drug_smiles'] and
                        sample['protein_sequence'] and
                        len(sample['drug_smiles']) > 2 and
                        len(sample['protein_sequence']) > 3):
                        valid_samples.append(sample)

                except (ValueError, TypeError, KeyError):
                    continue

            logger.info(f"   Valid samples: {len(valid_samples)}/{len(df)}")
            return valid_samples

        except Exception as e:
            logger.error(f"Error loading KIBA: {e}")
            return []

    def create_splits(self, data: List[Dict], train_ratio: float = 0.7,
                     val_ratio: float = 0.15, seed: int = 42) -> Tuple[List, List, List]:
        """Create train/val/test splits"""
        np.random.seed(seed)
        data = data.copy()
        np.random.shuffle(data)

        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        logger.info(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data

    def get_statistics(self, data: List[Dict]) -> Dict:
        """Get dataset statistics"""
        if not data:
            return {}

        affinities = [d['affinity'] for d in data]
        smiles_lens = [len(d['drug_smiles']) for d in data]
        seq_lens = [len(d['protein_sequence']) for d in data]

        return {
            'num_samples': len(data),
            'affinity_mean': np.mean(affinities),
            'affinity_std': np.std(affinities),
            'affinity_min': np.min(affinities),
            'affinity_max': np.max(affinities),
            'smiles_len_mean': np.mean(smiles_lens),
            'smiles_len_max': np.max(smiles_lens),
            'seq_len_mean': np.mean(seq_lens),
            'seq_len_max': np.max(seq_lens),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_real_dataset(source: str = 'davis') -> Tuple[List, List, List]:
    """
    Convenience function to load and split real dataset

    Args:
        source: 'davis' or 'kiba'

    Returns:
        (train_data, val_data, test_data)
    """
    loader = DAVISDatasetLoader(data_dir="data")

    if source.lower() == 'davis':
        data = loader.load_davis()
    elif source.lower() == 'kiba':
        data = loader.load_kiba()
    else:
        logger.error(f"Unknown source: {source}")
        return [], [], []

    if not data:
        logger.error("Failed to load dataset")
        return [], [], []

    stats = loader.get_statistics(data)
    logger.info(f"Dataset Statistics: {stats}")

    return loader.create_splits(data)


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REAL DATASET LOADER - DAVIS & KIBA")
    print("=" * 70 + "\n")

    loader = DAVISDatasetLoader(data_dir="data")

    # Try loading DAVIS
    print("Loading DAVIS dataset...")
    davis_data = loader.load_davis()
    if davis_data:
        print(f"✅ DAVIS loaded: {len(davis_data)} samples")
        stats = loader.get_statistics(davis_data)
        print(f"   Affinity range: {stats['affinity_min']:.2f} - {stats['affinity_max']:.2f}")
        print(f"   Mean: {stats['affinity_mean']:.2f} ± {stats['affinity_std']:.2f}")
        print(f"   Sample 1: {davis_data[0]}")

        # Create splits
        train, val, test = loader.create_splits(davis_data)
        print(f"\n Splits created: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Try loading KIBA
    print("\nLoading KIBA dataset...")
    kiba_data = loader.load_kiba()
    if kiba_data:
        print(f"✅ KIBA loaded: {len(kiba_data)} samples")
        stats = loader.get_statistics(kiba_data)
        print(f"   Affinity range: {stats['affinity_min']:.2f} - {stats['affinity_max']:.2f}")
        print(f"   Mean: {stats['affinity_mean']:.2f} ± {stats['affinity_std']:.2f}")

    print("\n" + "=" * 70)
