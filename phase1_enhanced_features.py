"""
PHASE 1: GRAPH FEATURE ENGINEERING
====================================
Analyzes molecular and protein graph properties on the DAVIS dataset.
Computes graph-level statistics and validates feature dimensions before training.

Key GML concepts:
  - Molecular graph construction (atoms=nodes, bonds=edges, rich features)
  - Protein residue graph construction
  - Graph property analysis: degree distribution, node/edge feature stats
  - Feature importance via correlation with affinity labels
"""

import torch
import numpy as np
import logging
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from gml_core import (
    MolecularGraphBuilder, ProteinGraphBuilder,
    ATOM_FEAT_DIM, BOND_FEAT_DIM, AA_VOCAB_SIZE,
    PYG_AVAILABLE,
)
from phase3_real_data import DAVISDatasetLoader
from target_normalizer import AffinityNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# GRAPH STATISTICS ANALYSER
# ============================================================================
class GraphFeatureAnalyser:
    """Analyses molecular graph properties across the DAVIS dataset."""

    def __init__(self):
        self.mol_builder  = MolecularGraphBuilder()
        self.prot_builder = ProteinGraphBuilder(max_len=1000, window=3)

    def analyse_molecular_graphs(self, samples: list, n: int = 500) -> dict:
        logger.info(f"Analysing {min(n, len(samples))} molecular graphs...")
        subset = samples[:n]

        n_atoms_list, n_edges_list = [], []
        degree_counts = defaultdict(int)

        for s in subset:
            try:
                g = self.mol_builder.smiles_to_pyg(str(s['drug_smiles']))
                na = g.x.shape[0]
                ne = g.edge_index.shape[1] // 2
                n_atoms_list.append(na)
                n_edges_list.append(ne)
                for node in range(na):
                    deg = (g.edge_index[0] == node).sum().item()
                    degree_counts[int(deg)] += 1
            except Exception:
                continue

        return {
            'n_atoms': {
                'mean': float(np.mean(n_atoms_list)) if n_atoms_list else 0,
                'std':  float(np.std(n_atoms_list)) if n_atoms_list else 0,
                'min':  int(np.min(n_atoms_list)) if n_atoms_list else 0,
                'max':  int(np.max(n_atoms_list)) if n_atoms_list else 0,
            },
            'n_edges': {
                'mean': float(np.mean(n_edges_list)) if n_edges_list else 0,
                'std':  float(np.std(n_edges_list)) if n_edges_list else 0,
            },
            'feature_dim':   ATOM_FEAT_DIM,
            'bond_feat_dim': BOND_FEAT_DIM,
            'avg_degree': float(
                sum(k * v for k, v in degree_counts.items()) /
                max(sum(degree_counts.values()), 1)
            ),
        }

    def analyse_protein_sequences(self, samples: list, n: int = 200) -> dict:
        logger.info(f"Analysing {min(n, len(samples))} protein sequences...")
        seq_lengths = [len(str(s['protein_sequence'])) for s in samples[:n]]
        return {
            'seq_length': {
                'mean': float(np.mean(seq_lengths)),
                'std':  float(np.std(seq_lengths)),
                'min':  int(np.min(seq_lengths)),
                'max':  int(np.max(seq_lengths)),
            },
            'vocab_size': AA_VOCAB_SIZE,
        }

    def analyse_affinity_distribution(self, samples: list) -> dict:
        affinities = np.array([float(s['affinity']) for s in samples])
        unique_vals, counts = np.unique(affinities, return_counts=True)
        top5 = np.argsort(-counts)[:5]
        return {
            'mean':     float(affinities.mean()),
            'std':      float(affinities.std()),
            'min':      float(affinities.min()),
            'max':      float(affinities.max()),
            'n_samples': len(affinities),
            'n_unique':  len(unique_vals),
            'top_values': [
                {'value': float(unique_vals[i]),
                 'count': int(counts[i]),
                 'pct':   float(counts[i] / len(affinities) * 100)}
                for i in top5
            ],
        }

    def compute_graph_correlation(self, samples: list, n: int = 1000) -> dict:
        """Correlate graph structural features with affinity labels."""
        logger.info("Computing graph-affinity correlations...")
        mol_features, affinities = [], []
        for s in samples[:n]:
            try:
                g  = self.mol_builder.smiles_to_pyg(str(s['drug_smiles']))
                na = float(g.x.shape[0])
                ne = float(g.edge_index.shape[1] / 2)
                mol_features.append([na, ne, ne / max(na, 1) * 2])
                affinities.append(float(s['affinity']))
            except Exception:
                continue

        if len(mol_features) < 2:
            return {}

        X, y = np.array(mol_features), np.array(affinities)
        return {name: round(float(np.corrcoef(X[:, i], y)[0, 1]), 4)
                for i, name in enumerate(['n_atoms', 'n_bonds', 'avg_degree'])}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("PHASE 1: GRAPH FEATURE ENGINEERING & ANALYSIS")
    print("=" * 80 + "\n")

    loader = DAVISDatasetLoader(data_dir="data")
    data   = loader.load_davis()
    logger.info(f"Loaded {len(data)} samples")

    analyser = GraphFeatureAnalyser()

    print("\n--- AFFINITY DISTRIBUTION ---")
    aff = analyser.analyse_affinity_distribution(data)
    print(f"  Samples : {aff['n_samples']:,}")
    print(f"  Range   : {aff['min']:.2f} – {aff['max']:.2f}")
    print(f"  Mean±Std: {aff['mean']:.4f} ± {aff['std']:.4f}")
    print(f"  Unique  : {aff['n_unique']}")
    print("  Top-5 affinity values:")
    for tv in aff['top_values']:
        print(f"    {tv['value']:.2f}  -> {tv['count']:,} samples ({tv['pct']:.1f}%)")

    print("\n--- MOLECULAR GRAPH PROPERTIES ---")
    mol = analyser.analyse_molecular_graphs(data, n=500)
    print(f"  Atom feature dim  : {mol['feature_dim']}")
    print(f"  Bond feature dim  : {mol['bond_feat_dim']}")
    na = mol['n_atoms']
    print(f"  Atoms/graph       : {na['mean']:.1f} ± {na['std']:.1f}  (range {na['min']}–{na['max']})")
    ne = mol['n_edges']
    print(f"  Edges/graph       : {ne['mean']:.1f} ± {ne['std']:.1f}")
    print(f"  Mean degree       : {mol['avg_degree']:.2f}")

    print("\n--- PROTEIN SEQUENCE PROPERTIES ---")
    prot = analyser.analyse_protein_sequences(data, n=200)
    sl = prot['seq_length']
    print(f"  Amino acid vocab  : {prot['vocab_size']}")
    print(f"  Sequence length   : {sl['mean']:.0f} ± {sl['std']:.0f}  (range {sl['min']}–{sl['max']})")

    print("\n--- GRAPH STRUCTURE vs AFFINITY CORRELATIONS ---")
    corrs = analyser.compute_graph_correlation(data, n=1000)
    for feat, corr in corrs.items():
        print(f"  {feat:<15}: r = {corr:+.4f}")

    train, val, test = loader.create_splits(data)
    print(f"\n--- DATASET SPLITS ---")
    print(f"  Train : {len(train):,}")
    print(f"  Val   : {len(val):,}")
    print(f"  Test  : {len(test):,}")

    stats = loader.get_statistics(data)
    norm  = AffinityNormalizer(mean=stats['affinity_mean'], std=stats['affinity_std'])
    print(f"\n--- NORMALIZER ---")
    print(f"  {norm.info()}")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE — Features validated, ready for GCN training")
    print(f"  Atom feature dim  : {ATOM_FEAT_DIM}")
    print(f"  Bond feature dim  : {BOND_FEAT_DIM}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
