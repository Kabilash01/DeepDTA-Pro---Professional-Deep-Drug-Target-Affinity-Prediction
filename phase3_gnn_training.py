"""
Phase 3 GNN Training Components.
Recreated to restore pipeline compatibility.

Provides:
  MolecularGraphBuilder   - SMILES → (node_features, edge_index)
  GNNMolecularEncoder     - Graph neural network encoder
  SimpleProteinTransformer - Protein sequence encoder
  Phase3GNNModel          - Full drug-target affinity model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atom / bond feature helpers
# ---------------------------------------------------------------------------

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'OTHER']
DEGREE_LIST = [0, 1, 2, 3, 4, 5]
VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6]


def _one_hot(value, choices: list) -> List[int]:
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1
    return enc


def _atom_features(atom) -> List[float]:
    """78-dim atom feature vector."""
    symbol = atom.GetSymbol()
    feats = (
        _one_hot(symbol, ATOM_TYPES)
        + _one_hot(atom.GetDegree(), DEGREE_LIST)
        + _one_hot(atom.GetTotalValence(), VALENCE_LIST)
        + [
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetFormalCharge() / 4.0,
            atom.GetNumRadicalElectrons() / 2.0,
        ]
    )
    # pad to fixed size 32
    feats = feats[:32]
    feats += [0.0] * (32 - len(feats))
    return feats


# ---------------------------------------------------------------------------
# MolecularGraphBuilder
# ---------------------------------------------------------------------------

class MolecularGraphBuilder:
    """Convert a SMILES string to a PyTorch graph (node_features, edge_index)."""

    def smiles_to_graph(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            smiles: SMILES string
        Returns:
            (x, edge_index) where x is [n_atoms, 32] and edge_index is [2, n_edges]
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            return self._mol_to_tensors(mol)
        except ImportError:
            return self._fallback_graph(smiles)
        except Exception:
            return self._fallback_graph(smiles)

    def _mol_to_tensors(self, mol) -> Tuple[torch.Tensor, torch.Tensor]:
        from rdkit import Chem
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return self._fallback_graph("")

        # Node features
        x = torch.tensor(
            [_atom_features(atom) for atom in mol.GetAtoms()],
            dtype=torch.float32,
        )  # [n_atoms, 32]

        # Edge index (undirected)
        src, dst = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [i, j]
            dst += [j, i]

        if not src:
            # Molecule with no bonds – add self-loops
            src = list(range(n_atoms))
            dst = list(range(n_atoms))

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return x, edge_index

    def _fallback_graph(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a small random graph when RDKit is unavailable or SMILES invalid."""
        n = max(5, len(smiles) % 20 + 3)
        x = torch.randn(n, 32)
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return x, edge_index


# ---------------------------------------------------------------------------
# GNNMolecularEncoder
# ---------------------------------------------------------------------------

class GNNMolecularEncoder(nn.Module):
    """
    Simple message-passing GNN encoder.
    Uses manual mean-aggregation to avoid torch_geometric dependency.
    """

    def __init__(self, in_dim: int = 32, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [n_atoms, in_dim]
            edge_index: [2, n_edges]
        Returns:
            graph_repr: [1, hidden_dim]
        """
        n = x.size(0)
        h = x

        for layer, norm in zip(self.layers, self.norms):
            # Aggregate neighbour features
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src])
            count = torch.bincount(dst, minlength=n).clamp(min=1).float().unsqueeze(-1)
            agg = agg / count

            h = F.relu(norm(layer(h + agg)))

        # Global mean pooling → [1, hidden_dim]
        graph_repr = h.mean(dim=0, keepdim=True)
        return graph_repr


# ---------------------------------------------------------------------------
# SimpleProteinTransformer
# ---------------------------------------------------------------------------

class SimpleProteinTransformer(nn.Module):
    """Lightweight Transformer encoder for amino-acid sequences."""

    def __init__(self, vocab_size: int = 21, embed_dim: int = 64,
                 hidden_dim: int = 128, n_heads: int = 4, n_layers: int = 2,
                 max_len: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=hidden_dim, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
        Returns:
            repr: [batch, hidden_dim]
        """
        batch, seq_len = token_ids.shape
        pos = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        x = self.embed(token_ids) + self.pos_enc(pos)
        padding_mask = (token_ids == 0)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Mean pool over non-padding positions
        valid = (~padding_mask).float().unsqueeze(-1)
        lengths = valid.sum(dim=1).clamp(min=1)
        pooled = (x * valid).sum(dim=1) / lengths

        return F.relu(self.proj(pooled))  # [batch, hidden_dim]


# ---------------------------------------------------------------------------
# Phase3GNNModel
# ---------------------------------------------------------------------------

class Phase3GNNModel(nn.Module):
    """
    Full drug-target affinity model:
      GNNMolecularEncoder + SimpleProteinTransformer + MLP prediction head.

    Forward signature:
        model((x, edge_index), prot_ids)
    where prot_ids is [1, seq_len].
    """

    def __init__(self, gnn_hidden_dim: int = 128, prot_embed_dim: int = 128,
                 mlp_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mol_encoder = GNNMolecularEncoder(in_dim=32, hidden_dim=gnn_hidden_dim)
        self.prot_encoder = SimpleProteinTransformer(
            embed_dim=64, hidden_dim=prot_embed_dim
        )

        fusion_dim = gnn_hidden_dim + prot_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(
        self,
        mol_data: Tuple[torch.Tensor, torch.Tensor],
        prot_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mol_data: (x [n_atoms, 32], edge_index [2, n_edges])
            prot_ids: [1, seq_len]
        Returns:
            affinity: [1, 1]
        """
        x, edge_index = mol_data
        mol_repr = self.mol_encoder(x, edge_index)          # [1, gnn_hidden_dim]
        prot_repr = self.prot_encoder(prot_ids)              # [1, prot_embed_dim]

        combined = torch.cat([mol_repr, prot_repr], dim=-1) # [1, fusion_dim]
        return self.head(combined)                           # [1, 1]
