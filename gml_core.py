"""
GML CORE: Full Graph Machine Learning Building Blocks
=======================================================
Provides all shared GML components used across all 7 pipeline phases:

  MolecularGraphBuilder     - SMILES → PyG Data (rich atom/bond features)
  ProteinGraphBuilder       - Protein sequence → residue graph
  GCNEncoder                - Graph Convolutional Network encoder
  GATEncoder                - Graph Attention Network encoder
  GINEncoder                - Graph Isomorphism Network encoder
  GlobalGraphPooling        - Mean + Max + Attention pooling
  CrossGraphAttention       - Drug-Protein cross-attention fusion
  DTAPredictor              - Full GNN-based DTA predictor
  GraphDataset              - PyTorch Dataset wrapper for DTA samples
  build_dataloader          - Build PyG DataLoader from sample list
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyG imports (required)
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, GINConv, GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    raise ImportError(
        "PyTorch Geometric is required. Install with:\n"
        "  pip install torch_geometric"
    )

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available — using fallback atom features")

# ---------------------------------------------------------------------------
# Atom & bond feature constants
# ---------------------------------------------------------------------------
ATOM_SYMBOLS   = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                   'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                   'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                   'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                   'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'OTHER']  # 44 symbols
DEGREE_LIST    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VALENCE_LIST   = [0, 1, 2, 3, 4, 5, 6]
HYBRID_LIST    = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER']
BOND_TYPES     = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'OTHER']  # 5 types

ATOM_FEAT_DIM  = len(ATOM_SYMBOLS) + len(DEGREE_LIST) + len(VALENCE_LIST) + len(HYBRID_LIST) + 5
# = 44 + 11 + 7 + 6 + 5 = 73 atom features
BOND_FEAT_DIM  = len(BOND_TYPES) + 3  # bond_type + is_conjugated + is_ring + stereo = 8

AA_VOCAB = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8,
    'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
    'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'X': 0
}
AA_VOCAB_SIZE = 21


# ---------------------------------------------------------------------------
# One-hot helper
# ---------------------------------------------------------------------------
def _one_hot(value, choices: list) -> List[float]:
    enc = [0.0] * (len(choices) + 1)
    try:
        enc[choices.index(value)] = 1.0
    except ValueError:
        enc[-1] = 1.0
    return enc


# ---------------------------------------------------------------------------
# Atom features (73-dim)
# ---------------------------------------------------------------------------
def atom_features(atom) -> List[float]:
    from rdkit.Chem import rdchem
    hybrid_map = {
        rdchem.HybridizationType.SP:    'SP',
        rdchem.HybridizationType.SP2:   'SP2',
        rdchem.HybridizationType.SP3:   'SP3',
        rdchem.HybridizationType.SP3D:  'SP3D',
        rdchem.HybridizationType.SP3D2: 'SP3D2',
    }
    h = hybrid_map.get(atom.GetHybridization(), 'OTHER')
    feats = (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)          # 44+1=45 → but capped at 44+1
        + _one_hot(atom.GetDegree(), DEGREE_LIST)          # 12
        + _one_hot(atom.GetTotalValence(), VALENCE_LIST)   # 8
        + _one_hot(h, HYBRID_LIST)                         # 7
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetNumImplicitHs()) / 4.0,
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons()),
        ]
    )
    # Ensure fixed length = ATOM_FEAT_DIM
    feats = feats[:ATOM_FEAT_DIM]
    feats += [0.0] * (ATOM_FEAT_DIM - len(feats))
    return feats


# ---------------------------------------------------------------------------
# Bond features (8-dim)
# ---------------------------------------------------------------------------
def bond_features(bond) -> List[float]:
    from rdkit.Chem import rdchem
    btype_map = {
        rdchem.BondType.SINGLE:   'SINGLE',
        rdchem.BondType.DOUBLE:   'DOUBLE',
        rdchem.BondType.TRIPLE:   'TRIPLE',
        rdchem.BondType.AROMATIC: 'AROMATIC',
    }
    bt = btype_map.get(bond.GetBondType(), 'OTHER')
    stereo = int(bond.GetStereo()) / 6.0  # normalize stereo enum
    feats = (
        _one_hot(bt, BOND_TYPES)                       # 6
        + [
            float(bond.GetIsConjugated()),             # 1
            float(bond.IsInRing()),                    # 1
            stereo,                                    # 1
        ]
    )
    feats = feats[:BOND_FEAT_DIM]
    feats += [0.0] * (BOND_FEAT_DIM - len(feats))
    return feats


# ============================================================================
# MOLECULAR GRAPH BUILDER
# ============================================================================
class MolecularGraphBuilder:
    """Convert SMILES string to a PyG Data object with rich atom/bond features."""

    def smiles_to_pyg(self, smiles: str) -> Data:
        """
        Build a PyG Data graph from a SMILES string.

        Returns:
            Data with:
              .x          [n_atoms, ATOM_FEAT_DIM]
              .edge_index [2, 2*n_bonds]
              .edge_attr  [2*n_bonds, BOND_FEAT_DIM]
        """
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return self._mol_to_data(mol)
            except Exception:
                pass
        return self._fallback_data(smiles)

    def _mol_to_data(self, mol) -> Data:
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return self._fallback_data("")

        x = torch.tensor(
            [atom_features(atom) for atom in mol.GetAtoms()],
            dtype=torch.float32,
        )  # [n_atoms, ATOM_FEAT_DIM]

        src_list, dst_list, edge_feats = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            src_list += [i, j]
            dst_list += [j, i]
            edge_feats += [bf, bf]

        if not src_list:
            src_list = list(range(n_atoms))
            dst_list = list(range(n_atoms))
            edge_feats = [[0.0] * BOND_FEAT_DIM] * n_atoms

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _fallback_data(self, smiles: str) -> Data:
        n = max(5, (len(smiles) % 18) + 4)
        x = torch.randn(n, ATOM_FEAT_DIM)
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.zeros(len(src), BOND_FEAT_DIM)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================================
# PROTEIN GRAPH BUILDER (residue-level contact graph)
# ============================================================================
class ProteinGraphBuilder:
    """
    Build a residue-level graph from a protein amino acid sequence.
    Nodes = residues (AA embedding), edges = sequential + co-occurrence contacts.
    """

    def __init__(self, max_len: int = 1000, window: int = 3):
        self.max_len = max_len
        self.window  = window  # residues within this window are connected

    def sequence_to_pyg(self, sequence: str) -> Data:
        seq = sequence.upper()[:self.max_len]
        n = len(seq)
        if n == 0:
            n = 4
            seq = 'GGGG'

        # Node features: one-hot AA + positional
        ids = [AA_VOCAB.get(aa, 0) for aa in seq]
        x   = F.one_hot(torch.tensor(ids, dtype=torch.long),
                        num_classes=AA_VOCAB_SIZE).float()  # [n, 21]
        # Add positional encoding
        pos = torch.arange(n).float().unsqueeze(1) / max(n - 1, 1)  # [n, 1]
        x   = torch.cat([x, pos], dim=1)  # [n, 22]

        # Edges: sliding window contacts
        src_list, dst_list = [], []
        for i in range(n):
            for j in range(i + 1, min(i + self.window + 1, n)):
                src_list += [i, j]
                dst_list += [j, i]

        if not src_list:
            src_list = list(range(n))
            dst_list = list(range(n))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)


# ============================================================================
# GCN ENCODER
# ============================================================================
class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder.
    Stacks GCNConv layers with BatchNorm + ReLU + skip connections.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout  = nn.Dropout(dropout)
        self.out_dim  = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(norm(conv(h, edge_index)))
            h_new = self.dropout(h_new)
            h = h + h_new  # residual
        return global_pool(h, batch)  # [B, 2*hidden_dim]


# ============================================================================
# GAT ENCODER
# ============================================================================
class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.
    Multi-head attention with skip connections.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 n_layers: int = 4, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        head_dim = hidden_dim // n_heads
        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATConv(hidden_dim, head_dim, heads=n_heads,
                        dropout=dropout, concat=True)
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(norm(conv(h, edge_index)))
            h_new = self.dropout(h_new)
            h = h + h_new
        return global_pool(h, batch)  # [B, 2*hidden_dim]


# ============================================================================
# GIN ENCODER
# ============================================================================
class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network encoder.
    Theoretically the most expressive of GCN/GAT/GIN.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(norm(conv(h, edge_index)))
            h_new = self.dropout(h_new)
            h = h + h_new
        return global_pool(h, batch)  # [B, 2*hidden_dim]


# ============================================================================
# GLOBAL GRAPH POOLING (mean + max)
# ============================================================================
def global_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Concatenate mean and max global pooling. Returns [B, 2*D]."""
    g_mean = global_mean_pool(h, batch)  # [B, D]
    g_max  = global_max_pool(h, batch)   # [B, D]
    return torch.cat([g_mean, g_max], dim=-1)  # [B, 2D]


class GlobalGraphPooling(nn.Module):
    """Learnable pooling: mean + max concatenated → linear projection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, out_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(global_pool(h, batch)))


# ============================================================================
# PROTEIN TRANSFORMER ENCODER (sequence-level)
# ============================================================================
class ProteinTransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder for protein sequences.
    Input: token IDs [B, L]. Output: [B, out_dim].
    """

    def __init__(self, vocab_size: int = AA_VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 4,
                 max_len: int = 1200, dropout: float = 0.1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = nn.Embedding(max_len, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.proj    = nn.Linear(embed_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        pos  = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x    = self.input_norm(self.embed(token_ids) + self.pos_enc(pos))
        mask = (token_ids == 0)  # padding mask

        x = self.transformer(x, src_key_padding_mask=mask)

        valid   = (~mask).float().unsqueeze(-1)
        lengths = valid.sum(dim=1).clamp(min=1.0)
        pooled  = (x * valid).sum(dim=1) / lengths
        return F.relu(self.proj(pooled))  # [B, hidden_dim]


# ============================================================================
# CROSS-GRAPH ATTENTION FUSION
# ============================================================================
class CrossGraphAttention(nn.Module):
    """
    Cross-attention between drug graph representation and protein representation.
    Drug queries attend over protein keys/values and vice versa.
    """

    def __init__(self, drug_dim: int, prot_dim: int,
                 hidden_dim: int = 256, n_heads: int = 8):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)

        self.drug2prot = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=0.1
        )
        self.prot2drug = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=0.1
        )
        self.norm_drug = nn.LayerNorm(hidden_dim)
        self.norm_prot = nn.LayerNorm(hidden_dim)
        self.out_dim   = hidden_dim * 2

    def forward(self, drug_repr: torch.Tensor,
                prot_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_repr: [B, drug_dim]
            prot_repr: [B, prot_dim]
        Returns:
            fused: [B, hidden_dim*2]
        """
        d = self.drug_proj(drug_repr).unsqueeze(1)  # [B, 1, H]
        p = self.prot_proj(prot_repr).unsqueeze(1)  # [B, 1, H]

        d_attn, _ = self.drug2prot(d, p, p)         # drug queries over prot
        p_attn, _ = self.prot2drug(p, d, d)         # prot queries over drug

        d_out = self.norm_drug((d + d_attn).squeeze(1))  # [B, H]
        p_out = self.norm_prot((p + p_attn).squeeze(1))  # [B, H]

        return torch.cat([d_out, p_out], dim=-1)          # [B, 2H]


# ============================================================================
# 1D CNN BOND ENCODER
# ============================================================================
class BondCNNEncoder(nn.Module):
    """
    1D CNN over bond features to capture local bond-pattern context before GNN.
    Reads edge_attr [E, BOND_FEAT_DIM], groups edges per molecule via batch-aware
    scatter, returns a per-atom bond-context vector that is concatenated with atom features.

    Implementation: we project bond features then scatter-mean them onto their
    source atom — each atom aggregates its own bond contexts before GNN layers.
    Output: per-atom delta tensor [N_atoms, out_dim] added to atom input.
    """

    def __init__(self, bond_feat_dim: int = BOND_FEAT_DIM, out_dim: int = 32,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_d = bond_feat_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = out_dim
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, atom_feat_dim]
            edge_index: [2, E]
            edge_attr:  [E, BOND_FEAT_DIM]
        Returns:
            atom_bond_ctx: [N, out_dim]  (scatter-mean of bond MLP outputs onto src atom)
        """
        n_atoms = x.size(0)
        if edge_attr is None or edge_attr.size(0) == 0:
            return x.new_zeros(n_atoms, self.out_dim)

        bond_ctx = self.mlp(edge_attr.float()).float()   # [E, out_dim] — force float32 output
        src = edge_index[0]

        # scatter mean: for each atom, average its outgoing bond contexts
        out   = torch.zeros(n_atoms, self.out_dim, dtype=torch.float32, device=x.device)
        count = torch.zeros(n_atoms, 1,             dtype=torch.float32, device=x.device)
        bond_ctx = bond_ctx.to(out.dtype)  # Ensure same dtype for scatter_add_
        out.scatter_add_(0, src.unsqueeze(1).expand_as(bond_ctx), bond_ctx)
        count.scatter_add_(0, src.unsqueeze(1), torch.ones(src.size(0), 1, dtype=torch.float32, device=x.device))
        count.clamp_(min=1.0)
        return out / count


# ============================================================================
# HYBRID CNN-TRANSFORMER PROTEIN ENCODER
# ============================================================================
class HybridProteinEncoder(nn.Module):
    """
    Hybrid CNN-Transformer encoder for protein sequences.
    CNN captures local motifs (α-helices, β-sheets, active-site patterns).
    Transformer handles long-range dependencies.
    Input: token IDs [B, L]. Output: [B, hidden_dim].
    """

    def __init__(self, vocab_size: int = AA_VOCAB_SIZE, embed_dim: int = 128,
                 hidden_dim: int = 192, n_heads: int = 8, n_transformer_layers: int = 2,
                 max_len: int = 1200, dropout: float = 0.1,
                 cnn_channels: int = 128, cnn_kernels: tuple = (3, 7, 11)):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = nn.Embedding(max_len, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Multi-scale 1D CNN branch: captures local motifs at 3 scales
        self.cnn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, cnn_channels, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for k in cnn_kernels
        ])
        cnn_out_dim = cnn_channels * len(cnn_kernels)  # 384
        self.cnn_proj = nn.Linear(cnn_out_dim, embed_dim)
        self.cnn_norm = nn.LayerNorm(embed_dim)

        # Transformer branch: long-range dependencies
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_transformer_layers, enable_nested_tensor=False
        )

        # Merge CNN + Transformer
        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.proj    = nn.Linear(embed_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        pos  = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        emb  = self.input_norm(self.embed(token_ids) + self.pos_enc(pos))  # [B, L, E]
        mask = (token_ids == 0)

        # CNN branch: [B, E, L] → multi-scale → [B, L, cnn_out]
        emb_t = emb.transpose(1, 2)  # [B, E, L]
        cnn_outs = []
        for branch in self.cnn_branches:
            c = branch(emb_t)    # [B, cnn_channels, L'] - same length due to padding
            # trim/pad to L if needed
            if c.size(2) != L:
                c = c[:, :, :L]
            cnn_outs.append(c)
        cnn_cat = torch.cat(cnn_outs, dim=1)     # [B, cnn_out_dim, L]
        cnn_seq = self.cnn_norm(self.cnn_proj(cnn_cat.transpose(1, 2)))  # [B, L, E]

        # Transformer branch
        tf_seq = self.transformer(emb, src_key_padding_mask=mask)  # [B, L, E]

        # Merge
        merged = self.merge(torch.cat([cnn_seq, tf_seq], dim=-1))  # [B, L, E]

        # Masked mean pooling
        valid   = (~mask).float().unsqueeze(-1)
        lengths = valid.sum(dim=1).clamp(min=1.0)
        pooled  = (merged * valid).sum(dim=1) / lengths              # [B, E]

        return F.relu(self.proj(pooled))  # [B, hidden_dim]


# ============================================================================
# GATED BILINEAR FUSION
# ============================================================================
class GatedBilinearFusion(nn.Module):
    """
    Bilinear + Cross-Attention + Gated Ensemble fusion.

    Three parallel views of the drug-protein interaction:
      1. Bilinear: captures pairwise feature interactions z_D^T W z_P
      2. Cross-attention: asymmetric drug-query / protein-query attention
      3. Element-wise: Hadamard product after projection to common dim

    A learnable gate combines all three views, then a shape-complementarity
    kernel refines the joint embedding.
    """

    def __init__(self, drug_dim: int, prot_dim: int,
                 hidden_dim: int = 192, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)

        # View 1: Bilinear
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        # View 2: Cross-attention (same as before)
        self.drug2prot = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        self.prot2drug = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm_d = nn.LayerNorm(hidden_dim)
        self.norm_p = nn.LayerNorm(hidden_dim)

        # View 3: Hadamard (element-wise product)
        self.hadamard_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating: 3 views each of dim hidden_dim → softmax gate over 3
        self.gate_fc = nn.Linear(hidden_dim * 3, 3)

        # Shape Complementarity Kernel refinement
        self.sc_kernel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sc_norm = nn.LayerNorm(hidden_dim)

        # Final projection to output
        self.out_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_norm = nn.LayerNorm(hidden_dim * 2)
        self.out_dim  = hidden_dim * 2

        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_repr: torch.Tensor,
                prot_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_repr: [B, drug_dim]
            prot_repr: [B, prot_dim]
        Returns:
            fused: [B, hidden_dim*2]
        """
        d = F.relu(self.drug_proj(drug_repr))   # [B, H]
        p = F.relu(self.prot_proj(prot_repr))   # [B, H]

        # View 1: Bilinear
        v1 = F.relu(self.bilinear(d, p))        # [B, H]

        # View 2: Cross-attention
        d_q = d.unsqueeze(1)                    # [B, 1, H]
        p_q = p.unsqueeze(1)                    # [B, 1, H]
        d_attn, _ = self.drug2prot(d_q, p_q, p_q)
        p_attn, _ = self.prot2drug(p_q, d_q, d_q)
        d_ca = self.norm_d((d_q + d_attn).squeeze(1))  # [B, H]
        p_ca = self.norm_p((p_q + p_attn).squeeze(1))  # [B, H]
        v2 = d_ca + p_ca                               # [B, H]

        # View 3: Hadamard
        v3 = F.relu(self.hadamard_proj(d * p))  # [B, H]

        # Gated combination
        concat3 = torch.cat([v1, v2, v3], dim=-1)          # [B, 3H]
        gate = torch.softmax(self.gate_fc(concat3), dim=-1) # [B, 3]
        fused = gate[:, 0:1] * v1 + gate[:, 1:2] * v2 + gate[:, 2:3] * v3  # [B, H]

        # Shape complementarity kernel refinement
        sc = self.sc_kernel(fused)                  # [B, H]
        fused = self.sc_norm(fused + sc)             # residual + LN

        # Project to out_dim
        out = F.relu(self.out_norm(self.out_proj(fused)))  # [B, 2H]
        return self.dropout(out)


# ============================================================================
# ENHANCED DTA PREDICTOR (with bond CNN + hybrid protein + gated fusion)
# ============================================================================
class EnhancedDTAPredictor(nn.Module):
    """
    Enhanced DTA predictor using:
      - Bond CNN augmented GNN (GIN with bond-context injection)
      - Hybrid CNN-Transformer protein encoder
      - Gated Bilinear fusion with shape complementarity kernel
    """

    def __init__(
        self,
        gnn_type: str      = 'gin',
        mol_in_dim: int    = ATOM_FEAT_DIM,
        gnn_hidden: int    = 192,
        gnn_layers: int    = 4,
        prot_embed: int    = 128,
        prot_hidden: int   = 192,
        prot_layers: int   = 4,
        fusion_hidden: int = 192,
        dropout: float     = 0.1,
        bond_cnn_dim: int  = 32,
    ):
        super().__init__()
        self.gnn_type = gnn_type

        # Bond CNN pre-encoder: adds bond context to atom features
        self.bond_cnn = BondCNNEncoder(BOND_FEAT_DIM, bond_cnn_dim, dropout=dropout)
        aug_mol_dim   = mol_in_dim + bond_cnn_dim  # 73 + 32 = 105

        # Molecular GNN encoder with augmented atom features
        if gnn_type == 'gcn':
            self.mol_encoder = GCNEncoder(aug_mol_dim, gnn_hidden, gnn_layers, dropout)
        elif gnn_type == 'gat':
            self.mol_encoder = GATEncoder(aug_mol_dim, gnn_hidden, gnn_layers, n_heads=8, dropout=dropout)
        elif gnn_type == 'gin':
            self.mol_encoder = GINEncoder(aug_mol_dim, gnn_hidden, gnn_layers, dropout)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.mol_pool = nn.Linear(gnn_hidden * 2, gnn_hidden)

        # Hybrid protein encoder
        self.prot_encoder = HybridProteinEncoder(
            embed_dim=prot_embed, hidden_dim=prot_hidden,
            n_transformer_layers=max(2, prot_layers - 2),
            dropout=dropout,
        )

        # Gated bilinear fusion with shape complementarity
        self.fusion = GatedBilinearFusion(
            drug_dim=gnn_hidden, prot_dim=prot_hidden,
            hidden_dim=fusion_hidden, n_heads=8, dropout=dropout,
        )

        fused_dim = fusion_hidden * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
        )

    def forward(self, mol_data, prot_ids: torch.Tensor) -> torch.Tensor:
        # BondCNN runs in float32 to avoid AMP dtype conflicts in scatter_add_
        with torch.amp.autocast('cuda', enabled=False):
            x_f32    = mol_data.x.float()
            ea       = getattr(mol_data, 'edge_attr', None)
            bond_ctx = self.bond_cnn(x_f32, mol_data.edge_index,
                                     ea.float() if ea is not None else ea)
            x_aug = torch.cat([x_f32, bond_ctx], dim=-1)  # [N, 105]

        # GNN encoding
        mol_h    = self.mol_encoder(x_aug, mol_data.edge_index, mol_data.batch)
        mol_repr = F.relu(self.mol_pool(mol_h))   # [B, gnn_hidden]

        # Hybrid protein encoding
        prot_repr = self.prot_encoder(prot_ids)    # [B, prot_hidden]

        # Gated fusion
        fused = self.fusion(mol_repr, prot_repr)   # [B, 2*fusion_hidden]

        return self.head(fused)                    # [B, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# FULL DTA PREDICTOR
# ============================================================================
class DTAPredictor(nn.Module):
    """
    Full Drug-Target Affinity predictor combining:
      - Molecular GNN encoder (GCN / GAT / GIN)
      - Protein Transformer encoder
      - CrossGraphAttention fusion
      - MLP prediction head

    gnn_type: 'gcn' | 'gat' | 'gin'
    """

    def __init__(
        self,
        gnn_type: str      = 'gin',
        mol_in_dim: int    = ATOM_FEAT_DIM,
        gnn_hidden: int    = 256,
        gnn_layers: int    = 4,
        prot_embed: int    = 128,
        prot_hidden: int   = 256,
        prot_layers: int   = 4,
        fusion_hidden: int = 256,
        dropout: float     = 0.2,
    ):
        super().__init__()
        self.gnn_type = gnn_type

        # Molecular encoder
        if gnn_type == 'gcn':
            self.mol_encoder = GCNEncoder(mol_in_dim, gnn_hidden, gnn_layers, dropout)
        elif gnn_type == 'gat':
            self.mol_encoder = GATEncoder(mol_in_dim, gnn_hidden, gnn_layers,
                                          n_heads=8, dropout=dropout)
        elif gnn_type == 'gin':
            self.mol_encoder = GINEncoder(mol_in_dim, gnn_hidden, gnn_layers, dropout)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        # global_pool returns [B, 2*gnn_hidden] — project down to gnn_hidden
        self.mol_pool = nn.Linear(gnn_hidden * 2, gnn_hidden)

        # Protein encoder
        self.prot_encoder = ProteinTransformerEncoder(
            embed_dim=prot_embed, hidden_dim=prot_hidden,
            n_layers=prot_layers, dropout=dropout
        )

        # Cross-attention fusion
        self.fusion = CrossGraphAttention(
            drug_dim=gnn_hidden, prot_dim=prot_hidden,
            hidden_dim=fusion_hidden, n_heads=8
        )

        # MLP prediction head
        fused_dim = fusion_hidden * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
        )

    def forward(self, mol_data: Data, prot_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mol_data: batched PyG Data (x, edge_index, batch)
            prot_ids: [B, seq_len]
        Returns:
            affinity: [B, 1]
        """
        # Molecular encoding
        mol_h = self.mol_encoder(mol_data.x, mol_data.edge_index, mol_data.batch)
        mol_repr = F.relu(self.mol_pool(mol_h))   # [B, gnn_hidden]

        # Protein encoding
        prot_repr = self.prot_encoder(prot_ids)    # [B, prot_hidden]

        # Cross-attention fusion
        fused = self.fusion(mol_repr, prot_repr)   # [B, 2*fusion_hidden]

        return self.head(fused)                    # [B, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# GRAPH DATASET
# ============================================================================
class GraphDTADataset(Dataset):
    """
    PyTorch Dataset that pre-builds molecular + protein data for DTA samples.
    Caches all graphs and protein token IDs in RAM on first construction so
    __getitem__ is a pure tensor lookup (no RDKit calls during training).

    Each sample: {'drug_smiles': str, 'protein_sequence': str, 'affinity': float}
    """

    def __init__(self, samples: List[Dict], max_prot_len: int = 1000,
                 normalizer=None):
        self.max_prot_len = max_prot_len
        self.normalizer   = normalizer
        mol_builder       = MolecularGraphBuilder()

        logger.info(f"  Pre-building {len(samples)} graphs (one-time cache)...")
        self.mol_cache   = []
        self.prot_cache  = []
        self.target_cache = []

        for s in samples:
            # Molecular graph
            self.mol_cache.append(mol_builder.smiles_to_pyg(str(s['drug_smiles'])))

            # Protein token IDs
            seq = str(s['protein_sequence']).upper()[:max_prot_len]
            seq = seq.ljust(max_prot_len, 'X')
            self.prot_cache.append(
                torch.tensor([AA_VOCAB.get(aa, 0) for aa in seq], dtype=torch.long)
            )

            # Target
            affinity = float(s['affinity'])
            if normalizer is not None:
                affinity = normalizer.normalize(affinity)
            self.target_cache.append(torch.tensor([affinity], dtype=torch.float32))

        logger.info(f"  Cache ready.")

    def __len__(self):
        return len(self.mol_cache)

    def __getitem__(self, idx: int):
        return self.mol_cache[idx], self.prot_cache[idx], self.target_cache[idx]


def collate_fn(batch):
    """Custom collate for (PyG Data, prot_ids, target) tuples."""
    mol_graphs, prot_ids_list, targets = zip(*batch)
    mol_batch  = Batch.from_data_list(list(mol_graphs))
    prot_batch = torch.stack(list(prot_ids_list), dim=0)
    tgt_batch  = torch.stack(list(targets), dim=0)
    return mol_batch, prot_batch, tgt_batch


def balance_samples(samples: List[Dict], majority_value: float = 5.0,
                    keep_ratio: float = 0.3, seed: int = 42) -> List[Dict]:
    """
    Undersample the majority class (affinity == majority_value).
    keep_ratio: fraction of majority samples to keep (0.3 = keep 30%).
    All non-majority samples are kept.
    This fixes the DAVIS dataset skew where 69.6% of labels are 5.0.
    """
    rng = np.random.default_rng(seed)
    majority = [s for s in samples if float(s['affinity']) == majority_value]
    minority = [s for s in samples if float(s['affinity']) != majority_value]
    n_keep   = max(len(minority), int(len(majority) * keep_ratio))
    kept_maj = majority[:n_keep] if n_keep >= len(majority) else \
               list(rng.choice(majority, size=n_keep, replace=False))  # type: ignore
    balanced = minority + kept_maj
    rng.shuffle(balanced)
    logger.info(
        f"  Balanced dataset: {len(minority)} minority + {len(kept_maj)} majority "
        f"= {len(balanced)} total (was {len(samples)})"
    )
    return balanced


def build_dataloader(samples: List[Dict], batch_size: int = 32,
                     shuffle: bool = True, max_prot_len: int = 1000,
                     normalizer=None, num_workers: int = 0,
                     balance: bool = False):
    """
    Build a DataLoader from DTA samples.
    Set balance=True on training sets to undersample the affinity=5.0 majority.
    """
    if balance and shuffle:  # only balance training sets
        samples = balance_samples(samples)
    dataset = GraphDTADataset(samples, max_prot_len=max_prot_len,
                              normalizer=normalizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================================
# UTILITIES
# ============================================================================
def sequence_to_ids(sequence: str, max_len: int = 1000) -> torch.Tensor:
    """Convert a protein sequence string to padded token-ID tensor [1, max_len]."""
    seq = sequence.upper()[:max_len].ljust(max_len, 'X')
    return torch.tensor([AA_VOCAB.get(aa, 0) for aa in seq],
                        dtype=torch.long).unsqueeze(0)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Return MSE, RMSE, MAE, R² from numpy arrays."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse  = float(mean_squared_error(targets, preds))
    mae  = float(mean_absolute_error(targets, preds))
    r2   = float(r2_score(targets, preds))
    rmse = float(np.sqrt(mse))
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def concordance_index(preds: np.ndarray, targets: np.ndarray,
                      max_pairs: int = 50_000) -> float:
    """
    Compute Concordance Index (CI) — standard DTA metric.
    Randomly samples up to max_pairs to keep runtime O(max_pairs).
    """
    n = len(preds)
    rng = np.random.default_rng(0)
    idx_i = rng.integers(0, n, size=max_pairs)
    idx_j = rng.integers(0, n, size=max_pairs)
    mask  = targets[idx_i] != targets[idx_j]
    ti, tj = targets[idx_i][mask], targets[idx_j][mask]
    pi, pj = preds[idx_i][mask],   preds[idx_j][mask]
    total  = mask.sum()
    if total == 0:
        return 0.5
    concordant = float(((ti > tj) == (pi > pj)).sum()) + 0.5 * float((pi == pj).sum())
    return concordant / total
