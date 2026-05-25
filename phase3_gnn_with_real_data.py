"""
PHASE 3: GAT TRAINING — GRAPH ATTENTION NETWORK
=================================================
Upgrades from GCN to GAT (Graph Attention Network) which learns
to attend to the most relevant neighboring atoms.

Key GML concepts:
  - GATConv: multi-head attention over node neighbors
  - Edge features fed into attention (bond type, aromaticity)
  - Concatenated head outputs -> richer representations

Performance improvements over v1:
  - CosineAnnealingWarmRestarts instead of OneCycleLR (more stable)
  - Epochs 30 -> 50
  - Batch 32 -> 64, gradient accumulation
  - Lower dropout 0.2 -> 0.15
  - Protein length 1000 -> 1200
  - Edge features passed to GATConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from gml_core import (
    DTAPredictor, build_dataloader, compute_metrics,
    ATOM_FEAT_DIM, BOND_FEAT_DIM, PYG_AVAILABLE,
    ProteinTransformerEncoder, CrossGraphAttention,
)
from phase3_real_data import DAVISDatasetLoader
from target_normalizer import AffinityNormalizer

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    GATV2_AVAILABLE = True
except ImportError:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    GATV2_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# IMPROVED GAT ENCODER — uses GATv2Conv with edge features
# ============================================================================
class ImprovedGATEncoder(nn.Module):
    """
    GATv2 encoder with bond features passed as edge attributes.
    GATv2 fixes the expressivity limitation in GATv1 by using dynamic attention.
    """

    def __init__(self, in_dim: int = ATOM_FEAT_DIM, hidden_dim: int = 256,
                 edge_dim: int = BOND_FEAT_DIM, n_layers: int = 4,
                 n_heads: int = 8, dropout: float = 0.15):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        head_dim = hidden_dim // n_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            if GATV2_AVAILABLE:
                self.convs.append(
                    GATv2Conv(hidden_dim, head_dim, heads=n_heads,
                              edge_dim=edge_dim, dropout=dropout, concat=True)
                )
            else:
                from torch_geometric.nn import GATConv
                self.convs.append(
                    GATConv(hidden_dim, head_dim, heads=n_heads,
                            dropout=dropout, concat=True)
                )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            if GATV2_AVAILABLE and edge_attr is not None:
                h_new = F.relu(norm(conv(h, edge_index, edge_attr=edge_attr)))
            else:
                h_new = F.relu(norm(conv(h, edge_index)))
            h_new = self.dropout(h_new)
            h = h + h_new  # residual

        g_mean = global_mean_pool(h, batch)
        g_max  = global_max_pool(h, batch)
        return torch.cat([g_mean, g_max], dim=-1)  # [B, 2*hidden]


# ============================================================================
# FULL GAT DTA MODEL
# ============================================================================
class GATDTAPredictor(nn.Module):
    def __init__(self, mol_in_dim=ATOM_FEAT_DIM, edge_dim=BOND_FEAT_DIM,
                 gnn_hidden=256, gnn_layers=4, prot_embed=128, prot_hidden=256,
                 prot_layers=4, fusion_hidden=256, dropout=0.15):
        super().__init__()
        self.mol_encoder  = ImprovedGATEncoder(
            mol_in_dim, gnn_hidden, edge_dim, gnn_layers, n_heads=8, dropout=dropout
        )
        self.mol_proj     = nn.Linear(gnn_hidden * 2, gnn_hidden)
        self.prot_encoder = ProteinTransformerEncoder(
            embed_dim=prot_embed, hidden_dim=prot_hidden,
            n_layers=prot_layers, dropout=dropout,
        )
        self.fusion = CrossGraphAttention(
            drug_dim=gnn_hidden, prot_dim=prot_hidden,
            hidden_dim=fusion_hidden, n_heads=8,
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

    def forward(self, mol_data, prot_ids):
        edge_attr = getattr(mol_data, 'edge_attr', None)
        mol_h  = self.mol_encoder(mol_data.x, mol_data.edge_index,
                                   mol_data.batch, edge_attr)
        mol_r  = F.relu(self.mol_proj(mol_h))
        prot_r = self.prot_encoder(prot_ids)
        fused  = self.fusion(mol_r, prot_r)
        return self.head(fused)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# GAT TRAINER
# ============================================================================
class GATTrainer:

    def __init__(self, config: dict, normalizer: AffinityNormalizer):
        self.config     = config
        self.normalizer = normalizer
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}", flush=True)
        print(f"GATv2 available: {GATV2_AVAILABLE}", flush=True)

        self.model = GATDTAPredictor(
            mol_in_dim  = ATOM_FEAT_DIM,
            edge_dim    = BOND_FEAT_DIM,
            gnn_hidden  = config.get('gnn_hidden', 256),
            gnn_layers  = config.get('gnn_layers', 4),
            prot_embed  = config.get('prot_embed', 128),
            prot_hidden = config.get('prot_hidden', 256),
            prot_layers = config.get('prot_layers', 4),
            dropout     = config.get('dropout', 0.15),
        ).to(self.device)
        print(f"GAT model parameters: {self.model.count_parameters():,}", flush=True)

        self.optimizer   = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 5e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
        )
        self.criterion   = nn.HuberLoss(delta=0.5)
        self.accum_steps = config.get('accum_steps', 2)
        self.best_val_r2 = float('-inf')
        self.best_state  = None

    def _warmup_lr(self, epoch, warmup_epochs=3):
        if epoch < warmup_epochs:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.config.get('lr', 5e-4) * (epoch + 1) / warmup_epochs

    def train_epoch(self, loader, epoch=0) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        self.optimizer.zero_grad()
        bar = tqdm(loader, desc=f"  Epoch {epoch+1}", leave=True, dynamic_ncols=True)

        for step, (mol_batch, prot_ids, targets) in enumerate(bar):
            mol_batch = mol_batch.to(self.device)
            prot_ids  = prot_ids.to(self.device)
            targets   = targets.to(self.device)

            preds = self.model(mol_batch, prot_ids)
            loss  = self.criterion(preds, targets) / self.accum_steps
            loss.backward()

            if (step + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps * targets.size(0)
            n += targets.size(0)
            bar.set_postfix(loss=f"{total_loss/max(n,1):.4f}")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_targets = [], []
        for mol_batch, prot_ids, targets in loader:
            mol_batch = mol_batch.to(self.device)
            prot_ids  = prot_ids.to(self.device)
            preds = self.model(mol_batch, prot_ids).cpu().numpy().flatten()
            all_preds.extend(self.normalizer.denormalize_array(preds))
            all_targets.extend(self.normalizer.denormalize_array(targets.numpy().flatten()))
        return compute_metrics(np.array(all_preds), np.array(all_targets))

    def train(self, train_data, val_data, test_data) -> dict:
        cfg     = self.config
        max_len = cfg.get('max_prot_len', 1200)

        train_loader = build_dataloader(
            train_data, batch_size=cfg.get('batch_size', 64),
            shuffle=True,  normalizer=self.normalizer,
            balance=True,  max_prot_len=max_len,
        )
        val_loader = build_dataloader(
            val_data, batch_size=cfg.get('batch_size', 64),
            shuffle=False, normalizer=self.normalizer,
            max_prot_len=max_len,
        )
        test_loader = build_dataloader(
            test_data, batch_size=cfg.get('batch_size', 64),
            shuffle=False, normalizer=self.normalizer,
            max_prot_len=max_len,
        )

        epochs    = cfg.get('epochs', 50)
        # CosineAnnealingWarmRestarts: restarts every T_0 epochs, much more stable than OneCycleLR
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=2, eta_min=1e-6
        )

        logger.info(f"Starting GAT training for {epochs} epochs "
                    f"(accum_steps={self.accum_steps})...")

        for epoch in range(epochs):
            t0 = time.time()
            self._warmup_lr(epoch)
            tr_loss = self.train_epoch(train_loader, epoch)
            val_m   = self.evaluate(val_loader)
            if epoch >= 3:
                scheduler.step()

            if val_m['r2'] > self.best_val_r2:
                self.best_val_r2 = val_m['r2']
                self.best_state  = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(
                f"Epoch {epoch+1:3d}/{epochs} ({time.time()-t0:.0f}s) | "
                f"Loss: {tr_loss:.4f} | "
                f"Val  R2={val_m['r2']:.4f}  RMSE={val_m['rmse']:.4f}  MAE={val_m['mae']:.4f}",
                flush=True
            )

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        test_m = self.evaluate(test_loader)
        print(f"Test R2={test_m['r2']:.4f}  RMSE={test_m['rmse']:.4f}  MAE={test_m['mae']:.4f}", flush=True)
        return {'best_val_r2': self.best_val_r2, **{f'test_{k}': v for k, v in test_m.items()}}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("PHASE 3: GATv2 — GRAPH ATTENTION NETWORK v2 WITH EDGE FEATURES")
    print("=" * 80 + "\n")

    loader = DAVISDatasetLoader(data_dir="data")
    data   = loader.load_davis()
    stats  = loader.get_statistics(data)
    train, val, test = loader.create_splits(data)

    normalizer = AffinityNormalizer(mean=stats['affinity_mean'], std=stats['affinity_std'])
    print(f"Normalizer: {normalizer.info()}", flush=True)

    config = {
        'epochs':       50,
        'batch_size':   32,
        'accum_steps':  4,
        'lr':           5e-4,
        'weight_decay': 1e-4,
        'gnn_hidden':   128,
        'gnn_layers':   4,
        'prot_embed':   64,
        'prot_hidden':  128,
        'prot_layers':  3,
        'dropout':      0.15,
        'max_prot_len': 800,
    }

    trainer = GATTrainer(config, normalizer)
    results = trainer.train(train, val, test)

    print("\n" + "=" * 80)
    print("PHASE 3 RESULTS (GATv2)")
    print("=" * 80)
    print(f"  Best Val R2   : {results['best_val_r2']:.4f}")
    print(f"  Test R2       : {results['test_r2']:.4f}")
    print(f"  Test RMSE     : {results['test_rmse']:.4f}")
    print(f"  Test MAE      : {results['test_mae']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
