"""
PHASE 4: GIN — GRAPH ISOMORPHISM NETWORK
==========================================
GIN is the most powerful GNN in the Weisfeiler-Lehman hierarchy.
Adds graph-level fingerprint features for richer representations.

Key GML concepts:
  - GINConv: epsilon-scaled self-features + neighbor aggregation
  - Trainable epsilon per layer (train_eps=True)
  - MLP inside each GIN layer (vs single linear in GCN/GAT)
  - Sum aggregation (not mean) — provably distinguishes more graph structures
  - Hierarchical graph readout: concatenate pooling from all layers (JK-net style)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from gml_core import (
    DTAPredictor, build_dataloader, compute_metrics,
    concordance_index, ATOM_FEAT_DIM, PYG_AVAILABLE,
    GINEncoder, ProteinTransformerEncoder, CrossGraphAttention,
)
from phase3_real_data import DAVISDatasetLoader, KIBADatasetLoader
from target_normalizer import AffinityNormalizer

try:
    from torch_geometric.nn import global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    import torch.nn.functional as F
    from torch_geometric.nn import GINConv
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# JK-GIN: Jumping Knowledge GIN with hierarchical readout
# ============================================================================
class JKGINEncoder(nn.Module):
    """
    GIN encoder with Jumping Knowledge (JK) connections.
    Concatenates graph-level pooled features from every layer, then projects
    through a 2-layer MLP (prevents bottleneck information loss).
    Per-layer residual GIN connections prevent oversmoothing.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 192,
                 n_layers: int = 5, dropout: float = 0.1):
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

        self.n_layers = n_layers
        self.dropout  = nn.Dropout(dropout)
        # JK: 2-layer MLP projection prevents the 640->128 information bottleneck.
        # Concatenated readouts: hidden*(n_layers+1)  ->  256  ->  hidden
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (n_layers + 1), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )
        self.out_dim  = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        layer_readouts = [global_mean_pool(h, batch)]  # layer 0

        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(norm(conv(h, edge_index)))
            h_new = self.dropout(h_new)
            h = h + h_new  # residual to prevent oversmoothing
            layer_readouts.append(global_mean_pool(h, batch))

        # JK concatenation + MLP projection
        jk = torch.cat(layer_readouts, dim=-1)  # [B, hidden*(n_layers+1)]
        return F.relu(self.jk_proj(jk))          # [B, hidden]


# ============================================================================
# FULL GIN MODEL
# ============================================================================
class GINDTAPredictor(nn.Module):
    """DTA predictor with JK-GIN molecular encoder."""

    def __init__(self, mol_in_dim=ATOM_FEAT_DIM, gnn_hidden=256,
                 gnn_layers=5, prot_embed=128, prot_hidden=256,
                 prot_layers=4, fusion_hidden=256, dropout=0.2):
        super().__init__()
        self.mol_encoder  = JKGINEncoder(mol_in_dim, gnn_hidden, gnn_layers, dropout)
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
        mol_r  = self.mol_encoder(mol_data.x, mol_data.edge_index, mol_data.batch)
        prot_r = self.prot_encoder(prot_ids)
        fused  = self.fusion(mol_r, prot_r)
        return self.head(fused)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TRAINER
# ============================================================================
class GINTrainer:
    def __init__(self, config: dict, normalizer: AffinityNormalizer):
        self.config     = config
        self.normalizer = normalizer
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp    = config.get('use_amp', True) and self.device.type == 'cuda'
        print(f"Device: {self.device} | AMP: {self.use_amp}", flush=True)

        self.model = GINDTAPredictor(
            mol_in_dim  = ATOM_FEAT_DIM,
            gnn_hidden  = config.get('gnn_hidden', 192),
            gnn_layers  = config.get('gnn_layers', 5),
            prot_embed  = config.get('prot_embed', 128),
            prot_hidden = config.get('prot_hidden', 192),
            prot_layers = config.get('prot_layers', 4),
            dropout     = config.get('dropout', 0.1),
        ).to(self.device)
        print(f"JK-GIN model parameters: {self.model.count_parameters():,}", flush=True)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1.5e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        self.criterion   = nn.HuberLoss(delta=0.5)
        self.scaler      = GradScaler(enabled=self.use_amp)
        self.accum_steps = config.get('accum_steps', 2)
        self.warmup_epochs = config.get('warmup_epochs', 3)
        self.base_lr       = config.get('lr', 1.5e-3)
        self.best_val_r2 = float('-inf')
        self.best_state  = None

    def _warmup_lr(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * scale

    def train_epoch(self, loader, epoch=0) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        pending = False  # whether we have un-stepped grads accumulating
        self.optimizer.zero_grad(set_to_none=True)
        bar = tqdm(loader, desc=f"  Epoch {epoch+1}", leave=True, dynamic_ncols=True)
        for step, (mol_batch, prot_ids, targets) in enumerate(bar):
            mol_batch = mol_batch.to(self.device, non_blocking=True)
            prot_ids  = prot_ids.to(self.device, non_blocking=True)
            targets   = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                preds = self.model(mol_batch, prot_ids)
                loss  = self.criterion(preds, targets) / self.accum_steps

            # skip step if loss is non-finite (FP16 overflow can produce nan/inf)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                pending = False
                continue

            self.scaler.scale(loss).backward()
            pending = True

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                pending = False

            total_loss += loss.item() * self.accum_steps * targets.size(0)
            n += targets.size(0)
            bar.set_postfix(loss=f"{total_loss/max(n,1):.4f}")

        # flush any leftover partial accumulation (only if pending)
        if pending:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
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
        p, t = np.array(all_preds), np.array(all_targets)
        # Guard against NaN/Inf from FP16 overflow corrupting predictions
        if not np.isfinite(p).all():
            print("  [WARN] Non-finite preds detected, returning sentinel metrics", flush=True)
            return {'r2': -1e9, 'rmse': float('inf'), 'mae': float('inf'), 'ci': 0.0}
        metrics = compute_metrics(p, t)
        metrics['ci'] = concordance_index(p, t)
        return metrics

    def train(self, train_data, val_data, test_data) -> dict:
        cfg = self.config
        max_len = cfg.get('max_prot_len', 1200)
        train_loader = build_dataloader(
            train_data, batch_size=cfg.get('batch_size', 16),
            shuffle=True,  normalizer=self.normalizer, balance=True,
            max_prot_len=max_len,
        )
        val_loader = build_dataloader(
            val_data, batch_size=cfg.get('batch_size', 16),
            shuffle=False, normalizer=self.normalizer,
            max_prot_len=max_len,
        )
        test_loader = build_dataloader(
            test_data, batch_size=cfg.get('batch_size', 16),
            shuffle=False, normalizer=self.normalizer,
            max_prot_len=max_len,
        )

        epochs = cfg.get('epochs', 30)
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        for epoch in range(epochs):
            t0 = time.time()
            self._warmup_lr(epoch)
            tr_loss = self.train_epoch(train_loader, epoch)
            val_m   = self.evaluate(val_loader)
            if epoch >= self.warmup_epochs:
                scheduler.step()

            if val_m['r2'] > self.best_val_r2:
                self.best_val_r2 = val_m['r2']
                self.best_state  = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(
                f"Epoch {epoch+1:3d}/{epochs} ({time.time()-t0:.0f}s) | "
                f"Loss: {tr_loss:.4f} | "
                f"Val  R2={val_m['r2']:.4f}  RMSE={val_m['rmse']:.4f}  CI={val_m['ci']:.4f}",
                flush=True
            )

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        test_m = self.evaluate(test_loader)
        print(f"Test  R2={test_m['r2']:.4f}  RMSE={test_m['rmse']:.4f}  CI={test_m['ci']:.4f}", flush=True)
        return {'best_val_r2': self.best_val_r2, **{f'test_{k}': v for k, v in test_m.items()}}

    def save_encoder(self, path: str):
        """Save mol_encoder + prot_encoder weights for transfer to other phases."""
        torch.save({
            'mol_encoder':  self.model.mol_encoder.state_dict(),
            'prot_encoder': self.model.prot_encoder.state_dict(),
            'fusion':       self.model.fusion.state_dict(),
        }, path)
        print(f"Saved pretrained encoder to {path}", flush=True)

    def load_encoder(self, path: str):
        """Load pretrained encoder weights from a previous training run."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.mol_encoder.load_state_dict(ckpt['mol_encoder'])
        self.model.prot_encoder.load_state_dict(ckpt['prot_encoder'])
        if 'fusion' in ckpt:
            self.model.fusion.load_state_dict(ckpt['fusion'])
        print(f"Loaded pretrained encoder from {path}", flush=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("PHASE 4: JK-GIN - TRANSFER LEARNING (KIBA -> DAVIS)")
    print("=" * 80 + "\n")

    # Shared model config (same hidden dims across both stages)
    base_config = {
        'batch_size':     12,
        'accum_steps':    4,        # effective batch = 48
        'lr':             8e-4,
        'warmup_epochs':  3,
        'weight_decay':   1e-4,
        'gnn_hidden':     192,
        'gnn_layers':     5,
        'prot_embed':     128,
        'prot_hidden':    192,
        'prot_layers':    4,
        'dropout':        0.1,
        'max_prot_len':   1200,
        'use_amp':        True,
    }

    ckpt_path = Path(__file__).parent / "phase4_pretrained_encoder.pt"

    # ------------------------------------------------------------------
    # STAGE 1: PRETRAIN on KIBA (~118k samples)
    # ------------------------------------------------------------------
    print("\n[STAGE 1] PRETRAIN on KIBA (~118k samples)\n", flush=True)
    kiba_loader = KIBADatasetLoader(data_dir="data")
    kiba_data   = kiba_loader.load_kiba()
    kiba_stats  = kiba_loader.get_statistics(kiba_data)
    k_train, k_val, k_test = kiba_loader.create_splits(kiba_data)

    kiba_normalizer = AffinityNormalizer(
        mean=kiba_stats['affinity_mean'], std=kiba_stats['affinity_std'])
    print(f"KIBA Normalizer: {kiba_normalizer.info()}", flush=True)

    pretrain_cfg = {**base_config, 'epochs': 15}  # shorter pretrain — just learn general patterns
    pretrainer = GINTrainer(pretrain_cfg, kiba_normalizer)
    kiba_results = pretrainer.train(k_train, k_val, k_test)

    print("\n--- KIBA PRETRAIN RESULTS ---", flush=True)
    print(f"  Test R2={kiba_results['test_r2']:.4f}  "
          f"RMSE={kiba_results['test_rmse']:.4f}  "
          f"CI={kiba_results['test_ci']:.4f}", flush=True)

    pretrainer.save_encoder(str(ckpt_path))

    # ------------------------------------------------------------------
    # STAGE 2: FINE-TUNE on DAVIS (~30k samples)
    # ------------------------------------------------------------------
    print("\n[STAGE 2] FINE-TUNE on DAVIS (~30k samples)\n", flush=True)
    davis_loader = DAVISDatasetLoader(data_dir="data")
    davis_data   = davis_loader.load_davis()
    davis_stats  = davis_loader.get_statistics(davis_data)
    d_train, d_val, d_test = davis_loader.create_splits(davis_data)

    davis_normalizer = AffinityNormalizer(
        mean=davis_stats['affinity_mean'], std=davis_stats['affinity_std'])
    print(f"DAVIS Normalizer: {davis_normalizer.info()}", flush=True)

    # Lower LR for fine-tuning (don't blow away pretrained features)
    finetune_cfg = {**base_config, 'epochs': 50, 'lr': 3e-4, 'warmup_epochs': 2}
    finetuner = GINTrainer(finetune_cfg, davis_normalizer)
    finetuner.load_encoder(str(ckpt_path))
    results = finetuner.train(d_train, d_val, d_test)

    print("\n" + "=" * 80)
    print("PHASE 4 RESULTS (TRANSFER LEARNING: KIBA -> DAVIS)")
    print("=" * 80)
    print(f"  KIBA pretrain Test R2 : {kiba_results['test_r2']:.4f}")
    print(f"  DAVIS finetune Test R2: {results['test_r2']:.4f}")
    print(f"  DAVIS finetune RMSE   : {results['test_rmse']:.4f}")
    print(f"  DAVIS finetune MAE    : {results['test_mae']:.4f}")
    print(f"  DAVIS finetune CI     : {results['test_ci']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
