"""
PHASE 2: GCN BASELINE TRAINING
================================
Trains a Graph Convolutional Network (GCN) as the GML baseline.

Key GML concepts:
  - GCNConv: aggregates neighbor node features via normalized sum
  - Global mean + max pooling for graph-level representation
  - Residual connections to prevent over-smoothing
  - Full drug-target affinity model: GCN + Transformer + Cross-Attention

Performance improvements over v1:
  - Epochs 30 -> 50
  - Batch size 32 -> 64 (better GPU utilization)
  - Gradient accumulation (effective batch = 128)
  - Warmup + CosineAnnealingLR
  - Label smoothing via soft targets
  - Protein sequence length 1000 -> 1200
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from gml_core import (
    DTAPredictor, build_dataloader, compute_metrics,
    ATOM_FEAT_DIM, PYG_AVAILABLE,
)
from phase3_real_data import DAVISDatasetLoader
from target_normalizer import AffinityNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# GCN TRAINER
# ============================================================================
class GCNTrainer:

    def __init__(self, config: dict, normalizer: AffinityNormalizer):
        self.config     = config
        self.normalizer = normalizer
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}", flush=True)

        self.model = DTAPredictor(
            gnn_type    = 'gcn',
            mol_in_dim  = ATOM_FEAT_DIM,
            gnn_hidden  = config.get('gnn_hidden', 256),
            gnn_layers  = config.get('gnn_layers', 4),
            prot_embed  = config.get('prot_embed', 128),
            prot_hidden = config.get('prot_hidden', 256),
            prot_layers = config.get('prot_layers', 4),
            dropout     = config.get('dropout', 0.15),
        ).to(self.device)
        print(f"GCN model parameters: {self.model.count_parameters():,}", flush=True)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
        )
        self.criterion   = nn.HuberLoss(delta=0.5)  # tighter delta for affinity range
        self.accum_steps = config.get('accum_steps', 2)  # gradient accumulation
        self.best_val_r2 = float('-inf')
        self.best_state  = None

    def _warmup_lr(self, epoch, warmup_epochs=3):
        if epoch < warmup_epochs:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.config.get('lr', 1e-3) * (epoch + 1) / warmup_epochs

    def train_epoch(self, loader, epoch) -> float:
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

        print("Building DataLoaders...", flush=True)
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
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs - 3, eta_min=1e-6)

        logger.info(f"Starting GCN training for {epochs} epochs "
                    f"(accum_steps={self.accum_steps}, "
                    f"effective_batch={cfg.get('batch_size',64)*self.accum_steps})...")

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
    print("PHASE 2: GCN BASELINE — GRAPH CONVOLUTIONAL NETWORK")
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
        'accum_steps':  4,      # effective batch = 128
        'lr':           1e-3,
        'weight_decay': 1e-4,
        'gnn_hidden':   128,
        'gnn_layers':   4,
        'prot_embed':   64,
        'prot_hidden':  128,
        'prot_layers':  3,
        'dropout':      0.15,
        'max_prot_len': 800,
    }

    trainer = GCNTrainer(config, normalizer)
    results = trainer.train(train, val, test)

    print("\n" + "=" * 80)
    print("PHASE 2 RESULTS (GCN BASELINE)")
    print("=" * 80)
    print(f"  Best Val R2   : {results['best_val_r2']:.4f}")
    print(f"  Test R2       : {results['test_r2']:.4f}")
    print(f"  Test RMSE     : {results['test_rmse']:.4f}")
    print(f"  Test MAE      : {results['test_mae']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
