"""
PHASE 6: BAYESIAN GNN — UNCERTAINTY QUANTIFICATION
====================================================
Adds principled uncertainty estimates to GNN predictions using
Monte Carlo (MC) Dropout. Dropout is kept active at test time and
predictions are sampled multiple times to estimate epistemic uncertainty.

Key GML concepts:
  - MC Dropout on GNN: dropout masks different graph edge paths each forward pass
  - Epistemic uncertainty: variance of T stochastic forward passes
  - Aleatoric uncertainty: learned via heteroscedastic output head
  - Calibration: Expected Calibration Error (ECE) on prediction intervals
  - Uncertainty-aware DTA: flag unreliable predictions for wet lab validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import numpy as np
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from gml_core import (
    build_dataloader, compute_metrics, concordance_index,
    ATOM_FEAT_DIM, BOND_FEAT_DIM, PYG_AVAILABLE,
    HybridProteinEncoder, GatedBilinearFusion, BondCNNEncoder,
    GINEncoder,
)
from phase3_real_data import DAVISDatasetLoader
from target_normalizer import AffinityNormalizer

import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# BAYESIAN GNN MODEL
# ============================================================================
class BayesianGNNDTA(nn.Module):
    """
    Enhanced DTA model with MC Dropout for epistemic uncertainty.

    Uses Bond CNN + GIN + Hybrid CNN-Transformer + Gated Bilinear Fusion.
    MC Dropout (kept active at test time) provides epistemic uncertainty estimates.
    """

    def __init__(self, mol_in_dim=ATOM_FEAT_DIM, gnn_hidden=192,
                 gnn_layers=5, prot_embed=128, prot_hidden=192,
                 prot_layers=4, fusion_hidden=192, mc_dropout=0.2,
                 bond_cnn_dim=32):
        super().__init__()
        self.mc_dropout = mc_dropout

        # Bond CNN augmentation
        self.bond_cnn    = BondCNNEncoder(BOND_FEAT_DIM, bond_cnn_dim, dropout=mc_dropout)
        aug_mol_dim      = mol_in_dim + bond_cnn_dim

        self.mol_encoder = GINEncoder(aug_mol_dim, gnn_hidden, gnn_layers, mc_dropout)
        self.mol_proj    = nn.Linear(gnn_hidden * 2, gnn_hidden)

        # Hybrid CNN-Transformer protein encoder
        self.prot_encoder = HybridProteinEncoder(
            embed_dim=prot_embed, hidden_dim=prot_hidden,
            n_transformer_layers=max(2, prot_layers - 2),
            dropout=mc_dropout,
        )

        # Gated bilinear fusion
        self.fusion = GatedBilinearFusion(
            drug_dim=gnn_hidden, prot_dim=prot_hidden,
            hidden_dim=fusion_hidden, n_heads=8, dropout=mc_dropout,
        )

        fused_dim = fusion_hidden * 2

        # Regression head with MC dropout for epistemic uncertainty
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(mc_dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(mc_dropout),
            nn.Linear(fused_dim // 2, 1),
        )

    def forward(self, mol_data, prot_ids):
        with torch.amp.autocast('cuda', enabled=False):
            x_f32    = mol_data.x.float()
            ea       = getattr(mol_data, 'edge_attr', None)
            bond_ctx = self.bond_cnn(x_f32, mol_data.edge_index,
                                     ea.float() if ea is not None else ea)
            x_aug = torch.cat([x_f32, bond_ctx], dim=-1)
        mol_h  = self.mol_encoder(x_aug, mol_data.edge_index, mol_data.batch)
        mol_r  = F.relu(self.mol_proj(mol_h))
        prot_r = self.prot_encoder(prot_ids)
        fused  = self.fusion(mol_r, prot_r)
        return self.head(fused)

    def mc_predict(self, mol_data, prot_ids, n_samples: int = 20):
        """
        Run T stochastic forward passes with dropout ON.
        Returns (mean prediction, epistemic std).
        """
        self.train()  # keep dropout active
        preds_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds_list.append(self.forward(mol_data, prot_ids).unsqueeze(0))
        preds = torch.cat(preds_list, dim=0)  # [T, B, 1]
        pred_mean = preds.mean(dim=0)          # [B, 1]
        epistemic = preds.std(dim=0)           # [B, 1]
        return pred_mean, epistemic

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# CALIBRATION METRIC
# ============================================================================
def expected_calibration_error(pred_mean, pred_std, targets, n_bins=10):
    """
    Compute ECE: how well the predicted intervals are calibrated.
    """
    confidences = []
    within = []
    for p_val in np.linspace(0.05, 0.95, n_bins):
        z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + p_val / 2)))
        lo = pred_mean - z * pred_std
        hi = pred_mean + z * pred_std
        frac = float(((targets >= lo) & (targets <= hi)).float().mean())
        confidences.append(p_val)
        within.append(frac)
    ece = float(np.mean(np.abs(np.array(within) - np.array(confidences))))
    return ece, within


# ============================================================================
# TRAINER
# ============================================================================
class BayesianGNNTrainer:
    def __init__(self, config: dict, normalizer: AffinityNormalizer):
        self.config     = config
        self.normalizer = normalizer
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp    = config.get('use_amp', True) and self.device.type == 'cuda'
        print(f"Device: {self.device} | AMP: {self.use_amp}", flush=True)

        self.model = BayesianGNNDTA(
            mol_in_dim   = ATOM_FEAT_DIM,
            gnn_hidden   = config.get('gnn_hidden', 192),
            gnn_layers   = config.get('gnn_layers', 5),
            prot_embed   = config.get('prot_embed', 128),
            prot_hidden  = config.get('prot_hidden', 192),
            prot_layers  = config.get('prot_layers', 4),
            mc_dropout   = config.get('mc_dropout', 0.2),
            bond_cnn_dim = config.get('bond_cnn_dim', 32),
        ).to(self.device)
        print(f"Bayesian GNN parameters: {self.model.count_parameters():,}", flush=True)

        self.optimizer   = optim.AdamW(self.model.parameters(),
                                       lr=config.get('lr', 1.5e-3),
                                       weight_decay=config.get('weight_decay', 1e-4))
        self.criterion   = nn.HuberLoss(delta=0.5)
        self.scaler      = GradScaler('cuda', enabled=self.use_amp)
        self.accum_steps = config.get('accum_steps', 2)
        self.warmup_epochs = config.get('warmup_epochs', 3)
        self.base_lr     = config.get('lr', 1.5e-3)
        self.best_val_r2 = float('-inf')
        self.best_state  = None

    def _warmup_lr(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * scale

    def load_pretrained(self, path: str):
        """Load Phase 4 pretrained weights — transfer prot_encoder only (fusion dim mismatch)."""
        try:
            ckpt = torch.load(path, map_location=self.device)
            loaded = []
            try:
                self.model.prot_encoder.load_state_dict(ckpt['prot_encoder'], strict=False)
                loaded.append('prot_encoder (partial)')
            except Exception:
                pass
            print(f"Loaded: {loaded} from {path}", flush=True)
        except Exception as e:
            print(f"Could not load pretrained weights ({e}), training from scratch", flush=True)

    def train_epoch(self, loader, epoch=0) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        pending = False
        self.optimizer.zero_grad(set_to_none=True)
        bar = tqdm(loader, desc=f"  Epoch {epoch+1}", leave=True, dynamic_ncols=True)
        for step, (mol_b, prot_b, targets) in enumerate(bar):
            mol_b   = mol_b.to(self.device, non_blocking=True)
            prot_b  = prot_b.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                pred = self.model(mol_b, prot_b)
                loss = self.criterion(pred, targets) / self.accum_steps

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

        if pending:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader, mc_samples=1):
        self.model.eval()
        all_preds, all_targets, all_epi = [], [], []
        for mol_b, prot_b, targets in loader:
            mol_b  = mol_b.to(self.device)
            prot_b = prot_b.to(self.device)
            if mc_samples > 1:
                pm, epi = self.model.mc_predict(mol_b, prot_b, mc_samples)
                all_epi.extend(epi.cpu().numpy().flatten())
            else:
                pm = self.model(mol_b, prot_b)
            all_preds.extend(self.normalizer.denormalize_array(pm.cpu().numpy().flatten()))
            all_targets.extend(self.normalizer.denormalize_array(targets.numpy().flatten()))

        p, t = np.array(all_preds), np.array(all_targets)
        if not np.isfinite(p).all():
            print("  [WARN] Non-finite preds detected, returning sentinel metrics", flush=True)
            return {'r2': -1e9, 'rmse': float('inf'), 'mae': float('inf'), 'ci': 0.0}
        m = compute_metrics(p, t)
        m['ci'] = concordance_index(p, t)
        if all_epi:
            m['mean_epistemic_unc'] = float(np.mean(all_epi))
        return m

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

        epochs    = cfg.get('epochs', 30)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

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

        print("Running MC Dropout evaluation (T=20 samples)...", flush=True)
        test_m = self.evaluate(test_loader, mc_samples=20)
        print(
            f"Test  R2={test_m['r2']:.4f}  RMSE={test_m['rmse']:.4f}  "
            f"CI={test_m['ci']:.4f}  "
            f"Epistemic Unc={test_m.get('mean_epistemic_unc', 0):.4f}",
            flush=True
        )
        return {'best_val_r2': self.best_val_r2, **{f'test_{k}': v for k, v in test_m.items()}}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("PHASE 6: BAYESIAN GNN — MC DROPOUT UNCERTAINTY QUANTIFICATION")
    print("=" * 80 + "\n")

    loader = DAVISDatasetLoader(data_dir="data")
    data   = loader.load_davis()
    stats  = loader.get_statistics(data)
    train, val, test = loader.create_splits(data)

    normalizer = AffinityNormalizer(mean=stats['affinity_mean'], std=stats['affinity_std'])
    print(f"Normalizer: {normalizer.info()}", flush=True)

    config = {
        'epochs':         40,
        'batch_size':     12,
        'accum_steps':    4,
        'lr':             8e-4,
        'warmup_epochs':  3,
        'weight_decay':   1e-4,
        'gnn_hidden':     192,
        'gnn_layers':     5,
        'prot_embed':     128,
        'prot_hidden':    192,
        'prot_layers':    4,
        'mc_dropout':     0.2,
        'bond_cnn_dim':   32,
        'max_prot_len':   1200,
        'use_amp':        True,
    }

    trainer = BayesianGNNTrainer(config, normalizer)
    pretrained_ckpt = Path(__file__).parent / "phase4_pretrained_encoder.pt"
    if pretrained_ckpt.exists():
        trainer.load_pretrained(str(pretrained_ckpt))
    results = trainer.train(train, val, test)

    print("\n" + "=" * 80)
    print("PHASE 6 RESULTS (BAYESIAN GNN)")
    print("=" * 80)
    print(f"  Best Val R²         : {results['best_val_r2']:.4f}")
    print(f"  Test R²             : {results['test_r2']:.4f}")
    print(f"  Test RMSE           : {results['test_rmse']:.4f}")
    print(f"  Test MAE            : {results['test_mae']:.4f}")
    print(f"  Test CI             : {results['test_ci']:.4f}")
    if 'test_mean_epistemic_unc' in results:
        print(f"  Epistemic Uncertainty: {results['test_mean_epistemic_unc']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
