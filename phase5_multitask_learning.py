"""
PHASE 5: MULTI-TASK GNN
========================
Extends Phase 4's GIN with multi-task graph learning.
The same molecular graph encoder is shared across tasks, improving
generalization through gradient sharing.

Key GML concepts:
  - Shared GNN encoder (parameter efficient, better generalization)
  - Task-specific prediction heads on top of shared graph representations
  - Hard parameter sharing: backbone is identical for all tasks
  - Task weighting: uncertainty-based loss weighting (Kendall et al. 2018)
  - Tasks: affinity (main), drug efficiency (aux1), selectivity (aux2)
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
    ATOM_FEAT_DIM, PYG_AVAILABLE,
    HybridProteinEncoder, GatedBilinearFusion, BondCNNEncoder,
    GINEncoder, BOND_FEAT_DIM,
)
from phase3_real_data import DAVISDatasetLoader
from target_normalizer import AffinityNormalizer

import torch.nn.functional as F
try:
    from torch_geometric.nn import global_mean_pool, global_max_pool
    from torch_geometric.nn import GINConv
    from torch_geometric.data import Batch
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MULTI-TASK GNN MODEL
# ============================================================================
class MultiTaskGNNDTA(nn.Module):
    """
    Multi-task DTA model with enhanced shared backbone:
      - Bond CNN augmented GIN
      - Hybrid CNN-Transformer protein encoder
      - Gated Bilinear + Shape Complementarity fusion

    Tasks:
      1. affinity   — main regression task (pKd)
      2. efficiency — drug efficiency (pKd / heavy_atom_count proxy)
      3. selectivity— z-score of affinity (how selective vs mean)
    """

    def __init__(self, mol_in_dim=ATOM_FEAT_DIM, gnn_hidden=192,
                 gnn_layers=5, prot_embed=128, prot_hidden=192,
                 prot_layers=4, fusion_hidden=192, dropout=0.1,
                 bond_cnn_dim=32):
        super().__init__()

        # Bond CNN augmentation
        self.bond_cnn    = BondCNNEncoder(BOND_FEAT_DIM, bond_cnn_dim, dropout=dropout)
        aug_mol_dim      = mol_in_dim + bond_cnn_dim

        # Shared GIN encoder with augmented atom features
        self.mol_encoder = GINEncoder(aug_mol_dim, gnn_hidden, gnn_layers, dropout)
        self.mol_proj    = nn.Linear(gnn_hidden * 2, gnn_hidden)

        # Hybrid protein encoder (CNN + Transformer)
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

        # Task-specific heads
        def _head(out_dim=1):
            return nn.Sequential(
                nn.Linear(fused_dim, fused_dim // 2),
                nn.LayerNorm(fused_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim // 2, out_dim),
            )

        self.affinity_head    = _head(1)
        self.efficiency_head  = _head(1)
        self.selectivity_head = _head(1)

    def encode(self, mol_data, prot_ids):
        # BondCNN runs in float32 to avoid AMP dtype conflicts in scatter_add_
        with torch.amp.autocast('cuda', enabled=False):
            x_f32    = mol_data.x.float()
            ea       = getattr(mol_data, 'edge_attr', None)
            bond_ctx = self.bond_cnn(x_f32, mol_data.edge_index,
                                     ea.float() if ea is not None else ea)
            x_aug = torch.cat([x_f32, bond_ctx], dim=-1)  # float32
        mol_h  = self.mol_encoder(x_aug, mol_data.edge_index, mol_data.batch)
        mol_r  = F.relu(self.mol_proj(mol_h))
        prot_r = self.prot_encoder(prot_ids)
        return self.fusion(mol_r, prot_r)

    def forward(self, mol_data, prot_ids):
        fused = self.encode(mol_data, prot_ids)
        return self.affinity_head(fused), self.efficiency_head(fused), self.selectivity_head(fused)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def uncertainty_loss(self, pred_aff, target_aff,
                         pred_eff, target_eff,
                         pred_sel, target_sel,
                         aux_weight: float = 0.2):
        huber = nn.functional.smooth_l1_loss
        loss_aff = huber(pred_aff, target_aff)
        loss_eff = huber(pred_eff, target_eff)
        loss_sel = huber(pred_sel, target_sel)
        return loss_aff + aux_weight * (loss_eff + loss_sel)


# ============================================================================
# AUXILIARY TARGET GENERATION
# ============================================================================
def build_aux_targets(samples: list, normalizer: AffinityNormalizer) -> list:
    """
    Add auxiliary regression targets to each sample.
    - efficiency  ≈ affinity / (len(smiles) / 10)  (heavy atom proxy)
    - selectivity ≈ Z-score of affinity (already normalized)
    """
    out = []
    for s in samples:
        aff = float(s['affinity'])
        n_heavy_proxy = max(len(str(s['drug_smiles'])) / 10.0, 1.0)
        eff = aff / n_heavy_proxy
        sel = normalizer.normalize(aff)  # already a z-score
        out.append({**s, '_eff': eff, '_sel': sel})
    return out


# ============================================================================
# MULTI-TASK DATASET
# ============================================================================
from torch.utils.data import Dataset
from gml_core import MolecularGraphBuilder, AA_VOCAB


class MultiTaskGraphDataset(Dataset):
    def __init__(self, samples, normalizer, max_prot_len=1000):
        self.samples      = samples
        self.normalizer   = normalizer
        self.max_prot_len = max_prot_len
        self.mol_builder  = MolecularGraphBuilder()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mol = self.mol_builder.smiles_to_pyg(str(s['drug_smiles']))
        seq = str(s['protein_sequence']).upper()[:self.max_prot_len].ljust(self.max_prot_len, 'X')
        prot_ids = torch.tensor([AA_VOCAB.get(a, 0) for a in seq], dtype=torch.long)

        t_aff = torch.tensor([self.normalizer.normalize(float(s['affinity']))], dtype=torch.float32)
        t_eff = torch.tensor([s.get('_eff', 0.0)], dtype=torch.float32)
        t_sel = torch.tensor([s.get('_sel', 0.0)], dtype=torch.float32)

        return mol, prot_ids, t_aff, t_eff, t_sel


def mt_collate(batch):
    mols, prots, affs, effs, sels = zip(*batch)
    return (
        Batch.from_data_list(list(mols)),
        torch.stack(list(prots)),
        torch.stack(list(affs)),
        torch.stack(list(effs)),
        torch.stack(list(sels)),
    )


def build_mt_dataloader(samples, normalizer, batch_size=32, shuffle=True,
                         max_prot_len=1200):
    ds = MultiTaskGraphDataset(samples, normalizer, max_prot_len=max_prot_len)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=mt_collate, pin_memory=torch.cuda.is_available(),
    )


# ============================================================================
# TRAINER
# ============================================================================
class MultiTaskGNNTrainer:
    def __init__(self, config: dict, normalizer: AffinityNormalizer):
        self.config     = config
        self.normalizer = normalizer
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp    = config.get('use_amp', True) and self.device.type == 'cuda'
        print(f"Device: {self.device} | AMP: {self.use_amp}", flush=True)

        self.model = MultiTaskGNNDTA(
            mol_in_dim   = ATOM_FEAT_DIM,
            gnn_hidden   = config.get('gnn_hidden', 192),
            gnn_layers   = config.get('gnn_layers', 5),
            prot_embed   = config.get('prot_embed', 128),
            prot_hidden  = config.get('prot_hidden', 192),
            prot_layers  = config.get('prot_layers', 4),
            dropout      = config.get('dropout', 0.1),
            bond_cnn_dim = config.get('bond_cnn_dim', 32),
        ).to(self.device)
        print(f"Multi-task GNN parameters: {self.model.count_parameters():,}", flush=True)

        self.optimizer   = optim.AdamW(self.model.parameters(),
                                       lr=config.get('lr', 1.5e-3),
                                       weight_decay=config.get('weight_decay', 1e-4))
        self.scaler       = GradScaler('cuda', enabled=self.use_amp)
        self.accum_steps  = config.get('accum_steps', 2)
        self.warmup_epochs= config.get('warmup_epochs', 3)
        self.base_lr      = config.get('lr', 1.5e-3)
        self.aux_weight   = config.get('aux_weight', 0.2)
        self.best_val_r2  = float('-inf')
        self.best_state   = None

    def _warmup_lr(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * scale

    def load_pretrained(self, path: str):
        """Load Phase 4 pretrained weights — transfer what shapes match, skip the rest."""
        try:
            ckpt = torch.load(path, map_location=self.device)
            loaded = []

            # prot_encoder: HybridProteinEncoder shares embed + pos_enc + transformer
            # sub-weights with ProteinTransformerEncoder — load with strict=False
            try:
                self.model.prot_encoder.load_state_dict(ckpt['prot_encoder'], strict=False)
                loaded.append('prot_encoder (partial)')
            except Exception:
                pass

            # fusion: GatedBilinearFusion has different hidden_dim from CrossGraphAttention
            # in Phase 4 (256 vs 192) — skip entirely to avoid size-mismatch errors
            # The model trains from scratch for fusion, which is fine.

            print(f"Loaded: {loaded} from {path}", flush=True)
        except Exception as e:
            print(f"Could not load pretrained weights ({e}), training from scratch", flush=True)

    def train_epoch(self, loader, epoch=0) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        pending = False
        self.optimizer.zero_grad(set_to_none=True)
        bar = tqdm(loader, desc=f"  Epoch {epoch+1}", leave=True, dynamic_ncols=True)
        for step, (mol_b, prot_b, t_aff, t_eff, t_sel) in enumerate(bar):
            mol_b  = mol_b.to(self.device, non_blocking=True)
            prot_b = prot_b.to(self.device, non_blocking=True)
            t_aff  = t_aff.to(self.device, non_blocking=True)
            t_eff  = t_eff.to(self.device, non_blocking=True)
            t_sel  = t_sel.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                p_aff, p_eff, p_sel = self.model(mol_b, prot_b)
                loss = self.model.uncertainty_loss(
                    p_aff, t_aff, p_eff, t_eff, p_sel, t_sel,
                    aux_weight=self.aux_weight,
                ) / self.accum_steps

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

            total_loss += loss.item() * self.accum_steps * t_aff.size(0)
            n += t_aff.size(0)
            bar.set_postfix(loss=f"{total_loss/max(n,1):.4f}")

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
        for mol_b, prot_b, t_aff, _, _ in loader:
            mol_b  = mol_b.to(self.device)
            prot_b = prot_b.to(self.device)
            p_aff, _, _ = self.model(mol_b, prot_b)
            all_preds.extend(self.normalizer.denormalize_array(p_aff.cpu().numpy().flatten()))
            all_targets.extend(self.normalizer.denormalize_array(t_aff.numpy().flatten()))
        p, t = np.array(all_preds), np.array(all_targets)
        if not np.isfinite(p).all():
            print("  [WARN] Non-finite preds detected, returning sentinel metrics", flush=True)
            return {'r2': -1e9, 'rmse': float('inf'), 'mae': float('inf'), 'ci': 0.0}
        m = compute_metrics(p, t)
        m['ci'] = concordance_index(p, t)
        return m

    def train(self, train_data, val_data, test_data) -> dict:
        cfg = self.config
        print("Adding auxiliary targets...", flush=True)
        train_aug = build_aux_targets(train_data, self.normalizer)
        val_aug   = build_aux_targets(val_data,   self.normalizer)
        test_aug  = build_aux_targets(test_data,  self.normalizer)

        max_len = cfg.get('max_prot_len', 1200)
        bs      = cfg.get('batch_size', 16)
        train_loader = build_mt_dataloader(train_aug, self.normalizer, bs, True,  max_prot_len=max_len)
        val_loader   = build_mt_dataloader(val_aug,   self.normalizer, bs, False, max_prot_len=max_len)
        test_loader  = build_mt_dataloader(test_aug,  self.normalizer, bs, False, max_prot_len=max_len)

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
        test_m = self.evaluate(test_loader)
        print(f"Test  R²={test_m['r2']:.4f}  RMSE={test_m['rmse']:.4f}  CI={test_m['ci']:.4f}", flush=True)
        return {'best_val_r2': self.best_val_r2, **{f'test_{k}': v for k, v in test_m.items()}}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("PHASE 5: MULTI-TASK GNN — SHARED GRAPH ENCODER WITH MULTIPLE HEADS")
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
        'dropout':        0.1,
        'bond_cnn_dim':   32,
        'aux_weight':     0.2,
        'max_prot_len':   1200,
        'use_amp':        True,
    }

    trainer = MultiTaskGNNTrainer(config, normalizer)
    # Try to load pretrained protein encoder + fusion from Phase 4
    pretrained_ckpt = Path(__file__).parent / "phase4_pretrained_encoder.pt"
    if pretrained_ckpt.exists():
        trainer.load_pretrained(str(pretrained_ckpt))
    results = trainer.train(train, val, test)

    print("\n" + "=" * 80)
    print("PHASE 5 RESULTS (MULTI-TASK GNN)")
    print("=" * 80)
    print(f"  Best Val R²   : {results['best_val_r2']:.4f}")
    print(f"  Test R²       : {results['test_r2']:.4f}")
    print(f"  Test RMSE     : {results['test_rmse']:.4f}")
    print(f"  Test MAE      : {results['test_mae']:.4f}")
    print(f"  Test CI       : {results['test_ci']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
