"""
PHASE 7: GNN ENSEMBLE — MULTIPLE ARCHITECTURES
================================================
Combines GCN, GAT, and GIN models into an ensemble.
Each model sees the same molecular graph but processes it differently,
so their errors are partially independent — ensemble averaging reduces variance.

Key GML concepts:
  - Heterogeneous ensemble: different GNN architectures (GCN, GAT, GIN)
  - Stochastic ensemble: same architecture, different random seeds
  - Weighted averaging: learn optimal combination weights
  - Ensemble uncertainty: std across member predictions
  - Model diversity: measured by pairwise prediction correlation
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
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from gml_core import (
    EnhancedDTAPredictor, build_dataloader, compute_metrics,
    concordance_index, ATOM_FEAT_DIM, PYG_AVAILABLE,
)
from phase3_real_data import DAVISDatasetLoader
from target_normalizer import AffinityNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# SINGLE MODEL TRAINER (shared across all ensemble members)
# ============================================================================
def train_single_model(gnn_type: str, config: dict, normalizer: AffinityNormalizer,
                       train_data, val_data, seed: int = 42):
    """Train one GNN ensemble member with AMP + warmup + gradient accumulation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = config.get('use_amp', True) and device.type == 'cuda'

    model  = EnhancedDTAPredictor(
        gnn_type     = gnn_type,
        mol_in_dim   = ATOM_FEAT_DIM,
        gnn_hidden   = config.get('gnn_hidden', 192),
        gnn_layers   = config.get('gnn_layers', 4),
        prot_embed   = config.get('prot_embed', 128),
        prot_hidden  = config.get('prot_hidden', 192),
        prot_layers  = config.get('prot_layers', 4),
        dropout      = config.get('dropout', 0.1),
        bond_cnn_dim = config.get('bond_cnn_dim', 32),
    ).to(device)

    # Warm-start: Phase 5 checkpoint has matching architecture (full transfer).
    # Fall back to Phase 4 prot_encoder-only transfer if Phase 5 not available.
    pretrained_path = config.get('pretrained_path', None)
    if pretrained_path and Path(pretrained_path).exists():
        try:
            ckpt = torch.load(pretrained_path, map_location=device)
            if 'model_state_dict' in ckpt and 'prot_encoder' not in ckpt:
                # Phase 5 checkpoint — full model weights, strip multi-task heads
                state = ckpt['model_state_dict']
                transfer_state = {}
                for k, v in state.items():
                    if k.startswith('affinity_head.'):
                        transfer_state[k.replace('affinity_head.', 'head.')] = v
                    elif not k.startswith('efficiency_head.') and not k.startswith('selectivity_head.') \
                         and not k.startswith('log_var'):
                        transfer_state[k] = v
                missing, unexpected = model.load_state_dict(transfer_state, strict=False)
                print(f"  [{gnn_type.upper()} s={seed}] Phase 5 warm-start: "
                      f"{len(transfer_state)} tensors, missing={len(missing)}", flush=True)
            else:
                # Phase 4 checkpoint — prot_encoder only
                try:
                    model.prot_encoder.load_state_dict(ckpt['prot_encoder'], strict=False)
                    print(f"  [{gnn_type.upper()} s={seed}] Loaded prot_encoder (partial)", flush=True)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [{gnn_type.upper()} s={seed}] Pretrained load skipped: {e}", flush=True)

    base_lr   = config.get('lr', 1.5e-3)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=config.get('weight_decay', 1e-4))
    criterion = nn.HuberLoss(delta=0.5)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.get('epochs', 20), eta_min=1e-6)
    scaler        = GradScaler('cuda', enabled=use_amp)
    accum_steps   = config.get('accum_steps', 2)
    warmup_epochs = config.get('warmup_epochs', 3)

    train_loader = build_dataloader(
        train_data, batch_size=config.get('batch_size', 16),
        shuffle=True,  normalizer=normalizer, balance=True,
        max_prot_len=config.get('max_prot_len', 1200),
    )
    val_loader = build_dataloader(
        val_data, batch_size=config.get('batch_size', 16),
        shuffle=False, normalizer=normalizer,
        max_prot_len=config.get('max_prot_len', 1200),
    )

    best_val_r2 = float('-inf')
    best_state  = None

    for epoch in range(config.get('epochs', 20)):
        # warmup
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * (epoch + 1) / warmup_epochs

        model.train()
        total_loss, n = 0.0, 0
        pending = False
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"  [{gnn_type.upper()} s={seed}] Ep {epoch+1}",
                   leave=True, dynamic_ncols=True)
        for step, (mol_b, prot_b, targets) in enumerate(bar):
            mol_b   = mol_b.to(device, non_blocking=True)
            prot_b  = prot_b.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast('cuda', enabled=use_amp):
                preds = model(mol_b, prot_b)
                loss  = criterion(preds, targets) / accum_steps

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                pending = False
                continue

            scaler.scale(loss).backward()
            pending = True

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = False

            total_loss += loss.item() * accum_steps * targets.size(0)
            n += targets.size(0)
            bar.set_postfix(loss=f"{total_loss/max(n,1):.4f}")

        if pending:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validation
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for mol_b, prot_b, targets in val_loader:
                mol_b  = mol_b.to(device)
                prot_b = prot_b.to(device)
                p = model(mol_b, prot_b).cpu().numpy().flatten()
                all_p.extend(normalizer.denormalize_array(p))
                all_t.extend(normalizer.denormalize_array(targets.numpy().flatten()))
        val_r2 = compute_metrics(np.array(all_p), np.array(all_t))['r2']

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state  = deepcopy(model.state_dict())

        print(f"  [{gnn_type.upper()} seed={seed}] Epoch {epoch+1:3d}/{config['epochs']} | Val R2={val_r2:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model.to(device), best_val_r2


# ============================================================================
# ENSEMBLE PREDICTOR
# ============================================================================
class GNNEnsemble(nn.Module):
    """
    Heterogeneous GNN ensemble with learnable weighting.
    Members: GCN, GAT, GIN (optionally repeated with different seeds).
    """

    def __init__(self, members: list):
        super().__init__()
        self.members = nn.ModuleList(members)
        n = len(members)
        # Learnable per-member weight (softmax-normalized at inference)
        self.weights = nn.Parameter(torch.ones(n) / n)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, mol_data, prot_ids, return_all=False):
        """
        Args:
            return_all: if True, return (weighted_mean, [member predictions])
        """
        preds = []
        for m in self.members:
            p = m(mol_data, prot_ids)     # [B, 1]
            preds.append(p)

        stack   = torch.stack(preds, dim=0)              # [M, B, 1]
        w       = torch.softmax(self.weights, dim=0)     # [M]
        w_view  = w.view(-1, 1, 1)
        weighted = (stack * w_view).sum(dim=0)           # [B, 1]

        if return_all:
            return weighted, [p.detach() for p in preds]
        return weighted

    def predict_with_uncertainty(self, mol_data, prot_ids):
        """Return (mean_pred, ensemble_std) — epistemic uncertainty proxy."""
        weighted, all_preds = self.forward(mol_data, prot_ids, return_all=True)
        if len(all_preds) > 1:
            stack = torch.stack(all_preds, dim=0)    # [M, B, 1]
            std   = stack.std(dim=0)                 # [B, 1]
        else:
            std = torch.zeros_like(weighted)
        return weighted, std

    def member_weights(self):
        w = torch.softmax(self.weights, dim=0).detach().cpu().numpy()
        return {f'member_{i}': float(w[i]) for i in range(len(w))}


# ============================================================================
# DIVERSITY METRICS
# ============================================================================
def ensemble_diversity(member_preds: list) -> dict:
    """Compute pairwise Pearson correlation between member predictions."""
    n = len(member_preds)
    if n < 2:
        return {}
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(np.corrcoef(member_preds[i], member_preds[j])[0, 1])
            corrs.append(c)
    return {
        'mean_pairwise_corr': float(np.mean(corrs)),
        'min_pairwise_corr':  float(np.min(corrs)),
        'diversity_score':    float(1.0 - np.mean(corrs)),
    }


# ============================================================================
# ENSEMBLE TRAINER
# ============================================================================
class EnsembleTrainer:
    def __init__(self, config: dict, normalizer: AffinityNormalizer):
        self.config     = config
        self.normalizer = normalizer
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_and_train(self, train_data, val_data, test_data) -> dict:
        cfg = self.config

        # Define ensemble members: (gnn_type, seed) pairs
        member_specs = [
            ('gcn', 42),
            ('gat', 42),
            ('gin', 42),
            ('gin', 123),   # second GIN with different seed
            ('gat', 7),     # second GAT with different seed
        ]

        print(f"Training {len(member_specs)} ensemble members...", flush=True)
        trained_members = []
        member_val_r2s  = []

        for gnn_type, seed in member_specs:
            print(f"\n--- Training {gnn_type.upper()} (seed={seed}) ---", flush=True)
            t0 = time.time()
            model, val_r2 = train_single_model(
                gnn_type, cfg, self.normalizer, train_data, val_data, seed
            )
            print(f"  Done in {time.time()-t0:.0f}s | Best Val R2={val_r2:.4f}", flush=True)
            trained_members.append(model)
            member_val_r2s.append(val_r2)

        # Assemble ensemble
        ensemble = GNNEnsemble(trained_members).to(self.device)
        print(f"\nEnsemble assembled. Member weights: {ensemble.member_weights()}", flush=True)

        # Evaluate on test set
        test_loader = build_dataloader(
            test_data, batch_size=cfg.get('batch_size', 16),
            shuffle=False, normalizer=self.normalizer,
            max_prot_len=cfg.get('max_prot_len', 1200),
        )

        ensemble.eval()
        all_preds, all_targets, all_stds = [], [], []
        member_preds_list = [[] for _ in trained_members]

        with torch.no_grad():
            for mol_b, prot_b, targets in tqdm(test_loader, desc="  Ensemble eval"):
                mol_b  = mol_b.to(self.device)
                prot_b = prot_b.to(self.device)
                weighted, ind_preds = ensemble(mol_b, prot_b, return_all=True)
                _, std = ensemble.predict_with_uncertainty(mol_b, prot_b)

                all_preds.extend(self.normalizer.denormalize_array(
                    weighted.cpu().numpy().flatten()))
                all_targets.extend(self.normalizer.denormalize_array(
                    targets.numpy().flatten()))
                all_stds.extend(std.cpu().numpy().flatten())

                for i, p in enumerate(ind_preds):
                    member_preds_list[i].extend(p.cpu().numpy().flatten())

        p_arr = np.array(all_preds)
        t_arr = np.array(all_targets)
        metrics = compute_metrics(p_arr, t_arr)
        metrics['ci']   = concordance_index(p_arr, t_arr)
        metrics['mean_ensemble_std'] = float(np.mean(all_stds))

        # Per-member metrics
        print("\n--- Per-Member Test Metrics ---", flush=True)
        for i, (gnn_type, seed) in enumerate(member_specs):
            m_preds = self.normalizer.denormalize_array(np.array(member_preds_list[i]))
            m_met   = compute_metrics(m_preds, t_arr)
            print(f"  {gnn_type.upper()} seed={seed}: R2={m_met['r2']:.4f}  RMSE={m_met['rmse']:.4f}", flush=True)

        # Diversity
        div = ensemble_diversity(member_preds_list)
        print(f"\nEnsemble diversity: {div}", flush=True)
        metrics.update(div)
        metrics['member_val_r2s'] = member_val_r2s

        # Save checkpoint
        checkpoint_path = Path(__file__).parent / 'models' / 'checkpoints' / 'phase7_ensemble_best_model.pth'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'ensemble_members': [m.state_dict() for m in trained_members],
            'member_specs': member_specs,
            'test_metrics': metrics,
            'config': self.config,
        }, checkpoint_path)
        print(f"Ensemble checkpoint saved to {checkpoint_path}", flush=True)

        return metrics


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("PHASE 7: GNN ENSEMBLE — GCN + GAT + GIN HETEROGENEOUS ENSEMBLE")
    print("=" * 80 + "\n")

    loader = DAVISDatasetLoader(data_dir="data")
    data   = loader.load_davis()
    stats  = loader.get_statistics(data)
    train, val, test = loader.create_splits(data)

    normalizer = AffinityNormalizer(mean=stats['affinity_mean'], std=stats['affinity_std'])
    print(f"Normalizer: {normalizer.info()}", flush=True)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}", flush=True)

    config = {
        'epochs':         25,
        'batch_size':     12,
        'accum_steps':    4,
        'lr':             8e-4,
        'warmup_epochs':  3,
        'weight_decay':   1e-4,
        'gnn_hidden':     192,
        'gnn_layers':     4,
        'prot_embed':     128,
        'prot_hidden':    192,
        'prot_layers':    4,
        'dropout':        0.1,
        'bond_cnn_dim':   32,
        'max_prot_len':   1200,
        'use_amp':        True,
        # Prefer Phase 5 checkpoint (same architecture = full transfer)
        'pretrained_path': str(
            Path(__file__).parent / "models" / "checkpoints" / "phase5_best_model.pth"
            if (Path(__file__).parent / "models" / "checkpoints" / "phase5_best_model.pth").exists()
            else Path(__file__).parent / "phase4_pretrained_encoder.pt"
        ),
    }

    trainer = EnsembleTrainer(config, normalizer)
    results = trainer.build_and_train(train, val, test)

    print("\n" + "=" * 80)
    print("PHASE 7 RESULTS (GNN ENSEMBLE)")
    print("=" * 80)
    print(f"  Test R²              : {results['r2']:.4f}")
    print(f"  Test RMSE            : {results['rmse']:.4f}")
    print(f"  Test MAE             : {results['mae']:.4f}")
    print(f"  Test CI              : {results['ci']:.4f}")
    print(f"  Ensemble Uncertainty : {results['mean_ensemble_std']:.4f}")
    print(f"  Diversity Score      : {results.get('diversity_score', 0):.4f}")
    print()
    print("  Member Val R²s:")
    specs = [('gcn',42),('gat',42),('gin',42),('gin',123),('gat',7)]
    for (gnn_type, seed), r2 in zip(specs, results['member_val_r2s']):
        print(f"    {gnn_type.upper()} seed={seed}: {r2:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
