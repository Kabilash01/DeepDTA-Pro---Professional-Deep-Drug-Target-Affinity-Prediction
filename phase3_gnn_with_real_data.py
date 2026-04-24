"""
PHASE 3: GNN TRAINING WITH REAL DAVIS/KIBA DATA
Full pipeline using Graph Neural Networks on production datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from pathlib import Path
import time
from tqdm import tqdm
import sys

# Import Phase 3 GNN components and data loader
sys.path.insert(0, str(Path(__file__).parent))
from phase3_gnn_training import (
    MolecularGraphBuilder,
    GNNMolecularEncoder,
    SimpleProteinTransformer,
    Phase3GNNModel
)
from phase3_real_data import DAVISDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 3 TRAINER WITH REAL DATA
# ============================================================================

class Phase3RealDataTrainer:
    """Trainer for Phase 3 GNN with DAVIS/KIBA real datasets"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Using device: {self.device}")

        self.model = Phase3GNNModel(gnn_hidden_dim=128, prot_embed_dim=128).to(self.device)
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        self.criterion = nn.MSELoss()
        self.graph_builder = MolecularGraphBuilder()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_r2 = float('-inf')

    def _sequence_to_ids(self, sequence: str, max_len: int = 100) -> torch.Tensor:
        """Convert protein sequence to token IDs"""
        aa_to_id = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
            'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 0
        }

        sequence = sequence.upper()[:max_len].ljust(max_len, 'X')
        ids = [aa_to_id.get(aa, 0) for aa in sequence]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

    def train_epoch(self, train_data, epoch, total_epochs):
        """Train one epoch on real data"""
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        pbar = tqdm(range(0, min(len(train_data), self.config.get('max_samples', 5000)),
                         self.config['batch_size']),
                    desc=f"Epoch {epoch + 1}/{total_epochs} Train")

        for batch_start in pbar:
            batch_end = min(batch_start + self.config['batch_size'], len(train_data))
            batch = train_data[batch_start:batch_end]

            self.optimizer.zero_grad()
            batch_loss_sum = 0.0
            batch_samples = 0

            for sample in batch:
                try:
                    # Build molecular graph
                    mol_graph = self.graph_builder.smiles_to_graph(sample['drug_smiles'])
                    x, edge_index = mol_graph
                    x, edge_index = x.to(self.device), edge_index.to(self.device)

                    # Convert protein to IDs
                    prot_ids = self._sequence_to_ids(sample['protein_sequence']).to(self.device)

                    # Forward pass
                    pred = self.model((x, edge_index), prot_ids)
                    target = torch.tensor([[sample['affinity']]], dtype=torch.float32).to(self.device)

                    loss = self.criterion(pred, target)
                    batch_loss_sum += loss.item()
                    batch_samples += 1

                    # Backward pass (gradient accumulation)
                    loss.backward()

                except Exception as e:
                    logger.debug(f"Sample error: {e}")
                    continue

            # Update weights
            if batch_samples > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss_avg = batch_loss_sum / batch_samples
                total_loss += batch_loss_avg
                num_samples += batch_samples
                pbar.set_postfix({'loss': f'{batch_loss_avg:.4f}'})

        avg_loss = total_loss / max(1, num_samples)
        return avg_loss

    def validate(self, val_data):
        """Validate on real data"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for sample in val_data[:min(len(val_data), self.config.get('max_eval_samples', 2000))]:
                try:
                    mol_graph = self.graph_builder.smiles_to_graph(sample['drug_smiles'])
                    x, edge_index = mol_graph
                    x, edge_index = x.to(self.device), edge_index.to(self.device)

                    prot_ids = self._sequence_to_ids(sample['protein_sequence']).to(self.device)

                    pred = self.model((x, edge_index), prot_ids)
                    target = torch.tensor([[sample['affinity']]], dtype=torch.float32).to(self.device)

                    loss = self.criterion(pred, target)
                    total_loss += loss.item()
                    predictions.append(pred.cpu().item())
                    targets.append(sample['affinity'])

                except Exception as e:
                    logger.debug(f"Validation error: {e}")
                    continue

        if len(predictions) == 0:
            return 0.0, 0.0, 0.0, 0.0

        avg_loss = total_loss / len(predictions)
        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        return avg_loss, mse, mae, r2

    def train(self, train_data, val_data, test_data):
        """Main training loop"""
        logger.info(f"🚀 Starting Phase 3 GNN Training with Real Data...")
        logger.info(f"   Train samples: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=self.config['epochs'],
            pct_start=0.3,
            anneal_strategy='cos'
        )

        for epoch in range(self.config['epochs']):
            start_time = time.time()

            train_loss = self.train_epoch(train_data, epoch, self.config['epochs'])
            val_loss, val_mse, val_mae, val_r2 = self.validate(val_data)

            scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_r2 = val_r2

            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config['epochs']} ({epoch_time:.2f}s) | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}, "
                f"MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}"
            )

        # Test evaluation
        logger.info("🧪 Final test evaluation...")
        test_loss, test_mse, test_mae, test_r2 = self.validate(test_data)

        logger.info(f"📊 Final Test Results:")
        logger.info(f"   Loss: {test_loss:.6f}")
        logger.info(f"   MSE: {test_mse:.4f}")
        logger.info(f"   MAE: {test_mae:.4f}")
        logger.info(f"   R²: {test_r2:.4f}")

        return {
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_loss': test_loss,
            'best_val_r2': self.best_r2
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("PHASE 3: GRAPH NEURAL NETWORK TRAINING WITH REAL DAVIS DATA")
    print("=" * 80 + "\n")

    # Load DAVIS dataset
    logger.info("🔍 Loading DAVIS dataset...")
    loader = DAVISDatasetLoader(data_dir="data")
    davis_data = loader.load_davis()

    if not davis_data:
        logger.error("Failed to load DAVIS dataset")
        return

    logger.info(f"✅ Loaded {len(davis_data)} valid samples")

    # Create splits
    train_data, val_data, test_data = loader.create_splits(davis_data)

    # Print statistics
    stats = loader.get_statistics(davis_data)
    logger.info(f"Dataset Statistics:")
    logger.info(f"   Affinity: {stats['affinity_min']:.2f} - {stats['affinity_max']:.2f}")
    logger.info(f"   Mean ± Std: {stats['affinity_mean']:.4f} ± {stats['affinity_std']:.4f}")

    # Configuration
    config = {
        'epochs': 10,  # Reduced for testing
        'batch_size': 8,  # Reasonable batch size for memory
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'max_samples': 3000,  # Limit for faster training
        'max_eval_samples': 1000  # Limit validation samples
    }

    logger.info(f"Configuration:")
    logger.info(f"   Epochs: {config['epochs']}")
    logger.info(f"   Batch Size: {config['batch_size']}")
    logger.info(f"   Learning Rate: {config['learning_rate']}")
    logger.info(f"   Max train samples: {config['max_samples']}")

    # Train
    trainer = Phase3RealDataTrainer(config)
    results = trainer.train(train_data, val_data, test_data)

    print("\n" + "=" * 80)
    print("🎉 PHASE 3 REAL DATA TRAINING COMPLETED!")
    print("=" * 80)
    print(f"✅ Final Test R²: {results['test_r2']:.4f}")
    print(f"✅ Final Test MSE: {results['test_mse']:.4f}")
    print(f"✅ Final Test MAE: {results['test_mae']:.4f}")
    print(f"✅ Best Validation R²: {results['best_val_r2']:.4f}")
    print("=" * 80 + "\n")

    # Performance expectations
    print("Expected Performance vs Baseline (Phase 2):")
    print(f"   Phase 2 Baseline R²: 0.5701 (optimized training)")
    print(f"   Phase 3 Target R²: 0.75-0.80 (with real data)")
    print(f"   Phase 3 Achieved R²: {results['test_r2']:.4f}")
    if results['test_r2'] > 0.57:
        print("   ✅ Improvement achieved!")
    print()


if __name__ == "__main__":
    main()
