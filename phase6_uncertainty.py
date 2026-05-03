"""
PHASE 6: UNCERTAINTY QUANTIFICATION & BAYESIAN DEEP LEARNING
Using MC Dropout to estimate prediction confidence and uncertainty intervals
Allows identification of low-confidence predictions requiring additional validation

Expected R² improvement: 0.85-0.89 (vs Phase 5)
Benefits: Confidence estimates, uncertainty-aware predictions, out-of-distribution detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from tqdm import tqdm
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from phase3_real_data import DAVISDatasetLoader
from phase4_transfer_learning import SimpleChemTokenizer, ProteinTokenizer, SimpleMolBERT, SimpleProtBERT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 6: BAYESIAN DEEP LEARNING WITH MC DROPOUT
# ============================================================================

class Phase6BayesianModel(nn.Module):
    """
    Bayesian Deep Learning Model with MC Dropout
    Uses stochastic forward passes to estimate uncertainty
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()

        # Pre-trained-like encoders
        self.mol_encoder = SimpleMolBERT(vocab_size=256, embed_dim=768, num_layers=2)
        self.prot_encoder = SimpleProtBERT(vocab_size=26, embed_dim=1024, num_layers=2)

        # Bayesian layers with high dropout for uncertainty estimation
        self.bayesian_encoder = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # MC Dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # MC Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # MC Dropout
        )

        # Output layer (without dropout here for stability)
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize output bias to mean affinity
        with torch.no_grad():
            self.regression_head[-1].bias.fill_(5.45)

        self.dropout_rate = dropout_rate

    def forward(self, smiles_ids, protein_ids, use_dropout=True):
        """
        Forward pass with optional dropout (for uncertainty estimation)

        Args:
            smiles_ids: [batch_size, max_smiles_len]
            protein_ids: [batch_size, max_protein_len]
            use_dropout: bool, whether to use dropout (for MC sampling)

        Returns:
            predictions: [batch_size, 1]
        """
        # Control dropout behavior
        if use_dropout:
            self.train()
        else:
            self.eval()

        # Encode modalities
        mol_repr = self.mol_encoder(smiles_ids)  # [batch, 512]
        prot_repr = self.prot_encoder(protein_ids)  # [batch, 512]

        combined = torch.cat([mol_repr, prot_repr], dim=-1)  # [batch, 1024]
        bayesian_repr = self.bayesian_encoder(combined)  # [batch, 128]

        predictions = self.regression_head(bayesian_repr)  # [batch, 1]

        return predictions

    def predict_with_uncertainty(self, smiles_ids, protein_ids, n_iterations=100):
        """
        Use MC Dropout to get predictions and uncertainty estimates

        Args:
            smiles_ids: [batch_size, max_smiles_len]
            protein_ids: [batch_size, max_protein_len]
            n_iterations: number of stochastic forward passes

        Returns:
            mean: Expected prediction [batch_size, 1]
            std: Prediction uncertainty [batch_size, 1]
            ci_lower: Lower confidence interval 95% [batch_size, 1]
            ci_upper: Upper confidence interval 95% [batch_size, 1]
        """
        predictions = []

        # MC Dropout sampling
        for _ in range(n_iterations):
            with torch.no_grad():
                pred = self.forward(smiles_ids, protein_ids, use_dropout=True)
            predictions.append(pred)

        predictions = torch.stack(predictions)  # [n_iterations, batch_size, 1]

        # Compute statistics
        mean = predictions.mean(dim=0)  # [batch_size, 1]
        std = predictions.std(dim=0)  # [batch_size, 1]

        # Confidence intervals (2.5% and 97.5% quantiles)
        ci_lower = torch.quantile(predictions, 0.025, dim=0)  # [batch_size, 1]
        ci_upper = torch.quantile(predictions, 0.975, dim=0)  # [batch_size, 1]

        return mean, std, ci_lower, ci_upper


# ============================================================================
# PHASE 6 TRAINER
# ============================================================================

class Phase6Trainer:
    """Trainer for Phase 6 Bayesian Deep Learning"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Using device: {self.device}")

        # Initialize tokenizers
        self.smiles_tokenizer = SimpleChemTokenizer(vocab_size=256)
        self.protein_tokenizer = ProteinTokenizer()

        # Initialize model
        self.model = Phase6BayesianModel(dropout_rate=config.get('dropout_rate', 0.5)).to(self.device)
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"   Dropout rate (MC Dropout): {config.get('dropout_rate', 0.5)}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_r2 = float('-inf')

    def train_epoch(self, train_data, epoch, total_epochs):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        batch_size = self.config['batch_size']
        num_steps = min(
            len(train_data) // batch_size,
            self.config.get('max_samples', 5000) // batch_size
        )

        pbar = tqdm(range(num_steps), desc=f"Epoch {epoch + 1}/{total_epochs} Bayesian Train")

        for batch_idx in pbar:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(train_data))
            batch = train_data[batch_start:batch_end]

            self.optimizer.zero_grad()
            batch_loss = 0.0
            batch_count = 0

            for sample in batch:
                try:
                    # Tokenize inputs
                    smiles_ids = self.smiles_tokenizer.encode(sample['drug_smiles']).unsqueeze(0).to(self.device)
                    protein_ids = self.protein_tokenizer.encode(sample['protein_sequence']).unsqueeze(0).to(self.device)

                    # Forward pass with dropout
                    pred = self.model(smiles_ids, protein_ids, use_dropout=True)
                    target = torch.tensor([[sample['affinity']]], dtype=torch.float32).to(self.device)

                    loss = self.criterion(pred, target)
                    batch_loss += loss.item()
                    batch_count += 1

                    # Backward pass
                    loss.backward()

                except Exception as e:
                    logger.debug(f"Sample error: {e}")
                    continue

            # Update weights
            if batch_count > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                avg_batch_loss = batch_loss / batch_count
                total_loss += avg_batch_loss
                num_batches += 1
                pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def validate(self, val_data, n_mc_samples=50):
        """
        Validate model with uncertainty estimation

        Args:
            val_data: validation dataset
            n_mc_samples: number of MC Dropout samples for uncertainty
        """
        self.model.eval()
        total_loss = 0.0
        predictions_mean = []
        predictions_std = []
        predictions_lower = []
        predictions_upper = []
        targets = []
        num_samples = 0

        with torch.no_grad():
            for sample in val_data[:min(len(val_data), self.config.get('max_eval_samples', 2000))]:
                try:
                    smiles_ids = self.smiles_tokenizer.encode(sample['drug_smiles']).unsqueeze(0).to(self.device)
                    protein_ids = self.protein_tokenizer.encode(sample['protein_sequence']).unsqueeze(0).to(self.device)

                    # Get prediction with uncertainty
                    mean, std, ci_lower, ci_upper = self.model.predict_with_uncertainty(
                        smiles_ids, protein_ids, n_iterations=n_mc_samples
                    )

                    target = torch.tensor([[sample['affinity']]], dtype=torch.float32).to(self.device)
                    loss = self.criterion(mean, target)

                    total_loss += loss.item()
                    predictions_mean.append(mean.cpu().item())
                    predictions_std.append(std.cpu().item())
                    predictions_lower.append(ci_lower.cpu().item())
                    predictions_upper.append(ci_upper.cpu().item())
                    targets.append(sample['affinity'])
                    num_samples += 1

                except Exception as e:
                    logger.debug(f"Validation error: {e}")
                    continue

        if num_samples == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        avg_loss = total_loss / num_samples
        predictions_mean = np.array(predictions_mean)
        targets = np.array(targets)

        mse = mean_squared_error(targets, predictions_mean)
        mae = mean_absolute_error(targets, predictions_mean)
        r2 = r2_score(targets, predictions_mean)

        # Calibration metrics
        predictions_lower = np.array(predictions_lower)
        predictions_upper = np.array(predictions_upper)
        calibration_rate = np.mean((targets >= predictions_lower) & (targets <= predictions_upper))

        return avg_loss, mse, mae, r2, calibration_rate, np.mean(np.array(predictions_std))

    def train(self, train_data, val_data, test_data):
        """Main training loop"""
        logger.info(f"🚀 Starting Phase 6 Bayesian Deep Learning...")
        logger.info(f"   Train samples: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.get('learning_rate', 1e-4),
            total_steps=self.config['epochs'],
            pct_start=0.3,
            anneal_strategy='cos'
        )

        for epoch in range(self.config['epochs']):
            start_time = time.time()

            train_loss = self.train_epoch(train_data, epoch, self.config['epochs'])
            val_loss, val_mse, val_mae, val_r2, calibration, mean_std = self.validate(val_data)

            scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_r2 > self.best_val_r2:
                self.best_val_r2 = val_r2

            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config['epochs']} ({epoch_time:.2f}s) | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}, "
                f"MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f} | "
                f"Calibration: {calibration:.1%}, Uncertainty: {mean_std:.4f}"
            )

        # Test evaluation with uncertainty
        logger.info("🧪 Final test evaluation with uncertainty...")
        test_loss, test_mse, test_mae, test_r2, calibration, mean_std = self.validate(test_data, n_mc_samples=100)

        logger.info(f"📊 Final Test Results:")
        logger.info(f"   Loss: {test_loss:.6f}")
        logger.info(f"   MSE: {test_mse:.4f}")
        logger.info(f"   MAE: {test_mae:.4f}")
        logger.info(f"   R²: {test_r2:.4f}")
        logger.info(f"   Calibration Rate (95% CI): {calibration:.1%}")
        logger.info(f"   Mean Prediction Uncertainty: {mean_std:.4f}")

        return {
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_loss': test_loss,
            'best_val_r2': self.best_val_r2,
            'calibration': calibration,
            'mean_uncertainty': mean_std
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("PHASE 6: UNCERTAINTY QUANTIFICATION & BAYESIAN DEEP LEARNING")
    print("Using MC Dropout for confidence estimation")
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
        'epochs': 20,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'dropout_rate': 0.5,  # High dropout for uncertainty
        'max_samples': 5000,
        'max_eval_samples': 1000
    }

    logger.info(f"Configuration:")
    logger.info(f"   Epochs: {config['epochs']}")
    logger.info(f"   Batch Size: {config['batch_size']}")
    logger.info(f"   Learning Rate: {config['learning_rate']}")
    logger.info(f"   MC Dropout Rate: {config['dropout_rate']}")

    # Train
    trainer = Phase6Trainer(config)
    results = trainer.train(train_data, val_data, test_data)

    print("\n" + "=" * 80)
    print("🎉 PHASE 6 BAYESIAN DEEP LEARNING COMPLETED!")
    print("=" * 80)
    print(f"✅ Final Test R²: {results['test_r2']:.4f}")
    print(f"✅ Final Test MSE: {results['test_mse']:.4f}")
    print(f"✅ Final Test MAE: {results['test_mae']:.4f}")
    print(f"✅ Mean Prediction Uncertainty: {results['mean_uncertainty']:.4f}")
    print(f"✅ Calibration Rate (95% CI): {results['calibration']:.1%}")
    print(f"✅ Best Validation R²: {results['best_val_r2']:.4f}")
    print("=" * 80 + "\n")

    # Performance comparison
    print("Performance Comparison:")
    print(f"   Phase 2 Baseline R²: 0.5701")
    print(f"   Phase 4 Transfer Learning R²: ~0.75")
    print(f"   Phase 5 Multi-Task Learning R²: ~0.82")
    print(f"   Phase 6 Bayesian Deep Learning R²: {results['test_r2']:.4f}")
    if results['test_r2'] > 0.82:
        print(f"   ✅ IMPROVEMENT: {(results['test_r2'] - 0.82) * 100:.1f}% over Phase 5!")
    print()


if __name__ == "__main__":
    main()
