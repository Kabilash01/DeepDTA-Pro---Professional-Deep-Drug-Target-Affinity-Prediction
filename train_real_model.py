"""
DeepDTA-Pro Model Training Script
Complete training pipeline for drug-target binding affinity prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import argparse
from pathlib import Path
import sys
import time
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from src.models.deepdta_pro import DeepDTAPro
    from src.data.molecular_features import MolecularFeatureExtractor
    from src.data.protein_features import ProteinFeatureExtractor
    from src.data.davis_loader import DavisDatasetLoader
    from src.utils.logger import Logger
    REAL_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Real models not available: {e}")
    print("📝 Using mock training for demonstration...")
    REAL_MODELS_AVAILABLE = False

class MockTrainingDataset:
    """Mock dataset for training demonstration"""
    def __init__(self, size=1000):
        self.size = size
        np.random.seed(42)
        
        # Generate realistic drug SMILES
        drug_templates = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen-like
            "CC(=O)OC1=CC=CC=C1C(=O)O",       # Aspirin-like
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine-like
            "CCCCCCCCCCCCCCCC(=O)O",          # Fatty acid-like
            "Nc1ncnc2c1ncn2C",                # Nucleotide-like
        ]
        
        protein_templates = [
            "MKKFFDSRREQGGSGLGSGSSGGGGSGGGYGNQDQSGGGGSGGGYGNQDQ",
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSL",
            "MGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVECLEQG",
        ]
        
        self.data = []
        for i in range(size):
            drug = np.random.choice(drug_templates) + f"C{i%10}"  # Add variation
            protein = np.random.choice(protein_templates) + "G" * (i%20)  # Add variation
            
            # Generate realistic binding affinity (lower = stronger binding)
            base_affinity = np.random.lognormal(mean=1.5, sigma=0.8)
            affinity = max(0.5, min(12.0, base_affinity))
            
            self.data.append({
                'drug_smiles': drug,
                'protein_sequence': protein,
                'binding_affinity': affinity
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DeepDTATrainer:
    """Complete training pipeline for DeepDTA-Pro"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize model
        if REAL_MODELS_AVAILABLE:
            self.model = DeepDTAPro(**config['model_params']).to(self.device)
            self.mol_extractor = MolecularFeatureExtractor()
            self.prot_extractor = ProteinFeatureExtractor()
        else:
            # Mock model for demonstration
            self.model = self._create_mock_model()
            self.mol_extractor = None
            self.prot_extractor = None
        
        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def _create_mock_model(self):
        """Create a simple mock model for demonstration"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(100, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1)
                )
            
            def forward(self, x):
                # Mock input processing
                batch_size = len(x) if isinstance(x, list) else 32
                device = next(self.fc.parameters()).device
                mock_features = torch.randn(batch_size, 100, device=device)
                return self.fc(mock_features)
                
        return MockModel().to(self.device)
    
    def load_dataset(self):
        """Load and prepare training dataset"""
        print("📊 Loading dataset...")
        
        if REAL_MODELS_AVAILABLE:
            try:
                # Try to load real DAVIS dataset
                loader = DavisDatasetLoader()
                data = loader.load_data()
                print(f"✅ Loaded {len(data)} real samples from DAVIS dataset")
            except Exception as e:
                print(f"⚠️  Failed to load real data: {e}")
                print("📝 Using mock dataset...")
                data = MockTrainingDataset(size=self.config['dataset_size'])
        else:
            data = MockTrainingDataset(size=self.config['dataset_size'])
            print(f"📝 Created mock dataset with {len(data)} samples")
        
        # Split dataset
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        test_size = len(data) - train_size - val_size
        
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        self.train_data = [data[i] for i in train_indices]
        self.val_data = [data[i] for i in val_indices]
        self.test_data = [data[i] for i in test_indices]
        
        print(f"📈 Dataset split: Train={len(self.train_data)}, Val={len(self.val_data)}, Test={len(self.test_data)}")
        
        return self.train_data, self.val_data, self.test_data
    
    def prepare_batch(self, batch_data):
        """Prepare batch for training"""
        if REAL_MODELS_AVAILABLE and self.mol_extractor and self.prot_extractor:
            # Real feature extraction
            mol_features = []
            prot_features = []
            targets = []
            
            for sample in batch_data:
                mol_feat = self.mol_extractor.extract_features(sample['drug_smiles'])
                prot_feat = self.prot_extractor.extract_features(sample['protein_sequence'])
                mol_features.append(mol_feat)
                prot_features.append(prot_feat)
                targets.append(sample['binding_affinity'])
            
            return mol_features, prot_features, torch.tensor(targets, dtype=torch.float32).to(self.device)
        else:
            # Mock feature preparation
            targets = torch.tensor([sample['binding_affinity'] for sample in batch_data], 
                                 dtype=torch.float32).to(self.device)
            return batch_data, None, targets
    
    def train_epoch(self, train_data):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Simple batching
        batch_size = self.config['batch_size']
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            # Prepare batch
            mol_features, prot_features, targets = self.prepare_batch(batch)
            
            # Forward pass
            if REAL_MODELS_AVAILABLE:
                predictions = self.model(mol_features, prot_features)
            else:
                predictions = self.model(mol_features).squeeze()
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_data):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            batch_size = self.config['batch_size']
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                
                mol_features, prot_features, batch_targets = self.prepare_batch(batch)
                
                if REAL_MODELS_AVAILABLE:
                    batch_preds = self.model(mol_features, prot_features)
                else:
                    batch_preds = self.model(mol_features).squeeze()
                
                loss = self.criterion(batch_preds, batch_targets)
                total_loss += loss.item()
                
                predictions.extend(batch_preds.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        avg_loss = total_loss / (len(val_data) // self.config['batch_size'])
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, mse, mae, r2, predictions, targets
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting training...")
        
        # Load dataset
        train_data, val_data, test_data = self.load_dataset()
        
        # Training loop
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_data)
            
            # Validate
            val_loss, val_mse, val_mae, val_r2, val_preds, val_targets = self.validate(val_data)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print()
        
        # Final test evaluation
        print("🧪 Final test evaluation...")
        test_loss, test_mse, test_mae, test_r2, test_preds, test_targets = self.validate(test_data)
        
        print(f"📊 Final Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        # Save results
        self.save_results(test_loss, test_mse, test_mae, test_r2, test_preds, test_targets)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    def save_results(self, test_loss, test_mse, test_mae, test_r2, predictions, targets):
        """Save training results and model"""
        output_dir = Path('outputs/training')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_metrics': {
                'loss': test_loss,
                'mse': test_mse,
                'mae': test_mae,
                'r2': test_r2
            }
        }, output_dir / f'deepdta_pro_model_{timestamp}.pth')
        
        # Save training metrics
        metrics = {
            'timestamp': timestamp,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_metrics': {
                'test_loss': test_loss,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
        }
        
        with open(output_dir / f'training_metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create training plots
        self.create_training_plots(output_dir, timestamp, predictions, targets)
        
        print(f"💾 Results saved to {output_dir}")
    
    def create_training_plots(self, output_dir, timestamp, predictions, targets):
        """Create training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training curves
        axes[0,0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0,0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_title('Training and Validation Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Prediction vs actual scatter plot
        axes[0,1].scatter(targets, predictions, alpha=0.6, color='green')
        axes[0,1].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
        axes[0,1].set_xlabel('Actual Binding Affinity')
        axes[0,1].set_ylabel('Predicted Binding Affinity')
        axes[0,1].set_title('Predictions vs Actual Values')
        axes[0,1].grid(True)
        
        # Residuals plot
        residuals = np.array(predictions) - np.array(targets)
        axes[1,0].scatter(predictions, residuals, alpha=0.6, color='purple')
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Predicted Values')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residual Plot')
        axes[1,0].grid(True)
        
        # Error distribution
        axes[1,1].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].set_xlabel('Residuals')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Residual Distribution')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'training_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved")

def main():
    parser = argparse.ArgumentParser(description='Train DeepDTA-Pro model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset_size', type=int, default=2000, help='Mock dataset size')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dataset_size': args.dataset_size,
        'model_params': {
            'molecular_node_features': 74,
            'molecular_edge_features': 12,
            'molecular_hidden_dim': 256,
            'molecular_num_layers': 5,
            'protein_vocab_size': 26,
            'protein_embedding_dim': 256,
            'protein_max_length': 1000,
            'fusion_hidden_dim': 512,
            'prediction_hidden_dims': [512, 256, 128],
            'dropout': 0.2
        }
    }
    
    print("🧬 DeepDTA-Pro Training Pipeline")
    print("=" * 50)
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Dataset Size: {config['dataset_size']}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = DeepDTATrainer(config)
    
    # Start training
    results = trainer.train()
    
    print("🎉 Training completed successfully!")
    print(f"Final Test R² Score: {results['test_r2']:.4f}")

if __name__ == "__main__":
    main()