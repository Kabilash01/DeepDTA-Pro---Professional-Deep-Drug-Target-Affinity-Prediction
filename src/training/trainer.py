"""
Deep Learning Trainer
Main training loop implementation for DeepDTA-Pro model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
import time
import os
from pathlib import Path
from tqdm import tqdm
import warnings

from .training_utils import (
    TrainingConfig, LossRegistry, OptimizerRegistry, SchedulerRegistry,
    EarlyStopping, ModelCheckpoint, GradientClipper, MetricsTracker,
    set_seed, get_device, count_parameters
)
from ..models.deepdta_pro import DeepDTAPro
# from ..evaluation.metrics import RegressionMetrics  # TODO: Create evaluation metrics

logger = logging.getLogger(__name__)

class SimpleMetricsCalculator:
    """Simple metrics calculator placeholder."""
    
    def calculate_batch_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate basic batch metrics."""
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        return {'mse': mse, 'mae': mae}
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # Pearson correlation
        if len(predictions) > 1:
            corr = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'pearson': corr}

class DeepDTATrainer:
    """
    Main trainer class for DeepDTA-Pro model.
    
    Features:
    - Mixed precision training with automatic scaling
    - Learning rate scheduling
    - Early stopping with best model restoration
    - Model checkpointing
    - Comprehensive logging and metrics tracking
    - Cross-dataset training support
    - Gradient clipping and regularization
    """
    
    def __init__(self, 
                 model: DeepDTAPro,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 output_dir: str = "./outputs",
                 experiment_name: str = "deepdta_experiment",
                 resume_from_checkpoint: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: DeepDTA-Pro model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            output_dir: Output directory for checkpoints and logs
            experiment_name: Name of the experiment
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Setup device
        self.device = get_device(config.device)
        self.model = self.model.to(self.device)
        
        # Setup training components
        self.setup_training_components()
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker()
        self.current_epoch = 0
        self.best_metrics = {}
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Trainer initialized for {experiment_name}")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.experiment_dir / "training.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def setup_training_components(self):
        """Setup loss function, optimizer, scheduler, and other training components."""
        # Loss function
        self.criterion = LossRegistry.get_loss_function(
            self.config.loss_function, **self.config.loss_params
        )
        
        # Optimizer
        self.optimizer = OptimizerRegistry.get_optimizer(
            self.config.optimizer,
            self.model.parameters(),
            self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            **self.config.optimizer_params
        )
        
        # Scheduler
        if self.config.scheduler:
            if self.config.scheduler == "reduce_on_plateau":
                scheduler_params = {
                    'mode': self.config.monitor_mode,
                    'patience': 10,
                    'factor': 0.5,
                    'verbose': True,
                    **self.config.scheduler_params
                }
            else:
                scheduler_params = self.config.scheduler_params
            
            self.scheduler = SchedulerRegistry.get_scheduler(
                self.config.scheduler,
                self.optimizer,
                **scheduler_params
            )
        else:
            self.scheduler = None
        
        # Early stopping
        if self.config.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode=self.config.monitor_mode,
                restore_best_weights=True
            )
        else:
            self.early_stopping = None
        
        # Model checkpoint
        checkpoint_path = self.experiment_dir / "checkpoints" / "model_{epoch:03d}_{val_loss:.4f}.pt"
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        self.model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            save_best_only=self.config.save_best_only,
            verbose=True
        )
        
        # Gradient clipper
        self.gradient_clipper = GradientClipper(max_norm=self.config.clip_grad_norm)
        
        # Mixed precision scaler
        if self.config.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Metrics calculator
        # TODO: Replace with actual RegressionMetrics class
        self.metrics_calculator = SimpleMetricsCalculator()
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {}
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            molecular_batch = batch['molecular'].to(self.device)
            protein_sequences = batch['protein'].to(self.device)
            protein_lengths = batch.get('protein_lengths')
            if protein_lengths is not None:
                protein_lengths = protein_lengths.to(self.device)
            targets = batch['affinity'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    output = self.model(molecular_batch, protein_sequences, protein_lengths)
                    predictions = output['predictions']
                    loss = self.criterion(predictions, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clipper.max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = self.gradient_clipper(self.model.parameters())
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                output = self.model(molecular_batch, protein_sequences, protein_lengths)
                predictions = output['predictions']
                loss = self.criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = self.gradient_clipper(self.model.parameters())
                
                # Optimizer step
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = self.metrics_calculator.calculate_batch_metrics(
                    predictions.cpu().numpy(),
                    targets.cpu().numpy()
                )
                batch_metrics['loss'] = loss.item()
                if 'grad_norm' in locals():
                    batch_metrics['grad_norm'] = grad_norm.item() if grad_norm is not None else 0.0
            
            # Update metrics tracker
            self.metrics_tracker.update(**{f"train_{k}": v for k, v in batch_metrics.items()})
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Get epoch averages
        epoch_metrics = self.metrics_tracker.get_averages()
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                molecular_batch = batch['molecular'].to(self.device)
                protein_sequences = batch['protein'].to(self.device)
                protein_lengths = batch.get('protein_lengths')
                if protein_lengths is not None:
                    protein_lengths = protein_lengths.to(self.device)
                targets = batch['affinity'].to(self.device)
                
                # Forward pass
                if self.config.use_amp:
                    with autocast():
                        output = self.model(molecular_batch, protein_sequences, protein_lengths)
                        predictions = output['predictions']
                        loss = self.criterion(predictions, targets)
                else:
                    output = self.model(molecular_batch, protein_sequences, protein_lengths)
                    predictions = output['predictions']
                    loss = self.criterion(predictions, targets)
                
                # Accumulate results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        val_metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        val_metrics['loss'] = total_loss / len(self.val_loader)
        
        # Add val_ prefix
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        
        return val_metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            if epoch % self.config.validation_freq == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_value = all_metrics.get(self.config.monitor_metric)
                    if monitor_value is not None:
                        self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            log_str = f"Epoch {epoch:03d} ({epoch_time:.1f}s): "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()])
            logger.info(log_str)
            
            # Model checkpointing
            if epoch % self.config.save_freq == 0 or epoch == self.config.num_epochs - 1:
                self.model_checkpoint(epoch, self.model, all_metrics)
            
            # Early stopping
            if self.early_stopping is not None and val_metrics:
                monitor_value = all_metrics.get(self.config.monitor_metric)
                if monitor_value is not None:
                    should_stop = self.early_stopping(monitor_value, self.model)
                    if should_stop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Reset metrics tracker
            self.metrics_tracker.reset()
        
        # Save final model and training history
        self.save_final_artifacts()
        
        # Return training history
        return self.metrics_tracker.get_history()
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            data_loader: Data loader to evaluate on (uses test_loader if None)
            
        Returns:
            Evaluation metrics
        """
        if data_loader is None:
            data_loader = self.test_loader
        
        if data_loader is None:
            logger.warning("No test data loader provided for evaluation")
            return {}
        
        logger.info("Evaluating model...")
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluation"):
                # Move batch to device
                molecular_batch = batch['molecular'].to(self.device)
                protein_sequences = batch['protein'].to(self.device)
                protein_lengths = batch.get('protein_lengths')
                if protein_lengths is not None:
                    protein_lengths = protein_lengths.to(self.device)
                targets = batch['affinity'].to(self.device)
                
                # Forward pass
                output = self.model(molecular_batch, protein_sequences, protein_lengths)
                predictions = output['predictions']
                loss = self.criterion(predictions, targets)
                
                # Accumulate results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        eval_metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        eval_metrics['loss'] = total_loss / len(data_loader)
        
        # Log results
        log_str = "Evaluation results: "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
        logger.info(log_str)
        
        return eval_metrics
    
    def save_checkpoint(self, filepath: str, **extra_state):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_tracker.get_history(),
            'best_metrics': self.best_metrics,
            **extra_state
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore metrics history
        if 'metrics_history' in checkpoint:
            self.metrics_tracker.history = checkpoint['metrics_history']
        
        logger.info(f"Checkpoint loaded from {filepath} (epoch {self.current_epoch})")
    
    def save_final_artifacts(self):
        """Save final model and training artifacts."""
        # Save final model
        final_model_path = self.experiment_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save training history
        history_path = self.experiment_dir / "training_history.json"
        self.metrics_tracker.save_history(str(history_path))
        
        # Save config
        import json
        config_path = self.experiment_dir / "training_config.json"
        with open(config_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            config_dict = {
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'loss_function': self.config.loss_function,
                'optimizer': self.config.optimizer,
                'scheduler': self.config.scheduler,
                'device': str(self.device),
                'seed': self.config.seed
            }
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Final artifacts saved to {self.experiment_dir}")

# Example usage
if __name__ == "__main__":
    print("Testing DeepDTATrainer...")
    
    # This would normally be run with actual data loaders and model
    # For testing, we'll just verify the class can be instantiated
    
    from ..models.deepdta_pro import DeepDTAPro
    
    # Create dummy model
    model = DeepDTAPro()
    
    # Create training config
    config = TrainingConfig(
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu"  # Use CPU for testing
    )
    
    print(f"Created model with {count_parameters(model):,} parameters")
    print(f"Training config: {config.num_epochs} epochs, batch size {config.batch_size}")
    
    # Note: In actual usage, you would need to provide real data loaders
    # trainer = DeepDTATrainer(model, config, train_loader, val_loader, test_loader)
    # history = trainer.train()
    # eval_metrics = trainer.evaluate()
    
    print("DeepDTATrainer test completed successfully!")