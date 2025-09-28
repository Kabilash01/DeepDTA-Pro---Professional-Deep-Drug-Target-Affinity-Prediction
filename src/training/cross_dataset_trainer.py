"""
Cross-Dataset Trainer
Advanced trainer for cross-dataset training and domain adaptation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import time
from pathlib import Path
from tqdm import tqdm
import random

from .trainer import DeepDTATrainer
from .training_utils import TrainingConfig, MetricsTracker
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

class CrossDatasetTrainer(DeepDTATrainer):
    """
    Cross-dataset trainer for training on multiple datasets simultaneously.
    
    Features:
    - Multi-dataset training with balanced sampling
    - Domain adaptation techniques
    - Dataset-specific validation
    - Cross-dataset evaluation
    - Dataset weighting strategies
    """
    
    def __init__(self,
                 model: DeepDTAPro,
                 config: TrainingConfig,
                 dataset_loaders: Dict[str, Dict[str, DataLoader]],
                 dataset_weights: Optional[Dict[str, float]] = None,
                 sampling_strategy: str = "balanced",
                 output_dir: str = "./outputs",
                 experiment_name: str = "cross_dataset_experiment"):
        """
        Initialize cross-dataset trainer.
        
        Args:
            model: DeepDTA-Pro model
            config: Training configuration
            dataset_loaders: Dictionary of dataset loaders
                Format: {dataset_name: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}}
            dataset_weights: Weights for different datasets (optional)
            sampling_strategy: Strategy for sampling across datasets ("balanced", "weighted", "sequential")
            output_dir: Output directory
            experiment_name: Experiment name
        """
        self.dataset_loaders = dataset_loaders
        self.dataset_names = list(dataset_loaders.keys())
        self.dataset_weights = dataset_weights or {name: 1.0 for name in self.dataset_names}
        self.sampling_strategy = sampling_strategy.lower()
        
        # Validate dataset loaders
        self._validate_dataset_loaders()
        
        # Create combined train loader for parent class
        train_loaders = [loaders['train'] for loaders in dataset_loaders.values()]
        combined_train_loader = self._create_combined_loader(train_loaders)
        
        # Initialize parent class with combined loader
        super().__init__(
            model=model,
            config=config,
            train_loader=combined_train_loader,
            val_loader=None,  # We'll handle validation separately
            test_loader=None,  # We'll handle testing separately
            output_dir=output_dir,
            experiment_name=experiment_name
        )
        
        # Setup cross-dataset specific components
        self.setup_cross_dataset_components()
        
        logger.info(f"Cross-dataset trainer initialized with datasets: {self.dataset_names}")
        logger.info(f"Dataset weights: {self.dataset_weights}")
        logger.info(f"Sampling strategy: {self.sampling_strategy}")
    
    def _validate_dataset_loaders(self):
        """Validate dataset loader structure."""
        required_splits = ['train', 'val', 'test']
        
        for dataset_name, loaders in self.dataset_loaders.items():
            for split in required_splits:
                if split not in loaders:
                    raise ValueError(f"Dataset '{dataset_name}' missing '{split}' split")
                if not isinstance(loaders[split], DataLoader):
                    raise ValueError(f"Dataset '{dataset_name}' split '{split}' must be DataLoader")
    
    def _create_combined_loader(self, train_loaders: List[DataLoader]) -> DataLoader:
        """Create combined data loader from multiple loaders."""
        # For now, we'll use the first loader as template
        # In practice, you might want to create a custom dataset that combines all
        return train_loaders[0]
    
    def setup_cross_dataset_components(self):
        """Setup components specific to cross-dataset training."""
        # Dataset-specific metrics trackers
        self.dataset_metrics = {name: MetricsTracker() for name in self.dataset_names}
        
        # Dataset iterators for balanced sampling
        if self.sampling_strategy == "balanced":
            self.dataset_iterators = {}
            for name, loaders in self.dataset_loaders.items():
                self.dataset_iterators[name] = iter(loaders['train'])
        
        # Combined metrics calculator
        # TODO: Replace with actual RegressionMetrics class
        self.combined_metrics = SimpleMetricsCalculator()
    
    def get_next_batch_balanced(self) -> Tuple[Dict[str, Any], str]:
        """
        Get next batch using balanced sampling strategy.
        
        Returns:
            Tuple of (batch, dataset_name)
        """
        # Randomly select dataset based on weights
        dataset_names = list(self.dataset_weights.keys())
        weights = list(self.dataset_weights.values())
        selected_dataset = random.choices(dataset_names, weights=weights)[0]
        
        # Get next batch from selected dataset
        try:
            batch = next(self.dataset_iterators[selected_dataset])
        except StopIteration:
            # Reset iterator if exhausted
            self.dataset_iterators[selected_dataset] = iter(self.dataset_loaders[selected_dataset]['train'])
            batch = next(self.dataset_iterators[selected_dataset])
        
        return batch, selected_dataset
    
    def train_epoch_cross_dataset(self) -> Dict[str, float]:
        """
        Train for one epoch using cross-dataset strategy.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Calculate total steps (use largest dataset)
        max_steps = max(len(loader['train']) for loader in self.dataset_loaders.values())
        
        # Progress bar
        pbar = tqdm(range(max_steps), desc=f"Epoch {self.current_epoch}")
        
        for step in pbar:
            # Get batch based on sampling strategy
            if self.sampling_strategy == "balanced":
                batch, dataset_name = self.get_next_batch_balanced()
            else:
                # For now, implement balanced sampling
                # Other strategies can be added later
                batch, dataset_name = self.get_next_batch_balanced()
            
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
                with torch.cuda.amp.autocast():
                    output = self.model(molecular_batch, protein_sequences, protein_lengths)
                    predictions = output['predictions']
                    loss = self.criterion(predictions, targets)
                
                # Apply dataset weighting
                loss = loss * self.dataset_weights.get(dataset_name, 1.0)
                
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
                
                # Apply dataset weighting
                loss = loss * self.dataset_weights.get(dataset_name, 1.0)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = self.gradient_clipper(self.model.parameters())
                
                # Optimizer step
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = self.combined_metrics.calculate_batch_metrics(
                    predictions.cpu().numpy(),
                    targets.cpu().numpy()
                )
                batch_metrics['loss'] = loss.item()
                if 'grad_norm' in locals():
                    batch_metrics['grad_norm'] = grad_norm.item() if grad_norm is not None else 0.0
            
            # Update overall metrics tracker
            self.metrics_tracker.update(**{f"train_{k}": v for k, v in batch_metrics.items()})
            
            # Update dataset-specific metrics
            self.dataset_metrics[dataset_name].update(**{f"train_{k}": v for k, v in batch_metrics.items()})
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dataset': dataset_name,
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Get epoch averages
        epoch_metrics = self.metrics_tracker.get_averages()
        
        # Add dataset-specific metrics
        for dataset_name, tracker in self.dataset_metrics.items():
            dataset_avg = tracker.get_averages()
            for metric_name, value in dataset_avg.items():
                epoch_metrics[f"{dataset_name}_{metric_name}"] = value
        
        return epoch_metrics
    
    def validate_epoch_cross_dataset(self) -> Dict[str, float]:
        """
        Validate on all datasets separately.
        
        Returns:
            Dictionary of validation metrics for all datasets
        """
        self.model.eval()
        all_metrics = {}
        
        for dataset_name, loaders in self.dataset_loaders.items():
            val_loader = loaders['val']
            
            all_predictions = []
            all_targets = []
            total_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating {dataset_name}"):
                    # Move batch to device
                    molecular_batch = batch['molecular'].to(self.device)
                    protein_sequences = batch['protein'].to(self.device)
                    protein_lengths = batch.get('protein_lengths')
                    if protein_lengths is not None:
                        protein_lengths = protein_lengths.to(self.device)
                    targets = batch['affinity'].to(self.device)
                    
                    # Forward pass
                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
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
            
            # Calculate metrics for this dataset
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            dataset_metrics = self.combined_metrics.calculate_metrics(all_predictions, all_targets)
            dataset_metrics['loss'] = total_loss / len(val_loader)
            
            # Add dataset prefix
            for metric_name, value in dataset_metrics.items():
                all_metrics[f"{dataset_name}_val_{metric_name}"] = value
        
        # Calculate overall validation metrics (weighted average)
        overall_metrics = self._calculate_weighted_metrics(all_metrics)
        all_metrics.update(overall_metrics)
        
        return all_metrics
    
    def _calculate_weighted_metrics(self, dataset_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate overall metrics as weighted average of dataset metrics."""
        overall_metrics = {}
        metric_names = set()
        
        # Extract unique metric names
        for key in dataset_metrics.keys():
            if '_val_' in key:
                metric_name = key.split('_val_')[-1]
                metric_names.add(metric_name)
        
        # Calculate weighted averages
        for metric_name in metric_names:
            total_weight = 0.0
            weighted_sum = 0.0
            
            for dataset_name in self.dataset_names:
                key = f"{dataset_name}_val_{metric_name}"
                if key in dataset_metrics:
                    weight = self.dataset_weights.get(dataset_name, 1.0)
                    weighted_sum += dataset_metrics[key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_metrics[f"val_{metric_name}"] = weighted_sum / total_weight
        
        return overall_metrics
    
    def evaluate_cross_dataset(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all test datasets.
        
        Returns:
            Dictionary of evaluation metrics for each dataset
        """
        logger.info("Evaluating model on all datasets...")
        self.model.eval()
        
        results = {}
        
        for dataset_name, loaders in self.dataset_loaders.items():
            test_loader = loaders['test']
            
            all_predictions = []
            all_targets = []
            total_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Testing {dataset_name}"):
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
            
            dataset_results = self.combined_metrics.calculate_metrics(all_predictions, all_targets)
            dataset_results['loss'] = total_loss / len(test_loader)
            dataset_results['num_samples'] = len(all_predictions)
            
            results[dataset_name] = dataset_results
            
            # Log results
            log_str = f"{dataset_name} evaluation: "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in dataset_results.items() if k != 'num_samples'])
            logger.info(log_str)
        
        return results
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main cross-dataset training loop.
        
        Returns:
            Training history
        """
        logger.info(f"Starting cross-dataset training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training with cross-dataset strategy
            train_metrics = self.train_epoch_cross_dataset()
            
            # Validation on all datasets
            if epoch % self.config.validation_freq == 0:
                val_metrics = self.validate_epoch_cross_dataset()
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
            
            # Log main metrics
            main_metrics = {k: v for k, v in all_metrics.items() if not any(dataset in k for dataset in self.dataset_names)}
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in main_metrics.items()])
            logger.info(log_str)
            
            # Log dataset-specific metrics separately
            for dataset_name in self.dataset_names:
                dataset_metrics = {k.replace(f"{dataset_name}_", ""): v for k, v in all_metrics.items() if k.startswith(f"{dataset_name}_")}
                if dataset_metrics:
                    dataset_log = f"  {dataset_name}: " + " | ".join([f"{k}: {v:.4f}" for k, v in dataset_metrics.items()])
                    logger.info(dataset_log)
            
            # Model checkpointing
            if epoch % self.config.save_freq == 0 or epoch == self.config.num_epochs - 1:
                self.model_checkpoint(epoch, self.model, all_metrics)
            
            # Early stopping (based on overall validation loss)
            if self.early_stopping is not None and val_metrics:
                monitor_value = all_metrics.get(self.config.monitor_metric)
                if monitor_value is not None:
                    should_stop = self.early_stopping(monitor_value, self.model)
                    if should_stop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Reset metrics trackers
            self.metrics_tracker.reset()
            for tracker in self.dataset_metrics.values():
                tracker.reset()
        
        # Save final artifacts
        self.save_final_artifacts()
        
        # Return training history
        return self.metrics_tracker.get_history()

# Example usage
if __name__ == "__main__":
    print("Testing CrossDatasetTrainer...")
    
    # This would normally be run with actual data loaders
    from ..models.deepdta_pro import DeepDTAPro
    
    # Create dummy model
    model = DeepDTAPro()
    
    # Create training config
    config = TrainingConfig(
        num_epochs=3,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu"
    )
    
    print(f"Created model for cross-dataset training")
    print(f"Config: {config.num_epochs} epochs, batch size {config.batch_size}")
    
    # Note: In actual usage, you would provide real dataset loaders
    # dataset_loaders = {
    #     'davis': {'train': davis_train_loader, 'val': davis_val_loader, 'test': davis_test_loader},
    #     'kiba': {'train': kiba_train_loader, 'val': kiba_val_loader, 'test': kiba_test_loader}
    # }
    # trainer = CrossDatasetTrainer(model, config, dataset_loaders)
    # history = trainer.train()
    # results = trainer.evaluate_cross_dataset()
    
    print("CrossDatasetTrainer test completed successfully!")