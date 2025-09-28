"""
Training Utilities
Utility functions and classes for model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import time
import os
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LossFunction(Enum):
    """Enumeration of available loss functions."""
    MSE = "mse"
    MAE = "mae" 
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"
    FOCAL = "focal"
    QUANTILE = "quantile"

class OptimizerType(Enum):
    """Enumeration of available optimizers."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"

class SchedulerType(Enum):
    """Enumeration of available learning rate schedulers."""
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    COSINE_ANNEALING = "cosine_annealing"
    STEP_LR = "step_lr"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad_norm: Optional[float] = 1.0
    
    # Loss function
    loss_function: str = "mse"
    loss_params: Dict[str, Any] = None
    
    # Optimizer
    optimizer: str = "adamw"
    optimizer_params: Dict[str, Any] = None
    
    # Scheduler
    scheduler: str = "reduce_on_plateau"
    scheduler_params: Dict[str, Any] = None
    
    # Validation
    validation_freq: int = 1
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_freq: int = 10
    save_best_only: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"  # "min" or "max"
    
    # Mixed precision training
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.loss_params is None:
            self.loss_params = {}
        if self.optimizer_params is None:
            self.optimizer_params = {}
        if self.scheduler_params is None:
            self.scheduler_params = {}

class LossRegistry:
    """Registry for loss functions."""
    
    @staticmethod
    def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
        """Get loss function by name."""
        loss_type = loss_type.lower()
        
        if loss_type == "mse":
            return nn.MSELoss(**kwargs)
        elif loss_type == "mae":
            return nn.L1Loss(**kwargs)
        elif loss_type == "huber":
            return nn.HuberLoss(**kwargs)
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss(**kwargs)
        elif loss_type == "focal":
            return FocalLoss(**kwargs)
        elif loss_type == "quantile":
            return QuantileLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

class FocalLoss(nn.Module):
    """Focal Loss for regression tasks with imbalanced targets."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Ground truth targets [batch_size, 1]
            
        Returns:
            Focal loss value
        """
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class QuantileLoss(nn.Module):
    """Quantile Loss for uncertainty-aware regression."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9], reduction: str = "mean"):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            predictions: Model predictions [batch_size, num_quantiles]
            targets: Ground truth targets [batch_size, 1]
            
        Returns:
            Quantile loss value
        """
        batch_size = targets.size(0)
        targets = targets.expand(-1, len(self.quantiles))  # [batch_size, num_quantiles]
        
        errors = targets - predictions
        quantiles = torch.tensor(self.quantiles, device=predictions.device)
        
        loss = torch.max((quantiles - 1) * errors, quantiles * errors)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class OptimizerRegistry:
    """Registry for optimizers."""
    
    @staticmethod
    def get_optimizer(optimizer_type: str, parameters, learning_rate: float, **kwargs) -> torch.optim.Optimizer:
        """Get optimizer by name."""
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate, **kwargs)
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(parameters, lr=learning_rate, **kwargs)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(parameters, lr=learning_rate, **kwargs)
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop(parameters, lr=learning_rate, **kwargs)
        elif optimizer_type == "adagrad":
            return torch.optim.Adagrad(parameters, lr=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

class SchedulerRegistry:
    """Registry for learning rate schedulers."""
    
    @staticmethod
    def get_scheduler(scheduler_type: str, optimizer: torch.optim.Optimizer, **kwargs):
        """Get scheduler by name."""
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_type == "cosine_annealing":
            return CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler_type == "step_lr":
            return StepLR(optimizer, **kwargs)
        elif scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

class EarlyStopping:
    """Early stopping to stop training when a monitored metric stops improving."""
    
    def __init__(self, 
                 patience: int = 20,
                 min_delta: float = 1e-4,
                 mode: str = "min",
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for metrics like accuracy
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_value = float('inf')
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_value = float('-inf')
    
    def __call__(self, current_value: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current value of monitored metric
            model: Model to save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
            return True
        
        return False

class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(self,
                 filepath: str,
                 monitor: str = "val_loss",
                 mode: str = "min",
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 verbose: bool = True):
        """
        Initialize model checkpoint.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: "min" or "max"
            save_best_only: Whether to save only best models
            save_weights_only: Whether to save only weights or full model
            verbose: Whether to print save messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode.lower()
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def __call__(self, epoch: int, model: nn.Module, metrics: Dict[str, float]):
        """
        Save checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch
            model: Model to save
            metrics: Dictionary of metrics
        """
        current_value = metrics.get(self.monitor)
        if current_value is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics. Available: {list(metrics.keys())}")
            return
        
        should_save = not self.save_best_only
        
        if self.save_best_only:
            if self.mode == "min" and current_value < self.best_value:
                should_save = True
                self.best_value = current_value
            elif self.mode == "max" and current_value > self.best_value:
                should_save = True
                self.best_value = current_value
        
        if should_save:
            # Format filepath with epoch and metric values
            formatted_filepath = self.filepath.format(
                epoch=epoch,
                **{k: f"{v:.4f}" for k, v in metrics.items()}
            )
            
            if self.save_weights_only:
                torch.save(model.state_dict(), formatted_filepath)
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'best_value': self.best_value
                }
                torch.save(checkpoint, formatted_filepath)
            
            if self.verbose:
                logger.info(f"Checkpoint saved: {formatted_filepath}")

class GradientClipper:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: Optional[float] = None, norm_type: float = 2.0):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum norm of gradients
            norm_type: Type of norm to use
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, parameters) -> Optional[float]:
        """
        Clip gradients.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Total norm of parameters if clipping is applied
        """
        if self.max_norm is not None:
            return torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)
        return None

class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.history[key] = []
            
            self.metrics[key].append(value)
            self.history[key].append(value)
    
    def reset(self):
        """Reset current epoch metrics."""
        self.metrics = {}
    
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics for current epoch."""
        return {key: np.mean(values) for key, values in self.metrics.items()}
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full training history."""
        return self.history
    
    def save_history(self, filepath: str):
        """Save training history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, filepath: str):
        """Load training history from file."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: str = "auto") -> torch.device:
    """Get appropriate device for training."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    return device

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_config(model: nn.Module, filepath: str):
    """Save model configuration."""
    config = {
        'model_class': model.__class__.__name__,
        'parameters': count_parameters(model),
        'architecture': str(model)
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)

# Example usage
if __name__ == "__main__":
    print("Testing training utilities...")
    
    # Test loss functions
    print("\nTesting loss functions...")
    mse_loss = LossRegistry.get_loss_function("mse")
    focal_loss = LossRegistry.get_loss_function("focal", alpha=1.0, gamma=2.0)
    quantile_loss = LossRegistry.get_loss_function("quantile", quantiles=[0.1, 0.5, 0.9])
    
    # Test data
    predictions = torch.randn(10, 1)
    targets = torch.randn(10, 1)
    
    mse_value = mse_loss(predictions, targets)
    focal_value = focal_loss(predictions, targets)
    
    print(f"MSE Loss: {mse_value.item():.4f}")
    print(f"Focal Loss: {focal_value.item():.4f}")
    
    # Test quantile loss
    quantile_predictions = torch.randn(10, 3)  # 3 quantiles
    quantile_value = quantile_loss(quantile_predictions, targets)
    print(f"Quantile Loss: {quantile_value.item():.4f}")
    
    # Test training config
    print("\nTesting training config...")
    config = TrainingConfig(
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-3
    )
    
    print(f"Config: epochs={config.num_epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
    
    # Test early stopping
    print("\nTesting early stopping...")
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    # Simulate training with improving then worsening loss
    dummy_model = nn.Linear(10, 1)
    losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9]  # Should stop after loss starts increasing
    
    for epoch, loss in enumerate(losses):
        should_stop = early_stopping(loss, dummy_model)
        print(f"Epoch {epoch}: Loss={loss:.2f}, Should stop: {should_stop}")
        if should_stop:
            break
    
    # Test metrics tracker
    print("\nTesting metrics tracker...")
    tracker = MetricsTracker()
    
    for epoch in range(3):
        # Simulate epoch metrics
        tracker.update(train_loss=1.0 - epoch * 0.1, val_loss=0.9 - epoch * 0.1)
        averages = tracker.get_averages()
        print(f"Epoch {epoch}: {averages}")
        tracker.reset()
    
    print("Training utilities test completed successfully!")