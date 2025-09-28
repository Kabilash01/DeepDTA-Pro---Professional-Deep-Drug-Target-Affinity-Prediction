"""
Training framework for DeepDTA-Pro.
Includes trainers, utilities, and cross-dataset training capabilities.
"""

from .trainer import DeepDTATrainer
from .cross_dataset_trainer import CrossDatasetTrainer
from .training_utils import (
    TrainingConfig, LossRegistry, OptimizerRegistry, SchedulerRegistry,
    EarlyStopping, ModelCheckpoint, GradientClipper, MetricsTracker,
    FocalLoss, QuantileLoss, LossFunction, OptimizerType, SchedulerType,
    set_seed, get_device, count_parameters
)

__all__ = [
    # Main trainers
    'DeepDTATrainer',
    'CrossDatasetTrainer',
    
    # Configuration
    'TrainingConfig',
    
    # Registries
    'LossRegistry',
    'OptimizerRegistry', 
    'SchedulerRegistry',
    
    # Training components
    'EarlyStopping',
    'ModelCheckpoint',
    'GradientClipper',
    'MetricsTracker',
    
    # Loss functions
    'FocalLoss',
    'QuantileLoss',
    
    # Enums
    'LossFunction',
    'OptimizerType',
    'SchedulerType',
    
    # Utilities
    'set_seed',
    'get_device',
    'count_parameters'
]

# Version info
__version__ = "1.0.0"
__author__ = "DeepDTA-Pro Development Team"
__description__ = "Advanced training framework for drug-target binding affinity prediction"