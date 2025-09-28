# 📖 DeepDTA-Pro API Documentation

## Table of Contents

1. [Core Models](#core-models)
2. [Data Processing](#data-processing)
3. [Training Framework](#training-framework)
4. [Evaluation & Metrics](#evaluation--metrics)
5. [Interpretability](#interpretability)
6. [Web Interface](#web-interface)
7. [Utilities](#utilities)

---

## Core Models

### DeepDTAPro

Main model class implementing the complete drug-target binding affinity prediction architecture.

```python
class DeepDTAPro(nn.Module):
    """
    Advanced drug-target binding affinity prediction model using Graph Neural Networks.
    
    Combines molecular graph representations with protein sequence encodings through
    attention mechanisms for accurate and interpretable binding predictions.
    
    Args:
        molecular_config (dict): Configuration for molecular GNN
        protein_config (dict): Configuration for protein encoder
        fusion_config (dict): Configuration for fusion network
    
    Example:
        >>> config = {
        ...     'molecular': {'hidden_dim': 128, 'num_layers': 3},
        ...     'protein': {'embedding_dim': 128, 'hidden_dim': 256},
        ...     'fusion': {'hidden_dim': 512, 'dropout': 0.1}
        ... }
        >>> model = DeepDTAPro(config)
        >>> prediction = model(mol_data, protein_data)
    """
    
    def __init__(self, molecular_config, protein_config, fusion_config):
        """Initialize DeepDTA-Pro model with component configurations."""
        
    def forward(self, molecular_data, protein_data):
        """
        Forward pass through the complete model.
        
        Args:
            molecular_data (dict): Molecular graph data with 'x', 'edge_index', 'batch'
            protein_data (dict): Protein data with 'sequence', 'embeddings'
        
        Returns:
            torch.Tensor: Predicted binding affinity scores
        """
        
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location='cpu'):
        """
        Load model from saved checkpoint.
        
        Args:
            checkpoint_path (str): Path to saved model checkpoint
            map_location (str): Device to load model on
        
        Returns:
            DeepDTAPro: Loaded model instance
        """
        
    def get_attention_weights(self, molecular_data, protein_data):
        """
        Extract attention weights for interpretability.
        
        Args:
            molecular_data (dict): Molecular graph data
            protein_data (dict): Protein sequence data
        
        Returns:
            dict: Attention weights for different components
        """
```

### MolecularGNN

Graph Neural Network for processing molecular representations.

```python
class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular feature extraction.
    
    Uses graph convolutions to process molecular graphs with node features
    representing atomic properties and edge features representing bonds.
    
    Args:
        input_dim (int): Input node feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output embedding dimension
        num_layers (int): Number of graph convolution layers
        dropout (float): Dropout rate
        activation (str): Activation function ('relu', 'gelu', 'swish')
    
    Example:
        >>> gnn = MolecularGNN(
        ...     input_dim=78, hidden_dim=128, output_dim=256,
        ...     num_layers=3, dropout=0.1
        ... )
        >>> mol_embedding = gnn(mol_data)
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                 dropout=0.1, activation='relu'):
        """Initialize molecular GNN with specified architecture."""
        
    def forward(self, data):
        """
        Process molecular graph data.
        
        Args:
            data (dict): Graph data with 'x', 'edge_index', 'edge_attr', 'batch'
        
        Returns:
            torch.Tensor: Molecular graph embeddings
        """
```

### ProteinEncoder

Sequence encoder for protein representations.

```python
class ProteinEncoder(nn.Module):
    """
    Protein sequence encoder using bidirectional LSTM/Transformer.
    
    Processes protein sequences to extract meaningful representations
    with attention mechanisms for variable-length sequences.
    
    Args:
        vocab_size (int): Amino acid vocabulary size
        embedding_dim (int): Embedding dimension
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of encoder layers
        encoder_type (str): 'lstm' or 'transformer'
        dropout (float): Dropout rate
    
    Example:
        >>> encoder = ProteinEncoder(
        ...     vocab_size=25, embedding_dim=128, hidden_dim=256,
        ...     num_layers=2, encoder_type='lstm'
        ... )
        >>> protein_embedding = encoder(protein_data)
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_layers=2, encoder_type='lstm', dropout=0.1):
        """Initialize protein encoder with specified architecture."""
        
    def forward(self, data):
        """
        Process protein sequence data.
        
        Args:
            data (dict): Protein data with 'sequence', 'length'
        
        Returns:
            torch.Tensor: Protein sequence embeddings
        """
```

---

## Data Processing

### MolecularFeatureExtractor

Extracts features from molecular SMILES strings.

```python
class MolecularFeatureExtractor:
    """
    Extract molecular features from SMILES strings.
    
    Converts SMILES representations to graph-based molecular features
    including atomic properties, bond information, and structural descriptors.
    
    Args:
        max_atoms (int): Maximum number of atoms per molecule
        feature_config (dict): Configuration for feature extraction
    
    Example:
        >>> extractor = MolecularFeatureExtractor(max_atoms=100)
        >>> features = extractor.extract_features("CCO")  # Ethanol
        >>> print(f"Graph has {features['x'].shape[0]} atoms")
    """
    
    def __init__(self, max_atoms=100, feature_config=None):
        """Initialize molecular feature extractor."""
        
    def extract_features(self, smiles):
        """
        Extract molecular features from SMILES string.
        
        Args:
            smiles (str): SMILES representation of molecule
        
        Returns:
            dict: Molecular graph data with node/edge features
                - x (torch.Tensor): Node features [num_atoms, num_features]
                - edge_index (torch.LongTensor): Edge connectivity [2, num_edges]
                - edge_attr (torch.Tensor): Edge features [num_edges, num_edge_features]
                - smiles (str): Original SMILES string
        
        Raises:
            ValueError: If SMILES string is invalid
        """
        
    def batch_extract(self, smiles_list, batch_size=32):
        """
        Extract features for multiple molecules in batches.
        
        Args:
            smiles_list (List[str]): List of SMILES strings
            batch_size (int): Batch size for processing
        
        Returns:
            List[dict]: List of molecular feature dictionaries
        """
        
    def get_molecular_descriptors(self, smiles):
        """
        Calculate molecular descriptors for a given SMILES.
        
        Args:
            smiles (str): SMILES representation
        
        Returns:
            dict: Molecular descriptors (MW, LogP, TPSA, etc.)
        """
```

### ProteinFeatureExtractor

Extracts features from protein sequences.

```python
class ProteinFeatureExtractor:
    """
    Extract protein features from amino acid sequences.
    
    Processes protein sequences to create numerical representations
    suitable for deep learning models with various encoding schemes.
    
    Args:
        max_length (int): Maximum sequence length
        encoding_type (str): Encoding method ('onehot', 'learned', 'physicochemical')
        feature_config (dict): Configuration for feature extraction
    
    Example:
        >>> extractor = ProteinFeatureExtractor(max_length=1000, encoding_type='learned')
        >>> features = extractor.extract_features("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
    """
    
    def __init__(self, max_length=1000, encoding_type='learned', feature_config=None):
        """Initialize protein feature extractor."""
        
    def extract_features(self, sequence):
        """
        Extract protein features from amino acid sequence.
        
        Args:
            sequence (str): Protein amino acid sequence
        
        Returns:
            dict: Protein feature data
                - sequence (torch.LongTensor): Encoded sequence [max_length]
                - length (int): Actual sequence length
                - mask (torch.BoolTensor): Sequence mask [max_length]
                - physicochemical (torch.Tensor): Physicochemical properties
        
        Raises:
            ValueError: If sequence contains invalid amino acids
        """
        
    def batch_extract(self, sequences, batch_size=32):
        """
        Extract features for multiple proteins in batches.
        
        Args:
            sequences (List[str]): List of protein sequences
            batch_size (int): Batch size for processing
        
        Returns:
            List[dict]: List of protein feature dictionaries
        """
        
    def get_physicochemical_properties(self, sequence):
        """
        Calculate physicochemical properties for protein sequence.
        
        Args:
            sequence (str): Amino acid sequence
        
        Returns:
            dict: Physicochemical properties (MW, pI, hydrophobicity, etc.)
        """
```

### DatasetLoader

Base class for dataset loading and preprocessing.

```python
class DatasetLoader:
    """
    Base class for loading and preprocessing drug-target datasets.
    
    Provides common functionality for data loading, preprocessing,
    and train/validation/test splitting with stratification options.
    
    Args:
        data_path (str): Path to dataset files
        preprocessing_config (dict): Preprocessing configuration
        split_config (dict): Data splitting configuration
    
    Example:
        >>> loader = DatasetLoader(
        ...     data_path='data/davis/',
        ...     preprocessing_config={'normalize': True, 'remove_outliers': True},
        ...     split_config={'train': 0.8, 'val': 0.1, 'test': 0.1}
        ... )
        >>> train_data, val_data, test_data = loader.load_and_split()
    """
    
    def __init__(self, data_path, preprocessing_config=None, split_config=None):
        """Initialize dataset loader with configuration."""
        
    def load_raw_data(self):
        """
        Load raw dataset from files.
        
        Returns:
            dict: Raw dataset with drugs, proteins, and interactions
        """
        
    def preprocess_data(self, raw_data):
        """
        Preprocess loaded data.
        
        Args:
            raw_data (dict): Raw dataset
        
        Returns:
            dict: Preprocessed dataset
        """
        
    def create_splits(self, data, split_config):
        """
        Create train/validation/test splits.
        
        Args:
            data (dict): Preprocessed dataset
            split_config (dict): Split configuration
        
        Returns:
            tuple: (train_data, val_data, test_data)
        """
```

---

## Training Framework

### DeepDTATrainer

Main training class for model training and evaluation.

```python
class DeepDTATrainer:
    """
    Comprehensive trainer for DeepDTA-Pro models.
    
    Handles model training, validation, checkpointing, and evaluation
    with support for various optimization strategies and regularization techniques.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Training configuration
        logger (Logger): Logger instance
    
    Example:
        >>> trainer = DeepDTATrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=training_config,
        ...     logger=logger
        ... )
        >>> trainer.train()
    """
    
    def __init__(self, model, train_loader, val_loader, config, logger=None):
        """Initialize trainer with model and data loaders."""
        
    def train(self):
        """
        Execute complete training loop.
        
        Returns:
            dict: Training results and metrics
        """
        
    def train_epoch(self, epoch):
        """
        Train model for one epoch.
        
        Args:
            epoch (int): Current epoch number
        
        Returns:
            dict: Epoch training metrics
        """
        
    def validate_epoch(self, epoch):
        """
        Validate model for one epoch.
        
        Args:
            epoch (int): Current epoch number
        
        Returns:
            dict: Epoch validation metrics
        """
        
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
```

### CrossDatasetTrainer

Trainer for cross-dataset learning scenarios.

```python
class CrossDatasetTrainer(DeepDTATrainer):
    """
    Trainer for cross-dataset learning and domain adaptation.
    
    Extends base trainer to handle multiple datasets with domain
    adaptation techniques and cross-dataset evaluation.
    
    Args:
        model (nn.Module): Model to train
        source_loader (DataLoader): Source dataset loader
        target_loader (DataLoader): Target dataset loader
        config (dict): Training configuration
        adaptation_config (dict): Domain adaptation configuration
    
    Example:
        >>> trainer = CrossDatasetTrainer(
        ...     model=model,
        ...     source_loader=davis_loader,
        ...     target_loader=kiba_loader,
        ...     config=training_config,
        ...     adaptation_config=adaptation_config
        ... )
        >>> trainer.train_cross_dataset()
    """
    
    def __init__(self, model, source_loader, target_loader, config, adaptation_config):
        """Initialize cross-dataset trainer."""
        
    def train_cross_dataset(self):
        """
        Execute cross-dataset training.
        
        Returns:
            dict: Cross-dataset training results
        """
        
    def domain_adaptation_loss(self, source_features, target_features):
        """
        Calculate domain adaptation loss.
        
        Args:
            source_features (torch.Tensor): Source domain features
            target_features (torch.Tensor): Target domain features
        
        Returns:
            torch.Tensor: Domain adaptation loss
        """
```

---

## Evaluation & Metrics

### MetricsCalculator

Comprehensive metrics calculation for model evaluation.

```python
class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics for binding affinity prediction.
    
    Provides 20+ metrics including regression metrics, correlation measures,
    and classification-based metrics with statistical significance testing.
    
    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.calculate_all_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.3f}")
        >>> print(f"Pearson R: {metrics['pearson_r']:.3f}")
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        
    def calculate_all_metrics(self, y_true, y_pred, threshold=None):
        """
        Calculate all available metrics.
        
        Args:
            y_true (np.ndarray): True binding affinities
            y_pred (np.ndarray): Predicted binding affinities
            threshold (float): Threshold for classification metrics
        
        Returns:
            dict: Comprehensive metrics dictionary
                - rmse (float): Root Mean Square Error
                - mae (float): Mean Absolute Error
                - mse (float): Mean Square Error
                - pearson_r (float): Pearson correlation coefficient
                - pearson_p (float): Pearson correlation p-value
                - spearman_rho (float): Spearman correlation coefficient
                - spearman_p (float): Spearman correlation p-value
                - r2_score (float): R-squared score
                - explained_variance (float): Explained variance score
                - max_error (float): Maximum absolute error
                - mean_error (float): Mean error (bias)
                - ... (additional metrics)
        """
        
    def calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression-specific metrics."""
        
    def calculate_correlation_metrics(self, y_true, y_pred):
        """Calculate correlation-based metrics."""
        
    def calculate_classification_metrics(self, y_true, y_pred, threshold):
        """Calculate classification metrics using threshold."""
        
    def calculate_confidence_intervals(self, y_true, y_pred, confidence=0.95):
        """
        Calculate confidence intervals for metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            confidence (float): Confidence level
        
        Returns:
            dict: Confidence intervals for key metrics
        """
```

### CrossValidator

Cross-validation framework for robust model evaluation.

```python
class CrossValidator:
    """
    Comprehensive cross-validation framework.
    
    Supports multiple CV strategies including stratified k-fold,
    time series splits, and custom splitting strategies.
    
    Args:
        model_class (class): Model class to instantiate
        cv_strategy (str): Cross-validation strategy
        n_splits (int): Number of CV splits
        metrics (List[str]): Metrics to calculate
        random_state (int): Random seed for reproducibility
    
    Example:
        >>> cv = CrossValidator(
        ...     model_class=DeepDTAPro,
        ...     cv_strategy='stratified',
        ...     n_splits=5,
        ...     metrics=['rmse', 'mae', 'pearson']
        ... )
        >>> results = cv.run_cross_validation(X, y, model_config)
    """
    
    def __init__(self, model_class, cv_strategy='stratified', n_splits=5, 
                 metrics=None, random_state=42):
        """Initialize cross-validator."""
        
    def run_cross_validation(self, X_data, y_data, model_config):
        """
        Run complete cross-validation.
        
        Args:
            X_data: Input features
            y_data: Target values
            model_config (dict): Model configuration
        
        Returns:
            dict: Cross-validation results with statistics
        """
        
    def print_results_summary(self):
        """Print formatted summary of CV results."""
        
    def plot_cv_results(self, save_path=None):
        """
        Plot cross-validation results.
        
        Args:
            save_path (str): Path to save plot
        """
```

---

## Interpretability

### AttentionVisualizer

Visualize and analyze attention mechanisms.

```python
class AttentionVisualizer:
    """
    Visualize attention weights for model interpretability.
    
    Provides comprehensive visualization of attention patterns
    in molecular and protein components of the model.
    
    Args:
        model (nn.Module): Trained model with attention mechanisms
        device (str): Device for computation
    
    Example:
        >>> visualizer = AttentionVisualizer(model)
        >>> attention_data = visualizer.get_attention_weights(mol_data, prot_data)
        >>> visualizer.plot_molecular_attention(drug_smiles, attention_data)
    """
    
    def __init__(self, model, device='cpu'):
        """Initialize attention visualizer."""
        
    def get_attention_weights(self, molecular_data, protein_data):
        """
        Extract attention weights from model.
        
        Args:
            molecular_data (dict): Molecular graph data
            protein_data (dict): Protein sequence data
        
        Returns:
            dict: Attention weights for different components
                - molecular_attention (torch.Tensor): Molecular attention weights
                - protein_attention (torch.Tensor): Protein attention weights
                - cross_attention (torch.Tensor): Cross-modal attention weights
        """
        
    def visualize_molecular_attention(self, smiles, attention_weights, save_path=None):
        """
        Visualize molecular attention on 2D structure.
        
        Args:
            smiles (str): SMILES string
            attention_weights (torch.Tensor): Attention weights per atom
            save_path (str): Path to save visualization
        """
        
    def visualize_protein_attention(self, sequence, attention_weights, save_path=None):
        """
        Visualize protein attention on sequence.
        
        Args:
            sequence (str): Protein sequence
            attention_weights (torch.Tensor): Attention weights per residue
            save_path (str): Path to save visualization
        """
        
    def plot_attention_heatmap(self, attention_matrix, labels=None, save_path=None):
        """
        Plot attention as heatmap.
        
        Args:
            attention_matrix (torch.Tensor): Attention weight matrix
            labels (List[str]): Labels for axes
            save_path (str): Path to save plot
        """
```

### DeepSHAPAnalyzer

SHAP-based feature importance analysis.

```python
class DeepSHAPAnalyzer:
    """
    Deep SHAP analysis for feature importance and model explanation.
    
    Provides SHAP-based explanations for individual predictions
    and global feature importance analysis.
    
    Args:
        model (nn.Module): Trained model to explain
        device (str): Device for computation
    
    Example:
        >>> analyzer = DeepSHAPAnalyzer(model)
        >>> analyzer.setup_explainer(background_data)
        >>> explanation = analyzer.explain_prediction(drug_data, protein_data)
    """
    
    def __init__(self, model, device='cpu'):
        """Initialize SHAP analyzer."""
        
    def setup_explainer(self, background_data, n_background=100):
        """
        Setup SHAP explainer with background data.
        
        Args:
            background_data (dict): Background dataset for SHAP
            n_background (int): Number of background samples
        """
        
    def explain_prediction(self, molecular_data, protein_data, 
                          drug_smiles=None, protein_sequence=None):
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            molecular_data (dict): Molecular input data
            protein_data (dict): Protein input data
            drug_smiles (str): SMILES string for visualization
            protein_sequence (str): Protein sequence for visualization
        
        Returns:
            dict: SHAP explanation with values and visualizations
        """
        
    def analyze_feature_importance(self, test_data, n_samples=1000):
        """
        Analyze global feature importance across test set.
        
        Args:
            test_data: Test dataset
            n_samples (int): Number of samples to analyze
        
        Returns:
            dict: Global feature importance analysis
        """
```

---

## Web Interface

### DeepDTAProApp

Main Streamlit web application class.

```python
class DeepDTAProApp:
    """
    Streamlit web application for DeepDTA-Pro.
    
    Provides interactive interface for drug-target binding prediction
    with single/batch processing, model interpretation, and results visualization.
    
    Example:
        >>> app = DeepDTAProApp()
        >>> app.run()
    """
    
    def __init__(self):
        """Initialize web application."""
        
    def run(self):
        """Run the complete web application."""
        
    def render_header(self):
        """Render application header and navigation."""
        
    def render_single_prediction_page(self):
        """Render single prediction interface."""
        
    def render_batch_prediction_page(self):
        """Render batch prediction interface."""
        
    def render_model_interpretation_page(self):
        """Render model interpretation interface."""
        
    def render_analytics_dashboard(self):
        """Render analytics and statistics dashboard."""
        
    def process_single_prediction(self, drug_smiles, protein_sequence):
        """
        Process single drug-protein prediction.
        
        Args:
            drug_smiles (str): Drug SMILES string
            protein_sequence (str): Protein amino acid sequence
        
        Returns:
            dict: Prediction results with confidence and interpretation
        """
        
    def process_batch_predictions(self, batch_data):
        """
        Process batch drug-protein predictions.
        
        Args:
            batch_data (pd.DataFrame): Batch input data
        
        Returns:
            pd.DataFrame: Batch prediction results
        """
```

---

## Utilities

### ConfigManager

Configuration management system.

```python
class ConfigManager:
    """
    Centralized configuration management.
    
    Handles loading, validation, and merging of configuration files
    with support for environment-specific overrides.
    
    Args:
        config_path (str): Path to main configuration file
        env (str): Environment ('dev', 'prod', 'test')
    
    Example:
        >>> config_manager = ConfigManager('configs/config.yaml', env='prod')
        >>> model_config = config_manager.get_model_config()
        >>> training_config = config_manager.get_training_config()
    """
    
    def __init__(self, config_path, env='dev'):
        """Initialize configuration manager."""
        
    def load_config(self, config_path):
        """Load configuration from file."""
        
    def get_model_config(self):
        """Get model configuration."""
        
    def get_training_config(self):
        """Get training configuration."""
        
    def get_data_config(self):
        """Get data processing configuration."""
        
    def validate_config(self, config):
        """Validate configuration parameters."""
```

### Logger

Advanced logging system.

```python
class Logger:
    """
    Advanced logging system for training and evaluation.
    
    Provides structured logging with multiple output formats
    and integration with experiment tracking systems.
    
    Args:
        name (str): Logger name
        log_dir (str): Directory for log files
        level (str): Logging level
        use_wandb (bool): Enable Weights & Biases integration
    
    Example:
        >>> logger = Logger('deepdta_training', 'logs/', level='INFO')
        >>> logger.info("Starting training...")
        >>> logger.log_metrics({'loss': 0.245, 'rmse': 0.312}, step=100)
    """
    
    def __init__(self, name, log_dir, level='INFO', use_wandb=False):
        """Initialize logger with configuration."""
        
    def info(self, message):
        """Log info message."""
        
    def warning(self, message):
        """Log warning message."""
        
    def error(self, message):
        """Log error message."""
        
    def log_metrics(self, metrics, step=None):
        """
        Log metrics for experiment tracking.
        
        Args:
            metrics (dict): Metrics to log
            step (int): Step number
        """
        
    def log_model_architecture(self, model):
        """Log model architecture summary."""
        
    def save_experiment_config(self, config):
        """Save experiment configuration."""
```

---

## Error Handling

### Custom Exceptions

```python
class DeepDTAError(Exception):
    """Base exception for DeepDTA-Pro."""
    pass

class DataProcessingError(DeepDTAError):
    """Exception for data processing errors."""
    pass

class ModelError(DeepDTAError):
    """Exception for model-related errors."""
    pass

class TrainingError(DeepDTAError):
    """Exception for training-related errors."""
    pass

class ConfigurationError(DeepDTAError):
    """Exception for configuration errors."""
    pass
```

---

## Type Hints

Common type definitions used throughout the API:

```python
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import numpy as np
import pandas as pd

# Data types
MolecularData = Dict[str, Union[torch.Tensor, str]]
ProteinData = Dict[str, Union[torch.Tensor, str, int]]
BatchData = Dict[str, torch.Tensor]
MetricsDict = Dict[str, float]
ConfigDict = Dict[str, Any]

# Model types
ModelOutput = torch.Tensor
AttentionWeights = Dict[str, torch.Tensor]
FeatureImportance = Dict[str, np.ndarray]

# Training types
TrainingResults = Dict[str, Union[float, List[float]]]
ValidationResults = Dict[str, Union[float, List[float]]]
```

---

For more detailed examples and usage patterns, see the [examples directory](../examples/) and [tutorial notebooks](../notebooks/).