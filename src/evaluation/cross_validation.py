"""
Cross-Validation Framework
Comprehensive cross-validation strategies for model evaluation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    train_test_split
)
from typing import Dict, List, Tuple, Optional, Union, Any, Generator, Callable
import logging
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from dataclasses import dataclass
import warnings

from .metrics import RegressionMetrics
from ..training.trainer import DeepDTATrainer
from ..training.training_utils import TrainingConfig
from ..models.deepdta_pro import DeepDTAPro

logger = logging.getLogger(__name__)

@dataclass
class CVResult:
    """Container for cross-validation results."""
    fold_results: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    ci_metrics: Dict[str, Tuple[float, float]]
    fold_predictions: List[np.ndarray]
    fold_targets: List[np.ndarray]
    metadata: Dict[str, Any]

class CrossValidator:
    """
    Comprehensive cross-validation framework for model evaluation.
    
    Supports multiple CV strategies:
    - K-Fold CV
    - Stratified K-Fold CV
    - Group K-Fold CV
    - Time Series CV
    - Leave-One-Group-Out CV
    - Custom split functions
    """
    
    def __init__(self,
                 cv_strategy: str = "kfold",
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42,
                 confidence_level: float = 0.95):
        """
        Initialize cross-validator.
        
        Args:
            cv_strategy: Cross-validation strategy
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed
            confidence_level: Confidence level for intervals
        """
        self.cv_strategy = cv_strategy.lower()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.confidence_level = confidence_level
        
        self.metrics_calculator = RegressionMetrics(confidence_level)
        
        # Initialize CV splitter
        self.cv_splitter = self._create_cv_splitter()
        
        logger.info(f"CrossValidator initialized with {cv_strategy} strategy, {n_splits} splits")
    
    def _create_cv_splitter(self):
        """Create appropriate CV splitter based on strategy."""
        if self.cv_strategy == "kfold":
            return KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_strategy == "stratified":
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_strategy == "group":
            return GroupKFold(n_splits=self.n_splits)
        elif self.cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def cross_validate_model(self,
                           model_factory: Callable,
                           training_config: TrainingConfig,
                           dataset,
                           groups: Optional[np.ndarray] = None,
                           stratify_targets: Optional[np.ndarray] = None,
                           output_dir: str = "./cv_results",
                           experiment_name: str = "cv_experiment") -> CVResult:
        """
        Perform cross-validation on a model.
        
        Args:
            model_factory: Function that creates a new model instance
            training_config: Training configuration
            dataset: Dataset for cross-validation
            groups: Group labels for GroupKFold
            stratify_targets: Targets for stratification
            output_dir: Output directory for results
            experiment_name: Experiment name
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {self.n_splits}-fold cross-validation...")
        
        # Create output directory
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for splitting
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        
        # Generate splits
        if self.cv_strategy == "group":
            if groups is None:
                raise ValueError("Groups must be provided for GroupKFold")
            splits = list(self.cv_splitter.split(indices, groups=groups))
        elif self.cv_strategy == "stratified":
            if stratify_targets is None:
                raise ValueError("Targets must be provided for StratifiedKFold")
            # Convert continuous targets to bins for stratification
            n_bins = min(10, len(np.unique(stratify_targets)))
            binned_targets = np.digitize(stratify_targets, 
                                       np.percentile(stratify_targets, 
                                                   np.linspace(0, 100, n_bins)))
            splits = list(self.cv_splitter.split(indices, binned_targets))
        else:
            splits = list(self.cv_splitter.split(indices))
        
        # Perform cross-validation
        fold_results = []
        fold_predictions = []
        fold_targets = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_splits}...")
            
            # Create fold datasets
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=training_config.num_workers,
                pin_memory=training_config.pin_memory
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_config.batch_size,
                shuffle=False,
                num_workers=training_config.num_workers,
                pin_memory=training_config.pin_memory
            )
            
            # Create model instance
            model = model_factory()
            
            # Create trainer
            fold_output_dir = output_path / f"fold_{fold_idx}"
            trainer = DeepDTATrainer(
                model=model,
                config=training_config,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=str(fold_output_dir),
                experiment_name=f"fold_{fold_idx}"
            )
            
            # Train model
            training_history = trainer.train()
            
            # Evaluate on validation set
            val_results = trainer.evaluate(val_loader)
            
            # Get predictions for this fold
            model.eval()
            fold_preds = []
            fold_targs = []
            
            with torch.no_grad():
                for batch in val_loader:
                    molecular_batch = batch['molecular'].to(trainer.device)
                    protein_sequences = batch['protein'].to(trainer.device)
                    protein_lengths = batch.get('protein_lengths')
                    if protein_lengths is not None:
                        protein_lengths = protein_lengths.to(trainer.device)
                    targets = batch['affinity'].to(trainer.device)
                    
                    # Forward pass
                    output = model(molecular_batch, protein_sequences, protein_lengths)
                    predictions = output['predictions']
                    
                    fold_preds.append(predictions.cpu().numpy())
                    fold_targs.append(targets.cpu().numpy())
            
            # Concatenate fold results
            fold_preds = np.concatenate(fold_preds, axis=0)
            fold_targs = np.concatenate(fold_targs, axis=0)
            
            # Calculate comprehensive metrics for this fold
            fold_metrics = self.metrics_calculator.calculate_all_metrics(fold_preds, fold_targs)
            
            # Store results
            fold_results.append(fold_metrics)
            fold_predictions.append(fold_preds)
            fold_targets.append(fold_targs)
            
            # Log fold results
            log_str = f"Fold {fold_idx + 1} results: "
            key_metrics = ['rmse', 'mae', 'pearson', 'c_index']
            log_str += " | ".join([f"{k}: {fold_metrics.get(k, 0):.4f}" for k in key_metrics])
            logger.info(log_str)
        
        # Calculate overall statistics
        mean_metrics, std_metrics, ci_metrics = self._calculate_cv_statistics(fold_results)
        
        # Create CV result
        cv_result = CVResult(
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            ci_metrics=ci_metrics,
            fold_predictions=fold_predictions,
            fold_targets=fold_targets,
            metadata={
                'cv_strategy': self.cv_strategy,
                'n_splits': self.n_splits,
                'dataset_size': dataset_size,
                'experiment_name': experiment_name
            }
        )
        
        # Save results
        self._save_cv_results(cv_result, output_path)
        
        # Log summary
        self._log_cv_summary(cv_result)
        
        return cv_result
    
    def _calculate_cv_statistics(self, 
                                fold_results: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[float, float]]]:
        """Calculate mean, std, and confidence intervals across folds."""
        if not fold_results:
            return {}, {}, {}
        
        # Get all metric names
        all_metrics = set()
        for fold_result in fold_results:
            all_metrics.update(fold_result.keys())
        
        mean_metrics = {}
        std_metrics = {}
        ci_metrics = {}
        
        for metric_name in all_metrics:
            # Extract values for this metric across folds
            values = []
            for fold_result in fold_results:
                if metric_name in fold_result:
                    value = fold_result[metric_name]
                    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                        values.append(value)
            
            if values:
                values = np.array(values)
                
                mean_metrics[metric_name] = np.mean(values)
                std_metrics[metric_name] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                
                # Calculate confidence interval
                if len(values) > 1:
                    alpha = 1 - self.confidence_level
                    ci_lower = np.percentile(values, (alpha/2) * 100)
                    ci_upper = np.percentile(values, (1 - alpha/2) * 100)
                    ci_metrics[metric_name] = (ci_lower, ci_upper)
                else:
                    ci_metrics[metric_name] = (values[0], values[0])
        
        return mean_metrics, std_metrics, ci_metrics
    
    def _save_cv_results(self, cv_result: CVResult, output_path: Path):
        """Save cross-validation results to disk."""
        # Save summary
        summary = {
            'mean_metrics': cv_result.mean_metrics,
            'std_metrics': cv_result.std_metrics,
            'ci_metrics': {k: list(v) for k, v in cv_result.ci_metrics.items()},
            'metadata': cv_result.metadata
        }
        
        with open(output_path / "cv_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        with open(output_path / "cv_detailed_results.pkl", 'wb') as f:
            pickle.dump(cv_result, f)
        
        # Save fold results as JSON
        with open(output_path / "fold_results.json", 'w') as f:
            json.dump(cv_result.fold_results, f, indent=2, default=str)
        
        logger.info(f"CV results saved to {output_path}")
    
    def _log_cv_summary(self, cv_result: CVResult):
        """Log cross-validation summary."""
        logger.info("Cross-Validation Summary:")
        logger.info("=" * 50)
        
        key_metrics = ['rmse', 'mae', 'pearson', 'spearman', 'c_index', 'r2']
        
        for metric in key_metrics:
            if metric in cv_result.mean_metrics:
                mean_val = cv_result.mean_metrics[metric]
                std_val = cv_result.std_metrics.get(metric, 0.0)
                ci_lower, ci_upper = cv_result.ci_metrics.get(metric, (0, 0))
                
                logger.info(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f} "
                           f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    def nested_cross_validation(self,
                              model_factory: Callable,
                              param_grid: Dict[str, List[Any]],
                              training_config: TrainingConfig,
                              dataset,
                              outer_cv: int = 5,
                              inner_cv: int = 3,
                              scoring_metric: str = 'rmse',
                              output_dir: str = "./nested_cv_results",
                              experiment_name: str = "nested_cv") -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased model evaluation.
        
        Args:
            model_factory: Function that creates model instances
            param_grid: Grid of hyperparameters to search
            training_config: Base training configuration
            dataset: Dataset for evaluation
            outer_cv: Number of outer CV folds
            inner_cv: Number of inner CV folds
            scoring_metric: Metric to optimize
            output_dir: Output directory
            experiment_name: Experiment name
            
        Returns:
            Nested CV results
        """
        logger.info(f"Starting nested cross-validation ({outer_cv}x{inner_cv} folds)...")
        
        # Create output directory
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Outer CV splits
        outer_splitter = KFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        
        outer_scores = []
        best_params_list = []
        
        for outer_fold, (train_val_indices, test_indices) in enumerate(outer_splitter.split(indices)):
            logger.info(f"Outer fold {outer_fold + 1}/{outer_cv}")
            
            # Create train+validation and test sets
            train_val_dataset = Subset(dataset, train_val_indices)
            test_dataset = Subset(dataset, test_indices)
            
            # Inner CV for hyperparameter optimization
            inner_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=self.random_state + outer_fold)
            
            best_score = float('inf') if scoring_metric in ['rmse', 'mae', 'mse'] else float('-inf')
            best_params = None
            
            # Grid search
            param_combinations = self._generate_param_combinations(param_grid)
            
            for param_idx, params in enumerate(param_combinations):
                logger.info(f"  Testing parameter combination {param_idx + 1}/{len(param_combinations)}")
                
                inner_scores = []
                
                # Inner CV evaluation
                for inner_fold, (train_indices, val_indices) in enumerate(inner_splitter.split(train_val_indices)):
                    # Create datasets
                    train_dataset = Subset(train_val_dataset, train_indices)
                    val_dataset = Subset(train_val_dataset, val_indices)
                    
                    # Create data loaders
                    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)
                    
                    # Create model with current parameters
                    model = model_factory(**params)
                    
                    # Create modified training config
                    modified_config = self._modify_config(training_config, params)
                    
                    # Train model
                    trainer = DeepDTATrainer(
                        model=model,
                        config=modified_config,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        output_dir=str(output_path / f"outer_{outer_fold}" / f"inner_{inner_fold}"),
                        experiment_name=f"inner_{inner_fold}"
                    )
                    
                    # Quick training (reduced epochs for speed)
                    original_epochs = modified_config.num_epochs
                    modified_config.num_epochs = min(20, original_epochs)
                    
                    trainer.train()
                    
                    # Evaluate
                    val_results = trainer.evaluate(val_loader)
                    inner_scores.append(val_results.get(scoring_metric, 0.0))
                
                # Average inner CV score
                avg_score = np.mean(inner_scores)
                
                # Check if this is the best parameter combination
                is_better = (avg_score < best_score if scoring_metric in ['rmse', 'mae', 'mse'] 
                           else avg_score > best_score)
                
                if is_better:
                    best_score = avg_score
                    best_params = params
            
            # Train final model with best parameters on full train+val set
            logger.info(f"Best parameters for outer fold {outer_fold + 1}: {best_params}")
            
            final_model = model_factory(**best_params)
            final_config = self._modify_config(training_config, best_params)
            
            train_val_loader = DataLoader(train_val_dataset, batch_size=training_config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False)
            
            final_trainer = DeepDTATrainer(
                model=final_model,
                config=final_config,
                train_loader=train_val_loader,
                val_loader=None,
                test_loader=test_loader,
                output_dir=str(output_path / f"final_outer_{outer_fold}"),
                experiment_name=f"final_outer_{outer_fold}"
            )
            
            final_trainer.train()
            test_results = final_trainer.evaluate(test_loader)
            
            outer_scores.append(test_results.get(scoring_metric, 0.0))
            best_params_list.append(best_params)
            
            logger.info(f"Outer fold {outer_fold + 1} test score: {test_results.get(scoring_metric, 0.0):.4f}")
        
        # Calculate final statistics
        final_score = np.mean(outer_scores)
        final_std = np.std(outer_scores, ddof=1)
        
        results = {
            'final_score': final_score,
            'final_std': final_std,
            'outer_fold_scores': outer_scores,
            'best_parameters': best_params_list,
            'scoring_metric': scoring_metric,
            'outer_cv': outer_cv,
            'inner_cv': inner_cv
        }
        
        # Save results
        with open(output_path / "nested_cv_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Nested CV completed. Final {scoring_metric}: {final_score:.4f} ± {final_std:.4f}")
        
        return results
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = []
        
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _modify_config(self, base_config: TrainingConfig, params: Dict[str, Any]) -> TrainingConfig:
        """Modify training config with hyperparameters."""
        # Create a copy of the config
        import copy
        modified_config = copy.deepcopy(base_config)
        
        # Update config with parameters
        for key, value in params.items():
            if hasattr(modified_config, key):
                setattr(modified_config, key, value)
        
        return modified_config

# Example usage
if __name__ == "__main__":
    print("Testing CrossValidator...")
    
    # This would normally be run with actual datasets and models
    from ..models.deepdta_pro import DeepDTAPro
    
    # Create model factory
    def model_factory(**kwargs):
        return DeepDTAPro(**kwargs)
    
    # Create training config
    config = TrainingConfig(
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu"
    )
    
    # Create cross-validator
    cv = CrossValidator(cv_strategy="kfold", n_splits=3)
    
    print(f"CrossValidator created with {cv.cv_strategy} strategy")
    print(f"Number of splits: {cv.n_splits}")
    
    # Note: In actual usage, you would run:
    # cv_results = cv.cross_validate_model(model_factory, config, dataset)
    # nested_results = cv.nested_cross_validation(model_factory, param_grid, config, dataset)
    
    print("CrossValidator test completed successfully!")