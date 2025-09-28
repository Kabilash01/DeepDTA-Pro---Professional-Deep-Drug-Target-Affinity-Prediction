"""
Evaluation Metrics
Comprehensive metrics for drug-target binding affinity prediction evaluation.
"""

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from sklearn.metrics import concordance_index_score
except ImportError:
    # In newer versions of scikit-learn, it might be in a different location
    try:
        from sksurv.metrics import concordance_index_censored as concordance_index_score
    except ImportError:
        # Fallback: implement a simple concordance index
        def concordance_index_score(y_true, y_pred):
            """Simple concordance index implementation."""
            n_concordant = 0
            n_pairs = 0
            
            for i in range(len(y_true)):
                for j in range(i + 1, len(y_true)):
                    if y_true[i] != y_true[j]:
                        n_pairs += 1
                        if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                           (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                            n_concordant += 1
            
            return n_concordant / n_pairs if n_pairs > 0 else 0.5
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    interpretation: Optional[str] = None

class RegressionMetrics:
    """
    Comprehensive regression metrics for binding affinity prediction.
    
    Includes traditional metrics, statistical tests, and domain-specific metrics.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize metrics calculator.
        
        Args:
            confidence_level: Confidence level for confidence intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_basic_metrics(self, 
                               predictions: np.ndarray, 
                               targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of basic metrics
        """
        # Ensure arrays are flattened
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Basic regression metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        # R-squared
        r2 = r2_score(targets, predictions)
        
        # Mean Absolute Percentage Error (MAPE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((targets - predictions) / targets)) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = float('inf')
        
        # Median Absolute Error
        median_ae = np.median(np.abs(targets - predictions))
        
        # Max error
        max_error = np.max(np.abs(targets - predictions))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'median_ae': median_ae,
            'max_error': max_error
        }
    
    def calculate_correlation_metrics(self, 
                                    predictions: np.ndarray, 
                                    targets: np.ndarray) -> Dict[str, MetricResult]:
        """
        Calculate correlation-based metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of correlation metrics with statistical significance
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        results = {}
        
        # Pearson correlation
        try:
            pearson_r, pearson_p = stats.pearsonr(predictions, targets)
            results['pearson'] = MetricResult(
                value=pearson_r,
                p_value=pearson_p,
                interpretation=self._interpret_correlation(pearson_r)
            )
        except Exception as e:
            logger.warning(f"Error calculating Pearson correlation: {e}")
            results['pearson'] = MetricResult(value=0.0)
        
        # Spearman correlation
        try:
            spearman_r, spearman_p = stats.spearmanr(predictions, targets)
            results['spearman'] = MetricResult(
                value=spearman_r,
                p_value=spearman_p,
                interpretation=self._interpret_correlation(spearman_r)
            )
        except Exception as e:
            logger.warning(f"Error calculating Spearman correlation: {e}")
            results['spearman'] = MetricResult(value=0.0)
        
        # Kendall's tau
        try:
            kendall_tau, kendall_p = stats.kendalltau(predictions, targets)
            results['kendall_tau'] = MetricResult(
                value=kendall_tau,
                p_value=kendall_p,
                interpretation=self._interpret_correlation(kendall_tau)
            )
        except Exception as e:
            logger.warning(f"Error calculating Kendall's tau: {e}")
            results['kendall_tau'] = MetricResult(value=0.0)
        
        return results
    
    def calculate_concordance_index(self, 
                                  predictions: np.ndarray, 
                                  targets: np.ndarray) -> MetricResult:
        """
        Calculate Concordance Index (C-Index).
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Concordance index result
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        try:
            # Calculate C-index
            c_index = concordance_index_score(targets, predictions)
            
            # Bootstrap confidence interval
            n_bootstrap = 1000
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(targets), size=len(targets), replace=True)
                boot_targets = targets[indices]
                boot_predictions = predictions[indices]
                
                try:
                    boot_score = concordance_index_score(boot_targets, boot_predictions)
                    bootstrap_scores.append(boot_score)
                except:
                    continue
            
            if bootstrap_scores:
                lower = np.percentile(bootstrap_scores, (self.alpha/2) * 100)
                upper = np.percentile(bootstrap_scores, (1 - self.alpha/2) * 100)
                ci = (lower, upper)
            else:
                ci = None
                
            return MetricResult(
                value=c_index,
                confidence_interval=ci,
                interpretation=self._interpret_c_index(c_index)
            )
            
        except Exception as e:
            logger.warning(f"Error calculating C-index: {e}")
            return MetricResult(value=0.5)
    
    def calculate_domain_specific_metrics(self, 
                                        predictions: np.ndarray, 
                                        targets: np.ndarray,
                                        threshold: float = 7.0) -> Dict[str, float]:
        """
        Calculate domain-specific metrics for drug-target interaction.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            threshold: Threshold for active/inactive classification
            
        Returns:
            Dictionary of domain-specific metrics
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Binary classification based on threshold
        pred_active = predictions >= threshold
        true_active = targets >= threshold
        
        # Classification metrics
        tp = np.sum(pred_active & true_active)
        tn = np.sum(~pred_active & ~true_active)
        fp = np.sum(pred_active & ~true_active)
        fn = np.sum(~pred_active & true_active)
        
        # Calculate classification metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # ROC AUC approximation using ranking
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(true_active.astype(int), predictions)
        except:
            auc = 0.5
        
        # Enrichment factor
        # Sort by predictions (descending)
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_targets = targets[sorted_indices]
        
        # Calculate enrichment for top 1%, 5%, 10%
        n_total = len(targets)
        n_active = np.sum(true_active)
        
        enrichment = {}
        for pct in [0.01, 0.05, 0.1]:
            n_top = int(n_total * pct)
            if n_top > 0:
                n_active_top = np.sum(sorted_targets[:n_top] >= threshold)
                expected = n_active * pct
                enrichment[f'ef_{int(pct*100)}'] = n_active_top / expected if expected > 0 else 0.0
            else:
                enrichment[f'ef_{int(pct*100)}'] = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc': auc,
            **enrichment
        }
    
    def calculate_distribution_metrics(self, 
                                     predictions: np.ndarray, 
                                     targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics comparing prediction and target distributions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of distribution metrics
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Kolmogorov-Smirnov test
        try:
            ks_statistic, ks_p_value = stats.ks_2samp(predictions, targets)
        except:
            ks_statistic, ks_p_value = 0.0, 1.0
        
        # Anderson-Darling test
        try:
            # Combine and sort data
            combined = np.concatenate([predictions, targets])
            combined_sorted = np.sort(combined)
            
            # Create empirical CDFs
            pred_cdf = np.searchsorted(combined_sorted, predictions, side='right') / len(combined_sorted)
            target_cdf = np.searchsorted(combined_sorted, targets, side='right') / len(combined_sorted)
            
            # Anderson-Darling statistic (simplified)
            ad_statistic = np.mean((pred_cdf - target_cdf) ** 2)
        except:
            ad_statistic = 0.0
        
        # Wasserstein distance (Earth Mover's Distance)
        try:
            wasserstein = stats.wasserstein_distance(predictions, targets)
        except:
            wasserstein = 0.0
        
        # Jensen-Shannon divergence (for discretized distributions)
        try:
            # Create histograms
            bins = np.linspace(min(np.min(predictions), np.min(targets)),
                              max(np.max(predictions), np.max(targets)), 50)
            pred_hist, _ = np.histogram(predictions, bins=bins, density=True)
            target_hist, _ = np.histogram(targets, bins=bins, density=True)
            
            # Normalize to probabilities
            pred_hist = pred_hist / np.sum(pred_hist)
            target_hist = target_hist / np.sum(target_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            pred_hist += epsilon
            target_hist += epsilon
            
            # Calculate JS divergence
            m = (pred_hist + target_hist) / 2
            js_div = 0.5 * stats.entropy(pred_hist, m) + 0.5 * stats.entropy(target_hist, m)
        except:
            js_div = 0.0
        
        return {
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'anderson_darling': ad_statistic,
            'wasserstein_distance': wasserstein,
            'js_divergence': js_div
        }
    
    def calculate_residual_metrics(self, 
                                 predictions: np.ndarray, 
                                 targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate residual-based metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of residual metrics
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        residuals = targets - predictions
        
        # Basic residual statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        # Shapiro-Wilk test for normality of residuals
        try:
            if len(residuals) > 3:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
            else:
                shapiro_stat, shapiro_p = 0.0, 1.0
        except:
            shapiro_stat, shapiro_p = 0.0, 1.0
        
        # Durbin-Watson test for autocorrelation
        try:
            diff = np.diff(residuals)
            dw_statistic = np.sum(diff**2) / np.sum(residuals**2)
        except:
            dw_statistic = 2.0
        
        return {
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'residual_skewness': skewness,
            'residual_kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'durbin_watson': dw_statistic
        }
    
    def calculate_all_metrics(self, 
                            predictions: np.ndarray, 
                            targets: np.ndarray,
                            threshold: float = 7.0) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            threshold: Threshold for domain-specific metrics
            
        Returns:
            Dictionary containing all metrics
        """
        results = {}
        
        # Basic metrics
        results.update(self.calculate_basic_metrics(predictions, targets))
        
        # Correlation metrics
        corr_metrics = self.calculate_correlation_metrics(predictions, targets)
        for name, metric_result in corr_metrics.items():
            results[name] = metric_result.value
            if metric_result.p_value is not None:
                results[f"{name}_p_value"] = metric_result.p_value
        
        # Concordance index
        c_index = self.calculate_concordance_index(predictions, targets)
        results['c_index'] = c_index.value
        if c_index.confidence_interval is not None:
            results['c_index_ci_lower'] = c_index.confidence_interval[0]
            results['c_index_ci_upper'] = c_index.confidence_interval[1]
        
        # Domain-specific metrics
        results.update(self.calculate_domain_specific_metrics(predictions, targets, threshold))
        
        # Distribution metrics
        results.update(self.calculate_distribution_metrics(predictions, targets))
        
        # Residual metrics
        results.update(self.calculate_residual_metrics(predictions, targets))
        
        return results
    
    def calculate_batch_metrics(self, 
                               predictions: np.ndarray, 
                               targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic metrics for a single batch (fast computation).
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of basic metrics
        """
        return self.calculate_basic_metrics(predictions, targets)
    
    def calculate_metrics(self, 
                         predictions: np.ndarray, 
                         targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics (compatible with trainer interface).
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of metrics
        """
        all_metrics = self.calculate_all_metrics(predictions, targets)
        
        # Convert MetricResult objects to float values for compatibility
        float_metrics = {}
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                float_metrics[key] = float(value)
            elif hasattr(value, 'value'):
                float_metrics[key] = float(value.value)
        
        return float_metrics
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        abs_r = abs(r)
        if abs_r >= 0.9:
            return "Very strong"
        elif abs_r >= 0.7:
            return "Strong"
        elif abs_r >= 0.5:
            return "Moderate"
        elif abs_r >= 0.3:
            return "Weak"
        else:
            return "Very weak"
    
    def _interpret_c_index(self, c: float) -> str:
        """Interpret concordance index."""
        if c >= 0.8:
            return "Excellent"
        elif c >= 0.7:
            return "Good"
        elif c >= 0.6:
            return "Fair"
        elif c >= 0.5:
            return "Poor"
        else:
            return "Random"

# Example usage and testing
if __name__ == "__main__":
    print("Testing RegressionMetrics...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic drug-target binding data
    true_values = np.random.uniform(4.0, 10.0, n_samples)  # pKd values
    noise = np.random.normal(0, 0.5, n_samples)
    predictions = true_values + noise + np.random.uniform(-0.2, 0.2, n_samples)
    
    # Initialize metrics calculator
    metrics_calc = RegressionMetrics()
    
    # Calculate all metrics
    print("Calculating comprehensive metrics...")
    all_metrics = metrics_calc.calculate_all_metrics(predictions, true_values)
    
    print("Basic Metrics:")
    basic_keys = ['mse', 'rmse', 'mae', 'r2']
    for key in basic_keys:
        if key in all_metrics:
            print(f"  {key.upper()}: {all_metrics[key]:.4f}")
    
    print("\nCorrelation Metrics:")
    corr_keys = ['pearson', 'spearman', 'kendall_tau']
    for key in corr_keys:
        if key in all_metrics:
            print(f"  {key.capitalize()}: {all_metrics[key]:.4f}")
            if f"{key}_p_value" in all_metrics:
                print(f"    p-value: {all_metrics[f'{key}_p_value']:.4e}")
    
    print(f"\nConcordance Index: {all_metrics.get('c_index', 0):.4f}")
    if 'c_index_ci_lower' in all_metrics:
        print(f"  95% CI: [{all_metrics['c_index_ci_lower']:.4f}, {all_metrics['c_index_ci_upper']:.4f}]")
    
    print("\nDomain-specific Metrics:")
    domain_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    for key in domain_keys:
        if key in all_metrics:
            print(f"  {key.capitalize()}: {all_metrics[key]:.4f}")
    
    print("\nEnrichment Factors:")
    ef_keys = ['ef_1', 'ef_5', 'ef_10']
    for key in ef_keys:
        if key in all_metrics:
            print(f"  EF{key.split('_')[1]}%: {all_metrics[key]:.2f}")
    
    print("\nDistribution Metrics:")
    dist_keys = ['ks_statistic', 'wasserstein_distance', 'js_divergence']
    for key in dist_keys:
        if key in all_metrics:
            print(f"  {key.replace('_', ' ').title()}: {all_metrics[key]:.4f}")
    
    print("RegressionMetrics test completed successfully!")