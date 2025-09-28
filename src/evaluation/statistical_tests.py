"""
Statistical Testing and Model Comparison
Statistical tests and utilities for comparing model performance.
"""

import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, kruskal
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class StatTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None
    significant: Optional[bool] = None

@dataclass 
class ComparisonResult:
    """Container for model comparison results."""
    model_names: List[str]
    metrics: Dict[str, List[float]]
    statistical_tests: Dict[str, StatTestResult]
    rankings: Dict[str, List[int]]
    summary: Dict[str, Any]

class StatisticalTester:
    """
    Statistical testing framework for model comparison.
    
    Provides various statistical tests for comparing model performance:
    - Paired t-test
    - Wilcoxon signed-rank test
    - Mann-Whitney U test
    - Kruskal-Wallis test
    - Friedman test
    - McNemar's test (for classification)
    - Effect size calculations
    """
    
    def __init__(self, alpha: float = 0.05, correction: str = "bonferroni"):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level
            correction: Multiple testing correction method
        """
        self.alpha = alpha
        self.correction = correction.lower()
        
        logger.info(f"StatisticalTester initialized with α={alpha}, correction={correction}")
    
    def paired_t_test(self, 
                     scores_a: np.ndarray, 
                     scores_b: np.ndarray,
                     alternative: str = "two-sided") -> StatTestResult:
        """
        Perform paired t-test for comparing two models.
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            alternative: Alternative hypothesis ("two-sided", "less", "greater")
            
        Returns:
            Statistical test result
        """
        # Check assumptions
        differences = scores_a - scores_b
        
        # Normality test
        if len(differences) >= 3:
            _, normality_p = stats.shapiro(differences)
        else:
            normality_p = 1.0
        
        # Perform t-test
        statistic, p_value = ttest_rel(scores_a, scores_b, alternative=alternative)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
        cohens_d = np.mean(differences) / pooled_std if pooled_std > 0 else 0.0
        
        # Confidence interval for mean difference
        se_diff = stats.sem(differences)
        ci = stats.t.interval(1 - self.alpha, len(differences) - 1, 
                             loc=np.mean(differences), scale=se_diff)
        
        # Interpretation
        interpretation = self._interpret_ttest(statistic, p_value, cohens_d, normality_p)
        
        return StatTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=ci,
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def wilcoxon_test(self, 
                     scores_a: np.ndarray, 
                     scores_b: np.ndarray,
                     alternative: str = "two-sided") -> StatTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            alternative: Alternative hypothesis
            
        Returns:
            Statistical test result
        """
        # Perform Wilcoxon test
        try:
            statistic, p_value = wilcoxon(scores_a, scores_b, alternative=alternative)
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return StatTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                interpretation="Test failed - identical distributions"
            )
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(scores_a)
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z from p-value
        effect_size = z_score / np.sqrt(n) if n > 0 else 0.0
        
        # Interpretation
        interpretation = self._interpret_wilcoxon(statistic, p_value, effect_size)
        
        return StatTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def mann_whitney_test(self, 
                         scores_a: np.ndarray, 
                         scores_b: np.ndarray,
                         alternative: str = "two-sided") -> StatTestResult:
        """
        Perform Mann-Whitney U test (non-parametric independent samples test).
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            alternative: Alternative hypothesis
            
        Returns:
            Statistical test result
        """
        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(scores_a, scores_b, alternative=alternative)
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(scores_a), len(scores_b)
        u1 = statistic
        u2 = n1 * n2 - u1
        effect_size = 1 - (2 * min(u1, u2)) / (n1 * n2)
        
        # Interpretation
        interpretation = self._interpret_mann_whitney(statistic, p_value, effect_size)
        
        return StatTestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def friedman_test(self, *score_arrays: np.ndarray) -> StatTestResult:
        """
        Perform Friedman test for comparing multiple models (non-parametric ANOVA).
        
        Args:
            *score_arrays: Score arrays for each model
            
        Returns:
            Statistical test result
        """
        # Perform Friedman test
        statistic, p_value = stats.friedmanchisquare(*score_arrays)
        
        # Calculate effect size (Kendall's W)
        k = len(score_arrays)  # number of models
        n = len(score_arrays[0])  # number of datasets/folds
        
        # Convert to ranks
        all_scores = np.column_stack(score_arrays)
        ranks = stats.rankdata(all_scores, axis=1)
        
        # Calculate Kendall's W
        rank_sums = np.sum(ranks, axis=0)
        w = (12 * np.var(rank_sums, ddof=1)) / (n**2 * (k**3 - k))
        
        # Interpretation
        interpretation = self._interpret_friedman(statistic, p_value, w)
        
        return StatTestResult(
            test_name="Friedman test",
            statistic=statistic,
            p_value=p_value,
            effect_size=w,
            interpretation=interpretation,
            significant=p_value < self.alpha
        )
    
    def nemenyi_post_hoc(self, 
                        score_arrays: List[np.ndarray],
                        model_names: List[str]) -> Dict[Tuple[str, str], StatTestResult]:
        """
        Perform Nemenyi post-hoc test after Friedman test.
        
        Args:
            score_arrays: List of score arrays for each model
            model_names: Names of the models
            
        Returns:
            Dictionary of pairwise comparison results
        """
        k = len(score_arrays)  # number of models
        n = len(score_arrays[0])  # number of datasets
        
        # Convert to ranks
        all_scores = np.column_stack(score_arrays)
        ranks = stats.rankdata(-all_scores, axis=1)  # Negative for descending order
        
        # Calculate average ranks
        avg_ranks = np.mean(ranks, axis=0)
        
        # Critical difference
        q_alpha = self._get_studentized_range_critical_value(k, np.inf, self.alpha)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
        
        # Pairwise comparisons
        results = {}
        
        for i in range(k):
            for j in range(i + 1, k):
                rank_diff = abs(avg_ranks[i] - avg_ranks[j])
                significant = rank_diff > cd
                
                # Approximate p-value (not exact for Nemenyi)
                z_score = rank_diff / np.sqrt(k * (k + 1) / (6 * n))
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                results[(model_names[i], model_names[j])] = StatTestResult(
                    test_name="Nemenyi post-hoc test",
                    statistic=rank_diff,
                    p_value=p_value,
                    effect_size=rank_diff / k,  # Normalized rank difference
                    interpretation=f"Rank difference: {rank_diff:.3f}, Critical difference: {cd:.3f}",
                    significant=significant
                )
        
        return results
    
    def bootstrap_test(self, 
                      scores_a: np.ndarray, 
                      scores_b: np.ndarray,
                      n_bootstrap: int = 10000,
                      statistic_func: callable = None) -> StatTestResult:
        """
        Perform bootstrap test for comparing two models.
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            n_bootstrap: Number of bootstrap samples
            statistic_func: Function to compute test statistic
            
        Returns:
            Statistical test result
        """
        if statistic_func is None:
            statistic_func = lambda a, b: np.mean(a) - np.mean(b)
        
        # Observed statistic
        observed_stat = statistic_func(scores_a, scores_b)
        
        # Bootstrap procedure
        combined = np.concatenate([scores_a, scores_b])
        n_a, n_b = len(scores_a), len(scores_b)
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample under null hypothesis (no difference)
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            boot_a = resampled[:n_a]
            boot_b = resampled[n_a:n_a+n_b]
            
            boot_stat = statistic_func(boot_a, boot_b)
            bootstrap_stats.append(boot_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate p-value
        p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_stats, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha/2) * 100)
        
        return StatTestResult(
            test_name="Bootstrap test",
            statistic=observed_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=f"Bootstrap p-value based on {n_bootstrap} samples",
            significant=p_value < self.alpha
        )
    
    def _interpret_ttest(self, statistic: float, p_value: float, 
                        cohens_d: float, normality_p: float) -> str:
        """Interpret t-test results."""
        interpretation = []
        
        if normality_p < 0.05:
            interpretation.append("Warning: normality assumption may be violated")
        
        if p_value < self.alpha:
            interpretation.append("Significant difference detected")
        else:
            interpretation.append("No significant difference")
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            interpretation.append("negligible effect size")
        elif abs(cohens_d) < 0.5:
            interpretation.append("small effect size")
        elif abs(cohens_d) < 0.8:
            interpretation.append("medium effect size")
        else:
            interpretation.append("large effect size")
        
        return "; ".join(interpretation)
    
    def _interpret_wilcoxon(self, statistic: float, p_value: float, effect_size: float) -> str:
        """Interpret Wilcoxon test results."""
        interpretation = []
        
        if p_value < self.alpha:
            interpretation.append("Significant difference detected")
        else:
            interpretation.append("No significant difference")
        
        # Effect size interpretation
        if abs(effect_size) < 0.1:
            interpretation.append("negligible effect")
        elif abs(effect_size) < 0.3:
            interpretation.append("small effect")
        elif abs(effect_size) < 0.5:
            interpretation.append("medium effect")
        else:
            interpretation.append("large effect")
        
        return "; ".join(interpretation)
    
    def _interpret_mann_whitney(self, statistic: float, p_value: float, effect_size: float) -> str:
        """Interpret Mann-Whitney test results."""
        interpretation = []
        
        if p_value < self.alpha:
            interpretation.append("Significant difference detected")
        else:
            interpretation.append("No significant difference")
        
        interpretation.append(f"Effect size (rank-biserial correlation): {effect_size:.3f}")
        
        return "; ".join(interpretation)
    
    def _interpret_friedman(self, statistic: float, p_value: float, kendalls_w: float) -> str:
        """Interpret Friedman test results."""
        interpretation = []
        
        if p_value < self.alpha:
            interpretation.append("Significant differences among models detected")
        else:
            interpretation.append("No significant differences among models")
        
        # Kendall's W interpretation
        if kendalls_w < 0.1:
            interpretation.append("very weak agreement")
        elif kendalls_w < 0.3:
            interpretation.append("weak agreement")
        elif kendalls_w < 0.5:
            interpretation.append("moderate agreement")
        elif kendalls_w < 0.7:
            interpretation.append("strong agreement")
        else:
            interpretation.append("very strong agreement")
        
        return "; ".join(interpretation)
    
    def _get_studentized_range_critical_value(self, k: int, df: float, alpha: float) -> float:
        """Get critical value for studentized range distribution (approximation)."""
        # This is an approximation - for exact values, use statistical tables
        if k == 2:
            return stats.norm.ppf(1 - alpha/2) * np.sqrt(2)
        elif k == 3:
            return 3.314 if alpha == 0.05 else 2.569
        elif k == 4:
            return 3.633 if alpha == 0.05 else 2.728
        elif k == 5:
            return 3.858 if alpha == 0.05 else 2.850
        else:
            # Rough approximation for larger k
            return stats.norm.ppf(1 - alpha/(2*k)) * np.sqrt(2)
    
    def apply_multiple_testing_correction(self, p_values: List[float]) -> List[float]:
        """Apply multiple testing correction."""
        p_values = np.array(p_values)
        
        if self.correction == "bonferroni":
            return np.minimum(p_values * len(p_values), 1.0)
        elif self.correction == "holm":
            return self._holm_correction(p_values)
        elif self.correction == "benjamini_hochberg":
            return self._benjamini_hochberg_correction(p_values)
        else:
            return p_values
    
    def _holm_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Holm correction."""
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        n = len(p_values)
        corrected = np.zeros_like(p_values)
        
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            corrected[idx] = min(p * (n - i), 1.0)
        
        return corrected
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Hochberg correction."""
        sorted_indices = np.argsort(p_values)[::-1]  # Descending order
        sorted_p = p_values[sorted_indices]
        
        n = len(p_values)
        corrected = np.zeros_like(p_values)
        
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            corrected[idx] = min(p * n / (n - i), 1.0)
        
        return corrected

class ModelComparator:
    """
    Comprehensive model comparison framework.
    
    Compares multiple models using various statistical tests and provides
    rankings, visualizations, and detailed comparison reports.
    """
    
    def __init__(self, alpha: float = 0.05, correction: str = "bonferroni"):
        """
        Initialize model comparator.
        
        Args:
            alpha: Significance level
            correction: Multiple testing correction
        """
        self.statistical_tester = StatisticalTester(alpha, correction)
        self.alpha = alpha
    
    def compare_models(self, 
                      model_results: Dict[str, Dict[str, List[float]]],
                      metrics_to_compare: List[str] = None) -> ComparisonResult:
        """
        Compare multiple models across multiple metrics.
        
        Args:
            model_results: Dictionary with model names as keys and metrics as values
                Format: {model_name: {metric_name: [fold_1_score, fold_2_score, ...]}}
            metrics_to_compare: List of metrics to compare (if None, use all)
            
        Returns:
            Comparison results
        """
        model_names = list(model_results.keys())
        
        if metrics_to_compare is None:
            # Find common metrics across all models
            all_metrics = set()
            for results in model_results.values():
                all_metrics.update(results.keys())
            metrics_to_compare = list(all_metrics)
        
        # Validate that all models have all requested metrics
        for model_name in model_names:
            for metric in metrics_to_compare:
                if metric not in model_results[model_name]:
                    raise ValueError(f"Model {model_name} missing metric {metric}")
        
        # Perform statistical tests
        statistical_tests = {}
        rankings = {}
        
        for metric in metrics_to_compare:
            logger.info(f"Comparing models on {metric}...")
            
            # Extract scores for this metric
            metric_scores = [model_results[name][metric] for name in model_names]
            
            # Perform Friedman test (overall comparison)
            if len(model_names) > 2:
                friedman_result = self.statistical_tester.friedman_test(*metric_scores)
                statistical_tests[f"{metric}_friedman"] = friedman_result
                
                # If significant, perform post-hoc tests
                if friedman_result.significant:
                    posthoc_results = self.statistical_tester.nemenyi_post_hoc(
                        metric_scores, model_names
                    )
                    statistical_tests[f"{metric}_posthoc"] = posthoc_results
            
            # Pairwise comparisons
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    name_a, name_b = model_names[i], model_names[j]
                    scores_a = np.array(metric_scores[i])
                    scores_b = np.array(metric_scores[j])
                    
                    # Paired t-test
                    ttest_result = self.statistical_tester.paired_t_test(scores_a, scores_b)
                    statistical_tests[f"{metric}_{name_a}_vs_{name_b}_ttest"] = ttest_result
                    
                    # Wilcoxon test
                    wilcoxon_result = self.statistical_tester.wilcoxon_test(scores_a, scores_b)
                    statistical_tests[f"{metric}_{name_a}_vs_{name_b}_wilcoxon"] = wilcoxon_result
            
            # Calculate rankings
            avg_scores = [np.mean(scores) for scores in metric_scores]
            
            # For loss metrics, lower is better; for others, higher is better
            is_loss_metric = any(loss_term in metric.lower() 
                               for loss_term in ['loss', 'error', 'mse', 'mae', 'rmse'])
            
            if is_loss_metric:
                ranking = np.argsort(avg_scores)  # Ascending order
            else:
                ranking = np.argsort(avg_scores)[::-1]  # Descending order
            
            rankings[metric] = [model_names[i] for i in ranking]
        
        # Create summary
        summary = self._create_comparison_summary(model_results, rankings, statistical_tests)
        
        return ComparisonResult(
            model_names=model_names,
            metrics={metric: [model_results[name][metric] for name in model_names] 
                    for metric in metrics_to_compare},
            statistical_tests=statistical_tests,
            rankings=rankings,
            summary=summary
        )
    
    def _create_comparison_summary(self, 
                                  model_results: Dict[str, Dict[str, List[float]]],
                                  rankings: Dict[str, List[str]],
                                  statistical_tests: Dict[str, StatTestResult]) -> Dict[str, Any]:
        """Create comparison summary."""
        model_names = list(model_results.keys())
        
        # Overall rankings (average rank across metrics)
        overall_ranks = {name: [] for name in model_names}
        
        for metric, ranking in rankings.items():
            for rank, model_name in enumerate(ranking):
                overall_ranks[model_name].append(rank + 1)  # 1-indexed
        
        avg_ranks = {name: np.mean(ranks) for name, ranks in overall_ranks.items()}
        overall_ranking = sorted(model_names, key=lambda x: avg_ranks[x])
        
        # Count significant differences
        significant_tests = sum(1 for test in statistical_tests.values() 
                              if hasattr(test, 'significant') and test.significant)
        total_tests = len([test for test in statistical_tests.values() 
                          if hasattr(test, 'significant')])
        
        return {
            'overall_ranking': overall_ranking,
            'average_ranks': avg_ranks,
            'significant_tests': significant_tests,
            'total_tests': total_tests,
            'best_model': overall_ranking[0] if overall_ranking else None,
            'worst_model': overall_ranking[-1] if overall_ranking else None
        }
    
    def save_comparison_report(self, 
                              comparison_result: ComparisonResult,
                              output_path: str):
        """Save detailed comparison report."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        import json
        
        # Convert results to serializable format
        serializable_summary = {
            'model_names': comparison_result.model_names,
            'rankings': comparison_result.rankings,
            'summary': comparison_result.summary,
            'statistical_tests_summary': {}
        }
        
        # Summarize statistical tests
        for test_name, test_result in comparison_result.statistical_tests.items():
            if hasattr(test_result, 'p_value'):
                serializable_summary['statistical_tests_summary'][test_name] = {
                    'test_name': test_result.test_name,
                    'p_value': test_result.p_value,
                    'significant': test_result.significant,
                    'effect_size': test_result.effect_size
                }
        
        with open(output_path / "comparison_summary.json", 'w') as f:
            json.dump(serializable_summary, f, indent=2, default=str)
        
        # Create detailed text report
        self._create_text_report(comparison_result, output_path / "comparison_report.txt")
        
        logger.info(f"Comparison report saved to {output_path}")
    
    def _create_text_report(self, result: ComparisonResult, file_path: Path):
        """Create detailed text report."""
        with open(file_path, 'w') as f:
            f.write("Model Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall summary
            f.write("Overall Summary:\n")
            f.write(f"Number of models: {len(result.model_names)}\n")
            f.write(f"Best model: {result.summary['best_model']}\n")
            f.write(f"Significant tests: {result.summary['significant_tests']}/{result.summary['total_tests']}\n\n")
            
            # Rankings
            f.write("Rankings by Metric:\n")
            for metric, ranking in result.rankings.items():
                f.write(f"{metric}: {' > '.join(ranking)}\n")
            f.write("\n")
            
            # Overall ranking
            f.write("Overall Ranking:\n")
            for i, model in enumerate(result.summary['overall_ranking']):
                avg_rank = result.summary['average_ranks'][model]
                f.write(f"{i+1}. {model} (avg rank: {avg_rank:.2f})\n")
            f.write("\n")
            
            # Statistical tests
            f.write("Statistical Test Results:\n")
            for test_name, test_result in result.statistical_tests.items():
                if hasattr(test_result, 'p_value'):
                    f.write(f"{test_name}:\n")
                    f.write(f"  p-value: {test_result.p_value:.4f}\n")
                    f.write(f"  Significant: {test_result.significant}\n")
                    if test_result.effect_size is not None:
                        f.write(f"  Effect size: {test_result.effect_size:.4f}\n")
                    f.write("\n")

# Example usage
if __name__ == "__main__":
    print("Testing StatisticalTester and ModelComparator...")
    
    # Generate dummy data for testing
    np.random.seed(42)
    
    # Simulate cross-validation results for 3 models
    model_results = {
        'Model_A': {
            'rmse': np.random.normal(0.8, 0.1, 5).tolist(),
            'mae': np.random.normal(0.6, 0.08, 5).tolist(),
            'pearson': np.random.normal(0.85, 0.05, 5).tolist()
        },
        'Model_B': {
            'rmse': np.random.normal(0.85, 0.12, 5).tolist(),
            'mae': np.random.normal(0.65, 0.09, 5).tolist(),
            'pearson': np.random.normal(0.82, 0.06, 5).tolist()
        },
        'Model_C': {
            'rmse': np.random.normal(0.9, 0.08, 5).tolist(),
            'mae': np.random.normal(0.7, 0.07, 5).tolist(),
            'pearson': np.random.normal(0.78, 0.07, 5).tolist()
        }
    }
    
    # Test statistical tester
    tester = StatisticalTester(alpha=0.05)
    
    scores_a = np.array(model_results['Model_A']['rmse'])
    scores_b = np.array(model_results['Model_B']['rmse'])
    
    # Paired t-test
    ttest_result = tester.paired_t_test(scores_a, scores_b)
    print(f"T-test result: p={ttest_result.p_value:.4f}, significant={ttest_result.significant}")
    
    # Wilcoxon test
    wilcoxon_result = tester.wilcoxon_test(scores_a, scores_b)
    print(f"Wilcoxon result: p={wilcoxon_result.p_value:.4f}, significant={wilcoxon_result.significant}")
    
    # Test model comparator
    comparator = ModelComparator(alpha=0.05)
    comparison_result = comparator.compare_models(model_results)
    
    print(f"Best model: {comparison_result.summary['best_model']}")
    print(f"Overall ranking: {comparison_result.summary['overall_ranking']}")
    print(f"Significant tests: {comparison_result.summary['significant_tests']}/{comparison_result.summary['total_tests']}")
    
    print("Statistical testing and model comparison test completed successfully!")