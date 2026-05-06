"""
Evaluation module for DeepDTA-Pro.

Provides comprehensive evaluation metrics, cross-validation strategies,
statistical testing, and baseline model comparisons.
"""

from .metrics import *
from .cross_validation import *
from .statistical_tests import *
from .baseline_models import *

__all__ = [
    # metrics
    'compute_metrics',
    'RMSEMetric',
    'MAEMetric',
    'PearsonMetric',
    'SpearmanMetric',
    'R2Metric',

    # cross_validation
    'CrossValidator',
    'KFoldCrossValidator',
    'StratifiedKFoldCrossValidator',

    # statistical_tests
    'paired_t_test',
    'mannwhitneyu_test',
    'wilcoxon_test',

    # baseline_models
    'LinearRegressionBaseline',
    'RandomForestBaseline',
    'SVRBaseline',
]
