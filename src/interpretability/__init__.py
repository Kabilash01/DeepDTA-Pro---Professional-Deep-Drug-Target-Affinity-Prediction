"""
Interpretability Module for DeepDTA-Pro
Model interpretation and explainability tools.
"""

from .attention_visualization import AttentionVisualizer, extract_attention_from_model, analyze_attention_patterns
from .shap_analysis import DeepSHAPAnalyzer, FeatureAttributionAnalyzer, molecular_descriptors_baseline, protein_features_baseline
from .molecular_interpretation import MolecularInterpreter, calculate_molecular_descriptors

__all__ = [
    'AttentionVisualizer',
    'extract_attention_from_model', 
    'analyze_attention_patterns',
    'DeepSHAPAnalyzer',
    'FeatureAttributionAnalyzer',
    'molecular_descriptors_baseline',
    'protein_features_baseline',
    'MolecularInterpreter',
    'calculate_molecular_descriptors'
]

__version__ = '1.0.0'