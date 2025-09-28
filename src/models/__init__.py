"""
DeepDTA-Pro Neural Network Models
Implementation of advanced graph neural networks for drug-target binding affinity prediction.
"""

# Main model
from .deepdta_pro import DeepDTAPro, create_deepdta_pro_model

# Individual components
from .molecular_gnn import MolecularGNN, GINLayer, MolecularGraphConv
from .protein_encoder import (
    ProteinEncoder, ProteinCNN, ProteinTransformer, 
    HybridProteinEncoder, AminoAcidVocabulary
)
from .fusion_network import (
    FusionNetwork, BilinearFusion, AttentionFusion, 
    GatedFusion, MultiModalFusion, AdaptiveFusion
)
from .prediction_head import (
    PredictionHead, MLPPredictionHead, AttentionPredictionHead,
    EnsemblePredictionHead, UncertaintyPredictionHead
)
from .attention_layers import (
    GraphAttention, AttentionPooling, CrossAttention,
    SelfAttention, PositionalEncoding
)
# from .baseline_models import BaselineModels  # TODO: Create baseline models

__all__ = [
    # Main model
    'DeepDTAPro',
    'create_deepdta_pro_model',
    
    # Molecular components
    'MolecularGNN',
    'GINLayer', 
    'MolecularGraphConv',
    
    # Protein components
    'ProteinEncoder',
    'ProteinCNN',
    'ProteinTransformer',
    'HybridProteinEncoder',
    'AminoAcidVocabulary',
    
    # Fusion components
    'FusionNetwork',
    'BilinearFusion',
    'AttentionFusion',
    'GatedFusion',
    'MultiModalFusion',
    'AdaptiveFusion',
    
    # Prediction components
    'PredictionHead',
    'MLPPredictionHead',
    'AttentionPredictionHead',
    'EnsemblePredictionHead',
    'UncertaintyPredictionHead',
    
    # Attention components
    'GraphAttention',
    'AttentionPooling',
    'CrossAttention',
    'SelfAttention',
    'PositionalEncoding'
]

# Version info
__version__ = "1.0.0"
__author__ = "DeepDTA-Pro Development Team"
__description__ = "Advanced Graph Neural Networks for Drug-Target Binding Affinity Prediction"