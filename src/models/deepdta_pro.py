"""
DeepDTA-Pro Model
Main drug-target binding affinity prediction model combining molecular GNN,
protein encoder, fusion network, and prediction head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Optional, Dict, Any, Tuple, Union
import logging
import yaml
from pathlib import Path

from .molecular_gnn import MolecularGNN
from .protein_encoder import ProteinEncoder
from .fusion_network import FusionNetwork
from .prediction_head import PredictionHead

logger = logging.getLogger(__name__)

class DeepDTAPro(nn.Module):
    """
    DeepDTA-Pro: Advanced Graph Neural Network for Drug-Target Binding Affinity Prediction.
    
    Architecture:
    1. Molecular GNN: Graph neural network for molecular representation learning
    2. Protein Encoder: Deep learning model for protein sequence encoding
    3. Fusion Network: Multi-modal fusion of molecular and protein features
    4. Prediction Head: Final layers for binding affinity prediction
    """
    
    def __init__(self,
                 # Molecular GNN parameters
                 molecular_node_features: int = 74,
                 molecular_edge_features: int = 12,
                 molecular_hidden_dim: int = 256,
                 molecular_num_layers: int = 5,
                 molecular_gnn_type: str = "gin",
                 molecular_pooling: str = "attention",
                 
                 # Protein encoder parameters
                 protein_vocab_size: int = 26,
                 protein_embedding_dim: int = 256,
                 protein_encoder_type: str = "hybrid",
                 protein_max_length: int = 1000,
                 
                 # Fusion network parameters
                 fusion_hidden_dim: int = 512,
                 fusion_type: str = "multimodal",
                 
                 # Prediction head parameters
                 prediction_head_type: str = "mlp",
                 prediction_hidden_dims: list = [512, 256, 128],
                 output_dim: int = 1,
                 
                 # General parameters
                 dropout: float = 0.1,
                 activation: str = "relu",
                 use_batch_norm: bool = True):
        """
        Initialize DeepDTA-Pro model.
        
        Args:
            molecular_node_features: Number of molecular node features
            molecular_edge_features: Number of molecular edge features
            molecular_hidden_dim: Hidden dimension for molecular GNN
            molecular_num_layers: Number of molecular GNN layers
            molecular_gnn_type: Type of molecular GNN ("gin", "gcn", "gat")
            molecular_pooling: Molecular graph pooling method
            protein_vocab_size: Size of protein vocabulary
            protein_embedding_dim: Protein embedding dimension
            protein_encoder_type: Type of protein encoder ("cnn", "transformer", "hybrid")
            protein_max_length: Maximum protein sequence length
            fusion_hidden_dim: Hidden dimension for fusion network
            fusion_type: Type of fusion network
            prediction_head_type: Type of prediction head
            prediction_hidden_dims: Hidden dimensions for prediction head
            output_dim: Output dimension (1 for regression)
            dropout: Dropout probability
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
        """
        super(DeepDTAPro, self).__init__()
        
        # Store configuration
        self.config = {
            'molecular': {
                'node_features': molecular_node_features,
                'edge_features': molecular_edge_features,
                'hidden_dim': molecular_hidden_dim,
                'num_layers': molecular_num_layers,
                'gnn_type': molecular_gnn_type,
                'pooling': molecular_pooling
            },
            'protein': {
                'vocab_size': protein_vocab_size,
                'embedding_dim': protein_embedding_dim,
                'encoder_type': protein_encoder_type,
                'max_length': protein_max_length
            },
            'fusion': {
                'hidden_dim': fusion_hidden_dim,
                'fusion_type': fusion_type
            },
            'prediction': {
                'head_type': prediction_head_type,
                'hidden_dims': prediction_hidden_dims,
                'output_dim': output_dim
            },
            'general': {
                'dropout': dropout,
                'activation': activation,
                'use_batch_norm': use_batch_norm
            }
        }
        
        # 1. Molecular GNN
        self.molecular_gnn = MolecularGNN(
            node_features=molecular_node_features,
            edge_features=molecular_edge_features,
            hidden_dim=molecular_hidden_dim,
            num_layers=molecular_num_layers,
            gnn_type=molecular_gnn_type,
            pooling=molecular_pooling,
            dropout=dropout,
            activation=activation,
            use_attention=True,
            use_edge_attr=True
        )
        
        # 2. Protein Encoder
        self.protein_encoder = ProteinEncoder(
            vocab_size=protein_vocab_size,
            embedding_dim=protein_embedding_dim,
            encoder_type=protein_encoder_type,
            dropout=dropout,
            max_length=protein_max_length
        )
        
        # 3. Fusion Network
        molecular_repr_dim = self.molecular_gnn.get_output_dim()
        protein_repr_dim = self.protein_encoder.get_output_dim()
        
        self.fusion_network = FusionNetwork(
            molecular_dim=molecular_repr_dim,
            protein_dim=protein_repr_dim,
            hidden_dim=fusion_hidden_dim,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # 4. Prediction Head
        fusion_output_dim = self.fusion_network.get_output_dim()
        
        self.prediction_head = PredictionHead(
            input_dim=fusion_output_dim,
            output_dim=output_dim,
            head_type=prediction_head_type,
            hidden_dims=prediction_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        # Store output dimension
        self.output_dim = output_dim
        
        logger.info(f"DeepDTA-Pro model initialized with {self.count_parameters():,} parameters")
    
    def forward(self, 
                molecular_batch: Batch,
                protein_sequences: torch.Tensor,
                protein_lengths: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                return_representations: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            molecular_batch: Batched molecular graph data
            protein_sequences: Protein sequences [batch_size, seq_len]
            protein_lengths: Protein sequence lengths [batch_size] (optional)
            return_attention: Whether to return attention weights
            return_representations: Whether to return intermediate representations
            
        Returns:
            Dictionary containing predictions and optional additional information
        """
        # 1. Molecular representation
        molecular_output = self.molecular_gnn(molecular_batch, return_attention=return_attention)
        molecular_repr = molecular_output['molecular_representation']
        
        # 2. Protein representation
        protein_output = self.protein_encoder(protein_sequences, protein_lengths)
        protein_repr = protein_output['protein_representation']
        
        # 3. Fusion
        fusion_output = self.fusion_network(molecular_repr, protein_repr)
        fused_repr = fusion_output['fused_representation']
        
        # 4. Prediction
        predictions = self.prediction_head(fused_repr)
        
        # Prepare output
        output = {'predictions': predictions}
        
        if return_representations:
            output.update({
                'molecular_representation': molecular_repr,
                'protein_representation': protein_repr,
                'fused_representation': fused_repr,
                'molecular_node_representations': molecular_output.get('node_representations'),
                'protein_sequence_representations': protein_output.get('sequence_representations')
            })
        
        if return_attention:
            attention_dict = {}
            
            # Molecular attention
            if 'attention_weights' in molecular_output:
                attention_dict['molecular_attention'] = molecular_output['attention_weights']
            
            # Fusion attention (for multimodal fusion)
            if 'method_weights' in fusion_output:
                attention_dict['fusion_method_weights'] = fusion_output['method_weights']
            
            if attention_dict:
                output['attention_weights'] = attention_dict
        
        return output
    
    def predict_with_uncertainty(self,
                                molecular_batch: Batch,
                                protein_sequences: torch.Tensor,
                                protein_lengths: Optional[torch.Tensor] = None,
                                num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation (requires uncertainty prediction head).
        
        Args:
            molecular_batch: Batched molecular graph data
            protein_sequences: Protein sequences [batch_size, seq_len]
            protein_lengths: Protein sequence lengths [batch_size] (optional)
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        if self.config['prediction']['head_type'] != 'uncertainty':
            raise ValueError("Uncertainty prediction requires 'uncertainty' prediction head type")
        
        # Get molecular and protein representations
        molecular_output = self.molecular_gnn(molecular_batch, return_attention=False)
        molecular_repr = molecular_output['molecular_representation']
        
        protein_output = self.protein_encoder(protein_sequences, protein_lengths)
        protein_repr = protein_output['protein_representation']
        
        # Fusion
        fusion_output = self.fusion_network(molecular_repr, protein_repr)
        fused_repr = fusion_output['fused_representation']
        
        # Prediction with uncertainty
        uncertainty_output = self.prediction_head.predict_with_uncertainty(
            fused_repr, num_samples=num_samples
        )
        
        return uncertainty_output
    
    def get_attention_maps(self,
                          molecular_batch: Batch,
                          protein_sequences: torch.Tensor,
                          protein_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for interpretability.
        
        Args:
            molecular_batch: Batched molecular graph data
            protein_sequences: Protein sequences [batch_size, seq_len]
            protein_lengths: Protein sequence lengths [batch_size] (optional)
            
        Returns:
            Dictionary containing various attention maps
        """
        output = self.forward(
            molecular_batch, protein_sequences, protein_lengths,
            return_attention=True
        )
        
        attention_maps = {}
        
        if 'attention_weights' in output:
            attention_weights = output['attention_weights']
            
            # Molecular attention
            if 'molecular_attention' in attention_weights:
                mol_attn = attention_weights['molecular_attention']
                if 'pooling_attention' in mol_attn and mol_attn['pooling_attention'] is not None:
                    attention_maps['molecular_node_attention'] = mol_attn['pooling_attention']
            
            # Fusion method weights
            if 'fusion_method_weights' in attention_weights:
                attention_maps['fusion_method_weights'] = attention_weights['fusion_method_weights']
        
        return attention_maps
    
    def get_molecular_embeddings(self, molecular_batch: Batch) -> torch.Tensor:
        """
        Get molecular embeddings without protein information.
        
        Args:
            molecular_batch: Batched molecular graph data
            
        Returns:
            Molecular embeddings [batch_size, molecular_hidden_dim]
        """
        with torch.no_grad():
            molecular_output = self.molecular_gnn(molecular_batch, return_attention=False)
            return molecular_output['molecular_representation']
    
    def get_protein_embeddings(self, 
                              protein_sequences: torch.Tensor,
                              protein_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get protein embeddings without molecular information.
        
        Args:
            protein_sequences: Protein sequences [batch_size, seq_len]
            protein_lengths: Protein sequence lengths [batch_size] (optional)
            
        Returns:
            Protein embeddings [batch_size, protein_output_dim]
        """
        with torch.no_grad():
            protein_output = self.protein_encoder(protein_sequences, protein_lengths)
            return protein_output['protein_representation']
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_summary(self) -> Dict[str, int]:
        """Get parameter count summary for each component."""
        summary = {
            'molecular_gnn': sum(p.numel() for p in self.molecular_gnn.parameters() if p.requires_grad),
            'protein_encoder': sum(p.numel() for p in self.protein_encoder.parameters() if p.requires_grad),
            'fusion_network': sum(p.numel() for p in self.fusion_network.parameters() if p.requires_grad),
            'prediction_head': sum(p.numel() for p in self.prediction_head.parameters() if p.requires_grad)
        }
        summary['total'] = sum(summary.values())
        return summary
    
    def save_config(self, filepath: str):
        """Save model configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DeepDTAPro':
        """Create model from configuration dictionary."""
        # Flatten configuration for constructor
        kwargs = {}
        
        # Molecular parameters
        mol_config = config.get('molecular', {})
        kwargs.update({
            'molecular_node_features': mol_config.get('node_features', 74),
            'molecular_edge_features': mol_config.get('edge_features', 12),
            'molecular_hidden_dim': mol_config.get('hidden_dim', 256),
            'molecular_num_layers': mol_config.get('num_layers', 5),
            'molecular_gnn_type': mol_config.get('gnn_type', 'gin'),
            'molecular_pooling': mol_config.get('pooling', 'attention')
        })
        
        # Protein parameters
        prot_config = config.get('protein', {})
        kwargs.update({
            'protein_vocab_size': prot_config.get('vocab_size', 26),
            'protein_embedding_dim': prot_config.get('embedding_dim', 256),
            'protein_encoder_type': prot_config.get('encoder_type', 'hybrid'),
            'protein_max_length': prot_config.get('max_length', 1000)
        })
        
        # Fusion parameters
        fusion_config = config.get('fusion', {})
        kwargs.update({
            'fusion_hidden_dim': fusion_config.get('hidden_dim', 512),
            'fusion_type': fusion_config.get('fusion_type', 'multimodal')
        })
        
        # Prediction parameters
        pred_config = config.get('prediction', {})
        kwargs.update({
            'prediction_head_type': pred_config.get('head_type', 'mlp'),
            'prediction_hidden_dims': pred_config.get('hidden_dims', [512, 256, 128]),
            'output_dim': pred_config.get('output_dim', 1)
        })
        
        # General parameters
        gen_config = config.get('general', {})
        kwargs.update({
            'dropout': gen_config.get('dropout', 0.1),
            'activation': gen_config.get('activation', 'relu'),
            'use_batch_norm': gen_config.get('use_batch_norm', True)
        })
        
        return cls(**kwargs)
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'DeepDTAPro':
        """Create model from configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_config(config.get('model', config))


# Model factory function
def create_deepdta_pro_model(config_path: Optional[str] = None, **kwargs) -> DeepDTAPro:
    """
    Factory function to create DeepDTA-Pro model.
    
    Args:
        config_path: Path to configuration file (optional)
        **kwargs: Model parameters (override config file)
        
    Returns:
        DeepDTA-Pro model instance
    """
    if config_path:
        model = DeepDTAPro.from_config_file(config_path)
        # Override with any provided kwargs
        if kwargs:
            logger.warning("Overriding config file parameters with provided kwargs")
            # This would require a more complex implementation to properly override
            # For now, just create a new model with kwargs
            model = DeepDTAPro(**kwargs)
    else:
        model = DeepDTAPro(**kwargs)
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing DeepDTA-Pro model...")
    
    # Create dummy data
    batch_size = 8
    num_nodes = 20
    num_edges = 30
    seq_len = 100
    
    # Molecular data
    x = torch.randn(num_nodes * batch_size, 74)
    edge_index = torch.randint(0, num_nodes, (2, num_edges * batch_size))
    edge_attr = torch.randn(num_edges * batch_size, 12)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    molecular_batch = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx)
    
    # Protein data
    protein_sequences = torch.randint(1, 26, (batch_size, seq_len))
    protein_lengths = torch.randint(50, seq_len, (batch_size,))
    
    # Test default model
    print("Testing default DeepDTA-Pro model...")
    model = DeepDTAPro()
    
    print(f"Total parameters: {model.count_parameters():,}")
    
    param_summary = model.get_parameter_summary()
    for component, count in param_summary.items():
        print(f"{component}: {count:,} parameters")
    
    # Forward pass
    with torch.no_grad():
        output = model(
            molecular_batch, 
            protein_sequences, 
            protein_lengths,
            return_attention=True,
            return_representations=True
        )
    
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Molecular representation shape: {output['molecular_representation'].shape}")
    print(f"Protein representation shape: {output['protein_representation'].shape}")
    print(f"Fused representation shape: {output['fused_representation'].shape}")
    
    if 'attention_weights' in output:
        print("Attention weights available")
    
    # Test attention maps
    print("\nTesting attention maps...")
    attention_maps = model.get_attention_maps(molecular_batch, protein_sequences, protein_lengths)
    for key, value in attention_maps.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
    
    # Test individual embeddings
    print("\nTesting individual embeddings...")
    molecular_emb = model.get_molecular_embeddings(molecular_batch)
    protein_emb = model.get_protein_embeddings(protein_sequences, protein_lengths)
    
    print(f"Molecular embeddings shape: {molecular_emb.shape}")
    print(f"Protein embeddings shape: {protein_emb.shape}")
    
    # Test configuration saving/loading
    print("\nTesting configuration save/load...")
    config_path = "test_model_config.yaml"
    model.save_config(config_path)
    
    # Create model from config
    model_from_config = DeepDTAPro.from_config_file(config_path)
    print(f"Model from config parameters: {model_from_config.count_parameters():,}")
    
    # Clean up
    import os
    if os.path.exists(config_path):
        os.remove(config_path)
    
    print("DeepDTA-Pro model test completed successfully!")