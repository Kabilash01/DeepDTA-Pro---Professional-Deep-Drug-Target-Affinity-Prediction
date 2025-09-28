"""
Prediction Head
Final layers for drug-target binding affinity prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)

class MLPPredictionHead(nn.Module):
    """
    Multi-layer perceptron prediction head.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 1,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 use_skip_connections: bool = True):
        """
        Initialize MLP prediction head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for regression, num_classes for classification)
            activation: Activation function
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_skip_connections: Whether to use skip connections
        """
        super(MLPPredictionHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_skip_connections = use_skip_connections
        
        # Build layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer_modules = []
            
            # Linear layer
            layer_modules.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layer_modules.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layer_modules.append(self._get_activation(activation))
            
            # Dropout
            if dropout > 0:
                layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
            
            # Skip connection projection (if dimensions don't match)
            if use_skip_connections and prev_dim != hidden_dim and i % 2 == 1:
                skip_proj = nn.Linear(prev_dim, hidden_dim)
                setattr(self, f'skip_proj_{i}', skip_proj)
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self.reset_parameters()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        elif activation.lower() == "elu":
            return nn.ELU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for layer in self.layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        residual = x
        
        for i, layer in enumerate(self.layers):
            out = layer(x)
            
            # Skip connection (every 2 layers)
            if self.use_skip_connections and i > 0 and i % 2 == 1:
                if hasattr(self, f'skip_proj_{i}'):
                    residual = getattr(self, f'skip_proj_{i}')(residual)
                
                if residual.shape == out.shape:
                    out = out + residual
            
            x = out
            residual = x
        
        # Output layer
        output = self.output_layer(x)
        
        return output


class AttentionPredictionHead(nn.Module):
    """
    Attention-based prediction head with interpretable attention weights.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 output_dim: int = 1,
                 dropout: float = 0.1):
        """
        Initialize attention prediction head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(AttentionPredictionHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and optional attention weights
        """
        # Input projection
        x = self.input_proj(x)  # [batch_size, hidden_dim]
        x = x.unsqueeze(1)      # [batch_size, 1, hidden_dim] (add sequence dimension)
        
        # Self-attention
        attn_out, attention_weights = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output layer
        output = self.output_layer(x)  # [batch_size, output_dim]
        
        result = {'predictions': output}
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result


class EnsemblePredictionHead(nn.Module):
    """
    Ensemble prediction head combining multiple prediction strategies.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_heads: int = 3,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 ensemble_method: str = "average"):
        """
        Initialize ensemble prediction head.
        
        Args:
            input_dim: Input feature dimension
            num_heads: Number of prediction heads
            hidden_dims: Hidden dimensions for MLP heads
            output_dim: Output dimension
            dropout: Dropout probability
            ensemble_method: Method to combine predictions ("average", "weighted", "learned")
        """
        super(EnsemblePredictionHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.ensemble_method = ensemble_method.lower()
        
        # Multiple prediction heads
        self.prediction_heads = nn.ModuleList()
        
        for i in range(num_heads):
            # Create different architectures for diversity
            if i == 0:
                # Standard MLP
                head = MLPPredictionHead(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    dropout=dropout
                )
            elif i == 1:
                # MLP with different hidden dimensions
                alt_hidden_dims = [dim * 2 for dim in hidden_dims[:-1]] + [hidden_dims[-1]]
                head = MLPPredictionHead(
                    input_dim=input_dim,
                    hidden_dims=alt_hidden_dims,
                    output_dim=output_dim,
                    dropout=dropout,
                    activation="gelu"
                )
            else:
                # MLP with skip connections
                head = MLPPredictionHead(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims + [hidden_dims[-1] // 2],
                    output_dim=output_dim,
                    dropout=dropout,
                    use_skip_connections=True
                )
            
            self.prediction_heads.append(head)
        
        # Ensemble combination
        if ensemble_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(num_heads))
        elif ensemble_method == "learned":
            self.ensemble_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], num_heads),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary containing ensemble predictions
        """
        # Get predictions from all heads
        head_predictions = []
        for head in self.prediction_heads:
            pred = head(x)
            head_predictions.append(pred)
        
        head_stack = torch.stack(head_predictions, dim=1)  # [batch_size, num_heads, output_dim]
        
        # Combine predictions
        if self.ensemble_method == "average":
            ensemble_pred = head_stack.mean(dim=1)
            weights = torch.ones(self.num_heads, device=x.device) / self.num_heads
            
        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_pred = (head_stack * weights.view(1, -1, 1)).sum(dim=1)
            
        elif self.ensemble_method == "learned":
            weights = self.ensemble_network(x)  # [batch_size, num_heads]
            ensemble_pred = (head_stack * weights.unsqueeze(-1)).sum(dim=1)
            weights = weights.mean(dim=0)  # Average weights across batch for reporting
        
        return {
            'predictions': ensemble_pred,
            'head_predictions': head_predictions,
            'ensemble_weights': weights
        }


class UncertaintyPredictionHead(nn.Module):
    """
    Prediction head with uncertainty estimation using Monte Carlo Dropout.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 1,
                 dropout: float = 0.2,
                 activation: str = "relu"):
        """
        Initialize uncertainty prediction head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability (higher for better uncertainty estimation)
            activation: Activation function
        """
        super(UncertaintyPredictionHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # Build layers with dropout
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)  # Always apply dropout for uncertainty
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        for layer in self.layers:
            x = layer(x)
        
        output = self.output_layer(x)
        return output
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout.
        
        Args:
            x: Input features [batch_size, input_dim]
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing mean predictions, uncertainties, and samples
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)  # [batch_size, output_dim]
        std_pred = predictions.std(dim=0)    # [batch_size, output_dim]
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_pred
        
        return {
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'all_predictions': predictions
        }


class PredictionHead(nn.Module):
    """
    Main prediction head that can switch between different prediction strategies.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 head_type: str = "mlp",
                 **kwargs):
        """
        Initialize prediction head.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            head_type: Type of prediction head ("mlp", "attention", "ensemble", "uncertainty")
            **kwargs: Additional arguments for specific head types
        """
        super(PredictionHead, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_type = head_type.lower()
        
        if self.head_type == "mlp":
            self.head = MLPPredictionHead(
                input_dim=input_dim,
                output_dim=output_dim,
                **kwargs
            )
            
        elif self.head_type == "attention":
            self.head = AttentionPredictionHead(
                input_dim=input_dim,
                output_dim=output_dim,
                **kwargs
            )
            
        elif self.head_type == "ensemble":
            self.head = EnsemblePredictionHead(
                input_dim=input_dim,
                output_dim=output_dim,
                **kwargs
            )
            
        elif self.head_type == "uncertainty":
            self.head = UncertaintyPredictionHead(
                input_dim=input_dim,
                output_dim=output_dim,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            **kwargs: Additional arguments for specific head types
            
        Returns:
            Predictions or dictionary containing predictions and additional info
        """
        if self.head_type in ["attention", "ensemble"]:
            output = self.head(x, **kwargs)
            return output['predictions'] if isinstance(output, dict) else output
        else:
            return self.head(x, **kwargs)
    
    def predict_with_uncertainty(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation (only for uncertainty head).
        
        Args:
            x: Input features [batch_size, input_dim]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        if self.head_type == "uncertainty":
            return self.head.predict_with_uncertainty(x, **kwargs)
        else:
            raise ValueError(f"Uncertainty prediction not supported for {self.head_type} head")
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights (only for attention head).
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Attention weights
        """
        if self.head_type == "attention":
            output = self.head(x, return_attention=True)
            return output['attention_weights']
        else:
            raise ValueError(f"Attention weights not available for {self.head_type} head")


# Example usage and testing
if __name__ == "__main__":
    print("Testing PredictionHead...")
    
    # Create dummy data
    batch_size = 32
    input_dim = 512
    output_dim = 1
    
    x = torch.randn(batch_size, input_dim)
    
    # Test different head types
    head_types = ["mlp", "attention", "ensemble", "uncertainty"]
    
    for head_type in head_types:
        print(f"\nTesting {head_type} prediction head...")
        
        pred_head = PredictionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            head_type=head_type
        )
        
        print(f"Model parameters: {sum(p.numel() for p in pred_head.parameters()):,}")
        
        with torch.no_grad():
            predictions = pred_head(x)
        
        print(f"Predictions shape: {predictions.shape}")
        
        # Test specific features
        if head_type == "uncertainty":
            print("Testing uncertainty estimation...")
            uncertainty_output = pred_head.predict_with_uncertainty(x, num_samples=10)
            print(f"Mean predictions shape: {uncertainty_output['mean_predictions'].shape}")
            print(f"Uncertainty shape: {uncertainty_output['epistemic_uncertainty'].shape}")
        
        elif head_type == "attention":
            print("Testing attention weights...")
            try:
                attention_weights = pred_head.get_attention_weights(x)
                print(f"Attention weights shape: {attention_weights.shape}")
            except Exception as e:
                print(f"Error getting attention weights: {e}")
    
    print("PredictionHead test completed successfully!")