"""
Molecular Graph Neural Network
Graph neural network for processing molecular structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Optional, Dict, Any, Tuple
import logging

from .attention_layers import GraphAttention, AttentionPooling

logger = logging.getLogger(__name__)

class GINLayer(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer with enhanced features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu", dropout: float = 0.1):
        """
        Initialize GIN layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            activation: Activation function name
            dropout: Dropout probability
        """
        super(GINLayer, self).__init__()
        
        # MLP for GIN
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self._get_activation(activation)
        )
        
        # GIN convolution
        self.gin_conv = GINConv(self.mlp, train_eps=True)
        
        # Residual connection (if dimensions match)
        self.use_residual = (input_dim == hidden_dim)
        if not self.use_residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
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
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Apply GIN convolution
        out = self.gin_conv(x, edge_index)
        
        # Residual connection
        if self.use_residual:
            out = out + x
        else:
            out = out + self.residual_proj(x)
        
        return self.dropout(out)


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular representation learning.
    
    Features:
    - Multiple GNN layers (GIN, GCN, GAT)
    - Attention mechanisms for interpretability
    - Hierarchical pooling
    - Skip connections
    - Batch normalization and dropout
    """
    
    def __init__(self, 
                 node_features: int = 74,
                 edge_features: int = 12, 
                 hidden_dim: int = 256,
                 num_layers: int = 5,
                 gnn_type: str = "gin",
                 pooling: str = "attention",
                 dropout: float = 0.1,
                 activation: str = "relu",
                 use_attention: bool = True,
                 use_edge_attr: bool = True):
        """
        Initialize molecular GNN.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ("gin", "gcn", "gat")
            pooling: Pooling method ("attention", "mean", "max", "add")
            dropout: Dropout probability
            activation: Activation function
            use_attention: Whether to use attention mechanisms
            use_edge_attr: Whether to use edge attributes
        """
        super(MolecularGNN, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.pooling = pooling.lower()
        self.use_attention = use_attention
        self.use_edge_attr = use_edge_attr
        
        # Input projection layers
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        if use_edge_attr and edge_features > 0:
            self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if self.gnn_type == "gin":
                layer = GINLayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout
                )
            elif self.gnn_type == "gcn":
                layer = GCNConv(hidden_dim, hidden_dim)
            elif self.gnn_type == "gat":
                layer = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // 8,  # 8 attention heads
                    heads=8,
                    dropout=dropout,
                    concat=True
                )
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}")
            
            self.gnn_layers.append(layer)
        
        # Attention mechanisms
        if use_attention:
            self.graph_attention = GraphAttention(
                input_dim=hidden_dim,
                attention_dim=hidden_dim,
                num_heads=8
            )
        
        # Pooling layer
        if pooling == "attention":
            self.pooling_layer = AttentionPooling(
                input_dim=hidden_dim,
                attention_dim=hidden_dim // 2
            )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_dim
        
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
        else:
            return nn.ReLU()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)
        
        if hasattr(self, 'edge_embedding'):
            nn.init.xavier_uniform_(self.edge_embedding.weight)
            nn.init.zeros_(self.edge_embedding.bias)
    
    def forward(self, batch: Batch, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            batch: Batched graph data from PyTorch Geometric
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing molecular representations and optional attention weights
        """
        x = batch.x  # Node features [num_nodes, node_features]
        edge_index = batch.edge_index  # Edge indices [2, num_edges]
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None  # Edge features
        batch_idx = batch.batch  # Batch indices [num_nodes]
        
        # Input embeddings
        x = self.node_embedding(x)  # [num_nodes, hidden_dim]
        
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)  # [num_edges, hidden_dim]
        
        # Store intermediate representations for skip connections
        layer_outputs = [x]
        attention_weights_list = []
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == "gin":
                x = gnn_layer(x, edge_index, edge_attr)
            elif self.gnn_type == "gcn":
                x = gnn_layer(x, edge_index)
            elif self.gnn_type == "gat":
                x = gnn_layer(x, edge_index)
            
            # Batch normalization
            x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Skip connection (every 2 layers)
            if i > 0 and i % 2 == 1:
                x = x + layer_outputs[-2]
            
            layer_outputs.append(x)
        
        # Apply graph attention if enabled
        graph_attention_weights = None
        if self.use_attention:
            x, graph_attention_weights = self.graph_attention(x, edge_index, batch_idx)
            attention_weights_list.append(graph_attention_weights)
        
        # Pooling to get graph-level representation
        pooling_attention_weights = None
        
        if self.pooling == "attention":
            graph_repr, pooling_attention_weights = self.pooling_layer(x, batch_idx)
            attention_weights_list.append(pooling_attention_weights)
        elif self.pooling == "mean":
            graph_repr = global_mean_pool(x, batch_idx)
        elif self.pooling == "max":
            graph_repr = global_max_pool(x, batch_idx)
        elif self.pooling == "add":
            graph_repr = global_add_pool(x, batch_idx)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Prepare output
        output = {
            'molecular_representation': graph_repr,  # [batch_size, hidden_dim]
            'node_representations': x,               # [num_nodes, hidden_dim]
            'batch_indices': batch_idx               # [num_nodes]
        }
        
        if return_attention:
            output['attention_weights'] = {
                'graph_attention': graph_attention_weights,
                'pooling_attention': pooling_attention_weights
            }
        
        return output
    
    def get_node_attention(self, batch: Batch) -> torch.Tensor:
        """
        Get attention weights for nodes in the molecular graph.
        
        Args:
            batch: Batched graph data
            
        Returns:
            Node attention weights
        """
        output = self.forward(batch, return_attention=True)
        return output['attention_weights']['pooling_attention']
    
    def get_molecular_fingerprint(self, batch: Batch) -> torch.Tensor:
        """
        Get molecular fingerprint (graph-level representation).
        
        Args:
            batch: Batched graph data
            
        Returns:
            Molecular fingerprints [batch_size, hidden_dim]
        """
        output = self.forward(batch, return_attention=False)
        return output['molecular_representation']
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class MolecularGraphConv(nn.Module):
    """
    Alternative molecular graph convolution with explicit edge handling.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, activation: str = "relu"):
        """
        Initialize molecular graph convolution.
        
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            activation: Activation function
        """
        super(MolecularGraphConv, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Node update networks
        self.node_network = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge network
        self.edge_network = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message aggregation
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.GELU()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        row, col = edge_index
        
        # Compute edge features
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_features = self.edge_network(edge_input)  # [num_edges, hidden_dim]
        
        # Aggregate messages
        messages = self.message_mlp(edge_features)  # [num_edges, hidden_dim]
        
        # Aggregate messages for each node
        num_nodes = x.size(0)
        aggregated_messages = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        aggregated_messages.index_add_(0, row, messages)
        
        # Update nodes
        node_input = torch.cat([x, aggregated_messages], dim=-1)
        updated_x = self.node_network(node_input)
        
        return updated_x


# Example usage and testing
if __name__ == "__main__":
    from torch_geometric.data import Data
    
    print("Testing MolecularGNN...")
    
    # Create dummy molecular graph data
    num_nodes = 20
    num_edges = 30
    batch_size = 4
    
    # Node features (atomic features)
    x = torch.randn(num_nodes * batch_size, 74)
    
    # Edge indices and features
    edge_index = torch.randint(0, num_nodes, (2, num_edges * batch_size))
    edge_attr = torch.randn(num_edges * batch_size, 12)
    
    # Batch indices
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Create batch
    batch_data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # Initialize model
    model = MolecularGNN(
        node_features=74,
        edge_features=12,
        hidden_dim=256,
        num_layers=5,
        gnn_type="gin",
        pooling="attention",
        use_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(batch_data, return_attention=True)
    
    print(f"Molecular representation shape: {output['molecular_representation'].shape}")
    print(f"Node representations shape: {output['node_representations'].shape}")
    print(f"Output dimension: {model.get_output_dim()}")
    
    # Test attention
    if 'attention_weights' in output:
        if output['attention_weights']['pooling_attention'] is not None:
            print(f"Pooling attention shape: {output['attention_weights']['pooling_attention'].shape}")
    
    print("MolecularGNN test completed successfully!")