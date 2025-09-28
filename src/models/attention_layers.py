"""
Attention Mechanisms
Various attention layers for molecular and protein representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Union
import math

class GraphAttention(nn.Module):
    """
    Graph attention mechanism for molecular graphs.
    """
    
    def __init__(self, input_dim: int, attention_dim: int = 128, num_heads: int = 4):
        """
        Initialize graph attention layer.
        
        Args:
            input_dim: Input feature dimension
            attention_dim: Attention dimension
            num_heads: Number of attention heads
        """
        super(GraphAttention, self).__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        # Attention projection layers
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        
        # Output projection
        self.output_proj = nn.Linear(attention_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        num_nodes = x.size(0)
        
        # Project to query, key, value
        q = self.query_proj(x)  # [num_nodes, attention_dim]
        k = self.key_proj(x)    # [num_nodes, attention_dim]
        v = self.value_proj(x)  # [num_nodes, attention_dim]
        
        # Reshape for multi-head attention
        q = q.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        k = k.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        v = v.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        
        # Compute attention scores for connected nodes
        row, col = edge_index
        
        # Get query and key for edges
        q_edges = q[row]  # [num_edges, num_heads, head_dim]
        k_edges = k[col]  # [num_edges, num_heads, head_dim]
        
        # Compute attention scores
        attention_scores = (q_edges * k_edges).sum(dim=-1) / math.sqrt(self.head_dim)  # [num_edges, num_heads]
        
        # Apply softmax over edges for each node
        attention_weights = softmax(attention_scores, row, num_nodes=num_nodes)  # [num_edges, num_heads]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        v_edges = v[col]  # [num_edges, num_heads, head_dim]
        attended_values = attention_weights.unsqueeze(-1) * v_edges  # [num_edges, num_heads, head_dim]
        
        # Aggregate attended values for each node
        attended_features = torch.zeros(num_nodes, self.num_heads, self.head_dim, 
                                       device=x.device, dtype=x.dtype)
        attended_features.index_add_(0, row, attended_values)
        
        # Reshape and project output
        attended_features = attended_features.view(num_nodes, self.attention_dim)  # [num_nodes, attention_dim]
        output = self.output_proj(attended_features)  # [num_nodes, input_dim]
        
        # Residual connection
        output = output + x
        
        # Return average attention weights across heads for interpretability
        avg_attention_weights = attention_weights.mean(dim=1)  # [num_edges]
        
        return output, avg_attention_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for graph-level representations.
    """
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        """
        Initialize attention pooling layer.
        
        Args:
            input_dim: Input feature dimension
            attention_dim: Attention dimension
        """
        super(AttentionPooling, self).__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for layer in self.attention_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Tuple of (pooled_features, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention_layer(x)  # [num_nodes, 1]
        attention_scores = attention_scores.squeeze(-1)  # [num_nodes]
        
        # Apply softmax within each graph
        attention_weights = softmax(attention_scores, batch)  # [num_nodes]
        
        # Weighted sum of node features
        weighted_features = x * attention_weights.unsqueeze(-1)  # [num_nodes, input_dim]
        
        # Pool by batch
        pooled_features = global_add_pool(weighted_features, batch)  # [batch_size, input_dim]
        
        return pooled_features, attention_weights


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism between molecular and protein representations.
    """
    
    def __init__(self, mol_dim: int, prot_dim: int, attention_dim: int = 256, num_heads: int = 8):
        """
        Initialize cross-attention layer.
        
        Args:
            mol_dim: Molecular feature dimension
            prot_dim: Protein feature dimension
            attention_dim: Attention dimension
            num_heads: Number of attention heads
        """
        super(CrossAttention, self).__init__()
        
        self.mol_dim = mol_dim
        self.prot_dim = prot_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        # Projection layers for molecular features (queries)
        self.mol_query_proj = nn.Linear(mol_dim, attention_dim)
        self.mol_key_proj = nn.Linear(mol_dim, attention_dim)
        self.mol_value_proj = nn.Linear(mol_dim, attention_dim)
        
        # Projection layers for protein features (keys and values)
        self.prot_key_proj = nn.Linear(prot_dim, attention_dim)
        self.prot_value_proj = nn.Linear(prot_dim, attention_dim)
        
        # Output projections
        self.mol_output_proj = nn.Linear(attention_dim, mol_dim)
        self.prot_output_proj = nn.Linear(attention_dim, prot_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for module in [self.mol_query_proj, self.mol_key_proj, self.mol_value_proj,
                      self.prot_key_proj, self.prot_value_proj,
                      self.mol_output_proj, self.prot_output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, mol_features: torch.Tensor, 
                prot_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            mol_features: Molecular features [batch_size, mol_dim]
            prot_features: Protein features [batch_size, seq_len, prot_dim]
            
        Returns:
            Tuple of (attended_mol_features, attended_prot_features, attention_weights)
        """
        batch_size, seq_len, _ = prot_features.shape
        
        # Project molecular features
        mol_q = self.mol_query_proj(mol_features)  # [batch_size, attention_dim]
        mol_k = self.mol_key_proj(mol_features)    # [batch_size, attention_dim]
        mol_v = self.mol_value_proj(mol_features)  # [batch_size, attention_dim]
        
        # Project protein features
        prot_k = self.prot_key_proj(prot_features)  # [batch_size, seq_len, attention_dim]
        prot_v = self.prot_value_proj(prot_features)  # [batch_size, seq_len, attention_dim]
        
        # Reshape for multi-head attention
        mol_q = mol_q.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        mol_k = mol_k.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        mol_v = mol_v.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        
        prot_k = prot_k.view(batch_size, seq_len, self.num_heads, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]
        prot_v = prot_v.view(batch_size, seq_len, self.num_heads, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]
        
        # Compute attention from molecule to protein
        mol_to_prot_scores = torch.einsum('bhd,bshd->bsh', mol_q, prot_k) / math.sqrt(self.head_dim)  # [batch_size, seq_len, num_heads]
        mol_to_prot_weights = F.softmax(mol_to_prot_scores, dim=1)  # [batch_size, seq_len, num_heads]
        mol_to_prot_weights = self.dropout(mol_to_prot_weights)
        
        # Attend to protein features
        attended_prot = torch.einsum('bsh,bshd->bhd', mol_to_prot_weights, prot_v)  # [batch_size, num_heads, head_dim]
        attended_prot = attended_prot.view(batch_size, self.attention_dim)  # [batch_size, attention_dim]
        attended_prot = self.mol_output_proj(attended_prot)  # [batch_size, mol_dim]
        
        # Compute attention from protein to molecule
        prot_to_mol_scores = torch.einsum('bshd,bhd->bsh', prot_k, mol_q)  # [batch_size, seq_len, num_heads]
        prot_to_mol_weights = F.softmax(prot_to_mol_scores, dim=-1)  # [batch_size, seq_len, num_heads]
        prot_to_mol_weights = self.dropout(prot_to_mol_weights)
        
        # Attend to molecular features (broadcast to sequence length)
        mol_v_expanded = mol_v.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch_size, seq_len, num_heads, head_dim]
        attended_mol_per_pos = torch.einsum('bsh,bshd->bshd', prot_to_mol_weights, mol_v_expanded)  # [batch_size, seq_len, num_heads, head_dim]
        attended_mol_per_pos = attended_mol_per_pos.view(batch_size, seq_len, self.attention_dim)  # [batch_size, seq_len, attention_dim]
        attended_mol_per_pos = self.prot_output_proj(attended_mol_per_pos)  # [batch_size, seq_len, prot_dim]
        
        # Residual connections
        attended_mol_features = attended_prot + mol_features
        attended_prot_features = attended_mol_per_pos + prot_features
        
        # Average attention weights across heads for interpretability
        avg_attention_weights = mol_to_prot_weights.mean(dim=-1)  # [batch_size, seq_len]
        
        return attended_mol_features, attended_prot_features, avg_attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence processing.
    """
    
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize self-attention layer.
        
        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(SelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len] (True for valid positions)
            
        Returns:
            Tuple of (output_features, attention_weights)
        """
        # Prepare attention mask (convert boolean mask to additive mask)
        attn_mask = None
        if mask is not None:
            # Convert to additive mask: 0 for valid positions, -inf for invalid
            attn_mask = torch.where(mask, 0.0, float('-inf'))
            attn_mask = attn_mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # [batch_size, seq_len, seq_len]
        
        # Self-attention
        attended_x, attention_weights = self.multihead_attn(
            query=x, key=x, value=x, 
            attn_mask=attn_mask,
            need_weights=True
        )
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + self.dropout(attended_x))
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model]
            
        Returns:
            Input with positional encoding added
        """
        if x.dim() == 3 and x.size(0) != self.pe.size(0):
            # Assume batch_first format [batch_size, seq_len, d_model]
            seq_len = x.size(1)
            return x + self.pe[:seq_len, :].transpose(0, 1).unsqueeze(0)
        else:
            # Assume seq_first format [seq_len, batch_size, d_model]
            return x + self.pe[:x.size(0), :]


# Example usage and testing
if __name__ == "__main__":
    # Test GraphAttention
    print("Testing GraphAttention...")
    graph_attn = GraphAttention(input_dim=64, attention_dim=128, num_heads=4)
    
    # Create dummy graph data
    num_nodes = 10
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    
    attended_x, attention_weights = graph_attn(x, edge_index)
    print(f"Graph attention output shape: {attended_x.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test AttentionPooling
    print("\nTesting AttentionPooling...")
    attn_pool = AttentionPooling(input_dim=64, attention_dim=32)
    
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])  # 3 graphs
    pooled_features, pool_weights = attn_pool(x, batch)
    print(f"Pooled features shape: {pooled_features.shape}")
    print(f"Pool attention weights shape: {pool_weights.shape}")
    
    # Test CrossAttention
    print("\nTesting CrossAttention...")
    cross_attn = CrossAttention(mol_dim=64, prot_dim=128, attention_dim=256, num_heads=8)
    
    batch_size = 4
    seq_len = 50
    mol_features = torch.randn(batch_size, 64)
    prot_features = torch.randn(batch_size, seq_len, 128)
    
    attended_mol, attended_prot, cross_weights = cross_attn(mol_features, prot_features)
    print(f"Attended molecular features shape: {attended_mol.shape}")
    print(f"Attended protein features shape: {attended_prot.shape}")
    print(f"Cross attention weights shape: {cross_weights.shape}")
    
    print("All attention mechanisms tested successfully!")