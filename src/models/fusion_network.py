"""
Fusion Network
Networks for combining molecular and protein representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math
import logging

from .attention_layers import CrossAttention

logger = logging.getLogger(__name__)

class BilinearFusion(nn.Module):
    """
    Bilinear fusion for combining molecular and protein representations.
    """
    
    def __init__(self, 
                 molecular_dim: int,
                 protein_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.1):
        """
        Initialize bilinear fusion.
        
        Args:
            molecular_dim: Molecular representation dimension
            protein_dim: Protein representation dimension
            hidden_dim: Hidden dimension for fusion
            dropout: Dropout probability
        """
        super(BilinearFusion, self).__init__()
        
        self.molecular_dim = molecular_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        # Bilinear transformation
        self.bilinear = nn.Bilinear(molecular_dim, protein_dim, hidden_dim)
        
        # Individual projections
        self.molecular_proj = nn.Linear(molecular_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, molecular_repr: torch.Tensor, protein_repr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            molecular_repr: Molecular representations [batch_size, molecular_dim]
            protein_repr: Protein representations [batch_size, protein_dim]
            
        Returns:
            Fused representations [batch_size, hidden_dim]
        """
        # Bilinear interaction
        bilinear_out = self.bilinear(molecular_repr, protein_repr)
        
        # Individual projections  
        mol_proj = self.molecular_proj(molecular_repr)
        prot_proj = self.protein_proj(protein_repr)
        
        # Combine all terms
        fused = bilinear_out + mol_proj + prot_proj
        
        # Layer normalization and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based fusion using cross-attention mechanisms.
    """
    
    def __init__(self,
                 molecular_dim: int,
                 protein_dim: int,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize attention fusion.
        
        Args:
            molecular_dim: Molecular representation dimension
            protein_dim: Protein representation dimension
            hidden_dim: Hidden dimension for fusion
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(AttentionFusion, self).__init__()
        
        self.molecular_dim = molecular_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.molecular_proj = nn.Linear(molecular_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # Cross-attention layers
        self.mol_to_prot_attention = CrossAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.prot_to_mol_attention = CrossAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, molecular_repr: torch.Tensor, protein_repr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            molecular_repr: Molecular representations [batch_size, molecular_dim]
            protein_repr: Protein representations [batch_size, protein_dim]
            
        Returns:
            Fused representations [batch_size, hidden_dim]
        """
        batch_size = molecular_repr.size(0)
        
        # Project to common dimension
        mol_proj = self.molecular_proj(molecular_repr)  # [batch_size, hidden_dim]
        prot_proj = self.protein_proj(protein_repr)     # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention
        mol_proj = mol_proj.unsqueeze(1)   # [batch_size, 1, hidden_dim]
        prot_proj = prot_proj.unsqueeze(1) # [batch_size, 1, hidden_dim]
        
        # Cross-attention
        mol_attended, _ = self.mol_to_prot_attention(mol_proj, prot_proj, prot_proj)
        prot_attended, _ = self.prot_to_mol_attention(prot_proj, mol_proj, mol_proj)
        
        # Remove sequence dimension
        mol_attended = mol_attended.squeeze(1)   # [batch_size, hidden_dim]
        prot_attended = prot_attended.squeeze(1) # [batch_size, hidden_dim]
        
        # Concatenate and fuse
        combined = torch.cat([mol_attended, prot_attended], dim=-1)
        fused = self.fusion_layer(combined)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion with learned gate weights.
    """
    
    def __init__(self,
                 molecular_dim: int,
                 protein_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.1):
        """
        Initialize gated fusion.
        
        Args:
            molecular_dim: Molecular representation dimension
            protein_dim: Protein representation dimension  
            hidden_dim: Hidden dimension for fusion
            dropout: Dropout probability
        """
        super(GatedFusion, self).__init__()
        
        self.molecular_dim = molecular_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.molecular_proj = nn.Linear(molecular_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(molecular_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Interaction layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, molecular_repr: torch.Tensor, protein_repr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            molecular_repr: Molecular representations [batch_size, molecular_dim]
            protein_repr: Protein representations [batch_size, protein_dim]
            
        Returns:
            Fused representations [batch_size, hidden_dim]
        """
        # Project to common dimension
        mol_proj = self.molecular_proj(molecular_repr)
        prot_proj = self.protein_proj(protein_repr)
        
        # Compute gate weights
        gate_input = torch.cat([molecular_repr, protein_repr], dim=-1)
        gate_weights = self.gate_network(gate_input)
        
        # Gated combination
        gated_repr = gate_weights * mol_proj + (1 - gate_weights) * prot_proj
        
        # Apply interaction layer
        fused = self.interaction_layer(gated_repr)
        
        return fused


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion combining multiple fusion strategies.
    """
    
    def __init__(self,
                 molecular_dim: int,
                 protein_dim: int,
                 hidden_dim: int,
                 fusion_methods: List[str] = ["bilinear", "attention", "gated"],
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multi-modal fusion.
        
        Args:
            molecular_dim: Molecular representation dimension
            protein_dim: Protein representation dimension
            hidden_dim: Hidden dimension for fusion
            fusion_methods: List of fusion methods to combine
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiModalFusion, self).__init__()
        
        self.molecular_dim = molecular_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.fusion_methods = fusion_methods
        
        # Initialize fusion modules
        self.fusion_modules = nn.ModuleDict()
        
        if "bilinear" in fusion_methods:
            self.fusion_modules["bilinear"] = BilinearFusion(
                molecular_dim, protein_dim, hidden_dim, dropout
            )
        
        if "attention" in fusion_methods:
            self.fusion_modules["attention"] = AttentionFusion(
                molecular_dim, protein_dim, hidden_dim, num_heads, dropout
            )
        
        if "gated" in fusion_methods:
            self.fusion_modules["gated"] = GatedFusion(
                molecular_dim, protein_dim, hidden_dim, dropout
            )
        
        # Combination layer
        num_methods = len(fusion_methods)
        self.combination_layer = nn.Sequential(
            nn.Linear(num_methods * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Method weights (learnable)
        self.method_weights = nn.Parameter(torch.ones(num_methods))
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, molecular_repr: torch.Tensor, protein_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            molecular_repr: Molecular representations [batch_size, molecular_dim]
            protein_repr: Protein representations [batch_size, protein_dim]
            
        Returns:
            Dictionary containing fused representations and method outputs
        """
        method_outputs = {}
        fusion_results = []
        
        # Apply each fusion method
        for i, method in enumerate(self.fusion_methods):
            fusion_module = self.fusion_modules[method]
            method_output = fusion_module(molecular_repr, protein_repr)
            
            # Weight by learnable method weights
            weighted_output = self.method_weights[i] * method_output
            
            method_outputs[method] = method_output
            fusion_results.append(weighted_output)
        
        # Combine all methods
        if len(fusion_results) == 1:
            combined = fusion_results[0]
        else:
            combined = torch.cat(fusion_results, dim=-1)
            combined = self.combination_layer(combined)
        
        return {
            'fused_representation': combined,
            'method_outputs': method_outputs,
            'method_weights': F.softmax(self.method_weights, dim=0)
        }


class FusionNetwork(nn.Module):
    """
    Main fusion network that can switch between different fusion strategies.
    """
    
    def __init__(self,
                 molecular_dim: int,
                 protein_dim: int,
                 hidden_dim: int = 512,
                 fusion_type: str = "multimodal",
                 **kwargs):
        """
        Initialize fusion network.
        
        Args:
            molecular_dim: Molecular representation dimension
            protein_dim: Protein representation dimension
            hidden_dim: Hidden dimension for fusion
            fusion_type: Type of fusion ("bilinear", "attention", "gated", "multimodal")
            **kwargs: Additional arguments for specific fusion methods
        """
        super(FusionNetwork, self).__init__()
        
        self.molecular_dim = molecular_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type.lower()
        
        if self.fusion_type == "bilinear":
            self.fusion = BilinearFusion(molecular_dim, protein_dim, hidden_dim, **kwargs)
            
        elif self.fusion_type == "attention":
            self.fusion = AttentionFusion(molecular_dim, protein_dim, hidden_dim, **kwargs)
            
        elif self.fusion_type == "gated":
            self.fusion = GatedFusion(molecular_dim, protein_dim, hidden_dim, **kwargs)
            
        elif self.fusion_type == "multimodal":
            self.fusion = MultiModalFusion(molecular_dim, protein_dim, hidden_dim, **kwargs)
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        self.output_dim = self.fusion.output_dim
    
    def forward(self, molecular_repr: torch.Tensor, protein_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            molecular_repr: Molecular representations [batch_size, molecular_dim]
            protein_repr: Protein representations [batch_size, protein_dim]
            
        Returns:
            Dictionary containing fusion results
        """
        if self.fusion_type == "multimodal":
            return self.fusion(molecular_repr, protein_repr)
        else:
            fused_repr = self.fusion(molecular_repr, protein_repr)
            return {'fused_representation': fused_repr}
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to select appropriate fusion strategies
    based on input characteristics.
    """
    
    def __init__(self,
                 molecular_dim: int,
                 protein_dim: int,
                 hidden_dim: int = 512,
                 num_experts: int = 3,
                 dropout: float = 0.1):
        """
        Initialize adaptive fusion.
        
        Args:
            molecular_dim: Molecular representation dimension
            protein_dim: Protein representation dimension
            hidden_dim: Hidden dimension for fusion
            num_experts: Number of expert fusion networks
            dropout: Dropout probability
        """
        super(AdaptiveFusion, self).__init__()
        
        self.molecular_dim = molecular_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Expert fusion networks
        self.experts = nn.ModuleList([
            BilinearFusion(molecular_dim, protein_dim, hidden_dim, dropout),
            AttentionFusion(molecular_dim, protein_dim, hidden_dim, dropout=dropout),
            GatedFusion(molecular_dim, protein_dim, hidden_dim, dropout)
        ])
        
        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(molecular_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, molecular_repr: torch.Tensor, protein_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            molecular_repr: Molecular representations [batch_size, molecular_dim]
            protein_repr: Protein representations [batch_size, protein_dim]
            
        Returns:
            Dictionary containing adaptive fusion results
        """
        batch_size = molecular_repr.size(0)
        
        # Compute gating weights
        gate_input = torch.cat([molecular_repr, protein_repr], dim=-1)
        gate_weights = self.gating_network(gate_input)  # [batch_size, num_experts]
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(molecular_repr, protein_repr)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_stack = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        fused_repr = (expert_stack * gate_weights).sum(dim=1)  # [batch_size, hidden_dim]
        
        return {
            'fused_representation': fused_repr,
            'gate_weights': gate_weights.squeeze(-1),
            'expert_outputs': expert_outputs
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing FusionNetwork...")
    
    # Create dummy data
    batch_size = 16
    molecular_dim = 256
    protein_dim = 512
    hidden_dim = 256
    
    molecular_repr = torch.randn(batch_size, molecular_dim)
    protein_repr = torch.randn(batch_size, protein_dim)
    
    # Test different fusion types
    fusion_types = ["bilinear", "attention", "gated", "multimodal"]
    
    for fusion_type in fusion_types:
        print(f"\nTesting {fusion_type} fusion...")
        
        fusion_net = FusionNetwork(
            molecular_dim=molecular_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            fusion_type=fusion_type
        )
        
        print(f"Model parameters: {sum(p.numel() for p in fusion_net.parameters()):,}")
        
        with torch.no_grad():
            output = fusion_net(molecular_repr, protein_repr)
        
        print(f"Fused representation shape: {output['fused_representation'].shape}")
        print(f"Output dimension: {fusion_net.get_output_dim()}")
        
        if fusion_type == "multimodal":
            print(f"Method weights: {output['method_weights']}")
    
    # Test adaptive fusion
    print(f"\nTesting adaptive fusion...")
    
    adaptive_fusion = AdaptiveFusion(
        molecular_dim=molecular_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim
    )
    
    print(f"Model parameters: {sum(p.numel() for p in adaptive_fusion.parameters()):,}")
    
    with torch.no_grad():
        output = adaptive_fusion(molecular_repr, protein_repr)
    
    print(f"Fused representation shape: {output['fused_representation'].shape}")
    print(f"Gate weights shape: {output['gate_weights'].shape}")
    print(f"Average gate weights: {output['gate_weights'].mean(dim=0)}")
    
    print("FusionNetwork test completed successfully!")