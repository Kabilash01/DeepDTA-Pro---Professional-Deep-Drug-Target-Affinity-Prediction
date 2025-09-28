"""
Protein Encoder
Deep learning models for protein sequence encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import logging

from .attention_layers import SelfAttention, PositionalEncoding, CrossAttention

logger = logging.getLogger(__name__)

class ProteinCNN(nn.Module):
    """
    Convolutional Neural Network for protein sequence encoding.
    """
    
    def __init__(self, 
                 vocab_size: int = 26,  # 20 amino acids + special tokens
                 embedding_dim: int = 128,
                 hidden_dims: list = [256, 512, 1024],
                 kernel_sizes: list = [3, 5, 7, 11],
                 dropout: float = 0.1,
                 max_length: int = 1000):
        """
        Initialize protein CNN.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Embedding dimension
            hidden_dims: Hidden dimensions for conv layers
            kernel_sizes: Kernel sizes for convolutions
            dropout: Dropout probability
            max_length: Maximum sequence length
        """
        super(ProteinCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multi-scale CNN layers
        self.conv_layers = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            input_dim = embedding_dim if i == 0 else hidden_dims[i-1]
            
            # Multiple kernel sizes for each layer
            multi_kernel_convs = nn.ModuleList()
            for kernel_size in kernel_sizes:
                conv = nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim // len(kernel_sizes), 
                             kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(hidden_dim // len(kernel_sizes)),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                multi_kernel_convs.append(conv)
            
            self.conv_layers.append(multi_kernel_convs)
        
        # Output dimension
        self.output_dim = hidden_dims[-1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Protein sequences [batch_size, seq_len]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Protein representations [batch_size, hidden_dims[-1]]
        """
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # Apply multi-scale convolutions
        for conv_group in self.conv_layers:
            # Apply different kernel sizes and concatenate
            conv_outputs = []
            for conv in conv_group:
                conv_out = conv(x)
                conv_outputs.append(conv_out)
            
            x = torch.cat(conv_outputs, dim=1)  # Concatenate along channel dimension
        
        # Global pooling
        x = self.global_pool(x)  # [batch_size, hidden_dims[-1], 1]
        x = x.squeeze(-1)       # [batch_size, hidden_dims[-1]]
        
        return x


class ProteinTransformer(nn.Module):
    """
    Transformer-based protein encoder with self-attention.
    """
    
    def __init__(self,
                 vocab_size: int = 26,
                 embedding_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 hidden_dim: int = 2048,
                 dropout: float = 0.1,
                 max_length: int = 1000,
                 use_positional_encoding: bool = True):
        """
        Initialize protein transformer.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension in feed-forward network
            dropout: Dropout probability
            max_length: Maximum sequence length
            use_positional_encoding: Whether to use positional encoding
        """
        super(ProteinTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.use_positional_encoding = use_positional_encoding
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_scale = math.sqrt(embedding_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embedding_dim, max_length, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = embedding_dim
    
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            x: Input sequences [batch_size, seq_len]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Padding mask [batch_size, seq_len]
        """
        if lengths is not None:
            batch_size, seq_len = x.shape
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
            return mask
        else:
            # Use padding token (0) to create mask
            return (x == 0)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Protein sequences [batch_size, seq_len]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Dictionary containing protein representations
        """
        # Create padding mask
        padding_mask = self.create_padding_mask(x, lengths)
        
        # Embedding
        x = self.embedding(x) * self.embedding_scale  # [batch_size, seq_len, embedding_dim]
        
        # Positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, src_key_padding_mask=padding_mask)
        
        x = self.layer_norm(x)
        
        # Global representation (mean pooling over non-padded positions)
        if lengths is not None:
            # Use actual lengths for mean pooling
            mask = ~padding_mask  # Inverse mask for valid positions
            global_repr = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        else:
            # Use all positions for mean pooling
            mask = ~padding_mask
            seq_lengths = mask.sum(dim=1)
            global_repr = (x * mask.unsqueeze(-1)).sum(dim=1) / seq_lengths.unsqueeze(-1)
        
        return {
            'sequence_representations': x,        # [batch_size, seq_len, embedding_dim]
            'global_representation': global_repr, # [batch_size, embedding_dim]
            'padding_mask': padding_mask          # [batch_size, seq_len]
        }


class HybridProteinEncoder(nn.Module):
    """
    Hybrid protein encoder combining CNN and Transformer architectures.
    """
    
    def __init__(self,
                 vocab_size: int = 26,
                 embedding_dim: int = 256,
                 cnn_hidden_dims: list = [128, 256, 512],
                 cnn_kernel_sizes: list = [3, 5, 7],
                 transformer_layers: int = 4,
                 transformer_heads: int = 8,
                 dropout: float = 0.1,
                 max_length: int = 1000,
                 fusion_method: str = "concat"):
        """
        Initialize hybrid protein encoder.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Embedding dimension
            cnn_hidden_dims: Hidden dimensions for CNN
            cnn_kernel_sizes: Kernel sizes for CNN
            transformer_layers: Number of transformer layers
            transformer_heads: Number of attention heads
            dropout: Dropout probability
            max_length: Maximum sequence length
            fusion_method: Method to fuse CNN and transformer features ("concat", "add", "gated")
        """
        super(HybridProteinEncoder, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN branch
        self.cnn_encoder = ProteinCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dims=cnn_hidden_dims,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            max_length=max_length
        )
        
        # Transformer branch
        self.transformer_encoder = ProteinTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
            max_length=max_length
        )
        
        # Fusion layer
        cnn_dim = cnn_hidden_dims[-1]
        transformer_dim = embedding_dim
        
        if fusion_method == "concat":
            self.output_dim = cnn_dim + transformer_dim
            self.fusion_layer = nn.Identity()
        elif fusion_method == "add":
            assert cnn_dim == transformer_dim, "Dimensions must match for addition fusion"
            self.output_dim = cnn_dim
            self.fusion_layer = nn.Identity()
        elif fusion_method == "gated":
            self.output_dim = max(cnn_dim, transformer_dim)
            self.gate = nn.Sequential(
                nn.Linear(cnn_dim + transformer_dim, self.output_dim),
                nn.Sigmoid()
            )
            self.cnn_proj = nn.Linear(cnn_dim, self.output_dim) if cnn_dim != self.output_dim else nn.Identity()
            self.transformer_proj = nn.Linear(transformer_dim, self.output_dim) if transformer_dim != self.output_dim else nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Protein sequences [batch_size, seq_len]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Dictionary containing protein representations
        """
        # CNN branch
        cnn_repr = self.cnn_encoder(x, lengths)
        
        # Transformer branch
        transformer_output = self.transformer_encoder(x, lengths)
        transformer_repr = transformer_output['global_representation']
        
        # Fusion
        if self.fusion_method == "concat":
            fused_repr = torch.cat([cnn_repr, transformer_repr], dim=-1)
        elif self.fusion_method == "add":
            fused_repr = cnn_repr + transformer_repr
        elif self.fusion_method == "gated":
            cnn_proj = self.cnn_proj(cnn_repr)
            transformer_proj = self.transformer_proj(transformer_repr)
            gate_weights = self.gate(torch.cat([cnn_repr, transformer_repr], dim=-1))
            fused_repr = gate_weights * cnn_proj + (1 - gate_weights) * transformer_proj
        
        return {
            'protein_representation': fused_repr,      # [batch_size, output_dim]
            'cnn_representation': cnn_repr,            # [batch_size, cnn_dim]
            'transformer_representation': transformer_repr,  # [batch_size, transformer_dim]
            'sequence_representations': transformer_output['sequence_representations'],  # [batch_size, seq_len, embedding_dim]
            'padding_mask': transformer_output['padding_mask']  # [batch_size, seq_len]
        }


class ProteinEncoder(nn.Module):
    """
    Main protein encoder that can switch between different architectures.
    """
    
    def __init__(self,
                 vocab_size: int = 26,
                 embedding_dim: int = 256,
                 encoder_type: str = "hybrid",
                 **kwargs):
        """
        Initialize protein encoder.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Embedding dimension
            encoder_type: Type of encoder ("cnn", "transformer", "hybrid")
            **kwargs: Additional arguments for specific encoders
        """
        super(ProteinEncoder, self).__init__()
        
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == "cnn":
            self.encoder = ProteinCNN(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                **kwargs
            )
            self.output_dim = self.encoder.output_dim
            
        elif self.encoder_type == "transformer":
            self.encoder = ProteinTransformer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                **kwargs
            )
            self.output_dim = self.encoder.output_dim
            
        elif self.encoder_type == "hybrid":
            self.encoder = HybridProteinEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                **kwargs
            )
            self.output_dim = self.encoder.output_dim
            
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Protein sequences [batch_size, seq_len]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Dictionary containing protein representations
        """
        if self.encoder_type == "cnn":
            repr = self.encoder(x, lengths)
            return {'protein_representation': repr}
        else:
            return self.encoder(x, lengths)
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


# Utility functions for protein processing
class AminoAcidVocabulary:
    """Amino acid vocabulary for protein sequence encoding."""
    
    # Standard amino acids
    AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    # Special tokens
    PAD_TOKEN = '<PAD>'    # Padding
    UNK_TOKEN = '<UNK>'    # Unknown amino acid
    START_TOKEN = '<START>'  # Start of sequence
    END_TOKEN = '<END>'    # End of sequence
    MASK_TOKEN = '<MASK>'  # Masked amino acid
    
    def __init__(self):
        # Create vocabulary
        self.vocab = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, 
                     self.END_TOKEN, self.MASK_TOKEN] + self.AMINO_ACIDS
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
    
    def encode(self, sequence: str, add_special_tokens: bool = False) -> list:
        """
        Encode protein sequence to token IDs.
        
        Args:
            sequence: Protein sequence string
            add_special_tokens: Whether to add START/END tokens
            
        Returns:
            List of token IDs
        """
        # Convert to uppercase and split
        tokens = list(sequence.upper())
        
        # Add special tokens if requested
        if add_special_tokens:
            tokens = [self.START_TOKEN] + tokens + [self.END_TOKEN]
        
        # Convert to IDs
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        
        return ids
    
    def decode(self, ids: list, remove_special_tokens: bool = True) -> str:
        """
        Decode token IDs to protein sequence.
        
        Args:
            ids: List of token IDs
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Protein sequence string
        """
        tokens = [self.id_to_token[id] for id in ids]
        
        if remove_special_tokens:
            special_tokens = {self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, 
                            self.END_TOKEN, self.MASK_TOKEN}
            tokens = [token for token in tokens if token not in special_tokens]
        
        return ''.join(tokens)


# Example usage and testing
if __name__ == "__main__":
    print("Testing ProteinEncoder...")
    
    # Create dummy data
    batch_size = 8
    seq_len = 100
    vocab_size = 26
    
    # Random protein sequences
    x = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid padding token (0)
    lengths = torch.randint(50, seq_len, (batch_size,))
    
    # Test different encoder types
    encoder_types = ["cnn", "transformer", "hybrid"]
    
    for encoder_type in encoder_types:
        print(f"\nTesting {encoder_type} encoder...")
        
        encoder = ProteinEncoder(
            vocab_size=vocab_size,
            embedding_dim=128,
            encoder_type=encoder_type
        )
        
        print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        
        with torch.no_grad():
            output = encoder(x, lengths)
        
        print(f"Protein representation shape: {output['protein_representation'].shape}")
        print(f"Output dimension: {encoder.get_output_dim()}")
    
    # Test amino acid vocabulary
    print("\nTesting AminoAcidVocabulary...")
    vocab = AminoAcidVocabulary()
    
    test_sequence = "MKFLVLLFNILCLFPVLAADNH"
    encoded = vocab.encode(test_sequence, add_special_tokens=True)
    decoded = vocab.decode(encoded, remove_special_tokens=True)
    
    print(f"Original: {test_sequence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    print("ProteinEncoder test completed successfully!")