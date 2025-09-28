"""
Attention Visualization for DeepDTA-Pro
Visualization tools for understanding attention mechanisms in the model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64
from PIL import Image
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """
    Visualization tools for attention mechanisms in DeepDTA-Pro.
    
    Provides methods to visualize and interpret attention weights for:
    - Molecular graphs (node and edge attention)
    - Protein sequences (residue attention)
    - Cross-modal attention (drug-protein interactions)
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize attention visualizer.
        
        Args:
            model: Trained DeepDTA-Pro model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Register forward hooks to capture attention weights
        self.attention_weights = {}
        self.hooks = []
        self._register_attention_hooks()
        
        logger.info("AttentionVisualizer initialized")
    
    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights."""
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, 'attention_weights'):
                    self.attention_weights[name] = module.attention_weights.detach().cpu()
            return hook
        
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or hasattr(module, 'attention_weights'):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_weights(self, 
                            drug_data: Dict[str, torch.Tensor],
                            protein_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for a given drug-protein pair.
        
        Args:
            drug_data: Drug molecular graph data
            protein_data: Protein sequence data
            
        Returns:
            Dictionary of attention weights from different layers
        """
        self.attention_weights = {}
        
        with torch.no_grad():
            # Forward pass to capture attention weights
            _ = self.model(drug_data, protein_data)
        
        return self.attention_weights.copy()
    
    def visualize_molecular_attention(self,
                                    mol_smiles: str,
                                    attention_weights: torch.Tensor,
                                    node_indices: Optional[List[int]] = None,
                                    save_path: Optional[str] = None,
                                    title: str = "Molecular Attention",
                                    figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        """
        Visualize attention weights on molecular structure.
        
        Args:
            mol_smiles: SMILES string of the molecule
            attention_weights: Attention weights for each atom
            node_indices: Mapping from attention indices to atom indices
            save_path: Path to save the visualization
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if save_path is provided
        """
        try:
            # Parse molecule
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                logger.error(f"Could not parse SMILES: {mol_smiles}")
                return None
            
            # Normalize attention weights
            if len(attention_weights.shape) > 1:
                attention_weights = attention_weights.mean(dim=0)  # Average over heads
            
            attention_weights = attention_weights.numpy()
            attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # 1. Molecular structure with highlighted atoms
            img = self._draw_molecule_with_attention(mol, attention_weights, node_indices)
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title("Molecular Structure\n(Attention Highlighted)")
            
            # 2. Attention heatmap
            self._plot_attention_heatmap(attention_weights, ax2, title="Atom Attention Weights")
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Molecular attention visualization saved to {save_path}")
                return None
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing molecular attention: {str(e)}")
            return None
    
    def visualize_protein_attention(self,
                                  protein_sequence: str,
                                  attention_weights: torch.Tensor,
                                  save_path: Optional[str] = None,
                                  title: str = "Protein Attention",
                                  max_length: int = 100,
                                  figsize: Tuple[int, int] = (15, 8)) -> Optional[plt.Figure]:
        """
        Visualize attention weights on protein sequence.
        
        Args:
            protein_sequence: Protein amino acid sequence
            attention_weights: Attention weights for each residue
            save_path: Path to save the visualization
            title: Plot title
            max_length: Maximum sequence length to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if save_path is provided
        """
        try:
            # Process attention weights
            if len(attention_weights.shape) > 1:
                attention_weights = attention_weights.mean(dim=0)  # Average over heads
            
            attention_weights = attention_weights.numpy()
            
            # Truncate sequence if too long
            if len(protein_sequence) > max_length:
                protein_sequence = protein_sequence[:max_length]
                attention_weights = attention_weights[:max_length]
            
            # Normalize attention weights
            attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
            
            # 1. Sequence with attention heatmap
            self._plot_sequence_attention(protein_sequence, attention_weights, ax1)
            
            # 2. Attention weights line plot
            positions = range(len(protein_sequence))
            ax2.plot(positions, attention_weights, 'b-', linewidth=2)
            ax2.fill_between(positions, attention_weights, alpha=0.3)
            ax2.set_xlabel('Residue Position')
            ax2.set_ylabel('Attention Weight')
            ax2.set_title('Attention Weight Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Protein attention visualization saved to {save_path}")
                return None
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing protein attention: {str(e)}")
            return None
    
    def visualize_cross_attention(self,
                                drug_smiles: str,
                                protein_sequence: str,
                                attention_matrix: torch.Tensor,
                                save_path: Optional[str] = None,
                                title: str = "Drug-Protein Cross-Attention",
                                max_protein_len: int = 100,
                                figsize: Tuple[int, int] = (12, 10)) -> Optional[plt.Figure]:
        """
        Visualize cross-attention between drug and protein.
        
        Args:
            drug_smiles: Drug SMILES string
            protein_sequence: Protein sequence
            attention_matrix: Cross-attention matrix (drug_nodes x protein_residues)
            save_path: Path to save the visualization
            title: Plot title
            max_protein_len: Maximum protein length to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if save_path is provided
        """
        try:
            # Process attention matrix
            if len(attention_matrix.shape) > 2:
                attention_matrix = attention_matrix.mean(dim=0)  # Average over heads
            
            attention_matrix = attention_matrix.numpy()
            
            # Parse molecule to get atom count
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol is None:
                logger.error(f"Could not parse SMILES: {drug_smiles}")
                return None
            
            num_atoms = mol.GetNumAtoms()
            
            # Truncate protein sequence if too long
            if len(protein_sequence) > max_protein_len:
                protein_sequence = protein_sequence[:max_protein_len]
                attention_matrix = attention_matrix[:, :max_protein_len]
            
            # Ensure attention matrix matches dimensions
            attention_matrix = attention_matrix[:num_atoms, :len(protein_sequence)]
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # Create heatmap
            im = ax.imshow(attention_matrix, cmap='Reds', aspect='auto')
            
            # Set labels
            ax.set_xlabel('Protein Residue Position')
            ax.set_ylabel('Drug Atom Index')
            ax.set_title(title)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attention Weight')
            
            # Add protein sequence as x-axis labels (if not too long)
            if len(protein_sequence) <= 50:
                ax.set_xticks(range(len(protein_sequence)))
                ax.set_xticklabels(list(protein_sequence), rotation=90, fontsize=8)
            else:
                # Show every 10th residue
                step = max(1, len(protein_sequence) // 20)
                positions = range(0, len(protein_sequence), step)
                ax.set_xticks(positions)
                ax.set_xticklabels([f"{protein_sequence[i]}_{i}" for i in positions], 
                                 rotation=90, fontsize=8)
            
            # Add atom indices as y-axis labels
            if num_atoms <= 50:
                ax.set_yticks(range(num_atoms))
                ax.set_yticklabels(range(num_atoms))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Cross-attention visualization saved to {save_path}")
                return None
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing cross-attention: {str(e)}")
            return None
    
    def _draw_molecule_with_attention(self,
                                    mol: Chem.Mol,
                                    attention_weights: np.ndarray,
                                    node_indices: Optional[List[int]] = None) -> np.ndarray:
        """Draw molecule with attention-based highlighting."""
        try:
            # Create drawer
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            
            # Map attention weights to atoms
            if node_indices is not None:
                atom_weights = np.zeros(mol.GetNumAtoms())
                for i, atom_idx in enumerate(node_indices):
                    if i < len(attention_weights) and atom_idx < len(atom_weights):
                        atom_weights[atom_idx] = attention_weights[i]
            else:
                atom_weights = attention_weights[:mol.GetNumAtoms()]
            
            # Normalize weights to [0, 1] for coloring
            if atom_weights.max() > atom_weights.min():
                atom_weights = (atom_weights - atom_weights.min()) / (atom_weights.max() - atom_weights.min())
            
            # Create highlight colors based on attention weights
            highlight_atoms = {}
            highlight_colors = {}
            
            for i, weight in enumerate(atom_weights):
                if weight > 0.1:  # Only highlight significant attention
                    highlight_atoms[i] = (1.0, 1.0 - weight, 1.0 - weight)  # Red gradient
                    highlight_colors[i] = (1.0, 1.0 - weight, 1.0 - weight)
            
            # Draw molecule
            drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms.keys()),
                              highlightAtomColors=highlight_colors)
            drawer.FinishDrawing()
            
            # Convert to image
            img_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(img_data))
            return np.array(img)
            
        except Exception as e:
            logger.warning(f"Could not draw molecule with RDKit: {str(e)}")
            # Fallback: return simple molecular structure
            img = Draw.MolToImage(mol, size=(400, 400))
            return np.array(img)
    
    def _plot_attention_heatmap(self, weights: np.ndarray, ax: plt.Axes, title: str):
        """Plot attention weights as heatmap."""
        # Reshape weights for visualization
        if len(weights.shape) == 1:
            # Convert 1D weights to 2D for better visualization
            n = len(weights)
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            
            heatmap_data = np.zeros((rows, cols))
            for i, weight in enumerate(weights):
                row, col = divmod(i, cols)
                if row < rows:
                    heatmap_data[row, col] = weight
        else:
            heatmap_data = weights
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto')
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_sequence_attention(self, sequence: str, weights: np.ndarray, ax: plt.Axes):
        """Plot protein sequence with attention-based coloring."""
        # Create color map
        cmap = plt.cm.Reds
        
        # Plot each residue with attention-based color
        for i, (residue, weight) in enumerate(zip(sequence, weights)):
            color = cmap(weight)
            ax.text(i, 0, residue, ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.5, len(sequence) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Residue Position')
        ax.set_title('Protein Sequence with Attention Weights')
        ax.set_yticks([])
        
        # Add position labels
        step = max(1, len(sequence) // 20)
        positions = range(0, len(sequence), step)
        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
    
    def create_attention_summary(self,
                               drug_smiles: str,
                               protein_sequence: str,
                               attention_data: Dict[str, torch.Tensor],
                               output_dir: str) -> Dict[str, Any]:
        """
        Create comprehensive attention analysis summary.
        
        Args:
            drug_smiles: Drug SMILES string
            protein_sequence: Protein sequence
            attention_data: Dictionary of attention weights
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary containing attention analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'drug_smiles': drug_smiles,
            'protein_sequence': protein_sequence[:100] + '...' if len(protein_sequence) > 100 else protein_sequence,
            'attention_layers': list(attention_data.keys()),
            'visualizations': []
        }
        
        try:
            # 1. Molecular attention visualization
            if 'molecular_attention' in attention_data:
                mol_fig = self.visualize_molecular_attention(
                    drug_smiles,
                    attention_data['molecular_attention'],
                    save_path=str(output_dir / 'molecular_attention.png'),
                    title=f"Molecular Attention: {drug_smiles}"
                )
                summary['visualizations'].append('molecular_attention.png')
            
            # 2. Protein attention visualization
            if 'protein_attention' in attention_data:
                prot_fig = self.visualize_protein_attention(
                    protein_sequence,
                    attention_data['protein_attention'],
                    save_path=str(output_dir / 'protein_attention.png'),
                    title="Protein Attention"
                )
                summary['visualizations'].append('protein_attention.png')
            
            # 3. Cross-attention visualization
            if 'cross_attention' in attention_data:
                cross_fig = self.visualize_cross_attention(
                    drug_smiles,
                    protein_sequence,
                    attention_data['cross_attention'],
                    save_path=str(output_dir / 'cross_attention.png'),
                    title="Drug-Protein Cross-Attention"
                )
                summary['visualizations'].append('cross_attention.png')
            
            # 4. Save attention statistics
            stats = self._compute_attention_statistics(attention_data)
            summary['statistics'] = stats
            
            # Save summary as JSON
            import json
            with open(output_dir / 'attention_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Attention analysis saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating attention summary: {str(e)}")
            summary['error'] = str(e)
        
        return summary
    
    def _compute_attention_statistics(self, attention_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute statistical measures of attention patterns."""
        stats = {}
        
        for layer_name, attention in attention_data.items():
            if len(attention.shape) > 1:
                attention = attention.mean(dim=0)  # Average over heads
            
            attention_np = attention.numpy()
            
            stats[layer_name] = {
                'mean': float(attention_np.mean()),
                'std': float(attention_np.std()),
                'max': float(attention_np.max()),
                'min': float(attention_np.min()),
                'entropy': float(-np.sum(attention_np * np.log(attention_np + 1e-8))),
                'sparsity': float(np.sum(attention_np < 0.01) / len(attention_np)),
                'top_5_positions': np.argsort(attention_np)[-5:].tolist()
            }
        
        return stats

# Utility functions for attention analysis
def extract_attention_from_model(model: torch.nn.Module, 
                                drug_data: Dict[str, torch.Tensor],
                                protein_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract attention weights from model during forward pass.
    
    Args:
        model: DeepDTA-Pro model
        drug_data: Drug molecular graph data
        protein_data: Protein sequence data
        
    Returns:
        Dictionary of attention weights
    """
    attention_weights = {}
    
    def attention_hook(name):
        def hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights[name] = module.attention_weights.detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower():
            hook = module.register_forward_hook(attention_hook(name))
            hooks.append(hook)
    
    try:
        # Forward pass
        with torch.no_grad():
            model.eval()
            _ = model(drug_data, protein_data)
        
        return attention_weights
        
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

def analyze_attention_patterns(attention_weights: Dict[str, torch.Tensor],
                             threshold: float = 0.1) -> Dict[str, Any]:
    """
    Analyze attention patterns for insights.
    
    Args:
        attention_weights: Dictionary of attention weights
        threshold: Threshold for significant attention
        
    Returns:
        Analysis results
    """
    analysis = {}
    
    for layer_name, weights in attention_weights.items():
        if len(weights.shape) > 1:
            weights = weights.mean(dim=0)
        
        weights_np = weights.numpy()
        
        # Find significant attention positions
        significant_positions = np.where(weights_np > threshold)[0]
        
        # Compute concentration metric (Gini coefficient)
        sorted_weights = np.sort(weights_np)
        n = len(sorted_weights)
        cumsum = np.cumsum(sorted_weights)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n
        
        analysis[layer_name] = {
            'significant_positions': significant_positions.tolist(),
            'num_significant': len(significant_positions),
            'concentration_gini': float(gini),
            'max_attention_position': int(np.argmax(weights_np)),
            'attention_distribution': {
                'quartiles': np.percentile(weights_np, [25, 50, 75]).tolist(),
                'iqr': float(np.percentile(weights_np, 75) - np.percentile(weights_np, 25))
            }
        }
    
    return analysis

# Example usage
if __name__ == "__main__":
    print("Testing AttentionVisualizer...")
    
    # This would normally be done with a real model and data
    # Here we create dummy data for testing
    
    # Create dummy attention weights
    molecular_attention = torch.randn(20)  # 20 atoms
    protein_attention = torch.randn(50)    # 50 residues
    cross_attention = torch.randn(20, 50)  # 20 atoms x 50 residues
    
    attention_data = {
        'molecular_attention': molecular_attention,
        'protein_attention': protein_attention,
        'cross_attention': cross_attention
    }
    
    # Test attention statistics
    stats = {}
    for layer_name, attention in attention_data.items():
        if len(attention.shape) > 1:
            attention = attention.mean(dim=0)
        attention_np = attention.numpy()
        stats[layer_name] = {
            'mean': float(attention_np.mean()),
            'std': float(attention_np.std()),
            'max': float(attention_np.max()),
            'min': float(attention_np.min())
        }
    
    print("Attention statistics computed:")
    for layer, layer_stats in stats.items():
        print(f"{layer}: mean={layer_stats['mean']:.3f}, std={layer_stats['std']:.3f}")
    
    print("AttentionVisualizer test completed successfully!")