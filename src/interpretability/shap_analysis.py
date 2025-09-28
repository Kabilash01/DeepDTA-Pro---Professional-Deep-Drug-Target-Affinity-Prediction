"""
SHAP Analysis for DeepDTA-Pro
SHAP (SHapley Additive exPlanations) analysis for interpreting model predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from pathlib import Path
import warnings
from tqdm import tqdm
import joblib
from functools import partial

# Try to import SHAP, but provide fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import pickle

logger = logging.getLogger(__name__)

class DeepSHAPAnalyzer:
    """
    SHAP analysis for DeepDTA-Pro model using DeepSHAP.
    
    Provides model interpretability through Shapley values for:
    - Molecular features (atoms, bonds, substructures)
    - Protein features (amino acids, domains)
    - Combined drug-protein interactions
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cpu',
                 background_size: int = 50):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained DeepDTA-Pro model
            device: Device for computation
            background_size: Size of background dataset for SHAP
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP package is required. Install with: pip install shap")
        
        self.model = model
        self.device = device
        self.background_size = background_size
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize SHAP explainer (will be set up with background data)
        self.explainer = None
        self.background_data = None
        
        logger.info("DeepSHAPAnalyzer initialized")
    
    def setup_explainer(self, 
                       background_data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
        """
        Set up SHAP explainer with background data.
        
        Args:
            background_data: List of (drug_data, protein_data) tuples for background
        """
        try:
            # Sample background data if needed
            if len(background_data) > self.background_size:
                indices = np.random.choice(len(background_data), self.background_size, replace=False)
                background_data = [background_data[i] for i in indices]
            
            self.background_data = background_data
            
            # Create wrapper function for the model
            def model_wrapper(data_batch):
                predictions = []
                for drug_data, protein_data in data_batch:
                    with torch.no_grad():
                        pred = self.model(drug_data, protein_data)
                        predictions.append(pred.cpu().numpy())
                return np.array(predictions)
            
            # Initialize DeepSHAP explainer
            self.explainer = shap.DeepExplainer(model_wrapper, background_data)
            
            logger.info(f"SHAP explainer set up with {len(background_data)} background samples")
            
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {str(e)}")
            raise
    
    def explain_prediction(self,
                          drug_data: Dict[str, torch.Tensor],
                          protein_data: Dict[str, torch.Tensor],
                          drug_smiles: str = None,
                          protein_sequence: str = None) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a single prediction.
        
        Args:
            drug_data: Drug molecular graph data
            protein_data: Protein sequence data
            drug_smiles: SMILES string for visualization
            protein_sequence: Protein sequence for visualization
            
        Returns:
            Dictionary containing SHAP explanations
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not set up. Call setup_explainer first.")
        
        try:
            # Get SHAP values
            test_data = [(drug_data, protein_data)]
            shap_values = self.explainer.shap_values(test_data)
            
            # Extract drug and protein SHAP values
            # Note: This assumes the model returns separate contributions
            # You may need to adapt this based on your model architecture
            
            explanation = {
                'prediction': float(self.model(drug_data, protein_data).item()),
                'drug_smiles': drug_smiles,
                'protein_sequence': protein_sequence,
                'shap_values': shap_values,
                'expected_value': float(self.explainer.expected_value)
            }
            
            # Process SHAP values for different components
            if isinstance(shap_values, (list, tuple)) and len(shap_values) >= 2:
                explanation['drug_shap_values'] = shap_values[0]
                explanation['protein_shap_values'] = shap_values[1]
            else:
                # If combined SHAP values, try to separate them
                explanation['combined_shap_values'] = shap_values
            
            logger.debug(f"Generated SHAP explanation for prediction: {explanation['prediction']:.4f}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise
    
    def batch_explain(self,
                     data_pairs: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
                     drug_smiles_list: List[str] = None,
                     protein_sequences: List[str] = None,
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanations for multiple predictions.
        
        Args:
            data_pairs: List of (drug_data, protein_data) tuples
            drug_smiles_list: List of SMILES strings
            protein_sequences: List of protein sequences
            batch_size: Batch size for processing
            
        Returns:
            List of SHAP explanations
        """
        explanations = []
        
        for i in tqdm(range(0, len(data_pairs), batch_size), desc="Generating SHAP explanations"):
            batch_pairs = data_pairs[i:i+batch_size]
            batch_smiles = drug_smiles_list[i:i+batch_size] if drug_smiles_list else [None] * len(batch_pairs)
            batch_sequences = protein_sequences[i:i+batch_size] if protein_sequences else [None] * len(batch_pairs)
            
            for j, ((drug_data, protein_data), smiles, sequence) in enumerate(zip(batch_pairs, batch_smiles, batch_sequences)):
                try:
                    explanation = self.explain_prediction(drug_data, protein_data, smiles, sequence)
                    explanations.append(explanation)
                except Exception as e:
                    logger.warning(f"Failed to explain sample {i+j}: {str(e)}")
                    explanations.append({'error': str(e)})
        
        return explanations
    
    def visualize_molecular_shap(self,
                               drug_smiles: str,
                               shap_values: np.ndarray,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
        """
        Visualize SHAP values for molecular features.
        
        Args:
            drug_smiles: SMILES string
            shap_values: SHAP values for molecular features
            save_path: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if saved
        """
        try:
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol is None:
                logger.error(f"Could not parse SMILES: {drug_smiles}")
                return None
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # 1. SHAP values bar plot
            if len(shap_values.shape) > 1:
                shap_values = shap_values.flatten()
            
            # Limit to significant SHAP values
            top_indices = np.argsort(np.abs(shap_values))[-20:]
            top_shap_values = shap_values[top_indices]
            
            colors = ['red' if x < 0 else 'blue' for x in top_shap_values]
            bars = ax1.barh(range(len(top_shap_values)), top_shap_values, color=colors, alpha=0.7)
            ax1.set_xlabel('SHAP Value')
            ax1.set_ylabel('Feature Index')
            ax1.set_title('Top Molecular Feature SHAP Values')
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_shap_values)):
                ax1.text(value + 0.001 * np.sign(value), bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center', fontsize=8)
            
            # 2. SHAP value distribution
            ax2.hist(shap_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')
            ax2.axvline(x=np.mean(shap_values), color='orange', linestyle='-', alpha=0.7, label='Mean')
            ax2.set_xlabel('SHAP Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('SHAP Value Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Molecular SHAP Analysis: {drug_smiles}', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Molecular SHAP visualization saved to {save_path}")
                return None
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing molecular SHAP: {str(e)}")
            return None
    
    def visualize_protein_shap(self,
                             protein_sequence: str,
                             shap_values: np.ndarray,
                             save_path: Optional[str] = None,
                             max_length: int = 100,
                             figsize: Tuple[int, int] = (15, 8)) -> Optional[plt.Figure]:
        """
        Visualize SHAP values for protein features.
        
        Args:
            protein_sequence: Protein amino acid sequence
            shap_values: SHAP values for protein features
            save_path: Path to save visualization
            max_length: Maximum sequence length to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if saved
        """
        try:
            if len(shap_values.shape) > 1:
                shap_values = shap_values.flatten()
            
            # Truncate sequence if too long
            if len(protein_sequence) > max_length:
                protein_sequence = protein_sequence[:max_length]
                shap_values = shap_values[:max_length]
            
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1])
            
            # 1. Sequence with SHAP-based coloring
            self._plot_sequence_shap(protein_sequence, shap_values, ax1)
            
            # 2. SHAP values line plot
            positions = range(len(protein_sequence))
            ax2.plot(positions, shap_values, 'b-', linewidth=2, marker='o', markersize=3)
            ax2.fill_between(positions, shap_values, alpha=0.3, 
                           where=(shap_values >= 0), color='blue', interpolate=True, label='Positive')
            ax2.fill_between(positions, shap_values, alpha=0.3, 
                           where=(shap_values < 0), color='red', interpolate=True, label='Negative')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Residue Position')
            ax2.set_ylabel('SHAP Value')
            ax2.set_title('SHAP Values Along Protein Sequence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Top contributing residues
            top_indices = np.argsort(np.abs(shap_values))[-10:]
            top_residues = [protein_sequence[i] for i in top_indices]
            top_shap_values = shap_values[top_indices]
            
            colors = ['red' if x < 0 else 'blue' for x in top_shap_values]
            bars = ax3.barh(range(len(top_shap_values)), top_shap_values, color=colors, alpha=0.7)
            ax3.set_xlabel('SHAP Value')
            ax3.set_ylabel('Residue')
            ax3.set_title('Top Contributing Residues')
            ax3.set_yticks(range(len(top_residues)))
            ax3.set_yticklabels([f'{res}{pos}' for res, pos in zip(top_residues, top_indices)])
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Protein SHAP visualization saved to {save_path}")
                return None
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing protein SHAP: {str(e)}")
            return None
    
    def _plot_sequence_shap(self, sequence: str, shap_values: np.ndarray, ax: plt.Axes):
        """Plot protein sequence with SHAP-based coloring."""
        # Normalize SHAP values for coloring
        abs_max = max(abs(shap_values.min()), abs(shap_values.max()))
        normalized_shap = shap_values / abs_max if abs_max > 0 else shap_values
        
        # Create color map (blue for positive, red for negative)
        colors = []
        for value in normalized_shap:
            if value >= 0:
                colors.append((1 - value, 1 - value, 1))  # Blue gradient
            else:
                colors.append((1, 1 + value, 1 + value))  # Red gradient
        
        # Plot each residue
        for i, (residue, color, shap_val) in enumerate(zip(sequence, colors, shap_values)):
            ax.text(i, 0, residue, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.5, len(sequence) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Residue Position')
        ax.set_title('Protein Sequence (SHAP-colored: Blue=Positive, Red=Negative)')
        ax.set_yticks([])
    
    def create_shap_summary(self,
                          explanations: List[Dict[str, Any]],
                          output_dir: str) -> Dict[str, Any]:
        """
        Create comprehensive SHAP analysis summary.
        
        Args:
            explanations: List of SHAP explanations
            output_dir: Directory to save results
            
        Returns:
            Summary dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Filter valid explanations
            valid_explanations = [exp for exp in explanations if 'error' not in exp]
            
            if not valid_explanations:
                logger.warning("No valid explanations found")
                return {'error': 'No valid explanations'}
            
            # Aggregate SHAP values
            all_drug_shap = []
            all_protein_shap = []
            predictions = []
            
            for exp in valid_explanations:
                predictions.append(exp['prediction'])
                
                if 'drug_shap_values' in exp:
                    all_drug_shap.append(exp['drug_shap_values'])
                if 'protein_shap_values' in exp:
                    all_protein_shap.append(exp['protein_shap_values'])
            
            summary = {
                'num_explanations': len(valid_explanations),
                'prediction_stats': {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions))
                }
            }
            
            # Analyze drug SHAP values
            if all_drug_shap:
                drug_shap_array = np.array(all_drug_shap)
                summary['drug_shap_analysis'] = {
                    'mean_importance': float(np.mean(np.abs(drug_shap_array))),
                    'std_importance': float(np.std(np.abs(drug_shap_array))),
                    'top_features': np.argsort(np.mean(np.abs(drug_shap_array), axis=0))[-10:].tolist()
                }
            
            # Analyze protein SHAP values
            if all_protein_shap:
                protein_shap_array = np.array(all_protein_shap)
                summary['protein_shap_analysis'] = {
                    'mean_importance': float(np.mean(np.abs(protein_shap_array))),
                    'std_importance': float(np.std(np.abs(protein_shap_array))),
                    'top_features': np.argsort(np.mean(np.abs(protein_shap_array), axis=0))[-10:].tolist()
                }
            
            # Create summary plots
            self._create_summary_plots(valid_explanations, output_dir)
            
            # Save summary
            import json
            with open(output_dir / 'shap_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save explanations
            with open(output_dir / 'explanations.pkl', 'wb') as f:
                pickle.dump(valid_explanations, f)
            
            logger.info(f"SHAP analysis summary saved to {output_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary: {str(e)}")
            return {'error': str(e)}
    
    def _create_summary_plots(self, explanations: List[Dict[str, Any]], output_dir: Path):
        """Create summary plots for SHAP analysis."""
        try:
            # 1. Prediction distribution
            predictions = [exp['prediction'] for exp in explanations]
            
            plt.figure(figsize=(10, 6))
            plt.hist(predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Predicted Binding Affinity')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predictions')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. SHAP importance correlation with predictions
            if len(explanations) > 1 and 'drug_shap_values' in explanations[0]:
                drug_importances = [np.mean(np.abs(exp['drug_shap_values'])) for exp in explanations]
                
                plt.figure(figsize=(8, 6))
                plt.scatter(predictions, drug_importances, alpha=0.6)
                plt.xlabel('Predicted Binding Affinity')
                plt.ylabel('Mean Drug SHAP Importance')
                plt.title('Prediction vs Drug Feature Importance')
                
                # Add correlation coefficient
                corr = np.corrcoef(predictions, drug_importances)[0, 1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
                
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'importance_correlation.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating summary plots: {str(e)}")

class FeatureAttributionAnalyzer:
    """
    Feature attribution analysis using various methods.
    Provides integrated gradients, guided backpropagation, and other attribution methods.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize feature attribution analyzer.
        
        Args:
            model: Trained model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("FeatureAttributionAnalyzer initialized")
    
    def integrated_gradients(self,
                           drug_data: Dict[str, torch.Tensor],
                           protein_data: Dict[str, torch.Tensor],
                           baseline_data: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None,
                           steps: int = 50) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients for feature attribution.
        
        Args:
            drug_data: Input drug data
            protein_data: Input protein data
            baseline_data: Baseline data (zeros if None)
            steps: Number of integration steps
            
        Returns:
            Dictionary of integrated gradients
        """
        # Create baseline if not provided
        if baseline_data is None:
            baseline_drug = {k: torch.zeros_like(v) for k, v in drug_data.items()}
            baseline_protein = {k: torch.zeros_like(v) for k, v in protein_data.items()}
        else:
            baseline_drug, baseline_protein = baseline_data
        
        # Compute integrated gradients
        integrated_grads = {'drug': {}, 'protein': {}}
        
        for step in tqdm(range(steps), desc="Computing integrated gradients"):
            alpha = step / (steps - 1) if steps > 1 else 1.0
            
            # Interpolate between baseline and input
            interp_drug = {}
            interp_protein = {}
            
            for key in drug_data:
                interp_drug[key] = baseline_drug[key] + alpha * (drug_data[key] - baseline_drug[key])
                interp_drug[key].requires_grad_(True)
            
            for key in protein_data:
                interp_protein[key] = baseline_protein[key] + alpha * (protein_data[key] - baseline_protein[key])
                interp_protein[key].requires_grad_(True)
            
            # Forward pass
            output = self.model(interp_drug, interp_protein)
            
            # Backward pass
            self.model.zero_grad()
            output.backward()
            
            # Accumulate gradients
            for key in drug_data:
                if key not in integrated_grads['drug']:
                    integrated_grads['drug'][key] = torch.zeros_like(drug_data[key])
                integrated_grads['drug'][key] += interp_drug[key].grad
            
            for key in protein_data:
                if key not in integrated_grads['protein']:
                    integrated_grads['protein'][key] = torch.zeros_like(protein_data[key])
                integrated_grads['protein'][key] += interp_protein[key].grad
        
        # Average and multiply by input difference
        for key in integrated_grads['drug']:
            integrated_grads['drug'][key] = (integrated_grads['drug'][key] / steps) * (drug_data[key] - baseline_drug[key])
        
        for key in integrated_grads['protein']:
            integrated_grads['protein'][key] = (integrated_grads['protein'][key] / steps) * (protein_data[key] - baseline_protein[key])
        
        return integrated_grads

# Example usage and utility functions
def molecular_descriptors_baseline(smiles_list: List[str]) -> np.ndarray:
    """
    Compute molecular descriptors as baseline features for SHAP analysis.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Array of molecular descriptors
    """
    descriptors = []
    
    descriptor_functions = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.TPSA,
        Descriptors.NumRotatableBonds,
        Descriptors.NumAromaticRings,
        Descriptors.NumSaturatedRings,
        Descriptors.FractionCsp3,
        Descriptors.NumAliphaticCarbocycles
    ]
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Use zeros for invalid molecules
            mol_descriptors = [0.0] * len(descriptor_functions)
        else:
            mol_descriptors = [func(mol) for func in descriptor_functions]
        
        descriptors.append(mol_descriptors)
    
    return np.array(descriptors)

def protein_features_baseline(sequences: List[str]) -> np.ndarray:
    """
    Compute protein features as baseline for SHAP analysis.
    
    Args:
        sequences: List of protein sequences
        
    Returns:
        Array of protein features
    """
    features = []
    
    for seq in sequences:
        # Basic features
        length = len(seq)
        
        # Amino acid composition
        aa_counts = {aa: seq.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        aa_freqs = [aa_counts[aa] / length if length > 0 else 0 for aa in 'ACDEFGHIKLMNPQRSTVWY']
        
        # Physicochemical properties
        hydrophobic = sum(1 for aa in seq if aa in 'AILMFPWYV') / length if length > 0 else 0
        charged = sum(1 for aa in seq if aa in 'DEKR') / length if length > 0 else 0
        polar = sum(1 for aa in seq if aa in 'STNQC') / length if length > 0 else 0
        
        seq_features = [length] + aa_freqs + [hydrophobic, charged, polar]
        features.append(seq_features)
    
    return np.array(features)

# Example usage
if __name__ == "__main__":
    print("Testing SHAP analysis modules...")
    
    if SHAP_AVAILABLE:
        print("SHAP package available - full functionality enabled")
        
        # Test molecular descriptors
        test_smiles = ['CCO', 'CCN', 'CCC']
        mol_descriptors = molecular_descriptors_baseline(test_smiles)
        print(f"Molecular descriptors shape: {mol_descriptors.shape}")
        
        # Test protein features
        test_sequences = ['ACDEFG', 'GGHHII', 'KKLMM']
        prot_features = protein_features_baseline(test_sequences)
        print(f"Protein features shape: {prot_features.shape}")
        
        print("SHAP analysis test completed successfully!")
    else:
        print("SHAP package not available - limited functionality")