"""
Molecular Interpretation for DeepDTA-Pro
Analysis of molecular substructures and their importance for binding predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from collections import defaultdict, Counter
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors, AllChem
from rdkit.Chem import Draw, rdDepictor, rdRGroupDecomposition
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.DataStructs import TanimotoSimilarity
import io
from PIL import Image

logger = logging.getLogger(__name__)

class MolecularInterpreter:
    """
    Molecular interpretation framework for understanding structure-activity relationships.
    
    Provides analysis of:
    - Functional group importance
    - Substructure contributions
    - Pharmacophore identification
    - Fragment-based analysis
    - Chemical space exploration
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize molecular interpreter.
        
        Args:
            model: Trained DeepDTA-Pro model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Predefined functional groups and pharmacophores
        self.functional_groups = self._define_functional_groups()
        self.pharmacophores = self._define_pharmacophores()
        
        logger.info("MolecularInterpreter initialized")
    
    def _define_functional_groups(self) -> Dict[str, str]:
        """Define common functional groups as SMARTS patterns."""
        return {
            'alcohol': '[OH]',
            'aldehyde': '[CX3H1](=O)[#6]',
            'ketone': '[#6][CX3](=O)[#6]',
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'ester': '[#6][CX3](=O)[OX2H0][#6]',
            'amide': '[CX3](=[OX1])[NX3H2]',
            'amine_primary': '[NX3;H2;!$(NC=O)]',
            'amine_secondary': '[NX3;H1;!$(NC=O)]',
            'amine_tertiary': '[NX3;H0;!$(NC=O);!$([N-]);!$(N=*)]',
            'aromatic_ring': 'c1ccccc1',
            'benzene': 'c1ccccc1',
            'pyridine': 'n1ccccc1',
            'pyrimidine': 'n1cccncc1',
            'imidazole': 'c1cnc[nH]1',
            'thiazole': 'c1cscn1',
            'furan': 'o1cccc1',
            'thiophene': 's1cccc1',
            'halogen': '[F,Cl,Br,I]',
            'nitro': '[N+](=O)[O-]',
            'sulfonamide': '[#16X4]([NX3])(=[OX1])(=[OX1])[#6]',
            'hydroxyl': '[OH]',
            'thiol': '[SH]',
            'phosphate': '[PX4](=[OX1])([OX2H,OX1-])([OX2H,OX1-])[OX2H,OX1-]'
        }
    
    def _define_pharmacophores(self) -> Dict[str, str]:
        """Define pharmacophore patterns."""
        return {
            'hydrogen_bond_donor': '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
            'hydrogen_bond_acceptor': '[!$([#6,H0,-,-2,-3])]',
            'positive_ionizable': '[#7;+,$([#7;!n][CX4;!$(C(=[#7,#8,#15,#16])),!$(C#[#7,#8,#15,#16])])]',
            'negative_ionizable': '[O,S;-1]',
            'aromatic': '[a]',
            'hydrophobic': '[#6+0;!$(*~[#7,#8,F]);!$([#6]=[#8])]',
            'pi_stacking': '[a;r6]'
        }
    
    def analyze_functional_groups(self, 
                                smiles_list: List[str],
                                predictions: List[float],
                                activity_threshold: float = 7.0) -> Dict[str, Any]:
        """
        Analyze the importance of functional groups for activity.
        
        Args:
            smiles_list: List of SMILES strings
            predictions: Corresponding binding affinity predictions
            activity_threshold: Threshold for defining active compounds
            
        Returns:
            Dictionary containing functional group analysis
        """
        results = {
            'functional_group_counts': defaultdict(list),
            'active_compounds': [],
            'inactive_compounds': [],
            'group_statistics': {}
        }
        
        # Classify compounds
        for smiles, pred in zip(smiles_list, predictions):
            if pred >= activity_threshold:
                results['active_compounds'].append((smiles, pred))
            else:
                results['inactive_compounds'].append((smiles, pred))
        
        # Count functional groups in each compound
        for smiles, pred in zip(smiles_list, predictions):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            is_active = pred >= activity_threshold
            
            for group_name, smarts in self.functional_groups.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    count = len(matches)
                    
                    results['functional_group_counts'][group_name].append({
                        'smiles': smiles,
                        'prediction': pred,
                        'count': count,
                        'is_active': is_active
                    })
        
        # Calculate statistics for each functional group
        for group_name, group_data in results['functional_group_counts'].items():
            active_data = [d for d in group_data if d['is_active']]
            inactive_data = [d for d in group_data if not d['is_active']]
            
            active_counts = [d['count'] for d in active_data]
            inactive_counts = [d['count'] for d in inactive_data]
            
            results['group_statistics'][group_name] = {
                'active_mean_count': np.mean(active_counts) if active_counts else 0,
                'inactive_mean_count': np.mean(inactive_counts) if inactive_counts else 0,
                'active_frequency': len([c for c in active_counts if c > 0]) / len(active_counts) if active_counts else 0,
                'inactive_frequency': len([c for c in inactive_counts if c > 0]) / len(inactive_counts) if inactive_counts else 0,
                'enrichment_ratio': None
            }
            
            # Calculate enrichment ratio
            active_freq = results['group_statistics'][group_name]['active_frequency']
            inactive_freq = results['group_statistics'][group_name]['inactive_frequency']
            
            if inactive_freq > 0:
                results['group_statistics'][group_name]['enrichment_ratio'] = active_freq / inactive_freq
        
        return results
    
    def fragment_based_analysis(self,
                              smiles_list: List[str],
                              predictions: List[float],
                              fragment_method: str = 'brics') -> Dict[str, Any]:
        """
        Perform fragment-based analysis of molecular activity.
        
        Args:
            smiles_list: List of SMILES strings
            predictions: Corresponding predictions
            fragment_method: Fragmentation method ('brics', 'recap', or 'morgan')
            
        Returns:
            Dictionary containing fragment analysis
        """
        from rdkit.Chem import BRICS, Recap
        
        fragment_data = defaultdict(list)
        fragment_activities = defaultdict(list)
        
        for smiles, pred in zip(smiles_list, predictions):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Fragment the molecule
            if fragment_method == 'brics':
                fragments = BRICS.BRICSDecompose(mol)
            elif fragment_method == 'recap':
                recap_tree = Recap.RecapDecompose(mol)
                fragments = recap_tree.GetLeaves().keys() if recap_tree else []
            elif fragment_method == 'morgan':
                # Use Morgan fingerprint bits as "fragments"
                info = {}
                fp = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=info)
                fragments = [str(bit) for bit in info.keys()]
            else:
                logger.warning(f"Unknown fragment method: {fragment_method}")
                continue
            
            # Store fragment data
            for fragment in fragments:
                fragment_data[fragment].append({
                    'smiles': smiles,
                    'prediction': pred,
                    'fragment': fragment
                })
                fragment_activities[fragment].append(pred)
        
        # Analyze fragment contributions
        fragment_analysis = {}
        
        for fragment, activities in fragment_activities.items():
            if len(activities) >= 3:  # Minimum occurrence threshold
                fragment_analysis[fragment] = {
                    'count': len(activities),
                    'mean_activity': np.mean(activities),
                    'std_activity': np.std(activities),
                    'median_activity': np.median(activities),
                    'molecules': [d['smiles'] for d in fragment_data[fragment]]
                }
        
        # Sort fragments by mean activity
        sorted_fragments = sorted(fragment_analysis.items(), 
                                key=lambda x: x[1]['mean_activity'], reverse=True)
        
        return {
            'fragment_method': fragment_method,
            'fragment_analysis': fragment_analysis,
            'top_fragments': sorted_fragments[:20],
            'bottom_fragments': sorted_fragments[-20:] if len(sorted_fragments) > 20 else []
        }
    
    def substructure_importance_analysis(self,
                                       drug_smiles: str,
                                       protein_data: Dict[str, torch.Tensor],
                                       reference_smiles: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the importance of molecular substructures using occlusion analysis.
        
        Args:
            drug_smiles: SMILES string of the drug
            protein_data: Protein data for binding partner
            reference_smiles: Reference molecule for comparison
            
        Returns:
            Dictionary containing substructure importance analysis
        """
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None:
            return {'error': 'Invalid SMILES'}
        
        # Get original prediction
        original_drug_data = self._smiles_to_model_input(drug_smiles)
        if original_drug_data is None:
            return {'error': 'Could not convert SMILES to model input'}
        
        with torch.no_grad():
            original_pred = self.model(original_drug_data, protein_data).item()
        
        # Analyze atom importance through systematic removal/modification
        atom_importance = []
        
        for atom_idx in range(mol.GetNumAtoms()):
            # Try to remove the atom and measure prediction change
            modified_mol = self._remove_atom(mol, atom_idx)
            
            if modified_mol is not None:
                modified_smiles = Chem.MolToSmiles(modified_mol)
                modified_drug_data = self._smiles_to_model_input(modified_smiles)
                
                if modified_drug_data is not None:
                    with torch.no_grad():
                        modified_pred = self.model(modified_drug_data, protein_data).item()
                    
                    importance = original_pred - modified_pred
                    atom_importance.append({
                        'atom_idx': atom_idx,
                        'atom_symbol': mol.GetAtomWithIdx(atom_idx).GetSymbol(),
                        'importance': importance,
                        'original_pred': original_pred,
                        'modified_pred': modified_pred
                    })
        
        # Analyze bond importance
        bond_importance = []
        
        for bond in mol.GetBonds():
            # Try to remove the bond
            modified_mol = self._remove_bond(mol, bond.GetIdx())
            
            if modified_mol is not None:
                # Get largest fragment after bond removal
                fragments = Chem.GetMolFrags(modified_mol, asMols=True)
                if fragments:
                    largest_fragment = max(fragments, key=lambda x: x.GetNumAtoms())
                    fragment_smiles = Chem.MolToSmiles(largest_fragment)
                    fragment_drug_data = self._smiles_to_model_input(fragment_smiles)
                    
                    if fragment_drug_data is not None:
                        with torch.no_grad():
                            fragment_pred = self.model(fragment_drug_data, protein_data).item()
                        
                        importance = original_pred - fragment_pred
                        bond_importance.append({
                            'bond_idx': bond.GetIdx(),
                            'atom1': bond.GetBeginAtom().GetSymbol(),
                            'atom2': bond.GetEndAtom().GetSymbol(),
                            'bond_type': str(bond.GetBondType()),
                            'importance': importance,
                            'fragment_smiles': fragment_smiles,
                            'fragment_pred': fragment_pred
                        })
        
        return {
            'original_smiles': drug_smiles,
            'original_prediction': original_pred,
            'atom_importance': sorted(atom_importance, key=lambda x: abs(x['importance']), reverse=True),
            'bond_importance': sorted(bond_importance, key=lambda x: abs(x['importance']), reverse=True),
            'most_important_atoms': sorted(atom_importance, key=lambda x: x['importance'], reverse=True)[:5],
            'most_important_bonds': sorted(bond_importance, key=lambda x: x['importance'], reverse=True)[:5]
        }
    
    def _smiles_to_model_input(self, smiles: str) -> Optional[Dict[str, torch.Tensor]]:
        """Convert SMILES to model input format."""
        # This is a placeholder - you'll need to implement this based on your data processing pipeline
        # It should convert SMILES to the same format as your training data
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Placeholder: create dummy molecular graph data
            # Replace this with your actual molecular featurization
            num_atoms = mol.GetNumAtoms()
            
            return {
                'x': torch.randn(num_atoms, 64).to(self.device),  # Node features
                'edge_index': torch.randint(0, num_atoms, (2, num_atoms * 2)).to(self.device),  # Edge indices
                'edge_attr': torch.randn(num_atoms * 2, 10).to(self.device),  # Edge features
                'batch': torch.zeros(num_atoms, dtype=torch.long).to(self.device)
            }
            
        except Exception as e:
            logger.error(f"Error converting SMILES to model input: {str(e)}")
            return None
    
    def _remove_atom(self, mol: Chem.Mol, atom_idx: int) -> Optional[Chem.Mol]:
        """Remove an atom from a molecule."""
        try:
            # Create editable molecule
            em = Chem.EditableMol(mol)
            
            # Remove the atom (this also removes associated bonds)
            em.RemoveAtom(atom_idx)
            
            # Get the modified molecule
            modified_mol = em.GetMol()
            
            # Sanitize the molecule
            Chem.SanitizeMol(modified_mol)
            
            return modified_mol
            
        except Exception:
            return None
    
    def _remove_bond(self, mol: Chem.Mol, bond_idx: int) -> Optional[Chem.Mol]:
        """Remove a bond from a molecule."""
        try:
            # Create editable molecule
            em = Chem.EditableMol(mol)
            
            # Remove the bond
            bond = mol.GetBondWithIdx(bond_idx)
            em.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
            # Get the modified molecule
            modified_mol = em.GetMol()
            
            # Sanitize the molecule
            Chem.SanitizeMol(modified_mol)
            
            return modified_mol
            
        except Exception:
            return None
    
    def visualize_substructure_importance(self,
                                        drug_smiles: str,
                                        importance_data: Dict[str, Any],
                                        save_path: Optional[str] = None,
                                        figsize: Tuple[int, int] = (15, 10)) -> Optional[plt.Figure]:
        """
        Visualize substructure importance on molecular structure.
        
        Args:
            drug_smiles: SMILES string
            importance_data: Results from substructure_importance_analysis
            save_path: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if saved
        """
        try:
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol is None:
                return None
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            
            # 1. Original molecule
            img = Draw.MolToImage(mol, size=(300, 300))
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f'Original Molecule\nPrediction: {importance_data["original_prediction"]:.3f}')
            
            # 2. Atom importance heatmap
            if importance_data['atom_importance']:
                atom_importances = [d['importance'] for d in importance_data['atom_importance']]
                atom_indices = [d['atom_idx'] for d in importance_data['atom_importance']]
                
                # Create highlighted molecule based on importance
                highlight_atoms = {}
                highlight_colors = {}
                
                # Normalize importances for coloring
                if len(atom_importances) > 0:
                    max_imp = max(abs(imp) for imp in atom_importances)
                    if max_imp > 0:
                        for idx, imp in zip(atom_indices, atom_importances):
                            normalized_imp = abs(imp) / max_imp
                            if normalized_imp > 0.1:  # Only highlight significant atoms
                                highlight_atoms[idx] = (1.0, 1.0 - normalized_imp, 1.0 - normalized_imp)
                                highlight_colors[idx] = (1.0, 1.0 - normalized_imp, 1.0 - normalized_imp)
                
                # Draw highlighted molecule
                if highlight_atoms:
                    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
                    drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms.keys()),
                                      highlightAtomColors=highlight_colors)
                    drawer.FinishDrawing()
                    
                    img_data = drawer.GetDrawingText()
                    img = Image.open(io.BytesIO(img_data))
                    ax2.imshow(img)
                else:
                    ax2.imshow(Draw.MolToImage(mol, size=(300, 300)))
                
                ax2.axis('off')
                ax2.set_title('Atom Importance\n(Red = High Importance)')
            
            # 3. Top important atoms bar plot
            if importance_data['most_important_atoms']:
                top_atoms = importance_data['most_important_atoms'][:10]
                atom_labels = [f"{a['atom_symbol']}_{a['atom_idx']}" for a in top_atoms]
                importances = [a['importance'] for a in top_atoms]
                
                colors = ['red' if imp > 0 else 'blue' for imp in importances]
                bars = ax3.barh(range(len(importances)), importances, color=colors, alpha=0.7)
                ax3.set_xlabel('Importance (Δ Prediction)')
                ax3.set_ylabel('Atom')
                ax3.set_title('Top Atom Contributions')
                ax3.set_yticks(range(len(atom_labels)))
                ax3.set_yticklabels(atom_labels)
                ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # 4. Top important bonds bar plot
            if importance_data['most_important_bonds']:
                top_bonds = importance_data['most_important_bonds'][:10]
                bond_labels = [f"{b['atom1']}-{b['atom2']}_{b['bond_idx']}" for b in top_bonds]
                importances = [b['importance'] for b in top_bonds]
                
                colors = ['red' if imp > 0 else 'blue' for imp in importances]
                bars = ax4.barh(range(len(importances)), importances, color=colors, alpha=0.7)
                ax4.set_xlabel('Importance (Δ Prediction)')
                ax4.set_ylabel('Bond')
                ax4.set_title('Top Bond Contributions')
                ax4.set_yticks(range(len(bond_labels)))
                ax4.set_yticklabels(bond_labels)
                ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            plt.suptitle(f'Substructure Importance Analysis: {drug_smiles}', fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Substructure importance visualization saved to {save_path}")
                return None
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing substructure importance: {str(e)}")
            return None
    
    def pharmacophore_analysis(self,
                             smiles_list: List[str],
                             predictions: List[float],
                             activity_threshold: float = 7.0) -> Dict[str, Any]:
        """
        Analyze pharmacophore patterns in active vs inactive compounds.
        
        Args:
            smiles_list: List of SMILES strings
            predictions: Corresponding predictions
            activity_threshold: Threshold for defining active compounds
            
        Returns:
            Dictionary containing pharmacophore analysis
        """
        active_mols = []
        inactive_mols = []
        
        for smiles, pred in zip(smiles_list, predictions):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            if pred >= activity_threshold:
                active_mols.append((mol, smiles, pred))
            else:
                inactive_mols.append((mol, smiles, pred))
        
        # Count pharmacophore features
        pharmacophore_counts = {name: {'active': [], 'inactive': []} 
                              for name in self.pharmacophores.keys()}
        
        for mol, smiles, pred in active_mols:
            for pharm_name, smarts in self.pharmacophores.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    pharmacophore_counts[pharm_name]['active'].append(len(matches))
        
        for mol, smiles, pred in inactive_mols:
            for pharm_name, smarts in self.pharmacophores.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    pharmacophore_counts[pharm_name]['inactive'].append(len(matches))
        
        # Calculate statistics
        pharmacophore_stats = {}
        for pharm_name, counts in pharmacophore_counts.items():
            active_counts = counts['active']
            inactive_counts = counts['inactive']
            
            pharmacophore_stats[pharm_name] = {
                'active_mean': np.mean(active_counts) if active_counts else 0,
                'inactive_mean': np.mean(inactive_counts) if inactive_counts else 0,
                'active_presence': len([c for c in active_counts if c > 0]) / len(active_counts) if active_counts else 0,
                'inactive_presence': len([c for c in inactive_counts if c > 0]) / len(inactive_counts) if inactive_counts else 0
            }
            
            # Calculate enrichment
            active_pres = pharmacophore_stats[pharm_name]['active_presence']
            inactive_pres = pharmacophore_stats[pharm_name]['inactive_presence']
            
            if inactive_pres > 0:
                pharmacophore_stats[pharm_name]['enrichment'] = active_pres / inactive_pres
            else:
                pharmacophore_stats[pharm_name]['enrichment'] = float('inf') if active_pres > 0 else 1.0
        
        return {
            'num_active': len(active_mols),
            'num_inactive': len(inactive_mols),
            'pharmacophore_statistics': pharmacophore_stats,
            'most_enriched': sorted(pharmacophore_stats.items(), 
                                  key=lambda x: x[1]['enrichment'], reverse=True)[:5]
        }

# Utility functions
def calculate_molecular_descriptors(smiles_list: List[str]) -> pd.DataFrame:
    """Calculate molecular descriptors for a list of SMILES."""
    descriptor_data = []
    
    descriptor_functions = {
        'MW': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'HBD': Descriptors.NumHDonors,
        'HBA': Descriptors.NumHAcceptors,
        'TPSA': Descriptors.TPSA,
        'RotBonds': Descriptors.NumRotatableBonds,
        'AromaticRings': Descriptors.NumAromaticRings,
        'SaturatedRings': Descriptors.NumSaturatedRings,
        'FractionCsp3': Descriptors.FractionCsp3,
        'MolMR': Descriptors.MolMR
    }
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            descriptor_values = [np.nan] * len(descriptor_functions)
        else:
            descriptor_values = [func(mol) for func in descriptor_functions.values()]
        
        descriptor_data.append([smiles] + descriptor_values)
    
    columns = ['SMILES'] + list(descriptor_functions.keys())
    return pd.DataFrame(descriptor_data, columns=columns)

# Example usage
if __name__ == "__main__":
    print("Testing MolecularInterpreter...")
    
    # Test molecular descriptors
    test_smiles = ['CCO', 'CCN', 'c1ccccc1', 'CC(=O)O']
    descriptors_df = calculate_molecular_descriptors(test_smiles)
    print(f"Molecular descriptors calculated for {len(test_smiles)} compounds")
    print(descriptors_df.head())
    
    # Test functional group detection
    interpreter = MolecularInterpreter(None)  # Mock model for testing
    
    # Test functional group patterns
    test_mol = Chem.MolFromSmiles('CC(=O)O')  # Acetic acid
    for group_name, smarts in interpreter.functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            matches = test_mol.GetSubstructMatches(pattern)
            if matches:
                print(f"Found {group_name}: {len(matches)} matches")
    
    print("MolecularInterpreter test completed successfully!")