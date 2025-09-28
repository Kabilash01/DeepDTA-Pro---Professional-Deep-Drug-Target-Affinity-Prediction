"""
Molecular Feature Extraction
Converts SMILES strings to molecular graphs and features for GNN input.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from torch_geometric.data import Data, Batch
import pickle
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class MolecularFeatureExtractor:
    """
    Extracts molecular features and constructs molecular graphs from SMILES strings.
    
    Features include:
    - Atomic features (element, charge, hybridization, etc.)
    - Bond features (bond type, stereo, conjugation, etc.)
    - Molecular graphs for GNN processing
    """
    
    def __init__(self, max_atoms: int = 100, include_hydrogens: bool = False):
        """
        Initialize molecular feature extractor.
        
        Args:
            max_atoms: Maximum number of atoms per molecule
            include_hydrogens: Whether to include hydrogen atoms
        """
        self.max_atoms = max_atoms
        self.include_hydrogens = include_hydrogens
        
        # Feature dimensions
        self.atom_feature_dim = 74  # Total atomic features
        self.bond_feature_dim = 12  # Total bond features
        
        # Atomic feature mappings
        self.atom_features = {
            'atomic_num': list(range(1, 119)),  # 1-118 (all elements)
            'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
            'hybridization': [0, 1, 2, 3, 4, 5],  # SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED
            'num_radical_electrons': [0, 1, 2, 3, 4],
            'degree': [0, 1, 2, 3, 4, 5, 6],
            'total_valence': [0, 1, 2, 3, 4, 5, 6, 7, 8]
        }
        
        # Bond feature mappings
        self.bond_features = {
            'bond_type': [0, 1, 2, 3, 4],  # SINGLE, DOUBLE, TRIPLE, AROMATIC, UNSPECIFIED
            'stereo': [0, 1, 2, 3, 4, 5],  # STEREONONE, STEREOANY, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS
            'conjugated': [0, 1],
            'in_ring': [0, 1]
        }
        
        # Cache for processed molecules  
        self._molecule_cache = {}
    
    def _get_atom_features(self, atom) -> List[float]:
        """
        Extract atomic features from RDKit atom object.
        
        Args:
            atom: RDKit atom object
            
        Returns:
            List of atomic features
        """
        try:
            from rdkit import Chem
            
            features = []
            
            # Atomic number (one-hot encoded)
            atomic_num = atom.GetAtomicNum()
            features.extend([1.0 if atomic_num == x else 0.0 for x in self.atom_features['atomic_num']])
            
            # Formal charge (one-hot encoded)
            formal_charge = atom.GetFormalCharge()
            features.extend([1.0 if formal_charge == x else 0.0 for x in self.atom_features['formal_charge']])
            
            # Hybridization (one-hot encoded)
            hybridization = int(atom.GetHybridization())
            features.extend([1.0 if hybridization == x else 0.0 for x in self.atom_features['hybridization']])
            
            # Number of radical electrons (one-hot encoded)
            num_radical = atom.GetNumRadicalElectrons()
            features.extend([1.0 if num_radical == x else 0.0 for x in self.atom_features['num_radical_electrons']])
            
            # Degree (one-hot encoded)
            degree = atom.GetDegree()
            features.extend([1.0 if degree == x else 0.0 for x in self.atom_features['degree']])
            
            # Total valence (one-hot encoded)
            total_valence = atom.GetTotalValence()
            features.extend([1.0 if total_valence == x else 0.0 for x in self.atom_features['total_valence']])
            
            # Additional boolean features
            features.append(1.0 if atom.GetIsAromatic() else 0.0)
            features.append(1.0 if atom.IsInRing() else 0.0)
            features.append(1.0 if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0.0)
            
            # Mass (normalized)
            mass = atom.GetMass()
            features.append(mass / 100.0)  # Rough normalization
            
            return features
            
        except ImportError:
            logger.error("RDKit not available for atom feature extraction")
            return [0.0] * self.atom_feature_dim
        except Exception as e:
            logger.warning(f"Error extracting atom features: {e}")
            return [0.0] * self.atom_feature_dim
    
    def _get_bond_features(self, bond) -> List[float]:
        """
        Extract bond features from RDKit bond object.
        
        Args:
            bond: RDKit bond object
            
        Returns:
            List of bond features
        """
        try:
            from rdkit import Chem
            
            features = []
            
            # Bond type (one-hot encoded)
            bond_type = int(bond.GetBondType())
            features.extend([1.0 if bond_type == x else 0.0 for x in self.bond_features['bond_type']])
            
            # Stereo (one-hot encoded)
            stereo = int(bond.GetStereo())
            features.extend([1.0 if stereo == x else 0.0 for x in self.bond_features['stereo']])
            
            # Additional boolean features
            features.append(1.0 if bond.GetIsConjugated() else 0.0)
            features.append(1.0 if bond.IsInRing() else 0.0)
            
            return features
            
        except ImportError:
            logger.error("RDKit not available for bond feature extraction")
            return [0.0] * self.bond_feature_dim
        except Exception as e:
            logger.warning(f"Error extracting bond features: {e}")
            return [0.0] * self.bond_feature_dim
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES string to PyTorch Geometric graph.
        
        Args:
            smiles: SMILES string
            
        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        if smiles in self._molecule_cache:
            return self._molecule_cache[smiles]
        
        try:
            from rdkit import Chem
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens if specified
            if self.include_hydrogens:
                mol = Chem.AddHs(mol)
            
            # Check molecule size
            if mol.GetNumAtoms() > self.max_atoms:
                logger.warning(f"Molecule too large ({mol.GetNumAtoms()} atoms): {smiles}")
                return None
            
            # Extract atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                atom_features.append(features)
            
            # Extract bonds and bond features
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                
                bond_feat = self._get_bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
            
            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # Handle molecules with no bonds (single atoms)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, self.bond_feature_dim), dtype=torch.float)
            
            # Create PyTorch Geometric data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles,
                num_atoms=mol.GetNumAtoms()
            )
            
            # Cache the result
            self._molecule_cache[smiles] = data
            
            return data
            
        except ImportError:
            logger.error("RDKit not available for molecular graph construction")
            return None
        except Exception as e:
            logger.error(f"Error converting SMILES to graph: {smiles}, Error: {e}")
            return None
    
    def extract_molecular_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        Extract traditional molecular descriptors.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of molecular descriptors
        """
        descriptors = {}
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return descriptors
            
            # Basic descriptors
            descriptors['mol_weight'] = Descriptors.MolWt(mol)
            descriptors['log_p'] = Descriptors.MolLogP(mol)
            descriptors['num_h_donors'] = Descriptors.NumHDonors(mol)
            descriptors['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['tpsa'] = Descriptors.TPSA(mol)
            descriptors['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['fraction_sp3'] = rdMolDescriptors.CalcFractionCsp3(mol)
            
            # Lipinski's Rule of Five
            descriptors['lipinski_violations'] = sum([
                descriptors['mol_weight'] > 500,
                descriptors['log_p'] > 5,
                descriptors['num_h_donors'] > 5,
                descriptors['num_h_acceptors'] > 10
            ])
            
            return descriptors
            
        except ImportError:
            logger.error("RDKit not available for descriptor calculation")
            return descriptors
        except Exception as e:
            logger.warning(f"Error calculating descriptors for {smiles}: {e}")
            return descriptors
    
    def process_smiles_list(self, smiles_list: List[str], 
                           include_descriptors: bool = True) -> Dict[str, Any]:
        """
        Process a list of SMILES strings to extract molecular features.
        
        Args:
            smiles_list: List of SMILES strings
            include_descriptors: Whether to include traditional descriptors
            
        Returns:
            Dictionary containing processed molecular data
        """
        logger.info(f"Processing {len(smiles_list)} SMILES strings...")
        
        graphs = []
        descriptors_list = []
        valid_smiles = []
        invalid_count = 0
        
        for i, smiles in enumerate(smiles_list):
            # Convert to graph
            graph = self.smiles_to_graph(smiles)
            
            if graph is not None:
                graphs.append(graph)
                valid_smiles.append(smiles)
                
                # Extract descriptors if requested
                if include_descriptors:
                    desc = self.extract_molecular_descriptors(smiles)
                    descriptors_list.append(desc)
                
            else:
                invalid_count += 1
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(smiles_list)} molecules")
        
        logger.info(f"Successfully processed {len(graphs)} molecules, {invalid_count} failed")
        
        result = {
            'graphs': graphs,
            'valid_smiles': valid_smiles,
            'invalid_count': invalid_count,
            'success_rate': len(graphs) / len(smiles_list) if smiles_list else 0.0
        }
        
        if include_descriptors:
            result['descriptors'] = descriptors_list
        
        return result
    
    def create_batch(self, graphs: List[Data]) -> Batch:
        """
        Create a batch of molecular graphs.
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            
        Returns:
            Batched graph data
        """
        return Batch.from_data_list(graphs)
    
    def save_processed_molecules(self, processed_data: Dict[str, Any], 
                                output_path: str) -> None:
        """
        Save processed molecular data.
        
        Args:
            processed_data: Dictionary containing processed molecular data
            output_path: Path to save the data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Saved processed molecular data to {output_path}")
    
    def load_processed_molecules(self, input_path: str) -> Dict[str, Any]:
        """
        Load processed molecular data.
        
        Args:
            input_path: Path to the saved data
            
        Returns:
            Dictionary containing processed molecular data
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded processed molecular data from {input_path}")
        return data
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions of atom and bond features.
        
        Returns:
            Tuple of (atom_feature_dim, bond_feature_dim)
        """
        return self.atom_feature_dim, self.bond_feature_dim
    
    def clear_cache(self) -> None:
        """Clear the molecule cache."""
        self._molecule_cache.clear()
        logger.info("Cleared molecule cache")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example SMILES strings
    test_smiles = [
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Imatinib
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    # Initialize extractor
    extractor = MolecularFeatureExtractor()
    
    # Process molecules
    processed = extractor.process_smiles_list(test_smiles)
    
    print(f"Processed {len(processed['graphs'])} molecules")
    print(f"Success rate: {processed['success_rate']:.2%}")
    
    # Print feature dimensions
    atom_dim, bond_dim = extractor.get_feature_dimensions()
    print(f"Atom feature dimension: {atom_dim}")
    print(f"Bond feature dimension: {bond_dim}")