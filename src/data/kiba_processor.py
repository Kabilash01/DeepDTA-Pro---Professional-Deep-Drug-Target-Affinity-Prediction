"""
KIBA Dataset Processor
Handles processing of the KIBA (Kinase Inhibitor BioActivity) dataset.
"""

import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml

# Setup logging
logger = logging.getLogger(__name__)

class KIBAProcessor:
    """
    Processor for the KIBA dataset containing kinase inhibitor bioactivity data.
    
    The KIBA dataset contains:
    - ligands_can.txt: Canonical SMILES for ligands
    - ligands_iso.txt: Isomeric SMILES for ligands  
    - proteins.txt: Protein sequences (text format)
    - proteins.csv: Protein information (CSV format)
    - Y: Binding affinity matrix (text format)
    - kiba_binding_affinity_v2.txt: Additional binding data
    """
    
    def __init__(self, data_path: str = "data/raw/kiba", config_path: str = "configs/data_config.yaml"):
        """
        Initialize KIBA dataset processor.
        
        Args:
            data_path: Path to raw KIBA dataset files
            config_path: Path to data configuration file
        """
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Data storage
        self.ligands_can = []
        self.ligands_iso = []
        self.proteins_sequences = []
        self.proteins_df = None
        self.affinity_matrix = None
        self.binding_data = None
        self.processed_data = None
        
        # Statistics
        self.stats = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load data processing configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('data', {})
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'kiba': {
                'ligands_can_file': 'ligands_can.txt',
                'ligands_iso_file': 'ligands_iso.txt',
                'proteins_file': 'proteins.txt',
                'proteins_csv_file': 'proteins.csv',
                'affinity_matrix_file': 'Y',
                'binding_file': 'kiba_binding_affinity_v2.txt',
                'log_transform': False
            },
            'quality_control': {
                'remove_duplicates': True,
                'valid_smiles_only': True,
                'valid_protein_only': True,
                'min_protein_length': 50,
                'max_protein_length': 2000
            }
        }
    
    def load_raw_data(self) -> None:
        """Load raw KIBA dataset files."""
        logger.info("Loading KIBA dataset files...")
        
        kiba_config = self.config.get('kiba', {})
        
        # Load canonical ligands
        ligands_can_file = self.data_path / kiba_config.get('ligands_can_file', 'ligands_can.txt')
        if ligands_can_file.exists():
            self.ligands_can = self._load_text_file(ligands_can_file)
            logger.info(f"Loaded {len(self.ligands_can)} canonical ligands")
        else:
            logger.warning(f"Canonical ligands file not found: {ligands_can_file}")
        
        # Load isomeric ligands
        ligands_iso_file = self.data_path / kiba_config.get('ligands_iso_file', 'ligands_iso.txt')
        if ligands_iso_file.exists():
            self.ligands_iso = self._load_text_file(ligands_iso_file)
            logger.info(f"Loaded {len(self.ligands_iso)} isomeric ligands")
        else:
            logger.warning(f"Isomeric ligands file not found: {ligands_iso_file}")
        
        # Load protein sequences (text format)
        proteins_file = self.data_path / kiba_config.get('proteins_file', 'proteins.txt')
        if proteins_file.exists():
            self.proteins_sequences = self._load_text_file(proteins_file)
            logger.info(f"Loaded {len(self.proteins_sequences)} protein sequences")
        else:
            logger.warning(f"Proteins file not found: {proteins_file}")
        
        # Load protein information (CSV format) if available
        proteins_csv_file = self.data_path / kiba_config.get('proteins_csv_file', 'proteins.csv')
        if proteins_csv_file.exists():
            self.proteins_df = pd.read_csv(proteins_csv_file)
            logger.info(f"Loaded protein CSV with {len(self.proteins_df)} entries")
        
        # Load affinity matrix
        affinity_matrix_file = self.data_path / kiba_config.get('affinity_matrix_file', 'Y')
        if affinity_matrix_file.exists():
            self.affinity_matrix = self._load_affinity_matrix(affinity_matrix_file)
            logger.info(f"Loaded affinity matrix with shape {self.affinity_matrix.shape}")
        else:
            logger.warning(f"Affinity matrix file not found: {affinity_matrix_file}")
        
        # Load additional binding data if available
        binding_file = self.data_path / kiba_config.get('binding_file', 'kiba_binding_affinity_v2.txt')
        if binding_file.exists():
            try:
                self.binding_data = pd.read_csv(binding_file, sep='\\t')
                logger.info(f"Loaded additional binding data with {len(self.binding_data)} entries")
            except Exception as e:
                logger.warning(f"Could not load binding data file: {e}")
        
        # Store initial statistics
        self.stats['raw_ligands_can'] = len(self.ligands_can)
        self.stats['raw_ligands_iso'] = len(self.ligands_iso)
        self.stats['raw_proteins'] = len(self.proteins_sequences)
        if self.affinity_matrix is not None:
            self.stats['raw_matrix_shape'] = self.affinity_matrix.shape
    
    def _load_text_file(self, file_path: Path) -> List[str]:
        """Load text file with one entry per line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def _load_affinity_matrix(self, file_path: Path) -> Optional[np.ndarray]:
        """Load affinity matrix from file."""
        try:
            # Try different formats
            if file_path.suffix in ['.txt', '.tsv']:
                # Text format with tab/space separation
                matrix = np.loadtxt(file_path, dtype=float)
            elif file_path.suffix == '.csv':
                # CSV format
                matrix = np.loadtxt(file_path, delimiter=',', dtype=float)
            else:
                # Generic text format
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse lines into matrix
                matrix_data = []
                for line in lines:
                    if line.strip():
                        # Try different separators
                        if '\\t' in line:
                            row = [float(x) if x.strip() != 'nan' else np.nan for x in line.strip().split('\\t')]
                        elif ',' in line:
                            row = [float(x) if x.strip() != 'nan' else np.nan for x in line.strip().split(',')]
                        else:
                            row = [float(x) if x.strip() != 'nan' else np.nan for x in line.strip().split()]
                        matrix_data.append(row)
                
                matrix = np.array(matrix_data, dtype=float)
            
            return matrix
            
        except Exception as e:
            logger.error(f"Error loading affinity matrix {file_path}: {e}")
            return None
    
    def validate_data(self) -> None:
        """Validate and clean the dataset."""
        logger.info("Validating KIBA dataset...")
        
        quality_config = self.config.get('quality_control', {})
        
        # Validate SMILES strings (use canonical if available, otherwise isomeric)
        ligands_to_use = self.ligands_can if self.ligands_can else self.ligands_iso
        
        if quality_config.get('valid_smiles_only', True) and ligands_to_use:
            logger.info("Validating SMILES strings...")
            
            # Import RDKit for validation (if available)
            try:
                from rdkit import Chem
                
                valid_ligands = []
                valid_indices = []
                
                for idx, smiles in enumerate(ligands_to_use):
                    if isinstance(smiles, str) and smiles.strip():
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            valid_ligands.append(smiles)
                            valid_indices.append(idx)
                        else:
                            logger.warning(f"Invalid SMILES at index {idx}: {smiles}")
                    else:
                        logger.warning(f"Empty/invalid SMILES at index {idx}")
                
                # Update ligands lists
                if self.ligands_can:
                    self.ligands_can = [self.ligands_can[i] for i in valid_indices]
                if self.ligands_iso:
                    self.ligands_iso = [self.ligands_iso[i] for i in valid_indices]
                
                # Update affinity matrix
                if self.affinity_matrix is not None:
                    self.affinity_matrix = self.affinity_matrix[valid_indices, :]
                
                logger.info(f"Retained {len(valid_ligands)} ligands with valid SMILES")
                
            except ImportError:
                logger.warning("RDKit not available for SMILES validation")
        
        # Validate protein sequences
        if quality_config.get('valid_protein_only', True) and self.proteins_sequences:
            logger.info("Validating protein sequences...")
            
            valid_proteins = []
            valid_indices = []
            min_length = quality_config.get('min_protein_length', 50)
            max_length = quality_config.get('max_protein_length', 2000)
            
            for idx, sequence in enumerate(self.proteins_sequences):
                if isinstance(sequence, str) and sequence.strip():
                    seq_len = len(sequence.strip())
                    if min_length <= seq_len <= max_length:
                        # Check for valid amino acids
                        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                        if all(aa.upper() in valid_aa for aa in sequence.strip()):
                            valid_proteins.append(sequence)
                            valid_indices.append(idx)
                        else:
                            logger.warning(f"Invalid amino acids in protein {idx}")
                    else:
                        logger.warning(f"Protein {idx} length {seq_len} outside range [{min_length}, {max_length}]")
                else:
                    logger.warning(f"Empty/invalid protein sequence at index {idx}")
            
            # Update proteins list
            self.proteins_sequences = valid_proteins
            
            # Update affinity matrix
            if self.affinity_matrix is not None:
                self.affinity_matrix = self.affinity_matrix[:, valid_indices]
            
            logger.info(f"Retained {len(valid_proteins)} proteins with valid sequences")
        
        # Store validation statistics
        self.stats['valid_ligands'] = len(self.ligands_can) if self.ligands_can else len(self.ligands_iso)
        self.stats['valid_proteins'] = len(self.proteins_sequences)
        if self.affinity_matrix is not None:
            self.stats['valid_matrix_shape'] = self.affinity_matrix.shape
    
    def process_affinities(self) -> None:
        """Process binding affinity values from the matrix."""
        logger.info("Processing binding affinities...")
        
        if self.affinity_matrix is None:
            logger.warning("No affinity matrix available for processing")
            return
        
        kiba_config = self.config.get('kiba', {})
        
        # KIBA scores are typically already in a processed form (log-scale)
        # so we usually don't need additional log transformation
        if kiba_config.get('log_transform', False):
            # Apply log transformation if specified
            positive_mask = self.affinity_matrix > 0
            self.affinity_matrix[positive_mask] = np.log10(self.affinity_matrix[positive_mask])
            logger.info("Applied log transformation to KIBA scores")
        
        # Calculate statistics (excluding NaN values)
        valid_affinities = self.affinity_matrix[~np.isnan(self.affinity_matrix)]
        
        if len(valid_affinities) > 0:
            self.stats['affinity_range'] = (float(np.min(valid_affinities)), float(np.max(valid_affinities)))
            self.stats['affinity_mean'] = float(np.mean(valid_affinities))
            self.stats['affinity_std'] = float(np.std(valid_affinities))
            self.stats['valid_affinities'] = len(valid_affinities)
            self.stats['total_possible_pairs'] = self.affinity_matrix.size
            self.stats['missing_affinities'] = np.sum(np.isnan(self.affinity_matrix))
            
            logger.info(f"Processed {len(valid_affinities)} affinity values")
            logger.info(f"Affinity range: [{self.stats['affinity_range'][0]:.3f}, {self.stats['affinity_range'][1]:.3f}]")
            logger.info(f"Affinity mean ± std: {self.stats['affinity_mean']:.3f} ± {self.stats['affinity_std']:.3f}")
            logger.info(f"Missing values: {self.stats['missing_affinities']} / {self.stats['total_possible_pairs']}")
    
    def create_interaction_pairs(self) -> List[Dict[str, Any]]:
        """Create drug-protein interaction pairs from the affinity matrix."""
        logger.info("Creating drug-protein interaction pairs...")
        
        if self.affinity_matrix is None:
            logger.error("No affinity matrix available")
            return []
        
        pairs = []
        ligands_to_use = self.ligands_can if self.ligands_can else self.ligands_iso
        
        # Iterate through the affinity matrix
        for drug_idx in range(self.affinity_matrix.shape[0]):
            for protein_idx in range(self.affinity_matrix.shape[1]):
                affinity = self.affinity_matrix[drug_idx, protein_idx]
                
                # Skip missing values
                if np.isnan(affinity):
                    continue
                
                # Get drug and protein information
                drug_smiles = ligands_to_use[drug_idx] if drug_idx < len(ligands_to_use) else ""
                protein_sequence = self.proteins_sequences[protein_idx] if protein_idx < len(self.proteins_sequences) else ""
                
                # Skip if missing essential data
                if not drug_smiles or not protein_sequence:
                    continue
                
                # Get additional information if available
                drug_name = f"KIBA_Drug_{drug_idx}"
                protein_name = f"KIBA_Protein_{protein_idx}"
                
                if self.proteins_df is not None and protein_idx < len(self.proteins_df):
                    protein_info = self.proteins_df.iloc[protein_idx]
                    protein_name = protein_info.get('name', protein_name)
                
                # Create interaction pair
                pair = {
                    'drug_id': drug_idx,
                    'protein_id': protein_idx,
                    'smiles': drug_smiles,
                    'protein_sequence': protein_sequence,
                    'affinity': float(affinity),
                    'drug_name': drug_name,
                    'protein_name': protein_name,
                }
                
                pairs.append(pair)
        
        logger.info(f"Created {len(pairs)} drug-protein interaction pairs")
        return pairs
    
    def process_dataset(self) -> Dict[str, Any]:
        """
        Complete KIBA dataset processing pipeline.
        
        Returns:
            Dictionary containing processed data and statistics
        """
        logger.info("Starting KIBA dataset processing...")
        
        try:
            # Load raw data
            self.load_raw_data()
            
            # Validate and clean data
            self.validate_data()
            
            # Process affinity values
            self.process_affinities()
            
            # Create interaction pairs
            interaction_pairs = self.create_interaction_pairs()
            
            # Compile processed data
            ligands_to_use = self.ligands_can if self.ligands_can else self.ligands_iso
            
            drugs_data = [
                {'drug_id': i, 'smiles': smiles, 'name': f'KIBA_Drug_{i}'}
                for i, smiles in enumerate(ligands_to_use)
            ]
            
            proteins_data = [
                {'protein_id': i, 'sequence': seq, 'name': f'KIBA_Protein_{i}'}
                for i, seq in enumerate(self.proteins_sequences)
            ]
            
            # Add protein information from CSV if available
            if self.proteins_df is not None:
                for i, (idx, row) in enumerate(self.proteins_df.iterrows()):
                    if i < len(proteins_data):
                        proteins_data[i].update(row.to_dict())
            
            self.processed_data = {
                'dataset_name': 'kiba',
                'interaction_pairs': interaction_pairs,
                'drugs_data': drugs_data,
                'proteins_data': proteins_data,
                'statistics': self.stats,
                'config': self.config
            }
            
            # Save processed data
            self.save_processed_data()
            
            logger.info("KIBA dataset processing completed successfully!")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error processing KIBA dataset: {e}")
            raise
    
    def save_processed_data(self, output_path: Optional[str] = None) -> None:
        """Save processed data to pickle file.""" 
        if output_path is None:
            output_path = "data/processed/kiba_data.pkl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.processed_data, f)
        
        logger.info(f"Saved processed KIBA data to {output_path}")
    
    def load_processed_data(self, input_path: Optional[str] = None) -> Dict[str, Any]:
        """Load previously processed data."""
        if input_path is None:
            input_path = "data/processed/kiba_data.pkl"
        
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            self.processed_data = pickle.load(f)
        
        logger.info(f"Loaded processed KIBA data from {input_path}")
        return self.processed_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()
    
    def print_summary(self) -> None:
        """Print dataset summary."""
        print("\n" + "="*50)
        print("KIBA DATASET SUMMARY")
        print("="*50)
        
        if self.stats:
            print(f"Raw data:")
            print(f"  - Canonical ligands: {self.stats.get('raw_ligands_can', 'N/A')}")
            print(f"  - Isomeric ligands: {self.stats.get('raw_ligands_iso', 'N/A')}")
            print(f"  - Proteins: {self.stats.get('raw_proteins', 'N/A')}")
            if 'raw_matrix_shape' in self.stats:
                print(f"  - Matrix shape: {self.stats['raw_matrix_shape']}")
            
            print(f"\nProcessed data:")
            print(f"  - Valid ligands: {self.stats.get('valid_ligands', 'N/A')}")
            print(f"  - Valid proteins: {self.stats.get('valid_proteins', 'N/A')}")
            if 'valid_matrix_shape' in self.stats:
                print(f"  - Matrix shape: {self.stats['valid_matrix_shape']}")
            
            if 'affinity_range' in self.stats:
                print(f"\nAffinity statistics:")
                print(f"  - Range: [{self.stats['affinity_range'][0]:.3f}, {self.stats['affinity_range'][1]:.3f}]")
                print(f"  - Mean ± Std: {self.stats['affinity_mean']:.3f} ± {self.stats['affinity_std']:.3f}")
                print(f"  - Valid entries: {self.stats.get('valid_affinities', 'N/A')}")
                print(f"  - Missing entries: {self.stats.get('missing_affinities', 'N/A')}")
        
        print("="*50 + "\n")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Process KIBA dataset
    processor = KIBAProcessor()
    processed_data = processor.process_dataset()
    processor.print_summary()