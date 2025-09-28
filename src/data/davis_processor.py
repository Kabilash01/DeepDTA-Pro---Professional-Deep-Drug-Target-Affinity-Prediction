"""
Davis Dataset Processor
Handles processing of the Davis drug-target interaction dataset.
"""

import os
import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import yaml

# Setup logging
logger = logging.getLogger(__name__)

class DavisProcessor:
    """
    Processor for the Davis dataset containing drug-protein binding affinities.
    
    The Davis dataset contains:
    - drugs.csv: Drug information with SMILES strings
    - proteins.csv: Protein sequences and information  
    - drug_protein_affinity.csv: Binding affinity values (Kd in nM)
    """
    
    def __init__(self, data_path: str = "data/raw/davis", config_path: str = "configs/data_config.yaml"):
        """
        Initialize Davis dataset processor.
        
        Args:
            data_path: Path to raw Davis dataset files
            config_path: Path to data configuration file
        """
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Data storage
        self.drugs_df = None
        self.proteins_df = None
        self.affinity_df = None
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
            'davis': {
                'drugs_file': 'drugs.csv',
                'proteins_file': 'proteins.csv', 
                'affinity_file': 'drug_protein_affinity.csv',
                'affinity_column': 'affinity',
                'log_transform': True
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
        """Load raw Davis dataset files."""
        logger.info("Loading Davis dataset files...")
        
        davis_config = self.config.get('davis', {})
        
        # Load drugs data
        drugs_file = self.data_path / davis_config.get('drugs_file', 'drugs.csv')
        if not drugs_file.exists():
            raise FileNotFoundError(f"Drugs file not found: {drugs_file}")
        
        self.drugs_df = pd.read_csv(drugs_file)
        logger.info(f"Loaded {len(self.drugs_df)} drugs")
        
        # Load proteins data
        proteins_file = self.data_path / davis_config.get('proteins_file', 'proteins.csv')
        if not proteins_file.exists():
            raise FileNotFoundError(f"Proteins file not found: {proteins_file}")
        
        self.proteins_df = pd.read_csv(proteins_file)
        logger.info(f"Loaded {len(self.proteins_df)} proteins")
        
        # Load affinity data
        affinity_file = self.data_path / davis_config.get('affinity_file', 'drug_protein_affinity.csv')
        if not affinity_file.exists():
            raise FileNotFoundError(f"Affinity file not found: {affinity_file}")
        
        self.affinity_df = pd.read_csv(affinity_file)
        logger.info(f"Loaded {len(self.affinity_df)} drug-protein interactions")
        
        # Store initial statistics
        self.stats['raw_drugs'] = len(self.drugs_df)
        self.stats['raw_proteins'] = len(self.proteins_df)
        self.stats['raw_interactions'] = len(self.affinity_df)
    
    def validate_data(self) -> None:
        """Validate and clean the dataset."""
        logger.info("Validating Davis dataset...")
        
        quality_config = self.config.get('quality_control', {})
        
        # Validate SMILES strings
        if quality_config.get('valid_smiles_only', True):
            logger.info("Validating SMILES strings...")
            valid_smiles = []
            
            for idx, row in self.drugs_df.iterrows():
                smiles = row.get('smiles', '')
                if isinstance(smiles, str) and smiles.strip():
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        valid_smiles.append(True)
                    else:
                        valid_smiles.append(False)
                        logger.warning(f"Invalid SMILES at index {idx}: {smiles}")
                else:
                    valid_smiles.append(False)
                    logger.warning(f"Empty/invalid SMILES at index {idx}")
            
            # Filter drugs with valid SMILES
            self.drugs_df = self.drugs_df[valid_smiles].reset_index(drop=True)
            logger.info(f"Retained {len(self.drugs_df)} drugs with valid SMILES")
        
        # Validate protein sequences
        if quality_config.get('valid_protein_only', True):
            logger.info("Validating protein sequences...")
            valid_proteins = []
            min_length = quality_config.get('min_protein_length', 50)
            max_length = quality_config.get('max_protein_length', 2000)
            
            for idx, row in self.proteins_df.iterrows():
                sequence = row.get('sequence', '')
                if isinstance(sequence, str) and sequence.strip():
                    seq_len = len(sequence.strip())
                    if min_length <= seq_len <= max_length:
                        # Check for valid amino acids
                        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                        if all(aa.upper() in valid_aa for aa in sequence.strip()):
                            valid_proteins.append(True)
                        else:
                            valid_proteins.append(False)
                            logger.warning(f"Invalid amino acids in protein {idx}")
                    else:
                        valid_proteins.append(False)
                        logger.warning(f"Protein {idx} length {seq_len} outside range [{min_length}, {max_length}]")
                else:
                    valid_proteins.append(False)
                    logger.warning(f"Empty/invalid protein sequence at index {idx}")
            
            # Filter proteins with valid sequences
            self.proteins_df = self.proteins_df[valid_proteins].reset_index(drop=True)
            logger.info(f"Retained {len(self.proteins_df)} proteins with valid sequences")
        
        # Filter interactions to keep only valid drug-protein pairs
        if 'drug_id' in self.affinity_df.columns and 'protein_id' in self.affinity_df.columns:
            valid_drug_ids = set(self.drugs_df.index) if 'drug_id' not in self.drugs_df.columns else set(self.drugs_df['drug_id'])
            valid_protein_ids = set(self.proteins_df.index) if 'protein_id' not in self.proteins_df.columns else set(self.proteins_df['protein_id'])
            
            mask = (self.affinity_df['drug_id'].isin(valid_drug_ids) & 
                   self.affinity_df['protein_id'].isin(valid_protein_ids))
            self.affinity_df = self.affinity_df[mask].reset_index(drop=True)
            logger.info(f"Retained {len(self.affinity_df)} valid drug-protein interactions")
        
        # Remove duplicates if specified
        if quality_config.get('remove_duplicates', True):
            initial_count = len(self.affinity_df)
            if 'drug_id' in self.affinity_df.columns and 'protein_id' in self.affinity_df.columns:
                self.affinity_df = self.affinity_df.drop_duplicates(
                    subset=['drug_id', 'protein_id']
                ).reset_index(drop=True)
                removed = initial_count - len(self.affinity_df)
                if removed > 0:
                    logger.info(f"Removed {removed} duplicate interactions")
        
        # Store validation statistics
        self.stats['valid_drugs'] = len(self.drugs_df)
        self.stats['valid_proteins'] = len(self.proteins_df)
        self.stats['valid_interactions'] = len(self.affinity_df)
    
    def process_affinities(self) -> None:
        """Process binding affinity values."""
        logger.info("Processing binding affinities...")
        
        davis_config = self.config.get('davis', {})
        affinity_col = davis_config.get('affinity_column', 'affinity')
        
        if affinity_col not in self.affinity_df.columns:
            # Try common column names
            possible_cols = ['affinity', 'binding_affinity', 'kd', 'Kd', 'pKd', 'value']
            for col in possible_cols:
                if col in self.affinity_df.columns:
                    affinity_col = col
                    break
            else:
                raise ValueError(f"No affinity column found. Available columns: {self.affinity_df.columns.tolist()}")
        
        # Extract affinity values
        affinities = self.affinity_df[affinity_col].values
        
        # Handle missing values
        valid_mask = pd.notna(affinities)
        affinities = affinities[valid_mask]
        self.affinity_df = self.affinity_df[valid_mask].reset_index(drop=True)
        
        # Log transform if specified (Davis uses Kd values in nM, so log transform is common)
        if davis_config.get('log_transform', True):
            # Convert to pKd scale: pKd = -log10(Kd_nM * 1e-9)
            affinities = np.maximum(affinities, 1e-3)  # Avoid log(0)
            affinities = -np.log10(affinities * 1e-9)
            logger.info("Applied log transformation (pKd scale)")
        
        # Update affinity values
        self.affinity_df[affinity_col] = affinities
        
        # Store statistics
        self.stats['affinity_range'] = (float(np.min(affinities)), float(np.max(affinities)))
        self.stats['affinity_mean'] = float(np.mean(affinities))
        self.stats['affinity_std'] = float(np.std(affinities))
        
        logger.info(f"Processed {len(affinities)} affinity values")
        logger.info(f"Affinity range: [{self.stats['affinity_range'][0]:.3f}, {self.stats['affinity_range'][1]:.3f}]")
        logger.info(f"Affinity mean ± std: {self.stats['affinity_mean']:.3f} ± {self.stats['affinity_std']:.3f}")
    
    def create_interaction_pairs(self) -> List[Dict[str, Any]]:
        """Create drug-protein interaction pairs with features."""
        logger.info("Creating drug-protein interaction pairs...")
        
        pairs = []
        davis_config = self.config.get('davis', {})
        affinity_col = davis_config.get('affinity_column', 'affinity')
        
        # Determine column mappings
        drug_id_col = 'drug_id' if 'drug_id' in self.affinity_df.columns else None
        protein_id_col = 'protein_id' if 'protein_id' in self.affinity_df.columns else None
        
        for idx, row in self.affinity_df.iterrows():
            try:
                # Get drug information
                if drug_id_col:
                    drug_idx = self.drugs_df[self.drugs_df['drug_id'] == row[drug_id_col]].index[0]
                else:
                    drug_idx = row.get('drug_id', idx) if 'drug_id' in row else idx
                
                drug_data = self.drugs_df.iloc[drug_idx]
                
                # Get protein information  
                if protein_id_col:
                    protein_idx = self.proteins_df[self.proteins_df['protein_id'] == row[protein_id_col]].index[0]
                else:
                    protein_idx = row.get('protein_id', idx) if 'protein_id' in row else idx
                
                protein_data = self.proteins_df.iloc[protein_idx]
                
                # Create interaction pair
                pair = {
                    'drug_id': drug_idx,
                    'protein_id': protein_idx,
                    'smiles': drug_data.get('smiles', ''),
                    'protein_sequence': protein_data.get('sequence', ''),
                    'affinity': row[affinity_col],
                    'drug_name': drug_data.get('name', f'Drug_{drug_idx}'),
                    'protein_name': protein_data.get('name', f'Protein_{protein_idx}'),
                }
                
                pairs.append(pair)
                
            except Exception as e:
                logger.warning(f"Error processing interaction {idx}: {e}")
                continue
        
        logger.info(f"Created {len(pairs)} drug-protein interaction pairs")
        return pairs
    
    def process_dataset(self) -> Dict[str, Any]:
        """
        Complete Davis dataset processing pipeline.
        
        Returns:
            Dictionary containing processed data and statistics
        """
        logger.info("Starting Davis dataset processing...")
        
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
            self.processed_data = {
                'dataset_name': 'davis',
                'interaction_pairs': interaction_pairs,
                'drugs_data': self.drugs_df.to_dict('records'),
                'proteins_data': self.proteins_df.to_dict('records'),
                'statistics': self.stats,
                'config': self.config
            }
            
            # Save processed data
            self.save_processed_data()
            
            logger.info("Davis dataset processing completed successfully!")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error processing Davis dataset: {e}")
            raise
    
    def save_processed_data(self, output_path: Optional[str] = None) -> None:
        """Save processed data to pickle file."""
        if output_path is None:
            output_path = "data/processed/davis_data.pkl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.processed_data, f)
        
        logger.info(f"Saved processed Davis data to {output_path}")
    
    def load_processed_data(self, input_path: Optional[str] = None) -> Dict[str, Any]:
        """Load previously processed data."""
        if input_path is None:
            input_path = "data/processed/davis_data.pkl"
        
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            self.processed_data = pickle.load(f)
        
        logger.info(f"Loaded processed Davis data from {input_path}")
        return self.processed_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()
    
    def print_summary(self) -> None:
        """Print dataset summary."""
        print("\n" + "="*50)
        print("DAVIS DATASET SUMMARY")
        print("="*50)
        
        if self.stats:
            print(f"Raw data:")
            print(f"  - Drugs: {self.stats.get('raw_drugs', 'N/A')}")
            print(f"  - Proteins: {self.stats.get('raw_proteins', 'N/A')}")
            print(f"  - Interactions: {self.stats.get('raw_interactions', 'N/A')}")
            
            print(f"\nProcessed data:")
            print(f"  - Valid drugs: {self.stats.get('valid_drugs', 'N/A')}")
            print(f"  - Valid proteins: {self.stats.get('valid_proteins', 'N/A')}")
            print(f"  - Valid interactions: {self.stats.get('valid_interactions', 'N/A')}")
            
            if 'affinity_range' in self.stats:
                print(f"\nAffinity statistics:")
                print(f"  - Range: [{self.stats['affinity_range'][0]:.3f}, {self.stats['affinity_range'][1]:.3f}]")
                print(f"  - Mean ± Std: {self.stats['affinity_mean']:.3f} ± {self.stats['affinity_std']:.3f}")
        
        print("="*50 + "\n")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Process Davis dataset
    processor = DavisProcessor()
    processed_data = processor.process_dataset()
    processor.print_summary()