"""
Dataset Merger
Combines Davis and KIBA datasets for cross-dataset training and evaluation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Setup logging
logger = logging.getLogger(__name__)

class DatasetMerger:
    """
    Merges Davis and KIBA datasets for unified training and cross-dataset evaluation.
    
    Handles:
    - Data format harmonization
    - Affinity scale alignment
    - Overlapping compound/protein detection
    - Cross-dataset validation splits
    """
    
    def __init__(self, davis_data: Optional[Dict[str, Any]] = None, 
                 kiba_data: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset merger.
        
        Args:
            davis_data: Processed Davis dataset
            kiba_data: Processed KIBA dataset
        """
        self.davis_data = davis_data
        self.kiba_data = kiba_data
        
        # Merged data storage
        self.merged_data = None
        self.cross_dataset_mapping = {}
        
        # Statistics
        self.merge_stats = {}
    
    def load_datasets(self, davis_path: str = "data/processed/davis_data.pkl",
                     kiba_path: str = "data/processed/kiba_data.pkl") -> None:
        """
        Load preprocessed datasets.
        
        Args:
            davis_path: Path to Davis dataset
            kiba_path: Path to KIBA dataset
        """
        logger.info("Loading preprocessed datasets...")
        
        # Load Davis data
        if Path(davis_path).exists():
            with open(davis_path, 'rb') as f:
                self.davis_data = pickle.load(f)
            logger.info(f"Loaded Davis dataset with {len(self.davis_data['interaction_pairs'])} interactions")
        else:
            logger.warning(f"Davis dataset not found at {davis_path}")
        
        # Load KIBA data
        if Path(kiba_path).exists():
            with open(kiba_path, 'rb') as f:
                self.kiba_data = pickle.load(f)
            logger.info(f"Loaded KIBA dataset with {len(self.kiba_data['interaction_pairs'])} interactions")
        else:
            logger.warning(f"KIBA dataset not found at {kiba_path}")
    
    def analyze_overlap(self) -> Dict[str, Any]:
        """
        Analyze overlap between Davis and KIBA datasets.
        
        Returns:
            Dictionary containing overlap analysis results
        """
        logger.info("Analyzing dataset overlap...")
        
        if not self.davis_data or not self.kiba_data:
            logger.error("Both datasets must be loaded for overlap analysis")
            return {}
        
        # Extract SMILES and sequences
        davis_smiles = set()
        davis_sequences = set()
        
        for pair in self.davis_data['interaction_pairs']:
            davis_smiles.add(pair['smiles'])
            davis_sequences.add(pair['protein_sequence'])
        
        kiba_smiles = set()
        kiba_sequences = set()
        
        for pair in self.kiba_data['interaction_pairs']:
            kiba_smiles.add(pair['smiles'])
            kiba_sequences.add(pair['protein_sequence'])
        
        # Calculate overlaps
        smiles_overlap = davis_smiles.intersection(kiba_smiles)
        sequence_overlap = davis_sequences.intersection(kiba_sequences)
        
        # Find overlapping interactions
        davis_pairs = set()
        for pair in self.davis_data['interaction_pairs']:
            davis_pairs.add((pair['smiles'], pair['protein_sequence']))
        
        kiba_pairs = set()
        for pair in self.kiba_data['interaction_pairs']:
            kiba_pairs.add((pair['smiles'], pair['protein_sequence']))
        
        interaction_overlap = davis_pairs.intersection(kiba_pairs)
        
        overlap_stats = {
            'davis_unique_smiles': len(davis_smiles),
            'kiba_unique_smiles': len(kiba_smiles),
            'overlapping_smiles': len(smiles_overlap),
            'davis_unique_sequences': len(davis_sequences),
            'kiba_unique_sequences': len(kiba_sequences),
            'overlapping_sequences': len(sequence_overlap),
            'davis_interactions': len(davis_pairs),
            'kiba_interactions': len(kiba_pairs),
            'overlapping_interactions': len(interaction_overlap),
            'overlap_smiles_list': list(smiles_overlap),
            'overlap_sequences_list': list(sequence_overlap),
            'overlap_interactions_list': list(interaction_overlap)
        }
        
        # Calculate percentages
        overlap_stats['smiles_overlap_pct'] = len(smiles_overlap) / len(davis_smiles.union(kiba_smiles)) * 100
        overlap_stats['sequence_overlap_pct'] = len(sequence_overlap) / len(davis_sequences.union(kiba_sequences)) * 100
        overlap_stats['interaction_overlap_pct'] = len(interaction_overlap) / len(davis_pairs.union(kiba_pairs)) * 100
        
        logger.info(f"SMILES overlap: {len(smiles_overlap)} ({overlap_stats['smiles_overlap_pct']:.1f}%)")
        logger.info(f"Sequence overlap: {len(sequence_overlap)} ({overlap_stats['sequence_overlap_pct']:.1f}%)")
        logger.info(f"Interaction overlap: {len(interaction_overlap)} ({overlap_stats['interaction_overlap_pct']:.1f}%)")
        
        self.merge_stats['overlap_analysis'] = overlap_stats
        return overlap_stats
    
    def harmonize_affinities(self, method: str = "standardize") -> None:
        """
        Harmonize affinity scales between datasets.
        
        Args:
            method: Harmonization method ('standardize', 'normalize', 'none')
        """
        logger.info(f"Harmonizing affinities using method: {method}")
        
        if not self.davis_data or not self.kiba_data:
            logger.error("Both datasets must be loaded for affinity harmonization")
            return
        
        # Extract affinity values
        davis_affinities = np.array([pair['affinity'] for pair in self.davis_data['interaction_pairs']])
        kiba_affinities = np.array([pair['affinity'] for pair in self.kiba_data['interaction_pairs']])
        
        logger.info(f"Davis affinities - Mean: {np.mean(davis_affinities):.3f}, Std: {np.std(davis_affinities):.3f}")
        logger.info(f"KIBA affinities - Mean: {np.mean(kiba_affinities):.3f}, Std: {np.std(kiba_affinities):.3f}")
        
        if method == "standardize":
            # Standardize both datasets to have mean=0, std=1
            davis_scaler = StandardScaler()
            kiba_scaler = StandardScaler()
            
            davis_affinities_norm = davis_scaler.fit_transform(davis_affinities.reshape(-1, 1)).ravel()
            kiba_affinities_norm = kiba_scaler.fit_transform(kiba_affinities.reshape(-1, 1)).ravel()
            
            # Store scalers for inverse transformation
            self.merge_stats['davis_scaler'] = davis_scaler
            self.merge_stats['kiba_scaler'] = kiba_scaler
            
        elif method == "normalize":
            # Normalize both datasets to [0, 1] range
            davis_scaler = MinMaxScaler()
            kiba_scaler = MinMaxScaler()
            
            davis_affinities_norm = davis_scaler.fit_transform(davis_affinities.reshape(-1, 1)).ravel()
            kiba_affinities_norm = kiba_scaler.fit_transform(kiba_affinities.reshape(-1, 1)).ravel()
            
            # Store scalers for inverse transformation
            self.merge_stats['davis_scaler'] = davis_scaler
            self.merge_stats['kiba_scaler'] = kiba_scaler
            
        elif method == "none":
            # No harmonization
            davis_affinities_norm = davis_affinities
            kiba_affinities_norm = kiba_affinities
            
        else:
            raise ValueError(f"Unknown harmonization method: {method}")
        
        # Update affinity values in the datasets
        for i, pair in enumerate(self.davis_data['interaction_pairs']):
            pair['affinity_original'] = pair['affinity']
            pair['affinity'] = float(davis_affinities_norm[i])
        
        for i, pair in enumerate(self.kiba_data['interaction_pairs']):
            pair['affinity_original'] = pair['affinity']
            pair['affinity'] = float(kiba_affinities_norm[i])
        
        logger.info("Affinity harmonization completed")
        
        # Update statistics
        self.merge_stats['harmonization_method'] = method
        self.merge_stats['davis_affinities_harmonized'] = {
            'mean': float(np.mean(davis_affinities_norm)),
            'std': float(np.std(davis_affinities_norm)),
            'min': float(np.min(davis_affinities_norm)),
            'max': float(np.max(davis_affinities_norm))
        }
        self.merge_stats['kiba_affinities_harmonized'] = {
            'mean': float(np.mean(kiba_affinities_norm)),
            'std': float(np.std(kiba_affinities_norm)),
            'min': float(np.min(kiba_affinities_norm)),
            'max': float(np.max(kiba_affinities_norm))
        }
    
    def create_unified_dataset(self, remove_duplicates: bool = True) -> Dict[str, Any]:
        """
        Create a unified dataset from Davis and KIBA data.
        
        Args:
            remove_duplicates: Whether to remove duplicate interactions
            
        Returns:
            Unified dataset dictionary
        """
        logger.info("Creating unified dataset...")
        
        if not self.davis_data or not self.kiba_data:
            logger.error("Both datasets must be loaded for merging")
            return {}
        
        # Combine interaction pairs
        all_interactions = []
        
        # Add Davis interactions
        for pair in self.davis_data['interaction_pairs']:
            pair_copy = pair.copy()
            pair_copy['dataset'] = 'davis'
            all_interactions.append(pair_copy)
        
        # Add KIBA interactions
        for pair in self.kiba_data['interaction_pairs']:
            pair_copy = pair.copy()
            pair_copy['dataset'] = 'kiba'
            all_interactions.append(pair_copy)
        
        logger.info(f"Combined {len(all_interactions)} interactions from both datasets")
        
        # Remove duplicates if requested
        if remove_duplicates:
            seen_pairs = set()
            unique_interactions = []
            duplicate_count = 0
            
            for interaction in all_interactions:
                pair_key = (interaction['smiles'], interaction['protein_sequence'])
                
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    unique_interactions.append(interaction)
                else:
                    duplicate_count += 1
            
            all_interactions = unique_interactions
            logger.info(f"Removed {duplicate_count} duplicate interactions")
        
        # Combine drug and protein data
        all_drugs = []
        all_proteins = []
        
        # Add unique drugs
        seen_smiles = set()
        drug_id_counter = 0
        
        for drug_data in self.davis_data['drugs_data'] + self.kiba_data['drugs_data']:
            smiles = drug_data.get('smiles', '')
            if smiles and smiles not in seen_smiles:
                drug_copy = drug_data.copy()
                drug_copy['unified_drug_id'] = drug_id_counter
                all_drugs.append(drug_copy)
                seen_smiles.add(smiles)
                drug_id_counter += 1
        
        # Add unique proteins
        seen_sequences = set()
        protein_id_counter = 0
        
        for protein_data in self.davis_data['proteins_data'] + self.kiba_data['proteins_data']:
            sequence = protein_data.get('sequence', '')
            if sequence and sequence not in seen_sequences:
                protein_copy = protein_data.copy()
                protein_copy['unified_protein_id'] = protein_id_counter
                all_proteins.append(protein_copy)
                seen_sequences.add(sequence)
                protein_id_counter += 1
        
        # Create unified dataset
        unified_data = {
            'dataset_name': 'davis_kiba_merged',
            'interaction_pairs': all_interactions,
            'drugs_data': all_drugs,
            'proteins_data': all_proteins,
            'source_datasets': ['davis', 'kiba'],
            'merge_statistics': self.merge_stats,
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Update statistics
        self.merge_stats['unified_interactions'] = len(all_interactions)
        self.merge_stats['unified_drugs'] = len(all_drugs)
        self.merge_stats['unified_proteins'] = len(all_proteins)
        
        logger.info(f"Created unified dataset with:")
        logger.info(f"  - {len(all_interactions)} interactions")
        logger.info(f"  - {len(all_drugs)} unique drugs")
        logger.info(f"  - {len(all_proteins)} unique proteins")
        
        self.merged_data = unified_data
        return unified_data
    
    def create_cross_dataset_splits(self) -> Dict[str, Any]:
        """
        Create splits for cross-dataset evaluation.
        
        Returns:
            Dictionary containing cross-dataset split information
        """
        logger.info("Creating cross-dataset evaluation splits...")
        
        if not self.davis_data or not self.kiba_data:
            logger.error("Both datasets must be loaded for cross-dataset splits")
            return {}
        
        # Extract interaction pairs by dataset
        davis_interactions = [(i, pair) for i, pair in enumerate(self.davis_data['interaction_pairs'])]
        kiba_interactions = [(i, pair) for i, pair in enumerate(self.kiba_data['interaction_pairs'])]
        
        # Create different cross-dataset evaluation scenarios
        cross_splits = {
            'davis_train_kiba_test': {
                'train_dataset': 'davis',
                'test_dataset': 'kiba',
                'train_indices': list(range(len(davis_interactions))),
                'test_indices': list(range(len(kiba_interactions))),
                'description': 'Train on Davis, test on KIBA'
            },
            'kiba_train_davis_test': {
                'train_dataset': 'kiba',
                'test_dataset': 'davis',
                'train_indices': list(range(len(kiba_interactions))),
                'test_indices': list(range(len(davis_interactions))),
                'description': 'Train on KIBA, test on Davis'
            }
        }
        
        # Find overlapping interactions for fair evaluation
        davis_pairs = {(pair['smiles'], pair['protein_sequence']): i 
                      for i, pair in enumerate(self.davis_data['interaction_pairs'])}
        kiba_pairs = {(pair['smiles'], pair['protein_sequence']): i 
                     for i, pair in enumerate(self.kiba_data['interaction_pairs'])}
        
        overlapping_pairs = set(davis_pairs.keys()).intersection(set(kiba_pairs.keys()))
        
        if overlapping_pairs:
            davis_overlap_indices = [davis_pairs[pair] for pair in overlapping_pairs]
            kiba_overlap_indices = [kiba_pairs[pair] for pair in overlapping_pairs]
            
            cross_splits['overlapping_pairs'] = {
                'davis_indices': davis_overlap_indices,
                'kiba_indices': kiba_overlap_indices,
                'num_pairs': len(overlapping_pairs),
                'description': 'Overlapping drug-protein pairs between datasets'
            }
        
        logger.info(f"Created cross-dataset splits with {len(overlapping_pairs)} overlapping pairs")
        
        return cross_splits
    
    def merge_datasets(self, harmonization_method: str = "standardize",
                      remove_duplicates: bool = True) -> Dict[str, Any]:
        """
        Complete dataset merging pipeline.
        
        Args:
            harmonization_method: Method for affinity harmonization
            remove_duplicates: Whether to remove duplicate interactions
            
        Returns:
            Merged dataset dictionary
        """
        logger.info("Starting dataset merging pipeline...")
        
        try:
            # Analyze overlap between datasets
            overlap_stats = self.analyze_overlap()
            
            # Harmonize affinity scales
            self.harmonize_affinities(method=harmonization_method)
            
            # Create unified dataset
            unified_data = self.create_unified_dataset(remove_duplicates=remove_duplicates)
            
            # Create cross-dataset splits
            cross_splits = self.create_cross_dataset_splits()
            unified_data['cross_dataset_splits'] = cross_splits
            
            # Save merged dataset
            self.save_merged_data()
            
            logger.info("Dataset merging completed successfully!")
            return unified_data
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise
    
    def save_merged_data(self, output_path: str = "data/processed/combined_data.pkl") -> None:
        """
        Save merged dataset.
        
        Args:
            output_path: Path to save merged data
        """
        if self.merged_data is None:
            logger.error("No merged data to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.merged_data, f)
        
        logger.info(f"Saved merged dataset to {output_path}")
    
    def load_merged_data(self, input_path: str = "data/processed/combined_data.pkl") -> Dict[str, Any]:
        """
        Load merged dataset.
        
        Args:
            input_path: Path to merged data file
            
        Returns:
            Merged dataset dictionary
        """
        with open(input_path, 'rb') as f:
            self.merged_data = pickle.load(f)
        
        logger.info(f"Loaded merged dataset from {input_path}")
        return self.merged_data
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get merge statistics."""
        return self.merge_stats.copy()
    
    def print_merge_summary(self) -> None:
        """Print merge summary."""
        print("\n" + "="*60)
        print("DATASET MERGE SUMMARY")
        print("="*60)
        
        if self.merge_stats:
            if 'overlap_analysis' in self.merge_stats:
                overlap = self.merge_stats['overlap_analysis']
                print(f"Dataset Overlap:")
                print(f"  - Overlapping SMILES: {overlap['overlapping_smiles']} ({overlap['smiles_overlap_pct']:.1f}%)")
                print(f"  - Overlapping sequences: {overlap['overlapping_sequences']} ({overlap['sequence_overlap_pct']:.1f}%)")
                print(f"  - Overlapping interactions: {overlap['overlapping_interactions']} ({overlap['interaction_overlap_pct']:.1f}%)")
            
            print(f"\nMerged Dataset:")
            print(f"  - Total interactions: {self.merge_stats.get('unified_interactions', 'N/A')}")
            print(f"  - Unique drugs: {self.merge_stats.get('unified_drugs', 'N/A')}")
            print(f"  - Unique proteins: {self.merge_stats.get('unified_proteins', 'N/A')}")
            
            if 'harmonization_method' in self.merge_stats:
                print(f"\nAffinity Harmonization: {self.merge_stats['harmonization_method']}")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize merger and merge datasets
    merger = DatasetMerger()
    
    # Load datasets (assuming they exist)
    try:
        merger.load_datasets()
        merged_data = merger.merge_datasets()
        merger.print_merge_summary()
    except Exception as e:
        logger.error(f"Error in example usage: {e}")