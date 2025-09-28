"""
Data Splitter
Creates train/validation/test splits for machine learning models with various strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from collections import defaultdict
import random

# Setup logging
logger = logging.getLogger(__name__)

class DataSplitter:
    """
    Creates various data splits for training and evaluation.
    
    Supports:
    - Random splits
    - Scaffold splits (based on molecular scaffolds)
    - Temporal splits (if timestamp information available)
    - Cross-validation folds
    - Cold-start scenarios (unseen drugs/proteins)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        random.seed(random_state)
        
        # Split configurations
        self.split_configs = {}
        self.splits_data = {}
    
    def random_split(self, data: List[Dict[str, Any]], 
                    test_size: float = 0.2, 
                    val_size: float = 0.15,
                    stratify_column: Optional[str] = None) -> Dict[str, List[int]]:
        """
        Create random train/validation/test splits.
        
        Args:
            data: List of data samples
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining data)
            stratify_column: Column name for stratified splitting
            
        Returns:
            Dictionary with train/val/test indices
        """
        logger.info(f"Creating random split (test: {test_size}, val: {val_size})")
        
        indices = list(range(len(data)))
        
        # Stratification setup
        stratify_values = None
        if stratify_column:
            try:
                values = [sample[stratify_column] for sample in data]
                # Bin continuous values for stratification
                if isinstance(values[0], (int, float)):
                    # Create bins for continuous values
                    bins = np.percentile(values, [0, 25, 50, 75, 100])
                    stratify_values = np.digitize(values, bins)
                else:
                    stratify_values = values
            except KeyError:
                logger.warning(f"Stratify column '{stratify_column}' not found")
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_values
        )
        
        # Second split: separate validation from training
        if val_size > 0:
            # Adjust stratify values for remaining data
            remaining_stratify = None
            if stratify_values is not None:
                remaining_stratify = [stratify_values[i] for i in train_val_indices]
            
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_size / (1 - test_size),  # Adjust for remaining data
                random_state=self.random_state,
                stratify=remaining_stratify
            )
        else:
            train_indices = train_val_indices
            val_indices = []
        
        split_info = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'split_type': 'random',
            'test_size': test_size,
            'val_size': val_size,
            'stratify_column': stratify_column
        }
        
        logger.info(f"Random split created: train={len(train_indices)}, "
                   f"val={len(val_indices)}, test={len(test_indices)}")
        
        return split_info
    
    def scaffold_split(self, data: List[Dict[str, Any]], 
                      smiles_column: str = 'smiles',
                      test_size: float = 0.2, 
                      val_size: float = 0.15) -> Dict[str, List[int]]:
        """
        Create scaffold-based splits to avoid data leakage.
        
        Args:
            data: List of data samples
            smiles_column: Column name containing SMILES strings
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            Dictionary with train/val/test indices
        """
        logger.info(f"Creating scaffold split (test: {test_size}, val: {val_size})")
        
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            # Extract scaffolds for each molecule
            scaffolds = defaultdict(list)
            
            for idx, sample in enumerate(data):
                smiles = sample.get(smiles_column, '')
                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                            scaffolds[scaffold].append(idx)
                        else:
                            # Handle invalid SMILES
                            scaffolds['invalid'].append(idx)
                    except Exception as e:
                        logger.warning(f"Error processing SMILES {smiles}: {e}")
                        scaffolds['invalid'].append(idx)
                else:
                    scaffolds['empty'].append(idx)
            
            # Sort scaffolds by size (largest first) for better balance
            scaffold_groups = list(scaffolds.values())
            scaffold_groups.sort(key=len, reverse=True)
            
            # Distribute scaffolds to splits
            train_indices = []
            val_indices = []
            test_indices = []
            
            train_target = len(data) * (1 - test_size - val_size)
            val_target = len(data) * val_size
            
            for scaffold_indices in scaffold_groups:
                # Assign to the split that needs more data
                train_size = len(train_indices)
                val_size_current = len(val_indices)
                test_size_current = len(test_indices)
                
                if test_size_current < len(data) * test_size:
                    test_indices.extend(scaffold_indices)
                elif val_size_current < val_target:
                    val_indices.extend(scaffold_indices)
                else:
                    train_indices.extend(scaffold_indices)
            
        except ImportError:
            logger.warning("RDKit not available for scaffold splitting, using random split")
            return self.random_split(data, test_size=test_size, val_size=val_size)
        
        split_info = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'split_type': 'scaffold',
            'test_size': test_size,
            'val_size': val_size,
            'num_scaffolds': len(scaffolds)
        }
        
        logger.info(f"Scaffold split created: train={len(train_indices)}, "
                   f"val={len(val_indices)}, test={len(test_indices)}")
        logger.info(f"Number of unique scaffolds: {len(scaffolds)}")
        
        return split_info
    
    def cold_start_split(self, data: List[Dict[str, Any]], 
                        split_by: str = 'drug',  # 'drug', 'protein', or 'both'
                        test_size: float = 0.2,
                        val_size: float = 0.15) -> Dict[str, List[int]]:
        """
        Create cold-start splits where test set contains unseen drugs/proteins.
        
        Args:
            data: List of data samples
            split_by: What to split by ('drug', 'protein', or 'both')
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            Dictionary with train/val/test indices
        """
        logger.info(f"Creating cold-start split by {split_by} (test: {test_size}, val: {val_size})")
        
        if split_by == 'drug':
            # Group by unique drugs (SMILES)
            entity_groups = defaultdict(list)
            for idx, sample in enumerate(data):
                smiles = sample.get('smiles', '')
                entity_groups[smiles].append(idx)
        
        elif split_by == 'protein':
            # Group by unique proteins (sequences)
            entity_groups = defaultdict(list)
            for idx, sample in enumerate(data):
                sequence = sample.get('protein_sequence', '')
                entity_groups[sequence].append(idx)
        
        elif split_by == 'both':
            # This is more complex - need to ensure test set has both unseen drugs AND proteins
            return self._cold_start_both_split(data, test_size, val_size)
        
        else:
            raise ValueError(f"Invalid split_by value: {split_by}. Must be 'drug', 'protein', or 'both'")
        
        # Sort entity groups by size for better balance
        entities = list(entity_groups.keys())
        entity_sizes = [(entity, len(entity_groups[entity])) for entity in entities]
        entity_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute entities to splits
        train_indices = []
        val_indices = []
        test_indices = []
        
        total_samples = len(data)
        test_target = total_samples * test_size
        val_target = total_samples * val_size
        
        for entity, size in entity_sizes:
            indices = entity_groups[entity]
            
            # Assign to the split that needs more data
            current_test_size = len(test_indices)
            current_val_size = len(val_indices)
            
            if current_test_size < test_target:
                test_indices.extend(indices)
            elif current_val_size < val_target:
                val_indices.extend(indices)
            else:
                train_indices.extend(indices)
        
        split_info = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'split_type': f'cold_start_{split_by}',
            'test_size': test_size,
            'val_size': val_size,
            'num_unique_entities': len(entities)
        }
        
        logger.info(f"Cold-start {split_by} split created: train={len(train_indices)}, "
                   f"val={len(val_indices)}, test={len(test_indices)}")
        logger.info(f"Number of unique {split_by}s: {len(entities)}")
        
        return split_info
    
    def _cold_start_both_split(self, data: List[Dict[str, Any]], 
                              test_size: float, val_size: float) -> Dict[str, List[int]]:
        """Create cold-start split with both unseen drugs and proteins."""
        logger.info("Creating cold-start split with both unseen drugs and proteins")
        
        # First, identify all unique drugs and proteins
        unique_drugs = set()
        unique_proteins = set()
        
        for sample in data:
            unique_drugs.add(sample.get('smiles', ''))
            unique_proteins.add(sample.get('protein_sequence', ''))
        
        # Randomly assign some drugs and proteins to test set
        drugs_list = list(unique_drugs)
        proteins_list = list(unique_proteins)
        
        test_drugs = set(self.rng.choice(drugs_list, size=int(len(drugs_list) * test_size), replace=False))
        test_proteins = set(self.rng.choice(proteins_list, size=int(len(proteins_list) * test_size), replace=False))
        
        # Assign remaining drugs and proteins to val set
        remaining_drugs = unique_drugs - test_drugs
        remaining_proteins = unique_proteins - test_proteins
        
        val_drugs = set(self.rng.choice(list(remaining_drugs), 
                                       size=int(len(remaining_drugs) * val_size / (1 - test_size)), 
                                       replace=False))
        val_proteins = set(self.rng.choice(list(remaining_proteins), 
                                          size=int(len(remaining_proteins) * val_size / (1 - test_size)), 
                                          replace=False))
        
        # Assign samples to splits based on drug and protein membership
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx, sample in enumerate(data):
            drug = sample.get('smiles', '')
            protein = sample.get('protein_sequence', '')
            
            # Test set: samples with test drugs OR test proteins
            if drug in test_drugs or protein in test_proteins:
                test_indices.append(idx)
            # Val set: samples with val drugs OR val proteins (but not test)
            elif drug in val_drugs or protein in val_proteins:
                val_indices.append(idx)
            # Train set: everything else
            else:
                train_indices.append(idx)
        
        split_info = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'split_type': 'cold_start_both',
            'test_size': test_size,
            'val_size': val_size,
            'test_drugs': list(test_drugs),
            'test_proteins': list(test_proteins),
            'val_drugs': list(val_drugs),
            'val_proteins': list(val_proteins)
        }
        
        return split_info
    
    def create_cv_folds(self, data: List[Dict[str, Any]], 
                       n_folds: int = 5,
                       split_type: str = 'random',
                       stratify_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Create cross-validation folds.
        
        Args:
            data: List of data samples
            n_folds: Number of CV folds
            split_type: Type of CV ('random', 'stratified')
            stratify_column: Column for stratified CV
            
        Returns:
            Dictionary with CV fold information
        """
        logger.info(f"Creating {n_folds}-fold cross-validation ({split_type})")
        
        indices = list(range(len(data)))
        
        if split_type == 'stratified' and stratify_column:
            try:
                values = [sample[stratify_column] for sample in data]
                # Bin continuous values for stratification
                if isinstance(values[0], (int, float)):
                    bins = np.percentile(values, [0, 25, 50, 75, 100])
                    stratify_values = np.digitize(values, bins)
                else:
                    stratify_values = values
                
                kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                cv_splits = list(kfold.split(indices, stratify_values))
                
            except Exception as e:
                logger.warning(f"Stratified CV failed: {e}, using random CV")
                kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                cv_splits = list(kfold.split(indices))
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            cv_splits = list(kfold.split(indices))
        
        # Format CV splits
        cv_folds = {}
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
            cv_folds[f'fold_{fold_idx}'] = {
                'train': train_indices.tolist(),
                'val': val_indices.tolist(),
                'fold_number': fold_idx
            }
        
        cv_info = {
            'folds': cv_folds,
            'n_folds': n_folds,
            'split_type': split_type,
            'stratify_column': stratify_column,
            'total_samples': len(data)
        }
        
        logger.info(f"Created {n_folds} CV folds")
        
        return cv_info
    
    def create_multiple_splits(self, data: List[Dict[str, Any]], 
                              split_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create multiple splits with different strategies.
        
        Args:
            data: List of data samples
            split_configs: Dictionary of split configurations
            
        Returns:
            Dictionary containing all splits
        """
        logger.info(f"Creating {len(split_configs)} different splits")
        
        all_splits = {}
        
        for split_name, config in split_configs.items():
            logger.info(f"Creating split: {split_name}")
            
            split_type = config.get('type', 'random')
            
            try:
                if split_type == 'random':
                    split_info = self.random_split(
                        data,
                        test_size=config.get('test_size', 0.2),
                        val_size=config.get('val_size', 0.15),
                        stratify_column=config.get('stratify_column')
                    )
                
                elif split_type == 'scaffold':
                    split_info = self.scaffold_split(
                        data,
                        smiles_column=config.get('smiles_column', 'smiles'),
                        test_size=config.get('test_size', 0.2),
                        val_size=config.get('val_size', 0.15)
                    )
                
                elif split_type.startswith('cold_start'):
                    split_by = config.get('split_by', 'drug')
                    split_info = self.cold_start_split(
                        data,
                        split_by=split_by,
                        test_size=config.get('test_size', 0.2),
                        val_size=config.get('val_size', 0.15)
                    )
                
                elif split_type == 'cv':
                    split_info = self.create_cv_folds(
                        data,
                        n_folds=config.get('n_folds', 5),
                        split_type=config.get('cv_type', 'random'),
                        stratify_column=config.get('stratify_column')
                    )
                
                else:
                    logger.warning(f"Unknown split type: {split_type}")
                    continue
                
                all_splits[split_name] = split_info
                
            except Exception as e:
                logger.error(f"Error creating split {split_name}: {e}")
                continue
        
        logger.info(f"Successfully created {len(all_splits)} splits")
        
        return all_splits
    
    def save_splits(self, splits_data: Dict[str, Any], 
                   output_path: str) -> None:
        """
        Save splits to file.
        
        Args:
            splits_data: Dictionary containing split information
            output_path: Path to save the splits
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        splits_serializable = convert_arrays(splits_data)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(splits_serializable, f, indent=2)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(splits_serializable, f)
        
        logger.info(f"Saved splits to {output_path}")
    
    def load_splits(self, input_path: str) -> Dict[str, Any]:
        """
        Load splits from file.
        
        Args:
            input_path: Path to the splits file
            
        Returns:
            Dictionary containing split information
        """
        input_path = Path(input_path)
        
        if input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                splits_data = json.load(f)
        else:
            with open(input_path, 'rb') as f:
                splits_data = pickle.load(f)
        
        logger.info(f"Loaded splits from {input_path}")
        return splits_data
    
    def print_split_summary(self, splits_data: Dict[str, Any]) -> None:
        """Print summary of splits."""
        print("\n" + "="*60)
        print("DATA SPLITS SUMMARY")
        print("="*60)
        
        for split_name, split_info in splits_data.items():
            print(f"\n{split_name.upper()}:")
            
            if 'folds' in split_info:
                # Cross-validation splits
                print(f"  Type: {split_info.get('split_type', 'unknown')} CV")
                print(f"  Number of folds: {split_info.get('n_folds', 'unknown')}")
                print(f"  Total samples: {split_info.get('total_samples', 'unknown')}")
            else:
                # Regular train/val/test splits
                print(f"  Type: {split_info.get('split_type', 'unknown')}")
                print(f"  Train: {len(split_info.get('train', []))}")
                print(f"  Validation: {len(split_info.get('val', []))}")
                print(f"  Test: {len(split_info.get('test', []))}")
                
                total = len(split_info.get('train', [])) + len(split_info.get('val', [])) + len(split_info.get('test', []))
                if total > 0:
                    train_pct = len(split_info.get('train', [])) / total * 100
                    val_pct = len(split_info.get('val', [])) / total * 100
                    test_pct = len(split_info.get('test', [])) / total * 100
                    print(f"  Percentages: {train_pct:.1f}% / {val_pct:.1f}% / {test_pct:.1f}%")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example data
    example_data = [
        {'smiles': 'CCO', 'protein_sequence': 'ACDEFGH', 'affinity': 5.2},
        {'smiles': 'CCC', 'protein_sequence': 'ACDEFGH', 'affinity': 6.1},
        {'smiles': 'CCO', 'protein_sequence': 'HIJKLM', 'affinity': 4.8},
        # ... more data
    ]
    
    # Initialize splitter
    splitter = DataSplitter(random_state=42)
    
    # Create different types of splits
    split_configs = {
        'random_split': {
            'type': 'random',
            'test_size': 0.2,
            'val_size': 0.15
        },
        'scaffold_split': {
            'type': 'scaffold',
            'test_size': 0.2,
            'val_size': 0.15
        },
        'cold_drug_split': {
            'type': 'cold_start',
            'split_by': 'drug',
            'test_size': 0.2,
            'val_size': 0.15
        },
        'cv_5fold': {
            'type': 'cv',
            'n_folds': 5,
            'cv_type': 'random'
        }
    }
    
    # Create splits
    all_splits = splitter.create_multiple_splits(example_data, split_configs)
    
    # Print summary
    splitter.print_split_summary(all_splits)