"""
Data Validator
Validates data quality and consistency for drug-target interaction datasets.
"""

import logging
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
import warnings

# Setup logging
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates data quality and consistency for drug-target datasets.
    
    Performs:
    - SMILES validation
    - Protein sequence validation
    - Affinity value validation
    - Statistical consistency checks
    - Duplicate detection
    - Missing value analysis
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize data validator.
        
        Args:
            strict_mode: Whether to use strict validation rules
        """
        self.strict_mode = strict_mode
        
        # Validation results
        self.validation_results = {}
        self.error_log = []
        self.warning_log = []
        
        # Valid amino acids
        self.valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        
        # Statistics
        self.stats = {}
    
    def validate_smiles(self, smiles_list: List[str], 
                       sample_ids: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Validate SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            sample_ids: Optional list of sample identifiers
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {len(smiles_list)} SMILES strings...")
        
        if sample_ids is None:
            sample_ids = list(range(len(smiles_list)))
        
        validation_results = {
            'total_smiles': len(smiles_list),
            'valid_smiles': 0,
            'invalid_smiles': 0,
            'empty_smiles': 0,
            'rdkit_parseable': 0,
            'rdkit_unparseable': 0,
            'invalid_entries': [],
            'duplicate_smiles': {},
            'smiles_statistics': {}
        }
        
        # Check for empty/null SMILES
        empty_count = 0
        for i, smiles in enumerate(smiles_list):
            if not smiles or (isinstance(smiles, str) and not smiles.strip()):
                empty_count += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'smiles': smiles,
                    'error': 'Empty SMILES'
                })
        
        validation_results['empty_smiles'] = empty_count
        
        # Basic SMILES format validation
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=+-#@/\\\\.')
        
        basic_valid = []
        for i, smiles in enumerate(smiles_list):
            if isinstance(smiles, str) and smiles.strip():
                smiles_clean = smiles.strip()
                # Check for valid characters
                if all(c in valid_chars for c in smiles_clean):
                    basic_valid.append((i, smiles_clean))
                else:
                    validation_results['invalid_entries'].append({
                        'index': i,
                        'sample_id': sample_ids[i],
                        'smiles': smiles,
                        'error': 'Invalid characters in SMILES'
                    })
        
        validation_results['valid_smiles'] = len(basic_valid)
        validation_results['invalid_smiles'] = len(smiles_list) - len(basic_valid) - empty_count
        
        # RDKit validation (if available)
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            rdkit_valid = []
            molecular_stats = {
                'molecular_weights': [],
                'num_atoms': [],
                'num_bonds': [],
                'num_rings': []
            }
            
            for i, smiles_clean in basic_valid:
                try:
                    mol = Chem.MolFromSmiles(smiles_clean)
                    if mol is not None:
                        rdkit_valid.append((i, smiles_clean, mol))
                        
                        # Collect molecular statistics
                        molecular_stats['molecular_weights'].append(Descriptors.MolWt(mol))
                        molecular_stats['num_atoms'].append(mol.GetNumAtoms())
                        molecular_stats['num_bonds'].append(mol.GetNumBonds())
                        molecular_stats['num_rings'].append(Descriptors.RingCount(mol))
                        
                    else:
                        validation_results['invalid_entries'].append({
                            'index': i,
                            'sample_id': sample_ids[i],
                            'smiles': smiles_clean,
                            'error': 'RDKit parsing failed'
                        })
                except Exception as e:
                    validation_results['invalid_entries'].append({
                        'index': i,
                        'sample_id': sample_ids[i],
                        'smiles': smiles_clean,
                        'error': f'RDKit error: {str(e)}'
                    })
            
            validation_results['rdkit_parseable'] = len(rdkit_valid)
            validation_results['rdkit_unparseable'] = len(basic_valid) - len(rdkit_valid)
            
            # Calculate molecular statistics
            if molecular_stats['molecular_weights']:
                validation_results['smiles_statistics'] = {
                    'mol_weight': {
                        'mean': np.mean(molecular_stats['molecular_weights']),
                        'std': np.std(molecular_stats['molecular_weights']),
                        'min': np.min(molecular_stats['molecular_weights']),
                        'max': np.max(molecular_stats['molecular_weights'])
                    },
                    'num_atoms': {
                        'mean': np.mean(molecular_stats['num_atoms']),
                        'std': np.std(molecular_stats['num_atoms']),
                        'min': np.min(molecular_stats['num_atoms']),
                        'max': np.max(molecular_stats['num_atoms'])
                    },
                    'num_rings': {
                        'mean': np.mean(molecular_stats['num_rings']),
                        'std': np.std(molecular_stats['num_rings']),
                        'min': np.min(molecular_stats['num_rings']),
                        'max': np.max(molecular_stats['num_rings'])
                    }
                }
            
        except ImportError:
            logger.warning("RDKit not available for SMILES validation")
            validation_results['rdkit_parseable'] = 0
            validation_results['rdkit_unparseable'] = 0
        
        # Check for duplicates
        smiles_counts = Counter([s.strip() for s in smiles_list if isinstance(s, str) and s.strip()])
        duplicates = {smiles: count for smiles, count in smiles_counts.items() if count > 1}
        validation_results['duplicate_smiles'] = duplicates
        
        # Log validation results
        logger.info(f"SMILES validation completed:")
        logger.info(f"  - Valid: {validation_results['valid_smiles']}")
        logger.info(f"  - Invalid: {validation_results['invalid_smiles']}")
        logger.info(f"  - Empty: {validation_results['empty_smiles']}")
        logger.info(f"  - RDKit parseable: {validation_results['rdkit_parseable']}")
        logger.info(f"  - Duplicates: {len(duplicates)}")
        
        return validation_results
    
    def validate_protein_sequences(self, sequences: List[str],
                                  sample_ids: Optional[List[Any]] = None,
                                  min_length: int = 10,
                                  max_length: int = 5000) -> Dict[str, Any]:
        """
        Validate protein sequences.
        
        Args:
            sequences: List of protein sequences
            sample_ids: Optional list of sample identifiers
            min_length: Minimum valid sequence length
            max_length: Maximum valid sequence length
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {len(sequences)} protein sequences...")
        
        if sample_ids is None:
            sample_ids = list(range(len(sequences)))
        
        validation_results = {
            'total_sequences': len(sequences),
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'empty_sequences': 0,
            'too_short': 0,
            'too_long': 0,
            'invalid_amino_acids': 0,
            'invalid_entries': [],
            'duplicate_sequences': {},
            'sequence_statistics': {}
        }
        
        valid_sequences = []
        sequence_stats = {
            'lengths': [],
            'amino_acid_composition': defaultdict(int),
            'hydrophobic_content': [],
            'charged_content': []
        }
        
        hydrophobic_aa = set('AILMFPWV')
        charged_aa = set('DEKR')
        
        for i, sequence in enumerate(sequences):
            # Check for empty sequences
            if not sequence or (isinstance(sequence, str) and not sequence.strip()):
                validation_results['empty_sequences'] += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'sequence': sequence,
                    'error': 'Empty sequence'
                })
                continue
            
            seq_clean = sequence.strip().upper() if isinstance(sequence, str) else str(sequence).strip().upper()
            
            # Check sequence length
            if len(seq_clean) < min_length:
                validation_results['too_short'] += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'sequence': seq_clean,
                    'error': f'Sequence too short ({len(seq_clean)} < {min_length})'
                })
                continue
            
            if len(seq_clean) > max_length:
                validation_results['too_long'] += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'sequence': seq_clean,
                    'error': f'Sequence too long ({len(seq_clean)} > {max_length})'
                })
                continue
            
            # Check for valid amino acids
            invalid_aa = set(seq_clean) - self.valid_amino_acids
            if invalid_aa:
                validation_results['invalid_amino_acids'] += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'sequence': seq_clean,
                    'error': f'Invalid amino acids: {invalid_aa}'
                })
                continue
            
            # Valid sequence
            valid_sequences.append(seq_clean)
            
            # Collect statistics
            sequence_stats['lengths'].append(len(seq_clean))
            
            for aa in seq_clean:
                sequence_stats['amino_acid_composition'][aa] += 1
            
            hydrophobic_count = sum(1 for aa in seq_clean if aa in hydrophobic_aa)
            charged_count = sum(1 for aa in seq_clean if aa in charged_aa)
            
            sequence_stats['hydrophobic_content'].append(hydrophobic_count / len(seq_clean))
            sequence_stats['charged_content'].append(charged_count / len(seq_clean))
        
        validation_results['valid_sequences'] = len(valid_sequences)
        validation_results['invalid_sequences'] = len(sequences) - len(valid_sequences) - validation_results['empty_sequences']
        
        # Calculate sequence statistics
        if sequence_stats['lengths']:
            validation_results['sequence_statistics'] = {
                'length': {
                    'mean': np.mean(sequence_stats['lengths']),
                    'std': np.std(sequence_stats['lengths']),
                    'min': np.min(sequence_stats['lengths']),
                    'max': np.max(sequence_stats['lengths'])
                },
                'hydrophobic_content': {
                    'mean': np.mean(sequence_stats['hydrophobic_content']),
                    'std': np.std(sequence_stats['hydrophobic_content'])
                },
                'charged_content': {
                    'mean': np.mean(sequence_stats['charged_content']),
                    'std': np.std(sequence_stats['charged_content'])
                },
                'amino_acid_frequency': dict(sequence_stats['amino_acid_composition'])
            }
        
        # Check for duplicates
        sequence_counts = Counter(valid_sequences)
        duplicates = {seq: count for seq, count in sequence_counts.items() if count > 1}
        validation_results['duplicate_sequences'] = duplicates
        
        # Log validation results
        logger.info(f"Protein sequence validation completed:")
        logger.info(f"  - Valid: {validation_results['valid_sequences']}")
        logger.info(f"  - Invalid: {validation_results['invalid_sequences']}")
        logger.info(f"  - Empty: {validation_results['empty_sequences']}")
        logger.info(f"  - Too short: {validation_results['too_short']}")
        logger.info(f"  - Too long: {validation_results['too_long']}")
        logger.info(f"  - Invalid amino acids: {validation_results['invalid_amino_acids']}")
        logger.info(f"  - Duplicates: {len(duplicates)}")
        
        return validation_results
    
    def validate_affinities(self, affinities: List[Union[float, int]],
                           sample_ids: Optional[List[Any]] = None,
                           expected_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Validate binding affinity values.
        
        Args:
            affinities: List of affinity values
            sample_ids: Optional list of sample identifiers
            expected_range: Expected range of valid affinity values
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {len(affinities)} affinity values...")
        
        if sample_ids is None:
            sample_ids = list(range(len(affinities)))
        
        validation_results = {
            'total_affinities': len(affinities),
            'valid_affinities': 0,
            'invalid_affinities': 0,
            'missing_values': 0,
            'out_of_range': 0,
            'negative_values': 0,
            'infinite_values': 0,
            'invalid_entries': [],
            'affinity_statistics': {}
        }
        
        valid_affinities = []
        
        for i, affinity in enumerate(affinities):
            # Check for missing values
            if affinity is None or pd.isna(affinity):
                validation_results['missing_values'] += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'affinity': affinity,
                    'error': 'Missing value'
                })
                continue
            
            # Try to convert to float
            try:
                affinity_float = float(affinity)
            except (ValueError, TypeError):
                validation_results['invalid_affinities'] += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'affinity': affinity,
                    'error': 'Cannot convert to float'
                })
                continue
            
            # Check for infinite values
            if np.isinf(affinity_float):
                validation_results['infinite_values'] += 1
                validation_results['invalid_entries'].append({
                    'index': i,
                    'sample_id': sample_ids[i],
                    'affinity': affinity_float,
                    'error': 'Infinite value'
                })
                continue
            
            # Check for negative values (might be valid depending on scale)
            if affinity_float < 0:
                validation_results['negative_values'] += 1
                if self.strict_mode:
                    validation_results['invalid_entries'].append({
                        'index': i,
                        'sample_id': sample_ids[i],
                        'affinity': affinity_float,
                        'error': 'Negative value'
                    })
                    continue
            
            # Check range if specified
            if expected_range:
                if not (expected_range[0] <= affinity_float <= expected_range[1]):
                    validation_results['out_of_range'] += 1
                    validation_results['invalid_entries'].append({
                        'index': i,
                        'sample_id': sample_ids[i],
                        'affinity': affinity_float,
                        'error': f'Out of range [{expected_range[0]}, {expected_range[1]}]'
                    })
                    continue
            
            # Valid affinity
            valid_affinities.append(affinity_float)
        
        validation_results['valid_affinities'] = len(valid_affinities)
        validation_results['invalid_affinities'] = len(affinities) - len(valid_affinities) - validation_results['missing_values']
        
        # Calculate affinity statistics
        if valid_affinities:
            validation_results['affinity_statistics'] = {
                'mean': np.mean(valid_affinities),
                'std': np.std(valid_affinities),
                'min': np.min(valid_affinities),
                'max': np.max(valid_affinities),
                'median': np.median(valid_affinities),
                'percentile_25': np.percentile(valid_affinities, 25),
                'percentile_75': np.percentile(valid_affinities, 75)
            }
        
        # Log validation results
        logger.info(f"Affinity validation completed:")
        logger.info(f"  - Valid: {validation_results['valid_affinities']}")
        logger.info(f"  - Invalid: {validation_results['invalid_affinities']}")
        logger.info(f"  - Missing: {validation_results['missing_values']}")
        logger.info(f"  - Out of range: {validation_results['out_of_range']}")
        logger.info(f"  - Negative: {validation_results['negative_values']}")
        logger.info(f"  - Infinite: {validation_results['infinite_values']}")
        
        return validation_results
    
    def validate_dataset(self, data: List[Dict[str, Any]],
                        smiles_column: str = 'smiles',
                        protein_column: str = 'protein_sequence',
                        affinity_column: str = 'affinity') -> Dict[str, Any]:
        """
        Validate complete dataset.
        
        Args:
            data: List of data samples
            smiles_column: Column name for SMILES
            protein_column: Column name for protein sequences
            affinity_column: Column name for affinity values
            
        Returns:
            Dictionary with complete validation results
        """
        logger.info(f"Validating complete dataset with {len(data)} samples")
        
        # Extract data columns
        smiles_list = []
        sequences_list = []
        affinities_list = []
        sample_ids = []
        
        for i, sample in enumerate(data):
            sample_ids.append(sample.get('drug_id', i))  # Use drug_id if available, otherwise index
            smiles_list.append(sample.get(smiles_column, ''))
            sequences_list.append(sample.get(protein_column, ''))
            affinities_list.append(sample.get(affinity_column, None))
        
        # Validate each component
        smiles_validation = self.validate_smiles(smiles_list, sample_ids)
        protein_validation = self.validate_protein_sequences(sequences_list, sample_ids)
        affinity_validation = self.validate_affinities(affinities_list, sample_ids)
        
        # Overall dataset validation
        overall_validation = {
            'total_samples': len(data),
            'completely_valid_samples': 0,
            'samples_with_errors': 0,
            'error_breakdown': defaultdict(int)
        }
        
        # Count completely valid samples
        smiles_invalid_indices = {entry['index'] for entry in smiles_validation['invalid_entries']}
        protein_invalid_indices = {entry['index'] for entry in protein_validation['invalid_entries']}
        affinity_invalid_indices = {entry['index'] for entry in affinity_validation['invalid_entries']}
        
        all_invalid_indices = smiles_invalid_indices.union(protein_invalid_indices).union(affinity_invalid_indices)
        
        overall_validation['completely_valid_samples'] = len(data) - len(all_invalid_indices)
        overall_validation['samples_with_errors'] = len(all_invalid_indices)
        
        # Error breakdown
        for entry in smiles_validation['invalid_entries']:
            overall_validation['error_breakdown']['smiles_errors'] += 1
        
        for entry in protein_validation['invalid_entries']:
            overall_validation['error_breakdown']['protein_errors'] += 1
        
        for entry in affinity_validation['invalid_entries']:
            overall_validation['error_breakdown']['affinity_errors'] += 1
        
        # Compile complete validation results
        complete_validation = {
            'dataset_info': {
                'total_samples': len(data),
                'smiles_column': smiles_column,
                'protein_column': protein_column,
                'affinity_column': affinity_column
            },
            'smiles_validation': smiles_validation,
            'protein_validation': protein_validation,
            'affinity_validation': affinity_validation,
            'overall_validation': overall_validation,
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Store results
        self.validation_results = complete_validation
        
        logger.info(f"Dataset validation completed:")
        logger.info(f"  - Total samples: {len(data)}")
        logger.info(f"  - Completely valid: {overall_validation['completely_valid_samples']}")
        logger.info(f"  - Samples with errors: {overall_validation['samples_with_errors']}")
        
        return complete_validation
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.validation_results:
            return {'error': 'No validation results available'}
        
        summary = {
            'total_samples': self.validation_results['dataset_info']['total_samples'],
            'valid_samples': self.validation_results['overall_validation']['completely_valid_samples'],
            'error_samples': self.validation_results['overall_validation']['samples_with_errors'],
            'success_rate': self.validation_results['overall_validation']['completely_valid_samples'] / 
                           self.validation_results['dataset_info']['total_samples'] * 100,
            'error_breakdown': dict(self.validation_results['overall_validation']['error_breakdown'])
        }
        
        return summary
    
    def print_validation_report(self) -> None:
        """Print detailed validation report."""
        if not self.validation_results:
            print("No validation results available.")
            return
        
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT")
        print("="*80)
        
        # Dataset info
        info = self.validation_results['dataset_info']
        print(f"Dataset: {info['total_samples']} samples")
        print(f"Columns: SMILES={info['smiles_column']}, Protein={info['protein_column']}, Affinity={info['affinity_column']}")
        
        # Overall results
        overall = self.validation_results['overall_validation']
        success_rate = overall['completely_valid_samples'] / info['total_samples'] * 100
        print(f"\nOVERALL RESULTS:")
        print(f"  - Valid samples: {overall['completely_valid_samples']} ({success_rate:.1f}%)")
        print(f"  - Error samples: {overall['samples_with_errors']}")
        
        # SMILES validation
        smiles = self.validation_results['smiles_validation']
        print(f"\nSMILES VALIDATION:")
        print(f"  - Valid: {smiles['valid_smiles']}")
        print(f"  - Invalid: {smiles['invalid_smiles']}")
        print(f"  - Empty: {smiles['empty_smiles']}")
        print(f"  - RDKit parseable: {smiles['rdkit_parseable']}")
        print(f"  - Duplicates: {len(smiles['duplicate_smiles'])}")
        
        # Protein validation
        protein = self.validation_results['protein_validation']
        print(f"\nPROTEIN VALIDATION:")
        print(f"  - Valid: {protein['valid_sequences']}")
        print(f"  - Invalid: {protein['invalid_sequences']}")
        print(f"  - Empty: {protein['empty_sequences']}")
        print(f"  - Too short: {protein['too_short']}")
        print(f"  - Too long: {protein['too_long']}")
        print(f"  - Invalid amino acids: {protein['invalid_amino_acids']}")
        print(f"  - Duplicates: {len(protein['duplicate_sequences'])}")
        
        # Affinity validation
        affinity = self.validation_results['affinity_validation']
        print(f"\nAFFINITY VALIDATION:")
        print(f"  - Valid: {affinity['valid_affinities']}")
        print(f"  - Invalid: {affinity['invalid_affinities']}")
        print(f"  - Missing: {affinity['missing_values']}")
        print(f"  - Out of range: {affinity['out_of_range']}")
        print(f"  - Negative: {affinity['negative_values']}")
        print(f"  - Infinite: {affinity['infinite_values']}")
        
        # Statistics
        if 'affinity_statistics' in affinity and affinity['affinity_statistics']:
            stats = affinity['affinity_statistics']
            print(f"\nAFFINITY STATISTICS:")
            print(f"  - Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  - Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  - Median: {stats['median']:.3f}")
        
        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example data with some errors
    example_data = [
        {'smiles': 'CCO', 'protein_sequence': 'ACDEFGH', 'affinity': 5.2},
        {'smiles': 'INVALID_SMILES', 'protein_sequence': 'ACDEFGH', 'affinity': 6.1},
        {'smiles': 'CCC', 'protein_sequence': 'INVALID123', 'affinity': 4.8},
        {'smiles': 'CCO', 'protein_sequence': 'ACDEFGHIKLMNPQRSTVWY', 'affinity': None},
        {'smiles': '', 'protein_sequence': '', 'affinity': 'invalid'},
    ]
    
    # Initialize validator
    validator = DataValidator(strict_mode=False)
    
    # Validate dataset
    results = validator.validate_dataset(example_data)
    
    # Print report
    validator.print_validation_report()
    
    # Get summary
    summary = validator.get_validation_summary()
    print(f"Validation summary: {summary}")