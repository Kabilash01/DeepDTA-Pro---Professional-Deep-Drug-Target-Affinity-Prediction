"""
Protein Feature Extraction
Processes protein sequences for neural network input including tokenization and embeddings.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
from pathlib import Path
import re

# Setup logging
logger = logging.getLogger(__name__)

class ProteinFeatureExtractor:
    """
    Extracts features from protein sequences for deep learning models.
    
    Features include:
    - Amino acid tokenization
    - Sequence encoding
    - Physicochemical properties
    - Secondary structure predictions (if available)
    """
    
    def __init__(self, max_length: int = 2000, padding_token: int = 0, unknown_token: int = 1):
        """
        Initialize protein feature extractor.
        
        Args:
            max_length: Maximum sequence length
            padding_token: Token ID for padding
            unknown_token: Token ID for unknown amino acids
        """
        self.max_length = max_length
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        
        # Standard amino acid vocabulary
        self.amino_acid_vocab = {
            'A': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16,
            'S': 17, 'T': 18, 'V': 19, 'W': 20, 'Y': 21,
            'X': 1,  # Unknown amino acid
            '<PAD>': 0,  # Padding token
        }
        
        # Reverse vocabulary for decoding
        self.vocab_to_aa = {v: k for k, v in self.amino_acid_vocab.items()}
        
        # Amino acid physicochemical properties
        self.aa_properties = self._get_aa_properties()
        
        # Cache for processed sequences
        self._sequence_cache = {}
    
    def _get_aa_properties(self) -> Dict[str, Dict[str, float]]:
        """
        Get physicochemical properties for amino acids.
        
        Returns:
            Dictionary mapping amino acids to their properties
        """
        properties = {
            'A': {'hydropathy': 1.8, 'volume': 88.6, 'polarity': 8.1, 'charge': 0},
            'C': {'hydropathy': 2.5, 'volume': 108.5, 'polarity': 5.5, 'charge': 0},
            'D': {'hydropathy': -3.5, 'volume': 111.1, 'polarity': 13.0, 'charge': -1},
            'E': {'hydropathy': -3.5, 'volume': 138.4, 'polarity': 12.3, 'charge': -1},
            'F': {'hydropathy': 2.8, 'volume': 189.9, 'polarity': 5.2, 'charge': 0},
            'G': {'hydropathy': -0.4, 'volume': 60.1, 'polarity': 9.0, 'charge': 0},
            'H': {'hydropathy': -3.2, 'volume': 153.2, 'polarity': 10.4, 'charge': 0},
            'I': {'hydropathy': 4.5, 'volume': 166.7, 'polarity': 5.2, 'charge': 0},
            'K': {'hydropathy': -3.9, 'volume': 168.6, 'polarity': 11.3, 'charge': 1},
            'L': {'hydropathy': 3.8, 'volume': 166.7, 'polarity': 4.9, 'charge': 0},
            'M': {'hydropathy': 1.9, 'volume': 162.9, 'polarity': 5.7, 'charge': 0},
            'N': {'hydropathy': -3.5, 'volume': 114.1, 'polarity': 11.6, 'charge': 0},
            'P': {'hydropathy': -1.6, 'volume': 112.7, 'polarity': 8.0, 'charge': 0},
            'Q': {'hydropathy': -3.5, 'volume': 143.8, 'polarity': 10.5, 'charge': 0},
            'R': {'hydropathy': -4.5, 'volume': 173.4, 'polarity': 10.5, 'charge': 1},
            'S': {'hydropathy': -0.8, 'volume': 89.0, 'polarity': 9.2, 'charge': 0},
            'T': {'hydropathy': -0.7, 'volume': 116.1, 'polarity': 8.6, 'charge': 0},
            'V': {'hydropathy': 4.2, 'volume': 140.0, 'polarity': 5.9, 'charge': 0},
            'W': {'hydropathy': -0.9, 'volume': 227.8, 'polarity': 5.4, 'charge': 0},
            'Y': {'hydropathy': -1.3, 'volume': 193.6, 'polarity': 6.2, 'charge': 0},
            'X': {'hydropathy': 0.0, 'volume': 100.0, 'polarity': 0.0, 'charge': 0},  # Unknown
        }
        
        # Normalize properties
        for aa in properties:
            props = properties[aa]
            # Normalize hydropathy to [0, 1]
            props['hydropathy_norm'] = (props['hydropathy'] + 4.5) / 9.0
            # Normalize volume to [0, 1] 
            props['volume_norm'] = props['volume'] / 300.0
            # Normalize polarity to [0, 1]
            props['polarity_norm'] = props['polarity'] / 15.0
            # Charge is already in a good range [-1, 1]
            props['charge_norm'] = (props['charge'] + 1) / 2.0
        
        return properties
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean protein sequence by removing invalid characters.
        
        Args:
            sequence: Raw protein sequence
            
        Returns:
            Cleaned sequence
        """
        if not isinstance(sequence, str):
            return ""
        
        # Convert to uppercase
        sequence = sequence.upper().strip()
        
        # Remove non-amino acid characters
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', sequence)
        
        return sequence
    
    def tokenize_sequence(self, sequence: str) -> List[int]:
        """
        Convert protein sequence to token IDs.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            List of token IDs
        """
        if sequence in self._sequence_cache:
            return self._sequence_cache[sequence]
        
        # Clean sequence
        clean_seq = self.clean_sequence(sequence)
        
        if not clean_seq:
            tokens = [self.padding_token]
        else:
            # Convert amino acids to token IDs
            tokens = []
            for aa in clean_seq:
                token_id = self.amino_acid_vocab.get(aa, self.unknown_token)
                tokens.append(token_id)
        
        # Cache result
        self._sequence_cache[sequence] = tokens
        
        return tokens
    
    def pad_sequence(self, tokens: List[int]) -> List[int]:
        """
        Pad or truncate sequence to fixed length.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Padded/truncated sequence
        """
        if len(tokens) > self.max_length:
            # Truncate from the end
            return tokens[:self.max_length]
        else:
            # Pad with padding tokens
            padding_needed = self.max_length - len(tokens)
            return tokens + [self.padding_token] * padding_needed
    
    def extract_physicochemical_features(self, sequence: str) -> np.ndarray:
        """
        Extract physicochemical features for each amino acid in the sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Array of physicochemical features [seq_len, num_features]
        """
        clean_seq = self.clean_sequence(sequence)
        
        features = []
        feature_names = ['hydropathy_norm', 'volume_norm', 'polarity_norm', 'charge_norm']
        
        for aa in clean_seq:
            aa_props = self.aa_properties.get(aa, self.aa_properties['X'])
            aa_features = [aa_props[prop] for prop in feature_names]
            features.append(aa_features)
        
        if not features:
            # Handle empty sequences
            features = [[0.0] * len(feature_names)]
        
        return np.array(features, dtype=np.float32)
    
    def extract_sequence_statistics(self, sequence: str) -> Dict[str, float]:
        """
        Extract global sequence statistics.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Dictionary of sequence statistics
        """
        clean_seq = self.clean_sequence(sequence)
        
        if not clean_seq:
            return {
                'length': 0,
                'hydropathy_mean': 0.0,
                'hydropathy_std': 0.0,
                'charge_sum': 0.0,
                'aromatic_fraction': 0.0,
                'polar_fraction': 0.0,
                'charged_fraction': 0.0
            }
        
        # Calculate amino acid composition
        aa_counts = {}
        for aa in clean_seq:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # Calculate physicochemical statistics
        hydropathy_values = [self.aa_properties[aa]['hydropathy'] for aa in clean_seq]
        charges = [self.aa_properties[aa]['charge'] for aa in clean_seq]
        
        # Count specific amino acid types
        aromatic_aas = set('FWY')
        polar_aas = set('NQST')
        charged_aas = set('DEKR')
        
        aromatic_count = sum(1 for aa in clean_seq if aa in aromatic_aas)
        polar_count = sum(1 for aa in clean_seq if aa in polar_aas)
        charged_count = sum(1 for aa in clean_seq if aa in charged_aas)
        
        stats = {
            'length': len(clean_seq),
            'hydropathy_mean': np.mean(hydropathy_values),
            'hydropathy_std': np.std(hydropathy_values),
            'charge_sum': sum(charges),
            'aromatic_fraction': aromatic_count / len(clean_seq),
            'polar_fraction': polar_count / len(clean_seq),
            'charged_fraction': charged_count / len(clean_seq)
        }
        
        return stats
    
    def extract_features(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Extract protein features from amino acid sequence (compatible with examples).
        
        Args:
            sequence: Protein amino acid sequence
            
        Returns:
            Dictionary containing:
            - sequence: Encoded sequence [max_length]
            - length: Actual sequence length
            - mask: Sequence mask [max_length] 
            - physicochemical: Physicochemical properties [max_length, n_props]
        """
        # Use existing processing pipeline
        processed = self.process_sequence(sequence, include_physicochemical=True, include_statistics=False)
        
        # Convert to expected format for examples
        sequence_tensor = torch.tensor(processed['padded_tokens'], dtype=torch.long)
        mask_tensor = torch.tensor(processed['mask'], dtype=torch.bool)
        
        # Get physicochemical features and pad/truncate to max_length
        physchem_features = processed['physicochemical_features']
        if len(physchem_features) < self.max_length:
            padding = np.zeros((self.max_length - len(physchem_features), physchem_features.shape[1]))
            physchem_features = np.vstack([physchem_features, padding])
        elif len(physchem_features) > self.max_length:
            physchem_features = physchem_features[:self.max_length]
        
        physicochemical_tensor = torch.tensor(physchem_features, dtype=torch.float32)
        
        return {
            'sequence': sequence_tensor,
            'length': processed['length'],
            'mask': mask_tensor,
            'physicochemical': physicochemical_tensor,
            'original_sequence': processed['clean_sequence']
        }

    def process_sequence(self, sequence: str, 
                        include_physicochemical: bool = True,
                        include_statistics: bool = True) -> Dict[str, Any]:
        """
        Process a single protein sequence to extract all features.
        
        Args:
            sequence: Protein sequence string
            include_physicochemical: Whether to include physicochemical features
            include_statistics: Whether to include sequence statistics
            
        Returns:
            Dictionary containing processed sequence data
        """
        # Tokenize sequence
        tokens = self.tokenize_sequence(sequence)
        padded_tokens = self.pad_sequence(tokens)
        
        result = {
            'sequence': sequence,
            'clean_sequence': self.clean_sequence(sequence),
            'tokens': tokens,
            'padded_tokens': padded_tokens,
            'length': len(tokens),
            'mask': [1 if token != self.padding_token else 0 for token in padded_tokens]
        }
        
        # Add physicochemical features
        if include_physicochemical:
            physchem_features = self.extract_physicochemical_features(sequence)
            result['physicochemical_features'] = physchem_features
        
        # Add sequence statistics
        if include_statistics:
            stats = self.extract_sequence_statistics(sequence)
            result['statistics'] = stats
        
        return result
    
    def process_sequence_list(self, sequences: List[str],
                             include_physicochemical: bool = True,
                             include_statistics: bool = True) -> Dict[str, Any]:
        """
        Process a list of protein sequences.
        
        Args:
            sequences: List of protein sequences
            include_physicochemical: Whether to include physicochemical features
            include_statistics: Whether to include sequence statistics
            
        Returns:
            Dictionary containing processed sequences data
        """
        logger.info(f"Processing {len(sequences)} protein sequences...")
        
        processed_sequences = []
        valid_sequences = []
        invalid_count = 0
        
        for i, sequence in enumerate(sequences):
            try:
                processed = self.process_sequence(
                    sequence, 
                    include_physicochemical=include_physicochemical,
                    include_statistics=include_statistics
                )
                
                if processed['length'] > 0:
                    processed_sequences.append(processed)
                    valid_sequences.append(sequence)
                else:
                    invalid_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing sequence {i}: {e}")
                invalid_count += 1
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(sequences)} sequences")
        
        logger.info(f"Successfully processed {len(processed_sequences)} sequences, {invalid_count} failed")
        
        result = {
            'processed_sequences': processed_sequences,
            'valid_sequences': valid_sequences,
            'invalid_count': invalid_count,
            'success_rate': len(processed_sequences) / len(sequences) if sequences else 0.0,
            'vocab_size': len(self.amino_acid_vocab),
            'max_length': self.max_length
        }
        
        return result
    
    def create_batch_tensors(self, processed_sequences: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Create batch tensors from processed sequences.
        
        Args:
            processed_sequences: List of processed sequence dictionaries
            
        Returns:
            Dictionary of batch tensors
        """
        if not processed_sequences:
            return {}
        
        # Extract data for batching
        batch_tokens = []
        batch_masks = []
        batch_lengths = []
        
        for seq_data in processed_sequences:
            batch_tokens.append(seq_data['padded_tokens'])
            batch_masks.append(seq_data['mask'])
            batch_lengths.append(seq_data['length'])
        
        # Convert to tensors
        batch_tensors = {
            'tokens': torch.tensor(batch_tokens, dtype=torch.long),
            'mask': torch.tensor(batch_masks, dtype=torch.bool),
            'lengths': torch.tensor(batch_lengths, dtype=torch.long)
        }
        
        # Add physicochemical features if available
        if 'physicochemical_features' in processed_sequences[0]:
            # Pad physicochemical features to max_length
            batch_physchem = []
            for seq_data in processed_sequences:
                features = seq_data['physicochemical_features']
                # Pad to max_length
                if len(features) < self.max_length:
                    padding = np.zeros((self.max_length - len(features), features.shape[1]))
                    features = np.vstack([features, padding])
                elif len(features) > self.max_length:
                    features = features[:self.max_length]
                
                batch_physchem.append(features)
            
            batch_tensors['physicochemical'] = torch.tensor(batch_physchem, dtype=torch.float)
        
        return batch_tensors
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to amino acid sequence.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Amino acid sequence string
        """
        sequence = ""
        for token in tokens:
            if token == self.padding_token:
                break  # Stop at padding
            aa = self.vocab_to_aa.get(token, 'X')
            if aa != '<PAD>':
                sequence += aa
        
        return sequence
    
    def save_processed_proteins(self, processed_data: Dict[str, Any], 
                               output_path: str) -> None:
        """
        Save processed protein data.
        
        Args:
            processed_data: Dictionary containing processed protein data
            output_path: Path to save the data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Saved processed protein data to {output_path}")
    
    def load_processed_proteins(self, input_path: str) -> Dict[str, Any]:
        """
        Load processed protein data.
        
        Args:
            input_path: Path to the saved data
            
        Returns:
            Dictionary containing processed protein data
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded processed protein data from {input_path}")
        return data
    
    def get_vocab_info(self) -> Dict[str, Any]:
        """
        Get vocabulary information.
        
        Returns:
            Dictionary with vocabulary details
        """
        return {
            'vocab_size': len(self.amino_acid_vocab),
            'vocab': self.amino_acid_vocab.copy(),
            'max_length': self.max_length,
            'padding_token': self.padding_token,
            'unknown_token': self.unknown_token
        }
    
    def clear_cache(self) -> None:
        """Clear the sequence cache."""
        self._sequence_cache.clear()
        logger.info("Cleared sequence cache")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example protein sequences
    test_sequences = [
        "MGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVECLEQGGMIDRPGKISERGFHLVL",  # ABL1 (truncated)
        "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEI",  # EGFR (truncated)
        "INVALID_SEQUENCE_WITH_NUMBERS123",  # Invalid sequence
    ]
    
    # Initialize extractor
    extractor = ProteinFeatureExtractor(max_length=100)
    
    # Process sequences
    processed = extractor.process_sequence_list(test_sequences)
    
    print(f"Processed {len(processed['processed_sequences'])} sequences")
    print(f"Success rate: {processed['success_rate']:.2%}")
    
    # Print vocabulary info
    vocab_info = extractor.get_vocab_info()
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Max sequence length: {vocab_info['max_length']}")
    
    # Example batch creation
    if processed['processed_sequences']:
        batch_tensors = extractor.create_batch_tensors(processed['processed_sequences'])
        print(f"Batch tensor shapes:")
        for key, tensor in batch_tensors.items():
            print(f"  {key}: {tensor.shape}")