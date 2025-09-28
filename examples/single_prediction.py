#!/usr/bin/env python3
"""
Example: Single Drug-Target Binding Affinity Prediction

This example demonstrates how to use DeepDTA-Pro for predicting binding affinity
between a single drug-target pair, including feature extraction, model loading,
prediction, and basic interpretation.

Usage:
    python single_prediction.py
    python single_prediction.py --drug_smiles "CCO" --protein_sequence "MKTV..."
"""

import argparse
import torch
import sys
import os
from pathlib import Path

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import from src directory structure
try:
    from src.models.deepdta_pro import DeepDTAPro
    from src.data.molecular_features import MolecularFeatureExtractor
    from src.data.protein_features import ProteinFeatureExtractor
    from src.interpretability.attention_visualization import AttentionVisualizer
    from src.utils.logger import Logger
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Creating mock classes for demonstration...")
    
    # Mock classes for demonstration
    class DeepDTAPro:
        def __init__(self, *args, **kwargs):
            pass
        def eval(self):
            pass
        def __call__(self, *args):
            import torch
            return torch.tensor([6.5 + torch.randn(1).item() * 0.5])
        @classmethod
        def load_from_checkpoint(cls, path, map_location='cpu'):
            return cls()
    
    class MolecularFeatureExtractor:
        def __init__(self, max_atoms=100):
            self.max_atoms = max_atoms
            
        def extract_features(self, smiles):
            import torch
            return {
                'x': torch.randn(20, 78),
                'edge_index': torch.randint(0, 20, (2, 40)),
                'smiles': smiles
            }
    
    class ProteinFeatureExtractor:
        def __init__(self, max_length=1000, encoding_type='learned'):
            self.max_length = max_length
            self.encoding_type = encoding_type
            
        def extract_features(self, sequence):
            import torch
            return {
                'sequence': torch.randint(0, 25, (min(len(sequence), self.max_length),)),
                'length': min(len(sequence), self.max_length),
                'mask': torch.ones(min(len(sequence), self.max_length), dtype=torch.bool)
            }
    
    class AttentionVisualizer:
        def __init__(self, model, device='cpu'):
            pass
        def get_attention_weights(self, mol_data, prot_data):
            import torch
            return {
                'molecular_attention': torch.randn(1, mol_data['x'].shape[0]),
                'protein_attention': torch.randn(1, prot_data['length'])
            }
        def visualize_molecular_attention(self, smiles, attention, save_path=None):
            if save_path:
                print(f"   🖼️ Mock molecular attention saved: {save_path}")
        def visualize_protein_attention(self, sequence, attention, save_path=None):
            if save_path:
                print(f"   🖼️ Mock protein attention saved: {save_path}")
    
    class Logger:
        def __init__(self, name, log_dir):
            self.name = name
        def info(self, message):
            pass
        def error(self, message):
            pass

def main():
    """Main function for single prediction example."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Single Drug-Target Prediction')
    parser.add_argument('--drug_smiles', type=str, 
                       default='CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
                       help='Drug SMILES string (default: Ibuprofen)')
    parser.add_argument('--protein_sequence', type=str,
                       default='MKKFFDSRREQGGSGLGSGSSGGGGSGGGYGNQDQSGGGGSGGGYGNQDQSGGGGSGGGYGNQDQSGGGGSGGGYGNQDQS',
                       help='Protein amino acid sequence')
    parser.add_argument('--model_path', type=str,
                       default='models/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate attention visualizations')
    parser.add_argument('--output_dir', type=str, default='outputs/single_prediction',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 DeepDTA-Pro Single Prediction Example")
    print(f"📱 Using device: {device}")
    print(f"💊 Drug SMILES: {args.drug_smiles}")
    print(f"🧬 Protein sequence: {args.protein_sequence[:50]}{'...' if len(args.protein_sequence) > 50 else ''}")
    print("-" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger('single_prediction', args.output_dir)
    logger.info("Starting single prediction example")
    
    try:
        # Step 1: Initialize feature extractors
        print("🔧 Initializing feature extractors...")
        mol_extractor = MolecularFeatureExtractor(max_atoms=100)
        prot_extractor = ProteinFeatureExtractor(max_length=1000)
        
        # Step 2: Extract molecular features
        print("⚗️ Extracting molecular features...")
        try:
            mol_data = mol_extractor.extract_features(args.drug_smiles)
            print(f"   ✅ Molecular graph: {mol_data['x'].shape[0]} atoms, {mol_data['edge_index'].shape[1]} bonds")
            logger.info(f"Extracted molecular features: {mol_data['x'].shape[0]} atoms")
        except Exception as e:
            print(f"   ❌ Error extracting molecular features: {e}")
            return
        
        # Step 3: Extract protein features
        print("🧬 Extracting protein features...")
        try:
            prot_data = prot_extractor.extract_features(args.protein_sequence)
            print(f"   ✅ Protein sequence: {prot_data['length']} residues")
            logger.info(f"Extracted protein features: {prot_data['length']} residues")
        except Exception as e:
            print(f"   ❌ Error extracting protein features: {e}")
            return
        
        # Step 4: Load trained model
        print("🤖 Loading trained model...")
        try:
            if os.path.exists(args.model_path):
                model = DeepDTAPro.load_from_checkpoint(args.model_path, map_location=device)
                model.eval()
                print(f"   ✅ Model loaded from {args.model_path}")
                logger.info(f"Model loaded from {args.model_path}")
            else:
                print(f"   ⚠️ Model file not found: {args.model_path}")
                print("   🔄 Creating demo model for illustration...")
                # Create demo model configuration
                model_config = {
                    'molecular': {
                        'input_dim': mol_data['x'].shape[1],
                        'hidden_dim': 128,
                        'output_dim': 256,
                        'num_layers': 3,
                        'dropout': 0.1
                    },
                    'protein': {
                        'vocab_size': 25,
                        'embedding_dim': 128,
                        'hidden_dim': 256,
                        'num_layers': 2,
                        'dropout': 0.1
                    },
                    'fusion': {
                        'input_dim': 512,
                        'hidden_dim': 256,
                        'output_dim': 1,
                        'dropout': 0.1
                    }
                }
                model = DeepDTAPro(
                    model_config['molecular'],
                    model_config['protein'],
                    model_config['fusion']
                )
                model.eval()
                print("   ✅ Demo model created (untrained)")
                logger.info("Created demo model for illustration")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return
        
        # Step 5: Prepare data for inference
        print("📦 Preparing data for inference...")
        try:
            # Convert to tensors and add batch dimension
            mol_batch = {
                'x': mol_data['x'].unsqueeze(0),
                'edge_index': mol_data['edge_index'],
                'batch': torch.zeros(mol_data['x'].shape[0], dtype=torch.long)
            }
            
            prot_batch = {
                'sequence': prot_data['sequence'].unsqueeze(0),
                'length': torch.tensor([prot_data['length']], dtype=torch.long),
                'mask': prot_data['mask'].unsqueeze(0)
            }
            
            print("   ✅ Data prepared for batch inference")
            logger.info("Data prepared for inference")
        except Exception as e:
            print(f"   ❌ Error preparing data: {e}")
            return
        
        # Step 6: Make prediction
        print("🔮 Making binding affinity prediction...")
        try:
            with torch.no_grad():
                prediction = model(mol_batch, prot_batch)
                binding_affinity = prediction.item()
            
            print(f"   🎯 Predicted binding affinity: {binding_affinity:.3f}")
            print(f"   📊 Binding strength: {get_binding_strength_description(binding_affinity)}")
            
            logger.info(f"Prediction completed: {binding_affinity:.3f}")
            
            # Save prediction results
            results = {
                'drug_smiles': args.drug_smiles,
                'protein_sequence': args.protein_sequence,
                'predicted_affinity': binding_affinity,
                'binding_strength': get_binding_strength_description(binding_affinity),
                'molecular_properties': get_molecular_properties(args.drug_smiles),
                'protein_properties': get_protein_properties(args.protein_sequence)
            }
            
            # Save to file
            import json
            with open(os.path.join(args.output_dir, 'prediction_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"   💾 Results saved to {args.output_dir}/prediction_results.json")
            
        except Exception as e:
            print(f"   ❌ Error making prediction: {e}")
            return
        
        # Step 7: Generate attention visualizations (if requested)
        if args.visualize:
            print("🎨 Generating attention visualizations...")
            try:
                visualizer = AttentionVisualizer(model, device=device)
                
                # Get attention weights
                attention_data = visualizer.get_attention_weights(mol_batch, prot_batch)
                print(f"   ✅ Extracted attention weights")
                
                # Visualize molecular attention
                mol_attention_path = os.path.join(args.output_dir, 'molecular_attention.png')
                visualizer.visualize_molecular_attention(
                    args.drug_smiles, 
                    attention_data['molecular_attention'][0],  # Remove batch dimension
                    save_path=mol_attention_path
                )
                print(f"   🖼️ Molecular attention saved: {mol_attention_path}")
                
                # Visualize protein attention
                prot_attention_path = os.path.join(args.output_dir, 'protein_attention.png')
                visualizer.visualize_protein_attention(
                    args.protein_sequence,
                    attention_data['protein_attention'][0],  # Remove batch dimension
                    save_path=prot_attention_path
                )
                print(f"   🖼️ Protein attention saved: {prot_attention_path}")
                
                logger.info("Attention visualizations generated")
                
            except Exception as e:
                print(f"   ⚠️ Could not generate visualizations: {e}")
                print("   💡 This is normal if optional dependencies are not installed")
        
        # Step 8: Print summary
        print("\n" + "="*70)
        print("📋 PREDICTION SUMMARY")
        print("="*70)
        print(f"Drug (SMILES): {args.drug_smiles}")
        print(f"Protein length: {len(args.protein_sequence)} residues")
        print(f"Predicted binding affinity: {binding_affinity:.3f}")
        print(f"Binding strength: {get_binding_strength_description(binding_affinity)}")
        print(f"Output directory: {args.output_dir}")
        print("="*70)
        
        logger.info("Single prediction example completed successfully")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def get_binding_strength_description(affinity_score):
    """
    Get human-readable binding strength description.
    
    Args:
        affinity_score (float): Predicted binding affinity score
    
    Returns:
        str: Binding strength description
    """
    if affinity_score >= 8.0:
        return "Very Strong (High Affinity)"
    elif affinity_score >= 7.0:
        return "Strong (Good Affinity)"
    elif affinity_score >= 6.0:
        return "Moderate (Medium Affinity)"
    elif affinity_score >= 5.0:
        return "Weak (Low Affinity)"
    else:
        return "Very Weak (Minimal Affinity)"

def get_molecular_properties(smiles):
    """
    Calculate basic molecular properties.
    
    Args:
        smiles (str): SMILES string
    
    Returns:
        dict: Molecular properties
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol)
        }
    except ImportError:
        return {"note": "RDKit not available for molecular property calculation"}

def get_protein_properties(sequence):
    """
    Calculate basic protein properties.
    
    Args:
        sequence (str): Protein sequence
    
    Returns:
        dict: Protein properties
    """
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    valid_residues = sum(1 for aa in sequence if aa in amino_acids)
    
    return {
        "length": len(sequence),
        "valid_residues": valid_residues,
        "validity": valid_residues / len(sequence) if len(sequence) > 0 else 0,
        "composition": {aa: sequence.count(aa) for aa in amino_acids if sequence.count(aa) > 0}
    }

if __name__ == "__main__":
    main()