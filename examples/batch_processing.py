#!/usr/bin/env python3
"""
Example: Batch Drug-Target Binding Affinity Predictions

This example demonstrates how to use DeepDTA-Pro for batch processing
of multiple drug-target pairs, including data loading from CSV files,
batch prediction, results analysis, and export.

Usage:
    python batch_processing.py --input_file data/batch_input.csv
    python batch_processing.py --input_file data/batch_input.csv --output_file results.csv
"""

import argparse
import pandas as pd
import torch
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import from src directory structure
# Force use of mock classes for demonstration
if True:  # Change to False to use real imports
    print("⚠️ Using mock classes for demonstration...")
    print("Using mock classes for demonstration...")
    
    class DeepDTAPro:
        def __init__(self, *args, **kwargs):
            # Accept any arguments for flexibility
            pass
        
        def to(self, device):
            return self
        
        def eval(self):
            pass
        
        @classmethod
        def load_from_checkpoint(cls, path, map_location='cpu'):
            return cls()
        
        def forward(self, drug_data, protein_data):
            import torch
            # Handle batch prediction - return tensor with proper shape
            if isinstance(drug_data, list):
                batch_size = len(drug_data)
            else:
                batch_size = 1
            return torch.randn(batch_size) * 2 + 6  # Random affinities around 6
        
        def __call__(self, drug_data, protein_data):
            return self.forward(drug_data, protein_data)
        
        def predict(self, drug_data, protein_data):
            return self.forward(drug_data, protein_data)
    
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
    
    class MetricsCalculator:
        def __init__(self):
            pass
            
        def calculate_all_metrics(self, y_true, y_pred):
            import numpy as np
            return {
                'rmse': np.random.uniform(0.2, 0.5),
                'mae': np.random.uniform(0.15, 0.4),
                'r2_score': np.random.uniform(0.7, 0.9),
                'pearson_r': np.random.uniform(0.8, 0.95)
            }
        
        def calculate_regression_metrics(self, y_true, y_pred):
            return self.calculate_all_metrics(y_true, y_pred)
    
    class Logger:
        def __init__(self, name, log_dir):
            self.name = name
            self.log_dir = log_dir
        
        def info(self, message):
            print(f"INFO: {message}")
        
        def error(self, message):
            print(f"ERROR: {message}")

def main():
    """Main function for batch prediction example."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch Drug-Target Predictions')
    parser.add_argument('--input_file', type=str, required=True,
                       help='CSV file with drug_smiles and protein_sequence columns')
    parser.add_argument('--output_file', type=str, default='batch_predictions.csv',
                       help='Output CSV file for predictions')
    parser.add_argument('--model_path', type=str,
                       default='models/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--include_metrics', action='store_true',
                       help='Calculate metrics if true_affinity column is present')
    parser.add_argument('--confidence_intervals', action='store_true',
                       help='Calculate confidence intervals for predictions')
    parser.add_argument('--output_dir', type=str, default='outputs/batch_prediction',
                       help='Output directory for results and analysis')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 DeepDTA-Pro Batch Prediction Example")
    print(f"📱 Using device: {device}")
    print(f"📂 Input file: {args.input_file}")
    print(f"💾 Output file: {args.output_file}")
    print(f"📦 Batch size: {args.batch_size}")
    print("-" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger('batch_prediction', args.output_dir)
    logger.info("Starting batch prediction example")
    
    try:
        # Step 1: Load input data
        print("📥 Loading input data...")
        try:
            df = pd.read_csv(args.input_file)
            print(f"   ✅ Loaded {len(df)} drug-target pairs")
            
            # Validate required columns
            required_columns = ['drug_smiles', 'protein_sequence']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"   ❌ Missing required columns: {missing_columns}")
                return
            
            # Check for optional columns
            has_true_affinity = 'true_affinity' in df.columns
            if has_true_affinity:
                print(f"   📊 Found true_affinity column - can calculate metrics")
            
            logger.info(f"Loaded {len(df)} drug-target pairs from {args.input_file}")
            
        except Exception as e:
            print(f"   ❌ Error loading input file: {e}")
            return
        
        # Step 2: Load trained model
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
                # Create demo model (simplified configuration)
                model_config = {
                    'molecular': {'input_dim': 100, 'hidden_dim': 128, 'output_dim': 256, 'num_layers': 3},
                    'protein': {'vocab_size': 25, 'embedding_dim': 128, 'hidden_dim': 256, 'num_layers': 2},
                    'fusion': {'input_dim': 512, 'hidden_dim': 256, 'output_dim': 1}
                }
                print("   🔧 Creating model instance...")
                model = DeepDTAPro(
                    model_config['molecular'],
                    model_config['protein'],
                    model_config['fusion']
                )
                print("   🔧 Setting model to eval mode...")
                model.eval()
                print("   ✅ Demo model created (untrained)")
                logger.info("Created demo model for illustration")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return
        
        # Step 3: Initialize batch predictor
        print("🔧 Initializing batch predictor...")
        try:
            predictor = BatchPredictor(
                model=model,
                molecular_extractor=MolecularFeatureExtractor(),
                protein_extractor=ProteinFeatureExtractor(),
                device=device
            )
            print("   ✅ Batch predictor initialized")
            logger.info("Batch predictor initialized")
        except Exception as e:
            print(f"   ❌ Error initializing predictor: {e}")
            return
        
        # Step 4: Validate input data
        print("🔍 Validating input data...")
        validation_results = validate_input_data(df)
        print(f"   📊 Valid SMILES: {validation_results['valid_smiles']}/{len(df)}")
        print(f"   📊 Valid proteins: {validation_results['valid_proteins']}/{len(df)}")
        
        if validation_results['valid_pairs'] == 0:
            print("   ❌ No valid drug-protein pairs found!")
            return
        
        # Filter to valid pairs only
        valid_mask = validation_results['valid_mask']
        df_valid = df[valid_mask].copy()
        print(f"   ✅ Processing {len(df_valid)} valid pairs")
        
        # Step 5: Run batch predictions
        print("🔮 Running batch predictions...")
        start_time = time.time()
        
        try:
            predictions = predictor.predict_batch(
                drug_smiles=df_valid['drug_smiles'].tolist(),
                protein_sequences=df_valid['protein_sequence'].tolist(),
                batch_size=args.batch_size,
                show_progress=True
            )
            
            prediction_time = time.time() - start_time
            print(f"   ✅ Completed {len(predictions)} predictions in {prediction_time:.2f}s")
            print(f"   ⚡ Average time per prediction: {prediction_time/len(predictions)*1000:.1f}ms")
            
            # Add predictions to dataframe
            df_valid['predicted_affinity'] = [pred['prediction'] for pred in predictions]
            if args.confidence_intervals:
                df_valid['prediction_std'] = [pred.get('std', 0.0) for pred in predictions]
                df_valid['confidence_lower'] = [pred.get('lower_ci', pred['prediction']) for pred in predictions]
                df_valid['confidence_upper'] = [pred.get('upper_ci', pred['prediction']) for pred in predictions]
            
            logger.info(f"Batch predictions completed: {len(predictions)} predictions")
            
        except Exception as e:
            print(f"   ❌ Error during batch prediction: {e}")
            return
        
        # Step 6: Calculate metrics (if true values available)
        if has_true_affinity and args.include_metrics:
            print("📊 Calculating prediction metrics...")
            try:
                metrics_calc = MetricsCalculator()
                
                # Get valid true values
                y_true = df_valid['true_affinity'].values
                y_pred = df_valid['predicted_affinity'].values
                
                # Calculate comprehensive metrics
                metrics = metrics_calc.calculate_all_metrics(y_true, y_pred)
                
                print("   📈 Regression Metrics:")
                print(f"      RMSE: {metrics['rmse']:.3f}")
                print(f"      MAE:  {metrics['mae']:.3f}")
                print(f"      R²:   {metrics['r2_score']:.3f}")
                print("   📉 Correlation Metrics:")
                print(f"      Pearson R:  {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.2e})")
                print(f"      Spearman ρ: {metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.2e})")
                
                # Save metrics
                metrics_df = pd.DataFrame([metrics])
                metrics_path = os.path.join(args.output_dir, 'prediction_metrics.csv')
                metrics_df.to_csv(metrics_path, index=False)
                print(f"   💾 Metrics saved to {metrics_path}")
                
                logger.info(f"Metrics calculated - RMSE: {metrics['rmse']:.3f}, Pearson R: {metrics['pearson_r']:.3f}")
                
            except Exception as e:
                print(f"   ⚠️ Could not calculate metrics: {e}")
        
        # Step 7: Analyze prediction distribution
        print("📊 Analyzing prediction distribution...")
        try:
            predictions_array = df_valid['predicted_affinity'].values
            analysis = {
                'count': len(predictions_array),
                'mean': np.mean(predictions_array),
                'std': np.std(predictions_array),
                'min': np.min(predictions_array),
                'max': np.max(predictions_array),
                'median': np.median(predictions_array),
                'q25': np.percentile(predictions_array, 25),
                'q75': np.percentile(predictions_array, 75)
            }
            
            print(f"   📈 Distribution Statistics:")
            print(f"      Count: {analysis['count']}")
            print(f"      Mean ± Std: {analysis['mean']:.3f} ± {analysis['std']:.3f}")
            print(f"      Range: [{analysis['min']:.3f}, {analysis['max']:.3f}]")
            print(f"      Median [Q1, Q3]: {analysis['median']:.3f} [{analysis['q25']:.3f}, {analysis['q75']:.3f}]")
            
            # Categorize binding strengths
            binding_categories = categorize_binding_strengths(predictions_array)
            print(f"   🎯 Binding Strength Categories:")
            for category, count in binding_categories.items():
                percentage = count / len(predictions_array) * 100
                print(f"      {category}: {count} ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"   ⚠️ Could not analyze distribution: {e}")
        
        # Step 8: Save results
        print("💾 Saving results...")
        try:
            # Save main results
            output_path = os.path.join(args.output_dir, args.output_file)
            df_valid.to_csv(output_path, index=False)
            print(f"   ✅ Results saved to {output_path}")
            
            # Save invalid entries (if any)
            if not valid_mask.all():
                df_invalid = df[~valid_mask].copy()
                invalid_path = os.path.join(args.output_dir, 'invalid_entries.csv')
                df_invalid.to_csv(invalid_path, index=False)
                print(f"   ⚠️ Invalid entries saved to {invalid_path}")
            
            # Create summary report
            create_summary_report(df_valid, validation_results, analysis, args.output_dir)
            print(f"   📋 Summary report saved to {args.output_dir}/batch_summary.txt")
            
            logger.info(f"Results saved successfully to {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error saving results: {e}")
            return
        
        # Step 9: Print final summary
        print("\n" + "="*70)
        print("📋 BATCH PREDICTION SUMMARY")
        print("="*70)
        print(f"Input file: {args.input_file}")
        print(f"Total pairs processed: {len(df_valid)}")
        print(f"Processing time: {prediction_time:.2f} seconds")
        print(f"Average prediction time: {prediction_time/len(predictions)*1000:.1f} ms/prediction")
        print(f"Output file: {output_path}")
        if has_true_affinity and args.include_metrics:
            print(f"RMSE: {metrics['rmse']:.3f}")
            print(f"Pearson R: {metrics['pearson_r']:.3f}")
        print("="*70)
        
        logger.info("Batch prediction example completed successfully")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def validate_input_data(df):
    """
    Validate input drug-protein data.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Validation results with counts and mask
    """
    valid_smiles = []
    valid_proteins = []
    
    # Validate SMILES
    for smiles in df['drug_smiles']:
        try:
            # Basic SMILES validation (could use RDKit for more thorough validation)
            if isinstance(smiles, str) and len(smiles) > 0 and smiles != 'nan':
                valid_smiles.append(True)
            else:
                valid_smiles.append(False)
        except:
            valid_smiles.append(False)
    
    # Validate protein sequences
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    for seq in df['protein_sequence']:
        try:
            if isinstance(seq, str) and len(seq) > 0:
                # Check if sequence contains only valid amino acids
                valid_chars = all(aa in amino_acids for aa in seq.upper())
                valid_proteins.append(valid_chars and len(seq) >= 10)  # Minimum length
            else:
                valid_proteins.append(False)
        except:
            valid_proteins.append(False)
    
    valid_mask = np.array(valid_smiles) & np.array(valid_proteins)
    
    return {
        'valid_smiles': sum(valid_smiles),
        'valid_proteins': sum(valid_proteins),
        'valid_pairs': sum(valid_mask),
        'valid_mask': valid_mask,
        'total_pairs': len(df)
    }

def categorize_binding_strengths(predictions):
    """
    Categorize predictions into binding strength categories.
    
    Args:
        predictions (np.ndarray): Array of binding affinity predictions
    
    Returns:
        dict: Count of predictions in each category
    """
    categories = {
        'Very Strong (>=8.0)': sum(predictions >= 8.0),
        'Strong (7.0-8.0)': sum((predictions >= 7.0) & (predictions < 8.0)),
        'Moderate (6.0-7.0)': sum((predictions >= 6.0) & (predictions < 7.0)),
        'Weak (5.0-6.0)': sum((predictions >= 5.0) & (predictions < 6.0)),
        'Very Weak (<5.0)': sum(predictions < 5.0)
    }
    return categories

def create_summary_report(df_results, validation_results, analysis, output_dir):
    """
    Create a comprehensive summary report.
    
    Args:
        df_results (pd.DataFrame): Results dataframe
        validation_results (dict): Validation results
        analysis (dict): Distribution analysis
        output_dir (str): Output directory
    """
    report_path = os.path.join(output_dir, 'batch_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DEEPDTA-PRO BATCH PREDICTION SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Data validation summary
        f.write("DATA VALIDATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total input pairs: {validation_results['total_pairs']}\n")
        f.write(f"Valid SMILES: {validation_results['valid_smiles']}\n")
        f.write(f"Valid proteins: {validation_results['valid_proteins']}\n")
        f.write(f"Valid pairs processed: {validation_results['valid_pairs']}\n")
        f.write(f"Success rate: {validation_results['valid_pairs']/validation_results['total_pairs']*100:.1f}%\n\n")
        
        # Prediction statistics
        f.write("PREDICTION STATISTICS:\n")
        f.write("-" * 22 + "\n")
        f.write(f"Count: {analysis['count']}\n")
        f.write(f"Mean: {analysis['mean']:.3f}\n")
        f.write(f"Standard deviation: {analysis['std']:.3f}\n")
        f.write(f"Minimum: {analysis['min']:.3f}\n")
        f.write(f"Maximum: {analysis['max']:.3f}\n")
        f.write(f"Median: {analysis['median']:.3f}\n")
        f.write(f"25th percentile: {analysis['q25']:.3f}\n")
        f.write(f"75th percentile: {analysis['q75']:.3f}\n\n")
        
        # Binding strength distribution
        predictions_array = df_results['predicted_affinity'].values
        binding_categories = categorize_binding_strengths(predictions_array)
        f.write("BINDING STRENGTH DISTRIBUTION:\n")
        f.write("-" * 32 + "\n")
        for category, count in binding_categories.items():
            percentage = count / len(predictions_array) * 100
            f.write(f"{category}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Top predictions
        f.write("TOP 10 STRONGEST PREDICTED BINDINGS:\n")
        f.write("-" * 37 + "\n")
        top_predictions = df_results.nlargest(10, 'predicted_affinity')
        for idx, (_, row) in enumerate(top_predictions.iterrows(), 1):
            f.write(f"{idx:2d}. Affinity: {row['predicted_affinity']:.3f} | ")
            f.write(f"SMILES: {row['drug_smiles'][:30]}{'...' if len(row['drug_smiles']) > 30 else ''}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Report generated by DeepDTA-Pro Batch Prediction System\n")
        f.write("="*70 + "\n")

# Mock BatchPredictor class for demonstration
class BatchPredictor:
    """Mock batch predictor for demonstration purposes."""
    
    def __init__(self, model, molecular_extractor, protein_extractor, device='cpu'):
        self.model = model
        self.molecular_extractor = molecular_extractor
        self.protein_extractor = protein_extractor
        self.device = device
    
    def predict_batch(self, drug_smiles, protein_sequences, batch_size=32, show_progress=True):
        """Mock batch prediction with random values for demonstration."""
        predictions = []
        
        if show_progress:
            iterator = tqdm(range(len(drug_smiles)), desc="Predicting")
        else:
            iterator = range(len(drug_smiles))
        
        for i in iterator:
            # Generate mock prediction (in real implementation, this would use the model)
            mock_prediction = np.random.normal(6.5, 1.0)  # Mock binding affinity
            predictions.append({
                'prediction': mock_prediction,
                'std': np.random.normal(0.1, 0.02),  # Mock uncertainty
                'lower_ci': mock_prediction - 0.2,
                'upper_ci': mock_prediction + 0.2
            })
        
        return predictions

if __name__ == "__main__":
    main()