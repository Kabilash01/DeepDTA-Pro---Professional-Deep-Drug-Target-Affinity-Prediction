"""
Web Interface Utilities
Helper functions and utilities for the DeepDTA-Pro web interface.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import io
import base64
from pathlib import Path
import json
import tempfile
import zipfile
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultsExporter:
    """Utility class for exporting prediction results in various formats."""
    
    @staticmethod
    def to_csv(results: List[Dict[str, Any]], filename: str = None) -> str:
        """Export results to CSV format."""
        df = pd.DataFrame(results)
        
        if filename is None:
            filename = f"deepdta_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return df.to_csv(index=False)
    
    @staticmethod
    def to_json(results: List[Dict[str, Any]], filename: str = None) -> str:
        """Export results to JSON format."""
        if filename is None:
            filename = f"deepdta_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return json.dumps(results, indent=2, default=str)
    
    @staticmethod
    def to_excel(results: List[Dict[str, Any]], filename: str = None) -> bytes:
        """Export results to Excel format."""
        df = pd.DataFrame(results)
        
        if filename is None:
            filename = f"deepdta_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Add summary sheet if multiple results
            if len(results) > 1:
                summary_data = {
                    'Metric': ['Count', 'Mean Affinity', 'Std Affinity', 'Min Affinity', 'Max Affinity'],
                    'Value': [
                        len(results),
                        np.mean([r.get('predicted_affinity', 0) for r in results]),
                        np.std([r.get('predicted_affinity', 0) for r in results]),
                        np.min([r.get('predicted_affinity', 0) for r in results]),
                        np.max([r.get('predicted_affinity', 0) for r in results])
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return output.getvalue()
    
    @staticmethod
    def create_report_package(results: List[Dict[str, Any]], 
                             visualizations: Dict[str, bytes] = None,
                             model_info: Dict[str, Any] = None) -> bytes:
        """Create a comprehensive report package as ZIP file."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add results CSV
            csv_content = ResultsExporter.to_csv(results)
            zip_file.writestr("predictions.csv", csv_content)
            
            # Add results JSON
            json_content = ResultsExporter.to_json(results)
            zip_file.writestr("predictions.json", json_content)
            
            # Add Excel file
            excel_content = ResultsExporter.to_excel(results)
            zip_file.writestr("predictions.xlsx", excel_content)
            
            # Add visualizations if provided
            if visualizations:
                for viz_name, viz_data in visualizations.items():
                    zip_file.writestr(f"visualizations/{viz_name}", viz_data)
            
            # Add model info if provided
            if model_info:
                model_info_content = json.dumps(model_info, indent=2, default=str)
                zip_file.writestr("model_info.json", model_info_content)
            
            # Add README
            readme_content = """
# DeepDTA-Pro Prediction Results

This package contains the following files:

## Data Files
- `predictions.csv`: Results in CSV format
- `predictions.json`: Results in JSON format  
- `predictions.xlsx`: Results in Excel format with summary sheet

## Visualizations
- `visualizations/`: Folder containing visualization images

## Model Information
- `model_info.json`: Model configuration and metadata

## Usage
The CSV file can be opened in Excel, Google Sheets, or any data analysis tool.
The JSON file can be loaded into Python using `json.load()`.

For questions or support, please refer to the DeepDTA-Pro documentation.
"""
            zip_file.writestr("README.md", readme_content)
        
        return zip_buffer.getvalue()

class DataValidator:
    """Utility class for validating input data."""
    
    @staticmethod
    def validate_smiles(smiles: str) -> Tuple[bool, str]:
        """
        Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not smiles or not smiles.strip():
            return False, "SMILES string is empty"
        
        smiles = smiles.strip()
        
        # Basic character validation
        valid_chars = set('()[]{}+-=#$:\\/@.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        invalid_chars = set(smiles) - valid_chars
        
        if invalid_chars:
            return False, f"Invalid characters in SMILES: {', '.join(invalid_chars)}"
        
        # Basic structure validation
        if smiles.count('(') != smiles.count(')'):
            return False, "Unmatched parentheses in SMILES"
        
        if smiles.count('[') != smiles.count(']'):
            return False, "Unmatched brackets in SMILES"
        
        # Length validation
        if len(smiles) > 500:
            return False, "SMILES string too long (>500 characters)"
        
        return True, ""
    
    @staticmethod
    def validate_protein_sequence(sequence: str) -> Tuple[bool, str]:
        """
        Validate protein sequence.
        
        Args:
            sequence: Protein sequence to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sequence or not sequence.strip():
            return False, "Protein sequence is empty"
        
        sequence = sequence.strip().upper()
        
        # Valid amino acid characters
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_aa = set(sequence) - valid_aa
        
        if invalid_aa:
            return False, f"Invalid amino acids: {', '.join(invalid_aa)}"
        
        # Length validation
        if len(sequence) < 10:
            return False, "Protein sequence too short (<10 amino acids)"
        
        if len(sequence) > 5000:
            return False, "Protein sequence too long (>5000 amino acids)"
        
        return True, ""
    
    @staticmethod
    def validate_batch_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate batch prediction data.
        
        Args:
            df: DataFrame containing batch data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_cols = ['drug_smiles', 'protein_sequence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            return False, errors
        
        # Check for empty data
        if len(df) == 0:
            errors.append("No data rows found")
            return False, errors
        
        # Validate individual entries
        for idx, row in df.iterrows():
            # Validate SMILES
            smiles_valid, smiles_error = DataValidator.validate_smiles(row['drug_smiles'])
            if not smiles_valid:
                errors.append(f"Row {idx+1} - SMILES: {smiles_error}")
            
            # Validate protein sequence
            seq_valid, seq_error = DataValidator.validate_protein_sequence(row['protein_sequence'])
            if not seq_valid:
                errors.append(f"Row {idx+1} - Protein: {seq_error}")
        
        return len(errors) == 0, errors

class VisualizationHelper:
    """Helper class for creating visualizations."""
    
    @staticmethod
    def create_distribution_plot(values: List[float], 
                               title: str = "Distribution",
                               xlabel: str = "Value",
                               bins: int = 30) -> bytes:
        """Create a distribution plot and return as bytes."""
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_val + std_val:.3f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_val - std_val:.3f}')
        plt.legend()
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer.getvalue()
    
    @staticmethod
    def create_scatter_plot(x_values: List[float], 
                          y_values: List[float],
                          title: str = "Scatter Plot",
                          xlabel: str = "X",
                          ylabel: str = "Y") -> bytes:
        """Create a scatter plot and return as bytes."""
        plt.figure(figsize=(10, 8))
        plt.scatter(x_values, y_values, alpha=0.6, s=50)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add trend line
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        plt.plot(x_values, p(x_values), "r--", alpha=0.8, label='Trend line')
        plt.legend()
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer.getvalue()
    
    @staticmethod
    def create_box_plot(data_dict: Dict[str, List[float]], 
                       title: str = "Box Plot",
                       ylabel: str = "Value") -> bytes:
        """Create a box plot and return as bytes."""
        plt.figure(figsize=(12, 8))
        
        labels = list(data_dict.keys())
        values = list(data_dict.values())
        
        box_plot = plt.boxplot(values, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer.getvalue()

class ModelInfoExtractor:
    """Extract information from trained models."""
    
    @staticmethod
    def extract_model_info(model_path: str) -> Dict[str, Any]:
        """Extract model information from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            info = {
                'model_path': model_path,
                'file_size_mb': Path(model_path).stat().st_size / (1024 * 1024),
                'architecture': 'DeepDTA-Pro',
                'framework': 'PyTorch'
            }
            
            # Extract training info if available
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    info['training_epoch'] = checkpoint['epoch']
                
                if 'train_loss' in checkpoint:
                    info['final_train_loss'] = checkpoint['train_loss']
                
                if 'val_loss' in checkpoint:
                    info['final_val_loss'] = checkpoint['val_loss']
                
                if 'config' in checkpoint:
                    info['model_config'] = checkpoint['config']
                
                if 'model_state_dict' in checkpoint:
                    # Count parameters
                    model_state = checkpoint['model_state_dict']
                    total_params = sum(p.numel() for p in model_state.values())
                    info['total_parameters'] = total_params
                    info['total_parameters_formatted'] = f"{total_params:,}"
                    
                    # Model size estimation
                    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
                    info['parameter_size_mb'] = param_size_mb
            
            return info
            
        except Exception as e:
            logger.error(f"Error extracting model info: {str(e)}")
            return {
                'model_path': model_path,
                'error': str(e)
            }

class SessionManager:
    """Manage user session data and state."""
    
    def __init__(self):
        self.session_data = {}
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Save a prediction and return session ID."""
        session_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        self.session_data[session_id] = {
            'timestamp': datetime.now(),
            'type': 'single_prediction',
            'data': prediction_data
        }
        
        return session_id
    
    def save_batch_results(self, batch_results: List[Dict[str, Any]]) -> str:
        """Save batch results and return session ID."""
        session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        self.session_data[session_id] = {
            'timestamp': datetime.now(),
            'type': 'batch_prediction',
            'data': batch_results,
            'summary': {
                'total_predictions': len(batch_results),
                'successful_predictions': len([r for r in batch_results if 'error' not in r]),
                'mean_affinity': np.mean([r.get('predicted_affinity', 0) for r in batch_results if 'predicted_affinity' in r])
            }
        }
        
        return session_id
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data by ID."""
        return self.session_data.get(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with summary information."""
        sessions = []
        
        for session_id, session_info in self.session_data.items():
            sessions.append({
                'session_id': session_id,
                'timestamp': session_info['timestamp'],
                'type': session_info['type'],
                'summary': session_info.get('summary', {})
            })
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)

# Utility functions
def format_large_number(num: int) -> str:
    """Format large numbers with appropriate suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)

def calculate_model_size(num_parameters: int) -> str:
    """Calculate and format model size in MB."""
    size_mb = num_parameters * 4 / (1024 * 1024)  # Assuming float32
    return f"{size_mb:.1f} MB"

def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def create_download_link(data: bytes, filename: str, text: str = "Download") -> str:
    """Create a download link for binary data."""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Example usage and testing
if __name__ == "__main__":
    print("Testing web interface utilities...")
    
    # Test data validation
    validator = DataValidator()
    
    # Test SMILES validation
    test_smiles = ["CCO", "invalid_smiles", "C1=CC=CC=C1"]
    for smiles in test_smiles:
        is_valid, error = validator.validate_smiles(smiles)
        print(f"SMILES '{smiles}': {'Valid' if is_valid else f'Invalid - {error}'}")
    
    # Test protein validation
    test_proteins = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWYX", "ACG"]
    for protein in test_proteins:
        is_valid, error = validator.validate_protein_sequence(protein)
        print(f"Protein '{protein}': {'Valid' if is_valid else f'Invalid - {error}'}")
    
    # Test results exporter
    test_results = [
        {'drug_smiles': 'CCO', 'protein_sequence': 'ACDEFG', 'predicted_affinity': 7.5, 'confidence': 0.85},
        {'drug_smiles': 'CCN', 'protein_sequence': 'GHIJKL', 'predicted_affinity': 6.2, 'confidence': 0.72}
    ]
    
    exporter = ResultsExporter()
    csv_content = exporter.to_csv(test_results)
    print(f"CSV export: {len(csv_content)} characters")
    
    json_content = exporter.to_json(test_results)
    print(f"JSON export: {len(json_content)} characters")
    
    print("Web interface utilities test completed successfully!")