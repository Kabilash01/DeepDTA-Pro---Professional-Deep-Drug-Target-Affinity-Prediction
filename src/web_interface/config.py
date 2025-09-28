"""
Web Interface Configuration
Configuration settings and constants for the DeepDTA-Pro web interface.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Application settings
APP_CONFIG = {
    'title': 'DeepDTA-Pro: Drug Discovery Platform',
    'description': 'Advanced Drug-Target Binding Affinity Prediction using Graph Neural Networks',
    'version': '1.0.0',
    'author': 'DeepDTA-Pro Team',
    'contact_email': 'contact@deepdta-pro.com',
    'github_url': 'https://github.com/your-repo/deepdta-pro',
    'documentation_url': 'https://deepdta-pro.readthedocs.io'
}

# File paths and directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'
TEMP_DIR = BASE_DIR / 'temp'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'default_model_path': str(MODELS_DIR / 'checkpoints' / 'best_model.pth'),
    'config_filename': 'config.json',
    'supported_formats': ['.pth', '.pt', '.pkl'],
    'max_model_size_mb': 500,
    'device': 'cpu'  # Can be changed to 'cuda' if GPU is available
}

# Data processing configuration
DATA_CONFIG = {
    'max_smiles_length': 500,
    'max_protein_length': 5000,
    'min_protein_length': 10,
    'batch_size_limits': {
        'min': 1,
        'max': 1000,
        'default': 32
    },
    'supported_file_formats': ['.csv', '.xlsx', '.tsv'],
    'max_file_size_mb': 50,
    'required_columns': ['drug_smiles', 'protein_sequence'],
    'optional_columns': ['compound_name', 'target_name', 'compound_id', 'target_id']
}

# Visualization configuration
VIZ_CONFIG = {
    'default_figure_size': (10, 8),
    'default_dpi': 300,
    'color_palettes': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#9467bd'
    },
    'plot_themes': {
        'default': 'plotly_white',
        'dark': 'plotly_dark',
        'minimal': 'simple_white'
    },
    'molecular_viz': {
        'image_size': (300, 300),
        'highlight_color': (1.0, 0.7, 0.7),
        'bond_color': (0.0, 0.0, 0.0),
        'atom_font_size': 12
    }
}

# Prediction configuration
PREDICTION_CONFIG = {
    'confidence_threshold': 0.5,
    'affinity_range': {
        'min': 4.0,
        'max': 12.0,
        'typical_range': (5.0, 10.0)
    },
    'output_precision': 3,
    'batch_processing': {
        'chunk_size': 100,
        'progress_update_frequency': 10,
        'timeout_seconds': 300
    }
}

# Interpretability configuration
INTERPRETABILITY_CONFIG = {
    'attention_visualization': {
        'max_sequence_length': 100,
        'top_attention_positions': 20,
        'color_scheme': 'Reds',
        'figure_size': (12, 8)
    },
    'shap_analysis': {
        'background_samples': 50,
        'max_features_display': 20,
        'explanation_timeout': 60
    },
    'molecular_interpretation': {
        'importance_threshold': 0.1,
        'max_substructures': 10,
        'fragment_methods': ['brics', 'recap', 'morgan']
    }
}

# UI/UX configuration
UI_CONFIG = {
    'theme': {
        'primary_color': '#1f77b4',
        'background_color': '#ffffff',
        'secondary_background_color': '#f0f2f6',
        'text_color': '#262730',
        'font': 'sans serif'
    },
    'layout': {
        'sidebar_width': 300,
        'main_content_padding': '1rem',
        'max_content_width': '1200px'
    },
    'components': {
        'input_height': 150,
        'button_height': 40,
        'metric_card_height': 100,
        'chart_height': 400
    },
    'messages': {
        'loading_model': 'Loading model, please wait...',
        'processing_prediction': 'Processing prediction...',
        'batch_processing': 'Processing batch predictions...',
        'generating_visualization': 'Generating visualization...',
        'exporting_results': 'Exporting results...'
    }
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'caching': {
        'enable_model_cache': True,
        'enable_prediction_cache': False,
        'cache_ttl_seconds': 3600
    },
    'memory_limits': {
        'max_batch_size': 1000,
        'max_concurrent_requests': 10,
        'memory_threshold_mb': 2048
    },
    'optimization': {
        'use_mixed_precision': False,
        'enable_torch_compile': False,
        'num_threads': 4
    }
}

# Security and validation configuration
SECURITY_CONFIG = {
    'input_validation': {
        'sanitize_inputs': True,
        'max_input_length': 10000,
        'allowed_characters': {
            'smiles': set('()[]{}+-=#$:\\/@.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            'protein': set('ACDEFGHIKLMNPQRSTVWY')
        }
    },
    'file_upload': {
        'scan_uploads': True,
        'allowed_extensions': ['.csv', '.xlsx', '.tsv'],
        'max_file_size_mb': 50,
        'quarantine_suspicious_files': True
    },
    'rate_limiting': {
        'enable': False,
        'max_requests_per_minute': 60,
        'max_requests_per_hour': 1000
    }
}

# Error handling configuration
ERROR_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': str(BASE_DIR / 'logs' / 'app.log')
    },
    'error_messages': {
        'model_not_loaded': 'Model not loaded. Please load a model first.',
        'invalid_smiles': 'Invalid SMILES string. Please check the molecular structure.',
        'invalid_protein': 'Invalid protein sequence. Please use single-letter amino acid codes.',
        'prediction_failed': 'Prediction failed. Please check your inputs and try again.',
        'file_too_large': 'File too large. Maximum file size is {max_size} MB.',
        'unsupported_format': 'Unsupported file format. Supported formats: {formats}',
        'batch_processing_failed': 'Batch processing failed. Please check your data format.',
        'export_failed': 'Export failed. Please try again.',
        'visualization_failed': 'Visualization generation failed.',
        'interpretation_failed': 'Model interpretation failed.'
    },
    'user_feedback': {
        'show_detailed_errors': False,
        'enable_error_reporting': True,
        'contact_support_message': 'If the problem persists, please contact support.'
    }
}

# Feature flags
FEATURE_FLAGS = {
    'enable_batch_prediction': True,
    'enable_model_interpretation': True,
    'enable_attention_visualization': True,
    'enable_shap_analysis': True,
    'enable_molecular_interpretation': True,
    'enable_3d_visualization': False,  # Requires py3Dmol
    'enable_advanced_analytics': True,
    'enable_model_comparison': True,
    'enable_api_endpoints': False,
    'enable_user_accounts': False,
    'enable_result_sharing': False,
    'enable_experimental_features': False
}

# External dependencies configuration
DEPENDENCIES_CONFIG = {
    'required': [
        'torch>=1.9.0',
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'streamlit>=1.10.0'
    ],
    'optional': {
        'rdkit': {
            'package': 'rdkit-pypi',
            'features': ['molecular_visualization', 'smiles_validation', 'descriptor_calculation']
        },
        'plotly': {
            'package': 'plotly>=5.0.0',
            'features': ['interactive_plots', 'advanced_visualization']
        },
        'shap': {
            'package': 'shap>=0.40.0',
            'features': ['model_interpretation', 'feature_importance']
        },
        'py3dmol': {
            'package': 'py3Dmol',
            'features': ['3d_molecular_visualization']
        },
        'openpyxl': {
            'package': 'openpyxl',
            'features': ['excel_export']
        }
    },
    'check_availability': True,
    'warn_missing_optional': True
}

# API configuration (for future use)
API_CONFIG = {
    'enable_api': False,
    'host': '0.0.0.0',
    'port': 8000,
    'cors_origins': ['*'],
    'rate_limiting': {
        'requests_per_minute': 100,
        'requests_per_hour': 1000
    },
    'authentication': {
        'enable': False,
        'method': 'api_key',
        'token_expiry_hours': 24
    },
    'documentation': {
        'enable_swagger': True,
        'title': 'DeepDTA-Pro API',
        'description': 'API for drug-target binding affinity prediction'
    }
}

# Development configuration
DEV_CONFIG = {
    'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true',
    'reload_on_change': True,
    'show_warnings': True,
    'profiling': {
        'enable': False,
        'profile_predictions': False,
        'profile_visualizations': False
    },
    'testing': {
        'use_mock_data': False,
        'mock_prediction_delay': 0.5,
        'test_data_path': str(BASE_DIR / 'tests' / 'data')
    }
}

# Export all configurations
ALL_CONFIGS = {
    'app': APP_CONFIG,
    'model': MODEL_CONFIG,
    'data': DATA_CONFIG,
    'visualization': VIZ_CONFIG,
    'prediction': PREDICTION_CONFIG,
    'interpretability': INTERPRETABILITY_CONFIG,
    'ui': UI_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'security': SECURITY_CONFIG,
    'error': ERROR_CONFIG,
    'features': FEATURE_FLAGS,
    'dependencies': DEPENDENCIES_CONFIG,
    'api': API_CONFIG,
    'development': DEV_CONFIG
}

def get_config(config_name: str = None) -> Dict[str, Any]:
    """
    Get configuration by name or all configurations.
    
    Args:
        config_name: Name of the configuration section
        
    Returns:
        Configuration dictionary
    """
    if config_name is None:
        return ALL_CONFIGS
    
    return ALL_CONFIGS.get(config_name, {})

def update_config(config_name: str, updates: Dict[str, Any]) -> bool:
    """
    Update configuration values.
    
    Args:
        config_name: Name of the configuration section
        updates: Dictionary of updates to apply
        
    Returns:
        Success status
    """
    if config_name not in ALL_CONFIGS:
        return False
    
    ALL_CONFIGS[config_name].update(updates)
    return True

def validate_config() -> List[str]:
    """
    Validate configuration settings and return any issues.
    
    Returns:
        List of validation issues
    """
    issues = []
    
    # Validate paths
    if not Path(MODEL_CONFIG['default_model_path']).parent.exists():
        issues.append(f"Model directory does not exist: {Path(MODEL_CONFIG['default_model_path']).parent}")
    
    # Validate data limits
    if DATA_CONFIG['max_smiles_length'] <= 0:
        issues.append("max_smiles_length must be positive")
    
    if DATA_CONFIG['max_protein_length'] <= DATA_CONFIG['min_protein_length']:
        issues.append("max_protein_length must be greater than min_protein_length")
    
    # Validate batch size limits
    batch_limits = DATA_CONFIG['batch_size_limits']
    if batch_limits['min'] >= batch_limits['max']:
        issues.append("Batch size min must be less than max")
    
    if not (batch_limits['min'] <= batch_limits['default'] <= batch_limits['max']):
        issues.append("Default batch size must be within min/max limits")
    
    # Validate performance settings
    if PERFORMANCE_CONFIG['memory_limits']['max_batch_size'] > 10000:
        issues.append("max_batch_size seems too large (>10000)")
    
    return issues

# Example usage
if __name__ == "__main__":
    print("DeepDTA-Pro Web Interface Configuration")
    print("=" * 50)
    
    # Print basic app info
    app_config = get_config('app')
    print(f"Title: {app_config['title']}")
    print(f"Version: {app_config['version']}")
    print(f"Author: {app_config['author']}")
    
    # Validate configuration
    issues = validate_config()
    if issues:
        print(f"\nConfiguration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration validation passed!")
    
    # Show feature flags
    print(f"\nEnabled features:")
    for feature, enabled in FEATURE_FLAGS.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature}")
    
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    
    print("\nConfiguration loaded successfully!")