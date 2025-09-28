"""
Web Interface Module for DeepDTA-Pro
Streamlit-based interactive web application for drug discovery.
"""

from .app import DeepDTAProApp, main
from .utils import (
    ResultsExporter, 
    DataValidator, 
    VisualizationHelper, 
    ModelInfoExtractor, 
    SessionManager,
    format_large_number,
    calculate_model_size,
    truncate_text,
    create_download_link
)
from .config import (
    get_config,
    update_config,
    validate_config,
    ALL_CONFIGS,
    APP_CONFIG,
    MODEL_CONFIG,
    DATA_CONFIG,
    VIZ_CONFIG,
    PREDICTION_CONFIG,
    INTERPRETABILITY_CONFIG,
    UI_CONFIG,
    PERFORMANCE_CONFIG,
    SECURITY_CONFIG,
    ERROR_CONFIG,
    FEATURE_FLAGS,
    DEPENDENCIES_CONFIG
)

__all__ = [
    # Main app
    'DeepDTAProApp',
    'main',
    
    # Utilities
    'ResultsExporter',
    'DataValidator', 
    'VisualizationHelper',
    'ModelInfoExtractor',
    'SessionManager',
    'format_large_number',
    'calculate_model_size',
    'truncate_text',
    'create_download_link',
    
    # Configuration
    'get_config',
    'update_config',
    'validate_config',
    'ALL_CONFIGS',
    'APP_CONFIG',
    'MODEL_CONFIG',
    'DATA_CONFIG',
    'VIZ_CONFIG',
    'PREDICTION_CONFIG',
    'INTERPRETABILITY_CONFIG',
    'UI_CONFIG',
    'PERFORMANCE_CONFIG',
    'SECURITY_CONFIG',
    'ERROR_CONFIG',
    'FEATURE_FLAGS',
    'DEPENDENCIES_CONFIG'
]

__version__ = '1.0.0'

# Module information
MODULE_INFO = {
    'name': 'DeepDTA-Pro Web Interface',
    'version': __version__,
    'description': 'Interactive web application for drug-target binding affinity prediction',
    'framework': 'Streamlit',
    'features': [
        'Single drug-protein prediction',
        'Batch prediction processing',
        'Model interpretation and visualization',
        'Results export and analysis',
        'Interactive molecular visualization',
        'Attention mechanism visualization',
        'SHAP-based feature importance',
        'Comprehensive analytics dashboard'
    ],
    'requirements': [
        'streamlit>=1.10.0',
        'torch>=1.9.0',
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0'
    ],
    'optional_requirements': [
        'rdkit-pypi',
        'plotly>=5.0.0',
        'shap>=0.40.0',
        'py3Dmol',
        'openpyxl'
    ]
}

def get_module_info():
    """Get information about the web interface module."""
    return MODULE_INFO.copy()

def check_dependencies():
    """Check if all required and optional dependencies are available."""
    import importlib
    
    status = {
        'required': {},
        'optional': {},
        'missing_required': [],
        'missing_optional': []
    }
    
    # Check required dependencies
    required_packages = [
        'streamlit', 'torch', 'numpy', 'pandas', 
        'matplotlib', 'seaborn'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            status['required'][package] = True
        except ImportError:
            status['required'][package] = False
            status['missing_required'].append(package)
    
    # Check optional dependencies
    optional_packages = [
        'rdkit', 'plotly', 'shap', 'py3Dmol', 'openpyxl'
    ]
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            status['optional'][package] = True
        except ImportError:
            status['optional'][package] = False
            status['missing_optional'].append(package)
    
    return status

def print_dependency_status():
    """Print the status of all dependencies."""
    status = check_dependencies()
    
    print("DeepDTA-Pro Web Interface - Dependency Status")
    print("=" * 50)
    
    print("\nRequired Dependencies:")
    for package, available in status['required'].items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {package}")
    
    print("\nOptional Dependencies:")
    for package, available in status['optional'].items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {package}")
    
    if status['missing_required']:
        print(f"\n⚠️  Missing required dependencies: {', '.join(status['missing_required'])}")
        print("   Install with: pip install " + " ".join(status['missing_required']))
    
    if status['missing_optional']:
        print(f"\n💡 Missing optional dependencies: {', '.join(status['missing_optional'])}")
        print("   Some features may be limited. Install with:")
        for package in status['missing_optional']:
            if package == 'rdkit':
                print("   pip install rdkit-pypi")
            else:
                print(f"   pip install {package}")
    
    if not status['missing_required']:
        print("\n✅ All required dependencies are available!")

# Startup checks
def run_startup_checks():
    """Run startup checks for the web interface."""
    print("Running DeepDTA-Pro Web Interface startup checks...")
    
    # Check dependencies
    dep_status = check_dependencies()
    if dep_status['missing_required']:
        print(f"❌ Missing required dependencies: {', '.join(dep_status['missing_required'])}")
        return False
    
    # Validate configuration
    config_issues = validate_config()
    if config_issues:
        print("⚠️  Configuration issues found:")
        for issue in config_issues:
            print(f"   - {issue}")
    
    # Check directories
    from .config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, TEMP_DIR
    
    directories = {
        'Data': DATA_DIR,
        'Models': MODELS_DIR,
        'Outputs': OUTPUTS_DIR,
        'Temp': TEMP_DIR
    }
    
    for name, path in directories.items():
        if path.exists():
            print(f"✓ {name} directory: {path}")
        else:
            print(f"⚠️  {name} directory created: {path}")
            path.mkdir(parents=True, exist_ok=True)
    
    print("Startup checks completed!")
    return True

# Example usage
if __name__ == "__main__":
    print_dependency_status()
    print()
    run_startup_checks()