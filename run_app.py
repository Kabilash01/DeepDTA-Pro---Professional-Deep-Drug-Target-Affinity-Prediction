#!/usr/bin/env python3
"""
DeepDTA-Pro Web Interface Runner
Script to launch the Streamlit web application.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_streamlit_installation():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available."""
    print("Streamlit not found. Installing...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlit'])
        print("Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install Streamlit. Please install manually:")
        print("pip install streamlit")
        return False

def run_app(port: int = 8501, 
           host: str = "localhost",
           model_path: str = None,
           debug: bool = False,
           auto_open: bool = True):
    """
    Run the DeepDTA-Pro web application.
    
    Args:
        port: Port number for the web server
        host: Host address
        model_path: Path to the trained model
        debug: Enable debug mode
        auto_open: Automatically open browser
    """
    # Check Streamlit installation
    if not check_streamlit_installation():
        if not install_streamlit():
            return False
    
    # Set environment variables
    if model_path:
        os.environ['DEEPDTA_MODEL_PATH'] = str(Path(model_path).absolute())
    
    if debug:
        os.environ['DEBUG'] = 'true'
        os.environ['STREAMLIT_LOGGER_LEVEL'] = 'debug'
    
    # Streamlit app path
    app_path = src_dir / 'web_interface' / 'app.py'
    
    if not app_path.exists():
        logger.error(f"App file not found: {app_path}")
        return False
    
    # Build Streamlit command
    cmd = [
        'streamlit', 'run', str(app_path),
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', str(not auto_open).lower(),
        '--server.runOnSave', str(debug).lower(),
        '--browser.gatherUsageStats', 'false'
    ]
    
    # Additional Streamlit config
    if debug:
        cmd.extend(['--logger.level', 'debug'])
    
    print(f"Starting DeepDTA-Pro Web Interface...")
    print(f"URL: http://{host}:{port}")
    if model_path:
        print(f"Model: {model_path}")
    print("-" * 50)
    
    try:
        # Run Streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        print("\nShutting down...")
        return True
    
    return True

def main():
    """Main entry point for the runner script."""
    parser = argparse.ArgumentParser(
        description="DeepDTA-Pro Web Interface Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                           # Run with default settings
  python run_app.py --port 8080               # Run on port 8080
  python run_app.py --model path/to/model.pth # Load specific model
  python run_app.py --debug                   # Run in debug mode
  python run_app.py --host 0.0.0.0 --port 8501 # Allow external access
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='Port number for the web server (default: 8501)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address (default: localhost, use 0.0.0.0 for external access)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to the trained DeepDTA-Pro model file'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--no-open',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install missing dependencies'
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        try:
            from web_interface import check_dependencies, print_dependency_status
            print_dependency_status()
        except ImportError:
            print("Cannot import web_interface module. Please check your installation.")
        return
    
    # Install dependencies if requested
    if args.install_deps:
        deps_to_install = [
            'streamlit>=1.10.0',
            'torch>=1.9.0',
            'numpy>=1.20.0',
            'pandas>=1.3.0',
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0'
        ]
        
        print("Installing required dependencies...")
        for dep in deps_to_install:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                print(f"✓ Installed {dep}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {dep}")
        
        print("\nOptional dependencies (install if needed):")
        optional_deps = [
            'rdkit-pypi  # For molecular visualization',
            'shap>=0.40.0  # For model interpretation',
            'py3Dmol  # For 3D molecular visualization',
            'openpyxl  # For Excel export'
        ]
        
        for dep in optional_deps:
            print(f"  pip install {dep}")
        
        return
    
    # Validate model path if provided
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return
        if not model_path.suffix in ['.pth', '.pt', '.pkl']:
            logger.warning(f"Unusual model file extension: {model_path.suffix}")
    
    # Run the application
    success = run_app(
        port=args.port,
        host=args.host,
        model_path=args.model,
        debug=args.debug,
        auto_open=not args.no_open
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()