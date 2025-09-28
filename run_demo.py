#!/usr/bin/env python3
"""
DeepDTA-Pro Demo Launcher
Single-command demo launcher for the complete drug discovery system.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'torch', 'torch_geometric', 'rdkit', 'streamlit', 
        'pandas', 'numpy', 'plotly', 'shap'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install dependencies: pip install -r requirements.txt")
        return False
    
    return True

def prepare_demo_data():
    """Prepare demo data if not already processed."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.data.davis_processor import DavisProcessor
        from src.data.kiba_processor import KIBAProcessor
        
        # Check if processed data exists
        processed_dir = Path("data/processed")
        if not (processed_dir / "davis_data.pkl").exists():
            logger.info("Processing Davis dataset...")
            davis_processor = DavisProcessor()
            davis_processor.process_dataset()
        
        if not (processed_dir / "kiba_data.pkl").exists():
            logger.info("Processing KIBA dataset...")
            kiba_processor = KIBAProcessor()
            kiba_processor.process_dataset()
        
        logger.info("Demo data preparation completed.")
        return True
        
    except Exception as e:
        logger.error(f"Error preparing demo data: {e}")
        return False

def launch_web_interface(port=8501, host="localhost"):
    """Launch the Streamlit web interface."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Launching web interface at http://{host}:{port}")
        
        # Launch Streamlit app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error launching web interface: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")
        return True

def run_quick_test():
    """Run a quick prediction test to verify system functionality."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.inference.predictor import DrugTargetPredictor
        
        logger.info("Running quick system test...")
        
        # Initialize predictor
        predictor = DrugTargetPredictor()
        
        # Test with example molecule and protein
        test_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"  # Imatinib
        test_protein = "MGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVECLEQGGMIDRPGKISERGFHLVLKAPALVVNHFTTHYSDAGVKVIVECENGGMVTGRVRAGTYHYLLLLPNSGTRYIIGVPYLRLGQGRPDMGNSQPGKLSKDQLAILVNVPSGKERLVFQAIAECFQSHFVVVDNQGVLHNELKCVELARNTLNKAGCDMMLKVQHVRDRVLDPKLKSGGQTLGGCVHTGSRILIQEQMQFAGPGQYVMEKNRCPRYQLLLDTLSFNQFEPQCSFTKDIVHFQDKNHYQTWKQMVLHVNPSALGSQVMQQRSDNELRSGQYFHHYSRNYGLCATGWPVTLMQIPVPYKLEYLVIPQPGKTLYKIDNGMYGTGQLQQVLNDLELLLQRTILASGYIQLPNDKDILELLVQYERNNQAVQNFLMQQYVEALFAEYFHRFGTTKHLQQLIRPKSSQERALGTRVMWSRTVCKQLRLKDNKRRQQQGLNVKASDSGVEFKQGAFLDVQHLYRVRLLGLDPGDLDLSWLMVPPPTGAEVVVQTLTFPASVPVEEAFSCLARRQTSDQMSQLLLMLLLNAGDYQLLQVEGVPACVQCGLLPKSLSYQVFQDPQDLGTVTDWQSFSLSVLYKGLKGPCSRRTLCNQPHRSFLLTYQHGGKRLAHILELLCEAKEGFRMERLRPLVEVDPQTLLPPLPLDDFKTCRLLLDGGQEENIAPDLVLDADGDNFGFFDGEILVPPAVQNLLGFMRQMRKPSRYGCQGTVTTTTGTAEVSKALDDALESFMRQKCPTQIDPSGITLRQEFRTKIMLEQEGTPEESEEEEEEEEEKRMPEYTALLEFAKAGDRVKFWEKDIKGPQGEYEMGKGSPRLPDRFNMQGAEQMADQQKDKQDNPDYELVKRDFSPEKLELQRQTMTELCPELRVQGRKLQSLAERLGEQMPEEDSEASLQLPPNRGVYIFVTEGKTPGQGAYLDAWWQHFVSRIASSSEVSTQTLQSSRRQLSLEEQAQDHSQQELQRMPGSPPSEGTSPLMSRMPTFDKDMYKDLEQHKPQVPSLHHNSNRQRYHLAAGKDVEQLLQRYCSADKASPSQGQGQGQGQGQGQGQGQGQGEESVSLQ"
        
        # Make prediction
        result = predictor.predict(test_smiles, test_protein)
        
        logger.info(f"Quick test completed successfully!")
        logger.info(f"Predicted binding affinity: {result['affinity']:.3f}")
        logger.info(f"Confidence interval: [{result['confidence_lower']:.3f}, {result['confidence_upper']:.3f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False

def main():
    """Main demo launcher function."""
    parser = argparse.ArgumentParser(
        description="DeepDTA-Pro Demo Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        choices=["web", "test", "prepare"], 
        default="web",
        help="Demo mode: web interface, quick test, or data preparation"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port for web interface"
    )
    
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host for web interface"
    )
    
    parser.add_argument(
        "--skip-checks", 
        action="store_true",
        help="Skip dependency and data checks"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting DeepDTA-Pro Demo...")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Check dependencies unless skipped
    if not args.skip_checks:
        logger.info("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
    
    # Execute based on mode
    if args.mode == "prepare":
        logger.info("Preparing demo data...")
        if prepare_demo_data():
            logger.info("Data preparation completed successfully!")
        else:
            logger.error("Data preparation failed!")
            sys.exit(1)
    
    elif args.mode == "test":
        logger.info("Running quick test...")
        if not args.skip_checks:
            prepare_demo_data()
        
        if run_quick_test():
            logger.info("Quick test passed!")
        else:
            logger.error("Quick test failed!")
            sys.exit(1)
    
    elif args.mode == "web":
        logger.info("Launching web interface...")
        
        # Prepare data if needed
        if not args.skip_checks:
            prepare_demo_data()
        
        # Launch web interface
        launch_web_interface(port=args.port, host=args.host)
    
    logger.info("Demo completed.")

if __name__ == "__main__":
    main()