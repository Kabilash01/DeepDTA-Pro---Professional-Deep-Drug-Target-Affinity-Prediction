"""
Quick Test Script for Real Training
Run this to test the training pipeline
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Test the real training pipeline"""
    print("🚀 Testing DeepDTA-Pro Real Training Pipeline")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    
    print(f"📂 Project directory: {project_dir}")
    print("🎯 Starting training with 5 epochs for testing...")
    
    try:
        # Run the real trainer
        cmd = [
            sys.executable, 
            "real_trainer.py",
            "--epochs", "5",
            "--batch_size", "8", 
            "--lr", "0.001",
            "--dataset", "davis"
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("📊 Training output:")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print("Error output:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running training: {e}")

if __name__ == "__main__":
    main()