#!/usr/bin/env python3
"""
List available trained model checkpoints.
"""
import os
import sys
from pathlib import Path
import torch

# Fix encoding on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def list_checkpoints():
    ckpt_dir = Path(__file__).parent / 'models' / 'checkpoints'

    if not ckpt_dir.exists():
        print("[ERROR] No checkpoints directory found at:", ckpt_dir)
        return

    files = sorted(ckpt_dir.glob('*.pth'))

    if not files:
        print("[ERROR] No checkpoint files found in:", ckpt_dir)
        return

    print("\n" + "="*80)
    print("AVAILABLE TRAINED MODELS")
    print("="*80 + "\n")

    for checkpoint_path in files:
        size_mb = checkpoint_path.stat().st_size / (1024**2)

        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')

            # Extract metadata
            metrics = ckpt.get('test_metrics', {})
            best_r2 = ckpt.get('best_val_r2', metrics.get('r2'))
            test_r2 = metrics.get('r2')
            test_rmse = metrics.get('rmse')
            test_mae = metrics.get('mae')
            test_ci = metrics.get('ci')
            epoch = ckpt.get('epoch')

            print(f"[MODEL] {checkpoint_path.name}")
            print(f"   Size: {size_mb:.1f} MB")
            if best_r2 is not None:
                print(f"   Best Val R²: {best_r2:.4f}")
            if test_r2 is not None:
                print(f"   Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | CI: {test_ci:.4f}")
            if epoch is not None:
                print(f"   Trained for {epoch} epochs")
            print()
        except Exception as e:
            print(f"[WARNING] {checkpoint_path.name} - Error loading: {e}\n")

    print("="*80 + "\n")

if __name__ == "__main__":
    list_checkpoints()
