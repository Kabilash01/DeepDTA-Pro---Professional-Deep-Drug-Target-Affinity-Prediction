# 🚀 Quick Start Execution Guide

## For Impatient Users (TL;DR)

### 1-Minute Setup
```bash
# Create environment
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1

# Install everything
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric "numpy<2" pandas streamlit rdkit-pypi pytorch-lightning

# Run!
python run_app.py
```

Then open: **http://localhost:8501**

---

## Quick Commands Reference

### Installation
```bash
# Full setup (Windows PowerShell)
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Examples
```bash
# Single prediction
python examples/single_prediction.py

# Batch processing
python examples/batch_processing.py

# Train model
python examples/train_model.py

# Evaluate model
python examples/evaluate_model.py
```

### Web Interface
```bash
# Standard launch
python run_app.py

# Custom port
python run_app.py --port 8502

# Debug mode
python run_app.py --debug
```

### Pipeline
```bash
# Full training
python run_pipeline.py

# With custom config
python run_pipeline.py --config configs/custom_config.yaml
```

---

## Verification Checklist

✅ = Ready to use
❌ = Needs attention

Run this after setup:
```bash
# Quick verification script
python -c "
import sys
checks = [
    ('PyTorch', lambda: __import__('torch')),
    ('PyTorch Geometric', lambda: __import__('torch_geometric')),
    ('Pandas', lambda: __import__('pandas')),
    ('NumPy', lambda: __import__('numpy')),
    ('RDKit', lambda: __import__('rdkit')),
    ('Streamlit', lambda: __import__('streamlit')),
]

for name, check in checks:
    try:
        check()
        print(f'✅ {name}')
    except Exception as e:
        print(f'❌ {name}: {e}')
"
```

---

## Expected Output Examples

### Single Prediction
```
🧪 DeepDTA-Pro Single Prediction
================================

Drug SMILES: CC(=O)Oc1ccccc1C(=O)O
Protein Seq: MKTAYIAKQRQISFVK...
Predicted Affinity: 7.35
Binding Strength: MODERATE
```

### Batch Processing
```
📊 Processing batch...
[████████░░] 80%
✅ Successfully predicted 100 pairs
📈 Average affinity: 6.82
💾 Results saved to: outputs/batch_results.csv
```

### Training
```
🏋️  Training Model...
Epoch 1/100: [████░░░░░░] Loss: 0.456
Epoch 2/100: [██████░░░░] Loss: 0.389
...
✅ Training complete!
Best model saved: models/best_model.pth
```

---

## Common Issues & Quick Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: rdkit` | `pip install rdkit-pypi` |
| `_ARRAY_API not found` | `pip install "numpy<2"` |
| `Port 8501 in use` | `python run_app.py --port 8502` |
| `CUDA out of memory` | Reduce batch_size in config |
| `No module torch_geometric` | `pip install torch-geometric` |

---

## File Locations

| What | Where |
|------|-------|
| Main web app | `run_app.py` |
| Examples | `examples/` |
| Models | `models/` |
| Configs | `configs/` |
| Results | `outputs/` |
| Docs | `docs/` |

---

## Next Steps

1. Run web interface: `python run_app.py`
2. Try example predictions
3. Train on your data
4. Export results
5. Explore visualizations

**Happy predicting!** 🎯
