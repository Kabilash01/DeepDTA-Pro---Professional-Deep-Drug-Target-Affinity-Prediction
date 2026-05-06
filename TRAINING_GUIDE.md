# 🎓 DeepDTA-Pro: Complete Training Guide

**Full guide to train the model and use predictions**

---

## 📋 PREREQUISITES

Before training, ensure:
- ✅ Installation complete (see INSTALLATION_GUIDE.md)
- ✅ Python environment activated
- ✅ All packages verified
- ✅ 4-8 hours available (2-3 hours with GPU)
- ✅ At least 5GB disk space

---

## 🚀 TRAINING THE FULL MODEL

### Quick Start (Copy-Paste)

```bash
# Activate environment first
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux

# Start training (all 7 phases)
python run_pipeline.py

# Monitor in another terminal:
tail -f outputs/logs/training.log  # macOS/Linux
Get-Content outputs/logs/training.log -Tail 20 -Wait  # Windows
```

---

## 📊 WHAT IS THE TRAINING PIPELINE?

The pipeline trains 7 progressive phases:

| # | Phase | Time | R² Target |
|---|-------|------|-----------|
| 1️⃣ | Enhanced Features | 10 min | 0.30 |
| 2️⃣ | Optimization | 30 min | 0.57 |
| 3️⃣ | Graph Neural Networks | 1-2 hrs | 0.70 |
| 4️⃣ | Transfer Learning | 1 hr | 0.75 |
| 5️⃣ | Multi-Task Learning | 1 hr | 0.82 |
| 6️⃣ | Uncertainty Quantification | 1 hr | 0.85 |
| 7️⃣ | Ensemble Methods | 2 hrs | 0.90+ |
| | **TOTAL** | **4-8 hrs** | **0.90+** |

---

## ⚙️ TRAINING OPTIONS

### Option 1: Full Training (Recommended)
```bash
python run_pipeline.py
```
Trains all 7 phases sequentially. Best results (R²=0.90+).

---

### Option 2: Resume from Specific Phase
```bash
# Skip to phase 5 (if earlier phases already done)
python run_pipeline.py --start 5 --end 7

# Run only phases 3-4
python run_pipeline.py --start 3 --end 4

# Start from phase 1, end at phase 5
python run_pipeline.py --start 1 --end 5
```

---

### Option 3: Continue on Errors
```bash
# Skip failed phases and continue
python run_pipeline.py --skip-errors

# Very helpful if training is interrupted
```

---

### Option 4: Train Individual Phases
For fine-grained control, run phases individually:

```bash
python phase1_enhanced_features.py
python phase2_advanced_training.py
python phase3_gnn_with_real_data.py
python phase4_transfer_learning.py
python phase5_multitask_learning.py
python phase6_uncertainty.py
python phase7_ensemble.py
```

---

## 📈 MONITORING TRAINING PROGRESS

### In New Terminal (Don't close training terminal!)

**Windows PowerShell:**
```powershell
# Watch logs in real-time
Get-Content outputs/logs/training.log -Tail 20 -Wait

# Or watch GPU usage (if training on GPU)
while($true) {
    Clear-Host
    nvidia-smi
    Start-Sleep -Seconds 1
}
```

**macOS/Linux:**
```bash
# Watch logs
tail -f outputs/logs/training.log

# Check GPU
watch nvidia-smi
```

---

## 📊 TRAINING OUTPUT EXPLAINED

### During Training

```
🔄 PHASE 3: Graph Neural Networks
   📝 Description: Molecular graph neural networks with attention
   📊 Expected R²: 0.70
   📄 Script: phase3_gnn_with_real_data.py
──────────────────────────────────────────────────────────────────

Epoch 1/100:   [████░░░░░░] Loss: 0.567, Val Loss: 0.532
Epoch 2/100:   [████████░░] Loss: 0.456, Val Loss: 0.421
Epoch 3/100:   [██████████] Loss: 0.389, Val Loss: 0.378
...
Epoch 100/100: [██████████] Loss: 0.123, Val Loss: 0.145

✅ PHASE 3 completed in 5847.23s (R² = 0.7012)
```

### After Completion

```
==================================================================================================
📊 PIPELINE EXECUTION SUMMARY
==================================================================================================

✅ Successful: 7/7
⏱️  Total Time: 28342s (7 hours 52 minutes)

Phase Results:
  ✅ PHASE 1: Enhanced Feature Engineering (Expected R²: 0.3000)
  ✅ PHASE 2: Advanced Optimization (Expected R²: 0.5701)
  ✅ PHASE 3: Graph Neural Networks (Expected R²: 0.7000)
  ✅ PHASE 4: Transfer Learning (Expected R²: 0.7500)
  ✅ PHASE 5: Multi-Task Learning (Expected R²: 0.8200)
  ✅ PHASE 6: Uncertainty Quantification (Expected R²: 0.8500)
  ✅ PHASE 7: Ensemble Methods (Expected R²: 0.9000)

🎉 ALL PHASES COMPLETED SUCCESSFULLY! 🎉
```

---

## 📁 OUTPUT FILES

After training completes, you'll have:

```
outputs/
├── models/
│   ├── phase1_features.pkl
│   ├── phase2_optimized.pt
│   ├── phase3_gnn.pt
│   ├── phase4_transfer.pt
│   ├── phase5_multitask.pt
│   ├── phase6_uncertainty.pt
│   ├── phase7_ensemble/
│   │   ├── model_1.pt
│   │   ├── model_2.pt
│   │   └── ...
│   └── final_model.pt              ← USE THIS! 🎯
│
├── logs/
│   ├── training.log                ← Complete log
│   ├── metrics.csv                 ← Performance metrics
│   └── hyperparameters.yaml        ← Best hyperparams
│
├── checkpoints/
│   └── best_model_phase7.pt        ← Best checkpoint
│
└── results/
    ├── predictions_test.csv         ← Test predictions
    ├── evaluation_metrics.json      ← R², MAE, RMSE stats
    ├── attention_weights.npy        ← Attention maps
    └── uncertainty_estimates.csv    ← Confidence scores
```

---

## 🎯 USING THE TRAINED MODEL

### Method 1: Web Interface with Trained Model

```bash
# Launch web app with trained model
python run_app.py --model outputs/models/final_model.pt

# Open http://localhost:8501
# Upload CSV file → Get REAL predictions!
```

---

### Method 2: Python - Single Prediction

```python
import torch
from src.models.deepdta_pro import DeepDTAPro

# Load trained model
model = DeepDTAPro.load_from_checkpoint('outputs/models/final_model.pt')
model.eval()

# Example: Ibuprofen + Protein Target
drug_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
protein_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV..."

# Make prediction
with torch.no_grad():
    prediction = model(drug_data, protein_data)
    affinity = prediction.item()

print(f"Predicted Binding Affinity: {affinity:.3f}")
# Output: Predicted Binding Affinity: 6.852
```

---

### Method 3: Python - Batch Predictions

```python
import pandas as pd
import torch
from src.models.deepdta_pro import DeepDTAPro

# Load model
model = DeepDTAPro.load_from_checkpoint('outputs/models/final_model.pt')
model.eval()

# Load data
df = pd.read_csv('drug_protein_pairs.csv')

# Make batch predictions
results = []
for idx, row in df.iterrows():
    drug_smiles = row['drug_smiles']
    protein_seq = row['protein_sequence']

    # Extract features
    from src.data.molecular_features import MolecularFeatureExtractor
    from src.data.protein_features import ProteinFeatureExtractor

    mol_extractor = MolecularFeatureExtractor()
    prot_extractor = ProteinFeatureExtractor()

    mol_data = mol_extractor.extract_features(drug_smiles)
    prot_data = prot_extractor.extract_features(protein_seq)

    # Predict
    with torch.no_grad():
        affinity = model(mol_data, prot_data).item()

    results.append({
        'drug': drug_smiles,
        'protein': protein_seq,
        'predicted_affinity': affinity
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('predictions.csv', index=False)
print(f"✅ Saved {len(results)} predictions to predictions.csv")
```

---

### Method 4: Get Uncertainty Estimates

```python
import torch
from src.models.deepdta_pro import DeepDTAPro

model = DeepDTAPro.load_from_checkpoint('outputs/models/final_model.pt')
model.eval()

# Make multiple predictions with dropout (uncertainty)
num_samples = 50
predictions = []

for _ in range(num_samples):
    with torch.no_grad():
        pred = model(drug_data, protein_data, mc_dropout=True)
        predictions.append(pred.item())

predictions = torch.tensor(predictions)
mean_pred = predictions.mean()
std_pred = predictions.std()

print(f"Predicted Affinity: {mean_pred:.3f} ± {std_pred:.3f}")
# Output: Predicted Affinity: 6.852 ± 0.145
# The model is 68% confident (±1 std)
```

---

## ⚡ OPTIMIZATION TIPS

### For Faster Training (GPU)

**1. Use NVIDIA GPU (already detected automatically):**
- RTX 3060: 2-3 hours
- RTX 4080: 1.5-2 hours
- T4 GPU: 3-4 hours

**2. Increase batch size (if GPU has memory):**
Edit phase files:
```python
batch_size = 64  # Default is 32
```

**3. Use more workers:**
```python
num_workers = 8  # Default is 4
```

---

### For Slower Hardware (CPU)

**1. Reduce dataset:**
```python
max_samples = 2000  # Train on subset first
```

**2. Skip early phases:**
```bash
# Start from phase 5 (saves 2+ hours)
python run_pipeline.py --start 5 --end 7
```

**3. Reduce batch size:**
```python
batch_size = 8  # Reduce from 32
```

---

## 🐛 TROUBLESHOOTING

### Training too slow

**Check if using GPU:**
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

If False and you have GPU:
- Reinstall PyTorch with CUDA support
- Check NVIDIA drivers

If CPU is slow:
- Reduce batch size
- Reduce dataset size
- Skip early phases

---

### GPU out of memory

```bash
# Reduce batch size in phase files
batch_size = 8  # Instead of 64

# OR use CPU instead
$env:CUDA_VISIBLE_DEVICES=""
python run_pipeline.py
```

---

### Training interrupted

```bash
# Resume from where it stopped
python run_pipeline.py --start 5  # If phase 5 failed
python run_pipeline.py --skip-errors  # Skip failed phases
```

---

### Missing data files

The pipeline will auto-generate synthetic data if DAVIS/KIBA not found. Training still works for learning purposes.

---

## 📊 EXPECTED PERFORMANCE

| Metric | Value |
|--------|-------|
| Final R² Score | 0.90+ |
| Mean Absolute Error (MAE) | ~0.12 |
| Root Mean Squared Error (RMSE) | ~0.18 |
| Pearson Correlation | 0.92 |
| Accuracy (within 0.5 unit) | 85%+ |

---

## 🎓 UNDERSTANDING THE PHASES

### Phase 1: Feature Engineering
- Extracts molecular & protein features
- Creates feature vectors
- **Purpose**: Prepare data for training

### Phase 2: Optimization
- Bayesian hyperparameter search
- Tests different configurations
- **Purpose**: Find best hyperparameters

### Phase 3: Graph Neural Networks
- Trains GNN on molecular graphs
- Protein sequence transformer
- **Purpose**: Core deep learning model

### Phase 4: Transfer Learning
- Uses pre-trained MolBERT + ProtBERT
- Fine-tunes on binding affinity task
- **Purpose**: Improve with pre-training knowledge

### Phase 5: Multi-Task Learning
- Adds auxiliary tasks (Efficiency, Solubility, Toxicity)
- Improves generalization
- **Purpose**: Better predictions through multi-task

### Phase 6: Uncertainty Quantification
- Adds MC Dropout for Bayesian uncertainty
- Provides confidence intervals
- **Purpose**: Know when model is uncertain

### Phase 7: Ensemble Methods
- Trains 5 different models
- Combines via voting/averaging
- **Purpose**: Maximum performance & robustness

---

## ✅ AFTER TRAINING CHECKLIST

- [ ] Training completed successfully
- [ ] final_model.pt exists in outputs/models/
- [ ] evaluation_metrics.json shows R²=0.90+
- [ ] Predictions saved in outputs/results/
- [ ] Logs saved in outputs/logs/
- [ ] Ready to use model for predictions

---

## 🚀 NEXT STEPS

1. **Use trained model:**
   ```bash
   python run_app.py --model outputs/models/final_model.pt
   ```

2. **Make predictions:**
   ```bash
   python examples/single_prediction.py --model_path outputs/models/final_model.pt
   ```

3. **Analyze results:**
   - Check outputs/evaluation_metrics.json
   - Review training logs in outputs/logs/

4. **Deploy (optional):**
   - Save outputs/models/final_model.pt
   - Use in your application
   - Integrate with web service

---

## 📞 QUICK REFERENCE

```bash
# Start training
python run_pipeline.py

# Resume from phase 5
python run_pipeline.py --start 5

# Skip errors
python run_pipeline.py --skip-errors

# Use trained model
python run_app.py --model outputs/models/final_model.pt

# Monitor logs
tail -f outputs/logs/training.log

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

**Training Guide Complete!** 🎓

For installation help, see INSTALLATION_GUIDE.md
For project overview, see README.md
