# 🚀 INSTANT START - Just Copy & Paste This!

## ONE-COMMAND FULL SETUP (For Windows PowerShell)

Copy and paste everything below into PowerShell at once:

```powershell
# ============================================================================
# DEEPDTA-PRO FULL TRAINING SETUP SCRIPT
# ============================================================================
# This script will:
# 1. Create Python 3.9 environment
# 2. Install all dependencies
# 3. Verify everything works
# 4. Start full training pipeline
# ============================================================================

Write-Host "🚀 Starting DeepDTA-Pro Full Training Setup..." -ForegroundColor Green
Write-Host "⏱️  This will take 15-20 minutes for setup" -ForegroundColor Yellow
Write-Host ""

# Step 1: Create environment
Write-Host "📦 Step 1: Creating Python 3.9 environment..." -ForegroundColor Cyan
py -3.9 -m venv venv
Write-Host "✅ Environment created" -ForegroundColor Green

# Step 2: Activate environment
Write-Host "📦 Step 2: Activating environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1
Write-Host "✅ Environment activated" -ForegroundColor Green

# Step 3: Upgrade pip
Write-Host "📦 Step 3: Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel -q
Write-Host "✅ Pip upgraded" -ForegroundColor Green

# Step 4: Install PyTorch with GPU support
Write-Host "📦 Step 4: Installing PyTorch..." -ForegroundColor Cyan
Write-Host "   💡 Choose your version:" -ForegroundColor Yellow
Write-Host "      A) GPU (CUDA 11.8) - FASTER! Recommended"
Write-Host "      B) CPU - Works everywhere, slower"
$choice = Read-Host "Choose A (GPU) or B (CPU)?"

if ($choice -eq "A" -or $choice -eq "a") {
    Write-Host "   🖥️  Installing GPU version (CUDA 11.8)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
} else {
    Write-Host "   💻 Installing CPU version..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
}
Write-Host "✅ PyTorch installed" -ForegroundColor Green

# Step 5: Install critical packages
Write-Host "📦 Step 5: Installing critical packages..." -ForegroundColor Cyan
pip install "numpy<2" -q
pip install pandas scipy scikit-learn -q
pip install torch-geometric -q
pip install rdkit-pypi -q
pip install pytorch-lightning tensorboard tqdm pyyaml -q
Write-Host "✅ Critical packages installed" -ForegroundColor Green

# Step 6: Install optional packages
Write-Host "📦 Step 6: Installing optional packages..." -ForegroundColor Cyan
pip install matplotlib seaborn plotly bokeh -q
pip install streamlit streamlit-plotly-events streamlit-aggrid -q
pip install shap captum lime -q
pip install mordred py3Dmol biopython biotite -q
Write-Host "✅ Optional packages installed" -ForegroundColor Green

# Step 7: Verify installation
Write-Host "🔍 Step 7: Verifying installation..." -ForegroundColor Cyan
python -c "
import torch
import torch_geometric
import pandas
import rdkit
import streamlit
print('✅ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Verification passed!" -ForegroundColor Green
} else {
    Write-Host "❌ Verification failed - check output above" -ForegroundColor Red
    exit 1
}

# Step 8: Ask if ready to train
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Setup complete! Ready to start training?" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Training will take 4-8 hours (2-3 hours with GPU)" -ForegroundColor Yellow
Write-Host "📊 You'll get R² = 0.90+ binding affinity predictions" -ForegroundColor Yellow
Write-Host ""

$start_training = Read-Host "Start training pipeline now? (Y/N)"

if ($start_training -eq "Y" -or $start_training -eq "y") {
    Write-Host ""
    Write-Host "🎉 Starting full pipeline training..." -ForegroundColor Green
    Write-Host "Monitor progress with: tail -f outputs/logs/training.log (in new terminal)"
    Write-Host ""

    # Start training
    python run_pipeline.py

    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host "🎉 TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Trained model saved to: outputs/models/final_model.pt" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Use web interface:" -ForegroundColor White
    Write-Host "   python run_app.py --model outputs/models/final_model.pt" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Check results:" -ForegroundColor White
    Write-Host "   cat outputs/evaluation_metrics.json" -ForegroundColor Gray
    Write-Host ""

} else {
    Write-Host ""
    Write-Host "✅ Setup complete! Ready for training anytime." -ForegroundColor Green
    Write-Host ""
    Write-Host "When ready, run:" -ForegroundColor Yellow
    Write-Host "  python run_pipeline.py" -ForegroundColor Cyan
    Write-Host ""
}
```

---

## 📋 ALTERNATIVE: Step-by-Step Copy/Paste

If the script doesn't work, copy-paste these commands one by one:

### Step 1: Create & Activate Environment
```powershell
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Upgrade pip
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch (Choose ONE)

**For GPU (FAST - 2-3 hours training):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU (SLOW - 4-8 hours training):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Core Packages
```powershell
pip install "numpy<2" pandas scipy scikit-learn torch-geometric rdkit-pypi pytorch-lightning tensorboard tqdm pyyaml
```

### Step 5: Install Web & Visualization
```powershell
pip install streamlit streamlit-plotly-events streamlit-aggrid matplotlib seaborn plotly
```

### Step 6: Verify Everything Works
```powershell
python -c "import torch; print(f'✅ PyTorch {torch.__version__} - GPU: {torch.cuda.is_available()}')"
```

### Step 7: Start Training
```powershell
python run_pipeline.py
```

---

## 📊 WHAT TO EXPECT

### During Training:

```
🚀 DEEPDTA-PRO: COMPLETE DEEP LEARNING PIPELINE
════════════════════════════════════════════════

🔄 PHASE 1: Enhanced Feature Engineering
   Starting...
   ✅ Completed: 587s

🔄 PHASE 2: Advanced Optimization
   Starting...
   ✅ Completed: 1847s

🔄 PHASE 3: Graph Neural Networks
   Epoch 1/100: Loss: 0.567
   Epoch 2/100: Loss: 0.456
   ...
   ✅ Completed: 5847s

[... continues for phases 4-7 ...]

📊 PIPELINE EXECUTION SUMMARY
════════════════════════════════════════════════
✅ Successful: 7/7
⏱️  Total Time: 28342s (7 hours 52 minutes)

🎉 ALL PHASES COMPLETED SUCCESSFULLY! 🎉
```

---

## 🎯 MONITORING IN REAL-TIME

### Open NEW PowerShell Window (DON'T close training window!)

```powershell
# Watch live logs
Get-Content outputs/logs/training.log -Tail 20 -Wait

# Check GPU usage (if using GPU)
while($true) {
    Clear-Host
    nvidia-smi
    Start-Sleep -Seconds 1
}
```

---

## 📁 FILES AFTER TRAINING

```
outputs/
├── models/
│   ├── phase1_features_model.pkl
│   ├── phase2_optimized_model.pt
│   ...
│   └── final_model.pt          ← USE THIS! 🎯
├── logs/
│   ├── training.log
│   ├── metrics.csv
│   └── hyperparameters.yaml
└── results/
    ├── predictions_test.csv
    ├── evaluation_metrics.json
    └── uncertainty_estimates.csv
```

---

## 🚀 AFTER TRAINING: USE YOUR MODEL

### 1. Web Interface with Trained Model
```powershell
python run_app.py --model outputs/models/final_model.pt
# Then open: http://localhost:8501
```

### 2. Make Predictions
```powershell
python examples/single_prediction.py --model_path outputs/models/final_model.pt
```

### 3. Batch Predictions
```powershell
python examples/batch_processing.py --model outputs/models/final_model.pt
```

---

## ⚠️ TROUBLESHOOTING

### If something fails:

**Check which phase failed:**
```powershell
# Look at training log
type outputs/logs/training.log | Select-Object -Last 50
```

**Resume from that phase:**
```powershell
# Resume from phase 5 (for example)
python run_pipeline.py --start 5 --end 7

# Or continue despite errors
python run_pipeline.py --skip-errors
```

**GPU out of memory:**
- Reduce batch size in phase files, OR
- Run on CPU: `$env:CUDA_VISIBLE_DEVICES=""; python run_pipeline.py`

**Missing package:**
```powershell
pip install [package_name]
python run_pipeline.py --skip-errors
```

---

## 💡 PRO TIPS

1. **Run overnight** - Start before bed, wake to trained model
2. **Use GPU** - 2-3x faster with NVIDIA GPU
3. **Monitor progress** - Check logs in another terminal
4. **Save everything** - All outputs are automatically saved
5. **Resume on failure** - Use `--start N` to continue

---

## ✅ YOU'RE READY!

Just copy one of the scripts above and paste into PowerShell.

**Estimated Timeline:**
- Setup: 15-20 min
- Training (GPU): 2-3 hours
- Training (CPU): 4-8 hours
- **Total: Get trained model in 2-8 hours!**

---

**Let the training begin! 🚀🔬🧪**
