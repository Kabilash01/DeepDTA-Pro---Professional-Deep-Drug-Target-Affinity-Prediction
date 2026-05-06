# 🚀 DeepDTA-Pro: Setup & Installation Guide

**Complete guide to install and run DeepDTA-Pro**

---

## 📋 Prerequisites

- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.9+ (recommended 3.9)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **GPU** (optional): NVIDIA GPU with CUDA 11.8+ (2-3x faster training)

---

## 🛠️ STEP-BY-STEP INSTALLATION

### Step 1: Navigate to Project Folder

```bash
cd c:\DeepDTA-Pro---Professional-Deep-Drug-Target-Affinity-Prediction
```

---

### Step 2: Create Python Virtual Environment

**Windows (PowerShell):**
```powershell
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3.9 -m venv venv
source venv/bin/activate
```

**Verify activation:**
```bash
python --version  # Should show Python 3.9.x
```

---

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

### Step 4: Install PyTorch

**Choose ONE based on your hardware:**

#### Option A: GPU Support (NVIDIA CUDA 11.8) - RECOMMENDED FOR SPEED
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
**Training time**: 2-3 hours

#### Option B: CPU Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
**Training time**: 4-8 hours

---

### Step 5: Install Core Dependencies

```bash
pip install "numpy<2"
pip install pandas scipy scikit-learn
pip install torch-geometric
pip install rdkit-pypi
pip install pytorch-lightning tensorboard tqdm pyyaml
```

---

### Step 6: Install Web Interface (Optional but Recommended)

```bash
pip install streamlit
pip install streamlit-plotly-events streamlit-aggrid
pip install matplotlib seaborn plotly
```

---

### Step 7: Verify Installation

```bash
python -c "
import torch
import torch_geometric
import pandas
import rdkit
print('✅ PyTorch:', torch.__version__)
print('✅ Torch Geometric')
print('✅ Pandas')
print('✅ RDKit')
print('✅ GPU Available:', torch.cuda.is_available())
"
```

**Expected output:**
```
✅ PyTorch: 2.0.1+cu118
✅ Torch Geometric
✅ Pandas
✅ RDKit
✅ GPU Available: True  (or False if no GPU)
```

---

## 🚀 QUICK START (Copy-Paste)

**Windows PowerShell - All in one:**

```powershell
# 1. Setup
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install packages (choose GPU or CPU)
# For GPU (faster):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU (slower):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies
pip install "numpy<2" pandas scipy scikit-learn torch-geometric rdkit-pypi pytorch-lightning tensorboard tqdm pyyaml streamlit matplotlib seaborn

# 4. Verify
python -c "import torch; print('✅ Ready!' if torch.__version__ else 'Error')"

# 5. Run web interface (no training needed to see it work)
python run_app.py
```

Then open: **http://localhost:8501** 🌐

---

## 🧪 TEST YOUR INSTALLATION

### Run Quick Example (No Training)
```bash
python examples/single_prediction.py
```

**Output:**
```
🧪 DeepDTA-Pro Single Prediction Example
📱 Using device: cpu
💊 Drug SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
🧬 Protein Sequence: MKTAYIAKQRQISFVKSHFSRQ...
⏱️  Processing... (takes ~2 seconds)
📊 Predicted Affinity: 6.85
✅ Example completed successfully!
```

---

## ⚡ WHAT WORKS NOW (Without Training)

✅ Web interface: `python run_app.py` - See UI immediately
✅ Examples: `python examples/single_prediction.py` - Mock predictions
✅ Batch processing: Upload CSV for batch predictions
✅ Explore code structure and modules

❌ Real predictions (need to train first)
❌ Accurate affinity predictions (R²=0.90)

---

## 🎓 NEXT STEPS

### Option 1: Just Play Around (15 minutes)
```bash
python run_app.py
# Explore UI, upload test CSV, see mock predictions
```

### Option 2: Train Full Model (4-8 hours)
```bash
python run_pipeline.py
# Trains all 7 phases, creates real model (R²=0.90+)
```

See **TRAINING_GUIDE.md** for detailed training instructions.

---

## 🐛 TROUBLESHOOTING

### Problem: "ModuleNotFoundError: rdkit"
```bash
pip install rdkit-pypi
```

### Problem: "numpy compatibility error"
```bash
pip install "numpy<2"
```

### Problem: "CUDA out of memory"
```bash
# Use CPU instead
$env:CUDA_VISIBLE_DEVICES=""
python run_app.py
```

### Problem: "Port 8501 already in use"
```bash
python run_app.py --port 8502
```

### Problem: "torch_geometric not found"
```bash
pip install torch-geometric
```

---

## 📊 PROJECT STRUCTURE

```
DeepDTA-Pro/
├── README.md                    ← Overview
├── INSTALLATION_GUIDE.md        ← This file
├── TRAINING_GUIDE.md            ← Training instructions
│
├── run_app.py                   ← Run web interface
├── run_pipeline.py              ← Start training
├── setup.py                     ← Setup script
│
├── phase1_enhanced_features.py  ← Phase 1
├── phase2_advanced_training.py  ← Phase 2
├── phase3_gnn_with_real_data.py ← Phase 3
├── phase4_transfer_learning.py  ← Phase 4
├── phase5_multitask_learning.py ← Phase 5
├── phase6_uncertainty.py        ← Phase 6
├── phase7_ensemble.py           ← Phase 7
│
├── src/                         ← Source code
│   ├── data/                    ← Data processing
│   ├── models/                  ← Neural networks
│   ├── training/                ← Training utilities
│   ├── evaluation/              ← Metrics
│   ├── web_interface/           ← Streamlit app
│   └── ...
│
├── examples/                    ← Usage examples
├── configs/                     ← Configuration files
├── data/                        ← Datasets (when downloaded)
├── models/                      ← Saved models
├── outputs/                     ← Training results
├── requirements.txt             ← Dependencies
└── ...
```

---

## ✅ VERIFICATION CHECKLIST

After installation, verify:

- [ ] Python environment created
- [ ] Python 3.9.x showing
- [ ] pip upgraded
- [ ] All packages installed
- [ ] Test example runs successfully
- [ ] Web interface launches (http://localhost:8501)
- [ ] GPU detected (if you have GPU)

---

## 📞 QUICK COMMANDS REFERENCE

```bash
# Activate environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux

# Run web interface
python run_app.py

# Run example
python examples/single_prediction.py

# Start training
python run_pipeline.py

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Deactivate environment
deactivate
```

---

## 🎯 NEXT: Read TRAINING_GUIDE.md for Full Training Instructions

After successful installation:
1. Try web interface: `python run_app.py`
2. Run example: `python examples/single_prediction.py`
3. See TRAINING_GUIDE.md for full model training

---

**Setup Complete! Ready to use or train!** 🚀
