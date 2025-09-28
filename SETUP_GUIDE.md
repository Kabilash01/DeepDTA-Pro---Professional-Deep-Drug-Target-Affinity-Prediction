# 🚀 DeepDTA-Pro Setup and Usage Guide

This guide will walk you through setting up and running the DeepDTA-Pro drug discovery system.

## 📋 Prerequisites

- **Python 3.9 recommended** (for best compatibility with molecular libraries)
- Multiple Python versions available: 3.13, 3.10, 3.9, 3.8 (check with `py -0`)
- Windows PowerShell or Command Prompt
- Git (optional, for cloning)

## 🛠️ Setup Instructions

### Step 1: Create Python Environment

```powershell
# Navigate to project directory
cd C:\DL-project

# Remove existing environment if upgrading
Remove-Item -Recurse -Force deepdta_pro\venv

# Create virtual environment with Python 3.9 (recommended)
py -3.9 -m venv deepdta_pro\venv

# Activate virtual environment (Windows)
deepdta_pro\venv\Scripts\Activate.ps1

# Verify Python version
python --version  # Should show Python 3.9.x
```

### Step 2: Install Dependencies

```powershell
# Install core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install "numpy<2"  # Important: NumPy v1.x for RDKit compatibility
pip install pandas scikit-learn matplotlib seaborn
pip install rdkit-pypi
pip install pytorch-lightning
pip install tensorboard
pip install tqdm
pip install pyyaml
```

### Step 3: Verify Installation

```powershell
# Check if environment is working
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric: OK')"
```

## 🎯 Running Examples

The project includes 4 comprehensive examples:

### 1. Single Prediction Example

Predict binding affinity for one drug-protein pair:

```powershell
python examples\single_prediction.py
```

**Output:**
- Predicted binding affinity score
- Binding strength classification
- Results saved to `outputs/single_prediction/`

### 2. Batch Processing Example

Process multiple drug-protein pairs:

```powershell
python examples\batch_processing.py
```

**Features:**
- Batch processing for efficiency
- Progress tracking
- Results analysis and visualization

### 3. Model Training Example

Train a new model from scratch:

```powershell
python examples\train_model.py
```

**Features:**
- Custom dataset loading
- Training progress monitoring
- Model checkpointing
- Validation metrics

### 4. Model Evaluation Example

Evaluate model performance:

```powershell
python examples\evaluate_model.py
```

**Features:**
- Comprehensive metrics (MSE, MAE, R², etc.)
- Performance visualization
- Statistical analysis

## 📁 Project Structure

```
deepdta_pro/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Neural network models
│   ├── training/          # Training utilities
│   └── utils/             # Helper functions
├── examples/              # Usage examples
├── data/                  # Dataset storage
├── models/                # Trained model storage
├── outputs/               # Results and visualizations
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🔧 Troubleshooting

### Common Issues

**1. NumPy Compatibility Error**
```
AttributeError: _ARRAY_API not found
```
**Solution:** Downgrade NumPy to v1.x:
```powershell
pip install "numpy<2"
```

**2. RDKit Import Error**
```
ModuleNotFoundError: No module named 'rdkit'
```
**Solution:** Install RDKit:
```powershell
pip install rdkit-pypi
```

**3. CUDA/GPU Issues**
```
RuntimeError: CUDA out of memory
```
**Solution:** The examples automatically use CPU if GPU is unavailable. To force CPU:
```python
device = torch.device('cpu')
```

**4. Missing Data Files**
The examples use mock data when real datasets aren't available. To use real data:
- Place DAVIS dataset in `../davis/`
- Place KIBA dataset in `../kiba/`

## 📊 Understanding Output

### Binding Affinity Scores

- **Score Range:** 0-12 (higher = stronger binding)
- **Strong Binding:** < 6.0
- **Moderate Binding:** 6.0-8.0  
- **Weak Binding:** > 8.0

### Output Files

Each example creates detailed output:
- **JSON Results:** Raw predictions and metadata
- **Visualizations:** Plots and charts (when applicable)
- **Logs:** Detailed execution information

## 🎓 Next Steps

1. **Explore Examples:** Run all 4 examples to understand functionality
2. **Customize Data:** Replace mock data with your datasets
3. **Modify Models:** Experiment with different architectures
4. **Train Models:** Use real data to train custom models
5. **Analyze Results:** Use built-in visualization tools

## 📚 Additional Resources

- **API Documentation:** See `docs/` directory
- **Model Architecture:** Check `src/models/` for implementations
- **Data Processing:** Review `src/data/` for preprocessing pipelines
- **Training Scripts:** Examine `src/training/` for training utilities

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure you're using the correct Python version (3.8+)
4. Check that your virtual environment is activated

## 🎉 Success!

If you can run the examples successfully, you have a working DeepDTA-Pro installation ready for drug discovery research!

---

**Last Updated:** January 2025
**Version:** 1.0.0