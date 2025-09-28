"""
DeepDTA-Pro: Advanced Drug-Target Binding Affinity Prediction
"""

__version__ = "1.0.0"
__author__ = "DeepDTA-Pro Team"
__email__ = "contact@deepdta-pro.com"

from . import data
from . import models
from . import training
from . import evaluation
from . import interpretability
from . import inference
from . import utils

__all__ = [
    "data",
    "models", 
    "training",
    "evaluation",
    "interpretability",
    "inference",
    "utils"
]