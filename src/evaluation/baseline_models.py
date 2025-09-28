"""
Baseline Models for Drug-Target Binding Affinity Prediction
Implementation of various baseline models for comparison with DeepDTA-Pro.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings

logger = logging.getLogger(__name__)

class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselineModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on test data."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}

class SimpleMLPBaseline(BaselineModel):
    """
    Simple Multi-Layer Perceptron baseline using molecular descriptors.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 64,
                 device: str = 'cpu'):
        super().__init__("Simple MLP")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        self._build_model()
    
    def _build_model(self):
        """Build MLP architecture."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleMLPBaseline':
        """Fit MLP to training data."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.debug(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions

class RandomForestBaseline(BaselineModel):
    """Random Forest baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__("Random Forest")
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        self.model = RandomForestRegressor(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestBaseline':
        """Fit Random Forest to training data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

class GradientBoostingBaseline(BaselineModel):
    """Gradient Boosting baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__("Gradient Boosting")
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = GradientBoostingRegressor(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingBaseline':
        """Fit Gradient Boosting to training data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

class SVRBaseline(BaselineModel):
    """Support Vector Regression baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__("SVR")
        
        # Default parameters
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'epsilon': 0.1
        }
        default_params.update(kwargs)
        
        self.model = SVR(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRBaseline':
        """Fit SVR to training data."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

class LinearRegressionBaseline(BaselineModel):
    """Linear Regression baseline model."""
    
    def __init__(self, regularization: str = None, alpha: float = 1.0):
        if regularization == 'ridge':
            name = "Ridge Regression"
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            name = "Lasso Regression"
            self.model = Lasso(alpha=alpha)
        elif regularization == 'elastic':
            name = "Elastic Net"
            self.model = ElasticNet(alpha=alpha)
        else:
            name = "Linear Regression"
            self.model = LinearRegression()
        
        super().__init__(name)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionBaseline':
        """Fit linear model to training data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

class KNNBaseline(BaselineModel):
    """K-Nearest Neighbors baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__("K-Nearest Neighbors")
        
        # Default parameters
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        }
        default_params.update(kwargs)
        
        self.model = KNeighborsRegressor(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNBaseline':
        """Fit KNN to training data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

class DeepDTABaseline(BaselineModel):
    """
    Simplified DeepDTA baseline (CNN-based) without graph components.
    """
    
    def __init__(self, 
                 drug_vocab_size: int,
                 protein_vocab_size: int,
                 drug_max_len: int = 100,
                 protein_max_len: int = 1000,
                 embedding_dim: int = 128,
                 cnn_filters: int = 32,
                 cnn_kernel_size: int = 4,
                 fc_dim: int = 1024,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 64,
                 device: str = 'cpu'):
        super().__init__("DeepDTA Baseline")
        
        self.drug_vocab_size = drug_vocab_size
        self.protein_vocab_size = protein_vocab_size
        self.drug_max_len = drug_max_len
        self.protein_max_len = protein_max_len
        self.embedding_dim = embedding_dim
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        self._build_model()
    
    def _build_model(self):
        """Build DeepDTA architecture."""
        self.model = DeepDTANet(
            drug_vocab_size=self.drug_vocab_size,
            protein_vocab_size=self.protein_vocab_size,
            drug_max_len=self.drug_max_len,
            protein_max_len=self.protein_max_len,
            embedding_dim=self.embedding_dim,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            fc_dim=self.fc_dim,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def fit(self, X: Tuple[np.ndarray, np.ndarray], y: np.ndarray) -> 'DeepDTABaseline':
        """
        Fit DeepDTA to training data.
        
        Args:
            X: Tuple of (drug_sequences, protein_sequences)
            y: Target values
        """
        drug_seqs, protein_seqs = X
        
        drug_tensor = torch.LongTensor(drug_seqs).to(self.device)
        protein_tensor = torch.LongTensor(protein_seqs).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(drug_tensor, protein_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_drugs, batch_proteins, batch_y in dataloader:
                self.optimizer.zero_grad()
                pred = self.model(batch_drugs, batch_proteins)
                loss = self.criterion(pred.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.debug(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        drug_seqs, protein_seqs = X
        
        self.model.eval()
        with torch.no_grad():
            drug_tensor = torch.LongTensor(drug_seqs).to(self.device)
            protein_tensor = torch.LongTensor(protein_seqs).to(self.device)
            predictions = self.model(drug_tensor, protein_tensor).cpu().numpy().flatten()
        
        return predictions

class DeepDTANet(nn.Module):
    """CNN-based network architecture for DeepDTA baseline."""
    
    def __init__(self, 
                 drug_vocab_size: int,
                 protein_vocab_size: int,
                 drug_max_len: int,
                 protein_max_len: int,
                 embedding_dim: int,
                 cnn_filters: int,
                 cnn_kernel_size: int,
                 fc_dim: int,
                 dropout: float):
        super().__init__()
        
        # Embedding layers
        self.drug_embedding = nn.Embedding(drug_vocab_size, embedding_dim)
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim)
        
        # CNN layers for drug
        self.drug_conv1 = nn.Conv1d(embedding_dim, cnn_filters, cnn_kernel_size)
        self.drug_conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, cnn_kernel_size)
        self.drug_conv3 = nn.Conv1d(cnn_filters * 2, cnn_filters * 3, cnn_kernel_size)
        
        # CNN layers for protein
        self.protein_conv1 = nn.Conv1d(embedding_dim, cnn_filters, cnn_kernel_size)
        self.protein_conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, cnn_kernel_size)
        self.protein_conv3 = nn.Conv1d(cnn_filters * 2, cnn_filters * 3, cnn_kernel_size)
        
        # Calculate feature dimensions after convolutions
        drug_feat_dim = self._get_conv_output_dim(drug_max_len, cnn_kernel_size, 3) * cnn_filters * 3
        protein_feat_dim = self._get_conv_output_dim(protein_max_len, cnn_kernel_size, 3) * cnn_filters * 3
        
        # Fully connected layers
        combined_dim = drug_feat_dim + protein_feat_dim
        self.fc1 = nn.Linear(combined_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim // 2)
        self.fc3 = nn.Linear(fc_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_conv_output_dim(self, input_dim: int, kernel_size: int, num_layers: int) -> int:
        """Calculate output dimension after convolution layers."""
        dim = input_dim
        for _ in range(num_layers):
            dim = dim - kernel_size + 1
        return max(1, dim)
    
    def forward(self, drug_seq: torch.Tensor, protein_seq: torch.Tensor) -> torch.Tensor:
        # Embeddings
        drug_embed = self.drug_embedding(drug_seq)  # (batch, seq_len, embed_dim)
        protein_embed = self.protein_embedding(protein_seq)
        
        # Transpose for conv1d (batch, embed_dim, seq_len)
        drug_embed = drug_embed.transpose(1, 2)
        protein_embed = protein_embed.transpose(1, 2)
        
        # Drug CNN
        drug_conv = F.relu(self.drug_conv1(drug_embed))
        drug_conv = F.relu(self.drug_conv2(drug_conv))
        drug_conv = F.relu(self.drug_conv3(drug_conv))
        drug_feat = torch.flatten(drug_conv, 1)
        
        # Protein CNN
        protein_conv = F.relu(self.protein_conv1(protein_embed))
        protein_conv = F.relu(self.protein_conv2(protein_conv))
        protein_conv = F.relu(self.protein_conv3(protein_conv))
        protein_feat = torch.flatten(protein_conv, 1)
        
        # Combine features
        combined = torch.cat([drug_feat, protein_feat], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        
        return output

class BaselineComparator:
    """
    Utility class for comparing multiple baseline models.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, model: BaselineModel):
        """Add a baseline model to the comparison."""
        self.models[model.name] = model
        logger.info(f"Added baseline model: {model.name}")
    
    def compare_models(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Compare all baseline models on given data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            verbose: Whether to print progress
            
        Returns:
            Dictionary of results for each model
        """
        results = {}
        
        for name, model in self.models.items():
            if verbose:
                logger.info(f"Training and evaluating {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(y_test, y_pred)
                
                results[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'pearson': pearson_r,
                    'pearson_p': pearson_p
                }
                
                if verbose:
                    logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson: {pearson_r:.4f}")
                    
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = {
                    'mse': float('inf'),
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'pearson': 0.0,
                    'pearson_p': 1.0,
                    'error': str(e)
                }
        
        self.results = results
        return results
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, BaselineModel]:
        """Get the best performing model based on a specific metric."""
        if not self.results:
            raise ValueError("No results available. Run compare_models first.")
        
        # For loss metrics, lower is better
        minimize_metrics = ['mse', 'rmse', 'mae']
        
        if metric in minimize_metrics:
            best_name = min(self.results.keys(), key=lambda x: self.results[x].get(metric, float('inf')))
        else:
            best_name = max(self.results.keys(), key=lambda x: self.results[x].get(metric, -float('inf')))
        
        return best_name, self.models[best_name]
    
    def print_results_table(self):
        """Print formatted results table."""
        if not self.results:
            print("No results available.")
            return
        
        print("\nBaseline Model Comparison Results")
        print("=" * 80)
        print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Pearson':<8}")
        print("-" * 80)
        
        for name, metrics in self.results.items():
            if 'error' not in metrics:
                print(f"{name:<20} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} "
                      f"{metrics['r2']:<8.4f} {metrics['pearson']:<8.4f}")
            else:
                print(f"{name:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}")

# Factory function to create baseline models
def create_baseline_models(feature_dim: int = None) -> List[BaselineModel]:
    """
    Factory function to create a standard set of baseline models.
    
    Args:
        feature_dim: Dimension of input features (for neural models)
        
    Returns:
        List of baseline models
    """
    models = []
    
    # Traditional ML models
    models.append(LinearRegressionBaseline())
    models.append(LinearRegressionBaseline(regularization='ridge', alpha=1.0))
    models.append(LinearRegressionBaseline(regularization='lasso', alpha=1.0))
    models.append(RandomForestBaseline(n_estimators=100))
    models.append(GradientBoostingBaseline(n_estimators=100))
    models.append(SVRBaseline(kernel='rbf', C=1.0))
    models.append(KNNBaseline(n_neighbors=5))
    
    # Neural network model (if feature dimension is provided)
    if feature_dim is not None:
        models.append(SimpleMLPBaseline(
            input_dim=feature_dim,
            hidden_dims=[512, 256, 128],
            epochs=50
        ))
    
    return models

# Example usage
if __name__ == "__main__":
    print("Testing baseline models...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples)
    X_test = np.random.randn(200, n_features)
    y_test = np.random.randn(200)
    
    # Create baseline comparator
    comparator = BaselineComparator()
    
    # Add models
    for model in create_baseline_models(feature_dim=n_features):
        comparator.add_model(model)
    
    # Compare models
    results = comparator.compare_models(X_train, y_train, X_test, y_test)
    
    # Print results
    comparator.print_results_table()
    
    # Get best model
    best_name, best_model = comparator.get_best_model('rmse')
    print(f"\nBest model by RMSE: {best_name}")
    
    print("Baseline models test completed successfully!")