"""
PHASE 3: GRAPH NEURAL NETWORKS FOR MOLECULAR ENCODING
Advanced DeepDTA-Pro with GNN-based molecular representation
Bridges Phase 2 features with graph-based learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from pathlib import Path
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLIFIED GRAPH REPRESENTATION
# ============================================================================

class MolecularGraphBuilder:
    """Build simplified molecular graphs from SMILES strings"""

    def __init__(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
            self.Chem = Chem
            self.AllChem = AllChem
            self.Descriptors = Descriptors
            self.rdkit_available = True
        except ImportError:
            logger.warning("RDKit not available - using mock graphs")
            self.rdkit_available = False

    def smiles_to_graph(self, smiles: str) -> tuple:
        """
        Convert SMILES to graph representation

        Returns:
            (node_features, edge_index) for molecular graph
        """
        if not self.rdkit_available:
            return self._mock_graph()

        try:
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._mock_graph()

            # Node features: atom properties
            node_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum() / 100.0,
                    atom.GetTotalDegree() / 10.0,
                    atom.GetTotalNumHs() / 10.0,
                    atom.GetFormalCharge() / 10.0,
                    int(atom.GetIsAromatic()),
                    int(atom.GetHybridization()),
                ]
                node_features.append(features)

            if not node_features:
                return self._mock_graph()

            # Edge index: connectivity
            edge_index = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected

            # Ensure minimum graph size
            min_nodes = 5
            if len(node_features) < min_nodes:
                pad_size = min_nodes - len(node_features)
                node_features.extend([[0.0] * 6 for _ in range(pad_size)])

                offset = len(node_features) - pad_size
                for i in range(pad_size):
                    edge_index.append([offset + i, (offset + i + 1) % min_nodes])

            x = torch.tensor(node_features[:min_nodes], dtype=torch.float32)

            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                # Fully connected fallback
                n = min(len(node_features), min_nodes)
                edge_index = torch.zeros((2, n * (n - 1)), dtype=torch.long)
                idx = 0
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            edge_index[0, idx] = i
                            edge_index[1, idx] = j
                            idx += 1

            return x, edge_index

        except Exception as e:
            logger.debug(f"Graph construction error: {e}")
            return self._mock_graph()

    def _mock_graph(self) -> tuple:
        """Generate mock molecular graph"""
        num_nodes = 5
        x = torch.randn(num_nodes, 6, dtype=torch.float32)

        # Random edges (fully connected)
        edge_index = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)
        idx = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index[0, idx] = i
                    edge_index[1, idx] = j
                    idx += 1

        return x, edge_index


# ============================================================================
# GRAPH NEURAL NETWORK FOR MOLECULAR ENCODING
# ============================================================================

class SimpleGATLayer(nn.Module):
    """Simple Graph Attention Layer (without torch_geometric dependency)"""

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.fc_out = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edges [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # Linear transformations
        Q = self.query(x)  # [N, out_channels]
        K = self.key(x)    # [N, out_channels]
        V = self.value(x)  # [N, out_channels]

        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)

        # Attention scores
        scores = torch.einsum('nhd,mhd->nmh', Q, K) / np.sqrt(self.head_dim)

        # Clamp edge indices to valid range to prevent indexing errors
        edge_index_clamped = edge_index.clamp(0, num_nodes - 1)

        # Mask attention to only connected nodes - use safe indexing
        mask = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=torch.bool)
        mask[edge_index_clamped[0], edge_index_clamped[1]] = True
        mask = mask.unsqueeze(-1)  # [N, N, 1]

        # Apply mask (set non-connected to -inf, others to 0)
        scores = scores.masked_fill(~mask, float('-inf'))
        attention = F.softmax(scores, dim=1)

        # Handle NaNs from softmax of all -inf (replace with uniform attention)
        attention = torch.nan_to_num(attention, nan=1.0/num_nodes)

        # Apply attention to values
        out = torch.einsum('nmh,mhd->nhd', attention, V)
        out = out.reshape(num_nodes, -1)

        return self.fc_out(out)


class GNNMolecularEncoder(nn.Module):
    """GNN-based molecular encoder using graph attention"""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.layers.append(SimpleGATLayer(input_dim, hidden_dim, num_heads=4))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(SimpleGATLayer(hidden_dim, hidden_dim, num_heads=4))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Graph-level representation [hidden_dim]
        """
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # Global average pooling
        graph_repr = x.mean(dim=0)
        return graph_repr


# ============================================================================
# GNN-ENHANCED PROTEIN ENCODER
# ============================================================================

class SimpleProteinTransformer(nn.Module):
    """Simplified protein transformer without full torch_geometric"""

    def __init__(self, vocab_size: int = 26, embed_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=0.3,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])

    def forward(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_ids: [batch_size, seq_len]

        Returns:
            [batch_size, embed_dim]
        """
        x = self.embedding(sequence_ids)

        for layer in self.transformer_layers:
            x = layer(x)

        # Mean pooling over sequence
        x = x.mean(dim=1)
        return x


# ============================================================================
# PHASE 3 GNN MODEL
# ============================================================================

class Phase3GNNModel(nn.Module):
    """Phase 3: GNN-based DeepDTA-Pro"""

    def __init__(self, gnn_hidden_dim: int = 128, prot_embed_dim: int = 128, fusion_dim: int = 256):
        super().__init__()

        # Molecular GNN encoder
        self.mol_encoder = GNNMolecularEncoder(input_dim=6, hidden_dim=gnn_hidden_dim, num_layers=3)

        # Protein transformer encoder
        self.prot_encoder = SimpleProteinTransformer(vocab_size=26, embed_dim=prot_embed_dim, num_layers=2)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(gnn_hidden_dim + prot_embed_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),  # Use LayerNorm instead of BatchNorm
            nn.Linear(128, 1)
        )

        # Initialize output layer to predict reasonable scale (5-11 affinity range)
        with torch.no_grad():
            self.fusion[-1].bias.fill_(7.5)  # Initialize to mean affinity (~7.5)

    def forward(self, mol_graph: tuple, prot_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mol_graph: (node_features, edge_index)
            prot_ids: protein sequence token IDs [1, seq_len]

        Returns:
            Affinity predictions [1, 1]
        """
        x, edge_index = mol_graph

        # Encode molecular graph -> [128] (global pooled)
        mol_repr = self.mol_encoder(x, edge_index)
        # Ensure shape [1, 128] for concatenation
        if mol_repr.dim() == 1:
            mol_repr = mol_repr.unsqueeze(0)  # [128] -> [1, 128]

        # Encode protein -> [1, 128]
        prot_repr = self.prot_encoder(prot_ids)
        # Ensure shape [1, 128]
        if prot_repr.dim() == 1:
            prot_repr = prot_repr.unsqueeze(0)

        # Fuse and predict
        combined = torch.cat([mol_repr, prot_repr], dim=-1)  # [1, 256]
        affinity = self.fusion(combined)  # [1, 1]

        return affinity


# ============================================================================
# PHASE 3 TRAINER
# ============================================================================

class Phase3GNNTrainer:
    """Trainer for Phase 3 with GNN molecular encoding"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Using device: {self.device}")

        self.model = Phase3GNNModel(gnn_hidden_dim=128, prot_embed_dim=128).to(self.device)
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        self.criterion = nn.MSELoss()
        self.graph_builder = MolecularGraphBuilder()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def _sequence_to_ids(self, sequence: str, max_len: int = 100) -> torch.Tensor:
        """Convert protein sequence to token IDs"""
        aa_to_id = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
            'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 0
        }

        sequence = sequence.upper()[:max_len].ljust(max_len, 'X')
        ids = [aa_to_id.get(aa, 0) for aa in sequence]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

    def train_epoch(self, train_data, epoch, total_epochs):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_samples = 0

        pbar = tqdm(range(0, len(train_data), self.config['batch_size']),
                    desc=f"Epoch {epoch + 1}/{total_epochs} GNN Train")

        for batch_start in pbar:
            batch_end = min(batch_start + self.config['batch_size'], len(train_data))
            batch = train_data[batch_start:batch_end]

            self.optimizer.zero_grad()
            batch_loss_sum = 0.0
            batch_samples = 0

            for sample in batch:
                try:
                    # Build molecular graph
                    mol_graph = self.graph_builder.smiles_to_graph(sample['drug_smiles'])
                    x, edge_index = mol_graph
                    x, edge_index = x.to(self.device), edge_index.to(self.device)

                    # Convert protein to IDs
                    prot_ids = self._sequence_to_ids(sample['protein_sequence']).to(self.device)

                    # Forward pass
                    pred = self.model((x, edge_index), prot_ids)
                    target = torch.tensor([[sample['affinity']]], dtype=torch.float32).to(self.device)

                    loss = self.criterion(pred, target)
                    batch_loss_sum += loss.item()
                    batch_samples += 1

                    # Backward pass (accumulate gradients)
                    loss.backward()

                except Exception as e:
                    logger.debug(f"Sample processing error: {e}")
                    continue

            # Update weights after processing batch
            if batch_samples > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss_avg = batch_loss_sum / batch_samples
                total_loss += batch_loss_avg
                num_samples += batch_samples
                pbar.set_postfix({'loss': f'{batch_loss_avg:.4f}'})

        return total_loss / max(1, num_samples) if num_samples > 0 else 0.0

    def validate(self, val_data):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for sample in val_data:
                try:
                    mol_graph = self.graph_builder.smiles_to_graph(sample['drug_smiles'])
                    x, edge_index = mol_graph
                    x, edge_index = x.to(self.device), edge_index.to(self.device)

                    prot_ids = self._sequence_to_ids(sample['protein_sequence']).to(self.device)

                    pred = self.model((x, edge_index), prot_ids)
                    target = torch.tensor([[sample['affinity']]], dtype=torch.float32).to(self.device)

                    loss = self.criterion(pred, target)
                    total_loss += loss.item()
                    predictions.append(pred.cpu().item())
                    targets.append(sample['affinity'])

                except Exception as e:
                    logger.debug(f"Validation error: {e}")
                    continue

        avg_loss = total_loss / max(1, len(val_data))
        predictions = np.array(predictions)
        targets = np.array(targets)

        if len(predictions) > 1:
            mse = mean_squared_error(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
        else:
            mse = mae = r2 = 0.0

        return avg_loss, mse, mae, r2

    def train(self, train_data, val_data, test_data):
        """Main training loop"""
        logger.info("🚀 Starting Phase 3 GNN Training...")

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=self.config['epochs'],
            pct_start=0.3,
            anneal_strategy='cos'
        )

        for epoch in range(self.config['epochs']):
            start_time = time.time()

            train_loss = self.train_epoch(train_data, epoch, self.config['epochs'])
            val_loss, val_mse, val_mae, val_r2 = self.validate(val_data)

            scheduler.step()  # Call after training epoch

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config['epochs']} ({epoch_time:.2f}s) | "
                f"Train Loss: {train_loss:.4f} | Val: Loss={val_loss:.4f}, R²={val_r2:.4f}"
            )

        # Test evaluation
        logger.info("🧪 Final test evaluation...")
        test_loss, test_mse, test_mae, test_r2 = self.validate(test_data)

        logger.info(f"📊 Final Test Results - R²: {test_r2:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

        return {'test_r2': test_r2, 'test_mse': test_mse, 'test_mae': test_mae}


# ============================================================================
# DEMO
# ============================================================================

def main():
    """Run Phase 3 GNN training"""
    print("\n" + "=" * 70)
    print("PHASE 3: GRAPH NEURAL NETWORK MOLECULAR ENCODING")
    print("=" * 70 + "\n")

    # Create synthetic data
    drug_templates = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CC(=O)OC1=CC=CC=C1C(=O)O"]
    protein_templates = ["MKKFFDSRREQGGSGLGSGSSGGGGSGGGYGNQDQSGGGGSGGGYGNQDQ"]

    np.random.seed(42)
    dataset = []
    for i in range(100):  # Smaller for GNN testing
        dataset.append({
            'drug_smiles': drug_templates[i % 2],
            'protein_sequence': protein_templates[0] + "G" * (i % 10),
            'affinity': np.random.uniform(1.0, 10.0)
        })

    # Split
    n = len(dataset)
    train_data = dataset[:int(0.7 * n)]
    val_data = dataset[int(0.7 * n):int(0.85 * n)]
    test_data = dataset[int(0.85 * n):]

    logger.info(f"✅ Dataset: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Config
    config = {
        'epochs': 20,
        'batch_size': 4,
        'learning_rate': 5e-4,
        'weight_decay': 1e-4
    }

    # Train
    trainer = Phase3GNNTrainer(config)
    results = trainer.train(train_data, val_data, test_data)

    print("\n" + "=" * 70)
    print("🎉 PHASE 3 COMPLETED!")
    print("=" * 70)
    print(f"✅ Final Test R²: {results['test_r2']:.4f}")
    print(f"✅ Final Test MSE: {results['test_mse']:.4f}")
    print(f"✅ Final Test MAE: {results['test_mae']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
