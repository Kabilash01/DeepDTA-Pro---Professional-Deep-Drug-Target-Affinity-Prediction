"""
DIAGNOSTIC SCRIPT: Debug loss computation issues
Identify why losses are 0 and gradients aren't flowing
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase3_gnn_training import MolecularGraphBuilder, Phase3GNNModel
from phase3_real_data import DAVISDatasetLoader


def diagnose_loss_computation():
    """Test forward pass and loss computation step by step"""

    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Loss Computation Analysis")
    print("=" * 80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 1. Load a sample from DAVIS
    print("Step 1: Loading sample from DAVIS dataset...")
    loader = DAVISDatasetLoader(data_dir="data")
    davis_data = loader.load_davis()

    if not davis_data:
        print("❌ Failed to load DAVIS data")
        return

    sample = davis_data[0]
    print(f"✅ Sample loaded:")
    print(f"   - SMILES: {sample['drug_smiles'][:50]}...")
    print(f"   - Protein length: {len(sample['protein_sequence'])}")
    print(f"   - Affinity: {sample['affinity']}\n")

    # 2. Build molecular graph
    print("Step 2: Building molecular graph...")
    graph_builder = MolecularGraphBuilder()
    mol_graph = graph_builder.smiles_to_graph(sample['drug_smiles'])
    x_raw, edge_index_raw = mol_graph

    print(f"✅ Graph built:")
    print(f"   - Node features shape: {x_raw.shape}")
    print(f"   - Edge index shape: {edge_index_raw.shape}")
    print(f"   - Node feature range: [{x_raw.min():.4f}, {x_raw.max():.4f}]\n")

    # 3. Move to device
    print("Step 3: Moving tensors to device...")
    x = x_raw.to(device)
    edge_index = edge_index_raw.to(device)

    # 4. Sequence to IDs
    print("Step 4: Converting protein sequence to token IDs...")
    aa_to_id = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 0
    }

    sequence = sample['protein_sequence'].upper()[:100].ljust(100, 'X')
    prot_ids = torch.tensor([aa_to_id.get(aa, 0) for aa in sequence], dtype=torch.long).unsqueeze(0).to(device)
    print(f"✅ Protein IDs shape: {prot_ids.shape}\n")

    # 5. Initialize model
    print("Step 5: Initializing Phase 3 GNN Model...")
    model = Phase3GNNModel(gnn_hidden_dim=128, prot_embed_dim=128).to(device)
    print(f"✅ Model loaded, parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # 6. Forward pass
    print("Step 6: Forward pass...")
    model.eval()
    with torch.no_grad():
        pred = model((x, edge_index), prot_ids)
        print(f"✅ Prediction output:")
        print(f"   - Shape: {pred.shape}")
        print(f"   - Value: {pred.item():.6f}")
        print(f"   - Is NaN: {torch.isnan(pred).any()}")
        print(f"   - Is Inf: {torch.isinf(pred).any()}\n")

    # 7. Target value
    print("Step 7: Target value...")
    target = torch.tensor([[sample['affinity']]], dtype=torch.float32).to(device)
    print(f"✅ Target:")
    print(f"   - Shape: {target.shape}")
    print(f"   - Value: {target.item():.6f}\n")

    # 8. Loss computation
    print("Step 8: Loss computation...")
    criterion = nn.MSELoss()
    model.train()

    # Forward with gradients enabled
    pred_grad = model((x, edge_index), prot_ids)
    loss = criterion(pred_grad, target)

    print(f"✅ Loss:")
    print(f"   - Shape: {loss.shape}")
    print(f"   - Value: {loss.item():.8f}")
    print(f"   - Is NaN: {torch.isnan(loss).any()}")
    print(f"   - Is Inf: {torch.isinf(loss).any()}")
    print(f"   - Requires grad: {loss.requires_grad}\n")

    # 9. Backward pass
    print("Step 9: Backward pass...")
    try:
        loss.backward()
        print(f"✅ Backward pass successful\n")

        # Check gradients
        print("Step 10: Checking gradients...")
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                if grad_norm > 0:
                    print(f"   - {name}: {grad_norm:.8f}")

        total_grad_norm = np.sqrt(total_grad_norm)
        print(f"\n✅ Total gradient norm: {total_grad_norm:.8f}")

        if total_grad_norm == 0:
            print("   ⚠️  WARNING: Gradient norm is 0 - gradients not flowing!\n")
        else:
            print("   ✅ Gradients flowing correctly!\n")

    except Exception as e:
        print(f"❌ Backward pass failed: {e}\n")

    # 11. Check prediction range
    print("Step 11: Analyzing prediction distribution...")
    with torch.no_grad():
        preds_batch = []
        for i in range(min(10, len(davis_data))):
            sample_i = davis_data[i]
            graph_i = graph_builder.smiles_to_graph(sample_i['drug_smiles'])
            x_i, edge_index_i = graph_i
            x_i, edge_index_i = x_i.to(device), edge_index_i.to(device)

            seq_i = sample_i['protein_sequence'].upper()[:100].ljust(100, 'X')
            prot_ids_i = torch.tensor([aa_to_id.get(aa, 0) for aa in seq_i], dtype=torch.long).unsqueeze(0).to(device)

            pred_i = model((x_i, edge_index_i), prot_ids_i)
            preds_batch.append(pred_i.item())

        preds_array = np.array(preds_batch)
        print(f"✅ Predictions from 10 samples:")
        print(f"   - Min: {preds_array.min():.6f}")
        print(f"   - Max: {preds_array.max():.6f}")
        print(f"   - Mean: {preds_array.mean():.6f}")
        print(f"   - Std: {preds_array.std():.6f}")
        print(f"   - All identical: {len(np.unique(preds_array)) == 1}\n")

    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    diagnose_loss_computation()
