"""Run all test_samples.csv pairs through the trained Phase 5 model
and print expected pKd values."""
import csv
from pathlib import Path
import torch

from gml_core import MolecularGraphBuilder, AA_VOCAB, ATOM_FEAT_DIM
from phase5_multitask_learning import MultiTaskGNNDTA
from target_normalizer import AffinityNormalizer

ROOT = Path(__file__).parent
CKPT = ROOT / "models" / "checkpoints" / "phase5_best_model.pth"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ckpt.get("config", {})
model = MultiTaskGNNDTA(
    mol_in_dim=ATOM_FEAT_DIM,
    gnn_hidden=cfg.get("gnn_hidden", 192), gnn_layers=cfg.get("gnn_layers", 5),
    prot_embed=cfg.get("prot_embed", 128), prot_hidden=cfg.get("prot_hidden", 192),
    prot_layers=cfg.get("prot_layers", 4), fusion_hidden=cfg.get("fusion_hidden", 192),
    dropout=cfg.get("dropout", 0.1), bond_cnn_dim=cfg.get("bond_cnn_dim", 32),
)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
norm = AffinityNormalizer(mean=5.7512, std=1.1676)
mb = MolecularGraphBuilder()


def prot_tensor(seq, MAX=1200):
    seq = seq.upper()[:MAX]
    ids = [AA_VOCAB.get(a, 0) for a in seq] + [0] * (MAX - len(seq))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def predict(smiles, protein):
    d = mb.smiles_to_pyg(smiles)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
    with torch.no_grad():
        out = model(d, prot_tensor(protein))
        return float(norm.denormalize(out[0].squeeze().item()))


rows = list(csv.DictReader((ROOT / "test_samples.csv").open(encoding="utf-8")))
print(f"{'Drug':<12}{'Target':<8}{'pred pKd':>10}  Binding")
print("-" * 44)
for r in rows:
    pkd = predict(r["drug_smiles"], r["protein_sequence"])
    cls = "Strong" if pkd >= 7 else ("Moderate" if pkd >= 5 else "Weak")
    print(f"{r['compound_name']:<12}{r['target_name']:<8}{pkd:>10.3f}  {cls}")
