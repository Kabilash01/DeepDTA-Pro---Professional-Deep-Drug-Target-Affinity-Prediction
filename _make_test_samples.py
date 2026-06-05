"""Generate test sample files for DeepDTA-Pro predictions.
Pulls the real kinase target sequences straight from the app so they match exactly.
"""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent

# Load SELECTIVITY_TARGETS dict from the app module without running Streamlit
app_path = ROOT / "src" / "web_interface" / "app.py"
src = app_path.read_text(encoding="utf-8")
# Extract just the SELECTIVITY_TARGETS literal by exec-ing a trimmed snippet
start = src.index("SELECTIVITY_TARGETS = {")
end = src.index("}", start) + 1
ns = {}
exec(src[start:end], ns)
TARGETS = ns["SELECTIVITY_TARGETS"]   # {name: sequence}

# Validated drug SMILES (kinase inhibitors + common references)
DRUGS = {
    "Imatinib":  "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
    "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4",
    "Erlotinib": "C#Cc1cccc(c1)Nc2ncnc3cc(c(cc23)OCCO)OCCO",
    "Sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Dasatinib": "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
    "Lapatinib": "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1",
    "Sunitinib": "CCN(CC)CCNC(=O)c1c(C)[nH]c(c1C)C=C1C(=O)Nc2ccc(F)cc21",
    "Aspirin":   "CC(=O)Oc1ccccc1C(=O)O",
}

# Build batch CSV: pair each drug with a chemically sensible primary target,
# plus a few cross-pairs so you can see selectivity differences.
target_names = list(TARGETS.keys())   # EGFR, BRAF, CDK2, VEGFR2, SRC

pairs = [
    ("Gefitinib", "EGFR"),    # gefitinib is an EGFR inhibitor (expect strong)
    ("Erlotinib", "EGFR"),    # EGFR inhibitor
    ("Lapatinib", "EGFR"),    # dual EGFR/HER2
    ("Sorafenib", "BRAF"),    # BRAF/VEGFR multikinase
    ("Sorafenib", "VEGFR2"),  # also hits VEGFR2
    ("Imatinib",  "SRC"),     # ABL/KIT; SRC family related
    ("Dasatinib", "SRC"),     # potent SRC/ABL inhibitor (expect strong)
    ("Sunitinib", "VEGFR2"),  # VEGFR multikinase
    ("Gefitinib", "CDK2"),    # off-target control (expect weaker)
    ("Aspirin",   "EGFR"),    # non-kinase-drug negative control (expect weak)
]

rows = []
for drug, tgt in pairs:
    rows.append({
        "compound_name":    drug,
        "drug_smiles":      DRUGS[drug],
        "target_name":      tgt,
        "protein_sequence": TARGETS[tgt],
    })

out_csv = ROOT / "test_samples.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["compound_name", "drug_smiles",
                                      "target_name", "protein_sequence"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {out_csv}  ({len(rows)} rows)")

# Also write a small README with single-prediction copy-paste pairs
readme = ROOT / "TEST_SAMPLES.md"
lines = [
    "# DeepDTA-Pro — Test Samples",
    "",
    "## Batch Prediction",
    "Upload `test_samples.csv` on the **Batch Prediction** page.",
    "Required columns: `drug_smiles`, `protein_sequence` (others are kept as metadata).",
    "",
    "## Single Prediction — copy/paste pairs",
    "",
]
for drug, tgt in pairs[:6]:
    seq = TARGETS[tgt]
    lines.append(f"### {drug} → {tgt}")
    lines.append(f"- **SMILES:** `{DRUGS[drug]}`")
    lines.append(f"- **Protein ({tgt}, {len(seq)} aa):**")
    lines.append("")
    lines.append("```")
    lines.append(seq)
    lines.append("```")
    lines.append("")

readme.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {readme}")
