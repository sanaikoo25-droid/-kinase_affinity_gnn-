import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from train_real_siamese_v2 import DistanceAwareEncoder, find_wt_graph, manifest, exp_ddg_table

raw_manifest = pd.read_csv("manifest.csv")

# Antisymmetric Siamese Architecture (Zero Bias -> f(0) = 0.000 strictly)
class AntisymmetricSiamese(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=64):
        super().__init__()
        self.encoder = DistanceAwareEncoder(in_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32, bias=False),
            nn.SiLU(),
            nn.Linear(32, 1, bias=False)
        )

    def forward(self, wt_g, mut_g):
        h_wt = self.encoder(wt_g)
        h_mut = self.encoder(mut_g)
        return self.head(h_mut - h_wt)

# Symmetrical pairing: WT->Mut is +dG, Mut->WT is -dG
training_pairs = []
for (target, mut, lig), ddg in exp_ddg_table.items():
    mut_rows = manifest[(manifest["target"] == target) & 
                        (manifest["mutation"] == mut) & 
                        (manifest["complex_id"].str.contains(lig, case=False, na=False))]
    if mut_rows.empty:
        continue
    lig_smiles = mut_rows.iloc[0]["ligand_smiles"]
    wt_path = find_wt_graph(target, lig_smiles)
    mut_path = mut_rows.iloc[0]["graph_path"]
    
    if wt_path and os.path.exists(wt_path) and os.path.exists(mut_path):
        training_pairs.append((wt_path, mut_path, ddg))
        training_pairs.append((mut_path, wt_path, -ddg))

model = AntisymmetricSiamese()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
criterion = nn.SmoothL1Loss()

print(f"Calibrating zero-intercept model on {len(training_pairs)} directional pairs...")
model.train()
for epoch in range(1, 81):
    total_loss = 0.0
    for g1_p, g2_p, target_val in training_pairs:
        g1 = torch.load(g1_p, weights_only=False)
        g2 = torch.load(g2_p, weights_only=False)
        y = torch.tensor([[target_val]], dtype=torch.float)

        optimizer.zero_grad()
        loss = criterion(model(g1, g2), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

model.eval()

predictions = []
with torch.no_grad():
    for _, row in raw_manifest.iterrows():
        cid = row["complex_id"]
        target = row["target"]
        mut = row["mutation"]
        lig_smiles = row["ligand_smiles"]
        lig_name = cid.split("_")[-1]

        if mut == "WT":
            predictions.append({
                "complex_id": cid,
                "target": target,
                "mutation": mut,
                "ligand": lig_name,
                "predicted_ddG": 0.000,
                "phenotype": "Reference (WT)"
            })
            continue

        wt_path = find_wt_graph(target, lig_smiles)
        mut_rows = manifest[(manifest["complex_id"] == cid)]
        if mut_rows.empty or not wt_path or not os.path.exists(wt_path):
            continue

        mut_path = mut_rows.iloc[0]["graph_path"]
        if not os.path.exists(mut_path):
            continue

        wt_g = torch.load(wt_path, weights_only=False)
        mut_g = torch.load(mut_path, weights_only=False)

        pred_ddg = model(wt_g, mut_g).item()

        predictions.append({
            "complex_id": cid,
            "target": target,
            "mutation": mut,
            "ligand": lig_name,
            "predicted_ddG": round(pred_ddg, 3),
            "phenotype": "Resistant" if pred_ddg >= 1.2 else ("Sensitive" if pred_ddg <= -0.4 else "Neutral")
        })

df_res = pd.DataFrame(predictions)
df_res.to_csv("full_screen_calibrated.csv", index=False)
print(f"\nScreening complete across {len(df_res)} complexes.")
print(df_res.head(15).to_string(index=False))
