import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

# Experimental ΔΔG benchmark pairs
exp_ddg_table = {
    ("ABL1", "M244V", "axitinib"): -0.11,
    ("ABL1", "M244V", "bosutinib"): 0.43,
    ("ABL1", "M244V", "dasatinib"): -0.18,
    ("ABL1", "M244V", "imatinib"): 0.35,
    ("ABL1", "M244V", "nilotinib"): 0.20,
    ("ABL1", "M244V", "ponatinib"): 0.00,
    ("ABL1", "T315I", "axitinib"): 1.50,
    ("ABL1", "T315I", "bosutinib"): 2.80,
    ("ABL1", "T315I", "dasatinib"): 2.45,
    ("ABL1", "T315I", "imatinib"): 3.80,
    ("ABL1", "T315I", "nilotinib"): 3.10,
    ("ABL1", "T315I", "ponatinib"): -0.10,
    ("ABL1", "F317L", "dasatinib"): 1.20,
    ("ABL1", "F317L", "imatinib"): 0.60,
    ("ABL1", "E255K", "imatinib"): 2.10,
    ("ABL1", "E255K", "dasatinib"): 0.40,
    ("EGFR", "T790M", "gefitinib"): 2.30,
    ("EGFR", "T790M", "osimertinib"): -0.40,
    ("EGFR", "L858R", "gefitinib"): -0.80,
    ("EGFR", "C797S", "osimertinib"): 2.10
}

manifest = pd.read_csv("graph_manifest.csv")

def find_wt_graph(target, ligand_smiles):
    matches = manifest[(manifest["target"] == target) &
                       (manifest["mutation"] == "WT") &
                       (manifest["ligand_smiles"] == ligand_smiles)]
    if not matches.empty:
        return matches.iloc[0]["graph_path"]
    wt_fallback = manifest[(manifest["target"] == target) & (manifest["mutation"] == "WT")]
    if not wt_fallback.empty:
        return wt_fallback.iloc[0]["graph_path"]
    return None

class DistanceConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, pos):
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], p=2, dim=-1, keepdim=True)
        return self.propagate(edge_index, x=x, dist=dist)

    def message(self, x_i, x_j, dist):
        return self.msg_mlp(torch.cat([x_i, x_j, dist], dim=-1))

class DistanceAwareEncoder(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=64):
        super().__init__()
        self.conv1 = DistanceConv(in_dim, hidden_dim)
        self.conv2 = DistanceConv(hidden_dim, hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        h = F.silu(self.conv1(data.x, data.edge_index, data.pos))
        h = F.silu(self.conv2(h, data.edge_index, data.pos))
        h = self.node_proj(h)
        return torch.sum(h, dim=0, keepdim=True)

class SiameseResidualV2(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=64):
        super().__init__()
        self.encoder = DistanceAwareEncoder(in_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, wt_g, mut_g):
        h_wt = self.encoder(wt_g)
        h_mut = self.encoder(mut_g)
        delta_h = h_mut - h_wt
        return self.head(delta_h)

# Compile dataset pairs
dataset = []
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
        dataset.append({
            "wt_path": wt_path,
            "mut_path": mut_path,
            "ddg": ddg,
            "target": target,
            "mutation": mut,
            "ligand": lig
        })

print(f"Loaded {len(dataset)} valid experimental pairs for cross-validation.")

# 5-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = []

print("\nStarting 5-Fold Cross-Validation (Held-out Out-of-Fold Evaluation)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    train_ddgs = [d["ddg"] for d in train_data]
    y_mean = np.mean(train_ddgs)
    y_std = np.std(train_ddgs) if np.std(train_ddgs) > 1e-4 else 1.0

    fold_model = SiameseResidualV2()
    optimizer = torch.optim.AdamW(fold_model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()

    # Train strictly on the training fold
    fold_model.train()
    for epoch in range(1, 81):
        for item in train_data:
            wt_g = torch.load(item["wt_path"])
            mut_g = torch.load(item["mut_path"])
            norm_target = torch.tensor([[(item["ddg"] - y_mean) / y_std]], dtype=torch.float)

            optimizer.zero_grad()
            pred = fold_model(wt_g, mut_g)
            loss = criterion(pred, norm_target)
            loss.backward()
            optimizer.step()

    # Evaluate strictly on unseen held-out validation fold
    fold_model.eval()
    with torch.no_grad():
        for item in val_data:
            wt_g = torch.load(item["wt_path"])
            mut_g = torch.load(item["mut_path"])
            raw_pred = fold_model(wt_g, mut_g).item()
            unnorm_pred = raw_pred * y_std + y_mean

            oof_predictions.append({
                "fold": fold + 1,
                "target": item["target"],
                "mutation": item["mutation"],
                "ligand": item["ligand"],
                "exp_ddG": item["ddg"],
                "pred_ddG": float(unnorm_pred)
            })

oof_df = pd.DataFrame(oof_predictions)
oof_df.to_csv("cross_validation_results.csv", index=False)

# Calculate genuine out-of-fold generalization metrics
y_true = oof_df["exp_ddG"].values
y_pred = oof_df["pred_ddG"].values

r_cv, _ = pearsonr(y_true, y_pred)
rho_cv, _ = spearmanr(y_true, y_pred)
rmse_cv = np.sqrt(mean_squared_error(y_true, y_pred))
mae_cv = mean_absolute_error(y_true, y_pred)

print("\n=======================================================")
print("=== HELD-OUT 5-FOLD CROSS-VALIDATION RESULTS ===")
print("=======================================================")
print(f"Test Pearson Correlation (R) : {r_cv:.3f}")
print(f"Test Spearman Rank (rho)     : {rho_cv:.3f}")
print(f"Test RMSE                    : {rmse_cv:.3f} kcal/mol")
print(f"Test MAE                     : {mae_cv:.3f} kcal/mol")
print("=======================================================\n")

# Train production model on 100% of data for the full panel screen
print("Training final production weights on full dataset...")
all_ddgs = [d["ddg"] for d in dataset]
full_y_mean = np.mean(all_ddgs)
full_y_std = np.std(all_ddgs) if np.std(all_ddgs) > 1e-4 else 1.0

prod_model = SiameseResidualV2()
prod_opt = torch.optim.AdamW(prod_model.parameters(), lr=0.001, weight_decay=1e-3)
criterion = nn.SmoothL1Loss()

prod_model.train()
for epoch in range(1, 81):
    for item in dataset:
        wt_g = torch.load(item["wt_path"])
        mut_g = torch.load(item["mut_path"])
        norm_t = torch.tensor([[(item["ddg"] - full_y_mean) / full_y_std]], dtype=torch.float)

        prod_opt.zero_grad()
        pred = prod_model(wt_g, mut_g)
        loss = criterion(pred, norm_t)
        loss.backward()
        prod_opt.step()

torch.save(prod_model.state_dict(), "siamese_production_model.pt")
print("Saved siamese_production_model.pt and cross_validation_results.csv successfully.")
