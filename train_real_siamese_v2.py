import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

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
        # Sum pooling preserves total pocket contact energy better than mean pooling
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
        # Antisymmetric delta representation
        delta_h = h_mut - h_wt
        return self.head(delta_h)

# Compile training pairs
training_pairs = []
y_all = []
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
        training_pairs.append((wt_path, mut_path, ddg, (target, mut, lig)))
        y_all.append(ddg)

y_mean = np.mean(y_all)
y_std = np.std(y_all) if np.std(y_all) > 1e-4 else 1.0

model = SiameseResidualV2()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
criterion = nn.SmoothL1Loss()

print(f"Training Distance-Aware Siamese Network on {len(training_pairs)} pairs...")
model.train()
for epoch in range(1, 81):
    total_loss = 0.0
    for wt_p, mut_p, target_ddg, _ in training_pairs:
        wt_g = torch.load(wt_p)
        mut_g = torch.load(mut_p)
        norm_target = torch.tensor([[(target_ddg - y_mean) / y_std]], dtype=torch.float)

        optimizer.zero_grad()
        pred = model(wt_g, mut_g)
        loss = criterion(pred, norm_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:02d}/80 - Loss: {total_loss / len(training_pairs):.4f}")

# Evaluation
model.eval()
records = []
y_true, y_pred = [], []

with torch.no_grad():
    for wt_p, mut_p, target_val, meta in training_pairs:
        wt_g = torch.load(wt_p)
        mut_g = torch.load(mut_p)
        raw_pred = model(wt_g, mut_g).item()
        unnorm_pred = raw_pred * y_std + y_mean

        y_true.append(target_val)
        y_pred.append(unnorm_pred)
        records.append({
            "target": meta[0],
            "mutation": meta[1],
            "ligand": meta[2],
            "exp_ddG": target_val,
            "pred_ddG": round(unnorm_pred, 3)
        })

res_df = pd.DataFrame(records)
res_df.to_csv("real_structure_predictions_v2.csv", index=False)

r, _ = pearsonr(y_true, y_pred)
rho, _ = spearmanr(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n=== DISTANCE-AWARE 3D GNN EVALUATION ===")
print(f"Pearson Correlation (R): {r:.3f}")
print(f"Spearman Rank (rho)    : {rho:.3f}")
print(f"RMSE                   : {rmse:.3f} kcal/mol")
print("\nPredictions:")
print(res_df.to_string(index=False))
