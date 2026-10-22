# Kinase Resistance Landscapes via Antisymmetric Siamese GNN

This repository provides the computational pipeline, model architectures, and screening datasets for predicting kinase inhibitor resistance shifts ($\Delta\Delta G$) using Boltz-generated structural ensembles and distance-aware Siamese Graph Neural Networks.

## Repository Structure
- `train_real_siamese_v2.py`: Model architecture and training script enforcing physical antisymmetry.
- `screen_all_complexes_v2.py`: Inference pipeline for full panel screening.
- `manifest.csv`: Dataset manifest mapping complexes to conformational graphs.
- `full_screen_calibrated.csv`: Calibrated predictions across all 52 kinase complexes.
- `structural_alignments/`: Aligned representative PDB structures for gatekeeper clash inspection.

## Quick Start
```bash
# Clone the repository
git clone [https://github.com/sanaikoo25/kinase_affinity_gnn.git](https://github.com/sanaikoo25/kinase_affinity_gnn.git)
cd kinase_affinity_gnn

# Run calibrated screening
python screen_all_complexes_v2.py

doi = doi.org/10.26434/chemrxiv.15009874/v1
