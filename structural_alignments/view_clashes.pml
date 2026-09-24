# PyMOL Inspection Script for Kinase Gatekeeper Steric Mechanism
reinitialize

# 1. Load ABL1 complexes
load structural_alignments/ABL1_WT_imatinib.pdb, abl1_wt_imat
load structural_alignments/ABL1_T315I_imatinib.pdb, abl1_t315i_imat
load structural_alignments/ABL1_T315I_ponatinib.pdb, abl1_t315i_ponat

# Align on backbone C-alpha
align abl1_t315i_imat, abl1_wt_imat
align abl1_t315i_ponat, abl1_wt_imat

# Display styling
hide everything
show cartoon, chain A
show sticks, chain B
show sticks, (resi 315 and chain A)

# Color coding
color gray80, abl1_wt_imat
color salmon, abl1_t315i_imat
color lightblue, abl1_t315i_ponat

color yellow, abl1_wt_imat and chain B
color firebrick, abl1_t315i_imat and chain B
color cyan, abl1_t315i_ponat and chain B

# Highlight gatekeeper residue 315
show spheres, (resi 315 and chain A)
set sphere_scale, 0.45

zoom (resi 315 and chain A), 8
center (resi 315 and chain A)
