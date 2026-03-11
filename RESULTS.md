# Functional Group Analysis — Experiment Results

## 1. Dataset Summary

| Property | Value |
|---|---|
| Source | ZINC15 database |
| Total molecules in file | 249455 |
| Molecules used | 249455 |
| Successfully parsed | 249455 (100.0%) |
| Parse failures | 0 |

### Molecular Property Distributions

| Property | Mean | Std | Min | Max |
|---|---|---|---|---|
| QED | 0.7283 | 0.1396 | 0.1118 | 0.9479 |
| logP | 2.4571 | 1.4343 | — | — |
| SAS | 3.0532 | 0.8348 | — | — |

![Property Distributions](figures/property_distributions_combined.svg)

*Figure 1: Distribution of QED, logP, and SAS across the full dataset. Red vertical lines indicate means.*

| | | |
|---|---|---|
| ![QED](figures/qed_distribution.svg) | ![logP](figures/logp_distribution.svg) | ![SAS](figures/sas_distribution.svg) |

*Figure 2: Individual property distributions with 50-bin histograms and mean indicators.*

## 2. Molecular Graph Statistics

| Property | Value |
|---|---|
| Atoms per molecule | 23.2 (range: 6–38) |
| Bonds per molecule | 24.9 (range: 5–45) |
| Total atoms processed | 5775223 |
| Total bonds processed | 6211984 |
| Node feature dimension | 29 |
| Edge feature dimension | 9 |

![Molecular Complexity](figures/molecule_complexity.svg)

*Figure 3: Molecular graph complexity — atoms vs. bonds colored by QED score (red=low, green=high).*

## 3. Functional Group Census

Detection of 22 functional group types across the entire dataset.

| Functional Group | Molecules | Prevalence (%) | Total Count | Mean per Mol |
|---|---|---|---|---|
| Phenyl (aromatic ring) | 207022 | 83.0 | 604208 | 2.42 |
| Amide (-CONH-) | 169695 | 68.0 | 221905 | 0.89 |
| Heterocycle | 144791 | 58.0 | 338210 | 1.36 |
| Ether (C-O-C) | 93022 | 37.3 | 122335 | 0.49 |
| Halide (C-X) | 87641 | 35.1 | 135900 | 0.54 |
| Secondary Amine (>NH) | 68322 | 27.4 | 73221 | 0.29 |
| Tertiary Amine (>N<) | 53797 | 21.6 | 59952 | 0.24 |
| Hydroxyl (-OH) | 27705 | 11.1 | 29863 | 0.12 |
| Thioether (C-S-C) | 27342 | 11.0 | 28238 | 0.11 |
| Sulfonyl (-SO₂-) | 27162 | 10.9 | 27887 | 0.11 |
| Ketone (>C=O) | 26296 | 10.5 | 28552 | 0.11 |
| Ester (-COO-) | 18118 | 7.3 | 18687 | 0.07 |
| Primary Amine (-NH₂) | 17796 | 7.1 | 18199 | 0.07 |
| Nitrile (-C≡N) | 12932 | 5.2 | 13595 | 0.05 |
| Nitro (-NO₂) | 10665 | 4.3 | 10944 | 0.04 |
| Carboxyl (-COOH) | 9373 | 3.8 | 9552 | 0.04 |
| Imine (C=N) | 6648 | 2.7 | 6877 | 0.03 |
| Sulfoxide (-SO-) | 2220 | 0.9 | 2224 | 0.01 |
| Aldehyde (-CHO) | 711 | 0.3 | 715 | 0.00 |
| Thiol (-SH) | 465 | 0.2 | 466 | 0.00 |
| Epoxide | 128 | 0.1 | 130 | 0.00 |
| Phosphate (-PO₄) | 79 | 0.0 | 81 | 0.00 |

### Functional Group Co-occurrence Patterns

Average number of distinct functional group types per molecule and distribution.

- **Functional group types detected**: 22 out of 22
- **Ubiquitous groups** (>50%): Ph (83%), CONH (68%), HetCyc (58%)
- **Rare groups** (<5%): NO2 (4.3%), COOH (3.8%), C=N (2.7%), SO (0.9%), CHO (0.3%), SH (0.2%), Epox (0.1%), PO4 (0.0%)

![FG Prevalence](figures/fg_prevalence.svg)

*Figure 4: Functional group prevalence across the dataset. Blue (>50%), green (10–50%), purple (<10%).*

## 4. Model Configuration

### VGAE Architecture

| Component | Configuration |
|---|---|
| GNN type | Graph Attention Network (GAT) |
| GNN layers | 3 |
| Hidden dimension | 64 |
| GNN output dimension | 32 |
| Latent dimension | 16 |
| Activation | ReLU (residual connections) |
| Pooling | Global attention pooling |
| Decoder | 16 → 64 → 128 → 29 (ReLU) |
| KL weight (β) | 0.001 |

### SOM Configuration

| Parameter | Value |
|---|---|
| Grid size | 10×10 (100 neurons) |
| Training epochs | 128 |
| Initial learning rate | 0.5 |
| Initial radius | 5.0 |
| Distance metric | Euclidean |
| Neighborhood | Gaussian |

## 5. VGAE Encoding Results

| Metric | Value |
|---|---|
| Mean reconstruction loss | 0.156732 |
| Mean pairwise embedding distance | 0.103368 |
| Embedding std (mean across dims) | 0.018115 |
| Embedding std range | [0.0050, 0.0397] |

### Latent Dimension Statistics

| Dim | Mean | Std | Min | Max |
|---|---|---|---|---|
| 0 | 0.2038 | 0.0358 | 0.0833 | 0.3352 |
| 1 | 0.3947 | 0.0278 | 0.2724 | 0.5122 |
| 2 | -0.2115 | 0.0155 | -0.2993 | -0.1193 |
| 3 | -0.2590 | 0.0175 | -0.3286 | -0.1551 |
| 4 | 0.1652 | 0.0157 | 0.0741 | 0.2312 |
| 5 | -0.3179 | 0.0144 | -0.3719 | -0.2206 |
| 6 | 0.0759 | 0.0050 | 0.0412 | 0.1032 |
| 7 | -0.1293 | 0.0075 | -0.1688 | -0.0903 |
| 8 | -0.0520 | 0.0157 | -0.1469 | 0.0336 |
| 9 | 0.2941 | 0.0218 | 0.1635 | 0.3733 |
| 10 | 0.1550 | 0.0130 | 0.0784 | 0.2080 |
| 11 | -0.0822 | 0.0129 | -0.1813 | -0.0026 |
| 12 | -0.0764 | 0.0209 | -0.1780 | -0.0264 |
| 13 | 0.1387 | 0.0397 | 0.0029 | 0.2753 |
| 14 | 0.3109 | 0.0151 | 0.1978 | 0.3597 |
| 15 | 0.0556 | 0.0115 | -0.0238 | 0.1108 |

![Reconstruction Loss](figures/reconstruction_loss_dist.svg)

*Figure 5: Distribution of VGAE reconstruction losses across all molecules.*

![Embedding Variance](figures/embedding_dim_variance.svg)

*Figure 6: Variance of each latent dimension — higher variance indicates more discriminative dimensions.*

## 6. Feature Importance Analysis

### 6.1 Latent Dimension ↔ Property Correlations

Pearson correlation (r) between each latent dimension and molecular properties.
Dimensions sorted by |r(QED)|.

| Dim | Variance | r(QED) | r(logP) | r(SAS) |
|---|---|---|---|---|
| 12 | 0.000438 | -0.2893 | +0.3730 | -0.5936 |
| 8 | 0.000245 | +0.2879 | -0.2350 | +0.3766 |
| 13 | 0.001576 | -0.2503 | +0.4888 | -0.5603 |
| 0 | 0.001281 | -0.2467 | +0.4815 | -0.5584 |
| 15 | 0.000132 | +0.2201 | -0.1381 | +0.3621 |
| 14 | 0.000228 | +0.1849 | +0.0636 | +0.2645 |
| 2 | 0.000241 | -0.1614 | +0.1637 | -0.2895 |
| 5 | 0.000209 | +0.1482 | -0.5524 | +0.4279 |
| 9 | 0.000477 | -0.1027 | +0.5191 | -0.3603 |
| 4 | 0.000247 | -0.0873 | +0.3429 | -0.2120 |
| 3 | 0.000307 | -0.0673 | -0.2681 | -0.0706 |
| 6 | 0.000025 | -0.0654 | -0.0919 | -0.1093 |
| 7 | 0.000056 | -0.0604 | +0.3897 | -0.1776 |
| 1 | 0.000771 | +0.0481 | +0.1537 | +0.1098 |
| 10 | 0.000169 | -0.0467 | +0.2524 | -0.1299 |
| 11 | 0.000167 | +0.0211 | -0.2831 | +0.1471 |

### 6.2 Functional Group ↔ Latent Space Encoding

Which latent dimensions best encode each functional group's presence.

| Functional Group | Prevalence (%) | Best Dim | |r| |
|---|---|---|---|
| Phenyl (aromatic ring) | 83.0 | 13 | 0.5595 |
| Heterocycle | 58.0 | 8 | 0.5011 |
| Nitrile (-C≡N) | 5.2 | 11 | 0.3789 |
| Amide (-CONH-) | 68.0 | 14 | 0.3779 |
| Sulfonyl (-SO₂-) | 10.9 | 14 | 0.3445 |
| Halide (C-X) | 35.1 | 5 | 0.2759 |
| Tertiary Amine (>N<) | 21.6 | 12 | 0.2629 |
| Ketone (>C=O) | 10.5 | 14 | 0.2413 |
| Ester (-COO-) | 7.3 | 14 | 0.2098 |
| Carboxyl (-COOH) | 3.8 | 4 | 0.1879 |
| Secondary Amine (>NH) | 27.4 | 14 | 0.1678 |
| Hydroxyl (-OH) | 11.1 | 8 | 0.1651 |
| Imine (C=N) | 2.7 | 14 | 0.1416 |
| Ether (C-O-C) | 37.3 | 12 | 0.1392 |
| Nitro (-NO₂) | 4.3 | 14 | 0.1322 |
| Primary Amine (-NH₂) | 7.1 | 11 | 0.1187 |
| Thioether (C-S-C) | 11.0 | 8 | 0.0586 |

### 6.3 Functional Group ↔ Molecular Property Correlations

Point-biserial correlation between FG presence and drug-likeness properties.

| Functional Group | r(QED) | r(logP) | r(SAS) |
|---|---|---|---|
| Phenyl (aromatic ring) | -0.0659 | +0.3974 | -0.4270 |
| Nitro (-NO₂) | -0.3206 | +0.0387 | -0.0730 |
| Amide (-CONH-) | +0.0240 | +0.0871 | -0.2900 |
| Carboxyl (-COOH) | +0.0013 | -0.2825 | +0.1299 |
| Halide (C-X) | +0.0063 | +0.2589 | -0.1633 |
| Imine (C=N) | -0.2247 | +0.0123 | -0.0266 |
| Ketone (>C=O) | -0.2142 | +0.0478 | -0.0618 |
| Secondary Amine (>NH) | +0.0615 | -0.1020 | +0.1774 |
| Primary Amine (-NH₂) | -0.0211 | -0.1642 | +0.1060 |
| Thioether (C-S-C) | -0.1537 | +0.1195 | -0.0027 |
| Ester (-COO-) | -0.1460 | +0.0148 | -0.0566 |
| Heterocycle | -0.0764 | +0.1292 | -0.0993 |
| Tertiary Amine (>N<) | +0.0680 | -0.1164 | +0.0337 |
| Hydroxyl (-OH) | +0.0210 | -0.0987 | +0.1057 |
| Sulfonyl (-SO₂-) | +0.0185 | -0.0605 | -0.0869 |
| Ether (C-O-C) | +0.0350 | +0.0456 | -0.0550 |
| Nitrile (-C≡N) | +0.0048 | +0.0375 | -0.0130 |

![Dim-Property Heatmap](figures/dim_property_heatmap.svg)

*Figure 7: Heatmap of Pearson correlations between latent dimensions and molecular properties. Blue = negative, red = positive.*

![FG-Property Correlations](figures/fg_property_correlations.svg)

*Figure 8: Point-biserial correlations between functional group presence and drug-likeness properties.*

## 7. Stratified Clustering Results

### Per-Stratum Overview

| Stratum | QED Range | Molecules | Active Clusters | QE | U-Matrix Mean | U-Matrix Max |
|---|---|---|---|---|---|---|
| 0 | [0, 0.399) | 6830 | 100 | 0.032171 | 0.0116 | 0.0177 |
| 1 | [0.399, 0.520) | 17622 | 100 | 0.032613 | 0.0119 | 0.0176 |
| 2 | [0.520, 0.694) | 60427 | 100 | 0.035004 | 0.0127 | 0.0190 |
| 3 | [0.694, 0.814) | 83673 | 100 | 0.035135 | 0.0121 | 0.0163 |
| 4 | [0.814, 1.0] | 80903 | 100 | 0.031593 | 0.0095 | 0.0136 |

**Total clustered**: 249455 molecules | **Avg QE**: 0.033303

![Latent Space PCA](figures/latent_space_pca.svg)

*Figure 9: PCA projection of 16-dimensional VGAE embeddings colored by QED stratum.*

![Stratum Properties](figures/stratum_property_comparison.svg)

*Figure 10: Mean ± std of molecular properties across QED strata.*

![U-Matrix](figures/umatrix_heatmaps.svg)

*Figure 11: SOM U-matrix heatmaps showing topological organization per stratum. Darker regions indicate cluster boundaries.*

### Stratum 0 ([0, 0.399)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 92.6 | 3.80 |
| Heterocycle | 69.1 | 1.97 |
| Amide (-CONH-) | 61.6 | 0.78 |
| Halide (C-X) | 36.2 | 0.56 |
| Ether (C-O-C) | 34.1 | 0.47 |
| Ketone (>C=O) | 31.6 | 0.36 |
| Thioether (C-S-C) | 31.4 | 0.32 |
| Nitro (-NO₂) | 29.6 | 0.30 |
| Secondary Amine (>NH) | 19.5 | 0.25 |
| Imine (C=N) | 19.2 | 0.20 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 99 | 557 | 0.345±0.047 | 4.51 | 2.46 | 0.0287 | Ph | HetCyc(1.4×), C-S-C(1.2×) | `O=C(Cc1n[nH]c(=O)c2ccccc12)NCc` |
| 0 | 337 | 0.327±0.052 | -0.31 | 4.20 | 0.0704 | NH2 | COOH(4.6×), NH2(3.4×), OH(3.2×) | `C[C@H]1OCC[C@H]1C(=O)NC(C)(C)/` |
| 94 | 184 | 0.345±0.053 | 4.39 | 2.79 | 0.0235 | Ph | N<(2.6×), C=O(1.7×), C-O-C(1.4×) | `CCCCCOc1ccc(/C=c2\sc3nc([C@@H]` |
| 49 | 173 | 0.336±0.056 | 3.68 | 2.49 | 0.0261 | Ph | SO2(2.3×), NO2(1.5×), HetCyc(1.3×) | `CC(=O)Nc1ccc(NC(=O)c2sc3nc(-c4` |
| 69 | 166 | 0.331±0.055 | 4.05 | 2.49 | 0.0233 | Ph | HetCyc(1.4×), SO2(1.2×), NO2(1.2×) | `Cc1ccc2sc(N(Cc3ccccn3)C(=O)c3c` |
| 59 | 165 | 0.330±0.056 | 3.93 | 2.45 | 0.0232 | Ph | HetCyc(1.4×), SO2(1.3×), C=O(1.2×) | `CC(=O)c1ccc(Nc2ncnc(Nc3ccccc3C` |
| 50 | 158 | 0.334±0.059 | 3.06 | 3.23 | 0.0320 | Ph | COOH(2.6×), N<(1.9×), COO(1.6×) | `O=C(/C=C/c1ccc([N+](=O)[O-])cc` |
| 2 | 156 | 0.338±0.042 | 1.17 | 3.07 | 0.0370 | Ph | NH2(3.6×), OH(2.2×), COOH(2.2×) | `Cc1ccc([N+](=O)[O-])cc1OCC(C)(` |
| 95 | 152 | 0.352±0.045 | 4.61 | 2.77 | 0.0211 | Ph | N<(1.9×), C-O-C(1.7×), HetCyc(1.4×) | `O=C(CNC(=O)c1ccc(Oc2ccccc2)cc1` |
| 5 | 136 | 0.335±0.047 | 2.45 | 2.60 | 0.0293 | Ph | NO2(2.2×), CN(2.1×), COOH(2.0×) | `O=C(CCC(=O)Nc1ccc(F)cc1)N/N=C\` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 17 | 18 | 0.008147 |
| 19 | 29 | 0.008209 |
| 47 | 48 | 0.008352 |
| 77 | 87 | 0.008387 |
| 77 | 78 | 0.008483 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 0 | 99 | 0.282837 |
| 0 | 98 | 0.260336 |
| 0 | 89 | 0.259392 |
| 0 | 97 | 0.258419 |
| 0 | 79 | 0.252227 |

Inter-cluster distance: mean=0.072156, min=0.008147, max=0.282837, 4950 pairs

### Stratum 1 ([0.399, 0.520)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 91.5 | 3.49 |
| Heterocycle | 68.5 | 1.98 |
| Amide (-CONH-) | 65.7 | 0.85 |
| Halide (C-X) | 36.5 | 0.55 |
| Ether (C-O-C) | 36.0 | 0.50 |
| Ketone (>C=O) | 24.0 | 0.27 |
| Secondary Amine (>NH) | 21.2 | 0.27 |
| Thioether (C-S-C) | 20.5 | 0.21 |
| Nitro (-NO₂) | 17.5 | 0.18 |
| Tertiary Amine (>N<) | 15.8 | 0.18 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 99 | 1336 | 0.470±0.035 | 4.13 | 2.42 | 0.0303 | Ph | HetCyc(1.4×) | `O=C(NCc1ccco1)c1ccc2ncc3c(O)n(` |
| 0 | 856 | 0.472±0.035 | -0.23 | 4.30 | 0.0660 | CONH | COOH(5.8×), OH(2.6×), NH2(1.9×) | `COCCCNC(=O)C(=O)NC[C@@H]1CCCO1` |
| 94 | 482 | 0.468±0.033 | 3.83 | 2.47 | 0.0243 | Ph | CN(1.6×), HetCyc(1.4×), C-S-C(1.2×) | `Cc1nc(-c2ccccn2)sc1C(=O)NCCc1n` |
| 50 | 443 | 0.465±0.031 | 2.49 | 2.64 | 0.0302 | Ph | NO2(2.7×), C=N(2.0×), CN(1.9×) | `CC[C@H](NC(=O)Nc1ccc(C)c(NC(C)` |
| 95 | 388 | 0.473±0.034 | 3.89 | 2.47 | 0.0228 | Ph | HetCyc(1.4×), CN(1.4×), C-X(1.3×) | `Cc1cc(C)cc(-c2nnc(Sc3cc(C)c([N` |
| 49 | 387 | 0.463±0.035 | 3.74 | 2.88 | 0.0227 | Ph | N<(2.6×), HetCyc(1.3×), C-O-C(1.2×) | `COC(=O)c1ccc2c(=O)n(-c3ccc(F)c` |
| 59 | 378 | 0.460±0.033 | 3.91 | 2.81 | 0.0233 | Ph | N<(2.3×), HetCyc(1.4×) | `COc1ccccc1-n1nc(C(=O)N2CCN(c3c` |
| 96 | 337 | 0.469±0.034 | 3.93 | 2.43 | 0.0243 | Ph | HetCyc(1.4×), C-S-C(1.3×), C-X(1.2×) | `Cc1cccnc1Nc1ncnc(NNC(=O)c2ccc(` |
| 30 | 323 | 0.467±0.029 | 1.73 | 2.82 | 0.0365 | Ph | NH2(2.2×), COOH(2.1×), CN(2.0×) | `CCCN(CC)C(=O)c1cc([N+](=O)[O-]` |
| 93 | 318 | 0.468±0.032 | 3.63 | 2.56 | 0.0241 | Ph | NH2(1.6×), CN(1.6×), HetCyc(1.4×) | `COc1ccc(NC(=O)[C@H](C)Sc2nnc3c` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 19 | 29 | 0.007799 |
| 18 | 27 | 0.008529 |
| 73 | 82 | 0.008876 |
| 28 | 37 | 0.009022 |
| 77 | 87 | 0.009062 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 0 | 99 | 0.289487 |
| 0 | 89 | 0.271236 |
| 0 | 98 | 0.262679 |
| 0 | 79 | 0.262329 |
| 0 | 88 | 0.255567 |

Inter-cluster distance: mean=0.074247, min=0.007799, max=0.289487, 4950 pairs

### Stratum 2 ([0.520, 0.694)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 84.5 | 2.77 |
| Amide (-CONH-) | 68.0 | 0.91 |
| Heterocycle | 61.7 | 1.55 |
| Ether (C-O-C) | 35.0 | 0.48 |
| Halide (C-X) | 34.9 | 0.54 |
| Secondary Amine (>NH) | 25.3 | 0.28 |
| Tertiary Amine (>N<) | 20.0 | 0.22 |
| Ketone (>C=O) | 14.4 | 0.16 |
| Thioether (C-S-C) | 13.2 | 0.14 |
| Sulfonyl (-SO₂-) | 11.4 | 0.12 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 99 | 4340 | 0.601±0.054 | 3.48 | 2.41 | 0.0362 | Ph | HetCyc(1.5×) | `Cc1nn(-c2ccccc2)c2sc(C(=O)NN(C` |
| 0 | 4278 | 0.628±0.047 | 0.08 | 4.31 | 0.0541 | CONH | COOH(4.0×), NH2(2.5×), OH(2.2×) | `CC(=O)N1CC[C@@H](COCC[NH3+])C1` |
| 49 | 1686 | 0.622±0.045 | 3.53 | 2.86 | 0.0248 | Ph | N<(2.2×), HetCyc(1.4×), C-O-C(1.2×) | `Cc1ccc(-c2ccc3c(c2)C[C@H](CNC(` |
| 96 | 1619 | 0.626±0.050 | 3.43 | 2.44 | 0.0247 | Ph | CN(1.5×), HetCyc(1.5×), C-S-C(1.3×) | `CCc1ccc(-c2nc(N)ccc2[N+](=O)[O` |
| 95 | 1596 | 0.625±0.049 | 3.30 | 2.48 | 0.0257 | Ph | HetCyc(1.5×), CN(1.4×), C-S-C(1.3×) | `CCCc1cc(NC(=O)Cc2csc(Cc3ccccc3` |
| 59 | 1546 | 0.631±0.045 | 3.66 | 2.80 | 0.0246 | Ph | N<(1.8×), HetCyc(1.4×) | `CCCc1nc(-n2cccc2)sc1C(=O)N[C@@` |
| 40 | 1436 | 0.623±0.044 | 1.35 | 3.07 | 0.0365 | Ph | NO2(2.1×), COOH(2.0×), NH2(2.0×) | `CC(C)CCC[C@H](C)NC(=O)C(=O)Nc1` |
| 94 | 1403 | 0.628±0.048 | 3.12 | 2.51 | 0.0276 | Ph | HetCyc(1.4×), C=N(1.4×), CN(1.4×) | `CCc1ccc(-c2nc(CC(=O)NCc3cn(C)n` |
| 97 | 1336 | 0.622±0.053 | 3.49 | 2.41 | 0.0257 | Ph | HetCyc(1.5×), CN(1.3×), C-S-C(1.2×) | `COc1ccc(NC(=O)Nc2nnc(CCc3ccccc` |
| 50 | 1279 | 0.620±0.044 | 1.52 | 2.87 | 0.0381 | Ph | COOH(2.7×), NH2(2.1×), COO(1.8×) | `COc1ccc(Br)cc1/C=N/NC(=O)C[C@H` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 29 | 39 | 0.008017 |
| 28 | 37 | 0.008256 |
| 17 | 26 | 0.008925 |
| 19 | 29 | 0.008936 |
| 67 | 68 | 0.009620 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 0 | 99 | 0.281090 |
| 10 | 99 | 0.275454 |
| 1 | 99 | 0.264209 |
| 0 | 89 | 0.258958 |
| 10 | 89 | 0.258605 |

Inter-cluster distance: mean=0.079498, min=0.008017, max=0.281090, 4950 pairs

### Stratum 3 ([0.694, 0.814)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 79.3 | 2.19 |
| Amide (-CONH-) | 68.4 | 0.90 |
| Heterocycle | 54.2 | 1.27 |
| Ether (C-O-C) | 36.9 | 0.48 |
| Halide (C-X) | 33.0 | 0.52 |
| Secondary Amine (>NH) | 28.2 | 0.30 |
| Tertiary Amine (>N<) | 22.2 | 0.25 |
| Hydroxyl (-OH) | 11.8 | 0.13 |
| Sulfonyl (-SO₂-) | 11.3 | 0.12 |
| Thioether (C-S-C) | 9.1 | 0.09 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 90 | 6185 | 0.747±0.031 | 3.04 | 2.44 | 0.0345 | Ph | HetCyc(1.7×), NH2(1.4×), Ph(1.2×) | `C[C@H](NC(=O)c1ccccc1O)c1cccc(` |
| 9 | 6175 | 0.753±0.033 | 1.16 | 4.24 | 0.0451 | CONH | NH2(2.0×), COOH(1.5×), OH(1.4×) | `C[C@H]1CCCN1C(=O)CC[NH+](C1CC1` |
| 3 | 3030 | 0.767±0.032 | 1.69 | 3.08 | 0.0338 | Ph | COOH(2.4×), OH(1.9×), NH2(1.5×) | `CC(C)[C@H](C)NC(=O)NCCNC(=O)c1` |
| 40 | 2511 | 0.762±0.034 | 2.86 | 2.60 | 0.0308 | Ph | CN(1.7×), C-X(1.5×), HetCyc(1.2×) | `CCn1nc(C)c(CNC(=O)c2cc3sccc3n2` |
| 59 | 2018 | 0.772±0.030 | 1.71 | 3.64 | 0.0329 | Ph | N<(1.8×), C-S-C(1.4×), C-O-C(1.2×) | `O=C(C[NH+]1C[C@H]2CC[C@@H]1CN(` |
| 50 | 2012 | 0.758±0.034 | 2.87 | 2.60 | 0.0315 | Ph | CN(1.6×), HetCyc(1.4×), C=O(1.3×) | `Cc1nn(C)c(Oc2cccnc2)c1NC(=O)c1` |
| 30 | 1965 | 0.762±0.035 | 2.93 | 2.57 | 0.0259 | Ph | CN(1.7×), C-X(1.6×), OH(1.4×) | `CCCc1cc(NC(=O)[C@@H](CC)Sc2ccc` |
| 49 | 1916 | 0.766±0.033 | 1.81 | 3.59 | 0.0270 | CONH | N<(1.9×), C-O-C(1.4×), CONH(1.2×) | `O=C(NCc1cccc(F)c1)[C@@H]1CCC(=` |
| 99 | 1817 | 0.766±0.032 | 2.21 | 3.58 | 0.0327 | Ph | NH2(1.8×), N<(1.8×), C-O-C(1.2×) | `Cc1nc(-c2ccco2)cc([C@H]2C[NH+]` |
| 54 | 1744 | 0.763±0.032 | 2.68 | 2.90 | 0.0245 | Ph | SO2(1.7×), C=O(1.5×), COO(1.4×) | `CCCc1cc(NC(=O)c2ccc(S[C@H]3CCO` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 79 | 89 | 0.008812 |
| 87 | 88 | 0.008996 |
| 56 | 57 | 0.009352 |
| 76 | 86 | 0.009391 |
| 96 | 97 | 0.009566 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 8 | 90 | 0.264135 |
| 7 | 90 | 0.257459 |
| 19 | 90 | 0.251650 |
| 9 | 90 | 0.250558 |
| 8 | 91 | 0.249996 |

Inter-cluster distance: mean=0.077973, min=0.008812, max=0.264135, 4950 pairs

### Stratum 4 ([0.814, 1.0]) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 83.0 | 2.05 |
| Amide (-CONH-) | 68.7 | 0.88 |
| Heterocycle | 56.1 | 1.12 |
| Ether (C-O-C) | 39.9 | 0.51 |
| Halide (C-X) | 37.1 | 0.57 |
| Secondary Amine (>NH) | 30.1 | 0.31 |
| Tertiary Amine (>N<) | 24.2 | 0.27 |
| Hydroxyl (-OH) | 11.6 | 0.12 |
| Sulfonyl (-SO₂-) | 11.0 | 0.11 |
| Thioether (C-S-C) | 7.4 | 0.08 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 9 | 5095 | 0.867±0.032 | 2.86 | 2.63 | 0.0321 | Ph | NH2(2.4×), CN(1.7×), C-X(1.7×) | `CCc1ccc(C[NH2+][C@@H](CC)c2ncc` |
| 99 | 3245 | 0.879±0.035 | 2.55 | 3.25 | 0.0307 | Ph | NH2(1.6×), CN(1.3×), HetCyc(1.3×) | `OC[C@@H]1CCCCN1c1nccc(OCc2cccc` |
| 0 | 2940 | 0.851±0.026 | 2.12 | 3.03 | 0.0344 | Ph | COOH(2.5×), COO(2.2×), SO(1.8×) | `Cc1cnc(C(C)(C)NC(=O)NC(C)C)s1` |
| 5 | 2680 | 0.861±0.033 | 2.63 | 2.69 | 0.0269 | Ph | COO(2.0×), SO(1.7×), HetCyc(1.3×) | `CCc1cnc(CNC(=O)Nc2cccc([C@@H](` |
| 40 | 2626 | 0.861±0.031 | 2.09 | 3.29 | 0.0297 | CONH | SO2(1.3×), COO(1.3×), N<(1.3×) | `CC(C)N(C(=O)[C@@H](C)[NH+]1CCC` |
| 90 | 2147 | 0.851±0.028 | 1.86 | 3.84 | 0.0452 | CONH | N<(1.7×), C-S-C(1.4×), C-O-C(1.3×) | `COC(=O)c1cc(CN2CCC([NH+]3CCC[C` |
| 4 | 2045 | 0.857±0.030 | 2.56 | 2.83 | 0.0287 | Ph | COO(1.5×), C=O(1.5×), HetCyc(1.5×) | `CCCCn1ncc(C(=O)N[C@@H](CO)c2cc` |
| 6 | 2009 | 0.864±0.033 | 2.72 | 2.61 | 0.0253 | Ph | SO(2.3×), COO(1.6×), CN(1.5×) | `Cc1nc(C)c(C(=O)NC[C@H](C)Cc2cc` |
| 30 | 1907 | 0.860±0.031 | 2.07 | 3.13 | 0.0262 | CONH | COO(1.9×), SO2(1.5×), SO(1.5×) | `CC[C@@H](C)C(=O)Nc1cccc(CNC(=O` |
| 93 | 1780 | 0.874±0.032 | 1.99 | 3.85 | 0.0289 | Ph | C-S-C(1.8×), NH2(1.5×), N<(1.5×) | `C[C@H]1CCC[C@@H](NC(=O)Nc2cccc` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 51 | 52 | 0.007153 |
| 37 | 38 | 0.007248 |
| 77 | 87 | 0.007423 |
| 67 | 77 | 0.007474 |
| 68 | 78 | 0.007482 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.173664 |
| 19 | 90 | 0.172423 |
| 29 | 90 | 0.164257 |
| 0 | 99 | 0.158284 |
| 18 | 90 | 0.157668 |

Inter-cluster distance: mean=0.063477, min=0.007153, max=0.173664, 4950 pairs

## 8. Cluster Functional Group Characterization

Summary of functional group signatures across the largest clusters in each stratum.
Enrichment ratio shows over-representation relative to the stratum population.

### Stratum 0 ([0, 0.399)) — Cluster FG Signatures

**Cluster 99 (557 molecules)** — representative: `O=C(Cc1n[nH]c(=O)c2ccccc12)NCc1cn(-c2ccc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.8 | 92.6 | 1.08× |
| Heterocycle | 97.1 | 69.1 | 1.41× |
| Amide (-CONH-) | 51.2 | 61.6 | 0.83× |
| Thioether (C-S-C) | 37.9 | 31.4 | 1.20× |
| Halide (C-X) | 33.6 | 36.2 | 0.93× |
| Ketone (>C=O) | 32.0 | 31.6 | 1.01× |
| Ether (C-O-C) | 22.6 | 34.1 | 0.66× |
| Secondary Amine (>NH) | 13.5 | 19.5 | 0.69× |

**Cluster 0 (337 molecules)** — representative: `C[C@H]1OCC[C@H]1C(=O)NC(C)(C)/C(N)=N/O`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Primary Amine (-NH₂) | 44.5 | 13.0 | 3.42× ⬆ |
| Amide (-CONH-) | 43.0 | 61.6 | 0.70× |
| Hydroxyl (-OH) | 37.4 | 11.6 | 3.23× ⬆ |
| Imine (C=N) | 36.5 | 19.2 | 1.90× ⬆ |
| Secondary Amine (>NH) | 34.1 | 19.5 | 1.75× ⬆ |
| Ether (C-O-C) | 27.6 | 34.1 | 0.81× |
| Tertiary Amine (>N<) | 21.4 | 11.6 | 1.84× ⬆ |
| Ester (-COO-) | 19.9 | 18.5 | 1.07× |

**Cluster 94 (184 molecules)** — representative: `CCCCCOc1ccc(/C=c2\sc3nc([C@@H]4COc5ccccc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.5 | 92.6 | 1.07× |
| Heterocycle | 91.8 | 69.1 | 1.33× |
| Amide (-CONH-) | 69.6 | 61.6 | 1.13× |
| Ketone (>C=O) | 54.3 | 31.6 | 1.72× ⬆ |
| Ether (C-O-C) | 48.4 | 34.1 | 1.42× |
| Tertiary Amine (>N<) | 30.4 | 11.6 | 2.61× ⬆ |
| Halide (C-X) | 27.7 | 36.2 | 0.76× |
| Thioether (C-S-C) | 26.6 | 31.4 | 0.85× |

**Cluster 49 (173 molecules)** — representative: `CC(=O)Nc1ccc(NC(=O)c2sc3nc(-c4cccc([N+](`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.4 | 92.6 | 1.07× |
| Heterocycle | 91.9 | 69.1 | 1.33× |
| Amide (-CONH-) | 56.1 | 61.6 | 0.91× |
| Nitro (-NO₂) | 43.9 | 29.6 | 1.48× |
| Ketone (>C=O) | 41.6 | 31.6 | 1.32× |
| Halide (C-X) | 37.0 | 36.2 | 1.02× |
| Thioether (C-S-C) | 24.9 | 31.4 | 0.79× |
| Imine (C=N) | 23.7 | 19.2 | 1.24× |

**Cluster 69 (166 molecules)** — representative: `Cc1ccc2sc(N(Cc3ccccn3)C(=O)c3ccc([N+](=O`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 92.6 | 1.08× |
| Heterocycle | 95.8 | 69.1 | 1.39× |
| Amide (-CONH-) | 62.7 | 61.6 | 1.02× |
| Ketone (>C=O) | 36.1 | 31.6 | 1.15× |
| Nitro (-NO₂) | 36.1 | 29.6 | 1.22× |
| Ether (C-O-C) | 29.5 | 34.1 | 0.86× |
| Thioether (C-S-C) | 29.5 | 31.4 | 0.94× |
| Halide (C-X) | 23.5 | 36.2 | 0.65× |

### Stratum 1 ([0.399, 0.520)) — Cluster FG Signatures

**Cluster 99 (1336 molecules)** — representative: `O=C(NCc1ccco1)c1ccc2ncc3c(O)n(-c4ccc(Cl)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.9 | 91.5 | 1.09× |
| Heterocycle | 98.9 | 68.5 | 1.44× |
| Amide (-CONH-) | 45.4 | 65.7 | 0.69× |
| Halide (C-X) | 34.8 | 36.5 | 0.95× |
| Ether (C-O-C) | 23.2 | 36.0 | 0.65× |
| Thioether (C-S-C) | 20.3 | 20.5 | 0.99× |
| Ketone (>C=O) | 16.8 | 24.0 | 0.70× |
| Secondary Amine (>NH) | 15.1 | 21.2 | 0.71× |

**Cluster 0 (856 molecules)** — representative: `COCCCNC(=O)C(=O)NC[C@@H]1CCCO1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 45.2 | 65.7 | 0.69× |
| Secondary Amine (>NH) | 37.6 | 21.2 | 1.77× ⬆ |
| Ether (C-O-C) | 25.8 | 36.0 | 0.72× |
| Hydroxyl (-OH) | 21.6 | 8.3 | 2.60× ⬆ |
| Ester (-COO-) | 20.1 | 13.9 | 1.45× |
| Tertiary Amine (>N<) | 16.6 | 15.8 | 1.05× |
| Carboxyl (-COOH) | 16.0 | 2.8 | 5.80× ⬆ |
| Primary Amine (-NH₂) | 15.3 | 8.0 | 1.92× ⬆ |

**Cluster 94 (482 molecules)** — representative: `Cc1nc(-c2ccccn2)sc1C(=O)NCCc1nc2cc(Cl)cc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.6 | 91.5 | 1.09× |
| Heterocycle | 96.3 | 68.5 | 1.41× |
| Amide (-CONH-) | 67.6 | 65.7 | 1.03× |
| Ether (C-O-C) | 42.9 | 36.0 | 1.19× |
| Halide (C-X) | 40.7 | 36.5 | 1.12× |
| Thioether (C-S-C) | 25.1 | 20.5 | 1.22× |
| Secondary Amine (>NH) | 21.2 | 21.2 | 1.00× |
| Ketone (>C=O) | 17.6 | 24.0 | 0.73× |

**Cluster 50 (443 molecules)** — representative: `CC[C@H](NC(=O)Nc1ccc(C)c(NC(C)=O)c1)c1cc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 95.3 | 91.5 | 1.04× |
| Amide (-CONH-) | 67.7 | 65.7 | 1.03× |
| Heterocycle | 56.4 | 68.5 | 0.82× |
| Nitro (-NO₂) | 47.9 | 17.5 | 2.73× ⬆ |
| Halide (C-X) | 40.9 | 36.5 | 1.12× |
| Ether (C-O-C) | 33.4 | 36.0 | 0.93× |
| Secondary Amine (>NH) | 25.5 | 21.2 | 1.20× |
| Imine (C=N) | 20.5 | 10.3 | 2.00× ⬆ |

**Cluster 95 (388 molecules)** — representative: `Cc1cc(C)cc(-c2nnc(Sc3cc(C)c([N+](=O)[O-]`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.7 | 91.5 | 1.09× |
| Heterocycle | 97.2 | 68.5 | 1.42× |
| Amide (-CONH-) | 58.8 | 65.7 | 0.89× |
| Halide (C-X) | 46.4 | 36.5 | 1.27× |
| Ether (C-O-C) | 34.3 | 36.0 | 0.95× |
| Thioether (C-S-C) | 23.5 | 20.5 | 1.14× |
| Secondary Amine (>NH) | 19.8 | 21.2 | 0.93× |
| Ketone (>C=O) | 19.6 | 24.0 | 0.82× |

### Stratum 2 ([0.520, 0.694)) — Cluster FG Signatures

**Cluster 99 (4340 molecules)** — representative: `Cc1nn(-c2ccccc2)c2sc(C(=O)NN(C)c3ccccc3)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.0 | 84.5 | 1.17× |
| Heterocycle | 95.0 | 61.7 | 1.54× ⬆ |
| Amide (-CONH-) | 43.0 | 68.0 | 0.63× |
| Halide (C-X) | 34.9 | 34.9 | 1.00× |
| Secondary Amine (>NH) | 19.0 | 25.3 | 0.75× |
| Ether (C-O-C) | 18.3 | 35.0 | 0.52× |
| Thioether (C-S-C) | 12.3 | 13.2 | 0.93× |
| Ketone (>C=O) | 11.5 | 14.4 | 0.80× |

**Cluster 0 (4278 molecules)** — representative: `CC(=O)N1CC[C@@H](COCC[NH3+])C1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 52.5 | 68.0 | 0.77× |
| Secondary Amine (>NH) | 41.5 | 25.3 | 1.64× ⬆ |
| Ether (C-O-C) | 28.5 | 35.0 | 0.81× |
| Hydroxyl (-OH) | 22.4 | 10.2 | 2.20× ⬆ |
| Tertiary Amine (>N<) | 21.7 | 20.0 | 1.09× |
| Carboxyl (-COOH) | 17.3 | 4.3 | 4.02× ⬆ |
| Primary Amine (-NH₂) | 17.0 | 6.9 | 2.47× ⬆ |
| Ester (-COO-) | 10.2 | 10.0 | 1.02× |

**Cluster 49 (1686 molecules)** — representative: `Cc1ccc(-c2ccc3c(c2)C[C@H](CNC(=O)CN2C[C@`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.0 | 84.5 | 1.17× |
| Heterocycle | 84.6 | 61.7 | 1.37× |
| Amide (-CONH-) | 70.4 | 68.0 | 1.04× |
| Ether (C-O-C) | 43.7 | 35.0 | 1.25× |
| Tertiary Amine (>N<) | 42.9 | 20.0 | 2.15× ⬆ |
| Halide (C-X) | 37.4 | 34.9 | 1.07× |
| Secondary Amine (>NH) | 18.9 | 25.3 | 0.74× |
| Ketone (>C=O) | 14.5 | 14.4 | 1.00× |

**Cluster 96 (1619 molecules)** — representative: `CCc1ccc(-c2nc(N)ccc2[N+](=O)[O-])cc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.5 | 84.5 | 1.16× |
| Heterocycle | 90.9 | 61.7 | 1.47× |
| Amide (-CONH-) | 71.0 | 68.0 | 1.05× |
| Halide (C-X) | 44.2 | 34.9 | 1.27× |
| Ether (C-O-C) | 33.8 | 35.0 | 0.97× |
| Secondary Amine (>NH) | 23.0 | 25.3 | 0.91× |
| Thioether (C-S-C) | 17.4 | 13.2 | 1.31× |
| Ketone (>C=O) | 10.6 | 14.4 | 0.73× |

**Cluster 95 (1596 molecules)** — representative: `CCCc1cc(NC(=O)Cc2csc(Cc3ccccc3)n2)n(C)n1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.0 | 84.5 | 1.16× |
| Heterocycle | 91.1 | 61.7 | 1.48× |
| Amide (-CONH-) | 72.1 | 68.0 | 1.06× |
| Halide (C-X) | 41.4 | 34.9 | 1.18× |
| Ether (C-O-C) | 35.8 | 35.0 | 1.02× |
| Secondary Amine (>NH) | 22.9 | 25.3 | 0.90× |
| Thioether (C-S-C) | 16.8 | 13.2 | 1.27× |
| Ketone (>C=O) | 10.6 | 14.4 | 0.73× |

### Stratum 3 ([0.694, 0.814)) — Cluster FG Signatures

**Cluster 90 (6185 molecules)** — representative: `C[C@H](NC(=O)c1ccccc1O)c1cccc(-n2ccnc2)c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.2 | 79.3 | 1.24× |
| Heterocycle | 93.4 | 54.2 | 1.72× ⬆ |
| Amide (-CONH-) | 51.9 | 68.4 | 0.76× |
| Halide (C-X) | 38.7 | 33.0 | 1.17× |
| Secondary Amine (>NH) | 24.7 | 28.2 | 0.88× |
| Ether (C-O-C) | 22.6 | 36.9 | 0.61× |
| Primary Amine (-NH₂) | 9.3 | 6.6 | 1.40× |
| Ketone (>C=O) | 7.8 | 9.0 | 0.86× |

**Cluster 9 (6175 molecules)** — representative: `C[C@H]1CCCN1C(=O)CC[NH+](C1CC1)[C@@H](C)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 52.9 | 68.4 | 0.77× |
| Ether (C-O-C) | 41.2 | 36.9 | 1.11× |
| Secondary Amine (>NH) | 37.3 | 28.2 | 1.32× |
| Tertiary Amine (>N<) | 30.2 | 22.2 | 1.36× |
| Hydroxyl (-OH) | 16.6 | 11.8 | 1.40× |
| Primary Amine (-NH₂) | 13.0 | 6.6 | 1.95× ⬆ |
| Thioether (C-S-C) | 11.2 | 9.1 | 1.23× |
| Sulfonyl (-SO₂-) | 9.4 | 11.3 | 0.83× |

**Cluster 3 (3030 molecules)** — representative: `CC(C)[C@H](C)NC(=O)NCCNC(=O)c1ccc(F)cc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 72.8 | 79.3 | 0.92× |
| Amide (-CONH-) | 65.7 | 68.4 | 0.96× |
| Halide (C-X) | 38.8 | 33.0 | 1.18× |
| Secondary Amine (>NH) | 38.4 | 28.2 | 1.36× |
| Heterocycle | 36.1 | 54.2 | 0.67× |
| Ether (C-O-C) | 31.8 | 36.9 | 0.86× |
| Hydroxyl (-OH) | 22.5 | 11.8 | 1.90× ⬆ |
| Sulfonyl (-SO₂-) | 14.7 | 11.3 | 1.30× |

**Cluster 40 (2511 molecules)** — representative: `CCn1nc(C)c(CNC(=O)c2cc3sccc3n2CC)c1C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.2 | 79.3 | 1.21× |
| Amide (-CONH-) | 72.2 | 68.4 | 1.05× |
| Heterocycle | 66.7 | 54.2 | 1.23× |
| Halide (C-X) | 49.0 | 33.0 | 1.48× |
| Ether (C-O-C) | 37.2 | 36.9 | 1.01× |
| Secondary Amine (>NH) | 30.1 | 28.2 | 1.07× |
| Hydroxyl (-OH) | 14.0 | 11.8 | 1.18× |
| Thioether (C-S-C) | 10.2 | 9.1 | 1.13× |

**Cluster 59 (2018 molecules)** — representative: `O=C(C[NH+]1C[C@H]2CC[C@@H]1CN(C(=O)C1CCC`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 77.6 | 79.3 | 0.98× |
| Amide (-CONH-) | 66.9 | 68.4 | 0.98× |
| Ether (C-O-C) | 44.8 | 36.9 | 1.21× |
| Tertiary Amine (>N<) | 39.0 | 22.2 | 1.76× ⬆ |
| Heterocycle | 31.6 | 54.2 | 0.58× |
| Secondary Amine (>NH) | 29.1 | 28.2 | 1.03× |
| Halide (C-X) | 23.3 | 33.0 | 0.71× |
| Hydroxyl (-OH) | 13.1 | 11.8 | 1.11× |

### Stratum 4 ([0.814, 1.0]) — Cluster FG Signatures

**Cluster 9 (5095 molecules)** — representative: `CCc1ccc(C[NH2+][C@@H](CC)c2nccs2)cc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 95.9 | 83.0 | 1.16× |
| Heterocycle | 68.6 | 56.1 | 1.22× |
| Halide (C-X) | 61.9 | 37.1 | 1.67× ⬆ |
| Amide (-CONH-) | 41.9 | 68.7 | 0.61× |
| Secondary Amine (>NH) | 36.6 | 30.1 | 1.22× |
| Ether (C-O-C) | 30.5 | 39.9 | 0.76× |
| Hydroxyl (-OH) | 17.4 | 11.6 | 1.50× |
| Primary Amine (-NH₂) | 17.3 | 7.2 | 2.41× ⬆ |

**Cluster 99 (3245 molecules)** — representative: `OC[C@@H]1CCCCN1c1nccc(OCc2ccccc2)n1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 94.3 | 83.0 | 1.14× |
| Heterocycle | 73.4 | 56.1 | 1.31× |
| Secondary Amine (>NH) | 39.0 | 30.1 | 1.30× |
| Ether (C-O-C) | 36.8 | 39.9 | 0.92× |
| Halide (C-X) | 31.6 | 37.1 | 0.85× |
| Amide (-CONH-) | 30.6 | 68.7 | 0.45× ⬇ |
| Tertiary Amine (>N<) | 30.1 | 24.2 | 1.24× |
| Hydroxyl (-OH) | 12.6 | 11.6 | 1.09× |

**Cluster 0 (2940 molecules)** — representative: `Cc1cnc(C(C)(C)NC(=O)NC(C)C)s1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 74.3 | 83.0 | 0.89× |
| Amide (-CONH-) | 72.6 | 68.7 | 1.06× |
| Halide (C-X) | 50.2 | 37.1 | 1.35× |
| Ether (C-O-C) | 33.2 | 39.9 | 0.83× |
| Heterocycle | 32.8 | 56.1 | 0.58× |
| Secondary Amine (>NH) | 30.2 | 30.1 | 1.00× |
| Sulfonyl (-SO₂-) | 18.8 | 11.0 | 1.71× ⬆ |
| Tertiary Amine (>N<) | 16.6 | 24.2 | 0.69× |

**Cluster 5 (2680 molecules)** — representative: `CCc1cnc(CNC(=O)Nc2cccc([C@@H](C)OC)c2)s1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 91.3 | 83.0 | 1.10× |
| Amide (-CONH-) | 79.6 | 68.7 | 1.16× |
| Heterocycle | 75.4 | 56.1 | 1.34× |
| Halide (C-X) | 45.5 | 37.1 | 1.23× |
| Ether (C-O-C) | 37.5 | 39.9 | 0.94× |
| Secondary Amine (>NH) | 27.5 | 30.1 | 0.92× |
| Tertiary Amine (>N<) | 11.4 | 24.2 | 0.47× ⬇ |
| Sulfonyl (-SO₂-) | 10.5 | 11.0 | 0.96× |

**Cluster 40 (2626 molecules)** — representative: `CC(C)N(C(=O)[C@@H](C)[NH+]1CCCN(C(=O)c2c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 82.2 | 68.7 | 1.20× |
| Phenyl (aromatic ring) | 57.0 | 83.0 | 0.69× |
| Ether (C-O-C) | 47.6 | 39.9 | 1.19× |
| Heterocycle | 42.6 | 56.1 | 0.76× |
| Halide (C-X) | 34.8 | 37.1 | 0.94× |
| Secondary Amine (>NH) | 31.5 | 30.1 | 1.05× |
| Tertiary Amine (>N<) | 30.5 | 24.2 | 1.26× |
| Sulfonyl (-SO₂-) | 14.7 | 11.0 | 1.34× |

## 9. Cluster Quality Analysis

### 9.1 Per-Stratum Quality Metrics

| Stratum | Silhouette | Davies-Bouldin | QE | Clusters | Gini | Singletons |
|---|---|---|---|---|---|---|
| 0 [0, 0.399) | -0.0287 | 3.8077 | 0.032171 | 100 | 0.436 | 0 |
| 1 [0.399, 0.520) | -0.0203 | 4.1117 | 0.032613 | 100 | 0.402 | 0 |
| 2 [0.520, 0.694) | -0.0124 | 4.3035 | 0.035004 | 100 | 0.448 | 0 |
| 3 [0.694, 0.814) | -0.0111 | 4.4541 | 0.035135 | 100 | 0.500 | 0 |
| 4 [0.814, 1.0] | -0.0355 | 5.1161 | 0.031593 | 100 | 0.453 | 0 |

**Interpretation guide:**
- **Silhouette** ∈ [-1, 1]: higher = better separation (>0.5 strong, >0.25 reasonable)
- **Davies-Bouldin**: lower = better separation (0 is optimal)
- **Gini coefficient**: 0 = equal sizes, 1 = maximally unequal

### 9.2 Cluster Size Distribution

| Stratum | Mean | Median | Std | Min | P25 | P75 | Max | Large |
|---|---|---|---|---|---|---|---|---|
| 0 | 68.3 | 46 | 70.6 | 5 | 32 | 90 | 557 | 2 |
| 1 | 176.2 | 131 | 167.9 | 26 | 81 | 227 | 1336 | 2 |
| 2 | 604.3 | 434 | 652.1 | 29 | 251 | 743 | 4340 | 2 |
| 3 | 836.7 | 572 | 973.4 | 11 | 249 | 1130 | 6185 | 4 |
| 4 | 809.0 | 566 | 777.9 | 80 | 323 | 995 | 5095 | 5 |

![Cluster Quality](figures/cluster_quality_comparison.svg)

*Figure 12: Comparison of cluster quality metrics across QED strata — silhouette score, Davies-Bouldin index, quantization error, and Gini coefficient.*

![Cluster Sizes](figures/cluster_size_distribution.svg)

*Figure 13: Distribution of cluster sizes within each QED stratum.*

## 10. Evaluation Summary

| Metric | Value |
|---|---|
| Total active clusters | 500 / 500 neurons |
| Cluster size (mean) | 498.9 |
| Cluster size (range) | 5 – 6185 |
| Average quantization error | 0.033303 |
| Average silhouette score | -0.0216 |
| Average Davies-Bouldin index | 4.3586 |
| Mean intra-cluster distance | 0.027009 |
| Functional group types detected | 22 / 22 |
| Strongest FG-property |r| | 0.1057 (Hydroxyl (-OH)) |

## 11. Performance

| Phase | Time |
|---|---|
| Data loading | 0.11s |
| Graph parsing + FG detection | 2.08s |
| VGAE encoding | 1037.67s |
| Importance analysis | 0.68s |
| SOM clustering + FG analysis | 93.20s |
| **Total** | **1153.36s** |

**Throughput**: 216 molecules/second

## 12. Methodology Comparison

| Aspect | Previous (Python) | Current (Rust + GNN) |
|---|---|---|
| Molecular representation | Flat 28-dim feature vector | Full molecular graph |
| Feature learning | Dense autoencoder (28→16→28) | Graph Attention Network (3 layers) |
| Latent model | Deterministic AE | Variational (VGAE with KL regularization) |
| Structure awareness | None (bag of atoms) | Message passing preserves bond topology |
| Pooling | N/A (fixed features) | Global attention pooling (learned) |
| Edge features | Not used | 9-dim bond features in attention |
| Functional group analysis | None | 22-type substructure detection + enrichment |
| Importance analysis | None | Dim-property correlation + FG-property correlation |
| Cluster characterization | Size + basic stats | FG signatures, enrichment, representatives |
| Implementation | Python/PyTorch | Rust/Burn (memory-safe, zero-cost abstractions) |

## 13. Output Files

```
results/
├── RESULTS.md              # This report
├── training_losses.csv     # Per-molecule reconstruction losses
├── figures/                # SVG visualizations
│   ├── qed_distribution.svg  # Property distribution
│   ├── logp_distribution.svg  # Property distribution
│   ├── sas_distribution.svg  # Property distribution
│   ├── property_distributions_combined.svg  # Property distribution
│   ├── fg_prevalence.svg  # Functional group prevalence
│   ├── latent_space_pca.svg  # Latent space PCA projection
│   ├── cluster_size_distribution.svg  # Cluster size distributions
│   ├── dim_property_heatmap.svg  # Dimension-property heatmap
│   ├── fg_property_correlations.svg  # FG-property correlations
│   ├── cluster_quality_comparison.svg  # Cluster quality comparison
│   ├── reconstruction_loss_dist.svg  # Reconstruction loss
│   ├── embedding_dim_variance.svg  # Embedding variance
│   ├── umatrix_heatmaps.svg  # SOM U-matrix
│   ├── stratum_property_comparison.svg  # Stratum properties
│   ├── molecule_complexity.svg  # Molecule complexity
│   ├── fg_enrichment_stratum_0.svg  # FG enrichment S0
│   ├── fg_enrichment_stratum_1.svg  # FG enrichment S1
│   ├── fg_enrichment_stratum_2.svg  # FG enrichment S2
│   ├── fg_enrichment_stratum_3.svg  # FG enrichment S3
│   ├── fg_enrichment_stratum_4.svg  # FG enrichment S4
│   ├── cluster_distance_matrix_stratum_0.svg  # Distances S0
│   ├── cluster_distance_matrix_stratum_1.svg  # Distances S1
│   ├── cluster_distance_matrix_stratum_2.svg  # Distances S2
│   ├── cluster_distance_matrix_stratum_3.svg  # Distances S3
│   ├── cluster_distance_matrix_stratum_4.svg  # Distances S4
└── group_0/
    ├── labeled_data.csv    # SMILES + properties + cluster label
    └── embeddings.csv      # 16-dim latent embeddings
└── group_1/
    ├── labeled_data.csv    # SMILES + properties + cluster label
    └── embeddings.csv      # 16-dim latent embeddings
└── group_2/
    ├── labeled_data.csv    # SMILES + properties + cluster label
    └── embeddings.csv      # 16-dim latent embeddings
└── group_3/
    ├── labeled_data.csv    # SMILES + properties + cluster label
    └── embeddings.csv      # 16-dim latent embeddings
└── group_4/
    ├── labeled_data.csv    # SMILES + properties + cluster label
    └── embeddings.csv      # 16-dim latent embeddings
```

## 14. Figure Index

| # | Figure | Description |
|---|---|---|
| Figure 1 | [figures/property_distributions_combined.svg](figures/property_distributions_combined.svg) | Molecular property distributions (QED, logP, SAS) |
| Figure 2 | [figures/qed_distribution.svg](figures/qed_distribution.svg) | Individual property histograms with mean indicators |
| Figure 3 | [figures/molecule_complexity.svg](figures/molecule_complexity.svg) | Molecular graph complexity scatter (atoms vs bonds) |
| Figure 4 | [figures/fg_prevalence.svg](figures/fg_prevalence.svg) | Functional group prevalence bar chart |
| Figure 5 | [figures/reconstruction_loss_dist.svg](figures/reconstruction_loss_dist.svg) | VGAE reconstruction loss distribution |
| Figure 6 | [figures/embedding_dim_variance.svg](figures/embedding_dim_variance.svg) | Latent dimension variance analysis |
| Figure 7 | [figures/dim_property_heatmap.svg](figures/dim_property_heatmap.svg) | Dimension–property correlation heatmap |
| Figure 8 | [figures/fg_property_correlations.svg](figures/fg_property_correlations.svg) | FG–property correlation heatmap |
| Figure 9 | [figures/latent_space_pca.svg](figures/latent_space_pca.svg) | PCA projection of latent space by stratum |
| Figure 10 | [figures/stratum_property_comparison.svg](figures/stratum_property_comparison.svg) | Stratum property comparison (mean ± std) |
| Figure 11 | [figures/umatrix_heatmaps.svg](figures/umatrix_heatmaps.svg) | SOM U-matrix heatmaps per stratum |
| Figure 12 | [figures/cluster_quality_comparison.svg](figures/cluster_quality_comparison.svg) | Cluster quality metrics comparison |
| Figure 13 | [figures/cluster_size_distribution.svg](figures/cluster_size_distribution.svg) | Cluster size distributions per stratum |
| Figure 14 | [figures/fg_enrichment_stratum_0.svg](figures/fg_enrichment_stratum_0.svg) | FG enrichment heatmap — Stratum 0 |
| Figure 15 | [figures/fg_enrichment_stratum_1.svg](figures/fg_enrichment_stratum_1.svg) | FG enrichment heatmap — Stratum 1 |
| Figure 16 | [figures/fg_enrichment_stratum_2.svg](figures/fg_enrichment_stratum_2.svg) | FG enrichment heatmap — Stratum 2 |
| Figure 17 | [figures/fg_enrichment_stratum_3.svg](figures/fg_enrichment_stratum_3.svg) | FG enrichment heatmap — Stratum 3 |
| Figure 18 | [figures/fg_enrichment_stratum_4.svg](figures/fg_enrichment_stratum_4.svg) | FG enrichment heatmap — Stratum 4 |

