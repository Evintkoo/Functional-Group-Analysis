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

## 2. Molecular Graph Statistics

| Property | Value |
|---|---|
| Atoms per molecule | 23.2 (range: 6–38) |
| Bonds per molecule | 24.9 (range: 5–45) |
| Total atoms processed | 5775223 |
| Total bonds processed | 6211984 |
| Node feature dimension | 29 |
| Edge feature dimension | 9 |

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
| Mean reconstruction loss | 0.167784 |
| Mean pairwise embedding distance | 0.118844 |
| Embedding std (mean across dims) | 0.022435 |
| Embedding std range | [0.0102, 0.0513] |

### Latent Dimension Statistics

| Dim | Mean | Std | Min | Max |
|---|---|---|---|---|
| 0 | -0.0134 | 0.0464 | -0.1480 | 0.1642 |
| 1 | -0.1702 | 0.0513 | -0.3193 | 0.0096 |
| 2 | 0.1986 | 0.0137 | 0.1174 | 0.2631 |
| 3 | -0.1327 | 0.0102 | -0.1728 | -0.0575 |
| 4 | 0.1262 | 0.0139 | 0.0538 | 0.1582 |
| 5 | -0.0668 | 0.0112 | -0.1181 | -0.0101 |
| 6 | -0.0077 | 0.0278 | -0.1255 | 0.0908 |
| 7 | -0.1310 | 0.0182 | -0.2211 | 0.0103 |
| 8 | 0.0305 | 0.0247 | -0.0424 | 0.1450 |
| 9 | -0.1104 | 0.0133 | -0.1604 | 0.0089 |
| 10 | -0.5381 | 0.0279 | -0.7095 | -0.3996 |
| 11 | 0.2774 | 0.0137 | 0.1960 | 0.3453 |
| 12 | -0.0277 | 0.0230 | -0.1185 | 0.0913 |
| 13 | -0.0622 | 0.0285 | -0.1627 | 0.0609 |
| 14 | -0.0573 | 0.0111 | -0.1480 | -0.0092 |
| 15 | -0.1223 | 0.0240 | -0.2296 | -0.0334 |

## 6. Feature Importance Analysis

### 6.1 Latent Dimension ↔ Property Correlations

Pearson correlation (r) between each latent dimension and molecular properties.
Dimensions sorted by |r(QED)|.

| Dim | Variance | r(QED) | r(logP) | r(SAS) |
|---|---|---|---|---|
| 12 | 0.000530 | +0.3081 | -0.3873 | +0.6373 |
| 13 | 0.000814 | +0.2994 | -0.3806 | +0.6561 |
| 5 | 0.000126 | -0.2911 | +0.2468 | -0.5654 |
| 1 | 0.002637 | +0.2871 | -0.4298 | +0.6225 |
| 0 | 0.002150 | +0.2864 | -0.4119 | +0.6297 |
| 6 | 0.000773 | -0.2779 | +0.3993 | -0.6120 |
| 10 | 0.000780 | -0.2778 | +0.1990 | -0.5514 |
| 2 | 0.000187 | +0.2777 | -0.3121 | +0.5765 |
| 8 | 0.000610 | +0.2381 | -0.4894 | +0.5535 |
| 3 | 0.000103 | +0.2180 | -0.3257 | +0.4989 |
| 4 | 0.000194 | -0.2027 | +0.4002 | -0.5566 |
| 11 | 0.000187 | -0.1893 | +0.3190 | -0.4336 |
| 14 | 0.000122 | -0.1869 | +0.3251 | -0.4801 |
| 15 | 0.000577 | -0.1806 | +0.4220 | -0.4225 |
| 9 | 0.000178 | -0.0436 | -0.2831 | +0.0042 |
| 7 | 0.000332 | -0.0103 | -0.2150 | +0.1058 |

### 6.2 Functional Group ↔ Latent Space Encoding

Which latent dimensions best encode each functional group's presence.

| Functional Group | Prevalence (%) | Best Dim | |r| |
|---|---|---|---|
| Phenyl (aromatic ring) | 83.0 | 8 | 0.5686 |
| Heterocycle | 58.0 | 14 | 0.4883 |
| Sulfonyl (-SO₂-) | 10.9 | 2 | 0.3757 |
| Tertiary Amine (>N<) | 21.6 | 11 | 0.2761 |
| Nitrile (-C≡N) | 5.2 | 14 | 0.2721 |
| Amide (-CONH-) | 68.0 | 9 | 0.2584 |
| Hydroxyl (-OH) | 11.1 | 3 | 0.2436 |
| Ether (C-O-C) | 37.3 | 3 | 0.2103 |
| Ketone (>C=O) | 10.5 | 5 | 0.2017 |
| Halide (C-X) | 35.1 | 2 | 0.1966 |
| Secondary Amine (>NH) | 27.4 | 12 | 0.1620 |
| Carboxyl (-COOH) | 3.8 | 9 | 0.1598 |
| Nitro (-NO₂) | 4.3 | 5 | 0.1586 |
| Ester (-COO-) | 7.3 | 9 | 0.1560 |
| Imine (C=N) | 2.7 | 10 | 0.1306 |
| Primary Amine (-NH₂) | 7.1 | 12 | 0.1024 |
| Thioether (C-S-C) | 11.0 | 2 | 0.0873 |

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

## 7. Stratified Clustering Results

### Per-Stratum Overview

| Stratum | QED Range | Molecules | Active Clusters | QE | U-Matrix Mean | U-Matrix Max |
|---|---|---|---|---|---|---|
| 0 | [0, 0.399) | 6830 | 100 | 0.028363 | 0.0131 | 0.0221 |
| 1 | [0.399, 0.520) | 17622 | 100 | 0.029087 | 0.0137 | 0.0224 |
| 2 | [0.520, 0.694) | 60427 | 100 | 0.031433 | 0.0158 | 0.0273 |
| 3 | [0.694, 0.814) | 83673 | 100 | 0.031549 | 0.0158 | 0.0226 |
| 4 | [0.814, 1.0] | 80903 | 100 | 0.027250 | 0.0120 | 0.0179 |

**Total clustered**: 249455 molecules | **Avg QE**: 0.029536

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
| 90 | 493 | 0.340±0.052 | 4.28 | 2.44 | 0.0280 | Ph | HetCyc(1.4×), C=O(1.3×) | `O=C(NCc1ccccc1-c1ccc(Cn2cccn2)` |
| 9 | 344 | 0.330±0.052 | 0.00 | 4.27 | 0.0723 | NH2 | OH(3.5×), NH2(3.5×), COOH(2.2×) | `C[C@@H]1CCC[C@H](NC(=O)C/C(N)=` |
| 50 | 173 | 0.335±0.051 | 3.90 | 2.43 | 0.0209 | Ph | SO2(2.7×), CN(1.8×), C=N(1.3×) | `O=C(/N=C1/c2ccccc2CN1c1ccc(I)c` |
| 40 | 148 | 0.329±0.060 | 3.55 | 2.52 | 0.0213 | Ph | SO2(3.1×), CN(2.2×), COOH(1.4×) | `Cc1c(NC(=O)/C=C/c2ccc(-c3ccccc` |
| 60 | 145 | 0.333±0.057 | 3.97 | 2.44 | 0.0220 | Ph | SO2(2.4×), COOH(1.9×), CN(1.8×) | `Cc1nccn1-c1ccc(C(=O)/C=C/c2ccc` |
| 94 | 141 | 0.350±0.042 | 4.44 | 2.62 | 0.0191 | Ph | C-S-C(1.6×), HetCyc(1.4×), C-O-C(1.3×) | `Cc1cccc(NC(=O)CSc2ncnc3c(-c4cc` |
| 93 | 128 | 0.349±0.038 | 4.47 | 2.52 | 0.0178 | Ph | C-S-C(1.6×), HetCyc(1.4×) | `CCn1nc(C)cc1-c1nnc(SCC(=O)c2cc` |
| 5 | 123 | 0.330±0.045 | 1.64 | 2.93 | 0.0286 | Ph | COOH(4.4×), CN(2.5×), NH2(2.1×) | `CCOC(=O)/C(CSc1ccc(Cl)cc1)=N\N` |
| 80 | 122 | 0.335±0.053 | 3.99 | 2.46 | 0.0210 | Ph | SO2(2.3×), C=O(1.4×), HetCyc(1.3×) | `Cc1nc2ccc(NC(=O)[C@@H](Cc3cccc` |
| 95 | 122 | 0.346±0.048 | 4.53 | 2.61 | 0.0190 | Ph | C-S-C(1.7×), C-O-C(1.4×), HetCyc(1.4×) | `COc1ccccc1/C=C/C(=O)Nc1nc2n(n1` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 91 | 92 | 0.007605 |
| 96 | 97 | 0.007656 |
| 76 | 87 | 0.007926 |
| 71 | 82 | 0.007983 |
| 81 | 82 | 0.008129 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.345952 |
| 9 | 80 | 0.329039 |
| 9 | 91 | 0.322025 |
| 9 | 70 | 0.321167 |
| 9 | 92 | 0.314723 |

Inter-cluster distance: mean=0.077107, min=0.007605, max=0.345952, 4950 pairs

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
| 90 | 1271 | 0.469±0.036 | 3.87 | 2.37 | 0.0293 | Ph | HetCyc(1.4×) | `Cn1ccnc1[C@H](NC(=O)c1ccc(-n2c` |
| 9 | 1009 | 0.473±0.034 | -0.02 | 4.45 | 0.0735 | CONH | COOH(3.9×), OH(2.8×), NH2(1.9×) | `[NH3+][C@H]1CCCCC[C@@H]1C(=O)N` |
| 93 | 436 | 0.466±0.034 | 3.56 | 2.41 | 0.0218 | Ph | SO2(2.0×), C=O(1.4×), C=N(1.3×) | `Cc1cc(C(=O)N/N=C2\C(=O)N(Cc3cc` |
| 94 | 430 | 0.465±0.036 | 3.45 | 2.46 | 0.0218 | Ph | SO2(2.5×), C=N(1.7×), CN(1.5×) | `Cc1ccc(C(=O)N/N=C\c2ccc(-c3ccc` |
| 40 | 392 | 0.464±0.034 | 4.02 | 2.71 | 0.0203 | Ph | N<(1.8×), HetCyc(1.4×), C-O-C(1.3×) | `Cc1cccc(-n2nc3c(c2-n2cccc2)CN(` |
| 50 | 388 | 0.467±0.033 | 4.13 | 2.62 | 0.0204 | Ph | HetCyc(1.4×) | `COc1ccc(-c2nc3n(n2)[C@H](c2ccn` |
| 95 | 374 | 0.469±0.035 | 3.21 | 2.44 | 0.0261 | Ph | SO2(2.2×), CN(1.9×), NO2(1.8×) | `Cc1ccc(CNC(=O)c2cc3cc(S(=O)(=O` |
| 4 | 317 | 0.474±0.033 | 3.32 | 3.17 | 0.0236 | Ph | N<(2.6×), C-O-C(1.7×), C-S-C(1.5×) | `CCOc1cccc2c1N[C@H](c1cc([N+](=` |
| 92 | 315 | 0.466±0.036 | 3.68 | 2.38 | 0.0221 | Ph | SO2(2.4×), C=O(1.3×), HetCyc(1.3×) | `O=C(/C=C/c1cccc([N+](=O)[O-])c` |
| 59 | 314 | 0.470±0.030 | 1.80 | 2.83 | 0.0313 | Ph | COOH(3.7×), CN(2.4×), C=N(2.0×) | `COc1c(Cl)cc(Cl)cc1/C=N/NC(N)=S` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 71 | 82 | 0.008060 |
| 12 | 23 | 0.008286 |
| 11 | 22 | 0.008349 |
| 70 | 80 | 0.008407 |
| 71 | 81 | 0.008420 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.350162 |
| 9 | 91 | 0.329981 |
| 9 | 80 | 0.322917 |
| 9 | 92 | 0.322138 |
| 9 | 70 | 0.315406 |

Inter-cluster distance: mean=0.080594, min=0.008060, max=0.350162, 4950 pairs

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
| 9 | 4849 | 0.595±0.053 | 3.20 | 2.37 | 0.0353 | Ph | HetCyc(1.5×), C=O(1.2×) | `Cc1cc(C)n(-c2nc(-c3ccccc3)cc(=` |
| 90 | 4772 | 0.631±0.046 | 0.27 | 4.55 | 0.0614 | CONH | COOH(2.8×), NH2(2.6×), OH(2.3×) | `CCCO[C@@H]1CC[C@]([NH2+]C2CC2)` |
| 4 | 1562 | 0.624±0.044 | 2.80 | 2.49 | 0.0252 | Ph | C=N(2.8×), CN(2.1×), SO2(2.0×) | `C[NH+](C)[C@H](CNC(=O)c1cccc(C` |
| 5 | 1382 | 0.624±0.047 | 2.86 | 2.45 | 0.0248 | Ph | C=N(2.4×), SO2(2.0×), CN(2.0×) | `CN(C)c1ccc(C(=O)N/N=C2\C(=O)N(` |
| 59 | 1325 | 0.631±0.047 | 3.62 | 2.71 | 0.0218 | Ph | HetCyc(1.4×), N<(1.4×), C-S-C(1.3×) | `COc1ccc(NC(=S)N2CCn3cccc3[C@@H` |
| 40 | 1256 | 0.621±0.043 | 1.57 | 3.15 | 0.0301 | Ph | CN(1.9×), NO2(1.9×), NH2(1.8×) | `CCC(CC)S(=O)(=O)/N=C(\[O-])[C@` |
| 7 | 1208 | 0.617±0.052 | 3.08 | 2.38 | 0.0234 | Ph | SO2(2.0×), CN(1.9×), C=N(1.7×) | `CCOC(=O)c1nn(-c2ccccc2C)c(=O)c` |
| 6 | 1149 | 0.622±0.050 | 2.97 | 2.42 | 0.0249 | Ph | SO2(2.1×), C=N(2.1×), CN(2.0×) | `CC(=O)c1cccc(S(=O)(=O)N(C)Cc2c` |
| 95 | 1081 | 0.633±0.044 | 2.48 | 3.44 | 0.0258 | Ph | N<(2.0×), C-O-C(1.5×), OH(1.3×) | `COc1ccc(N2C[C@@H](c3nnc(NC(=O)` |
| 49 | 1048 | 0.629±0.048 | 3.58 | 2.63 | 0.0226 | Ph | C-S-C(1.5×), HetCyc(1.4×) | `Cn1cccc1-c1csc(NC(=O)C2(c3cccc` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 12 | 23 | 0.008601 |
| 77 | 88 | 0.008752 |
| 11 | 22 | 0.008770 |
| 67 | 78 | 0.008921 |
| 16 | 27 | 0.009250 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.379524 |
| 8 | 90 | 0.350802 |
| 19 | 90 | 0.345682 |
| 7 | 90 | 0.340676 |
| 29 | 90 | 0.332756 |

Inter-cluster distance: mean=0.091386, min=0.008601, max=0.379524, 4950 pairs

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
| 0 | 7471 | 0.747±0.031 | 2.84 | 2.36 | 0.0329 | Ph | HetCyc(1.7×), C=O(1.6×), Ph(1.2×) | `Cc1cccc(Cl)c1NC(=O)c1ccc(-n2cn` |
| 99 | 6339 | 0.753±0.033 | 1.21 | 4.31 | 0.0530 | CONH | NH2(2.0×), OH(1.5×), COOH(1.4×) | `C[C@H]1CCCN1C(=O)CC[NH+](C1CC1` |
| 5 | 2561 | 0.762±0.033 | 2.70 | 2.58 | 0.0239 | Ph | CN(2.2×), SO2(2.2×), C-X(1.4×) | `CCNC(=O)NNC(=O)c1c(C)nn(-c2ccc` |
| 49 | 2168 | 0.765±0.033 | 2.11 | 3.39 | 0.0287 | Ph | OH(1.9×), NH2(1.6×), NH(1.4×) | `CCSCC[C@H](C)N(C)C(=O)c1ccc(Br` |
| 4 | 1962 | 0.762±0.035 | 2.60 | 2.48 | 0.0242 | Ph | SO2(2.8×), CN(1.7×), C-X(1.4×) | `O=C(COC(=O)c1ncc(Cl)c(Cl)c1Cl)` |
| 95 | 1704 | 0.771±0.031 | 1.77 | 3.71 | 0.0261 | Ph | N<(1.6×), NH2(1.5×), C-O-C(1.3×) | `O=C(NC[C@@H]1CCCO1)NC1CC[NH+](` |
| 96 | 1571 | 0.771±0.031 | 1.62 | 4.03 | 0.0255 | Ph | N<(1.8×), OH(1.7×), C-O-C(1.5×) | `O=C(NCCCOC[C@H]1CCCO1)N1CCO[C@` |
| 94 | 1485 | 0.770±0.031 | 1.70 | 3.57 | 0.0277 | Ph | NH2(1.9×), N<(1.6×), C-S-C(1.3×) | `C[C@H]1CCC[C@@H](NC(=O)CN2CC[N` |
| 50 | 1459 | 0.749±0.034 | 2.74 | 3.12 | 0.0235 | Ph | N<(1.7×), HetCyc(1.6×), Ph(1.2×) | `Cc1cnn([C@H]2CCCN(C(=O)c3cc(-c` |
| 30 | 1442 | 0.741±0.030 | 2.93 | 2.82 | 0.0235 | Ph | HetCyc(1.8×), Ph(1.2×) | `COc1ccc(C)c2c1N(C(=O)Cc1ccc(-n` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 62 | 71 | 0.007659 |
| 63 | 72 | 0.008161 |
| 73 | 82 | 0.008899 |
| 12 | 21 | 0.009185 |
| 13 | 22 | 0.009263 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 0 | 99 | 0.365982 |
| 1 | 99 | 0.339880 |
| 10 | 99 | 0.335335 |
| 2 | 99 | 0.328631 |
| 20 | 99 | 0.323455 |

Inter-cluster distance: mean=0.092552, min=0.007659, max=0.365982, 4950 pairs

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
| 90 | 5866 | 0.869±0.034 | 2.43 | 2.29 | 0.0312 | Ph | SO2(2.9×), COOH(2.7×), COO(1.7×) | `O=C(NCc1ccco1)C(=O)Nc1cc(Cl)cc` |
| 9 | 4193 | 0.859±0.031 | 1.89 | 4.22 | 0.0557 | C-O-C | C-S-C(1.8×), N<(1.7×), OH(1.6×) | `CC(C)(C)c1n[nH]cc1CN1CC[C@@H](` |
| 4 | 1815 | 0.857±0.030 | 2.26 | 3.41 | 0.0253 | CONH | SO(2.2×), CN(1.6×), OH(1.4×) | `CCC(=O)Nc1ccc([C@H](C)NC(=O)N2` |
| 5 | 1739 | 0.862±0.029 | 2.22 | 3.58 | 0.0215 | CONH | SO(1.5×), OH(1.4×), NH(1.2×) | `CCC[NH+]1[C@H]2CCC[C@@H]1CC(NC` |
| 94 | 1697 | 0.874±0.035 | 2.60 | 2.67 | 0.0224 | Ph | — | `Cc1ccc(-c2ncco2)cc1NC(=O)CCN1C` |
| 95 | 1682 | 0.876±0.036 | 2.51 | 2.71 | 0.0209 | Ph | C=O(1.3×), HetCyc(1.2×) | `CCn1cc[nH+]c(N2CC[C@H](Cc3cccc` |
| 19 | 1629 | 0.867±0.031 | 1.96 | 3.94 | 0.0250 | Ph | N<(1.7×), C-S-C(1.6×), C-O-C(1.6×) | `O=C(N[C@H]1C[C@H]2CCCc3cccc1c3` |
| 99 | 1439 | 0.879±0.035 | 2.37 | 3.29 | 0.0213 | Ph | N<(1.4×), HetCyc(1.4×), NH2(1.4×) | `O=C(COc1ccccc1)N1CCC[C@@H]1c1n` |
| 93 | 1425 | 0.873±0.034 | 2.65 | 2.67 | 0.0249 | Ph | NH2(1.6×), SO2(1.4×), C=O(1.4×) | `O=C(NC[C@@H](c1ccc(Cl)cc1)n1cc` |
| 29 | 1417 | 0.869±0.030 | 1.95 | 3.68 | 0.0222 | Ph | C-S-C(1.6×), N<(1.6×), C-O-C(1.5×) | `O=C(NC[C@@H](c1cccc(Cl)c1)N1CC` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 86 | 87 | 0.008454 |
| 76 | 87 | 0.008461 |
| 77 | 88 | 0.008534 |
| 96 | 97 | 0.008678 |
| 73 | 83 | 0.008721 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.267379 |
| 9 | 80 | 0.244640 |
| 9 | 91 | 0.242473 |
| 9 | 70 | 0.235841 |
| 9 | 92 | 0.233895 |

Inter-cluster distance: mean=0.074906, min=0.008454, max=0.267379, 4950 pairs

## 8. Cluster Functional Group Characterization

Summary of functional group signatures across the largest clusters in each stratum.
Enrichment ratio shows over-representation relative to the stratum population.

### Stratum 0 ([0, 0.399)) — Cluster FG Signatures

**Cluster 90 (493 molecules)** — representative: `O=C(NCc1ccccc1-c1ccc(Cn2cccn2)cc1)c1ccc(`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.8 | 92.6 | 1.08× |
| Heterocycle | 97.6 | 69.1 | 1.41× |
| Amide (-CONH-) | 56.4 | 61.6 | 0.92× |
| Ketone (>C=O) | 41.8 | 31.6 | 1.32× |
| Thioether (C-S-C) | 28.2 | 31.4 | 0.90× |
| Halide (C-X) | 26.6 | 36.2 | 0.73× |
| Nitro (-NO₂) | 19.3 | 29.6 | 0.65× |
| Ether (C-O-C) | 16.8 | 34.1 | 0.49× ⬇ |

**Cluster 9 (344 molecules)** — representative: `C[C@@H]1CCC[C@H](NC(=O)C/C(N)=N/O)C1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Primary Amine (-NH₂) | 45.3 | 13.0 | 3.48× ⬆ |
| Hydroxyl (-OH) | 41.0 | 11.6 | 3.54× ⬆ |
| Amide (-CONH-) | 41.0 | 61.6 | 0.67× |
| Imine (C=N) | 38.7 | 19.2 | 2.02× ⬆ |
| Secondary Amine (>NH) | 35.8 | 19.5 | 1.83× ⬆ |
| Ether (C-O-C) | 29.9 | 34.1 | 0.88× |
| Tertiary Amine (>N<) | 23.8 | 11.6 | 2.05× ⬆ |
| Ester (-COO-) | 18.6 | 18.5 | 1.00× |

**Cluster 50 (173 molecules)** — representative: `O=C(/N=C1/c2ccccc2CN1c1ccc(I)cc1)c1cccc(`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 92.6 | 1.08× |
| Heterocycle | 82.1 | 69.1 | 1.19× |
| Amide (-CONH-) | 64.7 | 61.6 | 1.05× |
| Ketone (>C=O) | 40.5 | 31.6 | 1.28× |
| Nitro (-NO₂) | 37.6 | 29.6 | 1.27× |
| Halide (C-X) | 36.4 | 36.2 | 1.00× |
| Imine (C=N) | 24.9 | 19.2 | 1.30× |
| Secondary Amine (>NH) | 19.7 | 19.5 | 1.01× |

**Cluster 40 (148 molecules)** — representative: `Cc1c(NC(=O)/C=C/c2ccc(-c3ccccc3[N+](=O)[`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 92.6 | 1.08× |
| Heterocycle | 74.3 | 69.1 | 1.08× |
| Amide (-CONH-) | 62.2 | 61.6 | 1.01× |
| Nitro (-NO₂) | 40.5 | 29.6 | 1.37× |
| Ketone (>C=O) | 35.8 | 31.6 | 1.13× |
| Halide (C-X) | 31.1 | 36.2 | 0.86× |
| Secondary Amine (>NH) | 25.0 | 19.5 | 1.28× |
| Imine (C=N) | 23.6 | 19.2 | 1.23× |

**Cluster 60 (145 molecules)** — representative: `Cc1nccn1-c1ccc(C(=O)/C=C/c2ccc([N+](=O)[`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 92.6 | 1.08× |
| Heterocycle | 92.4 | 69.1 | 1.34× |
| Amide (-CONH-) | 66.9 | 61.6 | 1.09× |
| Ketone (>C=O) | 44.8 | 31.6 | 1.42× |
| Nitro (-NO₂) | 34.5 | 29.6 | 1.16× |
| Halide (C-X) | 28.3 | 36.2 | 0.78× |
| Thioether (C-S-C) | 24.1 | 31.4 | 0.77× |
| Ether (C-O-C) | 22.8 | 34.1 | 0.67× |

### Stratum 1 ([0.399, 0.520)) — Cluster FG Signatures

**Cluster 90 (1271 molecules)** — representative: `Cn1ccnc1[C@H](NC(=O)c1ccc(-n2cnc3ccccc3c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.8 | 91.5 | 1.09× |
| Heterocycle | 98.0 | 68.5 | 1.43× |
| Amide (-CONH-) | 54.8 | 65.7 | 0.83× |
| Halide (C-X) | 29.7 | 36.5 | 0.81× |
| Ketone (>C=O) | 27.3 | 24.0 | 1.14× |
| Ether (C-O-C) | 16.6 | 36.0 | 0.46× ⬇ |
| Secondary Amine (>NH) | 11.3 | 21.2 | 0.53× |
| Thioether (C-S-C) | 11.3 | 20.5 | 0.55× |

**Cluster 9 (1009 molecules)** — representative: `[NH3+][C@H]1CCCCC[C@@H]1C(=O)N[C@@H](CCO`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 41.0 | 65.7 | 0.62× |
| Secondary Amine (>NH) | 40.6 | 21.2 | 1.91× ⬆ |
| Ether (C-O-C) | 29.2 | 36.0 | 0.81× |
| Hydroxyl (-OH) | 23.6 | 8.3 | 2.84× ⬆ |
| Tertiary Amine (>N<) | 20.3 | 15.8 | 1.28× |
| Ester (-COO-) | 18.6 | 13.9 | 1.34× |
| Primary Amine (-NH₂) | 15.4 | 8.0 | 1.93× ⬆ |
| Halide (C-X) | 13.8 | 36.5 | 0.38× ⬇ |

**Cluster 93 (436 molecules)** — representative: `Cc1cc(C(=O)N/N=C2\C(=O)N(Cc3cccc4ccccc34`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.3 | 91.5 | 1.09× |
| Heterocycle | 90.4 | 68.5 | 1.32× |
| Amide (-CONH-) | 74.1 | 65.7 | 1.13× |
| Halide (C-X) | 39.4 | 36.5 | 1.08× |
| Ketone (>C=O) | 33.0 | 24.0 | 1.37× |
| Ether (C-O-C) | 24.3 | 36.0 | 0.68× |
| Sulfonyl (-SO₂-) | 17.4 | 8.6 | 2.03× ⬆ |
| Secondary Amine (>NH) | 13.8 | 21.2 | 0.65× |

**Cluster 94 (430 molecules)** — representative: `Cc1ccc(C(=O)N/N=C\c2ccc(-c3ccc([N+](=O)[`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.1 | 91.5 | 1.08× |
| Heterocycle | 80.0 | 68.5 | 1.17× |
| Amide (-CONH-) | 71.6 | 65.7 | 1.09× |
| Halide (C-X) | 38.6 | 36.5 | 1.06× |
| Ketone (>C=O) | 29.8 | 24.0 | 1.24× |
| Ether (C-O-C) | 24.9 | 36.0 | 0.69× |
| Secondary Amine (>NH) | 22.6 | 21.2 | 1.06× |
| Sulfonyl (-SO₂-) | 21.4 | 8.6 | 2.49× ⬆ |

**Cluster 40 (392 molecules)** — representative: `Cc1cccc(-n2nc3c(c2-n2cccc2)CN(C(=O)COc2c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.7 | 91.5 | 1.09× |
| Heterocycle | 99.0 | 68.5 | 1.45× |
| Amide (-CONH-) | 67.9 | 65.7 | 1.03× |
| Ether (C-O-C) | 46.9 | 36.0 | 1.31× |
| Halide (C-X) | 34.9 | 36.5 | 0.96× |
| Tertiary Amine (>N<) | 28.8 | 15.8 | 1.82× ⬆ |
| Ketone (>C=O) | 20.4 | 24.0 | 0.85× |
| Secondary Amine (>NH) | 15.8 | 21.2 | 0.74× |

### Stratum 2 ([0.520, 0.694)) — Cluster FG Signatures

**Cluster 9 (4849 molecules)** — representative: `Cc1cc(C)n(-c2nc(-c3ccccc3)cc(=O)n2CC(=O)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.6 | 84.5 | 1.17× |
| Heterocycle | 94.2 | 61.7 | 1.53× ⬆ |
| Amide (-CONH-) | 56.5 | 68.0 | 0.83× |
| Halide (C-X) | 31.2 | 34.9 | 0.89× |
| Ketone (>C=O) | 18.0 | 14.4 | 1.25× |
| Secondary Amine (>NH) | 15.9 | 25.3 | 0.63× |
| Ether (C-O-C) | 15.7 | 35.0 | 0.45× ⬇ |
| Sulfonyl (-SO₂-) | 8.0 | 11.4 | 0.70× |

**Cluster 90 (4772 molecules)** — representative: `CCCO[C@@H]1CC[C@]([NH2+]C2CC2)(C(N)=O)C1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 44.6 | 68.0 | 0.66× |
| Secondary Amine (>NH) | 43.2 | 25.3 | 1.70× ⬆ |
| Ether (C-O-C) | 32.0 | 35.0 | 0.91× |
| Hydroxyl (-OH) | 23.7 | 10.2 | 2.33× ⬆ |
| Tertiary Amine (>N<) | 22.1 | 20.0 | 1.11× |
| Primary Amine (-NH₂) | 18.1 | 6.9 | 2.64× ⬆ |
| Carboxyl (-COOH) | 12.2 | 4.3 | 2.82× ⬆ |
| Ester (-COO-) | 8.4 | 10.0 | 0.84× |

**Cluster 4 (1562 molecules)** — representative: `C[NH+](C)[C@H](CNC(=O)c1cccc(CN2C(=O)c3c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 97.6 | 84.5 | 1.15× |
| Amide (-CONH-) | 76.7 | 68.0 | 1.13× |
| Heterocycle | 67.6 | 61.7 | 1.10× |
| Halide (C-X) | 43.4 | 34.9 | 1.24× |
| Ether (C-O-C) | 26.1 | 35.0 | 0.74× |
| Sulfonyl (-SO₂-) | 23.1 | 11.4 | 2.03× ⬆ |
| Secondary Amine (>NH) | 19.2 | 25.3 | 0.76× |
| Ketone (>C=O) | 17.6 | 14.4 | 1.22× |

**Cluster 5 (1382 molecules)** — representative: `CN(C)c1ccc(C(=O)N/N=C2\C(=O)N(Cc3ccccc3C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 97.8 | 84.5 | 1.16× |
| Amide (-CONH-) | 75.2 | 68.0 | 1.11× |
| Heterocycle | 74.2 | 61.7 | 1.20× |
| Halide (C-X) | 42.4 | 34.9 | 1.22× |
| Ether (C-O-C) | 23.3 | 35.0 | 0.67× |
| Sulfonyl (-SO₂-) | 22.8 | 11.4 | 2.00× ⬆ |
| Ketone (>C=O) | 21.1 | 14.4 | 1.46× |
| Secondary Amine (>NH) | 19.4 | 25.3 | 0.77× |

**Cluster 59 (1325 molecules)** — representative: `COc1ccc(NC(=S)N2CCn3cccc3[C@@H]2c2ccccc2`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.6 | 84.5 | 1.17× |
| Heterocycle | 86.2 | 61.7 | 1.40× |
| Amide (-CONH-) | 62.0 | 68.0 | 0.91× |
| Ether (C-O-C) | 43.5 | 35.0 | 1.24× |
| Halide (C-X) | 38.6 | 34.9 | 1.11× |
| Tertiary Amine (>N<) | 27.8 | 20.0 | 1.39× |
| Secondary Amine (>NH) | 21.1 | 25.3 | 0.83× |
| Thioether (C-S-C) | 16.6 | 13.2 | 1.25× |

### Stratum 3 ([0.694, 0.814)) — Cluster FG Signatures

**Cluster 0 (7471 molecules)** — representative: `Cc1cccc(Cl)c1NC(=O)c1ccc(-n2cncn2)c(F)c1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.3 | 79.3 | 1.24× |
| Heterocycle | 94.3 | 54.2 | 1.74× ⬆ |
| Amide (-CONH-) | 67.9 | 68.4 | 0.99× |
| Halide (C-X) | 35.3 | 33.0 | 1.07× |
| Ether (C-O-C) | 20.5 | 36.9 | 0.56× |
| Secondary Amine (>NH) | 17.7 | 28.2 | 0.63× |
| Ketone (>C=O) | 14.7 | 9.0 | 1.63× ⬆ |
| Sulfonyl (-SO₂-) | 9.4 | 11.3 | 0.83× |

**Cluster 99 (6339 molecules)** — representative: `C[C@H]1CCCN1C(=O)CC[NH+](C1CC1)[C@@H](C)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 51.4 | 68.4 | 0.75× |
| Ether (C-O-C) | 42.7 | 36.9 | 1.16× |
| Secondary Amine (>NH) | 38.1 | 28.2 | 1.35× |
| Tertiary Amine (>N<) | 29.2 | 22.2 | 1.32× |
| Hydroxyl (-OH) | 17.6 | 11.8 | 1.49× |
| Primary Amine (-NH₂) | 13.3 | 6.6 | 2.00× ⬆ |
| Thioether (C-S-C) | 12.2 | 9.1 | 1.35× |
| Ester (-COO-) | 7.0 | 7.0 | 1.00× |

**Cluster 5 (2561 molecules)** — representative: `CCNC(=O)NNC(=O)c1c(C)nn(-c2ccc(F)cc2)c1C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.3 | 79.3 | 1.21× |
| Amide (-CONH-) | 84.2 | 68.4 | 1.23× |
| Heterocycle | 47.0 | 54.2 | 0.87× |
| Halide (C-X) | 46.4 | 33.0 | 1.41× |
| Ether (C-O-C) | 31.5 | 36.9 | 0.85× |
| Secondary Amine (>NH) | 27.4 | 28.2 | 0.97× |
| Sulfonyl (-SO₂-) | 24.5 | 11.3 | 2.17× ⬆ |
| Ketone (>C=O) | 12.5 | 9.0 | 1.39× |

**Cluster 49 (2168 molecules)** — representative: `CCSCC[C@H](C)N(C)C(=O)c1ccc(Br)o1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 70.7 | 79.3 | 0.89× |
| Amide (-CONH-) | 65.5 | 68.4 | 0.96× |
| Secondary Amine (>NH) | 39.1 | 28.2 | 1.39× |
| Heterocycle | 34.8 | 54.2 | 0.64× |
| Halide (C-X) | 33.8 | 33.0 | 1.02× |
| Ether (C-O-C) | 33.0 | 36.9 | 0.89× |
| Hydroxyl (-OH) | 22.4 | 11.8 | 1.89× ⬆ |
| Tertiary Amine (>N<) | 14.7 | 22.2 | 0.66× |

**Cluster 4 (1962 molecules)** — representative: `O=C(COC(=O)c1ncc(Cl)c(Cl)c1Cl)NC(=O)NCc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.9 | 79.3 | 1.22× |
| Amide (-CONH-) | 79.5 | 68.4 | 1.16× |
| Heterocycle | 49.5 | 54.2 | 0.91× |
| Halide (C-X) | 46.0 | 33.0 | 1.39× |
| Sulfonyl (-SO₂-) | 31.8 | 11.3 | 2.82× ⬆ |
| Secondary Amine (>NH) | 29.5 | 28.2 | 1.04× |
| Ether (C-O-C) | 27.6 | 36.9 | 0.75× |
| Tertiary Amine (>N<) | 13.4 | 22.2 | 0.60× |

### Stratum 4 ([0.814, 1.0]) — Cluster FG Signatures

**Cluster 90 (5866 molecules)** — representative: `O=C(NCc1ccco1)C(=O)Nc1cc(Cl)ccc1Cl`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.5 | 83.0 | 1.16× |
| Amide (-CONH-) | 73.5 | 68.7 | 1.07× |
| Heterocycle | 60.6 | 56.1 | 1.08× |
| Halide (C-X) | 49.2 | 37.1 | 1.33× |
| Sulfonyl (-SO₂-) | 31.6 | 11.0 | 2.88× ⬆ |
| Secondary Amine (>NH) | 24.8 | 30.1 | 0.83× |
| Ether (C-O-C) | 22.7 | 39.9 | 0.57× |
| Nitrile (-C≡N) | 8.9 | 5.5 | 1.63× ⬆ |

**Cluster 9 (4193 molecules)** — representative: `CC(C)(C)c1n[nH]cc1CN1CC[C@@H](C[NH+]2CCC`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Ether (C-O-C) | 54.8 | 39.9 | 1.37× |
| Heterocycle | 44.2 | 56.1 | 0.79× |
| Secondary Amine (>NH) | 43.5 | 30.1 | 1.45× |
| Tertiary Amine (>N<) | 40.0 | 24.2 | 1.65× ⬆ |
| Amide (-CONH-) | 35.6 | 68.7 | 0.52× |
| Phenyl (aromatic ring) | 27.1 | 83.0 | 0.33× ⬇ |
| Hydroxyl (-OH) | 18.5 | 11.6 | 1.59× ⬆ |
| Halide (C-X) | 18.0 | 37.1 | 0.48× ⬇ |

**Cluster 4 (1815 molecules)** — representative: `CCC(=O)Nc1ccc([C@H](C)NC(=O)N2C[C@H](C)O`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 71.8 | 68.7 | 1.05× |
| Phenyl (aromatic ring) | 61.1 | 83.0 | 0.74× |
| Ether (C-O-C) | 43.0 | 39.9 | 1.08× |
| Heterocycle | 42.9 | 56.1 | 0.76× |
| Halide (C-X) | 35.8 | 37.1 | 0.97× |
| Secondary Amine (>NH) | 32.9 | 30.1 | 1.09× |
| Tertiary Amine (>N<) | 21.9 | 24.2 | 0.90× |
| Hydroxyl (-OH) | 16.0 | 11.6 | 1.38× |

**Cluster 5 (1739 molecules)** — representative: `CCC[NH+]1[C@H]2CCC[C@@H]1CC(NC(=O)c1ccc(`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 77.5 | 68.7 | 1.13× |
| Phenyl (aromatic ring) | 67.7 | 83.0 | 0.82× |
| Ether (C-O-C) | 46.4 | 39.9 | 1.16× |
| Halide (C-X) | 37.6 | 37.1 | 1.01× |
| Secondary Amine (>NH) | 37.3 | 30.1 | 1.24× |
| Heterocycle | 36.1 | 56.1 | 0.64× |
| Tertiary Amine (>N<) | 24.6 | 24.2 | 1.02× |
| Hydroxyl (-OH) | 15.9 | 11.6 | 1.36× |

**Cluster 94 (1697 molecules)** — representative: `Cc1ccc(-c2ncco2)cc1NC(=O)CCN1CCOC1=O`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.4 | 83.0 | 1.16× |
| Amide (-CONH-) | 69.9 | 68.7 | 1.02× |
| Heterocycle | 64.5 | 56.1 | 1.15× |
| Halide (C-X) | 41.4 | 37.1 | 1.12× |
| Ether (C-O-C) | 36.9 | 39.9 | 0.93× |
| Secondary Amine (>NH) | 26.7 | 30.1 | 0.89× |
| Tertiary Amine (>N<) | 19.1 | 24.2 | 0.79× |
| Sulfonyl (-SO₂-) | 9.7 | 11.0 | 0.88× |

## 9. Cluster Quality Analysis

### 9.1 Per-Stratum Quality Metrics

| Stratum | Silhouette | Davies-Bouldin | QE | Clusters | Gini | Singletons |
|---|---|---|---|---|---|---|
| 0 [0, 0.399) | -0.0203 | 3.3488 | 0.028363 | 100 | 0.382 | 0 |
| 1 [0.399, 0.520) | -0.0117 | 3.3535 | 0.029087 | 100 | 0.383 | 0 |
| 2 [0.520, 0.694) | -0.0027 | 3.3211 | 0.031433 | 100 | 0.402 | 0 |
| 3 [0.694, 0.814) | 0.0088 | 3.3084 | 0.031549 | 100 | 0.430 | 0 |
| 4 [0.814, 1.0] | -0.0302 | 3.6813 | 0.027250 | 100 | 0.369 | 0 |

**Interpretation guide:**
- **Silhouette** ∈ [-1, 1]: higher = better separation (>0.5 strong, >0.25 reasonable)
- **Davies-Bouldin**: lower = better separation (0 is optimal)
- **Gini coefficient**: 0 = equal sizes, 1 = maximally unequal

### 9.2 Cluster Size Distribution

| Stratum | Mean | Median | Std | Min | P25 | P75 | Max | Large |
|---|---|---|---|---|---|---|---|---|
| 0 | 68.3 | 50 | 62.4 | 12 | 33 | 86 | 493 | 2 |
| 1 | 176.2 | 143 | 166.6 | 36 | 85 | 208 | 1271 | 2 |
| 2 | 604.3 | 445 | 675.0 | 61 | 306 | 720 | 4849 | 2 |
| 3 | 836.7 | 574 | 987.8 | 87 | 383 | 892 | 7471 | 3 |
| 4 | 809.0 | 594 | 735.8 | 196 | 405 | 1001 | 5866 | 2 |

## 10. Evaluation Summary

| Metric | Value |
|---|---|
| Total active clusters | 500 / 500 neurons |
| Cluster size (mean) | 498.9 |
| Cluster size (range) | 12 – 7471 |
| Average quantization error | 0.029536 |
| Average silhouette score | -0.0112 |
| Average Davies-Bouldin index | 3.4026 |
| Mean intra-cluster distance | 0.021547 |
| Functional group types detected | 22 / 22 |
| Strongest FG-property |r| | 0.1057 (Hydroxyl (-OH)) |

## 11. Performance

| Phase | Time |
|---|---|
| Data loading | 0.12s |
| Graph parsing + FG detection | 2.06s |
| VGAE encoding | 720.12s |
| Importance analysis | 0.58s |
| SOM clustering + FG analysis | 44.18s |
| **Total** | **769.75s** |

**Throughput**: 324 molecules/second

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
