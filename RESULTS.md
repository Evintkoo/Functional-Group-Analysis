# Functional Group Analysis — Experiment Results

## 1. Dataset Summary

| Property | Value |
|---|---|
| Source | ZINC15 database |
| Total molecules in file | 249455 |
| Molecules used | 5000 |
| Successfully parsed | 5000 (100.0%) |
| Parse failures | 0 |

### Molecular Property Distributions

| Property | Mean | Std | Min | Max |
|---|---|---|---|---|
| QED | 0.7275 | 0.1388 | 0.1428 | 0.9475 |
| logP | 2.4781 | 1.4488 | — | — |
| SAS | 3.0380 | 0.8339 | — | — |

## 2. Molecular Graph Statistics

| Property | Value |
|---|---|
| Atoms per molecule | 23.1 (range: 8–37) |
| Bonds per molecule | 24.8 (range: 7–42) |
| Total atoms processed | 115506 |
| Total bonds processed | 124151 |
| Node feature dimension | 29 |
| Edge feature dimension | 9 |

## 3. Functional Group Census

Detection of 22 functional group types across the entire dataset.

| Functional Group | Molecules | Prevalence (%) | Total Count | Mean per Mol |
|---|---|---|---|---|
| Phenyl (aromatic ring) | 4103 | 82.1 | 12077 | 2.42 |
| Amide (-CONH-) | 3352 | 67.0 | 4354 | 0.87 |
| Heterocycle | 2888 | 57.8 | 6741 | 1.35 |
| Ether (C-O-C) | 1942 | 38.8 | 2563 | 0.51 |
| Halide (C-X) | 1756 | 35.1 | 2787 | 0.56 |
| Secondary Amine (>NH) | 1375 | 27.5 | 1472 | 0.29 |
| Tertiary Amine (>N<) | 1021 | 20.4 | 1148 | 0.23 |
| Sulfonyl (-SO₂-) | 600 | 12.0 | 615 | 0.12 |
| Thioether (C-S-C) | 566 | 11.3 | 580 | 0.12 |
| Hydroxyl (-OH) | 552 | 11.0 | 590 | 0.12 |
| Ketone (>C=O) | 537 | 10.7 | 584 | 0.12 |
| Ester (-COO-) | 389 | 7.8 | 402 | 0.08 |
| Primary Amine (-NH₂) | 372 | 7.4 | 385 | 0.08 |
| Nitrile (-C≡N) | 256 | 5.1 | 271 | 0.05 |
| Carboxyl (-COOH) | 187 | 3.7 | 191 | 0.04 |
| Nitro (-NO₂) | 181 | 3.6 | 184 | 0.04 |
| Imine (C=N) | 124 | 2.5 | 127 | 0.03 |
| Sulfoxide (-SO-) | 48 | 1.0 | 48 | 0.01 |
| Aldehyde (-CHO) | 10 | 0.2 | 10 | 0.00 |
| Thiol (-SH) | 8 | 0.2 | 8 | 0.00 |
| Epoxide | 3 | 0.1 | 3 | 0.00 |

### Functional Group Co-occurrence Patterns

Average number of distinct functional group types per molecule and distribution.

- **Functional group types detected**: 21 out of 22
- **Ubiquitous groups** (>50%): Ph (82%), CONH (67%), HetCyc (58%)
- **Rare groups** (<5%): COOH (3.7%), NO2 (3.6%), C=N (2.5%), SO (1.0%), CHO (0.2%), SH (0.2%), Epox (0.1%)

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
| Mean reconstruction loss | 0.172195 |
| Mean pairwise embedding distance | 0.109379 |
| Embedding std (mean across dims) | 0.021832 |
| Embedding std range | [0.0096, 0.0357] |

### Latent Dimension Statistics

| Dim | Mean | Std | Min | Max |
|---|---|---|---|---|
| 0 | -0.4598 | 0.0357 | -0.5392 | -0.3190 |
| 1 | -0.1244 | 0.0096 | -0.1557 | -0.0846 |
| 2 | 0.1261 | 0.0244 | 0.0516 | 0.2114 |
| 3 | 0.0560 | 0.0169 | 0.0202 | 0.1323 |
| 4 | 0.0504 | 0.0193 | -0.0499 | 0.1111 |
| 5 | -0.0478 | 0.0325 | -0.1579 | 0.0356 |
| 6 | -0.1051 | 0.0263 | -0.1835 | -0.0039 |
| 7 | 0.3060 | 0.0154 | 0.2646 | 0.3556 |
| 8 | 0.1801 | 0.0266 | 0.1147 | 0.2716 |
| 9 | 0.3376 | 0.0233 | 0.2585 | 0.4330 |
| 10 | 0.1328 | 0.0294 | 0.0533 | 0.2213 |
| 11 | -0.1525 | 0.0185 | -0.2256 | -0.1054 |
| 12 | -0.1161 | 0.0224 | -0.2162 | -0.0581 |
| 13 | -0.1959 | 0.0271 | -0.3010 | -0.1255 |
| 14 | 0.1840 | 0.0101 | 0.1241 | 0.2208 |
| 15 | 0.0723 | 0.0118 | 0.0404 | 0.1187 |

## 6. Feature Importance Analysis

### 6.1 Latent Dimension ↔ Property Correlations

Pearson correlation (r) between each latent dimension and molecular properties.
Dimensions sorted by |r(QED)|.

| Dim | Variance | r(QED) | r(logP) | r(SAS) |
|---|---|---|---|---|
| 9 | 0.000542 | +0.2792 | -0.3354 | +0.5607 |
| 8 | 0.000708 | +0.2773 | -0.4057 | +0.6511 |
| 13 | 0.000734 | -0.2763 | +0.3880 | -0.6144 |
| 0 | 0.001275 | +0.2730 | -0.3835 | +0.6183 |
| 5 | 0.001058 | -0.2605 | +0.4791 | -0.6120 |
| 2 | 0.000598 | +0.2534 | -0.0746 | +0.4832 |
| 10 | 0.000866 | +0.2499 | -0.4893 | +0.5634 |
| 4 | 0.000372 | -0.2389 | +0.2887 | -0.5821 |
| 3 | 0.000285 | +0.2254 | -0.4153 | +0.5352 |
| 1 | 0.000093 | +0.2246 | -0.3600 | +0.5674 |
| 12 | 0.000502 | -0.2028 | +0.4734 | -0.4808 |
| 6 | 0.000689 | +0.1970 | -0.5418 | +0.4815 |
| 7 | 0.000237 | +0.1871 | -0.5197 | +0.5342 |
| 14 | 0.000103 | -0.1340 | +0.1927 | -0.3006 |
| 15 | 0.000139 | +0.1053 | -0.3625 | +0.2614 |
| 11 | 0.000341 | +0.0303 | +0.3484 | -0.0277 |

### 6.2 Functional Group ↔ Latent Space Encoding

Which latent dimensions best encode each functional group's presence.

| Functional Group | Prevalence (%) | Best Dim | |r| |
|---|---|---|---|
| Phenyl (aromatic ring) | 82.1 | 10 | 0.5955 |
| Heterocycle | 57.8 | 7 | 0.4693 |
| Sulfonyl (-SO₂-) | 12.0 | 11 | 0.3890 |
| Amide (-CONH-) | 67.0 | 2 | 0.3353 |
| Tertiary Amine (>N<) | 20.4 | 3 | 0.2867 |
| Ketone (>C=O) | 10.7 | 2 | 0.2489 |
| Carboxyl (-COOH) | 3.7 | 11 | 0.2215 |
| Hydroxyl (-OH) | 11.0 | 0 | 0.2023 |
| Halide (C-X) | 35.1 | 3 | 0.2005 |
| Nitrile (-C≡N) | 5.1 | 14 | 0.1996 |
| Ester (-COO-) | 7.8 | 2 | 0.1767 |
| Ether (C-O-C) | 38.8 | 9 | 0.1713 |
| Secondary Amine (>NH) | 27.5 | 14 | 0.1519 |
| Nitro (-NO₂) | 3.6 | 2 | 0.1432 |
| Imine (C=N) | 2.5 | 2 | 0.1372 |
| Primary Amine (-NH₂) | 7.4 | 0 | 0.1265 |
| Thioether (C-S-C) | 11.3 | 6 | 0.0615 |

### 6.3 Functional Group ↔ Molecular Property Correlations

Point-biserial correlation between FG presence and drug-likeness properties.

| Functional Group | r(QED) | r(logP) | r(SAS) |
|---|---|---|---|
| Phenyl (aromatic ring) | -0.0486 | +0.4021 | -0.4327 |
| Carboxyl (-COOH) | -0.0222 | -0.3040 | +0.1422 |
| Nitro (-NO₂) | -0.2985 | +0.0161 | -0.0549 |
| Amide (-CONH-) | +0.0504 | +0.0951 | -0.2900 |
| Halide (C-X) | +0.0129 | +0.2693 | -0.1658 |
| Imine (C=N) | -0.2191 | +0.0036 | -0.0338 |
| Ketone (>C=O) | -0.1991 | +0.0434 | -0.0511 |
| Primary Amine (-NH₂) | -0.0334 | -0.1743 | +0.1172 |
| Thioether (C-S-C) | -0.1616 | +0.1161 | -0.0067 |
| Secondary Amine (>NH) | +0.0614 | -0.0998 | +0.1589 |
| Ester (-COO-) | -0.1527 | +0.0340 | -0.0631 |
| Heterocycle | -0.0816 | +0.1378 | -0.1248 |
| Hydroxyl (-OH) | +0.0397 | -0.0943 | +0.1366 |
| Tertiary Amine (>N<) | +0.0897 | -0.1355 | +0.0385 |
| Ether (C-O-C) | +0.0326 | +0.0532 | -0.0815 |
| Sulfonyl (-SO₂-) | +0.0237 | -0.0772 | -0.0790 |
| Nitrile (-C≡N) | -0.0290 | +0.0383 | +0.0010 |

## 7. Stratified Clustering Results

### Per-Stratum Overview

| Stratum | QED Range | Molecules | Active Clusters | QE | U-Matrix Mean | U-Matrix Max |
|---|---|---|---|---|---|---|
| 0 | [0, 0.399) | 122 | 61 | 0.030548 | 0.0133 | 0.0263 |
| 1 | [0.399, 0.520) | 376 | 89 | 0.030383 | 0.0122 | 0.0205 |
| 2 | [0.520, 0.694) | 1243 | 99 | 0.034197 | 0.0144 | 0.0271 |
| 3 | [0.694, 0.814) | 1645 | 99 | 0.035153 | 0.0139 | 0.0207 |
| 4 | [0.814, 1.0] | 1614 | 99 | 0.030387 | 0.0107 | 0.0168 |

**Total clustered**: 5000 molecules | **Avg QE**: 0.032134

### Stratum 0 ([0, 0.399)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 91.0 | 3.66 |
| Heterocycle | 66.4 | 1.93 |
| Amide (-CONH-) | 55.7 | 0.67 |
| Ether (C-O-C) | 36.9 | 0.52 |
| Ketone (>C=O) | 32.8 | 0.36 |
| Halide (C-X) | 28.7 | 0.45 |
| Nitro (-NO₂) | 27.9 | 0.28 |
| Thioether (C-S-C) | 25.4 | 0.25 |
| Imine (C=N) | 23.8 | 0.25 |
| Ester (-COO-) | 19.7 | 0.20 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 9 | 11 | 0.360±0.034 | 4.02 | 2.37 | 0.0335 | Ph | C=O(1.9×), C-S-C(1.8×), C=N(1.5×) | `CC(=O)NCc1ccc(C(=O)CSc2nnc3sc4` |
| 90 | 8 | 0.309±0.075 | 0.69 | 3.81 | 0.0529 | OH | OH(3.6×), N<(3.4×), NH2(2.7×) | `CCN(C[C@@H](C)/C(N)=N/O)C(=O)C` |
| 0 | 4 | 0.313±0.060 | 4.29 | 2.55 | 0.0192 | Ph | N<(3.4×), C-S-C(2.0×), NO2(1.8×) | `CCc1cccc(C)c1NC(=O)CSc1nc2cccc` |
| 5 | 4 | 0.345±0.037 | 4.44 | 2.71 | 0.0153 | Ph | C-O-C(2.0×), OH(1.8×), C=O(1.5×) | `O=[N+]([O-])c1c(Nc2ccc(F)c(F)c` |
| 6 | 4 | 0.301±0.045 | 4.47 | 2.50 | 0.0188 | Ph | C-S-C(2.0×), C=O(1.5×), HetCyc(1.5×) | `C=CCn1c(SCc2nnc([S-])n2-c2cccc` |
| 29 | 4 | 0.265±0.046 | 3.90 | 2.42 | 0.0196 | Ph | SO2(6.1×), CN(3.4×), NH2(2.7×) | `NS(=O)(=O)c1ccc(N/N=C/c2cn(-c3` |
| 49 | 4 | 0.356±0.061 | 2.85 | 2.41 | 0.0169 | Ph | NO2(2.7×), C=N(2.1×), SO2(2.0×) | `C/C(=N\NC(=O)c1ccc(Cl)c([N+](=` |
| 2 | 3 | 0.361±0.017 | 4.75 | 2.78 | 0.0186 | C-O-C | CN(4.5×), C-O-C(2.7×), NH2(1.8×) | `Cc1ccc2cccc(Nc3ncnc(Nc4ccc5c(c` |
| 4 | 3 | 0.335±0.051 | 4.23 | 2.48 | 0.0131 | Ph | C-X(2.3×), COO(1.7×), HetCyc(1.5×) | `Cc1ccc2[nH]nc(C(=O)NCc3nc(-c4c` |
| 25 | 3 | 0.350±0.047 | 4.64 | 2.61 | 0.0165 | CONH | CN(4.5×), C-O-C(2.7×), C=O(2.0×) | `CCCOc1cccc(C(=O)C2=C([O-])C(=O` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 58 | 69 | 0.011861 |
| 5 | 6 | 0.013257 |
| 39 | 49 | 0.014115 |
| 75 | 86 | 0.014591 |
| 37 | 48 | 0.016289 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.317856 |
| 6 | 90 | 0.304512 |
| 7 | 90 | 0.301779 |
| 29 | 90 | 0.297868 |
| 5 | 90 | 0.293845 |

Inter-cluster distance: mean=0.090554, min=0.011861, max=0.317856, 1830 pairs

### Stratum 1 ([0.399, 0.520)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 90.7 | 3.59 |
| Heterocycle | 70.5 | 2.07 |
| Amide (-CONH-) | 62.8 | 0.85 |
| Halide (C-X) | 36.2 | 0.56 |
| Ether (C-O-C) | 35.9 | 0.51 |
| Ketone (>C=O) | 22.1 | 0.24 |
| Secondary Amine (>NH) | 21.8 | 0.28 |
| Thioether (C-S-C) | 20.7 | 0.21 |
| Ester (-COO-) | 16.5 | 0.17 |
| Nitro (-NO₂) | 14.4 | 0.15 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 99 | 30 | 0.472±0.035 | 3.98 | 2.35 | 0.0321 | Ph | COO(1.4×), HetCyc(1.3×) | `CCCn1cc(NC(=O)c2cc3nc(-c4ccccc` |
| 0 | 23 | 0.475±0.035 | 0.01 | 4.37 | 0.0688 | COO | CHO(6.5×), COOH(5.4×), OH(3.4×) | `COC(=O)C(CC[C@@]1(C)[C@@H](C)C` |
| 50 | 12 | 0.478±0.032 | 1.92 | 2.89 | 0.0316 | Ph | SO2(4.5×), NO2(3.5×), COO(3.0×) | `Cc1nc([N+](=O)[O-])cn1CCC(=O)O` |
| 95 | 12 | 0.483±0.031 | 3.01 | 2.40 | 0.0181 | Ph | C=N(2.8×), SO2(2.2×), C=O(1.9×) | `Cn1cc(C(=O)Nc2ccccc2C(=O)NCCc2` |
| 94 | 11 | 0.467±0.027 | 3.58 | 2.30 | 0.0217 | Ph | C=O(3.3×), COOH(2.8×), COO(2.2×) | `CC(=O)Oc1ccc(C(=O)c2cccc(C)c2)` |
| 55 | 10 | 0.456±0.032 | 4.42 | 2.57 | 0.0165 | Ph | OH(2.6×), C-X(1.7×), C-S-C(1.4×) | `Cc1ccc(-c2cnc(CCC(=O)O[C@H](C)` |
| 59 | 9 | 0.473±0.038 | 4.39 | 2.31 | 0.0194 | Ph | NH(2.0×), C-O-C(1.9×), C-X(1.5×) | `COc1ccccc1CNC(=O)c1cc2sccc2n1C` |
| 2 | 7 | 0.468±0.037 | 1.68 | 3.74 | 0.0304 | N< | N<(6.6×), C-S-C(2.1×), SO2(1.9×) | `Cn1c(SCC(=O)N2CCN(Cc3c(F)cccc3` |
| 39 | 7 | 0.465±0.034 | 3.07 | 2.75 | 0.0149 | Ph | NH2(7.3×), CN(1.9×), C-O-C(1.6×) | `CCCc1nnc2sc(-c3cccc(NC(=O)c4cc` |
| 49 | 7 | 0.478±0.031 | 4.00 | 2.49 | 0.0208 | Ph | N<(2.2×), CN(1.9×), HetCyc(1.4×) | `COc1cccc([C@H]2C[C@H]2C(=O)Oc2` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 91 | 92 | 0.010394 |
| 18 | 28 | 0.010928 |
| 92 | 93 | 0.010978 |
| 77 | 86 | 0.011160 |
| 95 | 96 | 0.011820 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 0 | 99 | 0.308656 |
| 0 | 89 | 0.291785 |
| 0 | 98 | 0.290620 |
| 0 | 97 | 0.279253 |
| 0 | 79 | 0.276440 |

Inter-cluster distance: mean=0.076344, min=0.010394, max=0.308656, 3916 pairs

### Stratum 2 ([0.520, 0.694)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 82.8 | 2.76 |
| Amide (-CONH-) | 66.9 | 0.90 |
| Heterocycle | 62.4 | 1.56 |
| Ether (C-O-C) | 36.8 | 0.50 |
| Halide (C-X) | 34.4 | 0.56 |
| Secondary Amine (>NH) | 25.6 | 0.29 |
| Tertiary Amine (>N<) | 18.1 | 0.20 |
| Thioether (C-S-C) | 15.3 | 0.16 |
| Ketone (>C=O) | 13.6 | 0.15 |
| Sulfonyl (-SO₂-) | 12.1 | 0.12 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 0 | 93 | 0.625±0.046 | 0.29 | 4.54 | 0.0594 | CONH | OH(3.1×), NH2(2.5×), N<(1.5×) | `CC(C)[C@H](CNC(=O)N[C@H]1C[C@@` |
| 99 | 86 | 0.594±0.054 | 3.38 | 2.43 | 0.0341 | Ph | C=O(1.8×), HetCyc(1.5×), SO2(1.3×) | `O=Cc1ccc(OCc2ccn(-c3cccc(F)c3)` |
| 59 | 35 | 0.636±0.041 | 3.98 | 2.60 | 0.0258 | Ph | CN(2.2×), C-O-C(1.6×), C-S-C(1.5×) | `COc1ccccc1NC(=O)CSc1ccc(-c2ccc` |
| 95 | 34 | 0.616±0.049 | 2.83 | 2.58 | 0.0261 | Ph | SO2(2.7×), C=O(2.2×), COO(1.8×) | `CCOC(=O)c1ccccc1NC(=O)/C=C/c1c` |
| 50 | 30 | 0.628±0.039 | 1.38 | 3.28 | 0.0292 | CONH | COOH(3.7×), NO2(3.1×), C-S-C(1.7×) | `CC(=O)Nc1cccc(NC(=O)C(=O)NCC[N` |
| 94 | 28 | 0.611±0.046 | 2.78 | 2.55 | 0.0260 | Ph | C=N(4.3×), C=O(2.6×), SO2(2.4×) | `Cc1nc(NC(=O)c2nn(-c3ccc(Cl)cc3` |
| 96 | 27 | 0.606±0.052 | 3.09 | 2.43 | 0.0250 | Ph | SO2(2.4×), NO2(2.0×), C=O(1.6×) | `Cc1ccc(NS(=O)(=O)c2ccc(OCC(=O)` |
| 97 | 27 | 0.624±0.049 | 2.97 | 2.46 | 0.0251 | Ph | C=N(5.6×), SO2(2.7×), C=O(1.9×) | `Cc1nc(NC(=O)c2ccc3ccccc3n2)sc1` |
| 49 | 26 | 0.642±0.037 | 3.98 | 2.59 | 0.0250 | Ph | HetCyc(1.5×), CN(1.4×), C-X(1.3×) | `CN(Cc1nccn1Cc1ccccc1)C(=O)[C@@` |
| 9 | 23 | 0.645±0.039 | 2.84 | 3.49 | 0.0239 | Ph | OH(3.1×), N<(2.4×), NH(1.7×) | `COc1cccc(CN2CCc3nnc(CCc4ccccc4` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 19 | 29 | 0.008927 |
| 38 | 47 | 0.008946 |
| 68 | 78 | 0.010692 |
| 18 | 28 | 0.010984 |
| 95 | 96 | 0.011108 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 0 | 99 | 0.339045 |
| 0 | 89 | 0.317883 |
| 0 | 98 | 0.315141 |
| 0 | 97 | 0.309426 |
| 0 | 79 | 0.301968 |

Inter-cluster distance: mean=0.086385, min=0.008927, max=0.339045, 4851 pairs

### Stratum 3 ([0.694, 0.814)) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 77.6 | 2.15 |
| Amide (-CONH-) | 66.0 | 0.85 |
| Heterocycle | 53.3 | 1.21 |
| Ether (C-O-C) | 39.7 | 0.52 |
| Halide (C-X) | 33.9 | 0.53 |
| Secondary Amine (>NH) | 28.7 | 0.30 |
| Tertiary Amine (>N<) | 21.5 | 0.24 |
| Hydroxyl (-OH) | 12.7 | 0.14 |
| Sulfonyl (-SO₂-) | 12.7 | 0.13 |
| Ketone (>C=O) | 10.8 | 0.12 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 90 | 146 | 0.748±0.029 | 3.05 | 2.38 | 0.0329 | Ph | HetCyc(1.7×), NH2(1.3×), Ph(1.3×) | `C[C@H](NC(=O)Nc1ccncc1)c1nc(-c` |
| 9 | 116 | 0.750±0.033 | 1.26 | 4.34 | 0.0518 | CONH | NH2(2.3×), CN(2.0×), OH(1.6×) | `CC1(C)CCC[C@@H](C[NH+](CCO)C2C` |
| 40 | 50 | 0.767±0.027 | 2.48 | 2.62 | 0.0296 | Ph | SO2(4.7×), C-X(1.9×), C=O(1.5×) | `COc1ccc(CCCC(=O)Nc2cccc(S(N)(=` |
| 50 | 41 | 0.758±0.036 | 2.57 | 2.39 | 0.0267 | Ph | SO2(4.0×), COOH(1.6×), C-X(1.4×) | `CC(C)N(C(=O)CS(=O)(=O)Cc1ccc(C` |
| 54 | 37 | 0.760±0.033 | 2.89 | 2.80 | 0.0241 | Ph | COOH(1.8×), COO(1.8×), C-O-C(1.5×) | `COc1ccccc1[C@H]1CCCN1C(=O)Nc1c` |
| 49 | 33 | 0.761±0.036 | 1.51 | 4.15 | 0.0326 | Ph | OH(2.1×), NH(1.6×), N<(1.4×) | `COCCOc1cccc(C[NH+]2CCC2(C)C)c1` |
| 4 | 32 | 0.760±0.034 | 1.82 | 3.36 | 0.0304 | CONH | SO2(1.5×), CONH(1.3×), COO(1.2×) | `CC(C)[C@H]1CN(C(=O)CCC(N)=O)CC` |
| 3 | 30 | 0.768±0.029 | 1.60 | 3.26 | 0.0265 | CONH | SO2(3.7×), COO(3.0×), N<(1.5×) | `CCCNC(=O)NC(=O)COC(=O)C1(c2ccc` |
| 59 | 30 | 0.766±0.033 | 1.76 | 3.90 | 0.0344 | Ph | NH2(4.6×), OH(2.4×), N<(1.5×) | `Oc1ccc(C[NH2+]C[C@H]2CCCC[C@H]` |
| 93 | 30 | 0.746±0.035 | 3.15 | 2.75 | 0.0274 | Ph | HetCyc(1.8×), CN(1.5×), C-S-C(1.4×) | `CC1(C)CC[C@@H](CNC(=O)c2ccc3nn` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 75 | 86 | 0.010276 |
| 76 | 86 | 0.010377 |
| 16 | 27 | 0.010827 |
| 53 | 64 | 0.011131 |
| 66 | 77 | 0.011204 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.318692 |
| 9 | 80 | 0.298034 |
| 9 | 91 | 0.295067 |
| 9 | 70 | 0.291298 |
| 9 | 60 | 0.286769 |

Inter-cluster distance: mean=0.085676, min=0.010276, max=0.318692, 4851 pairs

### Stratum 4 ([0.814, 1.0]) — Detailed Analysis

#### Functional Group Distribution

| Functional Group | Prevalence (%) | Mean Count |
|---|---|---|
| Phenyl (aromatic ring) | 83.3 | 2.06 |
| Amide (-CONH-) | 70.1 | 0.89 |
| Heterocycle | 55.1 | 1.11 |
| Ether (C-O-C) | 40.4 | 0.51 |
| Halide (C-X) | 37.2 | 0.59 |
| Secondary Amine (>NH) | 29.7 | 0.30 |
| Tertiary Amine (>N<) | 23.8 | 0.27 |
| Sulfonyl (-SO₂-) | 12.2 | 0.12 |
| Hydroxyl (-OH) | 12.0 | 0.12 |
| Primary Amine (-NH₂) | 7.0 | 0.07 |

#### Top Clusters by Size

| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |
|---|---|---|---|---|---|---|---|---|
| 9 | 110 | 0.863±0.032 | 1.90 | 4.01 | 0.0470 | C-O-C | N<(1.9×), NH2(1.7×), C-O-C(1.4×) | `Fc1ccc(C[NH2+]C[C@@H]([C@H]2CC` |
| 90 | 94 | 0.874±0.033 | 2.40 | 2.33 | 0.0307 | Ph | SO2(4.2×), C=O(3.0×), COOH(2.5×) | `CC(C)NS(=O)(=O)c1ccc(C(=O)Nc2c` |
| 0 | 55 | 0.878±0.039 | 2.59 | 2.90 | 0.0246 | Ph | N<(1.8×), C-O-C(1.4×), OH(1.2×) | `O=C(c1ccc(O)cc1)N1CCN(CCc2cccc` |
| 99 | 39 | 0.859±0.028 | 1.63 | 3.06 | 0.0283 | CONH | COOH(4.0×), C=O(3.0×), SO2(2.9×) | `NC(=O)N1CCCN(C(=O)Nc2cc(Cl)ccc` |
| 40 | 36 | 0.873±0.034 | 2.92 | 2.82 | 0.0268 | Ph | CN(2.7×), OH(2.3×), NH2(2.0×) | `CCNc1ccc2c(OC)ccc(F)c2n1` |
| 49 | 34 | 0.856±0.028 | 2.10 | 3.20 | 0.0290 | CONH | SO(3.3×), COO(2.6×), N<(1.4×) | `O=C(CSCC(F)(F)F)N1CCN(c2ccc(Cl` |
| 59 | 34 | 0.862±0.033 | 2.29 | 3.09 | 0.0256 | CONH | C=O(2.1×), COO(1.8×), CONH(1.3×) | `Cc1ccc(NC(=O)N2CCCN(C(=O)C(C)(` |
| 50 | 32 | 0.854±0.030 | 2.85 | 2.81 | 0.0337 | Ph | NH2(3.6×), CN(3.1×), C-S-C(1.3×) | `CC(C)Cc1ccc([C@H](C)NC(=O)c2cc` |
| 3 | 31 | 0.872±0.038 | 2.44 | 3.56 | 0.0228 | Ph | OH(1.9×), C-S-C(1.8×), HetCyc(1.5×) | `Cn1cc(CCC[NH+]2CCC(c3ccc(F)cc3` |
| 39 | 31 | 0.861±0.030 | 2.34 | 3.46 | 0.0215 | CONH | C-S-C(1.8×), SO(1.8×) | `COCc1cc(C(=O)N[C@@H]2CCCC[C@H]` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 81 | 82 | 0.007674 |
| 10 | 20 | 0.008699 |
| 73 | 83 | 0.008965 |
| 11 | 22 | 0.009538 |
| 41 | 42 | 0.010023 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 9 | 90 | 0.228040 |
| 9 | 80 | 0.214741 |
| 9 | 70 | 0.204832 |
| 9 | 91 | 0.203341 |
| 19 | 90 | 0.203008 |

Inter-cluster distance: mean=0.069927, min=0.007674, max=0.228040, 4851 pairs

## 8. Cluster Functional Group Characterization

Summary of functional group signatures across the largest clusters in each stratum.
Enrichment ratio shows over-representation relative to the stratum population.

### Stratum 0 ([0, 0.399)) — Cluster FG Signatures

**Cluster 9 (11 molecules)** — representative: `CC(=O)NCc1ccc(C(=O)CSc2nnc3sc4ccccc4n23)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 91.0 | 1.10× |
| Heterocycle | 90.9 | 66.4 | 1.37× |
| Ketone (>C=O) | 63.6 | 32.8 | 1.94× ⬆ |
| Thioether (C-S-C) | 45.5 | 25.4 | 1.79× ⬆ |
| Imine (C=N) | 36.4 | 23.8 | 1.53× ⬆ |
| Amide (-CONH-) | 27.3 | 55.7 | 0.49× ⬇ |
| Hydroxyl (-OH) | 9.1 | 13.9 | 0.65× |
| Secondary Amine (>NH) | 9.1 | 18.9 | 0.48× ⬇ |

**Cluster 90 (8 molecules)** — representative: `CCN(C[C@@H](C)/C(N)=N/O)C(=O)C1(C)CCCC1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Hydroxyl (-OH) | 50.0 | 13.9 | 3.59× ⬆ |
| Primary Amine (-NH₂) | 50.0 | 18.9 | 2.65× ⬆ |
| Amide (-CONH-) | 50.0 | 55.7 | 0.90× |
| Ether (C-O-C) | 50.0 | 36.9 | 1.36× |
| Secondary Amine (>NH) | 37.5 | 18.9 | 1.99× ⬆ |
| Imine (C=N) | 37.5 | 23.8 | 1.58× ⬆ |
| Tertiary Amine (>N<) | 25.0 | 7.4 | 3.39× ⬆ |
| Ketone (>C=O) | 25.0 | 32.8 | 0.76× |

**Cluster 0 (4 molecules)** — representative: `CCc1cccc(C)c1NC(=O)CSc1nc2ccccc2c(=O)n1C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 91.0 | 1.10× |
| Heterocycle | 100.0 | 66.4 | 1.51× ⬆ |
| Ether (C-O-C) | 50.0 | 36.9 | 1.36× |
| Ketone (>C=O) | 50.0 | 32.8 | 1.53× ⬆ |
| Nitro (-NO₂) | 50.0 | 27.9 | 1.79× ⬆ |
| Thioether (C-S-C) | 50.0 | 25.4 | 1.97× ⬆ |
| Halide (C-X) | 50.0 | 28.7 | 1.74× ⬆ |
| Primary Amine (-NH₂) | 25.0 | 18.9 | 1.33× |

**Cluster 5 (4 molecules)** — representative: `O=[N+]([O-])c1c(Nc2ccc(F)c(F)c2)ncnc1Oc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 91.0 | 1.10× |
| Heterocycle | 100.0 | 66.4 | 1.51× ⬆ |
| Ether (C-O-C) | 75.0 | 36.9 | 2.03× ⬆ |
| Amide (-CONH-) | 50.0 | 55.7 | 0.90× |
| Ketone (>C=O) | 50.0 | 32.8 | 1.53× ⬆ |
| Hydroxyl (-OH) | 25.0 | 13.9 | 1.79× ⬆ |
| Primary Amine (-NH₂) | 25.0 | 18.9 | 1.33× |
| Secondary Amine (>NH) | 25.0 | 18.9 | 1.33× |

**Cluster 6 (4 molecules)** — representative: `C=CCn1c(SCc2nnc([S-])n2-c2ccccc2)nnc1-c1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 91.0 | 1.10× |
| Heterocycle | 100.0 | 66.4 | 1.51× ⬆ |
| Amide (-CONH-) | 75.0 | 55.7 | 1.35× |
| Ether (C-O-C) | 50.0 | 36.9 | 1.36× |
| Ketone (>C=O) | 50.0 | 32.8 | 1.53× ⬆ |
| Thioether (C-S-C) | 50.0 | 25.4 | 1.97× ⬆ |
| Halide (C-X) | 25.0 | 28.7 | 0.87× |

### Stratum 1 ([0.399, 0.520)) — Cluster FG Signatures

**Cluster 99 (30 molecules)** — representative: `CCCn1cc(NC(=O)c2cc3nc(-c4ccccc4)cc(-c4cc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 90.7 | 1.10× |
| Heterocycle | 90.0 | 70.5 | 1.28× |
| Amide (-CONH-) | 40.0 | 62.8 | 0.64× |
| Halide (C-X) | 36.7 | 36.2 | 1.01× |
| Ester (-COO-) | 23.3 | 16.5 | 1.42× |
| Thioether (C-S-C) | 16.7 | 20.7 | 0.80× |
| Ketone (>C=O) | 13.3 | 22.1 | 0.60× |
| Secondary Amine (>NH) | 10.0 | 21.8 | 0.46× ⬇ |

**Cluster 0 (23 molecules)** — representative: `COC(=O)C(CC[C@@]1(C)[C@@H](C)CC=C[C@H]1O`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Ester (-COO-) | 39.1 | 16.5 | 2.37× ⬆ |
| Secondary Amine (>NH) | 34.8 | 21.8 | 1.59× ⬆ |
| Hydroxyl (-OH) | 26.1 | 7.7 | 3.38× ⬆ |
| Amide (-CONH-) | 26.1 | 62.8 | 0.42× ⬇ |
| Carboxyl (-COOH) | 17.4 | 3.2 | 5.45× ⬆ |
| Ether (C-O-C) | 17.4 | 35.9 | 0.48× ⬇ |
| Tertiary Amine (>N<) | 13.0 | 13.0 | 1.00× |
| Ketone (>C=O) | 13.0 | 22.1 | 0.59× |

**Cluster 50 (12 molecules)** — representative: `Cc1nc([N+](=O)[O-])cn1CCC(=O)O[C@H](C)c1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 91.7 | 90.7 | 1.01× |
| Amide (-CONH-) | 50.0 | 62.8 | 0.80× |
| Ester (-COO-) | 50.0 | 16.5 | 3.03× ⬆ |
| Nitro (-NO₂) | 50.0 | 14.4 | 3.48× ⬆ |
| Ether (C-O-C) | 41.7 | 35.9 | 1.16× |
| Heterocycle | 41.7 | 70.5 | 0.59× |
| Secondary Amine (>NH) | 33.3 | 21.8 | 1.53× ⬆ |
| Sulfonyl (-SO₂-) | 33.3 | 7.4 | 4.48× ⬆ |

**Cluster 95 (12 molecules)** — representative: `Cn1cc(C(=O)Nc2ccccc2C(=O)NCCc2ccccc2)c(=`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 90.7 | 1.10× |
| Heterocycle | 100.0 | 70.5 | 1.42× |
| Amide (-CONH-) | 83.3 | 62.8 | 1.33× |
| Halide (C-X) | 66.7 | 36.2 | 1.84× ⬆ |
| Ketone (>C=O) | 41.7 | 22.1 | 1.89× ⬆ |
| Imine (C=N) | 25.0 | 8.8 | 2.85× ⬆ |
| Secondary Amine (>NH) | 16.7 | 21.8 | 0.76× |
| Tertiary Amine (>N<) | 16.7 | 13.0 | 1.28× |

**Cluster 94 (11 molecules)** — representative: `CC(=O)Oc1ccc(C(=O)c2cccc(C)c2)cc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 90.7 | 1.10× |
| Amide (-CONH-) | 72.7 | 62.8 | 1.16× |
| Ketone (>C=O) | 72.7 | 22.1 | 3.29× ⬆ |
| Heterocycle | 72.7 | 70.5 | 1.03× |
| Ester (-COO-) | 36.4 | 16.5 | 2.21× ⬆ |
| Halide (C-X) | 36.4 | 36.2 | 1.01× |
| Carboxyl (-COOH) | 9.1 | 3.2 | 2.85× ⬆ |
| Ether (C-O-C) | 9.1 | 35.9 | 0.25× ⬇ |

### Stratum 2 ([0.520, 0.694)) — Cluster FG Signatures

**Cluster 0 (93 molecules)** — representative: `CC(C)[C@H](CNC(=O)N[C@H]1C[C@@H]1C)N1CC[`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 49.5 | 66.9 | 0.74× |
| Secondary Amine (>NH) | 38.7 | 25.6 | 1.51× ⬆ |
| Ether (C-O-C) | 33.3 | 36.8 | 0.91× |
| Tertiary Amine (>N<) | 28.0 | 18.1 | 1.54× ⬆ |
| Hydroxyl (-OH) | 25.8 | 8.3 | 3.11× ⬆ |
| Primary Amine (-NH₂) | 19.4 | 7.6 | 2.53× ⬆ |
| Carboxyl (-COOH) | 6.5 | 4.5 | 1.43× |
| Thioether (C-S-C) | 6.5 | 15.3 | 0.42× ⬇ |

**Cluster 99 (86 molecules)** — representative: `O=Cc1ccc(OCc2ccn(-c3cccc(F)c3)n2)cc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 82.8 | 1.21× |
| Heterocycle | 93.0 | 62.4 | 1.49× |
| Amide (-CONH-) | 47.7 | 66.9 | 0.71× |
| Ketone (>C=O) | 24.4 | 13.6 | 1.80× ⬆ |
| Halide (C-X) | 24.4 | 34.4 | 0.71× |
| Ether (C-O-C) | 16.3 | 36.8 | 0.44× ⬇ |
| Sulfonyl (-SO₂-) | 16.3 | 12.1 | 1.34× |
| Thioether (C-S-C) | 12.8 | 15.3 | 0.84× |

**Cluster 59 (35 molecules)** — representative: `COc1ccccc1NC(=O)CSc1ccc(-c2ccccc2OC)nn1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 82.8 | 1.21× |
| Heterocycle | 88.6 | 62.4 | 1.42× |
| Ether (C-O-C) | 57.1 | 36.8 | 1.55× ⬆ |
| Amide (-CONH-) | 51.4 | 66.9 | 0.77× |
| Halide (C-X) | 48.6 | 34.4 | 1.41× |
| Secondary Amine (>NH) | 22.9 | 25.6 | 0.89× |
| Thioether (C-S-C) | 22.9 | 15.3 | 1.50× |
| Nitrile (-C≡N) | 11.4 | 5.3 | 2.15× ⬆ |

**Cluster 95 (34 molecules)** — representative: `CCOC(=O)c1ccccc1NC(=O)/C=C/c1ccc(Br)cc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 82.8 | 1.21× |
| Amide (-CONH-) | 82.4 | 66.9 | 1.23× |
| Heterocycle | 73.5 | 62.4 | 1.18× |
| Ether (C-O-C) | 35.3 | 36.8 | 0.96× |
| Sulfonyl (-SO₂-) | 32.4 | 12.1 | 2.66× ⬆ |
| Ketone (>C=O) | 29.4 | 13.6 | 2.16× ⬆ |
| Halide (C-X) | 29.4 | 34.4 | 0.86× |
| Tertiary Amine (>N<) | 17.6 | 18.1 | 0.97× |

**Cluster 50 (30 molecules)** — representative: `CC(=O)Nc1cccc(NC(=O)C(=O)NCC[NH+]2CCCC[C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 80.0 | 66.9 | 1.20× |
| Phenyl (aromatic ring) | 70.0 | 82.8 | 0.85× |
| Secondary Amine (>NH) | 33.3 | 25.6 | 1.30× |
| Halide (C-X) | 33.3 | 34.4 | 0.97× |
| Heterocycle | 33.3 | 62.4 | 0.53× |
| Ether (C-O-C) | 26.7 | 36.8 | 0.73× |
| Thioether (C-S-C) | 26.7 | 15.3 | 1.74× ⬆ |
| Nitro (-NO₂) | 23.3 | 7.5 | 3.12× ⬆ |

### Stratum 3 ([0.694, 0.814)) — Cluster FG Signatures

**Cluster 90 (146 molecules)** — representative: `C[C@H](NC(=O)Nc1ccncc1)c1nc(-c2ccc(Cl)cc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.3 | 77.6 | 1.28× |
| Heterocycle | 88.4 | 53.3 | 1.66× ⬆ |
| Amide (-CONH-) | 67.1 | 66.0 | 1.02× |
| Halide (C-X) | 38.4 | 33.9 | 1.13× |
| Ether (C-O-C) | 23.3 | 39.7 | 0.59× |
| Secondary Amine (>NH) | 18.5 | 28.7 | 0.64× |
| Ketone (>C=O) | 13.7 | 10.8 | 1.27× |
| Primary Amine (-NH₂) | 9.6 | 7.2 | 1.33× |

**Cluster 9 (116 molecules)** — representative: `CC1(C)CCC[C@@H](C[NH+](CCO)C2CCCCC2)C1=O`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 45.7 | 66.0 | 0.69× |
| Ether (C-O-C) | 44.8 | 39.7 | 1.13× |
| Secondary Amine (>NH) | 33.6 | 28.7 | 1.17× |
| Tertiary Amine (>N<) | 28.4 | 21.5 | 1.32× |
| Hydroxyl (-OH) | 19.8 | 12.7 | 1.56× ⬆ |
| Primary Amine (-NH₂) | 16.4 | 7.2 | 2.26× ⬆ |
| Thioether (C-S-C) | 10.3 | 9.4 | 1.11× |
| Nitrile (-C≡N) | 8.6 | 4.3 | 2.00× ⬆ |

**Cluster 40 (50 molecules)** — representative: `COc1ccc(CCCC(=O)Nc2cccc(S(N)(=O)=O)c2)cc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 77.6 | 1.29× |
| Amide (-CONH-) | 84.0 | 66.0 | 1.27× |
| Halide (C-X) | 66.0 | 33.9 | 1.95× ⬆ |
| Sulfonyl (-SO₂-) | 60.0 | 12.7 | 4.72× ⬆ |
| Heterocycle | 44.0 | 53.3 | 0.83× |
| Ether (C-O-C) | 34.0 | 39.7 | 0.86× |
| Tertiary Amine (>N<) | 30.0 | 21.5 | 1.39× |
| Secondary Amine (>NH) | 20.0 | 28.7 | 0.70× |

**Cluster 50 (41 molecules)** — representative: `CC(C)N(C(=O)CS(=O)(=O)Cc1ccc(Cl)c(Cl)c1)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 77.6 | 1.29× |
| Amide (-CONH-) | 70.7 | 66.0 | 1.07× |
| Sulfonyl (-SO₂-) | 51.2 | 12.7 | 4.03× ⬆ |
| Halide (C-X) | 48.8 | 33.9 | 1.44× |
| Heterocycle | 43.9 | 53.3 | 0.82× |
| Secondary Amine (>NH) | 36.6 | 28.7 | 1.28× |
| Ether (C-O-C) | 24.4 | 39.7 | 0.61× |
| Tertiary Amine (>N<) | 17.1 | 21.5 | 0.79× |

**Cluster 54 (37 molecules)** — representative: `COc1ccccc1[C@H]1CCCN1C(=O)Nc1ccc(C(=O)NC`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 91.9 | 77.6 | 1.18× |
| Amide (-CONH-) | 81.1 | 66.0 | 1.23× |
| Heterocycle | 70.3 | 53.3 | 1.32× |
| Ether (C-O-C) | 59.5 | 39.7 | 1.50× |
| Secondary Amine (>NH) | 32.4 | 28.7 | 1.13× |
| Halide (C-X) | 27.0 | 33.9 | 0.80× |
| Hydroxyl (-OH) | 18.9 | 12.7 | 1.49× |
| Tertiary Amine (>N<) | 16.2 | 21.5 | 0.75× |

### Stratum 4 ([0.814, 1.0]) — Cluster FG Signatures

**Cluster 9 (110 molecules)** — representative: `Fc1ccc(C[NH2+]C[C@@H]([C@H]2CCOC2)N2CCOC`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Ether (C-O-C) | 58.2 | 40.4 | 1.44× |
| Heterocycle | 45.5 | 55.1 | 0.82× |
| Tertiary Amine (>N<) | 44.5 | 23.8 | 1.87× ⬆ |
| Phenyl (aromatic ring) | 39.1 | 83.3 | 0.47× ⬇ |
| Secondary Amine (>NH) | 36.4 | 29.7 | 1.22× |
| Amide (-CONH-) | 34.5 | 70.1 | 0.49× ⬇ |
| Halide (C-X) | 26.4 | 37.2 | 0.71× |
| Hydroxyl (-OH) | 17.3 | 12.0 | 1.44× |

**Cluster 90 (94 molecules)** — representative: `CC(C)NS(=O)(=O)c1ccc(C(=O)Nc2ccc(Cl)cc2)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.8 | 83.3 | 1.16× |
| Amide (-CONH-) | 76.6 | 70.1 | 1.09× |
| Heterocycle | 53.2 | 55.1 | 0.96× |
| Sulfonyl (-SO₂-) | 51.1 | 12.2 | 4.18× ⬆ |
| Halide (C-X) | 44.7 | 37.2 | 1.20× |
| Secondary Amine (>NH) | 31.9 | 29.7 | 1.07× |
| Ether (C-O-C) | 14.9 | 40.4 | 0.37× ⬇ |
| Ketone (>C=O) | 12.8 | 4.2 | 3.03× ⬆ |

**Cluster 0 (55 molecules)** — representative: `O=C(c1ccc(O)cc1)N1CCN(CCc2ccccc2Cl)CC1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.4 | 83.3 | 1.16× |
| Amide (-CONH-) | 69.1 | 70.1 | 0.99× |
| Heterocycle | 65.5 | 55.1 | 1.19× |
| Ether (C-O-C) | 54.5 | 40.4 | 1.35× |
| Tertiary Amine (>N<) | 43.6 | 23.8 | 1.83× ⬆ |
| Halide (C-X) | 29.1 | 37.2 | 0.78× |
| Secondary Amine (>NH) | 23.6 | 29.7 | 0.79× |
| Hydroxyl (-OH) | 14.5 | 12.0 | 1.21× |

**Cluster 99 (39 molecules)** — representative: `NC(=O)N1CCCN(C(=O)Nc2cc(Cl)ccc2Cl)CC1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 87.2 | 70.1 | 1.24× |
| Phenyl (aromatic ring) | 74.4 | 83.3 | 0.89× |
| Ether (C-O-C) | 46.2 | 40.4 | 1.14× |
| Sulfonyl (-SO₂-) | 35.9 | 12.2 | 2.94× ⬆ |
| Tertiary Amine (>N<) | 33.3 | 23.8 | 1.40× |
| Heterocycle | 33.3 | 55.1 | 0.60× |
| Halide (C-X) | 25.6 | 37.2 | 0.69× |
| Secondary Amine (>NH) | 20.5 | 29.7 | 0.69× |

**Cluster 40 (36 molecules)** — representative: `CCNc1ccc2c(OC)ccc(F)c2n1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 91.7 | 83.3 | 1.10× |
| Heterocycle | 61.1 | 55.1 | 1.11× |
| Halide (C-X) | 55.6 | 37.2 | 1.49× |
| Amide (-CONH-) | 44.4 | 70.1 | 0.63× |
| Secondary Amine (>NH) | 38.9 | 29.7 | 1.31× |
| Ether (C-O-C) | 36.1 | 40.4 | 0.89× |
| Hydroxyl (-OH) | 27.8 | 12.0 | 2.31× ⬆ |
| Primary Amine (-NH₂) | 13.9 | 7.0 | 1.98× ⬆ |

## 9. Evaluation Summary

| Metric | Value |
|---|---|
| Total active clusters | 447 / 500 neurons |
| Cluster size (mean) | 11.2 |
| Cluster size (range) | 1 – 146 |
| Average quantization error | 0.032134 |
| Mean intra-cluster distance | 0.020789 |
| Functional group types detected | 21 / 22 |
| Strongest FG-property |r| | 0.1366 (Hydroxyl (-OH)) |

## 10. Performance

| Phase | Time |
|---|---|
| Data loading | 0.11s |
| Graph parsing + FG detection | 0.04s |
| VGAE encoding | 14.97s |
| Importance analysis | 0.01s |
| SOM clustering + FG analysis | 1.37s |
| **Total** | **16.55s** |

**Throughput**: 302 molecules/second

## 11. Methodology Comparison

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

## 12. Output Files

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
