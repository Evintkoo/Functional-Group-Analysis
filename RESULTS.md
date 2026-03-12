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

![Property Distributions](results/figures/property_distributions_combined.svg)

*Figure 1: Distribution of QED, logP, and SAS across the full dataset. Red vertical lines indicate means.*

| | | |
|---|---|---|
| ![QED](results/figures/qed_distribution.svg) | ![logP](results/figures/logp_distribution.svg) | ![SAS](results/figures/sas_distribution.svg) |

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

![Molecular Complexity](results/figures/molecule_complexity.svg)

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

![FG Prevalence](results/figures/fg_prevalence.svg)

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
| Grid selection | **Autotune** (multi-candidate evaluation) |
| Training epochs | 128 (full), 30–50 (autotune eval) |
| Initial learning rate | 0.5 |
| Distance metric | Euclidean |
| Neighborhood | Gaussian |
| Scoring | 0.4×QE + 0.3×TE + 0.3×ActiveRatio |

### Autotune Results (per Stratum)

**Stratum 0** — Best grid: **25×25**

| Grid | Neurons | Active | QE | TE | Score |
|---|---|---|---|---|---|
| 25×25 | 625 | 591 | 0.1772 | 0.1958 | 1.0000 |
| 30×30 | 900 | 807 | 0.1732 | 0.2092 | 1.0000 |
| 17×17 | 289 | 288 | 0.1852 | 0.1996 | 0.6220 |

**Stratum 1** — Best grid: **30×30**

| Grid | Neurons | Active | QE | TE | Score |
|---|---|---|---|---|---|
| 30×30 | 900 | 818 | 0.1808 | 0.2022 | 1.0000 |
| 27×27 | 729 | 681 | 0.1857 | 0.1852 | 0.8498 |

**Stratum 2** — Best grid: **30×30**

| Grid | Neurons | Active | QE | TE | Score |
|---|---|---|---|---|---|
| 30×30 | 900 | 802 | 0.1971 | 0.1942 | 1.0000 |

**Stratum 3** — Best grid: **30×30**

| Grid | Neurons | Active | QE | TE | Score |
|---|---|---|---|---|---|
| 30×30 | 900 | 760 | 0.2003 | 0.1766 | 1.0000 |

**Stratum 4** — Best grid: **30×30**

| Grid | Neurons | Active | QE | TE | Score |
|---|---|---|---|---|---|
| 30×30 | 900 | 787 | 0.1507 | 0.0706 | 1.0000 |

## 5. VGAE Encoding Results

| Metric | Value |
|---|---|
| Mean reconstruction loss | 0.050459 |
| Mean pairwise embedding distance | 1.372703 |
| Embedding std (mean across dims) | 0.172888 |
| Embedding std range | [0.0212, 0.9755] |

### Latent Dimension Statistics

| Dim | Mean | Std | Min | Max |
|---|---|---|---|---|
| 0 | 0.1060 | 0.9755 | -3.3241 | 2.9081 |
| 1 | 0.1234 | 0.2987 | -1.1649 | 1.3984 |
| 2 | 0.0005 | 0.0212 | -0.0795 | 0.0646 |
| 3 | -0.0559 | 0.1483 | -0.6948 | 0.5090 |
| 4 | -0.0067 | 0.0718 | -0.3646 | 0.2523 |
| 5 | -0.0070 | 0.1199 | -0.3652 | 0.4375 |
| 6 | 0.1750 | 0.3883 | -1.4129 | 1.8295 |
| 7 | 0.0041 | 0.0443 | -0.1196 | 0.1994 |
| 8 | -0.0444 | 0.1606 | -0.7276 | 0.6199 |
| 9 | 0.0255 | 0.0610 | -0.2744 | 0.2653 |
| 10 | -0.0123 | 0.0863 | -0.3669 | 0.2586 |
| 11 | 0.0575 | 0.1968 | -0.6325 | 0.8779 |
| 12 | 0.0096 | 0.0362 | -0.1206 | 0.1464 |
| 13 | -0.0221 | 0.0840 | -0.3732 | 0.3675 |
| 14 | 0.0091 | 0.0358 | -0.1448 | 0.1374 |
| 15 | -0.0138 | 0.0376 | -0.1952 | 0.1365 |

![Reconstruction Loss](results/figures/reconstruction_loss_dist.svg)

*Figure 5: Distribution of VGAE reconstruction losses across all molecules.*

![Embedding Variance](results/figures/embedding_dim_variance.svg)

*Figure 6: Variance of each latent dimension — higher variance indicates more discriminative dimensions.*

## 6. Feature Importance Analysis

### 6.1 Latent Dimension ↔ Property Correlations

Pearson correlation (r) between each latent dimension and molecular properties.
Dimensions sorted by |r(QED)|.

| Dim | Variance | r(QED) | r(logP) | r(SAS) |
|---|---|---|---|---|
| 5 | 0.014373 | +0.3215 | -0.4184 | +0.6307 |
| 7 | 0.001959 | +0.3145 | -0.3383 | +0.6274 |
| 2 | 0.000449 | -0.2974 | +0.4238 | -0.6131 |
| 0 | 0.951615 | +0.2907 | -0.4610 | +0.5782 |
| 12 | 0.001312 | +0.2724 | -0.4322 | +0.5216 |
| 10 | 0.007442 | +0.2507 | -0.4608 | +0.4940 |
| 15 | 0.001414 | -0.2287 | +0.0958 | -0.4499 |
| 4 | 0.005155 | -0.2180 | +0.1091 | -0.4249 |
| 9 | 0.003717 | -0.1971 | +0.0908 | -0.4124 |
| 14 | 0.001282 | +0.1894 | -0.4280 | +0.3819 |
| 13 | 0.007058 | +0.1642 | -0.0020 | +0.3144 |
| 1 | 0.089221 | -0.1171 | -0.0698 | -0.2299 |
| 8 | 0.025803 | +0.0976 | +0.1004 | +0.1912 |
| 6 | 0.150819 | -0.0913 | -0.1066 | -0.1816 |
| 3 | 0.022001 | +0.0642 | +0.1425 | +0.1316 |
| 11 | 0.038722 | -0.0212 | -0.2029 | -0.0421 |

### 6.2 Functional Group ↔ Latent Space Encoding

Which latent dimensions best encode each functional group's presence.

| Functional Group | Prevalence (%) | Best Dim | |r| |
|---|---|---|---|
| Phenyl (aromatic ring) | 83.0 | 0 | 0.5431 |
| Heterocycle | 58.0 | 10 | 0.4763 |
| Tertiary Amine (>N<) | 21.6 | 4 | 0.2645 |
| Halide (C-X) | 35.1 | 7 | 0.1819 |
| Amide (-CONH-) | 68.0 | 15 | 0.1812 |
| Sulfonyl (-SO₂-) | 10.9 | 15 | 0.1732 |
| Ketone (>C=O) | 10.5 | 12 | 0.1580 |
| Hydroxyl (-OH) | 11.1 | 12 | 0.1468 |
| Secondary Amine (>NH) | 27.4 | 12 | 0.1455 |
| Ether (C-O-C) | 37.3 | 7 | 0.1381 |
| Nitro (-NO₂) | 4.3 | 7 | 0.1330 |
| Carboxyl (-COOH) | 3.8 | 11 | 0.1288 |
| Ester (-COO-) | 7.3 | 15 | 0.1281 |
| Imine (C=N) | 2.7 | 7 | 0.1157 |
| Nitrile (-C≡N) | 5.2 | 9 | 0.1083 |
| Primary Amine (-NH₂) | 7.1 | 12 | 0.0840 |
| Thioether (C-S-C) | 11.0 | 2 | 0.0314 |

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

![Dim-Property Heatmap](results/figures/dim_property_heatmap.svg)

*Figure 7: Heatmap of Pearson correlations between latent dimensions and molecular properties. Blue = negative, red = positive.*

![FG-Property Correlations](results/figures/fg_property_correlations.svg)

*Figure 8: Point-biserial correlations between functional group presence and drug-likeness properties.*

## 7. Stratified Clustering Results

### Per-Stratum Overview

| Stratum | QED Range | Molecules | Active Clusters | QE | U-Matrix Mean | U-Matrix Max |
|---|---|---|---|---|---|---|
| 0 | [0, 0.399) | 6830 | 612 | 1.096530 | 0.0551 | 0.1110 |
| 1 | [0.399, 0.520) | 17622 | 896 | 1.048350 | 0.0485 | 0.1106 |
| 2 | [0.520, 0.694) | 60427 | 900 | 0.969829 | 0.0550 | 0.1413 |
| 3 | [0.694, 0.814) | 83673 | 900 | 0.935341 | 0.0556 | 0.1093 |
| 4 | [0.814, 1.0] | 80903 | 900 | 0.739544 | 0.0434 | 0.0738 |

**Total clustered**: 249455 molecules | **Avg QE**: 0.957919

![Latent Space UMAP](results/figures/latent_space_umap.svg)

*Figure 9: UMAP projection of 16-dimensional VGAE embeddings colored by QED stratum.*

![Stratum Properties](results/figures/stratum_property_comparison.svg)

*Figure 10: Mean ± std of molecular properties across QED strata.*

![U-Matrix](results/figures/umatrix_heatmaps.svg)

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
| 24 | 454 | 0.345±0.048 | 4.46 | 2.45 | 0.3120 | Ph | HetCyc(1.4×) | `COc1ccc2nc(N(Cc3cccnc3)C(=O)c3` |
| 600 | 302 | 0.328±0.052 | -0.19 | 4.28 | 0.7424 | NH2 | NH2(3.8×), OH(3.7×), COOH(2.5×) | `CCC(CC)(NC(=O)N1CCCCCC1)/C(N)=` |
| 624 | 91 | 0.347±0.052 | 4.41 | 3.02 | 0.2004 | Ph | N<(3.3×), C-O-C(1.6×), C-S-C(1.4×) | `O=C1/C(=C\c2cn(-c3ccccc3)nc2-c` |
| 324 | 90 | 0.342±0.047 | 4.53 | 2.75 | 0.2175 | Ph | N<(1.7×), HetCyc(1.4×), C-O-C(1.3×) | `NC(=O)c1c(NC(=O)c2ccc(-c3ccccc` |
| 12 | 79 | 0.324±0.055 | 4.11 | 2.43 | 0.1318 | Ph | COOH(1.7×), NO2(1.5×), C=O(1.3×) | `O=C(Cn1c(CCNC(=O)c2cccc(Br)c2)` |
| 0 | 78 | 0.348±0.046 | 2.71 | 2.64 | 0.1900 | Ph | SO2(2.8×), COOH(2.6×), CN(2.0×) | `Cc1ncc(/C=C/C(=O)OCCCn2nc3cccc` |
| 11 | 72 | 0.334±0.059 | 3.97 | 2.36 | 0.1169 | Ph | NO2(1.5×), C=N(1.3×), HetCyc(1.2×) | `CCOC(=O)c1ccc(-c2ccc(/C=N/n3c(` |
| 275 | 68 | 0.320±0.060 | 1.82 | 2.69 | 0.2244 | Ph | C=N(2.1×), NO2(1.8×), NH2(1.7×) | `CCO/C([O-])=C(\C#N)C(=O)c1cccc` |
| 18 | 63 | 0.324±0.063 | 4.02 | 2.42 | 0.1003 | Ph | CN(1.7×), C=O(1.5×), HetCyc(1.3×) | `CC(=O)Oc1ccc(-c2ccc(C(=O)Nc3cc` |
| 20 | 62 | 0.326±0.056 | 3.83 | 2.54 | 0.1723 | Ph | C=O(1.4×), HetCyc(1.4×) | `O=C(Nc1ccc(C(=O)Nc2ccccc2[N-]S` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 221 | 247 | 0.007301 |
| 40 | 65 | 0.007413 |
| 90 | 115 | 0.008612 |
| 496 | 521 | 0.008838 |
| 298 | 323 | 0.009612 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 24 | 600 | 4.185544 |
| 24 | 575 | 4.167718 |
| 23 | 600 | 3.874031 |
| 22 | 600 | 3.857116 |
| 49 | 600 | 3.850152 |

Inter-cluster distance: mean=0.885972, min=0.007301, max=4.185544, 186966 pairs

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
| 29 | 982 | 0.471±0.036 | 3.97 | 2.39 | 0.3233 | Ph | HetCyc(1.4×) | `O=C(NCCc1ccc(-c2ccccc2)cc1)c1c` |
| 870 | 756 | 0.475±0.034 | -0.16 | 4.49 | 0.7341 | CONH | COOH(4.4×), OH(2.9×), NH2(2.2×) | `COC[C@@]1(C)C[NH+]=C(N)N1C` |
| 14 | 194 | 0.469±0.031 | 3.49 | 2.41 | 0.1171 | Ph | SO2(1.9×), C=N(1.6×), HetCyc(1.3×) | `O=C(CCNC(=O)c1c[nH]c2nc(-c3ccc` |
| 479 | 191 | 0.468±0.036 | 3.94 | 2.76 | 0.2222 | Ph | N<(2.7×), HetCyc(1.4×) | `Cn1cc(C(=O)N2CCN(c3ccc(F)cc3)C` |
| 0 | 163 | 0.470±0.034 | 2.76 | 2.52 | 0.1591 | Ph | SO2(2.1×), NO2(2.0×), C=N(1.8×) | `C=CCOC(=O)/C(=C\c1ccco1)NC(=O)` |
| 13 | 157 | 0.469±0.036 | 3.39 | 2.42 | 0.1380 | Ph | SO2(1.9×), C=N(1.7×), HetCyc(1.3×) | `Cc1nc2cc(C(=O)OCC(=O)Nc3ccc(F)` |
| 899 | 147 | 0.471±0.033 | 3.95 | 3.28 | 0.1921 | Ph | N<(2.6×), NH(1.4×), OH(1.3×) | `O=C(CSc1nc(=O)n(Cc2ccncc2)c2c1` |
| 15 | 143 | 0.470±0.035 | 3.88 | 2.41 | 0.1205 | Ph | SO2(2.0×), CN(1.3×), C-X(1.2×) | `O=C(Cc1csc(-c2cccs2)n1)NCCn1cc` |
| 885 | 139 | 0.469±0.033 | 2.42 | 3.46 | 0.2482 | Ph | N<(2.8×), NH2(1.4×), NH(1.3×) | `Cc1ccc(N2C(=O)[C@@H]([C@H]3NCC` |
| 21 | 119 | 0.468±0.039 | 3.61 | 2.36 | 0.1520 | Ph | SO2(1.6×), C=N(1.5×), HetCyc(1.4×) | `Cc1sc2ncn3nc(Cn4cc([N+](=O)[O-` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 85 | 86 | 0.006379 |
| 266 | 268 | 0.006628 |
| 42 | 72 | 0.008580 |
| 49 | 80 | 0.008980 |
| 187 | 217 | 0.009071 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 29 | 840 | 4.496827 |
| 29 | 870 | 4.350712 |
| 29 | 810 | 4.349072 |
| 59 | 840 | 4.224448 |
| 29 | 780 | 4.218035 |

Inter-cluster distance: mean=0.941964, min=0.006379, max=4.496827, 400960 pairs

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
| 0 | 3712 | 0.633±0.045 | 0.20 | 4.68 | 0.6137 | NH | NH2(2.8×), OH(2.4×), COOH(2.3×) | `CC[NH2+][C@@]1(C(=O)[O-])CCC[C` |
| 899 | 3350 | 0.594±0.052 | 3.28 | 2.39 | 0.3823 | Ph | HetCyc(1.6×) | `Cc1ccccc1C(=O)Nc1ccc2c(=O)oc(-` |
| 884 | 710 | 0.631±0.046 | 3.67 | 2.80 | 0.2284 | Ph | N<(1.7×), HetCyc(1.3×) | `Cc1ccccc1-n1nnnc1S[C@@H](C(=O)` |
| 16 | 695 | 0.621±0.043 | 1.45 | 2.88 | 0.1864 | Ph | COOH(2.4×), NO2(2.0×), NH2(1.9×) | `COc1ccnc(COC(=O)/C=C(/C)C(C)(C` |
| 449 | 564 | 0.620±0.050 | 2.88 | 2.36 | 0.1764 | Ph | C=N(2.6×), NO2(2.2×), SO2(1.8×) | `COc1ccccc1Cc1nnc(NC(=O)CN2C(=O` |
| 870 | 518 | 0.621±0.045 | 3.23 | 3.15 | 0.2371 | Ph | N<(2.6×), C-O-C(1.4×), HetCyc(1.3×) | `O=c1occ(C[NH+]2CCc3ccccc3C2)c(` |
| 450 | 500 | 0.635±0.042 | 1.97 | 3.69 | 0.2634 | Ph | N<(2.1×), NH2(1.7×), C-S-C(1.4×) | `C1=C(CC[NH2+]Cc2ccco2)CCCC1` |
| 15 | 464 | 0.619±0.044 | 1.60 | 3.09 | 0.2079 | Ph | OH(2.1×), NO2(2.0×), NH2(2.0×) | `O=[N+]([O-])c1ccc(CN(CCO)CCO)c` |
| 659 | 428 | 0.605±0.053 | 3.32 | 2.34 | 0.1370 | Ph | CN(1.8×), SO2(1.7×), C=N(1.5×) | `O=C(CNC(=O)c1ccc2[nH]c(=S)oc2c` |
| 269 | 404 | 0.623±0.047 | 2.71 | 2.46 | 0.1524 | Ph | C=N(2.9×), COO(1.9×), NO2(1.9×) | `O=C(NNc1cccc(C(F)(F)F)n1)c1cc2` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 87 | 116 | 0.006094 |
| 743 | 772 | 0.006301 |
| 57 | 58 | 0.008883 |
| 837 | 838 | 0.011145 |
| 820 | 849 | 0.012396 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 4 | 899 | 4.512288 |
| 0 | 899 | 4.342086 |
| 6 | 899 | 4.322716 |
| 1 | 899 | 4.305764 |
| 5 | 899 | 4.297295 |

Inter-cluster distance: mean=1.033282, min=0.006094, max=4.512288, 404550 pairs

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
| 870 | 5072 | 0.754±0.033 | 1.18 | 4.40 | 0.5119 | CONH | NH2(2.1×), OH(1.5×), N<(1.4×) | `C[C@H]1C[C@@H](C)CN(C(=O)C[NH+` |
| 29 | 4622 | 0.748±0.031 | 2.91 | 2.33 | 0.3161 | Ph | HetCyc(1.8×), C=O(1.2×), Ph(1.2×) | `O=C(Cc1noc(-c2ccncc2)n1)Nc1cc(` |
| 420 | 1028 | 0.775±0.030 | 1.82 | 3.75 | 0.2875 | Ph | NH2(1.9×), N<(1.6×), C-S-C(1.3×) | `C[C@H]1CC(C(=O)N2CCc3ccc(F)cc3` |
| 479 | 923 | 0.758±0.036 | 2.56 | 2.48 | 0.1407 | Ph | SO2(2.1×), CN(1.8×), C-X(1.4×) | `COC(=O)c1ccc(=O)n(CC(=O)N(c2cc` |
| 871 | 855 | 0.749±0.033 | 1.22 | 3.90 | 0.2218 | CONH | NH2(1.9×), C-S-C(1.6×), OH(1.6×) | `CCN(C)C(=O)NCCCOC1CCOCC1` |
| 14 | 796 | 0.749±0.034 | 2.75 | 3.18 | 0.2595 | Ph | N<(2.0×), HetCyc(1.6×) | `CCc1nc([C@H]2CCC[NH+]2Cc2nc([O` |
| 59 | 794 | 0.742±0.030 | 2.94 | 2.35 | 0.1692 | Ph | C=O(1.8×), HetCyc(1.7×), Ph(1.2×) | `CN(C)c1ncc(-c2nccn2Cc2ccc(S(N)` |
| 509 | 768 | 0.764±0.033 | 2.87 | 2.42 | 0.1196 | Ph | SO2(1.8×), COO(1.7×), CN(1.6×) | `COc1ccccc1CNC(=O)NCc1ccc(C(N)=` |
| 569 | 743 | 0.762±0.035 | 2.90 | 2.47 | 0.0938 | Ph | SO2(2.0×), C-X(1.6×), COO(1.4×) | `COc1ccccc1NCC(=O)NNC(=O)c1cc(C` |
| 899 | 740 | 0.766±0.032 | 1.35 | 2.75 | 0.1664 | Ph | COOH(5.5×), COO(2.2×), SO2(2.2×) | `COC(=O)[C@@H](C)[C@H](C)S(=O)(` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 836 | 837 | 0.010562 |
| 773 | 774 | 0.010668 |
| 647 | 677 | 0.012180 |
| 746 | 777 | 0.012513 |
| 843 | 844 | 0.012832 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 29 | 871 | 3.993193 |
| 29 | 870 | 3.947579 |
| 29 | 876 | 3.866070 |
| 29 | 873 | 3.852270 |
| 28 | 871 | 3.848481 |

Inter-cluster distance: mean=1.030866, min=0.010562, max=3.993193, 404550 pairs

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
| 899 | 3347 | 0.871±0.034 | 2.51 | 2.25 | 0.2647 | Ph | COOH(3.7×), SO(1.6×), SO2(1.5×) | `CCOc1ccc(-c2nc(C(=O)[O-])cc(=O` |
| 29 | 1651 | 0.851±0.026 | 2.30 | 3.00 | 0.1768 | CONH | COOH(2.2×), COO(2.1×), SO(1.8×) | `C[NH+](C/C=C/c1ccco1)CCC(F)(F)` |
| 0 | 1615 | 0.847±0.026 | 1.81 | 4.24 | 0.5733 | C-O-C | C-S-C(1.9×), N<(1.8×), C-O-C(1.4×) | `N#C[C@@H](NC(=O)N1CCCCC1)C1CCC` |
| 870 | 1313 | 0.877±0.036 | 2.37 | 3.33 | 0.2338 | Ph | NH2(1.4×), HetCyc(1.4×), N<(1.4×) | `O=C(NC1CCSCC1)C1=NN(c2ccccc2)[` |
| 180 | 1054 | 0.874±0.032 | 1.97 | 3.94 | 0.2486 | Ph | C-S-C(1.7×), NH2(1.6×), N<(1.6×) | `CC(C)c1ccnc(N2CCC3(CC2)C[C@@H]` |
| 15 | 987 | 0.855±0.028 | 2.16 | 3.50 | 0.2653 | CONH | C-O-C(1.3×), NH(1.2×), COO(1.2×) | `CCC[NH+](C)C[C@@H]1CCN(C(=O)NC` |
| 14 | 872 | 0.860±0.030 | 2.19 | 3.46 | 0.1694 | CONH | N<(1.6×), OH(1.5×), C-O-C(1.4×) | `CC(C)[C@H](C)NC(=O)C[NH+](C)C1` |
| 30 | 862 | 0.870±0.031 | 1.75 | 4.28 | 0.2403 | HetCyc | N<(2.1×), C-O-C(1.4×), NH(1.4×) | `Cc1ccc(O)c(CN2CCC[C@@H](C[NH+]` |
| 479 | 771 | 0.857±0.031 | 2.58 | 2.64 | 0.1412 | Ph | SO(2.0×), SO2(1.9×), COO(1.8×) | `CC[C@H](CNC(=O)c1cc(C#N)cn1C)O` |
| 884 | 710 | 0.883±0.037 | 2.52 | 2.68 | 0.2091 | Ph | C=O(1.4×) | `CNc1nc(C(=O)N2CC=C(c3cccc(C)c3` |

#### Inter-Cluster Distance Analysis

**Most similar cluster pairs** (smallest embedding distance):

| Cluster A | Cluster B | Distance |
|---|---|---|
| 829 | 859 | 0.005331 |
| 837 | 868 | 0.006307 |
| 56 | 57 | 0.008813 |
| 504 | 534 | 0.009622 |
| 146 | 147 | 0.009999 |

**Most distant cluster pairs**:

| Cluster A | Cluster B | Distance |
|---|---|---|
| 0 | 898 | 3.072985 |
| 0 | 899 | 3.025156 |
| 0 | 896 | 2.931030 |
| 0 | 897 | 2.901108 |
| 0 | 869 | 2.836731 |

Inter-cluster distance: mean=0.832364, min=0.005331, max=3.072985, 404550 pairs

## 8. Cluster Functional Group Characterization

Summary of functional group signatures across the largest clusters in each stratum.
Enrichment ratio shows over-representation relative to the stratum population.

### Stratum 0 ([0, 0.399)) — Cluster FG Signatures

**Cluster 24 (454 molecules)** — representative: `COc1ccc2nc(N(Cc3cccnc3)C(=O)c3cc(=O)c4cc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.8 | 92.6 | 1.08× |
| Heterocycle | 98.7 | 69.1 | 1.43× |
| Amide (-CONH-) | 50.4 | 61.6 | 0.82× |
| Halide (C-X) | 33.9 | 36.2 | 0.94× |
| Ketone (>C=O) | 32.6 | 31.6 | 1.03× |
| Thioether (C-S-C) | 32.6 | 31.4 | 1.04× |
| Ether (C-O-C) | 18.5 | 34.1 | 0.54× |
| Secondary Amine (>NH) | 14.5 | 19.5 | 0.74× |

**Cluster 600 (302 molecules)** — representative: `CCC(CC)(NC(=O)N1CCCCCC1)/C(N)=N/O`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Primary Amine (-NH₂) | 49.0 | 13.0 | 3.77× ⬆ |
| Hydroxyl (-OH) | 43.4 | 11.6 | 3.75× ⬆ |
| Amide (-CONH-) | 41.1 | 61.6 | 0.67× |
| Imine (C=N) | 39.7 | 19.2 | 2.07× ⬆ |
| Secondary Amine (>NH) | 34.4 | 19.5 | 1.76× ⬆ |
| Ether (C-O-C) | 30.1 | 34.1 | 0.88× |
| Tertiary Amine (>N<) | 21.9 | 11.6 | 1.88× ⬆ |
| Ester (-COO-) | 16.6 | 18.5 | 0.89× |

**Cluster 624 (91 molecules)** — representative: `O=C1/C(=C\c2cn(-c3ccccc3)nc2-c2cccs2)SC(`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.9 | 92.6 | 1.07× |
| Heterocycle | 86.8 | 69.1 | 1.26× |
| Ether (C-O-C) | 56.0 | 34.1 | 1.64× ⬆ |
| Amide (-CONH-) | 54.9 | 61.6 | 0.89× |
| Thioether (C-S-C) | 44.0 | 31.4 | 1.40× |
| Tertiary Amine (>N<) | 38.5 | 11.6 | 3.30× ⬆ |
| Halide (C-X) | 30.8 | 36.2 | 0.85× |
| Ketone (>C=O) | 29.7 | 31.6 | 0.94× |

**Cluster 324 (90 molecules)** — representative: `NC(=O)c1c(NC(=O)c2ccc(-c3ccccc3)cc2)sc2c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 92.6 | 1.08× |
| Heterocycle | 93.3 | 69.1 | 1.35× |
| Amide (-CONH-) | 70.0 | 61.6 | 1.14× |
| Ether (C-O-C) | 45.6 | 34.1 | 1.33× |
| Thioether (C-S-C) | 38.9 | 31.4 | 1.24× |
| Ketone (>C=O) | 31.1 | 31.6 | 0.99× |
| Halide (C-X) | 26.7 | 36.2 | 0.74× |
| Tertiary Amine (>N<) | 20.0 | 11.6 | 1.72× ⬆ |

**Cluster 12 (79 molecules)** — representative: `O=C(Cn1c(CCNC(=O)c2cccc(Br)c2)nc2ccccc21`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 92.6 | 1.08× |
| Heterocycle | 83.5 | 69.1 | 1.21× |
| Amide (-CONH-) | 49.4 | 61.6 | 0.80× |
| Nitro (-NO₂) | 43.0 | 29.6 | 1.45× |
| Halide (C-X) | 43.0 | 36.2 | 1.19× |
| Ketone (>C=O) | 41.8 | 31.6 | 1.32× |
| Thioether (C-S-C) | 24.1 | 31.4 | 0.76× |
| Imine (C=N) | 20.3 | 19.2 | 1.06× |

### Stratum 1 ([0.399, 0.520)) — Cluster FG Signatures

**Cluster 29 (982 molecules)** — representative: `O=C(NCCc1ccc(-c2ccccc2)cc1)c1cnc2cc(-c3c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.9 | 91.5 | 1.09× |
| Heterocycle | 98.6 | 68.5 | 1.44× |
| Amide (-CONH-) | 47.1 | 65.7 | 0.72× |
| Halide (C-X) | 34.6 | 36.5 | 0.95× |
| Ketone (>C=O) | 22.6 | 24.0 | 0.94× |
| Thioether (C-S-C) | 15.8 | 20.5 | 0.77× |
| Ether (C-O-C) | 14.9 | 36.0 | 0.41× ⬇ |
| Secondary Amine (>NH) | 11.1 | 21.2 | 0.52× |

**Cluster 870 (756 molecules)** — representative: `COC[C@@]1(C)C[NH+]=C(N)N1C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 41.3 | 65.7 | 0.63× |
| Secondary Amine (>NH) | 38.2 | 21.2 | 1.80× ⬆ |
| Ether (C-O-C) | 28.4 | 36.0 | 0.79× |
| Hydroxyl (-OH) | 23.7 | 8.3 | 2.85× ⬆ |
| Ester (-COO-) | 18.7 | 13.9 | 1.34× |
| Tertiary Amine (>N<) | 18.1 | 15.8 | 1.14× |
| Primary Amine (-NH₂) | 17.6 | 8.0 | 2.21× ⬆ |
| Carboxyl (-COOH) | 12.0 | 2.8 | 4.36× ⬆ |

**Cluster 14 (194 molecules)** — representative: `O=C(CCNC(=O)c1c[nH]c2nc(-c3ccccc3)ccc12)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.0 | 91.5 | 1.08× |
| Heterocycle | 89.7 | 68.5 | 1.31× |
| Amide (-CONH-) | 72.2 | 65.7 | 1.10× |
| Halide (C-X) | 40.2 | 36.5 | 1.10× |
| Ether (C-O-C) | 27.3 | 36.0 | 0.76× |
| Ketone (>C=O) | 25.8 | 24.0 | 1.07× |
| Secondary Amine (>NH) | 17.0 | 21.2 | 0.80× |
| Imine (C=N) | 16.5 | 10.3 | 1.61× ⬆ |

**Cluster 479 (191 molecules)** — representative: `Cn1cc(C(=O)N2CCN(c3ccc(F)cc3)CC2)c2nn(-c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 100.0 | 91.5 | 1.09× |
| Heterocycle | 96.9 | 68.5 | 1.41× |
| Amide (-CONH-) | 68.6 | 65.7 | 1.04× |
| Tertiary Amine (>N<) | 42.4 | 15.8 | 2.68× ⬆ |
| Ether (C-O-C) | 34.0 | 36.0 | 0.95× |
| Halide (C-X) | 28.8 | 36.5 | 0.79× |
| Ketone (>C=O) | 20.4 | 24.0 | 0.85× |
| Secondary Amine (>NH) | 13.6 | 21.2 | 0.64× |

**Cluster 0 (163 molecules)** — representative: `C=CCOC(=O)/C(=C\c1ccco1)NC(=O)c1ccc(Br)c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 95.7 | 91.5 | 1.05× |
| Heterocycle | 78.5 | 68.5 | 1.15× |
| Amide (-CONH-) | 68.1 | 65.7 | 1.04× |
| Nitro (-NO₂) | 34.4 | 17.5 | 1.96× ⬆ |
| Halide (C-X) | 27.0 | 36.5 | 0.74× |
| Ketone (>C=O) | 25.2 | 24.0 | 1.05× |
| Imine (C=N) | 18.4 | 10.3 | 1.79× ⬆ |
| Ether (C-O-C) | 17.8 | 36.0 | 0.49× ⬇ |

### Stratum 2 ([0.520, 0.694)) — Cluster FG Signatures

**Cluster 0 (3712 molecules)** — representative: `CC[NH2+][C@@]1(C(=O)[O-])CCC[C@@H]1CC[NH`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Secondary Amine (>NH) | 46.6 | 25.3 | 1.84× ⬆ |
| Amide (-CONH-) | 39.8 | 68.0 | 0.59× |
| Ether (C-O-C) | 33.0 | 35.0 | 0.94× |
| Hydroxyl (-OH) | 24.3 | 10.2 | 2.39× ⬆ |
| Tertiary Amine (>N<) | 23.9 | 20.0 | 1.20× |
| Primary Amine (-NH₂) | 19.1 | 6.9 | 2.79× ⬆ |
| Carboxyl (-COOH) | 9.9 | 4.3 | 2.29× ⬆ |
| Ester (-COO-) | 6.7 | 10.0 | 0.67× |

**Cluster 899 (3350 molecules)** — representative: `Cc1ccccc1C(=O)Nc1ccc2c(=O)oc(-c3ccccc3C)`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.6 | 84.5 | 1.17× |
| Heterocycle | 96.4 | 61.7 | 1.56× ⬆ |
| Amide (-CONH-) | 47.9 | 68.0 | 0.70× |
| Halide (C-X) | 32.6 | 34.9 | 0.93× |
| Ketone (>C=O) | 16.1 | 14.4 | 1.12× |
| Secondary Amine (>NH) | 15.1 | 25.3 | 0.59× |
| Ether (C-O-C) | 14.1 | 35.0 | 0.40× ⬇ |
| Thioether (C-S-C) | 7.8 | 13.2 | 0.59× |

**Cluster 884 (710 molecules)** — representative: `Cc1ccccc1-n1nnnc1S[C@@H](C(=O)N1CCCC1)c1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 99.6 | 84.5 | 1.18× |
| Heterocycle | 83.2 | 61.7 | 1.35× |
| Amide (-CONH-) | 61.0 | 68.0 | 0.90× |
| Ether (C-O-C) | 38.2 | 35.0 | 1.09× |
| Halide (C-X) | 38.2 | 34.9 | 1.09× |
| Tertiary Amine (>N<) | 33.2 | 20.0 | 1.66× ⬆ |
| Secondary Amine (>NH) | 22.7 | 25.3 | 0.89× |
| Ketone (>C=O) | 15.9 | 14.4 | 1.10× |

**Cluster 16 (695 molecules)** — representative: `COc1ccnc(COC(=O)/C=C(/C)C(C)(C)C)c1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 80.4 | 84.5 | 0.95× |
| Amide (-CONH-) | 68.9 | 68.0 | 1.01× |
| Halide (C-X) | 41.4 | 34.9 | 1.19× |
| Heterocycle | 30.9 | 61.7 | 0.50× |
| Secondary Amine (>NH) | 30.1 | 25.3 | 1.19× |
| Ether (C-O-C) | 26.3 | 35.0 | 0.75× |
| Ester (-COO-) | 18.6 | 10.0 | 1.85× ⬆ |
| Nitro (-NO₂) | 18.4 | 9.2 | 2.01× ⬆ |

**Cluster 449 (564 molecules)** — representative: `COc1ccccc1Cc1nnc(NC(=O)CN2C(=O)c3ccccc3C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.1 | 84.5 | 1.14× |
| Amide (-CONH-) | 69.9 | 68.0 | 1.03× |
| Heterocycle | 65.8 | 61.7 | 1.07× |
| Halide (C-X) | 46.3 | 34.9 | 1.33× |
| Ether (C-O-C) | 21.3 | 35.0 | 0.61× |
| Sulfonyl (-SO₂-) | 20.7 | 11.4 | 1.82× ⬆ |
| Nitro (-NO₂) | 19.9 | 9.2 | 2.16× ⬆ |
| Ketone (>C=O) | 19.3 | 14.4 | 1.34× |

### Stratum 3 ([0.694, 0.814)) — Cluster FG Signatures

**Cluster 870 (5072 molecules)** — representative: `C[C@H]1C[C@@H](C)CN(C(=O)C[NH+](C)CC2CC[`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 45.6 | 68.4 | 0.67× |
| Ether (C-O-C) | 44.2 | 36.9 | 1.20× |
| Secondary Amine (>NH) | 39.6 | 28.2 | 1.41× |
| Tertiary Amine (>N<) | 31.3 | 22.2 | 1.41× |
| Hydroxyl (-OH) | 17.2 | 11.8 | 1.45× |
| Primary Amine (-NH₂) | 13.7 | 6.6 | 2.07× ⬆ |
| Thioether (C-S-C) | 11.5 | 9.1 | 1.27× |
| Ester (-COO-) | 6.5 | 7.0 | 0.94× |

**Cluster 29 (4622 molecules)** — representative: `O=C(Cc1noc(-c2ccncc2)n1)Nc1cc(Cl)cc(Cl)c`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 98.2 | 79.3 | 1.24× |
| Heterocycle | 95.9 | 54.2 | 1.77× ⬆ |
| Amide (-CONH-) | 64.6 | 68.4 | 0.94× |
| Halide (C-X) | 38.5 | 33.0 | 1.17× |
| Ether (C-O-C) | 18.8 | 36.9 | 0.51× |
| Secondary Amine (>NH) | 16.8 | 28.2 | 0.60× |
| Ketone (>C=O) | 11.2 | 9.0 | 1.24× |
| Primary Amine (-NH₂) | 8.2 | 6.6 | 1.23× |

**Cluster 420 (1028 molecules)** — representative: `C[C@H]1CC(C(=O)N2CCc3ccc(F)cc3C2)C[C@H](`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 69.4 | 79.3 | 0.87× |
| Amide (-CONH-) | 47.2 | 68.4 | 0.69× |
| Heterocycle | 41.1 | 54.2 | 0.76× |
| Ether (C-O-C) | 40.6 | 36.9 | 1.10× |
| Tertiary Amine (>N<) | 35.4 | 22.2 | 1.60× ⬆ |
| Halide (C-X) | 27.0 | 33.0 | 0.82× |
| Secondary Amine (>NH) | 25.2 | 28.2 | 0.89× |
| Hydroxyl (-OH) | 12.5 | 11.8 | 1.05× |

**Cluster 479 (923 molecules)** — representative: `COC(=O)c1ccc(=O)n(CC(=O)N(c2ccccc2)C(C)C`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 96.5 | 79.3 | 1.22× |
| Amide (-CONH-) | 77.0 | 68.4 | 1.13× |
| Heterocycle | 65.5 | 54.2 | 1.21× |
| Halide (C-X) | 47.7 | 33.0 | 1.44× |
| Ether (C-O-C) | 27.3 | 36.9 | 0.74× |
| Secondary Amine (>NH) | 26.2 | 28.2 | 0.93× |
| Sulfonyl (-SO₂-) | 24.1 | 11.3 | 2.13× ⬆ |
| Ketone (>C=O) | 11.8 | 9.0 | 1.31× |

**Cluster 871 (855 molecules)** — representative: `CCN(C)C(=O)NCCCOC1CCOCC1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 73.3 | 68.4 | 1.07× |
| Ether (C-O-C) | 37.0 | 36.9 | 1.00× |
| Secondary Amine (>NH) | 35.1 | 28.2 | 1.24× |
| Tertiary Amine (>N<) | 22.9 | 22.2 | 1.03× |
| Hydroxyl (-OH) | 18.8 | 11.8 | 1.59× ⬆ |
| Thioether (C-S-C) | 14.9 | 9.1 | 1.64× ⬆ |
| Sulfonyl (-SO₂-) | 12.9 | 11.3 | 1.14× |
| Primary Amine (-NH₂) | 12.5 | 6.6 | 1.89× ⬆ |

### Stratum 4 ([0.814, 1.0]) — Cluster FG Signatures

**Cluster 899 (3347 molecules)** — representative: `CCOc1ccc(-c2nc(C(=O)[O-])cc(=O)[nH]2)cc1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 95.6 | 83.0 | 1.15× |
| Amide (-CONH-) | 73.0 | 68.7 | 1.06× |
| Heterocycle | 69.2 | 56.1 | 1.23× |
| Halide (C-X) | 55.9 | 37.1 | 1.51× ⬆ |
| Ether (C-O-C) | 21.7 | 39.9 | 0.54× |
| Secondary Amine (>NH) | 19.9 | 30.1 | 0.66× |
| Sulfonyl (-SO₂-) | 16.8 | 11.0 | 1.53× ⬆ |
| Carboxyl (-COOH) | 11.5 | 3.1 | 3.67× ⬆ |

**Cluster 29 (1651 molecules)** — representative: `C[NH+](C/C=C/c1ccco1)CCC(F)(F)F`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Amide (-CONH-) | 81.6 | 68.7 | 1.19× |
| Phenyl (aromatic ring) | 72.8 | 83.0 | 0.88× |
| Halide (C-X) | 49.1 | 37.1 | 1.32× |
| Ether (C-O-C) | 34.6 | 39.9 | 0.87× |
| Heterocycle | 32.5 | 56.1 | 0.58× |
| Secondary Amine (>NH) | 29.3 | 30.1 | 0.98× |
| Hydroxyl (-OH) | 17.5 | 11.6 | 1.50× ⬆ |
| Tertiary Amine (>N<) | 17.5 | 24.2 | 0.72× |

**Cluster 0 (1615 molecules)** — representative: `N#C[C@@H](NC(=O)N1CCCCC1)C1CCCCC1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Ether (C-O-C) | 56.4 | 39.9 | 1.41× |
| Amide (-CONH-) | 48.0 | 68.7 | 0.70× |
| Tertiary Amine (>N<) | 43.7 | 24.2 | 1.81× ⬆ |
| Secondary Amine (>NH) | 41.1 | 30.1 | 1.37× |
| Heterocycle | 27.1 | 56.1 | 0.48× ⬇ |
| Hydroxyl (-OH) | 14.8 | 11.6 | 1.27× |
| Thioether (C-S-C) | 14.0 | 7.4 | 1.89× ⬆ |
| Phenyl (aromatic ring) | 11.5 | 83.0 | 0.14× ⬇ |

**Cluster 870 (1313 molecules)** — representative: `O=C(NC1CCSCC1)C1=NN(c2ccccc2)[C@@H](c2cc`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 90.8 | 83.0 | 1.09× |
| Heterocycle | 78.4 | 56.1 | 1.40× |
| Ether (C-O-C) | 38.4 | 39.9 | 0.96× |
| Secondary Amine (>NH) | 34.8 | 30.1 | 1.16× |
| Tertiary Amine (>N<) | 33.6 | 24.2 | 1.39× |
| Amide (-CONH-) | 33.1 | 68.7 | 0.48× ⬇ |
| Halide (C-X) | 25.4 | 37.1 | 0.68× |
| Primary Amine (-NH₂) | 10.4 | 7.2 | 1.44× |

**Cluster 180 (1054 molecules)** — representative: `CC(C)c1ccnc(N2CCC3(CC2)C[C@@H](O)CO3)n1`

| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |
|---|---|---|---|
| Phenyl (aromatic ring) | 73.0 | 83.0 | 0.88× |
| Ether (C-O-C) | 57.6 | 39.9 | 1.44× |
| Amide (-CONH-) | 42.5 | 68.7 | 0.62× |
| Tertiary Amine (>N<) | 37.9 | 24.2 | 1.56× ⬆ |
| Heterocycle | 36.8 | 56.1 | 0.66× |
| Secondary Amine (>NH) | 35.7 | 30.1 | 1.19× |
| Halide (C-X) | 25.8 | 37.1 | 0.70× |
| Hydroxyl (-OH) | 15.6 | 11.6 | 1.34× |

## 9. Cluster Quality Analysis

### 9.1 Per-Stratum Quality Metrics

| Stratum | Silhouette | Davies-Bouldin | QE | Clusters | Gini | Singletons |
|---|---|---|---|---|---|---|
| 0 [0, 0.399) | -0.2384 | 3.7045 | 1.096530 | 612 | 0.566 | 41 |
| 1 [0.399, 0.520) | -0.2302 | 4.2907 | 1.048350 | 896 | 0.554 | 8 |
| 2 [0.520, 0.694) | -0.1930 | 3.7107 | 0.969829 | 900 | 0.557 | 4 |
| 3 [0.694, 0.814) | -0.1654 | 3.9224 | 0.935341 | 900 | 0.615 | 4 |
| 4 [0.814, 1.0] | -0.2002 | 4.4723 | 0.739544 | 900 | 0.584 | 0 |

**Interpretation guide:**
- **Silhouette** ∈ [-1, 1]: higher = better separation (>0.5 strong, >0.25 reasonable)
- **Davies-Bouldin**: lower = better separation (0 is optimal)
- **Gini coefficient**: 0 = equal sizes, 1 = maximally unequal

### 9.2 Cluster Size Distribution

| Stratum | Mean | Median | Std | Min | P25 | P75 | Max | Large |
|---|---|---|---|---|---|---|---|---|
| 0 | 11.2 | 6 | 24.8 | 1 | 4 | 10 | 454 | 38 |
| 1 | 19.7 | 11 | 46.7 | 1 | 7 | 16 | 982 | 55 |
| 2 | 67.1 | 36 | 181.2 | 1 | 24 | 55 | 3712 | 59 |
| 3 | 93.0 | 42 | 256.1 | 1 | 24 | 82 | 5072 | 56 |
| 4 | 89.9 | 44 | 182.2 | 2 | 27 | 75 | 3347 | 64 |

![Cluster Quality](results/figures/cluster_quality_comparison.svg)

*Figure 12: Comparison of cluster quality metrics across QED strata — silhouette score, Davies-Bouldin index, quantization error, and Gini coefficient.*

![Cluster Sizes](results/figures/cluster_size_distribution.svg)

*Figure 13: Distribution of cluster sizes within each QED stratum.*

## 10. Evaluation Summary

| Metric | Value |
|---|---|
| Total active clusters | 4208 / 4225 neurons |
| Cluster size (mean) | 59.3 |
| Cluster size (range) | 1 – 5072 |
| Average quantization error | 0.957919 |
| Average silhouette score | -0.2054 |
| Average Davies-Bouldin index | 4.0201 |
| Mean intra-cluster distance | 0.076976 |
| Functional group types detected | 22 / 22 |
| Strongest FG-property |r| | 0.1057 (Hydroxyl (-OH)) |

## 11. Performance

| Phase | Time |
|---|---|
| Data loading | 0.12s |
| Graph parsing + FG detection | 1.86s |
| VGAE encoding | 0.10s |
| Importance analysis | 0.33s |
| SOM clustering + FG analysis | 14.01s |
| **Total** | **26.97s** |

**Throughput**: 9249 molecules/second

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
│   ├── latent_space_umap.svg  # Latent space UMAP projection
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
| Figure 1 | [figures/property_distributions_combined.svg](results/figures/property_distributions_combined.svg) | Molecular property distributions (QED, logP, SAS) |
| Figure 2 | [figures/qed_distribution.svg](results/figures/qed_distribution.svg) | Individual property histograms with mean indicators |
| Figure 3 | [figures/molecule_complexity.svg](results/figures/molecule_complexity.svg) | Molecular graph complexity scatter (atoms vs bonds) |
| Figure 4 | [figures/fg_prevalence.svg](results/figures/fg_prevalence.svg) | Functional group prevalence bar chart |
| Figure 5 | [figures/reconstruction_loss_dist.svg](results/figures/reconstruction_loss_dist.svg) | VGAE reconstruction loss distribution |
| Figure 6 | [figures/embedding_dim_variance.svg](results/figures/embedding_dim_variance.svg) | Latent dimension variance analysis |
| Figure 7 | [figures/dim_property_heatmap.svg](results/figures/dim_property_heatmap.svg) | Dimension–property correlation heatmap |
| Figure 8 | [figures/fg_property_correlations.svg](results/figures/fg_property_correlations.svg) | FG–property correlation heatmap |
| Figure 9 | [figures/latent_space_umap.svg](results/figures/latent_space_umap.svg) | UMAP projection of latent space by stratum |
| Figure 10 | [figures/stratum_property_comparison.svg](results/figures/stratum_property_comparison.svg) | Stratum property comparison (mean ± std) |
| Figure 11 | [figures/umatrix_heatmaps.svg](results/figures/umatrix_heatmaps.svg) | SOM U-matrix heatmaps per stratum |
| Figure 12 | [figures/cluster_quality_comparison.svg](results/figures/cluster_quality_comparison.svg) | Cluster quality metrics comparison |
| Figure 13 | [figures/cluster_size_distribution.svg](results/figures/cluster_size_distribution.svg) | Cluster size distributions per stratum |
| Figure 14 | [figures/fg_enrichment_stratum_0.svg](results/figures/fg_enrichment_stratum_0.svg) | FG enrichment heatmap — Stratum 0 |
| Figure 15 | [figures/fg_enrichment_stratum_1.svg](results/figures/fg_enrichment_stratum_1.svg) | FG enrichment heatmap — Stratum 1 |
| Figure 16 | [figures/fg_enrichment_stratum_2.svg](results/figures/fg_enrichment_stratum_2.svg) | FG enrichment heatmap — Stratum 2 |
| Figure 17 | [figures/fg_enrichment_stratum_3.svg](results/figures/fg_enrichment_stratum_3.svg) | FG enrichment heatmap — Stratum 3 |
| Figure 18 | [figures/fg_enrichment_stratum_4.svg](results/figures/fg_enrichment_stratum_4.svg) | FG enrichment heatmap — Stratum 4 |

