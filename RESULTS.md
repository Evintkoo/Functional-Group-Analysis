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

## 3. Model Configuration

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

## 4. VGAE Encoding Results

| Metric | Value |
|---|---|
| Mean reconstruction loss | 0.176418 |
| Mean pairwise embedding distance | 0.133487 |
| Embedding std (mean across dims) | 0.025042 |
| Embedding std range | [0.0057, 0.0432] |

### Latent Dimension Statistics

| Dim | Mean | Std | Min | Max |
|---|---|---|---|---|
| 0 | -0.1685 | 0.0419 | -0.3749 | -0.0711 |
| 1 | -0.0079 | 0.0252 | -0.1107 | 0.0917 |
| 2 | -0.1537 | 0.0432 | -0.3790 | -0.0543 |
| 3 | 0.4265 | 0.0256 | 0.3464 | 0.5005 |
| 4 | -0.1873 | 0.0285 | -0.2547 | -0.0338 |
| 5 | -0.1748 | 0.0117 | -0.2233 | -0.1106 |
| 6 | 0.4510 | 0.0389 | 0.2857 | 0.5818 |
| 7 | 0.1356 | 0.0299 | 0.0202 | 0.2110 |
| 8 | -0.3057 | 0.0283 | -0.3849 | -0.1767 |
| 9 | -0.0586 | 0.0162 | -0.1322 | 0.0088 |
| 10 | -0.5063 | 0.0319 | -0.6458 | -0.4205 |
| 11 | 0.1684 | 0.0057 | 0.1375 | 0.1862 |
| 12 | -0.0953 | 0.0096 | -0.1392 | -0.0581 |
| 13 | 0.1316 | 0.0143 | 0.0872 | 0.1899 |
| 14 | 0.1613 | 0.0387 | -0.0129 | 0.2712 |
| 15 | 0.1210 | 0.0110 | 0.0907 | 0.1778 |

## 5. Stratified Clustering Results

### Per-Stratum Overview

| Stratum | QED Range | Molecules | Active Clusters | QE | U-Matrix Mean | U-Matrix Max |
|---|---|---|---|---|---|---|
| 0 | [0, 0.399) | 122 | 63 | 0.038178 | 0.0176 | 0.0339 |
| 1 | [0.399, 0.520) | 376 | 93 | 0.037791 | 0.0158 | 0.0274 |
| 2 | [0.520, 0.694) | 1243 | 99 | 0.041300 | 0.0179 | 0.0291 |
| 3 | [0.694, 0.814) | 1645 | 100 | 0.041200 | 0.0170 | 0.0220 |
| 4 | [0.814, 1.0] | 1614 | 100 | 0.035609 | 0.0139 | 0.0189 |

**Total clustered**: 5000 molecules | **Avg QE**: 0.038816

### Stratum 0 — Top Clusters by Size

| Cluster | Size | Mean QED | Std QED | Mean logP | Mean SAS | Compactness |
|---|---|---|---|---|---|---|
| 99 | 10 | 0.3380 | 0.0442 | 4.1556 | 2.4879 | 0.0278 |
| 90 | 6 | 0.3593 | 0.0324 | 4.2690 | 2.3507 | 0.0356 |
| 0 | 5 | 0.3512 | 0.0390 | 0.6662 | 3.5074 | 0.0992 |
| 20 | 4 | 0.3333 | 0.0145 | 1.2284 | 3.5328 | 0.0413 |
| 36 | 4 | 0.3683 | 0.0295 | 3.1128 | 2.9574 | 0.0200 |
| 40 | 4 | 0.3201 | 0.0432 | 1.1159 | 2.9039 | 0.0307 |
| 3 | 3 | 0.2689 | 0.1105 | -0.1071 | 4.1435 | 0.0624 |
| 49 | 3 | 0.3213 | 0.0658 | 4.5952 | 2.7910 | 0.0120 |
| 55 | 3 | 0.3451 | 0.0345 | 3.8606 | 2.8076 | 0.0206 |
| 85 | 3 | 0.3723 | 0.0225 | 3.5081 | 2.4577 | 0.0128 |

### Stratum 1 — Top Clusters by Size

| Cluster | Size | Mean QED | Std QED | Mean logP | Mean SAS | Compactness |
|---|---|---|---|---|---|---|
| 9 | 23 | 0.4715 | 0.0319 | 4.0171 | 2.4286 | 0.0291 |
| 0 | 19 | 0.4797 | 0.0286 | 3.8731 | 2.3726 | 0.0330 |
| 90 | 18 | 0.4684 | 0.0392 | 0.3897 | 4.1443 | 0.0965 |
| 19 | 13 | 0.4618 | 0.0356 | 4.1011 | 2.6506 | 0.0276 |
| 4 | 10 | 0.4676 | 0.0372 | 3.6321 | 2.3424 | 0.0281 |
| 6 | 10 | 0.4787 | 0.0368 | 3.3883 | 2.4312 | 0.0172 |
| 59 | 9 | 0.4808 | 0.0316 | 3.4496 | 2.9820 | 0.0234 |
| 35 | 8 | 0.4745 | 0.0330 | 3.8193 | 2.4587 | 0.0201 |
| 46 | 8 | 0.4536 | 0.0330 | 3.5132 | 2.6571 | 0.0137 |
| 99 | 8 | 0.4587 | 0.0377 | 4.0198 | 2.8167 | 0.0242 |

### Stratum 2 — Top Clusters by Size

| Cluster | Size | Mean QED | Std QED | Mean logP | Mean SAS | Compactness |
|---|---|---|---|---|---|---|
| 0 | 71 | 0.6017 | 0.0549 | 3.4529 | 2.4670 | 0.0336 |
| 99 | 69 | 0.6266 | 0.0489 | -0.0525 | 4.1566 | 0.0912 |
| 5 | 43 | 0.6275 | 0.0432 | 3.4231 | 3.1044 | 0.0308 |
| 4 | 42 | 0.6283 | 0.0493 | 3.6492 | 2.7109 | 0.0265 |
| 90 | 39 | 0.6260 | 0.0517 | 2.6021 | 2.5326 | 0.0342 |
| 50 | 37 | 0.6212 | 0.0454 | 3.3665 | 2.3088 | 0.0302 |
| 40 | 30 | 0.6078 | 0.0519 | 3.3404 | 2.3644 | 0.0298 |
| 94 | 29 | 0.6268 | 0.0467 | 1.5650 | 2.9582 | 0.0469 |
| 30 | 27 | 0.6191 | 0.0546 | 3.4881 | 2.3305 | 0.0217 |
| 1 | 25 | 0.6119 | 0.0575 | 3.8676 | 2.5329 | 0.0236 |

### Stratum 3 — Top Clusters by Size

| Cluster | Size | Mean QED | Std QED | Mean logP | Mean SAS | Compactness |
|---|---|---|---|---|---|---|
| 9 | 99 | 0.7479 | 0.0304 | 3.0852 | 2.5278 | 0.0377 |
| 90 | 76 | 0.7444 | 0.0341 | 1.1654 | 3.7944 | 0.0678 |
| 0 | 51 | 0.7564 | 0.0320 | 1.0515 | 4.5949 | 0.0631 |
| 5 | 41 | 0.7641 | 0.0320 | 2.6398 | 3.2072 | 0.0277 |
| 99 | 41 | 0.7443 | 0.0279 | 3.1532 | 2.5024 | 0.0387 |
| 49 | 40 | 0.7473 | 0.0326 | 2.7334 | 2.6343 | 0.0292 |
| 19 | 36 | 0.7443 | 0.0314 | 3.2709 | 2.3537 | 0.0224 |
| 95 | 34 | 0.7657 | 0.0273 | 1.9536 | 2.7518 | 0.0462 |
| 4 | 33 | 0.7564 | 0.0337 | 2.2281 | 3.4543 | 0.0361 |
| 6 | 31 | 0.7592 | 0.0322 | 2.7654 | 2.8572 | 0.0280 |

### Stratum 4 — Top Clusters by Size

| Cluster | Size | Mean QED | Std QED | Mean logP | Mean SAS | Compactness |
|---|---|---|---|---|---|---|
| 90 | 98 | 0.8552 | 0.0281 | 2.1622 | 3.1861 | 0.0550 |
| 9 | 63 | 0.8774 | 0.0362 | 2.4801 | 2.9031 | 0.0306 |
| 99 | 46 | 0.8683 | 0.0315 | 2.4377 | 2.2585 | 0.0291 |
| 0 | 42 | 0.8593 | 0.0342 | 1.9037 | 4.1674 | 0.0479 |
| 49 | 42 | 0.8695 | 0.0351 | 2.5277 | 2.5889 | 0.0288 |
| 40 | 41 | 0.8599 | 0.0322 | 2.2240 | 3.5841 | 0.0314 |
| 50 | 39 | 0.8586 | 0.0314 | 2.3324 | 3.4679 | 0.0255 |
| 95 | 38 | 0.8599 | 0.0323 | 2.7697 | 2.5647 | 0.0306 |
| 98 | 32 | 0.8573 | 0.0324 | 2.4801 | 2.4020 | 0.0282 |
| 2 | 30 | 0.8737 | 0.0302 | 1.5747 | 3.8677 | 0.0332 |

## 6. Evaluation Summary

| Metric | Value |
|---|---|
| Total active clusters | 455 / 500 neurons |
| Cluster size (mean) | 11.0 |
| Cluster size (range) | 1 – 99 |
| Average quantization error | 0.038816 |
| Mean intra-cluster distance | 0.024229 |

## 7. Performance

| Phase | Time |
|---|---|
| Data loading | 0.12s |
| Graph parsing | 0.02s |
| VGAE encoding | 10.79s |
| SOM clustering | 0.61s |
| **Total** | **11.60s** |

**Throughput**: 431 molecules/second

## 8. Methodology Comparison

| Aspect | Previous (Python) | Current (Rust + GNN) |
|---|---|---|
| Molecular representation | Flat 28-dim feature vector | Full molecular graph |
| Feature learning | Dense autoencoder (28→16→28) | Graph Attention Network (3 layers) |
| Latent model | Deterministic AE | Variational (VGAE with KL regularization) |
| Structure awareness | None (bag of atoms) | Message passing preserves bond topology |
| Pooling | N/A (fixed features) | Global attention pooling (learned) |
| Edge features | Not used | 9-dim bond features in attention |
| Implementation | Python/PyTorch | Rust/Burn (memory-safe, zero-cost abstractions) |

## 9. Output Files

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
