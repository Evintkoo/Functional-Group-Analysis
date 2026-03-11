# Functional Group Analysis — Graph-Based Methodology (Rust)

## Overview

This project performs unsupervised discovery and characterization of functional-group patterns in ~250,000 drug-like molecules from the ZINC15 database. The pipeline is implemented in Rust using the **Burn** machine learning framework and represents molecules as **molecular graphs** rather than flat feature vectors.

### Pipeline Phases

| Phase | Description | Implementation |
|-------|-------------|----------------|
| **0** | Data ingestion | CSV parsing (SMILES, logP, QED, SAS) via `serde` + `csv` |
| **1** | Molecular graph construction + FG detection | SMILES → `petgraph::Graph<Atom, Bond, Undirected>` + 22-type substructure matching |
| **2** | Graph featurization | Node features (29-dim) + Edge features (9-dim) |
| **3** | Representation learning | Variational Graph Autoencoder (VGAE) with GAT encoder |
| **3.5** | Importance analysis | Latent dim ↔ property correlation, FG ↔ dim correlation, FG ↔ property correlation |
| **4** | Stratified clustering | QED-stratified Self-Organizing Map (SOM) on latent embeddings |
| **5** | Interpretation | FG enrichment per cluster, signature FGs, inter-cluster distances, representatives |

---

## Phase 0 — Data Ingestion

**Source**: ZINC15 database — 249,455 drug-like molecules

Each record contains:
- **SMILES**: Canonical molecular structure string
- **logP**: Octanol-water partition coefficient
- **QED**: Quantitative Estimate of Drug-likeness (0–1)
- **SAS**: Synthetic Accessibility Score

Implementation: `src/io/mod.rs` — streaming CSV deserialization with malformed-row recovery.

---

## Phase 1 — Molecular Graph Construction

**Module**: `src/smiles/mod.rs`

Molecules are represented as undirected graphs `Graph<Atom, Bond, Undirected>` where:
- **Nodes** = atoms (with element type, charge, chirality, aromaticity)
- **Edges** = bonds (with type, conjugation, ring membership, stereochemistry)

The SMILES parser handles:
- Organic subset atoms (`B, C, N, O, P, S, F, Cl, Br, I`)
- Bracket atoms with isotopes, chirality (`@`, `@@`), explicit H, and charges
- Bond types: single (`-`), double (`=`), triple (`#`), aromatic (`:`)
- Ring closures (single digit and `%nn` two-digit)
- Branches (parentheses)
- Disconnected fragments (`.`)
- Implicit aromaticity for lowercase SMILES atoms

Post-parsing ring/aromaticity detection via BFS identifies ring membership and conjugation.

### Molecular Graph Data Structure

```
Atom {
    element: Element,       // 48 supported elements + Unknown
    formal_charge: i8,
    chirality: Chirality,   // None | Clockwise | CounterClockwise
    is_aromatic: bool,
    explicit_h_count: u8,
    isotope: Option<u16>,
}

Bond {
    bond_type: BondType,    // Single | Double | Triple | Aromatic
    is_conjugated: bool,
    is_in_ring: bool,
    stereo: BondStereo,     // None | E | Z
}
```

---

## Phase 2 — Graph Featurization

**Module**: `src/features/mod.rs`

### Node Features (29 dimensions)

| Feature | Dims | Encoding |
|---------|------|----------|
| Atom type | 14 | One-hot (H, C, N, O, S, F, P, Cl, Br, I, B, Si, Se, Unknown) |
| Degree | 1 | Normalized by 6 |
| Formal charge | 1 | Normalized by 4 |
| Hybridization | 7 | One-hot (S, SP, SP², SP³, SP³D, SP³D², Other) |
| Is aromatic | 1 | Binary |
| Is in ring | 1 | Binary |
| Atomic mass | 1 | Normalized by 200 Da |
| Chirality | 3 | One-hot (None, CW, CCW) |

Hybridization is inferred from bond connectivity: triple bonds → SP, aromatic/double → SP², etc.

### Edge Features (9 dimensions)

| Feature | Dims | Encoding |
|---------|------|----------|
| Bond type | 4 | One-hot (Single, Double, Triple, Aromatic) |
| Is conjugated | 1 | Binary |
| Is in ring | 1 | Binary |
| Stereochemistry | 3 | One-hot (None, E, Z) |

### Graph Representation for GNN

The molecular features are packaged as:
- `node_features`: `[num_atoms × 29]` matrix
- `edge_features`: `[num_edges × 9]` matrix (bidirectional — each bond creates 2 directed edges)
- `edge_index`: COO-format `[[src, dst], ...]` for message passing

---

## Phase 3 — Variational Graph Autoencoder (VGAE)

**Modules**: `src/gnn/mod.rs`, `src/autoencoder/mod.rs`

### Architecture

```
Input Graph (N atoms, E bonds)
    │
    ├── Node features [N, 29]
    ├── Edge features [2E, 9]
    └── Edge index [2E, 2]
    │
    ▼
┌─────────────────────────────────┐
│  Input Projection: Linear(29→64)│
└──────────────┬──────────────────┘
               │
    ┌──────────▼──────────┐
    │   GAT Layer 1       │ ← Edge-aware attention message passing
    │   Linear(64→64)     │   score = LeakyReLU(a^T[Wh_src ‖ Wh_dst ‖ We])
    │   + Residual + ReLU │   softmax per destination → weighted aggregation
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   GAT Layer 2       │
    │   (same architecture)│
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   GAT Layer 3       │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Output Projection  │
    │  Linear(64→32)      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────┐
    │  Global Attention Pooling   │
    │  gate = Linear(32→1)        │
    │  α = softmax(gate(H))       │  → Graph-level embedding [1, 32]
    │  z = Σ αᵢ · hᵢ             │
    └──────────┬──────────────────┘
               │
    ┌──────────▼──────────┐     ┌──────────────────┐
    │  fc_mu: Linear(32→16)│     │ fc_logvar: 32→16 │
    └──────────┬──────────┘     └────────┬─────────┘
               │                          │
               ▼                          ▼
            μ [1,16]                log σ² [1,16]
               │                          │
               └──────┬──────────────────┘
                       │ Reparameterization
                       │ z = μ + σ · ε,  ε ~ N(0,1)
                       ▼
                   z [1, 16]
                       │
    ┌──────────────────▼──────────────────┐
    │  Decoder                             │
    │  Linear(16→64) + ReLU               │
    │  Linear(64→128) + ReLU              │
    │  Linear(128→29)                     │  → Reconstructed features
    └─────────────────────────────────────┘
```

### GAT Message Passing

Each GAT layer performs edge-aware attention:

1. **Project** all node features: `h = W_node · x`
2. **Project** edge features: `e = W_edge · edge_feat`
3. **Attention** per edge: `score = LeakyReLU(a^T [h_src ‖ h_dst ‖ e])`
4. **Normalize** via softmax per destination node
5. **Aggregate**: `h'_dst = Σ_src α_{src→dst} · (h_src + e_{src→dst})`

Isolated nodes (no edges) retain their projected features unchanged.

### Loss Function

```
L = L_recon + β · L_KL

L_recon = MSE(x̂, x)              // Reconstruction loss
L_KL = -0.5 · Σ(1 + log σ² - μ² - σ²)  // KL divergence
β = 0.001                          // KL weight (annealed)
```

### Key Improvement over Previous Approach

| Aspect | Previous (Python AE) | Current (Rust VGAE) |
|--------|---------------------|---------------------|
| Input | 28-dim flat vector (bag of atoms) | Full molecular graph |
| Topology | Lost entirely | Preserved via message passing |
| Feature learning | Dense layers only | Graph attention + dense decoder |
| Latent space | Deterministic | Variational (smoother, regularized) |
| Substructure sensitivity | None | Multi-hop neighborhood awareness |

---

## Phase 4 — Stratified SOM Clustering

**Module**: `src/som/mod.rs`

### QED Stratification

Molecules are divided into 5 strata based on QED valley detection:

| Stratum | QED Range | Description |
|---------|-----------|-------------|
| 0 | [0, 0.399) | Low drug-likeness |
| 1 | [0.399, 0.520) | Below-average |
| 2 | [0.520, 0.694) | Moderate |
| 3 | [0.694, 0.814) | Above-average |
| 4 | [0.814, 1.0] | High drug-likeness |

### Self-Organizing Map

**Grid**: 10×10 (100 neurons)
**Input**: 16-dimensional latent embeddings from VGAE

Training procedure:
1. **Initialization**: Random data sample per neuron
2. **Competitive learning** (128 epochs):
   - Find Best Matching Unit (BMU) via Euclidean distance
   - Update BMU and neighborhood with Gaussian kernel
   - Learning rate and radius decay linearly
3. **Output**: Cluster assignments + U-matrix for visualization

### Diagnostics

- **Quantization Error**: Mean distance from each point to its BMU
- **U-Matrix**: Inter-neuron distance matrix for boundary visualization

---

## Phase 5 — Interpretation

### 5a. Functional Group Detection

Each molecule undergoes substructure pattern matching to identify 22 functional group types:

| Category | Groups Detected |
|----------|----------------|
| **Oxygen-containing** | Hydroxyl (-OH), Carboxyl (-COOH), Ester (-COO-), Ether (C-O-C), Ketone (>C=O), Aldehyde (-CHO), Epoxide |
| **Nitrogen-containing** | Primary amine (-NH₂), Secondary amine (>NH), Tertiary amine (>N<), Amide (-CONH-), Nitro (-NO₂), Nitrile (-C≡N), Imine (C=N) |
| **Sulfur-containing** | Thiol (-SH), Thioether (C-S-C), Sulfonyl (-SO₂-), Sulfoxide (-SO-) |
| **Halogens** | Halide (C-F, C-Cl, C-Br, C-I) |
| **Ring systems** | Phenyl (aromatic carbocycle), Heterocycle (N/O/S in ring) |
| **Phosphorus** | Phosphate (-PO₄) |

Detection uses a two-pass algorithm:
1. **Complex groups first** (COOH, amide, ester, nitro) — claims atoms to prevent double-counting
2. **Simple groups** (OH, ketone, amine) — respects claimed atoms
3. **Ring analysis** — connected component counting for aromatic rings, cycle detection for epoxides

Implicit hydrogen counts are computed from valence rules for organic subset atoms (SMILES without brackets).

### 5b. Feature Importance Analysis

Three types of importance analysis are computed:

1. **Latent dimension ↔ Property correlation**: Pearson correlation between each of the 16 latent dimensions and molecular properties (QED, logP, SAS). Identifies which dimensions encode drug-likeness information.

2. **Functional group ↔ Latent dimension correlation**: Point-biserial correlation between FG presence (binary) and each latent dimension. Reveals which dimensions encode specific functional groups.

3. **Functional group ↔ Property correlation**: Point-biserial correlation between FG presence and molecular properties. Quantifies how functional groups influence drug-likeness.

### 5c. Cluster Characterization

For each cluster in each QED stratum:

- **FG census**: Prevalence and mean count of each functional group within the cluster
- **Enrichment analysis**: Cluster FG prevalence vs. stratum population — ratio > 1.0 indicates over-representation
- **Signature FGs**: Top functional groups with enrichment > 1.2× (distinguishing characteristics)
- **Dominant FG**: Most prevalent functional group in the cluster
- **Representative molecule**: SMILES closest to the cluster centroid in embedding space
- **Inter-cluster distances**: Pairwise Euclidean distances between cluster centroids, identifying most similar and most distinct cluster pairs

---

## Implementation

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | **Rust** (2021 edition) |
| ML framework | **Burn** 0.16 (ndarray backend) |
| Graph library | **petgraph** 0.7 |
| CSV I/O | **csv** + **serde** |
| Parallelism | **rayon** |
| Logging | **env_logger** |

### Module Structure

```
src/
├── main.rs                    # Entry point and CLI
├── smiles/mod.rs              # SMILES parser → molecular graph
├── features/mod.rs            # Atom/bond feature extraction
├── gnn/mod.rs                 # GAT layers + global attention pooling
├── autoencoder/mod.rs         # VGAE (encode, reparameterize, decode)
├── som/mod.rs                 # Self-Organizing Map
├── functional_groups/mod.rs   # 22-type FG detection + enrichment analysis
├── pipeline/mod.rs            # End-to-end pipeline orchestration
└── io/mod.rs                  # CSV loading and result output
```

### Running

```bash
cargo build --release
./target/release/functional_group_analysis [csv_path] [output_dir]

# Defaults:
#   csv_path = 250k_rndm_zinc_drugs_clean_3.csv
#   output_dir = results/
```

### Output Structure

```
results/
├── RESULTS.md              # Generated analysis report
├── training_losses.csv     # Epoch-wise train/val losses
└── group_{0..4}/
    ├── labeled_data.csv    # SMILES + properties + cluster label
    └── embeddings.csv      # 16-dim latent embeddings
```

---

## References

1. Kipf, T.N. & Welling, M. (2016). Variational Graph Auto-Encoders. *arXiv:1611.07308*
2. Veličković, P. et al. (2018). Graph Attention Networks. *ICLR 2018*
3. Kohonen, T. (1990). The Self-Organizing Map. *Proceedings of the IEEE*
4. Sterling, T. & Irwin, J.J. (2015). ZINC 15 – Ligand Discovery for Everyone. *J. Chem. Inf. Model.*
5. Bickerton, G.R. et al. (2012). Quantifying the chemical beauty of drugs. *Nature Chemistry*
