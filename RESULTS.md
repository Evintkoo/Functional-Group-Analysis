# Functional Group Analysis — Results

## Pipeline Configuration

| Parameter | Value |
|---|---|
| Total molecules in dataset | 249,455 |
| Implementation | Rust + Burn (ndarray backend) |
| GNN architecture | GAT (3 layers, edge-aware attention) |
| Node feature dim | 29 |
| Edge feature dim | 9 |
| Latent dimension | 16 |
| SOM grid | 10×10 (100 neurons) |
| QED strata | 5 |

## Architecture Summary

```
SMILES → Molecular Graph (petgraph)
    → Node features [N, 29] + Edge features [2E, 9]
    → GAT Encoder (3 layers, 64-dim hidden, residual + ReLU)
    → Global Attention Pooling → [1, 32]
    → Variational: μ, log σ² → z [1, 16]
    → SOM Clustering (10×10, 128 epochs)
```

## QED Stratification

| Stratum | QED Range | Description |
|---------|-----------|-------------|
| 0 | [0, 0.399) | Low drug-likeness |
| 1 | [0.399, 0.520) | Below-average |
| 2 | [0.520, 0.694) | Moderate |
| 3 | [0.694, 0.814) | Above-average |
| 4 | [0.814, 1.0] | High drug-likeness |

## Methodology Improvements

| Aspect | Previous (Python) | Current (Rust + GNN) |
|---|---|---|
| Molecular representation | Flat 28-dim feature vector | Full molecular graph (nodes + edges) |
| Feature learning | Dense autoencoder (28→512→128→16) | Graph Attention Network (3 layers) |
| Latent model | Deterministic AE | Variational (VGAE with KL regularization) |
| Structure awareness | None (bag of atoms) | Message passing preserves bond topology |
| Pooling | N/A (fixed features) | Global attention pooling (learned) |
| Edge information | Not used | 9-dim bond features in attention |
| Implementation | Python/PyTorch | Rust/Burn (memory-safe, zero-cost abstractions) |
| Parallelism | None | rayon-ready data pipeline |

## Module Inventory

| Module | Lines | Purpose |
|--------|-------|---------|
| `smiles/mod.rs` | ~490 | SMILES parser → molecular graph |
| `features/mod.rs` | ~170 | Atom/bond feature extraction (29+9 dims) |
| `gnn/mod.rs` | ~200 | GAT layers + global attention pooling |
| `autoencoder/mod.rs` | ~160 | VGAE (encode, reparameterize, decode) |
| `som/mod.rs` | ~220 | Self-Organizing Map with U-matrix |
| `pipeline/mod.rs` | ~240 | End-to-end orchestration |
| `io/mod.rs` | ~130 | CSV I/O and result serialization |
| `main.rs` | ~40 | CLI entry point |

## Test Results

```
running 10 tests
test features::tests::test_carbon_features ... ok
test features::tests::test_feature_dimensions ... ok
test io::tests::test_stratification ... ok
test smiles::tests::test_parse_benzene ... ok
test smiles::tests::test_parse_bracket_atom ... ok
test smiles::tests::test_parse_branch ... ok
test smiles::tests::test_parse_ethanol ... ok
test smiles::tests::test_parse_methane ... ok
test som::tests::test_som_basic ... ok
test som::tests::test_u_matrix ... ok

test result: ok. 10 passed; 0 failed; 0 ignored
```

## Output Files

When run on the ZINC dataset, the pipeline produces:

```
results/
├── RESULTS.md              # This report (auto-generated)
├── training_losses.csv     # VGAE training metrics
└── group_{0..4}/
    ├── labeled_data.csv    # smiles, logP, qed, SAS, cluster
    └── embeddings.csv      # 16-dim latent vectors per molecule
```

## Build & Run

```bash
# Build
cargo build --release

# Run (defaults to 250k_rndm_zinc_drugs_clean_3.csv → results/)
./target/release/functional_group_analysis

# Custom paths
./target/release/functional_group_analysis my_data.csv output/
```
