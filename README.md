# Functional Group Analysis of Drug-Like Chemical Space

Graph-based functional group analysis of drug-like molecules using Variational Graph Autoencoders (VGAE) and Self-Organising Maps (SOM), implemented in Rust with [Burn](https://burn.dev/).

## Overview

This project analyses the distribution and co-occurrence of functional groups across ~250k drug-like molecules from ZINC, using a pipeline that combines:

1. **SMILES parsing** — tokenises molecular strings into atoms, bonds, and ring structures
2. **Molecular graph construction** — builds atom-level graphs with node/edge features
3. **GNN + VGAE encoding** — learns latent representations of molecular graphs via variational graph autoencoding
4. **Self-Organising Map (SOM)** — clusters molecules in latent space to reveal functional group organisation
5. **Statistical enrichment** — identifies which functional groups are enriched in each SOM cluster (with FDR correction)

## Project Structure

```
├── src/                  # Rust source code
│   ├── main.rs           # Entry point / pipeline orchestration
│   ├── smiles/           # SMILES tokeniser and parser
│   ├── functional_groups/# Functional group detection (SMARTS-like)
│   ├── features/         # Node and edge feature extraction
│   ├── gnn/              # Graph neural network layers
│   ├── autoencoder/      # Variational graph autoencoder (VGAE)
│   ├── som/              # Self-Organising Map implementation
│   ├── stats/            # Enrichment analysis and FDR testing
│   ├── visualization/    # Plotting (plotters + SVG output)
│   ├── pipeline/         # End-to-end pipeline logic
│   └── io/               # CSV / data I/O
├── paper/                # LaTeX manuscript
│   ├── main.tex
│   └── main.pdf
├── results/              # Pipeline outputs (figures, checkpoints, cluster data)
└── 250k_rndm_zinc_drugs_clean_3.csv  # Input dataset
```

## Requirements

- Rust 1.75+ (2021 edition)
- GPU support via wgpu (optional; falls back to ndarray CPU backend)

## Usage

```bash
# Build
cargo build --release

# Run the full pipeline
cargo run --release
```

Pipeline outputs (figures, cluster assignments, training losses) are written to `results/`.

## Paper

The accompanying manuscript is in `paper/main.tex`. To compile:

```bash
cd paper && pdflatex main.tex
```

## License

See [LICENSE](LICENSE).
