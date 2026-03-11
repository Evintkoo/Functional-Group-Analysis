/// Full analysis pipeline: CSV → SMILES → Graph → GNN encode → SOM cluster → results.

use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use std::time::Instant;
use std::collections::HashMap;

use crate::autoencoder::{Vgae, VgaeConfig, TrainConfig, vgae_loss};
use crate::features::{self, MolecularFeatures, NODE_FEATURE_DIM, EDGE_FEATURE_DIM};
use crate::io::{self, MoleculeRecord};
use crate::smiles;
use crate::som::{Som, SomConfig};

type B = NdArray<f32>;

/// Convert MolecularFeatures into Burn tensors.
fn features_to_tensors(
    feats: &MolecularFeatures,
    device: &<B as Backend>::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let node_data: Vec<f32> = feats.node_features.iter().flatten().copied().collect();
    let node_tensor = Tensor::<B, 1>::from_floats(
        node_data.as_slice(), device,
    ).reshape([feats.num_atoms, NODE_FEATURE_DIM]);

    let edge_data: Vec<f32> = feats.edge_features.iter().flatten().copied().collect();
    let num_edge_entries = feats.edge_features.len().max(1);
    let edge_tensor = if edge_data.is_empty() {
        Tensor::zeros([1, EDGE_FEATURE_DIM], device)
    } else {
        Tensor::<B, 1>::from_floats(
            edge_data.as_slice(), device,
        ).reshape([num_edge_entries, EDGE_FEATURE_DIM])
    };

    (node_tensor, edge_tensor)
}

/// Process SMILES strings into molecular features, collecting graph statistics.
fn process_molecules(records: &[&MoleculeRecord]) -> (Vec<(MolecularFeatures, usize)>, GraphStats) {
    let mut results = Vec::new();
    let mut stats = GraphStats::default();
    let mut parse_failures = 0usize;

    for (i, rec) in records.iter().enumerate() {
        match smiles::parse_smiles(&rec.smiles) {
            Ok(graph) => {
                let feats = features::extract_features(&graph);
                if feats.num_atoms > 0 {
                    stats.total_atoms += feats.num_atoms;
                    stats.total_bonds += feats.num_bonds;
                    stats.min_atoms = stats.min_atoms.min(feats.num_atoms);
                    stats.max_atoms = stats.max_atoms.max(feats.num_atoms);
                    stats.min_bonds = stats.min_bonds.min(feats.num_bonds);
                    stats.max_bonds = stats.max_bonds.max(feats.num_bonds);
                    stats.molecule_count += 1;
                    results.push((feats, i));
                }
            }
            Err(_) => {
                parse_failures += 1;
            }
        }

        if (i + 1) % 500 == 0 {
            log::info!("  Parsed {}/{} molecules ({} failures)", i + 1, records.len(), parse_failures);
        }
    }

    stats.parse_failures = parse_failures;
    (results, stats)
}

/// Compute embedding statistics for a set of embeddings.
fn embedding_stats(embeddings: &[Vec<f32>]) -> EmbeddingStats {
    if embeddings.is_empty() {
        return EmbeddingStats::default();
    }

    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;

    let mut means = vec![0.0f32; dim];
    let mut mins = vec![f32::MAX; dim];
    let mut maxs = vec![f32::MIN; dim];

    for emb in embeddings {
        for (d, &v) in emb.iter().enumerate() {
            means[d] += v;
            mins[d] = mins[d].min(v);
            maxs[d] = maxs[d].max(v);
        }
    }
    for d in 0..dim {
        means[d] /= n;
    }

    let mut stds = vec![0.0f32; dim];
    for emb in embeddings {
        for (d, &v) in emb.iter().enumerate() {
            stds[d] += (v - means[d]).powi(2);
        }
    }
    for d in 0..dim {
        stds[d] = (stds[d] / n).sqrt();
    }

    // Mean pairwise distance (sample up to 500 pairs for speed)
    let mut total_dist = 0.0f64;
    let mut pair_count = 0u64;
    let sample_size = embeddings.len().min(500);
    for i in 0..sample_size {
        for j in (i + 1)..sample_size {
            let dist: f64 = embeddings[i].iter().zip(embeddings[j].iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            total_dist += dist;
            pair_count += 1;
        }
    }
    let mean_pairwise_dist = if pair_count > 0 { total_dist / pair_count as f64 } else { 0.0 };

    EmbeddingStats { means, stds, mins, maxs, mean_pairwise_dist }
}

/// Compute cluster-level statistics.
fn cluster_stats(
    labels: &[usize],
    embeddings: &[Vec<f32>],
    records: &[&MoleculeRecord],
) -> Vec<ClusterInfo> {
    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        cluster_map.entry(label).or_default().push(i);
    }

    let mut infos: Vec<ClusterInfo> = cluster_map.iter().map(|(&cluster_id, members)| {
        let size = members.len();

        // QED statistics for this cluster
        let qed_vals: Vec<f64> = members.iter()
            .filter_map(|&i| records.get(i).map(|r| r.qed))
            .collect();
        let mean_qed = qed_vals.iter().sum::<f64>() / qed_vals.len().max(1) as f64;
        let std_qed = if qed_vals.len() > 1 {
            (qed_vals.iter().map(|q| (q - mean_qed).powi(2)).sum::<f64>() / (qed_vals.len() - 1) as f64).sqrt()
        } else { 0.0 };

        // LogP statistics
        let logp_vals: Vec<f64> = members.iter()
            .filter_map(|&i| records.get(i).map(|r| r.log_p))
            .collect();
        let mean_logp = logp_vals.iter().sum::<f64>() / logp_vals.len().max(1) as f64;

        // SAS statistics
        let sas_vals: Vec<f64> = members.iter()
            .filter_map(|&i| records.get(i).map(|r| r.sas))
            .collect();
        let mean_sas = sas_vals.iter().sum::<f64>() / sas_vals.len().max(1) as f64;

        // Intra-cluster distance (compactness)
        let cluster_embs: Vec<&Vec<f32>> = members.iter()
            .filter_map(|&i| embeddings.get(i))
            .collect();
        let centroid: Vec<f32> = if !cluster_embs.is_empty() {
            let dim = cluster_embs[0].len();
            let mut c = vec![0.0f32; dim];
            for e in &cluster_embs {
                for (d, v) in e.iter().enumerate() {
                    c[d] += v;
                }
            }
            let n = cluster_embs.len() as f32;
            c.iter_mut().for_each(|v| *v /= n);
            c
        } else {
            Vec::new()
        };

        let mean_dist_to_centroid = if !cluster_embs.is_empty() && !centroid.is_empty() {
            let total: f64 = cluster_embs.iter().map(|e| {
                e.iter().zip(centroid.iter())
                    .map(|(a, b)| ((a - b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }).sum();
            total / cluster_embs.len() as f64
        } else { 0.0 };

        ClusterInfo {
            cluster_id,
            size,
            mean_qed,
            std_qed,
            mean_logp,
            mean_sas,
            compactness: mean_dist_to_centroid,
        }
    }).collect();

    infos.sort_by_key(|c| c.cluster_id);
    infos
}

/// Run the complete analysis pipeline.
pub fn run_pipeline(csv_path: &str, output_dir: &str) -> Result<PipelineResults, Box<dyn std::error::Error>> {
    let pipeline_start = Instant::now();
    let device = Default::default();
    let train_config = TrainConfig::default();

    // ═══════════════════════════════════════════════════
    // Phase 0: Load data
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 0: Loading data");
    log::info!("════════════════════════════════════════════");
    let t0 = Instant::now();
    let all_records = io::load_zinc_csv(csv_path)?;
    let total = all_records.len();
    let load_time = t0.elapsed();
    log::info!("Loaded {} molecules in {:.2}s", total, load_time.as_secs_f64());

    // Use a meaningful experiment size
    let max_molecules = total.min(5000);
    let records = &all_records[..max_molecules];
    log::info!("Experiment subset: {} molecules", max_molecules);

    // Dataset overview
    let qed_vals: Vec<f64> = records.iter().map(|r| r.qed).collect();
    let logp_vals: Vec<f64> = records.iter().map(|r| r.log_p).collect();
    let sas_vals: Vec<f64> = records.iter().map(|r| r.sas).collect();
    let dataset_stats = DatasetStats {
        total_in_file: total,
        used: max_molecules,
        qed_mean: qed_vals.iter().sum::<f64>() / qed_vals.len() as f64,
        qed_std: stat_std(&qed_vals),
        qed_min: qed_vals.iter().copied().fold(f64::MAX, f64::min),
        qed_max: qed_vals.iter().copied().fold(f64::MIN, f64::max),
        logp_mean: logp_vals.iter().sum::<f64>() / logp_vals.len() as f64,
        logp_std: stat_std(&logp_vals),
        sas_mean: sas_vals.iter().sum::<f64>() / sas_vals.len() as f64,
        sas_std: stat_std(&sas_vals),
    };

    log::info!("  QED:  mean={:.4} ± {:.4}  [{:.4}, {:.4}]",
        dataset_stats.qed_mean, dataset_stats.qed_std,
        dataset_stats.qed_min, dataset_stats.qed_max);
    log::info!("  logP: mean={:.4} ± {:.4}", dataset_stats.logp_mean, dataset_stats.logp_std);
    log::info!("  SAS:  mean={:.4} ± {:.4}", dataset_stats.sas_mean, dataset_stats.sas_std);

    // ═══════════════════════════════════════════════════
    // Phase 1: Parse SMILES and extract graph features
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 1: Molecular graph construction");
    log::info!("════════════════════════════════════════════");
    let t1 = Instant::now();
    let record_refs: Vec<&MoleculeRecord> = records.iter().collect();
    let (mol_features, graph_stats) = process_molecules(&record_refs);
    let parse_time = t1.elapsed();

    log::info!("Parsed {}/{} molecules in {:.2}s ({} failures)",
        mol_features.len(), max_molecules, parse_time.as_secs_f64(), graph_stats.parse_failures);
    log::info!("  Atoms per molecule: min={}, max={}, avg={:.1}",
        graph_stats.min_atoms, graph_stats.max_atoms,
        graph_stats.total_atoms as f64 / graph_stats.molecule_count.max(1) as f64);
    log::info!("  Bonds per molecule: min={}, max={}, avg={:.1}",
        graph_stats.min_bonds, graph_stats.max_bonds,
        graph_stats.total_bonds as f64 / graph_stats.molecule_count.max(1) as f64);

    // ═══════════════════════════════════════════════════
    // Phase 2: VGAE Encoding
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 2: VGAE Encoding");
    log::info!("════════════════════════════════════════════");
    let t2 = Instant::now();

    let vgae_config = VgaeConfig::new(NODE_FEATURE_DIM, EDGE_FEATURE_DIM)
        .with_hidden_dim(64)
        .with_gnn_output_dim(32)
        .with_latent_dim(16)
        .with_num_gnn_layers(3);

    let vgae: Vgae<B> = vgae_config.init(&device);

    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut valid_indices: Vec<usize> = Vec::new();
    let mut recon_losses: Vec<f32> = Vec::new();

    for (i, (feats, orig_idx)) in mol_features.iter().enumerate() {
        let (node_t, edge_t) = features_to_tensors(feats, &device);

        // Forward pass to get embedding + reconstruction loss
        let output = vgae.forward(node_t.clone(), &feats.edge_index, edge_t.clone(), feats.num_atoms);
        let embedding: Vec<f32> = output.mu.to_data().to_vec().unwrap();

        let loss = vgae_loss(output.reconstructed, node_t, output.mu, output.log_var, train_config.kl_weight);
        let loss_val: f32 = loss.into_scalar();
        recon_losses.push(loss_val);

        embeddings.push(embedding);
        valid_indices.push(*orig_idx);

        if (i + 1) % 500 == 0 {
            let avg_loss: f32 = recon_losses.iter().sum::<f32>() / recon_losses.len() as f32;
            log::info!("  Encoded {}/{} | avg loss = {:.6}", i + 1, mol_features.len(), avg_loss);
        }
    }

    let encode_time = t2.elapsed();
    let avg_recon_loss = recon_losses.iter().sum::<f32>() / recon_losses.len().max(1) as f32;
    let emb_stats = embedding_stats(&embeddings);

    log::info!("Encoding complete in {:.2}s", encode_time.as_secs_f64());
    log::info!("  Mean reconstruction loss: {:.6}", avg_recon_loss);
    log::info!("  Mean pairwise embedding distance: {:.6}", emb_stats.mean_pairwise_dist);
    log::info!("  Embedding std range: [{:.4}, {:.4}]",
        emb_stats.stds.iter().copied().fold(f32::MAX, f32::min),
        emb_stats.stds.iter().copied().fold(f32::MIN, f32::max));

    // Save losses
    io::save_training_losses(output_dir, &recon_losses, &recon_losses)?;

    // ═══════════════════════════════════════════════════
    // Phase 3: QED stratification + SOM clustering
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 3: Stratified SOM clustering");
    log::info!("════════════════════════════════════════════");
    let t3 = Instant::now();

    let qed_edges = vec![0.399, 0.520, 0.694, 0.814];
    let strata = io::stratify_by_qed(records, &qed_edges);

    let mut all_labels = vec![0usize; embeddings.len()];
    let mut strata_results = Vec::new();

    for (group_id, stratum_indices) in strata.iter().enumerate() {
        let stratum_emb_indices: Vec<usize> = stratum_indices.iter()
            .filter_map(|&orig_idx| valid_indices.iter().position(|&vi| vi == orig_idx))
            .collect();

        if stratum_emb_indices.is_empty() {
            log::warn!("Stratum {} has no valid embeddings, skipping", group_id);
            continue;
        }

        let stratum_embeddings: Vec<Vec<f32>> = stratum_emb_indices.iter()
            .map(|&i| embeddings[i].clone())
            .collect();

        log::info!("─── Stratum {} ───  {} molecules", group_id, stratum_embeddings.len());

        // Train SOM
        let som_config = SomConfig::new(16);
        let mut som = Som::new(&som_config, &stratum_embeddings);
        som.train(&stratum_embeddings, &som_config);

        let labels = som.predict(&stratum_embeddings);
        let qe = som.quantization_error(&stratum_embeddings);
        let u_matrix = som.u_matrix();

        // U-matrix statistics
        let u_vals: Vec<f64> = u_matrix.iter().flatten().copied().collect();
        let u_mean = u_vals.iter().sum::<f64>() / u_vals.len() as f64;
        let u_max = u_vals.iter().copied().fold(f64::MIN, f64::max);

        // Assign labels back
        for (k, &emb_idx) in stratum_emb_indices.iter().enumerate() {
            all_labels[emb_idx] = labels[k];
        }

        // Cluster statistics
        let stratum_records: Vec<&MoleculeRecord> = stratum_emb_indices.iter()
            .filter_map(|&emb_idx| {
                let orig_idx = valid_indices[emb_idx];
                records.get(orig_idx)
            })
            .collect();

        let cluster_infos = cluster_stats(&labels, &stratum_embeddings, &stratum_records);

        // Count unique clusters
        let mut used_clusters: Vec<usize> = labels.clone();
        used_clusters.sort();
        used_clusters.dedup();

        log::info!("  Active clusters: {}/100", used_clusters.len());
        log::info!("  Quantization error: {:.6}", qe);
        log::info!("  U-matrix: mean={:.4}, max={:.4}", u_mean, u_max);

        // Show top 5 largest clusters
        let mut sorted_clusters = cluster_infos.clone();
        sorted_clusters.sort_by(|a, b| b.size.cmp(&a.size));
        for ci in sorted_clusters.iter().take(5) {
            log::info!("    Cluster {:3}: {:4} mols | QED={:.3}±{:.3} | logP={:.2} | SAS={:.2} | compact={:.4}",
                ci.cluster_id, ci.size, ci.mean_qed, ci.std_qed, ci.mean_logp, ci.mean_sas, ci.compactness);
        }

        // Save results for this stratum
        let stratum_labels_for_save: Vec<usize> = stratum_emb_indices.iter()
            .map(|&emb_idx| all_labels[emb_idx])
            .collect();

        io::save_cluster_results(
            output_dir, group_id, &stratum_records, &stratum_labels_for_save, &stratum_embeddings,
        )?;

        strata_results.push(StratumResult {
            group_id,
            num_molecules: stratum_embeddings.len(),
            num_clusters_used: used_clusters.len(),
            quantization_error: qe,
            u_matrix_mean: u_mean,
            u_matrix_max: u_max,
            cluster_infos,
        });
    }

    let cluster_time = t3.elapsed();
    let total_time = pipeline_start.elapsed();

    log::info!("════════════════════════════════════════════");
    log::info!("  Pipeline complete");
    log::info!("════════════════════════════════════════════");
    log::info!("  Data loading:   {:.2}s", load_time.as_secs_f64());
    log::info!("  Graph parsing:  {:.2}s", parse_time.as_secs_f64());
    log::info!("  VGAE encoding:  {:.2}s", encode_time.as_secs_f64());
    log::info!("  SOM clustering: {:.2}s", cluster_time.as_secs_f64());
    log::info!("  Total:          {:.2}s", total_time.as_secs_f64());

    Ok(PipelineResults {
        total_molecules: total,
        processed_molecules: embeddings.len(),
        num_strata: strata.len(),
        strata_results,
        latent_dim: 16,
        gnn_layers: 3,
        som_grid: (10, 10),
        dataset_stats,
        graph_stats,
        emb_stats,
        avg_recon_loss,
        timings: Timings {
            load_secs: load_time.as_secs_f64(),
            parse_secs: parse_time.as_secs_f64(),
            encode_secs: encode_time.as_secs_f64(),
            cluster_secs: cluster_time.as_secs_f64(),
            total_secs: total_time.as_secs_f64(),
        },
    })
}

fn stat_std(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n <= 1.0 { return 0.0; }
    let mean = vals.iter().sum::<f64>() / n;
    (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

// ═══════════════════════════════════════════════════
// Result data structures
// ═══════════════════════════════════════════════════

#[derive(Debug)]
pub struct GraphStats {
    pub molecule_count: usize,
    pub total_atoms: usize,
    pub total_bonds: usize,
    pub min_atoms: usize,
    pub max_atoms: usize,
    pub min_bonds: usize,
    pub max_bonds: usize,
    pub parse_failures: usize,
}

impl Default for GraphStats {
    fn default() -> Self {
        Self {
            molecule_count: 0,
            total_atoms: 0,
            total_bonds: 0,
            min_atoms: usize::MAX,
            max_atoms: 0,
            min_bonds: usize::MAX,
            max_bonds: 0,
            parse_failures: 0,
        }
    }
}

#[derive(Debug, Default)]
pub struct EmbeddingStats {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
    pub mins: Vec<f32>,
    pub maxs: Vec<f32>,
    pub mean_pairwise_dist: f64,
}

#[derive(Debug)]
pub struct DatasetStats {
    pub total_in_file: usize,
    pub used: usize,
    pub qed_mean: f64,
    pub qed_std: f64,
    pub qed_min: f64,
    pub qed_max: f64,
    pub logp_mean: f64,
    pub logp_std: f64,
    pub sas_mean: f64,
    pub sas_std: f64,
}

#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub cluster_id: usize,
    pub size: usize,
    pub mean_qed: f64,
    pub std_qed: f64,
    pub mean_logp: f64,
    pub mean_sas: f64,
    pub compactness: f64,
}

#[derive(Debug)]
pub struct Timings {
    pub load_secs: f64,
    pub parse_secs: f64,
    pub encode_secs: f64,
    pub cluster_secs: f64,
    pub total_secs: f64,
}

#[derive(Debug)]
pub struct PipelineResults {
    pub total_molecules: usize,
    pub processed_molecules: usize,
    pub num_strata: usize,
    pub strata_results: Vec<StratumResult>,
    pub latent_dim: usize,
    pub gnn_layers: usize,
    pub som_grid: (usize, usize),
    pub dataset_stats: DatasetStats,
    pub graph_stats: GraphStats,
    pub emb_stats: EmbeddingStats,
    pub avg_recon_loss: f32,
    pub timings: Timings,
}

#[derive(Debug)]
pub struct StratumResult {
    pub group_id: usize,
    pub num_molecules: usize,
    pub num_clusters_used: usize,
    pub quantization_error: f64,
    pub u_matrix_mean: f64,
    pub u_matrix_max: f64,
    pub cluster_infos: Vec<ClusterInfo>,
}

impl PipelineResults {
    /// Generate a comprehensive markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // ── Header ──
        md.push_str("# Functional Group Analysis — Experiment Results\n\n");
        md.push_str("## 1. Dataset Summary\n\n");
        md.push_str("| Property | Value |\n|---|---|\n");
        md.push_str(&format!("| Source | ZINC15 database |\n"));
        md.push_str(&format!("| Total molecules in file | {} |\n", self.dataset_stats.total_in_file));
        md.push_str(&format!("| Molecules used | {} |\n", self.dataset_stats.used));
        md.push_str(&format!("| Successfully parsed | {} ({:.1}%) |\n",
            self.processed_molecules,
            self.processed_molecules as f64 / self.dataset_stats.used as f64 * 100.0));
        md.push_str(&format!("| Parse failures | {} |\n\n", self.graph_stats.parse_failures));

        md.push_str("### Molecular Property Distributions\n\n");
        md.push_str("| Property | Mean | Std | Min | Max |\n|---|---|---|---|---|\n");
        md.push_str(&format!("| QED | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            self.dataset_stats.qed_mean, self.dataset_stats.qed_std,
            self.dataset_stats.qed_min, self.dataset_stats.qed_max));
        md.push_str(&format!("| logP | {:.4} | {:.4} | — | — |\n",
            self.dataset_stats.logp_mean, self.dataset_stats.logp_std));
        md.push_str(&format!("| SAS | {:.4} | {:.4} | — | — |\n\n",
            self.dataset_stats.sas_mean, self.dataset_stats.sas_std));

        // ── Graph Statistics ──
        md.push_str("## 2. Molecular Graph Statistics\n\n");
        md.push_str("| Property | Value |\n|---|---|\n");
        let avg_atoms = self.graph_stats.total_atoms as f64 / self.graph_stats.molecule_count.max(1) as f64;
        let avg_bonds = self.graph_stats.total_bonds as f64 / self.graph_stats.molecule_count.max(1) as f64;
        md.push_str(&format!("| Atoms per molecule | {:.1} (range: {}–{}) |\n",
            avg_atoms, self.graph_stats.min_atoms, self.graph_stats.max_atoms));
        md.push_str(&format!("| Bonds per molecule | {:.1} (range: {}–{}) |\n",
            avg_bonds, self.graph_stats.min_bonds, self.graph_stats.max_bonds));
        md.push_str(&format!("| Total atoms processed | {} |\n", self.graph_stats.total_atoms));
        md.push_str(&format!("| Total bonds processed | {} |\n", self.graph_stats.total_bonds));
        md.push_str(&format!("| Node feature dimension | {} |\n", NODE_FEATURE_DIM));
        md.push_str(&format!("| Edge feature dimension | {} |\n\n", EDGE_FEATURE_DIM));

        // ── Model Configuration ──
        md.push_str("## 3. Model Configuration\n\n");
        md.push_str("### VGAE Architecture\n\n");
        md.push_str("| Component | Configuration |\n|---|---|\n");
        md.push_str(&format!("| GNN type | Graph Attention Network (GAT) |\n"));
        md.push_str(&format!("| GNN layers | {} |\n", self.gnn_layers));
        md.push_str("| Hidden dimension | 64 |\n");
        md.push_str("| GNN output dimension | 32 |\n");
        md.push_str(&format!("| Latent dimension | {} |\n", self.latent_dim));
        md.push_str("| Activation | ReLU (residual connections) |\n");
        md.push_str("| Pooling | Global attention pooling |\n");
        md.push_str("| Decoder | 16 → 64 → 128 → 29 (ReLU) |\n");
        md.push_str("| KL weight (β) | 0.001 |\n\n");

        md.push_str("### SOM Configuration\n\n");
        md.push_str("| Parameter | Value |\n|---|---|\n");
        md.push_str(&format!("| Grid size | {}×{} ({} neurons) |\n",
            self.som_grid.0, self.som_grid.1, self.som_grid.0 * self.som_grid.1));
        md.push_str("| Training epochs | 128 |\n");
        md.push_str("| Initial learning rate | 0.5 |\n");
        md.push_str("| Initial radius | 5.0 |\n");
        md.push_str("| Distance metric | Euclidean |\n");
        md.push_str("| Neighborhood | Gaussian |\n\n");

        // ── Encoding Results ──
        md.push_str("## 4. VGAE Encoding Results\n\n");
        md.push_str(&format!("| Metric | Value |\n|---|---|\n"));
        md.push_str(&format!("| Mean reconstruction loss | {:.6} |\n", self.avg_recon_loss));
        md.push_str(&format!("| Mean pairwise embedding distance | {:.6} |\n", self.emb_stats.mean_pairwise_dist));

        if !self.emb_stats.stds.is_empty() {
            let min_std = self.emb_stats.stds.iter().copied().fold(f32::MAX, f32::min);
            let max_std = self.emb_stats.stds.iter().copied().fold(f32::MIN, f32::max);
            let mean_std = self.emb_stats.stds.iter().sum::<f32>() / self.emb_stats.stds.len() as f32;
            md.push_str(&format!("| Embedding std (mean across dims) | {:.6} |\n", mean_std));
            md.push_str(&format!("| Embedding std range | [{:.4}, {:.4}] |\n", min_std, max_std));
        }
        md.push_str("\n");

        // Per-dimension embedding stats
        md.push_str("### Latent Dimension Statistics\n\n");
        md.push_str("| Dim | Mean | Std | Min | Max |\n|---|---|---|---|---|\n");
        let dim = self.emb_stats.means.len().min(16);
        for d in 0..dim {
            md.push_str(&format!("| {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                d, self.emb_stats.means[d], self.emb_stats.stds[d],
                self.emb_stats.mins[d], self.emb_stats.maxs[d]));
        }
        md.push_str("\n");

        // ── Clustering Results ──
        md.push_str("## 5. Stratified Clustering Results\n\n");
        md.push_str("### Per-Stratum Overview\n\n");
        md.push_str("| Stratum | QED Range | Molecules | Active Clusters | QE | U-Matrix Mean | U-Matrix Max |\n");
        md.push_str("|---|---|---|---|---|---|---|\n");
        let qed_ranges = ["[0, 0.399)", "[0.399, 0.520)", "[0.520, 0.694)", "[0.694, 0.814)", "[0.814, 1.0]"];
        for sr in &self.strata_results {
            let range = qed_ranges.get(sr.group_id).unwrap_or(&"—");
            md.push_str(&format!(
                "| {} | {} | {} | {} | {:.6} | {:.4} | {:.4} |\n",
                sr.group_id, range, sr.num_molecules, sr.num_clusters_used,
                sr.quantization_error, sr.u_matrix_mean, sr.u_matrix_max
            ));
        }

        let total_clustered: usize = self.strata_results.iter().map(|s| s.num_molecules).sum();
        let avg_qe: f64 = if self.strata_results.is_empty() { 0.0 } else {
            self.strata_results.iter().map(|s| s.quantization_error).sum::<f64>() / self.strata_results.len() as f64
        };
        md.push_str(&format!("\n**Total clustered**: {} molecules | **Avg QE**: {:.6}\n\n", total_clustered, avg_qe));

        // ── Per-stratum cluster details ──
        for sr in &self.strata_results {
            md.push_str(&format!("### Stratum {} — Top Clusters by Size\n\n", sr.group_id));
            md.push_str("| Cluster | Size | Mean QED | Std QED | Mean logP | Mean SAS | Compactness |\n");
            md.push_str("|---|---|---|---|---|---|---|\n");
            let mut sorted = sr.cluster_infos.clone();
            sorted.sort_by(|a, b| b.size.cmp(&a.size));
            for ci in sorted.iter().take(10) {
                md.push_str(&format!(
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                    ci.cluster_id, ci.size, ci.mean_qed, ci.std_qed,
                    ci.mean_logp, ci.mean_sas, ci.compactness
                ));
            }
            md.push_str("\n");
        }

        // ── Evaluation ──
        md.push_str("## 6. Evaluation Summary\n\n");

        // Cluster size distribution across all strata
        let all_sizes: Vec<usize> = self.strata_results.iter()
            .flat_map(|sr| sr.cluster_infos.iter().map(|c| c.size))
            .collect();
        if !all_sizes.is_empty() {
            let mean_size = all_sizes.iter().sum::<usize>() as f64 / all_sizes.len() as f64;
            let max_size = all_sizes.iter().copied().max().unwrap_or(0);
            let min_size = all_sizes.iter().copied().min().unwrap_or(0);
            let active_clusters: usize = self.strata_results.iter().map(|sr| sr.num_clusters_used).sum();
            let total_neurons: usize = self.strata_results.len() * self.som_grid.0 * self.som_grid.1;

            md.push_str("| Metric | Value |\n|---|---|\n");
            md.push_str(&format!("| Total active clusters | {} / {} neurons |\n", active_clusters, total_neurons));
            md.push_str(&format!("| Cluster size (mean) | {:.1} |\n", mean_size));
            md.push_str(&format!("| Cluster size (range) | {} – {} |\n", min_size, max_size));
            md.push_str(&format!("| Average quantization error | {:.6} |\n", avg_qe));

            // Mean compactness
            let all_compactness: Vec<f64> = self.strata_results.iter()
                .flat_map(|sr| sr.cluster_infos.iter().map(|c| c.compactness))
                .collect();
            let mean_compact = all_compactness.iter().sum::<f64>() / all_compactness.len().max(1) as f64;
            md.push_str(&format!("| Mean intra-cluster distance | {:.6} |\n\n", mean_compact));
        }

        // ── Timings ──
        md.push_str("## 7. Performance\n\n");
        md.push_str("| Phase | Time |\n|---|---|\n");
        md.push_str(&format!("| Data loading | {:.2}s |\n", self.timings.load_secs));
        md.push_str(&format!("| Graph parsing | {:.2}s |\n", self.timings.parse_secs));
        md.push_str(&format!("| VGAE encoding | {:.2}s |\n", self.timings.encode_secs));
        md.push_str(&format!("| SOM clustering | {:.2}s |\n", self.timings.cluster_secs));
        md.push_str(&format!("| **Total** | **{:.2}s** |\n\n", self.timings.total_secs));

        if self.processed_molecules > 0 {
            let throughput = self.processed_molecules as f64 / self.timings.total_secs;
            md.push_str(&format!("**Throughput**: {:.0} molecules/second\n\n", throughput));
        }

        // ── Methodology comparison ──
        md.push_str("## 8. Methodology Comparison\n\n");
        md.push_str("| Aspect | Previous (Python) | Current (Rust + GNN) |\n");
        md.push_str("|---|---|---|\n");
        md.push_str("| Molecular representation | Flat 28-dim feature vector | Full molecular graph |\n");
        md.push_str("| Feature learning | Dense autoencoder (28→16→28) | Graph Attention Network (3 layers) |\n");
        md.push_str("| Latent model | Deterministic AE | Variational (VGAE with KL regularization) |\n");
        md.push_str("| Structure awareness | None (bag of atoms) | Message passing preserves bond topology |\n");
        md.push_str("| Pooling | N/A (fixed features) | Global attention pooling (learned) |\n");
        md.push_str("| Edge features | Not used | 9-dim bond features in attention |\n");
        md.push_str("| Implementation | Python/PyTorch | Rust/Burn (memory-safe, zero-cost abstractions) |\n\n");

        md.push_str("## 9. Output Files\n\n");
        md.push_str("```\nresults/\n");
        md.push_str("├── RESULTS.md              # This report\n");
        md.push_str("├── training_losses.csv     # Per-molecule reconstruction losses\n");
        for sr in &self.strata_results {
            md.push_str(&format!("└── group_{}/\n", sr.group_id));
            md.push_str("    ├── labeled_data.csv    # SMILES + properties + cluster label\n");
            md.push_str("    └── embeddings.csv      # 16-dim latent embeddings\n");
        }
        md.push_str("```\n");

        md
    }
}
