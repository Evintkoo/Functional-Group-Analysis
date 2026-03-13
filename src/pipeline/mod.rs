/// Full analysis pipeline: CSV → SMILES → Graph → FG detect → GNN encode → SOM cluster → analysis → results.
/// Supports save/load of pipeline state for resumption and autotune for optimal SOM grid sizing.

use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use serde::{Serialize, Deserialize};
use std::time::Instant;
use std::collections::HashMap;

use crate::autoencoder::{Vgae, VgaeConfig, TrainConfig, vgae_loss};
use crate::features::{self, MolecularFeatures, NODE_FEATURE_DIM, EDGE_FEATURE_DIM};
use crate::functional_groups::{self, FunctionalGroup, FGProfile, FGCensus, fg_enrichment};
use crate::io::{self, MoleculeRecord};
use crate::smiles;
use crate::som::{self, Som, SomConfig, AutotuneResult};
use crate::stats;
use crate::visualization::{self, VisualizationData};

/// GPU-accelerated backend via Metal (macOS) / Vulkan (Linux) / DX12 (Windows).
type B = Wgpu;
/// Training backend with automatic differentiation on GPU.
type TrainB = Autodiff<Wgpu>;

/// Flush stderr to ensure log output is visible in non-TTY modes (pipes, redirection).
fn flush_log() {
    use std::io::Write;
    let _ = std::io::stderr().flush();
}

// ═══════════════════════════════════════════════════
// Pipeline state — save/load for resumption
// ═══════════════════════════════════════════════════

/// Serializable pipeline state snapshot. Saved after encoding + clustering to allow
/// resuming from a checkpoint without re-running the expensive VGAE encoding phase.
#[derive(Serialize, Deserialize)]
pub struct PipelineState {
    pub version: String,
    pub csv_path: String,
    pub total_molecules: usize,
    pub processed_molecules: usize,
    pub embeddings: Vec<Vec<f32>>,
    pub valid_indices: Vec<usize>,
    pub recon_losses: Vec<f32>,
    pub labels: Vec<usize>,
    pub som_states: Vec<SomState>,
    pub autotune_results: Vec<AutotuneResult>,
    pub phase_completed: u8,
}

/// Per-stratum SOM state for serialization.
#[derive(Clone, Serialize, Deserialize)]
pub struct SomState {
    pub stratum_id: usize,
    pub som: Som,
    pub config: SomConfig,
    pub best_grid: (usize, usize),
}

impl PipelineState {
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, &json)?;
        log::info!("Pipeline state saved to {} ({:.1} MB)", path, json.len() as f64 / 1_048_576.0);
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let state: PipelineState = serde_json::from_str(&json)?;
        log::info!("Pipeline state loaded from {} (phase {}, {} molecules, {} embeddings)",
            path, state.phase_completed, state.total_molecules, state.embeddings.len());
        Ok(state)
    }
}

// ═══════════════════════════════════════════════════
// Per-phase checkpoints for resumption
// ═══════════════════════════════════════════════════

#[derive(Serialize, Deserialize)]
struct Phase2Checkpoint {
    embeddings: Vec<Vec<f32>>,
    valid_indices: Vec<usize>,
    recon_losses: Vec<f32>,
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
}

fn save_phase_checkpoint(output_dir: &str, phase: u8, data: &impl serde::Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let dir = format!("{}/checkpoints", output_dir);
    std::fs::create_dir_all(&dir)?;
    let path = format!("{}/phase_{}.json", dir, phase);
    let json = serde_json::to_string(data)?;
    std::fs::write(&path, &json)?;
    log::info!("Checkpoint saved: {} ({:.1} MB)", path, json.len() as f64 / 1_048_576.0);
    Ok(())
}

fn load_phase_checkpoint<T: serde::de::DeserializeOwned>(output_dir: &str, phase: u8) -> Option<T> {
    let path = format!("{}/checkpoints/phase_{}.json", output_dir, phase);
    match std::fs::read_to_string(&path) {
        Ok(json) => match serde_json::from_str(&json) {
            Ok(data) => {
                log::info!("Loaded checkpoint: {}", path);
                Some(data)
            }
            Err(e) => {
                log::warn!("Failed to parse checkpoint {}: {}", path, e);
                None
            }
        }
        Err(_) => None,
    }
}

/// Convert MolecularFeatures into Burn tensors for inference.
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

/// Convert MolecularFeatures into Burn tensors for training (Autodiff backend).
fn features_to_train_tensors(
    feats: &MolecularFeatures,
    device: &<TrainB as Backend>::Device,
) -> (Tensor<TrainB, 2>, Tensor<TrainB, 2>) {
    let node_data: Vec<f32> = feats.node_features.iter().flatten().copied().collect();
    let node_tensor = Tensor::<TrainB, 1>::from_floats(
        node_data.as_slice(), device,
    ).reshape([feats.num_atoms, NODE_FEATURE_DIM]);

    let edge_data: Vec<f32> = feats.edge_features.iter().flatten().copied().collect();
    let num_edge_entries = feats.edge_features.len().max(1);
    let edge_tensor = if edge_data.is_empty() {
        Tensor::zeros([1, EDGE_FEATURE_DIM], device)
    } else {
        Tensor::<TrainB, 1>::from_floats(
            edge_data.as_slice(), device,
        ).reshape([num_edge_entries, EDGE_FEATURE_DIM])
    };

    (node_tensor, edge_tensor)
}

/// Process SMILES strings into molecular features + FG profiles.
fn process_molecules(records: &[&MoleculeRecord]) -> (Vec<(MolecularFeatures, usize, FGProfile)>, GraphStats) {
    let mut results = Vec::new();
    let mut stats = GraphStats::default();
    let mut parse_failures = 0usize;

    for (i, rec) in records.iter().enumerate() {
        match smiles::parse_smiles(&rec.smiles) {
            Ok(graph) => {
                let feats = features::extract_features(&graph);
                if feats.num_atoms > 0 {
                    let fg_profile = functional_groups::detect_functional_groups(&graph);

                    stats.total_atoms += feats.num_atoms;
                    stats.total_bonds += feats.num_bonds;
                    stats.min_atoms = stats.min_atoms.min(feats.num_atoms);
                    stats.max_atoms = stats.max_atoms.max(feats.num_atoms);
                    stats.min_bonds = stats.min_bonds.min(feats.num_bonds);
                    stats.max_bonds = stats.max_bonds.max(feats.num_bonds);
                    stats.per_mol_atoms.push(feats.num_atoms);
                    stats.per_mol_bonds.push(feats.num_bonds);
                    stats.molecule_count += 1;
                    results.push((feats, i, fg_profile));
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

/// Compute cluster-level statistics including FG profiles.
fn cluster_stats(
    labels: &[usize],
    embeddings: &[Vec<f32>],
    records: &[&MoleculeRecord],
    fg_profiles: &[FGProfile],
    population_census: &FGCensus,
) -> Vec<ClusterInfo> {
    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        cluster_map.entry(label).or_default().push(i);
    }

    let mut infos: Vec<ClusterInfo> = cluster_map.iter().map(|(&cluster_id, members)| {
        let size = members.len();

        let qed_vals: Vec<f64> = members.iter()
            .filter_map(|&i| records.get(i).map(|r| r.qed))
            .collect();
        let mean_qed = qed_vals.iter().sum::<f64>() / qed_vals.len().max(1) as f64;
        let std_qed = if qed_vals.len() > 1 {
            (qed_vals.iter().map(|q| (q - mean_qed).powi(2)).sum::<f64>() / (qed_vals.len() - 1) as f64).sqrt()
        } else { 0.0 };

        let logp_vals: Vec<f64> = members.iter()
            .filter_map(|&i| records.get(i).map(|r| r.log_p))
            .collect();
        let mean_logp = logp_vals.iter().sum::<f64>() / logp_vals.len().max(1) as f64;

        let sas_vals: Vec<f64> = members.iter()
            .filter_map(|&i| records.get(i).map(|r| r.sas))
            .collect();
        let mean_sas = sas_vals.iter().sum::<f64>() / sas_vals.len().max(1) as f64;

        // Intra-cluster compactness
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

        // FG census for this cluster
        let cluster_fg_profiles: Vec<FGProfile> = members.iter()
            .filter_map(|&i| fg_profiles.get(i).cloned())
            .collect();
        let cluster_census = FGCensus::from_profiles(&cluster_fg_profiles);
        let enrichment = fg_enrichment(&cluster_census, population_census);

        // --- Statistical enrichment with Fisher's exact test + BH-FDR ---
        let cluster_fg_counts: HashMap<String, usize> = FunctionalGroup::ALL.iter()
            .map(|&fg| (fg.name().to_string(), *cluster_census.prevalence.get(&fg).unwrap_or(&0)))
            .collect();
        let pop_fg_counts: HashMap<String, usize> = FunctionalGroup::ALL.iter()
            .map(|&fg| (fg.name().to_string(), *population_census.prevalence.get(&fg).unwrap_or(&0)))
            .collect();
        let enrichment_results = stats::enrichment_with_significance(
            &cluster_fg_counts,
            size,
            &pop_fg_counts,
            population_census.num_molecules,
            0.05, // FDR alpha
        );

        // Top signature FGs (enrichment > 1.2 AND FDR-significant)
        let signature_fgs: Vec<(FunctionalGroup, f64)> = enrichment.iter()
            .filter(|(fg, e)| {
                *e > 1.2 && enrichment_results.iter()
                    .any(|er| er.fg_name == fg.name() && er.significant)
            })
            .take(5)
            .cloned()
            .collect();

        // Dominant FG (most prevalent in this cluster)
        let dominant_fg = cluster_census.sorted_by_prevalence()
            .first()
            .map(|(fg, _, _)| *fg);

        // Representative SMILES (closest to centroid)
        let representative_smiles = if !cluster_embs.is_empty() && !centroid.is_empty() {
            let closest_idx = cluster_embs.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da: f64 = a.iter().zip(centroid.iter()).map(|(x, c)| ((x - c) as f64).powi(2)).sum();
                    let db: f64 = b.iter().zip(centroid.iter()).map(|(x, c)| ((x - c) as f64).powi(2)).sum();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let member_idx = members[closest_idx];
            records.get(member_idx).map(|r| r.smiles.clone())
        } else {
            None
        };

        ClusterInfo {
            cluster_id,
            size,
            mean_qed,
            std_qed,
            mean_logp,
            mean_sas,
            compactness: mean_dist_to_centroid,
            centroid,
            cluster_census,
            signature_fgs,
            dominant_fg,
            representative_smiles,
            enrichment_results,
        }
    }).collect();

    infos.sort_by_key(|c| c.cluster_id);
    infos
}

/// Compute Pearson correlation between two vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 3.0 { return 0.0; }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len().min(y.len()) {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x < 1e-12 || var_y < 1e-12 { return 0.0; }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute importance analysis: latent dimension correlations with properties + FGs.
fn importance_analysis(
    embeddings: &[Vec<f32>],
    records: &[&MoleculeRecord],
    fg_profiles: &[FGProfile],
    valid_indices: &[usize],
) -> ImportanceAnalysis {
    let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);

    // Property vectors aligned with embeddings
    let qed_vals: Vec<f64> = valid_indices.iter()
        .filter_map(|&i| records.get(i).map(|r| r.qed))
        .collect();
    let logp_vals: Vec<f64> = valid_indices.iter()
        .filter_map(|&i| records.get(i).map(|r| r.log_p))
        .collect();
    let sas_vals: Vec<f64> = valid_indices.iter()
        .filter_map(|&i| records.get(i).map(|r| r.sas))
        .collect();

    // Per-dimension correlations with properties
    let mut dim_correlations = Vec::new();
    for d in 0..dim {
        let dim_vals: Vec<f64> = embeddings.iter().map(|e| e[d] as f64).collect();
        let qed_corr = pearson_correlation(&dim_vals, &qed_vals);
        let logp_corr = pearson_correlation(&dim_vals, &logp_vals);
        let sas_corr = pearson_correlation(&dim_vals, &sas_vals);
        let variance = embeddings.iter().map(|e| e[d] as f64).collect::<Vec<_>>();
        let var = stat_var(&variance);
        dim_correlations.push(DimCorrelation {
            dim: d,
            qed_corr,
            logp_corr,
            sas_corr,
            variance: var,
        });
    }

    // Per-dimension correlation with FG presence (point-biserial for binary presence)
    let mut fg_dim_correlations: Vec<FGDimCorrelation> = Vec::new();
    for &fg in FunctionalGroup::ALL {
        let fg_presence: Vec<f64> = fg_profiles.iter()
            .map(|p| if p.has(fg) { 1.0 } else { 0.0 })
            .collect();
        let prevalence = fg_presence.iter().sum::<f64>() / fg_presence.len() as f64;
        if prevalence < 0.01 || prevalence > 0.99 { continue; } // Skip trivial cases

        let mut best_dim = 0;
        let mut best_corr = 0.0f64;
        for d in 0..dim {
            let dim_vals: Vec<f64> = embeddings.iter().map(|e| e[d] as f64).collect();
            let corr = pearson_correlation(&dim_vals, &fg_presence).abs();
            if corr > best_corr {
                best_corr = corr;
                best_dim = d;
            }
        }

        fg_dim_correlations.push(FGDimCorrelation {
            fg,
            best_dim,
            best_corr,
            prevalence,
        });
    }
    fg_dim_correlations.sort_by(|a, b| b.best_corr.partial_cmp(&a.best_corr).unwrap_or(std::cmp::Ordering::Equal));

    // FG ↔ property correlations
    let mut fg_property_correlations: Vec<FGPropertyCorrelation> = Vec::new();
    for &fg in FunctionalGroup::ALL {
        let fg_presence: Vec<f64> = fg_profiles.iter()
            .map(|p| if p.has(fg) { 1.0 } else { 0.0 })
            .collect();
        let prevalence = fg_presence.iter().sum::<f64>() / fg_presence.len() as f64;
        if prevalence < 0.01 { continue; }

        let qed_corr = pearson_correlation(&fg_presence, &qed_vals);
        let logp_corr = pearson_correlation(&fg_presence, &logp_vals);
        let sas_corr = pearson_correlation(&fg_presence, &sas_vals);

        fg_property_correlations.push(FGPropertyCorrelation {
            fg,
            qed_corr,
            logp_corr,
            sas_corr,
        });
    }

    ImportanceAnalysis {
        dim_correlations,
        fg_dim_correlations,
        fg_property_correlations,
    }
}

/// Compute inter-cluster distance matrix (top pairs only).
fn inter_cluster_distances(cluster_infos: &[ClusterInfo]) -> Vec<ClusterDistancePair> {
    let mut pairs = Vec::new();

    for i in 0..cluster_infos.len() {
        for j in (i + 1)..cluster_infos.len() {
            if cluster_infos[i].centroid.is_empty() || cluster_infos[j].centroid.is_empty() {
                continue;
            }
            let dist: f64 = cluster_infos[i].centroid.iter()
                .zip(cluster_infos[j].centroid.iter())
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            pairs.push(ClusterDistancePair {
                cluster_a: cluster_infos[i].cluster_id,
                cluster_b: cluster_infos[j].cluster_id,
                distance: dist,
            });
        }
    }

    pairs.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

/// Compute simplified silhouette score for clustering quality.
/// For each sample: s(i) = (b(i) - a(i)) / max(a(i), b(i))
/// where a(i) = mean distance to same-cluster members, b(i) = min mean distance to other clusters.
/// Subsampled for performance on large datasets.
fn silhouette_score(
    embeddings: &[Vec<f32>],
    labels: &[usize],
) -> ClusterQuality {
    let n = embeddings.len();
    if n < 2 { return ClusterQuality::default(); }

    // Build cluster membership
    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        cluster_map.entry(label).or_default().push(i);
    }

    let num_clusters = cluster_map.len();
    if num_clusters < 2 {
        return ClusterQuality { silhouette_mean: 0.0, silhouette_std: 0.0, num_clusters, davies_bouldin: 0.0 };
    }

    // Subsample for speed — take up to 2000 samples
    let sample_size = n.min(2000);
    let step = n / sample_size;
    let sample_indices: Vec<usize> = (0..n).step_by(step.max(1)).take(sample_size).collect();

    let mut silhouettes = Vec::with_capacity(sample_size);

    for &i in &sample_indices {
        let my_label = labels[i];
        let my_cluster = match cluster_map.get(&my_label) {
            Some(c) => c,
            None => continue,
        };

        // a(i): mean distance to same-cluster members
        let a = if my_cluster.len() > 1 {
            let sum: f64 = my_cluster.iter()
                .filter(|&&j| j != i)
                .map(|&j| euclidean_dist(&embeddings[i], &embeddings[j]))
                .sum();
            sum / (my_cluster.len() - 1) as f64
        } else {
            0.0
        };

        // b(i): min mean distance to nearest other cluster
        let b = cluster_map.iter()
            .filter(|(&label, _)| label != my_label)
            .map(|(_, members)| {
                let sum: f64 = members.iter()
                    .map(|&j| euclidean_dist(&embeddings[i], &embeddings[j]))
                    .sum();
                sum / members.len().max(1) as f64
            })
            .fold(f64::MAX, f64::min);

        let s = if a.max(b) > 0.0 { (b - a) / a.max(b) } else { 0.0 };
        silhouettes.push(s);
    }

    let mean = silhouettes.iter().sum::<f64>() / silhouettes.len().max(1) as f64;
    let std = if silhouettes.len() > 1 {
        (silhouettes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (silhouettes.len() - 1) as f64).sqrt()
    } else { 0.0 };

    // Davies-Bouldin index: average over clusters of max (s_i + s_j) / d(c_i, c_j)
    let cluster_ids: Vec<usize> = cluster_map.keys().copied().collect();
    let mut db_sum = 0.0;
    for &ci in &cluster_ids {
        let ci_members = &cluster_map[&ci];
        let ci_centroid = centroid_of(ci_members, embeddings);
        let si: f64 = ci_members.iter()
            .map(|&j| euclidean_dist(&embeddings[j], &ci_centroid))
            .sum::<f64>() / ci_members.len().max(1) as f64;

        let mut max_ratio = 0.0f64;
        for &cj in &cluster_ids {
            if ci == cj { continue; }
            let cj_members = &cluster_map[&cj];
            let cj_centroid = centroid_of(cj_members, embeddings);
            let sj: f64 = cj_members.iter()
                .map(|&j| euclidean_dist(&embeddings[j], &cj_centroid))
                .sum::<f64>() / cj_members.len().max(1) as f64;

            let d = euclidean_dist_f32(&ci_centroid, &cj_centroid);
            if d > 1e-10 {
                max_ratio = max_ratio.max((si + sj) / d);
            }
        }
        db_sum += max_ratio;
    }
    let davies_bouldin = db_sum / cluster_ids.len().max(1) as f64;

    ClusterQuality { silhouette_mean: mean, silhouette_std: std, num_clusters, davies_bouldin }
}

fn euclidean_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn euclidean_dist_f32(a: &[f32], b: &[f32]) -> f64 {
    euclidean_dist(a, b)
}

fn centroid_of(indices: &[usize], embeddings: &[Vec<f32>]) -> Vec<f32> {
    if indices.is_empty() { return Vec::new(); }
    let dim = embeddings[indices[0]].len();
    let mut c = vec![0.0f32; dim];
    for &i in indices {
        for (d, v) in embeddings[i].iter().enumerate() {
            c[d] += v;
        }
    }
    let n = indices.len() as f32;
    c.iter_mut().for_each(|v| *v /= n);
    c
}

/// Compute cluster size distribution statistics.
fn cluster_size_distribution(cluster_infos: &[ClusterInfo]) -> ClusterSizeStats {
    let sizes: Vec<usize> = cluster_infos.iter().map(|c| c.size).collect();
    if sizes.is_empty() { return ClusterSizeStats::default(); }

    let mut sorted = sizes.clone();
    sorted.sort();

    let n = sorted.len();
    let mean = sorted.iter().sum::<usize>() as f64 / n as f64;
    let median = if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) as f64 / 2.0 } else { sorted[n/2] as f64 };
    let std = (sorted.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / n.max(1) as f64).sqrt();
    let min = sorted[0];
    let max = sorted[n - 1];
    let p25 = sorted[n / 4];
    let p75 = sorted[3 * n / 4];

    // Gini coefficient for size inequality
    let total: f64 = sorted.iter().sum::<usize>() as f64;
    let mut gini_sum = 0.0;
    for (i, &s) in sorted.iter().enumerate() {
        gini_sum += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * s as f64;
    }
    let gini = if total > 0.0 { gini_sum / (n as f64 * total) } else { 0.0 };

    // Count singleton and large clusters
    let singletons = sorted.iter().filter(|&&s| s == 1).count();
    let large = sorted.iter().filter(|&&s| s as f64 > mean * 3.0).count();

    ClusterSizeStats { mean, median, std, min, max, p25, p75, gini, singletons, large_clusters: large, total_clusters: n }
}

/// Run the complete analysis pipeline.
pub fn run_pipeline(csv_path: &str, output_dir: &str) -> Result<PipelineResults, Box<dyn std::error::Error>> {
    let pipeline_start = Instant::now();
    let device = WgpuDevice::BestAvailable;
    let train_config = TrainConfig::default();

    log::info!("Using GPU-accelerated backend (Metal/wgpu)");

    // ═══════════════════════════════════════════════════
    // Phase 0: Load data
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 0: Loading data");
    log::info!("════════════════════════════════════════════");
    flush_log();
    let t0 = Instant::now();
    let all_records = io::load_zinc_csv(csv_path)?;
    let total = all_records.len();
    let load_time = t0.elapsed();
    log::info!("Loaded {} molecules in {:.2}s", total, load_time.as_secs_f64());

    let max_molecules = total;
    let records = &all_records[..max_molecules];
    log::info!("Processing all {} molecules", max_molecules);

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

    // ═══════════════════════════════════════════════════
    // Phase 1: Parse SMILES, extract features + FG profiles
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 1: Molecular graph construction + FG detection");
    log::info!("════════════════════════════════════════════");
    flush_log();
    let t1 = Instant::now();
    let record_refs: Vec<&MoleculeRecord> = records.iter().collect();
    let (mol_features, graph_stats) = process_molecules(&record_refs);
    let parse_time = t1.elapsed();

    log::info!("Parsed {}/{} molecules in {:.2}s ({} failures)",
        mol_features.len(), max_molecules, parse_time.as_secs_f64(), graph_stats.parse_failures);
    log::info!("  Atoms per molecule: min={}, max={}, avg={:.1}",
        graph_stats.min_atoms, graph_stats.max_atoms,
        graph_stats.total_atoms as f64 / graph_stats.molecule_count.max(1) as f64);

    // Global FG census
    let all_fg_profiles: Vec<FGProfile> = mol_features.iter().map(|(_, _, fg)| fg.clone()).collect();
    let global_fg_census = FGCensus::from_profiles(&all_fg_profiles);

    log::info!("  Functional groups detected:");
    for (fg, count, pct) in global_fg_census.sorted_by_prevalence().iter().take(10) {
        log::info!("    {:25} {:5} molecules ({:.1}%)", fg.name(), count, pct);
    }

    // ═══════════════════════════════════════════════════
    // Phase 2: VGAE Training + Encoding
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 2: VGAE Training + Encoding");
    log::info!("════════════════════════════════════════════");
    flush_log();
    let t2 = Instant::now();

    // Try to load Phase 2 checkpoint
    let phase2_loaded = load_phase_checkpoint::<Phase2Checkpoint>(output_dir, 2);

    let (embeddings, valid_indices, recon_losses, train_losses_per_epoch, val_losses_per_epoch, phase2_was_cached) =
    if let Some(ckpt) = phase2_loaded {
        log::info!("  ✓ Phase 2 loaded from checkpoint ({} embeddings)", ckpt.embeddings.len());
        (ckpt.embeddings, ckpt.valid_indices, ckpt.recon_losses, ckpt.train_losses, ckpt.val_losses, true)
    } else {

    let vgae_config = VgaeConfig::new(NODE_FEATURE_DIM, EDGE_FEATURE_DIM)
        .with_hidden_dim(64)
        .with_gnn_output_dim(32)
        .with_latent_dim(16)
        .with_num_gnn_layers(3);

    // ── 2a: Train VGAE with Adam optimizer on Autodiff<Wgpu> ──
    // Train on a representative sample for efficiency, then encode all molecules.
    let train_sample_size = 2000.min(mol_features.len());
    log::info!("  Training VGAE: {} epochs, lr={}, kl_weight={}, sample={}",
        train_config.num_epochs, train_config.learning_rate, train_config.kl_weight, train_sample_size);

    let mut vgae_train: Vgae<TrainB> = vgae_config.init(&device);
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .init::<TrainB, Vgae<TrainB>>();

    // Shuffle indices and pick train/val samples
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut all_indices: Vec<usize> = (0..mol_features.len()).collect();
    all_indices.shuffle(&mut rng);

    let sample_indices = &all_indices[..train_sample_size];
    let n_val = (train_sample_size as f32 * train_config.val_split) as usize;
    let n_train = train_sample_size - n_val;
    let train_indices = &sample_indices[..n_train];
    let val_indices = &sample_indices[n_train..];

    log::info!("  Train: {} molecules, Val: {} molecules", n_train, n_val);

    let mut train_losses_per_epoch = Vec::new();
    let mut val_losses_per_epoch = Vec::new();

    for epoch in 0..train_config.num_epochs {
        // ── Training pass ──
        let mut epoch_train_loss = 0.0f32;
        let mut train_count = 0usize;

        for &idx in train_indices {
            let (feats, _, _) = &mol_features[idx];
            let (node_t, edge_t) = features_to_train_tensors(feats, &device);

            let output = vgae_train.forward(node_t.clone(), &feats.edge_index, edge_t, feats.num_atoms);
            let loss = vgae_loss(output.reconstructed, node_t, output.mu, output.log_var, train_config.kl_weight);

            let loss_val: f32 = loss.clone().inner().into_scalar();
            epoch_train_loss += loss_val;
            train_count += 1;

            // Backpropagation
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &vgae_train);
            vgae_train = optimizer.step(train_config.learning_rate, vgae_train, grads);
        }

        let avg_train_loss = epoch_train_loss / train_count.max(1) as f32;
        train_losses_per_epoch.push(avg_train_loss);

        // ── Validation pass (no gradients) ──
        let mut epoch_val_loss = 0.0f32;
        let mut val_count = 0usize;

        for &idx in val_indices {
            let (feats, _, _) = &mol_features[idx];
            let (node_t, edge_t) = features_to_train_tensors(feats, &device);

            let output = vgae_train.forward(node_t.clone(), &feats.edge_index, edge_t, feats.num_atoms);
            let loss = vgae_loss(output.reconstructed, node_t, output.mu, output.log_var, train_config.kl_weight);
            let loss_val: f32 = loss.inner().into_scalar();
            epoch_val_loss += loss_val;
            val_count += 1;
        }

        let avg_val_loss = epoch_val_loss / val_count.max(1) as f32;
        val_losses_per_epoch.push(avg_val_loss);

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            let elapsed = t2.elapsed().as_secs_f64();
            log::info!("  Epoch {:3}/{}: train_loss={:.6}, val_loss={:.6} ({:.1}s elapsed)",
                epoch + 1, train_config.num_epochs, avg_train_loss, avg_val_loss, elapsed);
            flush_log();
        }
    }

    let train_time = t2.elapsed();
    log::info!("  VGAE training complete in {:.2}s", train_time.as_secs_f64());
    flush_log();
    log::info!("  Final: train_loss={:.6}, val_loss={:.6}",
        train_losses_per_epoch.last().unwrap_or(&0.0),
        val_losses_per_epoch.last().unwrap_or(&0.0));

    // ── 2b: Encode all molecules using trained model (inference, no autodiff) ──
    log::info!("  Encoding all {} molecules with trained VGAE...", mol_features.len());
    let t_enc = Instant::now();

    // Strip autodiff wrapper for inference
    let vgae: Vgae<B> = vgae_train.valid();

    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut valid_indices: Vec<usize> = Vec::new();

    // Use embed() — only runs encoder + attention pool + fc_mu (skips decoder entirely)
    for (i, (feats, orig_idx, _)) in mol_features.iter().enumerate() {
        let (node_t, edge_t) = features_to_tensors(feats, &device);
        let mu = vgae.embed(node_t, &feats.edge_index, edge_t);
        let embedding: Vec<f32> = mu.to_data().to_vec().unwrap();

        embeddings.push(embedding);
        valid_indices.push(*orig_idx);

        if (i + 1) % 10000 == 0 {
            log::info!("  Encoded {}/{}", i + 1, mol_features.len());
            flush_log();
        }
    }

    // Sample reconstruction loss on a subset for diagnostics
    let sample_size = 1000.min(mol_features.len());
    let mut sample_indices: Vec<usize> = (0..mol_features.len()).collect();
    {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        sample_indices.partial_shuffle(&mut rng, sample_size);
    }
    sample_indices.truncate(sample_size);

    let mut recon_losses: Vec<f32> = Vec::new();
    for &si in &sample_indices {
        let (feats, _, _) = &mol_features[si];
        let (node_t, edge_t) = features_to_tensors(feats, &device);
        let output = vgae.forward(node_t.clone(), &feats.edge_index, edge_t, feats.num_atoms);
        let loss = vgae_loss(output.reconstructed, node_t, output.mu, output.log_var, train_config.kl_weight);
        recon_losses.push(loss.into_scalar());
    }

    log::info!("Encoding complete in {:.2}s (inference: {:.2}s)",
        t2.elapsed().as_secs_f64(), t_enc.elapsed().as_secs_f64());
    flush_log();

    (embeddings, valid_indices, recon_losses, train_losses_per_epoch, val_losses_per_epoch, false)
    }; // end Phase 2 checkpoint check

    let encode_time = t2.elapsed();
    let avg_recon_loss = recon_losses.iter().sum::<f32>() / recon_losses.len().max(1) as f32;
    let emb_stats = embedding_stats(&embeddings);

    log::info!("Phase 2 complete in {:.2}s", encode_time.as_secs_f64());
    log::info!("  Mean reconstruction loss: {:.6}", avg_recon_loss);

    // Save Phase 2 checkpoint (only if freshly computed)
    if !phase2_was_cached {
        let phase2_ckpt = Phase2Checkpoint {
            embeddings: embeddings.clone(),
            valid_indices: valid_indices.clone(),
            recon_losses: recon_losses.clone(),
            train_losses: train_losses_per_epoch.clone(),
            val_losses: val_losses_per_epoch.clone(),
        };
        let _ = save_phase_checkpoint(output_dir, 2, &phase2_ckpt);
        io::save_training_losses(output_dir, &train_losses_per_epoch, &val_losses_per_epoch)?;
    }

    // ═══════════════════════════════════════════════════
    // Phase 3: Importance Analysis
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 3: Importance Analysis");
    log::info!("════════════════════════════════════════════");
    flush_log();
    let t_imp = Instant::now();
    let importance = importance_analysis(&embeddings, &record_refs, &all_fg_profiles, &valid_indices);
    let importance_time = t_imp.elapsed();

    log::info!("Importance analysis complete in {:.2}s", importance_time.as_secs_f64());
    flush_log();
    log::info!("  Top latent dimension correlations with QED:");
    let mut sorted_dims = importance.dim_correlations.clone();
    sorted_dims.sort_by(|a, b| b.qed_corr.abs().partial_cmp(&a.qed_corr.abs()).unwrap_or(std::cmp::Ordering::Equal));
    for dc in sorted_dims.iter().take(5) {
        log::info!("    Dim {:2}: r(QED)={:+.4}, r(logP)={:+.4}, r(SAS)={:+.4}, var={:.6}",
            dc.dim, dc.qed_corr, dc.logp_corr, dc.sas_corr, dc.variance);
    }

    log::info!("  Top FG ↔ latent dimension correlations:");
    for fgdc in importance.fg_dim_correlations.iter().take(5) {
        log::info!("    {:25} → dim {:2} (|r|={:.4}, prev={:.1}%)",
            fgdc.fg.name(), fgdc.best_dim, fgdc.best_corr, fgdc.prevalence * 100.0);
    }

    // ═══════════════════════════════════════════════════
    // Phase 4: QED stratification + SOM clustering + FG analysis
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 4: Stratified SOM clustering + FG analysis");
    log::info!("════════════════════════════════════════════");
    flush_log();
    let t3 = Instant::now();

    let qed_edges = vec![0.399, 0.520, 0.694, 0.814];
    let strata = io::stratify_by_qed(records, &qed_edges);

    // Try to load Phase 4 from existing pipeline_state.json
    let phase4_state = {
        let state_path = format!("{}/pipeline_state.json", output_dir);
        match PipelineState::load(&state_path) {
            Ok(state) if state.phase_completed >= 4 => {
                log::info!("  ✓ Phase 4 loaded from checkpoint ({} labels, {} SOM states)",
                    state.labels.len(), state.som_states.len());
                Some(state)
            }
            _ => None,
        }
    };

    let (all_labels, strata_results, autotune_results, som_states) = if let Some(ref state) = phase4_state {
        // Restore labels and SOM states; re-derive strata_results using restored SOMs
        let mut all_labels = state.labels.clone();
        let mut strata_results = Vec::new();
        let autotune_results = state.autotune_results.clone();
        let som_states_restored = std::mem::take(&mut { state.som_states.clone() });

        for ss in &som_states_restored {
            let group_id = ss.stratum_id;
            let stratum_indices = &strata[group_id];
            let stratum_emb_indices: Vec<usize> = stratum_indices.iter()
                .filter_map(|&orig_idx| valid_indices.iter().position(|&vi| vi == orig_idx))
                .collect();

            if stratum_emb_indices.is_empty() { continue; }

            let stratum_embeddings: Vec<Vec<f32>> = stratum_emb_indices.iter()
                .map(|&i| embeddings[i].clone())
                .collect();

            log::info!("─── Stratum {} (restored) ───  {} molecules", group_id, stratum_embeddings.len());

            let labels: Vec<usize> = stratum_emb_indices.iter()
                .map(|&emb_idx| all_labels[emb_idx])
                .collect();
            let qe = ss.som.quantization_error(&stratum_embeddings);
            let u_matrix = ss.som.u_matrix();
            let u_vals: Vec<f64> = u_matrix.iter().flatten().copied().collect();
            let u_mean = u_vals.iter().sum::<f64>() / u_vals.len() as f64;
            let u_max = u_vals.iter().copied().fold(f64::MIN, f64::max);

            let stratum_fg_profiles: Vec<FGProfile> = stratum_emb_indices.iter()
                .map(|&i| all_fg_profiles[i].clone())
                .collect();
            let stratum_census = FGCensus::from_profiles(&stratum_fg_profiles);

            let stratum_records: Vec<&MoleculeRecord> = stratum_emb_indices.iter()
                .filter_map(|&emb_idx| {
                    let orig_idx = valid_indices[emb_idx];
                    records.get(orig_idx)
                })
                .collect();

            let cluster_infos = cluster_stats(&labels, &stratum_embeddings, &stratum_records, &stratum_fg_profiles, &stratum_census);
            let cluster_distances = inter_cluster_distances(&cluster_infos);

            let mut used_clusters: Vec<usize> = labels.clone();
            used_clusters.sort();
            used_clusters.dedup();

            let cluster_quality = silhouette_score(&stratum_embeddings, &labels);
            let size_stats = cluster_size_distribution(&cluster_infos);

            log::info!("  Active clusters: {}/{} | QE: {:.6}", used_clusters.len(), ss.best_grid.0 * ss.best_grid.1, qe);

            let s_qed: Vec<f64> = stratum_records.iter().map(|r| r.qed).collect();
            let s_logp: Vec<f64> = stratum_records.iter().map(|r| r.log_p).collect();
            let s_sas: Vec<f64> = stratum_records.iter().map(|r| r.sas).collect();
            let s_qed_mean = s_qed.iter().sum::<f64>() / s_qed.len().max(1) as f64;
            let s_logp_mean = s_logp.iter().sum::<f64>() / s_logp.len().max(1) as f64;
            let s_sas_mean = s_sas.iter().sum::<f64>() / s_sas.len().max(1) as f64;

            strata_results.push(StratumResult {
                group_id,
                num_molecules: stratum_embeddings.len(),
                num_clusters_used: used_clusters.len(),
                quantization_error: qe,
                u_matrix_mean: u_mean,
                u_matrix_max: u_max,
                cluster_infos,
                stratum_census,
                cluster_distances,
                cluster_quality,
                size_stats,
                u_matrix,
                stratum_property_stats: (s_qed_mean, stat_std(&s_qed), s_logp_mean, stat_std(&s_logp), s_sas_mean, stat_std(&s_sas)),
            });
        }

        (all_labels, strata_results, autotune_results, som_states_restored)
    } else {

    let mut all_labels = vec![0usize; embeddings.len()];
    let mut strata_results = Vec::new();
    let mut autotune_results: Vec<AutotuneResult> = Vec::new();
    let mut som_states: Vec<SomState> = Vec::new();

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

        // Autotune SOM grid size for this stratum
        log::info!("  Running SOM autotune...");
        let at_result = som::autotune(&stratum_embeddings, 16, stratum_embeddings.len() > 20000);
        let best_grid = at_result.best_grid;
        log::info!("  Autotune selected {}×{} grid", best_grid.0, best_grid.1);

        autotune_results.push(at_result.clone());

        // Train SOM with best configuration (full epochs)
        let som_config = at_result.best_config.clone();
        let mut som = Som::new(&som_config, &stratum_embeddings);
        som.train(&stratum_embeddings, &som_config);

        let labels = som.predict(&stratum_embeddings);
        let qe = som.quantization_error(&stratum_embeddings);
        let u_matrix = som.u_matrix();

        let u_vals: Vec<f64> = u_matrix.iter().flatten().copied().collect();
        let u_mean = u_vals.iter().sum::<f64>() / u_vals.len() as f64;
        let u_max = u_vals.iter().copied().fold(f64::MIN, f64::max);

        for (k, &emb_idx) in stratum_emb_indices.iter().enumerate() {
            all_labels[emb_idx] = labels[k];
        }

        // Save SOM state for serialization
        som_states.push(SomState {
            stratum_id: group_id,
            som: som.clone(),
            config: som_config.clone(),
            best_grid,
        });

        // FG profiles for this stratum
        let stratum_fg_profiles: Vec<FGProfile> = stratum_emb_indices.iter()
            .map(|&i| all_fg_profiles[i].clone())
            .collect();
        let stratum_census = FGCensus::from_profiles(&stratum_fg_profiles);

        let stratum_records: Vec<&MoleculeRecord> = stratum_emb_indices.iter()
            .filter_map(|&emb_idx| {
                let orig_idx = valid_indices[emb_idx];
                records.get(orig_idx)
            })
            .collect();

        let cluster_infos = cluster_stats(&labels, &stratum_embeddings, &stratum_records, &stratum_fg_profiles, &stratum_census);

        // Inter-cluster distances
        let cluster_distances = inter_cluster_distances(&cluster_infos);

        let mut used_clusters: Vec<usize> = labels.clone();
        used_clusters.sort();
        used_clusters.dedup();

        // Cluster quality metrics
        let cluster_quality = silhouette_score(&stratum_embeddings, &labels);
        let size_stats = cluster_size_distribution(&cluster_infos);

        log::info!("  Active clusters: {}/{} | QE: {:.6}", used_clusters.len(), best_grid.0 * best_grid.1, qe);
        log::info!("  Silhouette: {:.4} | Davies-Bouldin: {:.4}", cluster_quality.silhouette_mean, cluster_quality.davies_bouldin);
        log::info!("  Cluster sizes: mean={:.1}, median={:.0}, gini={:.3}", size_stats.mean, size_stats.median, size_stats.gini);

        // Show top clusters with their signature FGs
        let mut sorted_clusters = cluster_infos.clone();
        sorted_clusters.sort_by(|a, b| b.size.cmp(&a.size));
        for ci in sorted_clusters.iter().take(3) {
            let sig: String = ci.signature_fgs.iter().take(3)
                .map(|(fg, e)| format!("{}({:.1}x)", fg.short_name(), e))
                .collect::<Vec<_>>()
                .join(", ");
            log::info!("    Cluster {:3}: {:4} mols | QED={:.3} | Signature: {}",
                ci.cluster_id, ci.size, ci.mean_qed, if sig.is_empty() { "none".to_string() } else { sig });
        }

        // Save results
        let stratum_labels_for_save: Vec<usize> = stratum_emb_indices.iter()
            .map(|&emb_idx| all_labels[emb_idx])
            .collect();

        io::save_cluster_results(
            output_dir, group_id, &stratum_records, &stratum_labels_for_save, &stratum_embeddings,
        )?;

        // Compute stratum property stats
        let s_qed: Vec<f64> = stratum_records.iter().map(|r| r.qed).collect();
        let s_logp: Vec<f64> = stratum_records.iter().map(|r| r.log_p).collect();
        let s_sas: Vec<f64> = stratum_records.iter().map(|r| r.sas).collect();
        let s_qed_mean = s_qed.iter().sum::<f64>() / s_qed.len().max(1) as f64;
        let s_logp_mean = s_logp.iter().sum::<f64>() / s_logp.len().max(1) as f64;
        let s_sas_mean = s_sas.iter().sum::<f64>() / s_sas.len().max(1) as f64;

        strata_results.push(StratumResult {
            group_id,
            num_molecules: stratum_embeddings.len(),
            num_clusters_used: used_clusters.len(),
            quantization_error: qe,
            u_matrix_mean: u_mean,
            u_matrix_max: u_max,
            cluster_infos,
            stratum_census,
            cluster_distances,
            cluster_quality,
            size_stats,
            u_matrix,
            stratum_property_stats: (s_qed_mean, stat_std(&s_qed), s_logp_mean, stat_std(&s_logp), s_sas_mean, stat_std(&s_sas)),
        });
    }

    (all_labels, strata_results, autotune_results, som_states)
    }; // end Phase 4 checkpoint check

    let cluster_time = t3.elapsed();

    // Save pipeline state checkpoint after clustering (only if not loaded from checkpoint)
    if phase4_state.is_none() {
        let state_path = format!("{}/pipeline_state.json", output_dir);
        let pipeline_state = PipelineState {
            version: "1.0".to_string(),
            csv_path: csv_path.to_string(),
            total_molecules: total,
            processed_molecules: embeddings.len(),
            embeddings: embeddings.clone(),
            valid_indices: valid_indices.clone(),
            recon_losses: recon_losses.clone(),
            labels: all_labels.clone(),
            som_states,
            autotune_results: autotune_results.clone(),
            phase_completed: 4,
        };
        if let Err(e) = pipeline_state.save(&state_path) {
            log::warn!("Failed to save pipeline state: {}", e);
        }
    }

    // ═══════════════════════════════════════════════════
    // Phase 5: Generate visualizations
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 5: Generating visualizations");
    log::info!("════════════════════════════════════════════");
    flush_log();
    let t_viz = Instant::now();

    // Build stratum labels for each embedding
    let mut emb_stratum_labels = vec![0usize; embeddings.len()];
    for sr in &strata_results {
        for (orig_idx_ref, stratum_idx) in strata[sr.group_id].iter().zip(std::iter::repeat(sr.group_id)) {
            if let Some(emb_pos) = valid_indices.iter().position(|&vi| vi == *orig_idx_ref) {
                emb_stratum_labels[emb_pos] = stratum_idx;
            }
        }
    }

    // Build visualization data
    let viz_data = VisualizationData {
        qed_vals: qed_vals.clone(),
        logp_vals: logp_vals.clone(),
        sas_vals: sas_vals.clone(),
        atom_counts: graph_stats.per_mol_atoms.clone(),
        bond_counts: graph_stats.per_mol_bonds.clone(),
        embeddings: embeddings.clone(),
        stratum_labels: emb_stratum_labels,
        recon_losses: recon_losses.clone(),
        dim_correlations: importance.dim_correlations.iter()
            .map(|dc| (dc.dim, dc.qed_corr, dc.logp_corr, dc.sas_corr))
            .collect(),
        dim_variances: importance.dim_correlations.iter()
            .map(|dc| (dc.dim, dc.variance))
            .collect(),
        fg_prevalence: global_fg_census.sorted_by_prevalence().iter()
            .map(|(fg, _count, pct)| (fg.name().to_string(), *pct))
            .collect(),
        fg_property_corr: importance.fg_property_correlations.iter()
            .map(|fpc| (fpc.fg.name().to_string(), fpc.qed_corr, fpc.logp_corr, fpc.sas_corr))
            .collect(),
        strata_quality: strata_results.iter()
            .map(|sr| (sr.group_id, sr.cluster_quality.silhouette_mean, sr.cluster_quality.davies_bouldin, sr.quantization_error, sr.size_stats.gini))
            .collect(),
        strata_cluster_sizes: strata_results.iter()
            .map(|sr| sr.cluster_infos.iter().map(|c| c.size).collect())
            .collect(),
        strata_property_stats: strata_results.iter()
            .map(|sr| {
                let (mq, sq, ml, sl, ms, ss) = sr.stratum_property_stats;
                (sr.group_id, mq, sq, ml, sl, ms, ss)
            })
            .collect(),
        u_matrices: strata_results.iter().map(|sr| sr.u_matrix.clone()).collect(),
        strata_fg_enrichments: strata_results.iter().map(|sr| {
            let cluster_enrichments: Vec<(usize, Vec<(String, f64)>)> = sr.cluster_infos.iter()
                .take(20)  // top 20 clusters for readability
                .map(|ci| {
                    let fgs: Vec<(String, f64)> = ci.signature_fgs.iter()
                        .map(|(fg, enrichment)| (fg.short_name().to_string(), *enrichment))
                        .collect();
                    (ci.cluster_id, fgs)
                })
                .collect();
            (sr.group_id, cluster_enrichments)
        }).collect(),
        strata_distances: strata_results.iter().map(|sr| {
            let dists: Vec<(usize, usize, f64)> = sr.cluster_distances.iter()
                .map(|d| (d.cluster_a, d.cluster_b, d.distance))
                .collect();
            (sr.group_id, dists, sr.num_clusters_used)
        }).collect(),
    };

    let figures = visualization::generate_all_figures(&viz_data, output_dir);

    let viz_time = t_viz.elapsed();
    log::info!("Visualization complete in {:.2}s ({} figures)", viz_time.as_secs_f64(), figures.len());
    flush_log();

    let total_time = pipeline_start.elapsed();

    log::info!("════════════════════════════════════════════");
    log::info!("  Pipeline complete");
    log::info!("════════════════════════════════════════════");
    flush_log();
    log::info!("  Total: {:.2}s", total_time.as_secs_f64());
    flush_log();

    // ═══════════════════════════════════════════════════
    // Cross-stratum statistical analyses
    // ═══════════════════════════════════════════════════
    log::info!("Computing cross-stratum statistical tests...");
    flush_log();

    // Linear regression: SAS ~ QED and logP ~ QED across stratum means
    let stratum_mean_qed: Vec<f64> = strata_results.iter().map(|sr| sr.stratum_property_stats.0).collect();
    let stratum_mean_sas: Vec<f64> = strata_results.iter().map(|sr| sr.stratum_property_stats.4).collect();
    let stratum_mean_logp: Vec<f64> = strata_results.iter().map(|sr| sr.stratum_property_stats.2).collect();

    let sas_qed_regression = stats::linear_regression(&stratum_mean_qed, &stratum_mean_sas);
    let logp_qed_regression = stats::linear_regression(&stratum_mean_qed, &stratum_mean_logp);

    log::info!("  SAS ~ QED regression: R²={:.4}, slope={:.3}±{:.3}, p={:.4}",
        sas_qed_regression.r_squared, sas_qed_regression.slope, sas_qed_regression.slope_se, sas_qed_regression.p_value);
    log::info!("  logP ~ QED regression: R²={:.4}, slope={:.3}±{:.3}, p={:.4}",
        logp_qed_regression.r_squared, logp_qed_regression.slope, logp_qed_regression.slope_se, logp_qed_regression.p_value);

    // Chi-squared test: sulfonyl--heterocycle co-occurrence
    let sulfonyl_presence: Vec<bool> = all_fg_profiles.iter().map(|p| p.has(FunctionalGroup::Sulfonyl)).collect();
    let heterocycle_presence: Vec<bool> = all_fg_profiles.iter().map(|p| p.has(FunctionalGroup::Heterocycle)).collect();
    let sulfonyl_heterocycle_chi2 = if sulfonyl_presence.len() > 10 {
        Some(stats::cooccurrence_chi_squared(&sulfonyl_presence, &heterocycle_presence))
    } else { None };

    // Co-occurrence rate: P(heterocycle | sulfonyl)
    let sulfonyl_count = sulfonyl_presence.iter().filter(|&&b| b).count();
    let both_count = sulfonyl_presence.iter().zip(heterocycle_presence.iter())
        .filter(|(&s, &h)| s && h).count();
    let sulfonyl_heterocycle_cooccurrence = if sulfonyl_count > 0 {
        both_count as f64 / sulfonyl_count as f64
    } else { 0.0 };

    if let Some(ref chi2) = sulfonyl_heterocycle_chi2 {
        log::info!("  Sulfonyl-Heterocycle co-occurrence: χ²={:.1}, p={}, cooccurrence rate={:.1}%",
            chi2.chi2, stats::format_p_value(chi2.p_value), sulfonyl_heterocycle_cooccurrence * 100.0);
    }

    let cross_stratum_stats = CrossStratumStats {
        sas_qed_regression,
        logp_qed_regression,
        sulfonyl_heterocycle_chi2,
        sulfonyl_heterocycle_cooccurrence,
        fsp3_ttest: None, // computed per-cluster in analysis
    };

    Ok(PipelineResults {
        total_molecules: total,
        processed_molecules: embeddings.len(),
        num_strata: strata.len(),
        strata_results,
        latent_dim: 16,
        gnn_layers: 3,
        som_grid: autotune_results.first().map(|a| a.best_grid).unwrap_or((10, 10)),
        autotune_results,
        dataset_stats,
        graph_stats,
        emb_stats,
        avg_recon_loss,
        global_fg_census,
        importance,
        cross_stratum_stats,
        timings: Timings {
            load_secs: load_time.as_secs_f64(),
            parse_secs: parse_time.as_secs_f64(),
            encode_secs: encode_time.as_secs_f64(),
            importance_secs: importance_time.as_secs_f64(),
            cluster_secs: cluster_time.as_secs_f64(),
            total_secs: total_time.as_secs_f64(),
        },
        figures,
    })
}

fn stat_std(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n <= 1.0 { return 0.0; }
    let mean = vals.iter().sum::<f64>() / n;
    (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn stat_var(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n <= 1.0 { return 0.0; }
    let mean = vals.iter().sum::<f64>() / n;
    vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
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
    pub per_mol_atoms: Vec<usize>,
    pub per_mol_bonds: Vec<usize>,
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
            per_mol_atoms: Vec::new(),
            per_mol_bonds: Vec::new(),
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
    pub centroid: Vec<f32>,
    pub cluster_census: FGCensus,
    pub signature_fgs: Vec<(FunctionalGroup, f64)>,
    pub dominant_fg: Option<FunctionalGroup>,
    pub representative_smiles: Option<String>,
    /// Enrichment results with Fisher's exact test p-values and BH-FDR correction.
    pub enrichment_results: Vec<stats::EnrichmentResult>,
}

#[derive(Debug, Clone)]
pub struct DimCorrelation {
    pub dim: usize,
    pub qed_corr: f64,
    pub logp_corr: f64,
    pub sas_corr: f64,
    pub variance: f64,
}

#[derive(Debug, Clone)]
pub struct FGDimCorrelation {
    pub fg: FunctionalGroup,
    pub best_dim: usize,
    pub best_corr: f64,
    pub prevalence: f64,
}

#[derive(Debug, Clone)]
pub struct FGPropertyCorrelation {
    pub fg: FunctionalGroup,
    pub qed_corr: f64,
    pub logp_corr: f64,
    pub sas_corr: f64,
}

#[derive(Debug, Clone)]
pub struct ImportanceAnalysis {
    pub dim_correlations: Vec<DimCorrelation>,
    pub fg_dim_correlations: Vec<FGDimCorrelation>,
    pub fg_property_correlations: Vec<FGPropertyCorrelation>,
}

#[derive(Debug, Clone)]
pub struct ClusterDistancePair {
    pub cluster_a: usize,
    pub cluster_b: usize,
    pub distance: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ClusterQuality {
    pub silhouette_mean: f64,
    pub silhouette_std: f64,
    pub num_clusters: usize,
    pub davies_bouldin: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ClusterSizeStats {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub min: usize,
    pub max: usize,
    pub p25: usize,
    pub p75: usize,
    pub gini: f64,
    pub singletons: usize,
    pub large_clusters: usize,
    pub total_clusters: usize,
}

#[derive(Debug)]
pub struct Timings {
    pub load_secs: f64,
    pub parse_secs: f64,
    pub encode_secs: f64,
    pub importance_secs: f64,
    pub cluster_secs: f64,
    pub total_secs: f64,
}

/// Cross-stratum statistical analysis results.
#[derive(Debug)]
pub struct CrossStratumStats {
    /// Linear regression: SAS ~ QED across stratum means
    pub sas_qed_regression: stats::LinearRegressionResult,
    /// Linear regression: logP ~ QED across stratum means
    pub logp_qed_regression: stats::LinearRegressionResult,
    /// Chi-squared test: sulfonyl--heterocycle co-occurrence
    pub sulfonyl_heterocycle_chi2: Option<stats::ChiSquaredResult>,
    /// Co-occurrence rate: P(heterocycle | sulfonyl)
    pub sulfonyl_heterocycle_cooccurrence: f64,
    /// Welch's t-test: Fsp3 of phenyl-free cluster vs dataset
    pub fsp3_ttest: Option<stats::WelchTTestResult>,
}

pub struct PipelineResults {
    pub total_molecules: usize,
    pub processed_molecules: usize,
    pub num_strata: usize,
    pub strata_results: Vec<StratumResult>,
    pub latent_dim: usize,
    pub gnn_layers: usize,
    pub som_grid: (usize, usize),
    pub autotune_results: Vec<AutotuneResult>,
    pub dataset_stats: DatasetStats,
    pub graph_stats: GraphStats,
    pub emb_stats: EmbeddingStats,
    pub avg_recon_loss: f32,
    pub global_fg_census: FGCensus,
    pub importance: ImportanceAnalysis,
    pub cross_stratum_stats: CrossStratumStats,
    pub timings: Timings,
    pub figures: Vec<(String, String)>,
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
    pub stratum_census: FGCensus,
    pub cluster_distances: Vec<ClusterDistancePair>,
    pub cluster_quality: ClusterQuality,
    pub size_stats: ClusterSizeStats,
    pub u_matrix: Vec<Vec<f64>>,
    pub stratum_property_stats: (f64, f64, f64, f64, f64, f64),  // mean_qed, std_qed, mean_logp, std_logp, mean_sas, std_sas
}

// ═══════════════════════════════════════════════════
// Markdown report generation
// ═══════════════════════════════════════════════════

impl PipelineResults {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // ── 1. Dataset Summary ──
        md.push_str("# Functional Group Analysis — Experiment Results\n\n");
        md.push_str("## 1. Dataset Summary\n\n");
        md.push_str("| Property | Value |\n|---|---|\n");
        md.push_str("| Source | ZINC15 database |\n");
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

        md.push_str("![Property Distributions](figures/property_distributions_combined.svg)\n\n");
        md.push_str("*Figure 1: Distribution of QED, logP, and SAS across the full dataset. Red vertical lines indicate means.*\n\n");
        md.push_str("| | | |\n|---|---|---|\n");
        md.push_str("| ![QED](figures/qed_distribution.svg) | ![logP](figures/logp_distribution.svg) | ![SAS](figures/sas_distribution.svg) |\n\n");
        md.push_str("*Figure 2: Individual property distributions with 50-bin histograms and mean indicators.*\n\n");

        // ── 2. Molecular Graph Statistics ──
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

        md.push_str("![Molecular Complexity](figures/molecule_complexity.svg)\n\n");
        md.push_str("*Figure 3: Molecular graph complexity — atoms vs. bonds colored by QED score (red=low, green=high).*\n\n");

        // ── 3. Functional Group Census ──
        md.push_str("## 3. Functional Group Census\n\n");
        md.push_str("Detection of 22 functional group types across the entire dataset.\n\n");
        md.push_str("| Functional Group | Molecules | Prevalence (%) | Total Count | Mean per Mol |\n");
        md.push_str("|---|---|---|---|---|\n");
        for (fg, count, pct) in self.global_fg_census.sorted_by_prevalence() {
            let total = *self.global_fg_census.total_count.get(&fg).unwrap_or(&0);
            let mean = *self.global_fg_census.mean_count.get(&fg).unwrap_or(&0.0);
            md.push_str(&format!("| {} | {} | {:.1} | {} | {:.2} |\n",
                fg.name(), count, pct, total, mean));
        }
        md.push_str("\n");

        // FG co-occurrence summary
        md.push_str("### Functional Group Co-occurrence Patterns\n\n");
        md.push_str("Average number of distinct functional group types per molecule and distribution.\n\n");
        // We don't have per-molecule diversity stored, but we can note the census data
        let total_fg_types_found = self.global_fg_census.sorted_by_prevalence().len();
        md.push_str(&format!("- **Functional group types detected**: {} out of 22\n", total_fg_types_found));
        let highly_prevalent: Vec<_> = self.global_fg_census.sorted_by_prevalence().iter()
            .filter(|(_, _, pct)| *pct > 50.0)
            .map(|(fg, _, pct)| format!("{} ({:.0}%)", fg.short_name(), pct))
            .collect();
        if !highly_prevalent.is_empty() {
            md.push_str(&format!("- **Ubiquitous groups** (>50%): {}\n", highly_prevalent.join(", ")));
        }
        let rare: Vec<_> = self.global_fg_census.sorted_by_prevalence().iter()
            .filter(|(_, _, pct)| *pct > 0.0 && *pct < 5.0)
            .map(|(fg, _, pct)| format!("{} ({:.1}%)", fg.short_name(), pct))
            .collect();
        if !rare.is_empty() {
            md.push_str(&format!("- **Rare groups** (<5%): {}\n", rare.join(", ")));
        }
        md.push_str("\n");

        md.push_str("![FG Prevalence](figures/fg_prevalence.svg)\n\n");
        md.push_str("*Figure 4: Functional group prevalence across the dataset. Blue (>50%), green (10–50%), purple (<10%).*\n\n");

        // ── 4. Model Configuration ──
        md.push_str("## 4. Model Configuration\n\n");
        md.push_str("### VGAE Architecture\n\n");
        md.push_str("| Component | Configuration |\n|---|---|\n");
        md.push_str("| GNN type | Graph Attention Network (GAT) |\n");
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
        md.push_str("| Grid selection | **Autotune** (multi-candidate evaluation) |\n");
        md.push_str("| Training epochs | 128 (full), 30–50 (autotune eval) |\n");
        md.push_str("| Initial learning rate | 0.5 |\n");
        md.push_str("| Distance metric | Euclidean |\n");
        md.push_str("| Neighborhood | Gaussian |\n");
        md.push_str("| Scoring | 0.4×QE + 0.3×TE + 0.3×ActiveRatio |\n\n");

        if !self.autotune_results.is_empty() {
            md.push_str("### Autotune Results (per Stratum)\n\n");
            for (i, at) in self.autotune_results.iter().enumerate() {
                md.push_str(&format!("**Stratum {}** — Best grid: **{}×{}**\n\n", i, at.best_grid.0, at.best_grid.1));
                md.push_str("| Grid | Neurons | Active | QE | TE | Score |\n|---|---|---|---|---|---|\n");
                for c in &at.candidates {
                    md.push_str(&format!("| {}×{} | {} | {} | {:.4} | {:.4} | {:.4} |\n",
                        c.grid_width, c.grid_height, c.num_clusters, c.active_clusters,
                        c.quantization_error, c.topographic_error, c.combined_score));
                }
                md.push_str("\n");
            }
        }

        // ── 5. VGAE Encoding Results ──
        md.push_str("## 5. VGAE Encoding Results\n\n");
        md.push_str("| Metric | Value |\n|---|---|\n");
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

        md.push_str("### Latent Dimension Statistics\n\n");
        md.push_str("| Dim | Mean | Std | Min | Max |\n|---|---|---|---|---|\n");
        let dim = self.emb_stats.means.len().min(16);
        for d in 0..dim {
            md.push_str(&format!("| {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                d, self.emb_stats.means[d], self.emb_stats.stds[d],
                self.emb_stats.mins[d], self.emb_stats.maxs[d]));
        }
        md.push_str("\n");

        md.push_str("![Reconstruction Loss](figures/reconstruction_loss_dist.svg)\n\n");
        md.push_str("*Figure 5: Distribution of VGAE reconstruction losses across all molecules.*\n\n");
        md.push_str("![Embedding Variance](figures/embedding_dim_variance.svg)\n\n");
        md.push_str("*Figure 6: Variance of each latent dimension — higher variance indicates more discriminative dimensions.*\n\n");

        // ── 6. Feature Importance Analysis ──
        md.push_str("## 6. Feature Importance Analysis\n\n");

        md.push_str("### 6.1 Latent Dimension ↔ Property Correlations\n\n");
        md.push_str("Pearson correlation (r) between each latent dimension and molecular properties.\n");
        md.push_str("Dimensions sorted by |r(QED)|.\n\n");
        md.push_str("| Dim | Variance | r(QED) | r(logP) | r(SAS) |\n");
        md.push_str("|---|---|---|---|---|\n");
        let mut sorted_dims = self.importance.dim_correlations.clone();
        sorted_dims.sort_by(|a, b| b.qed_corr.abs().partial_cmp(&a.qed_corr.abs()).unwrap_or(std::cmp::Ordering::Equal));
        for dc in &sorted_dims {
            md.push_str(&format!("| {} | {:.6} | {:+.4} | {:+.4} | {:+.4} |\n",
                dc.dim, dc.variance, dc.qed_corr, dc.logp_corr, dc.sas_corr));
        }
        md.push_str("\n");

        md.push_str("### 6.2 Functional Group ↔ Latent Space Encoding\n\n");
        md.push_str("Which latent dimensions best encode each functional group's presence.\n\n");
        md.push_str("| Functional Group | Prevalence (%) | Best Dim | |r| |\n");
        md.push_str("|---|---|---|---|\n");
        for fgdc in &self.importance.fg_dim_correlations {
            md.push_str(&format!("| {} | {:.1} | {} | {:.4} |\n",
                fgdc.fg.name(), fgdc.prevalence * 100.0, fgdc.best_dim, fgdc.best_corr));
        }
        md.push_str("\n");

        md.push_str("### 6.3 Functional Group ↔ Molecular Property Correlations\n\n");
        md.push_str("Point-biserial correlation between FG presence and drug-likeness properties.\n\n");
        md.push_str("| Functional Group | r(QED) | r(logP) | r(SAS) |\n");
        md.push_str("|---|---|---|---|\n");
        let mut sorted_fg_props = self.importance.fg_property_correlations.clone();
        sorted_fg_props.sort_by(|a, b| {
            let max_a = a.qed_corr.abs().max(a.logp_corr.abs()).max(a.sas_corr.abs());
            let max_b = b.qed_corr.abs().max(b.logp_corr.abs()).max(b.sas_corr.abs());
            max_b.partial_cmp(&max_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        for fpc in &sorted_fg_props {
            md.push_str(&format!("| {} | {:+.4} | {:+.4} | {:+.4} |\n",
                fpc.fg.name(), fpc.qed_corr, fpc.logp_corr, fpc.sas_corr));
        }
        md.push_str("\n");

        md.push_str("![Dim-Property Heatmap](figures/dim_property_heatmap.svg)\n\n");
        md.push_str("*Figure 7: Heatmap of Pearson correlations between latent dimensions and molecular properties. Blue = negative, red = positive.*\n\n");
        md.push_str("![FG-Property Correlations](figures/fg_property_correlations.svg)\n\n");
        md.push_str("*Figure 8: Point-biserial correlations between functional group presence and drug-likeness properties.*\n\n");

        // ── 7. Stratified Clustering Results ──
        md.push_str("## 7. Stratified Clustering Results\n\n");
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

        md.push_str("![Latent Space UMAP](figures/latent_space_umap.svg)\n\n");
        md.push_str("*Figure 9: UMAP projection of 16-dimensional VGAE embeddings colored by QED stratum.*\n\n");
        md.push_str("![Stratum Properties](figures/stratum_property_comparison.svg)\n\n");
        md.push_str("*Figure 10: Mean ± std of molecular properties across QED strata.*\n\n");
        md.push_str("![U-Matrix](figures/umatrix_heatmaps.svg)\n\n");
        md.push_str("*Figure 11: SOM U-matrix heatmaps showing topological organization per stratum. Darker regions indicate cluster boundaries.*\n\n");

        // ── Per-stratum detailed analysis ──
        for sr in &self.strata_results {
            let range = qed_ranges.get(sr.group_id).unwrap_or(&"—");
            md.push_str(&format!("### Stratum {} ({}) — Detailed Analysis\n\n", sr.group_id, range));

            // Stratum FG census
            md.push_str("#### Functional Group Distribution\n\n");
            md.push_str("| Functional Group | Prevalence (%) | Mean Count |\n|---|---|---|\n");
            for (fg, _, pct) in sr.stratum_census.sorted_by_prevalence().iter().take(10) {
                let mean = *sr.stratum_census.mean_count.get(fg).unwrap_or(&0.0);
                md.push_str(&format!("| {} | {:.1} | {:.2} |\n", fg.name(), pct, mean));
            }
            md.push_str("\n");

            // Top clusters with characterization
            md.push_str("#### Top Clusters by Size\n\n");
            md.push_str("| Cluster | Size | QED μ±σ | logP | SAS | Compact | Dominant FG | Signature FGs | Representative |\n");
            md.push_str("|---|---|---|---|---|---|---|---|---|\n");
            let mut sorted = sr.cluster_infos.clone();
            sorted.sort_by(|a, b| b.size.cmp(&a.size));
            for ci in sorted.iter().take(10) {
                let dominant = ci.dominant_fg
                    .map(|fg| fg.short_name().to_string())
                    .unwrap_or_else(|| "—".to_string());
                let sig: String = ci.signature_fgs.iter().take(3)
                    .map(|(fg, e)| format!("{}({:.1}×)", fg.short_name(), e))
                    .collect::<Vec<_>>()
                    .join(", ");
                let repr = ci.representative_smiles.as_deref().unwrap_or("—");
                // Truncate long SMILES
                let repr_short = if repr.len() > 30 { &repr[..30] } else { repr };
                md.push_str(&format!(
                    "| {} | {} | {:.3}±{:.3} | {:.2} | {:.2} | {:.4} | {} | {} | `{}` |\n",
                    ci.cluster_id, ci.size, ci.mean_qed, ci.std_qed,
                    ci.mean_logp, ci.mean_sas, ci.compactness,
                    dominant,
                    if sig.is_empty() { "—".to_string() } else { sig },
                    repr_short
                ));
            }
            md.push_str("\n");

            // Inter-cluster distances
            if !sr.cluster_distances.is_empty() {
                md.push_str("#### Inter-Cluster Distance Analysis\n\n");

                // Closest pairs
                md.push_str("**Most similar cluster pairs** (smallest embedding distance):\n\n");
                md.push_str("| Cluster A | Cluster B | Distance |\n|---|---|---|\n");
                for pair in sr.cluster_distances.iter().take(5) {
                    md.push_str(&format!("| {} | {} | {:.6} |\n",
                        pair.cluster_a, pair.cluster_b, pair.distance));
                }
                md.push_str("\n");

                // Most distant pairs
                md.push_str("**Most distant cluster pairs**:\n\n");
                md.push_str("| Cluster A | Cluster B | Distance |\n|---|---|---|\n");
                let n = sr.cluster_distances.len();
                for pair in sr.cluster_distances.iter().rev().take(5) {
                    md.push_str(&format!("| {} | {} | {:.6} |\n",
                        pair.cluster_a, pair.cluster_b, pair.distance));
                }
                md.push_str("\n");

                // Distance statistics
                let dists: Vec<f64> = sr.cluster_distances.iter().map(|p| p.distance).collect();
                let mean_dist = dists.iter().sum::<f64>() / dists.len() as f64;
                let min_dist = dists.first().copied().unwrap_or(0.0);
                let max_dist = dists.last().copied().unwrap_or(0.0);
                md.push_str(&format!("Inter-cluster distance: mean={:.6}, min={:.6}, max={:.6}, {} pairs\n\n",
                    mean_dist, min_dist, max_dist, n));
            }
        }

        // ── 8. Cluster Functional Group Characterization ──
        md.push_str("## 8. Cluster Functional Group Characterization\n\n");
        md.push_str("Summary of functional group signatures across the largest clusters in each stratum.\n");
        md.push_str("Enrichment ratio shows over-representation relative to the stratum population.\n\n");

        for sr in &self.strata_results {
            let range = qed_ranges.get(sr.group_id).unwrap_or(&"—");
            md.push_str(&format!("### Stratum {} ({}) — Cluster FG Signatures\n\n", sr.group_id, range));

            let mut sorted = sr.cluster_infos.clone();
            sorted.sort_by(|a, b| b.size.cmp(&a.size));

            for ci in sorted.iter().take(5) {
                md.push_str(&format!("**Cluster {} ({} molecules)**", ci.cluster_id, ci.size));
                if let Some(repr) = &ci.representative_smiles {
                    let short = if repr.len() > 40 { &repr[..40] } else { repr };
                    md.push_str(&format!(" — representative: `{}`", short));
                }
                md.push_str("\n\n");

                // FG breakdown with statistical significance
                md.push_str("| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment | p_adj (FDR) | Sig. |\n");
                md.push_str("|---|---|---|---|---|---|\n");

                let cluster_sorted = ci.cluster_census.sorted_by_prevalence();
                for (fg, _, cluster_pct) in cluster_sorted.iter().take(8) {
                    let stratum_pct = sr.stratum_census.prevalence_pct(*fg);
                    let enrichment = if stratum_pct > 0.0 { cluster_pct / stratum_pct } else { 0.0 };
                    let marker = if enrichment > 1.5 { " ⬆" } else if enrichment < 0.5 { " ⬇" } else { "" };
                    // Look up p-value from enrichment_results
                    let (p_str, sig_str) = ci.enrichment_results.iter()
                        .find(|er| er.fg_name == fg.name())
                        .map(|er| (stats::format_p_value(er.p_adjusted), if er.significant { "✓" } else { "" }))
                        .unwrap_or(("—".to_string(), ""));
                    md.push_str(&format!("| {} | {:.1} | {:.1} | {:.2}×{} | {} | {} |\n",
                        fg.name(), cluster_pct, stratum_pct, enrichment, marker, p_str, sig_str));
                }
                md.push_str("\n");
            }
        }

        // ── 9. Cluster Quality Analysis ──
        md.push_str("## 9. Cluster Quality Analysis\n\n");

        // Per-stratum cluster quality
        md.push_str("### 9.1 Per-Stratum Quality Metrics\n\n");
        md.push_str("| Stratum | Silhouette | Davies-Bouldin | QE | Clusters | Gini | Singletons |\n");
        md.push_str("|---|---|---|---|---|---|---|\n");
        let qed_ranges = ["[0, 0.399)", "[0.399, 0.520)", "[0.520, 0.694)", "[0.694, 0.814)", "[0.814, 1.0]"];
        for sr in &self.strata_results {
            let range = qed_ranges.get(sr.group_id).unwrap_or(&"—");
            md.push_str(&format!(
                "| {} {} | {:.4} | {:.4} | {:.6} | {} | {:.3} | {} |\n",
                sr.group_id, range,
                sr.cluster_quality.silhouette_mean,
                sr.cluster_quality.davies_bouldin,
                sr.quantization_error,
                sr.num_clusters_used,
                sr.size_stats.gini,
                sr.size_stats.singletons,
            ));
        }
        md.push_str("\n");

        md.push_str("**Interpretation guide:**\n");
        md.push_str("- **Silhouette** ∈ [-1, 1]: higher = better separation (>0.5 strong, >0.25 reasonable)\n");
        md.push_str("- **Davies-Bouldin**: lower = better separation (0 is optimal)\n");
        md.push_str("- **Gini coefficient**: 0 = equal sizes, 1 = maximally unequal\n\n");

        // Cluster size distributions
        md.push_str("### 9.2 Cluster Size Distribution\n\n");
        md.push_str("| Stratum | Mean | Median | Std | Min | P25 | P75 | Max | Large |\n");
        md.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for sr in &self.strata_results {
            md.push_str(&format!(
                "| {} | {:.1} | {:.0} | {:.1} | {} | {} | {} | {} | {} |\n",
                sr.group_id,
                sr.size_stats.mean, sr.size_stats.median, sr.size_stats.std,
                sr.size_stats.min, sr.size_stats.p25, sr.size_stats.p75, sr.size_stats.max,
                sr.size_stats.large_clusters,
            ));
        }
        md.push_str("\n");

        md.push_str("![Cluster Quality](figures/cluster_quality_comparison.svg)\n\n");
        md.push_str("*Figure 12: Comparison of cluster quality metrics across QED strata — silhouette score, Davies-Bouldin index, quantization error, and Gini coefficient.*\n\n");
        md.push_str("![Cluster Sizes](figures/cluster_size_distribution.svg)\n\n");
        md.push_str("*Figure 13: Distribution of cluster sizes within each QED stratum.*\n\n");

        // ── 9.5. Cross-Stratum Statistical Analysis ──
        md.push_str("## 9.5 Cross-Stratum Statistical Analysis\n\n");
        md.push_str("### Linear Regressions Across Stratum Means\n\n");
        md.push_str("| Regression | R² | Slope ± SE | p-value | n |\n");
        md.push_str("|---|---|---|---|---|\n");
        md.push_str(&format!("| SAS ~ QED | {:.4} | {:.3} ± {:.3} | {} | {} |\n",
            self.cross_stratum_stats.sas_qed_regression.r_squared,
            self.cross_stratum_stats.sas_qed_regression.slope,
            self.cross_stratum_stats.sas_qed_regression.slope_se,
            stats::format_p_value(self.cross_stratum_stats.sas_qed_regression.p_value),
            self.cross_stratum_stats.sas_qed_regression.n));
        md.push_str(&format!("| logP ~ QED | {:.4} | {:.3} ± {:.3} | {} | {} |\n\n",
            self.cross_stratum_stats.logp_qed_regression.r_squared,
            self.cross_stratum_stats.logp_qed_regression.slope,
            self.cross_stratum_stats.logp_qed_regression.slope_se,
            stats::format_p_value(self.cross_stratum_stats.logp_qed_regression.p_value),
            self.cross_stratum_stats.logp_qed_regression.n));

        md.push_str("**Interpretation**: Each 0.1 QED improvement costs approximately ");
        let sas_per_01_qed = self.cross_stratum_stats.sas_qed_regression.slope * 0.1;
        md.push_str(&format!("{:.2} SAS units of synthetic difficulty.\n\n", sas_per_01_qed));

        md.push_str("### Functional Group Co-occurrence (Chi-Squared Test)\n\n");
        if let Some(ref chi2) = self.cross_stratum_stats.sulfonyl_heterocycle_chi2 {
            md.push_str(&format!("**Sulfonyl–Heterocycle co-occurrence**: χ² = {:.1} (df = {}, p {})\n\n",
                chi2.chi2, chi2.df, stats::format_p_value(chi2.p_value)));
            md.push_str(&format!("P(heterocycle | sulfonyl) = {:.1}% vs P(heterocycle) overall = {:.1}%\n\n",
                self.cross_stratum_stats.sulfonyl_heterocycle_cooccurrence * 100.0,
                self.global_fg_census.prevalence_pct(FunctionalGroup::Heterocycle)));
        }

        md.push_str("### Enrichment Statistical Framework\n\n");
        md.push_str("All enrichment ratios in Section 8 are tested using **Fisher's exact test** (one-sided, right-tailed) ");
        md.push_str("with **Benjamini-Hochberg FDR correction** at α = 0.05. Only enrichments marked ✓ are statistically significant.\n\n");

        // Count significant enrichments across all strata
        let total_tests: usize = self.strata_results.iter()
            .flat_map(|sr| sr.cluster_infos.iter())
            .map(|ci| ci.enrichment_results.len())
            .sum();
        let sig_tests: usize = self.strata_results.iter()
            .flat_map(|sr| sr.cluster_infos.iter())
            .map(|ci| ci.enrichment_results.iter().filter(|er| er.significant).count())
            .sum();
        md.push_str(&format!("Total enrichment tests performed: {}\n", total_tests));
        md.push_str(&format!("Significant after FDR correction: {} ({:.1}%)\n\n", sig_tests, sig_tests as f64 / total_tests.max(1) as f64 * 100.0));

        // ── 10. Evaluation Summary ──
        md.push_str("## 10. Evaluation Summary\n\n");

        let all_sizes: Vec<usize> = self.strata_results.iter()
            .flat_map(|sr| sr.cluster_infos.iter().map(|c| c.size))
            .collect();
        if !all_sizes.is_empty() {
            let mean_size = all_sizes.iter().sum::<usize>() as f64 / all_sizes.len() as f64;
            let max_size = all_sizes.iter().copied().max().unwrap_or(0);
            let min_size = all_sizes.iter().copied().min().unwrap_or(0);
            let active_clusters: usize = self.strata_results.iter().map(|sr| sr.num_clusters_used).sum();
            let total_neurons: usize = self.autotune_results.iter()
                .map(|at| at.best_grid.0 * at.best_grid.1)
                .sum::<usize>()
                .max(self.strata_results.len() * self.som_grid.0 * self.som_grid.1);

            // Global averages
            let avg_silhouette = self.strata_results.iter()
                .map(|s| s.cluster_quality.silhouette_mean).sum::<f64>() / self.strata_results.len().max(1) as f64;
            let avg_db = self.strata_results.iter()
                .map(|s| s.cluster_quality.davies_bouldin).sum::<f64>() / self.strata_results.len().max(1) as f64;

            md.push_str("| Metric | Value |\n|---|---|\n");
            md.push_str(&format!("| Total active clusters | {} / {} neurons |\n", active_clusters, total_neurons));
            md.push_str(&format!("| Cluster size (mean) | {:.1} |\n", mean_size));
            md.push_str(&format!("| Cluster size (range) | {} – {} |\n", min_size, max_size));
            md.push_str(&format!("| Average quantization error | {:.6} |\n", avg_qe));
            md.push_str(&format!("| Average silhouette score | {:.4} |\n", avg_silhouette));
            md.push_str(&format!("| Average Davies-Bouldin index | {:.4} |\n", avg_db));

            let all_compactness: Vec<f64> = self.strata_results.iter()
                .flat_map(|sr| sr.cluster_infos.iter().map(|c| c.compactness))
                .collect();
            let mean_compact = all_compactness.iter().sum::<f64>() / all_compactness.len().max(1) as f64;
            md.push_str(&format!("| Mean intra-cluster distance | {:.6} |\n", mean_compact));

            let fg_types_detected = self.global_fg_census.sorted_by_prevalence().len();
            md.push_str(&format!("| Functional group types detected | {} / 22 |\n", fg_types_detected));

            if let Some(best) = self.importance.fg_property_correlations.first() {
                let max_r = best.qed_corr.abs().max(best.logp_corr.abs()).max(best.sas_corr.abs());
                md.push_str(&format!("| Strongest FG-property |r| | {:.4} ({}) |\n", max_r, best.fg.name()));
            }
            md.push_str("\n");
        }

        // ── 11. Performance ──
        md.push_str("## 11. Performance\n\n");
        md.push_str("| Phase | Time |\n|---|---|\n");
        md.push_str(&format!("| Data loading | {:.2}s |\n", self.timings.load_secs));
        md.push_str(&format!("| Graph parsing + FG detection | {:.2}s |\n", self.timings.parse_secs));
        md.push_str(&format!("| VGAE encoding | {:.2}s |\n", self.timings.encode_secs));
        md.push_str(&format!("| Importance analysis | {:.2}s |\n", self.timings.importance_secs));
        md.push_str(&format!("| SOM clustering + FG analysis | {:.2}s |\n", self.timings.cluster_secs));
        md.push_str(&format!("| **Total** | **{:.2}s** |\n\n", self.timings.total_secs));

        if self.processed_molecules > 0 {
            let throughput = self.processed_molecules as f64 / self.timings.total_secs;
            md.push_str(&format!("**Throughput**: {:.0} molecules/second\n\n", throughput));
        }

        // ── 11. Methodology Comparison ──
        md.push_str("## 12. Methodology Comparison\n\n");
        md.push_str("| Aspect | Previous (Python) | Current (Rust + GNN) |\n");
        md.push_str("|---|---|---|\n");
        md.push_str("| Molecular representation | Flat 28-dim feature vector | Full molecular graph |\n");
        md.push_str("| Feature learning | Dense autoencoder (28→16→28) | Graph Attention Network (3 layers) |\n");
        md.push_str("| Latent model | Deterministic AE | Variational (VGAE with KL regularization) |\n");
        md.push_str("| Structure awareness | None (bag of atoms) | Message passing preserves bond topology |\n");
        md.push_str("| Pooling | N/A (fixed features) | Global attention pooling (learned) |\n");
        md.push_str("| Edge features | Not used | 9-dim bond features in attention |\n");
        md.push_str("| Functional group analysis | None | 22-type substructure detection + enrichment |\n");
        md.push_str("| Importance analysis | None | Dim-property correlation + FG-property correlation |\n");
        md.push_str("| Cluster characterization | Size + basic stats | FG signatures, enrichment, representatives |\n");
        md.push_str("| Implementation | Python/PyTorch | Rust/Burn (memory-safe, zero-cost abstractions) |\n\n");

        // ── 12. Output Files ──
        md.push_str("## 13. Output Files\n\n");
        md.push_str("```\nresults/\n");
        md.push_str("├── RESULTS.md              # This report\n");
        md.push_str("├── training_losses.csv     # Per-molecule reconstruction losses\n");
        md.push_str("├── figures/                # SVG visualizations\n");
        for (path, desc) in &self.figures {
            md.push_str(&format!("│   ├── {}  # {}\n", path.replace("figures/", ""), desc));
        }
        for sr in &self.strata_results {
            md.push_str(&format!("└── group_{}/\n", sr.group_id));
            md.push_str("    ├── labeled_data.csv    # SMILES + properties + cluster label\n");
            md.push_str("    └── embeddings.csv      # 16-dim latent embeddings\n");
        }
        md.push_str("```\n\n");

        // Figure index
        md.push_str("## 14. Figure Index\n\n");
        md.push_str("| # | Figure | Description |\n|---|---|---|\n");
        let figure_descriptions = [
            ("Figure 1", "figures/property_distributions_combined.svg", "Molecular property distributions (QED, logP, SAS)"),
            ("Figure 2", "figures/qed_distribution.svg", "Individual property histograms with mean indicators"),
            ("Figure 3", "figures/molecule_complexity.svg", "Molecular graph complexity scatter (atoms vs bonds)"),
            ("Figure 4", "figures/fg_prevalence.svg", "Functional group prevalence bar chart"),
            ("Figure 5", "figures/reconstruction_loss_dist.svg", "VGAE reconstruction loss distribution"),
            ("Figure 6", "figures/embedding_dim_variance.svg", "Latent dimension variance analysis"),
            ("Figure 7", "figures/dim_property_heatmap.svg", "Dimension–property correlation heatmap"),
            ("Figure 8", "figures/fg_property_correlations.svg", "FG–property correlation heatmap"),
            ("Figure 9", "figures/latent_space_umap.svg", "UMAP projection of latent space by stratum"),
            ("Figure 10", "figures/stratum_property_comparison.svg", "Stratum property comparison (mean ± std)"),
            ("Figure 11", "figures/umatrix_heatmaps.svg", "SOM U-matrix heatmaps per stratum"),
            ("Figure 12", "figures/cluster_quality_comparison.svg", "Cluster quality metrics comparison"),
            ("Figure 13", "figures/cluster_size_distribution.svg", "Cluster size distributions per stratum"),
        ];
        for (num, path, desc) in &figure_descriptions {
            md.push_str(&format!("| {} | [{}]({}) | {} |\n", num, path, path, desc));
        }
        for (stratum_id, _, _) in &self.strata_results.iter()
            .map(|sr| (sr.group_id, &sr.cluster_distances, sr.num_clusters_used))
            .collect::<Vec<_>>()
        {
            let fig_num = 14 + stratum_id;
            let path = format!("figures/fg_enrichment_stratum_{}.svg", stratum_id);
            md.push_str(&format!("| Figure {} | [{}]({}) | FG enrichment heatmap — Stratum {} |\n", fig_num, path, path, stratum_id));
        }
        md.push_str("\n");

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── pearson_correlation ──

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Perfect positive correlation: r={}", r);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&x, &y);
        assert!((r + 1.0).abs() < 1e-10, "Perfect negative correlation: r={}", r);
    }

    #[test]
    fn test_pearson_uncorrelated() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 1.0, 5.0, 3.0];
        let r = pearson_correlation(&x, &y);
        assert!(r.abs() < 0.5, "Weakly correlated data: r={}", r);
    }

    #[test]
    fn test_pearson_constant_returns_zero() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let r = pearson_correlation(&x, &y);
        assert_eq!(r, 0.0, "Constant input should return 0");
    }

    #[test]
    fn test_pearson_too_short() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let r = pearson_correlation(&x, &y);
        assert_eq!(r, 0.0, "n < 3 should return 0");
    }

    // ── stat_std / stat_var ──

    #[test]
    fn test_stat_std_known() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = stat_std(&vals);
        // Population std=2.0, sample std≈2.138
        assert!((std - 2.138).abs() < 0.01, "std={}", std);
    }

    #[test]
    fn test_stat_var_known() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = stat_var(&vals);
        assert!((var - stat_std(&vals).powi(2)).abs() < 1e-10);
    }

    #[test]
    fn test_stat_std_single_value() {
        let vals = vec![42.0];
        assert_eq!(stat_std(&vals), 0.0);
    }

    #[test]
    fn test_stat_std_empty() {
        let vals: Vec<f64> = vec![];
        assert_eq!(stat_std(&vals), 0.0);
    }

    // ── euclidean_dist ──

    #[test]
    fn test_euclidean_dist_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let d = euclidean_dist(&a, &b);
        assert!((d - 5.0).abs() < 1e-10, "3-4-5 triangle: d={}", d);
    }

    #[test]
    fn test_euclidean_dist_same_point() {
        let a = vec![1.0f32, 2.0, 3.0];
        let d = euclidean_dist(&a, &a);
        assert!((d).abs() < 1e-10);
    }

    // ── centroid_of ──

    #[test]
    fn test_centroid_of_basic() {
        let embeddings = vec![
            vec![0.0f32, 0.0],
            vec![2.0f32, 4.0],
            vec![4.0f32, 2.0],
        ];
        let c = centroid_of(&[0, 1, 2], &embeddings);
        assert!((c[0] - 2.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_centroid_of_single() {
        let embeddings = vec![vec![3.0f32, 7.0]];
        let c = centroid_of(&[0], &embeddings);
        assert_eq!(c, vec![3.0, 7.0]);
    }

    #[test]
    fn test_centroid_of_empty() {
        let embeddings = vec![vec![1.0f32]];
        let c = centroid_of(&[], &embeddings);
        assert!(c.is_empty());
    }

    // ── cluster_size_distribution ──

    fn dummy_cluster(id: usize, size: usize, centroid: Vec<f32>) -> ClusterInfo {
        ClusterInfo {
            cluster_id: id,
            size,
            centroid,
            mean_qed: 0.0, std_qed: 0.0, mean_logp: 0.0, mean_sas: 0.0,
            compactness: 0.0,
            cluster_census: FGCensus::from_profiles(&[]),
            signature_fgs: vec![],
            dominant_fg: None,
            representative_smiles: None,
            enrichment_results: vec![],
        }
    }

    #[test]
    fn test_cluster_size_distribution_uniform() {
        let clusters: Vec<ClusterInfo> = (0..10).map(|i| {
            let mut c = dummy_cluster(i, 100, vec![0.0]);
            c.mean_qed = 0.5;
            c
        }).collect();

        let stats = cluster_size_distribution(&clusters);
        assert_eq!(stats.total_clusters, 10);
        assert!((stats.mean - 100.0).abs() < 1e-5);
        assert!((stats.median - 100.0).abs() < 1e-5);
        assert!((stats.std).abs() < 1e-5, "Uniform sizes should have zero std");
        assert!((stats.gini).abs() < 1e-5, "Uniform sizes should have zero Gini");
        assert_eq!(stats.singletons, 0);
    }

    #[test]
    fn test_cluster_size_distribution_varied() {
        let sizes = vec![1, 1, 5, 10, 50, 100];
        let clusters: Vec<ClusterInfo> = sizes.iter().enumerate()
            .map(|(i, &s)| dummy_cluster(i, s, vec![]))
            .collect();

        let stats = cluster_size_distribution(&clusters);
        assert_eq!(stats.singletons, 2);
        assert!(stats.gini > 0.3, "Unequal sizes should have high Gini: {}", stats.gini);
        assert_eq!(stats.min, 1);
        assert_eq!(stats.max, 100);
    }

    // ── inter_cluster_distances ──

    #[test]
    fn test_inter_cluster_distances_basic() {
        let clusters = vec![
            dummy_cluster(0, 10, vec![0.0, 0.0]),
            dummy_cluster(1, 10, vec![3.0, 4.0]),
        ];

        let pairs = inter_cluster_distances(&clusters);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_inter_cluster_distances_sorted() {
        let clusters = vec![
            dummy_cluster(0, 1, vec![0.0]),
            dummy_cluster(1, 1, vec![10.0]),
            dummy_cluster(2, 1, vec![1.0]),
        ];

        let pairs = inter_cluster_distances(&clusters);
        assert_eq!(pairs.len(), 3);
        // Sorted by distance: (0,2)=1.0, (1,2)=9.0, (0,1)=10.0
        assert!((pairs[0].distance - 1.0).abs() < 1e-5);
        assert!((pairs[1].distance - 9.0).abs() < 1e-5);
        assert!((pairs[2].distance - 10.0).abs() < 1e-5);
    }

    // ── silhouette_score ──

    #[test]
    fn test_silhouette_well_separated() {
        let embeddings: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.1],
            vec![10.0, 10.0], vec![10.1, 10.1], vec![10.0, 10.1],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let q = silhouette_score(&embeddings, &labels);
        assert!(q.silhouette_mean > 0.5, "Well-separated clusters should have high silhouette: {}", q.silhouette_mean);
    }

    // ── features_to_tensors (smoke test) ──

    #[test]
    fn test_features_to_tensors_basic() {
        let feats = MolecularFeatures {
            num_atoms: 2,
            num_bonds: 1,
            node_features: vec![vec![0.0; NODE_FEATURE_DIM]; 2],
            edge_features: vec![vec![0.0; EDGE_FEATURE_DIM]; 1],
            edge_index: vec![[0, 1]],
        };
        let device = WgpuDevice::BestAvailable;
        let (nodes, edges) = features_to_tensors(&feats, &device);
        assert_eq!(nodes.dims(), [2, NODE_FEATURE_DIM]);
        assert_eq!(edges.dims(), [1, EDGE_FEATURE_DIM]);
    }

    // ── Pipeline state save/load ──

    #[test]
    fn test_pipeline_state_save_load() {
        let state = PipelineState {
            version: "1.0".to_string(),
            csv_path: "test.csv".to_string(),
            total_molecules: 100,
            processed_molecules: 95,
            embeddings: vec![vec![0.1, 0.2, 0.3]; 3],
            valid_indices: vec![0, 1, 2],
            recon_losses: vec![0.01, 0.02, 0.03],
            labels: vec![0, 1, 0],
            som_states: vec![],
            autotune_results: vec![],
            phase_completed: 2,
        };

        let path = "/tmp/fga_test_pipeline_state.json";
        state.save(path).unwrap();
        let loaded = PipelineState::load(path).unwrap();

        assert_eq!(loaded.version, "1.0");
        assert_eq!(loaded.total_molecules, 100);
        assert_eq!(loaded.processed_molecules, 95);
        assert_eq!(loaded.embeddings.len(), 3);
        assert_eq!(loaded.phase_completed, 2);
        assert_eq!(loaded.labels, vec![0, 1, 0]);

        let _ = std::fs::remove_file(path);
    }
}
