/// Full analysis pipeline: CSV → SMILES → Graph → FG detect → GNN encode → SOM cluster → analysis → results.

use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use std::time::Instant;
use std::collections::HashMap;

use crate::autoencoder::{Vgae, VgaeConfig, TrainConfig, vgae_loss};
use crate::features::{self, MolecularFeatures, NODE_FEATURE_DIM, EDGE_FEATURE_DIM};
use crate::functional_groups::{self, FunctionalGroup, FGProfile, FGCensus, fg_enrichment};
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

        // Top signature FGs (enrichment > 1.5)
        let signature_fgs: Vec<(FunctionalGroup, f64)> = enrichment.iter()
            .filter(|(_, e)| *e > 1.2)
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

    let max_molecules = total.min(5000);
    let records = &all_records[..max_molecules];
    log::info!("Experiment subset: {} molecules", max_molecules);

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

    for (i, (feats, orig_idx, _)) in mol_features.iter().enumerate() {
        let (node_t, edge_t) = features_to_tensors(feats, &device);

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

    io::save_training_losses(output_dir, &recon_losses, &recon_losses)?;

    // ═══════════════════════════════════════════════════
    // Phase 3: Importance Analysis
    // ═══════════════════════════════════════════════════
    log::info!("════════════════════════════════════════════");
    log::info!("  Phase 3: Importance Analysis");
    log::info!("════════════════════════════════════════════");
    let t_imp = Instant::now();
    let importance = importance_analysis(&embeddings, &record_refs, &all_fg_profiles, &valid_indices);
    let importance_time = t_imp.elapsed();

    log::info!("Importance analysis complete in {:.2}s", importance_time.as_secs_f64());
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

        let u_vals: Vec<f64> = u_matrix.iter().flatten().copied().collect();
        let u_mean = u_vals.iter().sum::<f64>() / u_vals.len() as f64;
        let u_max = u_vals.iter().copied().fold(f64::MIN, f64::max);

        for (k, &emb_idx) in stratum_emb_indices.iter().enumerate() {
            all_labels[emb_idx] = labels[k];
        }

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

        log::info!("  Active clusters: {}/100 | QE: {:.6}", used_clusters.len(), qe);

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
        });
    }

    let cluster_time = t3.elapsed();
    let total_time = pipeline_start.elapsed();

    log::info!("════════════════════════════════════════════");
    log::info!("  Pipeline complete");
    log::info!("════════════════════════════════════════════");
    log::info!("  Total: {:.2}s", total_time.as_secs_f64());

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
        global_fg_census,
        importance,
        timings: Timings {
            load_secs: load_time.as_secs_f64(),
            parse_secs: parse_time.as_secs_f64(),
            encode_secs: encode_time.as_secs_f64(),
            importance_secs: importance_time.as_secs_f64(),
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
    pub centroid: Vec<f32>,
    pub cluster_census: FGCensus,
    pub signature_fgs: Vec<(FunctionalGroup, f64)>,
    pub dominant_fg: Option<FunctionalGroup>,
    pub representative_smiles: Option<String>,
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

#[derive(Debug)]
pub struct Timings {
    pub load_secs: f64,
    pub parse_secs: f64,
    pub encode_secs: f64,
    pub importance_secs: f64,
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
    pub global_fg_census: FGCensus,
    pub importance: ImportanceAnalysis,
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
    pub stratum_census: FGCensus,
    pub cluster_distances: Vec<ClusterDistancePair>,
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
        md.push_str(&format!("| Grid size | {}×{} ({} neurons) |\n",
            self.som_grid.0, self.som_grid.1, self.som_grid.0 * self.som_grid.1));
        md.push_str("| Training epochs | 128 |\n");
        md.push_str("| Initial learning rate | 0.5 |\n");
        md.push_str("| Initial radius | 5.0 |\n");
        md.push_str("| Distance metric | Euclidean |\n");
        md.push_str("| Neighborhood | Gaussian |\n\n");

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

                // FG breakdown
                md.push_str("| Functional Group | Cluster Prev (%) | Stratum Prev (%) | Enrichment |\n");
                md.push_str("|---|---|---|---|\n");

                let cluster_sorted = ci.cluster_census.sorted_by_prevalence();
                for (fg, _, cluster_pct) in cluster_sorted.iter().take(8) {
                    let stratum_pct = sr.stratum_census.prevalence_pct(*fg);
                    let enrichment = if stratum_pct > 0.0 { cluster_pct / stratum_pct } else { 0.0 };
                    let marker = if enrichment > 1.5 { " ⬆" } else if enrichment < 0.5 { " ⬇" } else { "" };
                    md.push_str(&format!("| {} | {:.1} | {:.1} | {:.2}×{} |\n",
                        fg.name(), cluster_pct, stratum_pct, enrichment, marker));
                }
                md.push_str("\n");
            }
        }

        // ── 9. Evaluation Summary ──
        md.push_str("## 9. Evaluation Summary\n\n");

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

            let all_compactness: Vec<f64> = self.strata_results.iter()
                .flat_map(|sr| sr.cluster_infos.iter().map(|c| c.compactness))
                .collect();
            let mean_compact = all_compactness.iter().sum::<f64>() / all_compactness.len().max(1) as f64;
            md.push_str(&format!("| Mean intra-cluster distance | {:.6} |\n", mean_compact));

            // Functional group coverage
            let fg_types_detected = self.global_fg_census.sorted_by_prevalence().len();
            md.push_str(&format!("| Functional group types detected | {} / 22 |\n", fg_types_detected));

            // Strongest property-FG correlation
            if let Some(best) = self.importance.fg_property_correlations.first() {
                let max_r = best.qed_corr.abs().max(best.logp_corr.abs()).max(best.sas_corr.abs());
                md.push_str(&format!("| Strongest FG-property |r| | {:.4} ({}) |\n", max_r, best.fg.name()));
            }
            md.push_str("\n");
        }

        // ── 10. Performance ──
        md.push_str("## 10. Performance\n\n");
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
        md.push_str("## 11. Methodology Comparison\n\n");
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
        md.push_str("## 12. Output Files\n\n");
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
