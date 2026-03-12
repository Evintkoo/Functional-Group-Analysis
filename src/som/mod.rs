/// Self-Organizing Map (SOM) for clustering graph-level latent embeddings.
/// Competitive learning on a 2D grid of neurons with neighborhood decay.
/// Includes autotune to find optimal grid size by evaluating multiple candidates.

use rand::Rng;
use serde::{Serialize, Deserialize};

/// SOM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub input_dim: usize,
    pub num_epochs: usize,
    pub initial_learning_rate: f64,
    pub initial_radius: f64,
}

impl SomConfig {
    pub fn new(input_dim: usize) -> Self {
        Self {
            grid_width: 10,
            grid_height: 10,
            input_dim,
            num_epochs: 128,
            initial_learning_rate: 0.5,
            initial_radius: 5.0,
        }
    }

    pub fn with_grid(mut self, width: usize, height: usize) -> Self {
        self.grid_width = width;
        self.grid_height = height;
        self.initial_radius = (width.max(height) as f64) / 2.0;
        self
    }
}

/// Result of autotune evaluation for a single grid configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutotuneCandidate {
    pub grid_width: usize,
    pub grid_height: usize,
    pub num_clusters: usize,
    pub active_clusters: usize,
    pub quantization_error: f64,
    pub topographic_error: f64,
    pub combined_score: f64,
}

/// Result of the full autotune search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutotuneResult {
    pub best_config: SomConfig,
    pub candidates: Vec<AutotuneCandidate>,
    pub best_grid: (usize, usize),
}

/// A trained Self-Organizing Map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Som {
    pub weights: Vec<Vec<f64>>,  // [grid_size, input_dim]
    pub grid_width: usize,
    pub grid_height: usize,
    pub input_dim: usize,
}

impl Som {
    /// Initialize SOM with random weights sampled from the data distribution.
    pub fn new(config: &SomConfig, data: &[Vec<f32>]) -> Self {
        let grid_size = config.grid_width * config.grid_height;
        let mut rng = rand::thread_rng();

        // Initialize weights from random data samples
        let weights: Vec<Vec<f64>> = (0..grid_size)
            .map(|_| {
                let idx = rng.gen_range(0..data.len());
                data[idx].iter().map(|&v| v as f64).collect()
            })
            .collect();

        Som {
            weights,
            grid_width: config.grid_width,
            grid_height: config.grid_height,
            input_dim: config.input_dim,
        }
    }

    /// Train the SOM on the provided data.
    pub fn train(&mut self, data: &[Vec<f32>], config: &SomConfig) {
        let mut rng = rand::thread_rng();
        let total_iterations = config.num_epochs * data.len();

        for epoch in 0..config.num_epochs {
            // Shuffle-like: random sample order per epoch
            let mut indices: Vec<usize> = (0..data.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            for (step, &idx) in indices.iter().enumerate() {
                let global_step = epoch * data.len() + step;
                let progress = global_step as f64 / total_iterations as f64;

                // Decay learning rate and radius
                let lr = config.initial_learning_rate * (1.0 - progress);
                let radius = config.initial_radius * (1.0 - progress).max(0.5);

                let input: Vec<f64> = data[idx].iter().map(|&v| v as f64).collect();

                // Find Best Matching Unit (BMU)
                let bmu = self.find_bmu(&input);

                // Update weights in neighborhood
                self.update_weights(bmu, &input, lr, radius);
            }

            if (epoch + 1) % 10 == 0 {
                let qe = self.quantization_error(data);
                log::info!("SOM epoch {}/{}: quantization error = {:.6}", epoch + 1, config.num_epochs, qe);
            }
        }
    }

    /// Find the Best Matching Unit for an input vector.
    pub fn find_bmu(&self, input: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;

        for (i, w) in self.weights.iter().enumerate() {
            let dist: f64 = w.iter().zip(input.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Update neuron weights based on neighborhood function.
    fn update_weights(&mut self, bmu: usize, input: &[f64], lr: f64, radius: f64) {
        let bmu_row = bmu / self.grid_width;
        let bmu_col = bmu % self.grid_width;
        let radius_sq = radius * radius;

        for i in 0..self.weights.len() {
            let row = i / self.grid_width;
            let col = i % self.grid_width;

            let dist_sq = ((row as f64 - bmu_row as f64).powi(2)
                + (col as f64 - bmu_col as f64).powi(2)) as f64;

            if dist_sq <= radius_sq * 4.0 {
                // Gaussian neighborhood function
                let influence = (-dist_sq / (2.0 * radius_sq)).exp();
                let effective_lr = lr * influence;

                for (w, x) in self.weights[i].iter_mut().zip(input.iter()) {
                    *w += effective_lr * (x - *w);
                }
            }
        }
    }

    /// Assign each data point to its BMU cluster.
    pub fn predict(&self, data: &[Vec<f32>]) -> Vec<usize> {
        data.iter()
            .map(|d| {
                let input: Vec<f64> = d.iter().map(|&v| v as f64).collect();
                self.find_bmu(&input)
            })
            .collect()
    }

    /// Mean quantization error across all data points.
    pub fn quantization_error(&self, data: &[Vec<f32>]) -> f64 {
        let total: f64 = data.iter()
            .map(|d| {
                let input: Vec<f64> = d.iter().map(|&v| v as f64).collect();
                let bmu = self.find_bmu(&input);
                self.weights[bmu].iter().zip(input.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum();
        total / data.len() as f64
    }

    /// Get cluster centers (neuron weight vectors).
    pub fn cluster_centers(&self) -> Vec<Vec<f32>> {
        self.weights.iter()
            .map(|w| w.iter().map(|&v| v as f32).collect())
            .collect()
    }

    /// Get 2D grid coordinates for a neuron index.
    pub fn neuron_coords(&self, idx: usize) -> (usize, usize) {
        (idx / self.grid_width, idx % self.grid_width)
    }

    /// Compute U-Matrix (unified distance matrix) for visualization.
    pub fn u_matrix(&self) -> Vec<Vec<f64>> {
        let mut u_mat = vec![vec![0.0; self.grid_width]; self.grid_height];

        for row in 0..self.grid_height {
            for col in 0..self.grid_width {
                let idx = row * self.grid_width + col;
                let mut sum_dist = 0.0;
                let mut count = 0;

                // Check 4-connected neighbors
                for (dr, dc) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0 && nr < self.grid_height as i32 && nc >= 0 && nc < self.grid_width as i32 {
                        let nidx = nr as usize * self.grid_width + nc as usize;
                        let dist: f64 = self.weights[idx].iter()
                            .zip(self.weights[nidx].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        sum_dist += dist;
                        count += 1;
                    }
                }

                u_mat[row][col] = if count > 0 { sum_dist / count as f64 } else { 0.0 };
            }
        }

        u_mat
    }

    /// Compute topographic error: fraction of data points where the two closest
    /// neurons (BMU and second-BMU) are NOT adjacent on the grid.
    pub fn topographic_error(&self, data: &[Vec<f32>]) -> f64 {
        if data.is_empty() { return 0.0; }
        let mut non_adjacent = 0usize;

        for d in data {
            let input: Vec<f64> = d.iter().map(|&v| v as f64).collect();
            let (bmu, second_bmu) = self.find_two_bmu(&input);
            let (r1, c1) = self.neuron_coords(bmu);
            let (r2, c2) = self.neuron_coords(second_bmu);
            let dr = (r1 as i32 - r2 as i32).unsigned_abs() as usize;
            let dc = (c1 as i32 - c2 as i32).unsigned_abs() as usize;
            if dr + dc > 1 {
                non_adjacent += 1;
            }
        }

        non_adjacent as f64 / data.len() as f64
    }

    /// Find the two closest neurons (BMU and second-BMU).
    fn find_two_bmu(&self, input: &[f64]) -> (usize, usize) {
        let mut best = (0, f64::MAX);
        let mut second = (0, f64::MAX);

        for (i, w) in self.weights.iter().enumerate() {
            let dist: f64 = w.iter().zip(input.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist < best.1 {
                second = best;
                best = (i, dist);
            } else if dist < second.1 {
                second = (i, dist);
            }
        }

        (best.0, second.0)
    }

    /// Count active clusters (neurons that are BMU for at least one data point).
    pub fn active_cluster_count(&self, data: &[Vec<f32>]) -> usize {
        let labels = self.predict(data);
        let mut seen = std::collections::HashSet::new();
        for l in labels {
            seen.insert(l);
        }
        seen.len()
    }

    /// Serialize SOM state to JSON.
    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Deserialize SOM state from JSON.
    pub fn load_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let som: Som = serde_json::from_str(&json)?;
        Ok(som)
    }
}

/// Autotune: evaluate multiple grid configurations and return the best one.
/// Scores combine normalized quantization error and topographic error.
/// Candidate grids are sized heuristically from sqrt(N) with scaling factors.
pub fn autotune(data: &[Vec<f32>], input_dim: usize, fast: bool) -> AutotuneResult {
    let n = data.len();
    let base = (n as f64).sqrt();

    // Candidate grid side lengths — from small to large
    let candidates: Vec<usize> = if fast {
        vec![
            (base * 0.3).ceil() as usize,
            (base * 0.5).ceil() as usize,
            (base * 0.7).ceil() as usize,
        ]
    } else {
        vec![
            (base * 0.2).ceil() as usize,
            (base * 0.3).ceil() as usize,
            (base * 0.5).ceil() as usize,
            (base * 0.7).ceil() as usize,
            (base * 1.0).ceil() as usize,
        ]
    };

    // Clamp to reasonable range
    let candidates: Vec<usize> = candidates.into_iter()
        .map(|s| s.clamp(3, 30))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    log::info!("SOM autotune: evaluating {} grid configurations for {} data points", candidates.len(), n);

    // Use fewer training epochs for autotune evaluation
    let autotune_epochs = if fast { 30 } else { 50 };

    // Subsample data for faster evaluation if dataset is large
    let eval_data: Vec<Vec<f32>> = if n > 5000 {
        let step = n / 5000;
        data.iter().step_by(step.max(1)).take(5000).cloned().collect()
    } else {
        data.to_vec()
    };

    let mut results: Vec<AutotuneCandidate> = Vec::new();

    for &side in &candidates {
        let config = SomConfig {
            grid_width: side,
            grid_height: side,
            input_dim,
            num_epochs: autotune_epochs,
            initial_learning_rate: 0.5,
            initial_radius: (side as f64) / 2.0,
        };

        let mut som = Som::new(&config, &eval_data);
        som.train(&eval_data, &config);

        let qe = som.quantization_error(&eval_data);
        let te = som.topographic_error(&eval_data);
        let active = som.active_cluster_count(&eval_data);

        log::info!("  Grid {}×{}: QE={:.4}, TE={:.4}, active={}/{}", 
            side, side, qe, te, active, side * side);

        results.push(AutotuneCandidate {
            grid_width: side,
            grid_height: side,
            num_clusters: side * side,
            active_clusters: active,
            quantization_error: qe,
            topographic_error: te,
            combined_score: 0.0, // computed below after normalization
        });
    }

    // Normalize and compute combined score
    // Lower QE is better, lower TE is better, more active clusters is better
    let qe_min = results.iter().map(|r| r.quantization_error).fold(f64::MAX, f64::min);
    let qe_max = results.iter().map(|r| r.quantization_error).fold(f64::MIN, f64::max);
    let te_min = results.iter().map(|r| r.topographic_error).fold(f64::MAX, f64::min);
    let te_max = results.iter().map(|r| r.topographic_error).fold(f64::MIN, f64::max);
    let ac_max = results.iter().map(|r| r.active_clusters).max().unwrap_or(1) as f64;

    for r in &mut results {
        let norm_qe = if (qe_max - qe_min).abs() > 1e-12 {
            (r.quantization_error - qe_min) / (qe_max - qe_min)
        } else { 0.0 };

        let norm_te = if (te_max - te_min).abs() > 1e-12 {
            (r.topographic_error - te_min) / (te_max - te_min)
        } else { 0.0 };

        let norm_ac = r.active_clusters as f64 / ac_max;

        // Score: minimize QE and TE, maximize active cluster ratio
        // Weights: QE (0.4), TE (0.3), active cluster utilization (0.3)
        r.combined_score = 1.0 - (0.4 * norm_qe + 0.3 * norm_te - 0.3 * norm_ac).clamp(0.0, 1.0);
    }

    // Pick the best
    results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap_or(std::cmp::Ordering::Equal));
    let best = &results[0];

    log::info!("SOM autotune best: {}×{} (score={:.4}, QE={:.4}, TE={:.4}, active={})",
        best.grid_width, best.grid_height, best.combined_score,
        best.quantization_error, best.topographic_error, best.active_clusters);

    let best_config = SomConfig {
        grid_width: best.grid_width,
        grid_height: best.grid_height,
        input_dim,
        num_epochs: 128,
        initial_learning_rate: 0.5,
        initial_radius: (best.grid_width.max(best.grid_height) as f64) / 2.0,
    };

    AutotuneResult {
        best_grid: (best.grid_width, best.grid_height),
        best_config,
        candidates: results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_som_basic() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0],
            vec![0.5, 0.5], vec![0.2, 0.8], vec![0.8, 0.2], vec![0.3, 0.7],
        ];

        let config = SomConfig {
            grid_width: 3,
            grid_height: 3,
            input_dim: 2,
            num_epochs: 10,
            initial_learning_rate: 0.5,
            initial_radius: 2.0,
        };

        let mut som = Som::new(&config, &data);
        som.train(&data, &config);
        let labels = som.predict(&data);
        assert_eq!(labels.len(), 8);

        let qe = som.quantization_error(&data);
        assert!(qe < 1.0, "Quantization error should be reasonable");
    }

    #[test]
    fn test_u_matrix() {
        let data: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let config = SomConfig {
            grid_width: 2, grid_height: 2, input_dim: 2,
            num_epochs: 5, initial_learning_rate: 0.3, initial_radius: 1.0,
        };
        let mut som = Som::new(&config, &data);
        som.train(&data, &config);
        let u = som.u_matrix();
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].len(), 2);
    }

    #[test]
    fn test_topographic_error_range() {
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.1, (i as f32 * 0.05).sin()])
            .collect();
        let config = SomConfig {
            grid_width: 4, grid_height: 4, input_dim: 2,
            num_epochs: 20, initial_learning_rate: 0.5, initial_radius: 2.0,
        };
        let mut som = Som::new(&config, &data);
        som.train(&data, &config);
        let te = som.topographic_error(&data);
        assert!(te >= 0.0 && te <= 1.0, "Topographic error must be in [0,1]: {}", te);
    }

    #[test]
    fn test_autotune_returns_result() {
        let data: Vec<Vec<f32>> = (0..200)
            .map(|i| vec![i as f32 * 0.01, (i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()])
            .collect();
        let result = autotune(&data, 3, true);
        assert!(!result.candidates.is_empty());
        assert!(result.best_grid.0 >= 3);
        assert!(result.best_grid.1 >= 3);
        assert!(result.best_config.input_dim == 3);
    }

    #[test]
    fn test_som_save_load() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 1.0], vec![1.0, 0.0], vec![0.5, 0.5],
        ];
        let config = SomConfig {
            grid_width: 2, grid_height: 2, input_dim: 2,
            num_epochs: 5, initial_learning_rate: 0.3, initial_radius: 1.0,
        };
        let mut som = Som::new(&config, &data);
        som.train(&data, &config);

        let path = "/tmp/fga_test_som_state.json";
        som.save_json(path).unwrap();
        let loaded = Som::load_json(path).unwrap();
        assert_eq!(som.weights.len(), loaded.weights.len());
        assert_eq!(som.grid_width, loaded.grid_width);
        for (w1, w2) in som.weights.iter().zip(loaded.weights.iter()) {
            for (a, b) in w1.iter().zip(w2.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_with_grid_builder() {
        let config = SomConfig::new(16).with_grid(8, 8);
        assert_eq!(config.grid_width, 8);
        assert_eq!(config.grid_height, 8);
        assert!((config.initial_radius - 4.0).abs() < 1e-10);
    }
}
