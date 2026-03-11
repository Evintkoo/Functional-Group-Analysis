/// Self-Organizing Map (SOM) for clustering graph-level latent embeddings.
/// Competitive learning on a 2D grid of neurons with neighborhood decay.

use rand::Rng;

/// SOM configuration.
#[derive(Debug, Clone)]
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
}

/// A trained Self-Organizing Map.
#[derive(Debug, Clone)]
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
}
