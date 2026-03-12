/// Variational Graph Autoencoder (VGAE) for molecular graphs.
/// Uses GAT encoder → global attention pooling → latent distribution → decoder.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

use crate::gnn::{
    GatEncoder, GatEncoderConfig, GlobalAttentionPool, GlobalAttentionPoolConfig,
};

/// VGAE configuration.
#[derive(Config, Debug)]
pub struct VgaeConfig {
    pub node_feature_dim: usize,
    pub edge_feature_dim: usize,
    #[config(default = "64")]
    pub hidden_dim: usize,
    #[config(default = "32")]
    pub gnn_output_dim: usize,
    #[config(default = "16")]
    pub latent_dim: usize,
    #[config(default = "3")]
    pub num_gnn_layers: usize,
}

/// Variational Graph Autoencoder.
#[derive(Module, Debug)]
pub struct Vgae<B: Backend> {
    encoder: GatEncoder<B>,
    pool: GlobalAttentionPool<B>,
    fc_mu: Linear<B>,
    fc_logvar: Linear<B>,
    decoder_fc1: Linear<B>,
    decoder_fc2: Linear<B>,
    decoder_fc3: Linear<B>,
    latent_dim: usize,
}

/// Output from VGAE forward pass.
pub struct VgaeOutput<B: Backend> {
    pub z: Tensor<B, 2>,           // Latent embedding [1, latent_dim]
    pub mu: Tensor<B, 2>,          // Mean [1, latent_dim]
    pub log_var: Tensor<B, 2>,     // Log variance [1, latent_dim]
    pub reconstructed: Tensor<B, 2>, // Reconstructed node features [num_nodes, node_feature_dim]
}

impl VgaeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Vgae<B> {
        let encoder_config = GatEncoderConfig::new(self.node_feature_dim, self.edge_feature_dim)
            .with_hidden_dim(self.hidden_dim)
            .with_output_dim(self.gnn_output_dim)
            .with_num_layers(self.num_gnn_layers);

        Vgae {
            encoder: encoder_config.init(device),
            pool: GlobalAttentionPoolConfig::new(self.gnn_output_dim).init(device),
            fc_mu: LinearConfig::new(self.gnn_output_dim, self.latent_dim).init(device),
            fc_logvar: LinearConfig::new(self.gnn_output_dim, self.latent_dim).init(device),
            decoder_fc1: LinearConfig::new(self.latent_dim, self.hidden_dim).init(device),
            decoder_fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim * 2).init(device),
            decoder_fc3: LinearConfig::new(self.hidden_dim * 2, self.node_feature_dim).init(device),
            latent_dim: self.latent_dim,
        }
    }
}

impl<B: Backend> Vgae<B> {
    /// Encode a molecular graph to a latent distribution.
    pub fn encode(
        &self,
        node_features: Tensor<B, 2>,
        edge_index: &[[usize; 2]],
        edge_features: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let node_embeddings = self.encoder.forward(node_features, edge_index, edge_features);
        let graph_embedding = self.pool.forward(node_embeddings); // [1, gnn_output_dim]

        let mu = self.fc_mu.forward(graph_embedding.clone());
        let log_var = self.fc_logvar.forward(graph_embedding);

        (mu, log_var)
    }

    /// Reparameterization trick: z = mu + std * epsilon
    pub fn reparameterize(&self, mu: Tensor<B, 2>, log_var: Tensor<B, 2>) -> Tensor<B, 2> {
        let std = (log_var.clone() / 2.0).exp();
        let eps = Tensor::random(
            mu.shape(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &mu.device(),
        );
        mu + std * eps
    }

    /// Decode latent vector to reconstructed node features.
    /// Outputs a single reconstructed feature vector (graph-level).
    pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(self.decoder_fc1.forward(z));
        let h = burn::tensor::activation::relu(self.decoder_fc2.forward(h));
        self.decoder_fc3.forward(h)
    }

    /// Decode a latent vector to multiple node features for a given number of atoms.
    pub fn decode_to_nodes(&self, z: Tensor<B, 2>, num_nodes: usize) -> Tensor<B, 2> {
        // Broadcast latent to each node position and decode
        let z_expanded = z.repeat_dim(0, num_nodes); // [num_nodes, latent_dim]
        let h = burn::tensor::activation::relu(self.decoder_fc1.forward(z_expanded));
        let h = burn::tensor::activation::relu(self.decoder_fc2.forward(h));
        self.decoder_fc3.forward(h)
    }

    /// Full forward pass: encode → reparameterize → decode.
    pub fn forward(
        &self,
        node_features: Tensor<B, 2>,
        edge_index: &[[usize; 2]],
        edge_features: Tensor<B, 2>,
        num_nodes: usize,
    ) -> VgaeOutput<B> {
        let (mu, log_var) = self.encode(node_features, edge_index, edge_features);
        let z = self.reparameterize(mu.clone(), log_var.clone());
        let reconstructed = self.decode_to_nodes(z.clone(), num_nodes);

        VgaeOutput {
            z,
            mu,
            log_var,
            reconstructed,
        }
    }

    /// Get graph-level latent embedding (mean, no sampling) for inference.
    pub fn embed(
        &self,
        node_features: Tensor<B, 2>,
        edge_index: &[[usize; 2]],
        edge_features: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let (mu, _) = self.encode(node_features, edge_index, edge_features);
        mu
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }
}

/// Compute VGAE loss: reconstruction + KL divergence.
pub fn vgae_loss<B: Backend>(
    reconstructed: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mu: Tensor<B, 2>,
    log_var: Tensor<B, 2>,
    kl_weight: f32,
) -> Tensor<B, 1> {
    // Reconstruction loss (MSE)
    let diff = reconstructed - target;
    let recon_loss = (diff.clone() * diff).mean();

    // KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    let kl = (Tensor::ones_like(&log_var) + log_var.clone() - mu.clone() * mu - log_var.exp())
        .sum()
        * (-0.5);

    recon_loss + kl * kl_weight
}

/// Training configuration for VGAE.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub num_epochs: usize,
    pub learning_rate: f64,
    pub kl_weight: f32,
    pub val_split: f32,
    pub output_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_epochs: 20,
            learning_rate: 1e-3,
            kl_weight: 0.001,
            val_split: 0.2,
            output_dir: "results".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_vgae() -> Vgae<B> {
        let device = Default::default();
        VgaeConfig::new(8, 3)
            .with_hidden_dim(16)
            .with_gnn_output_dim(8)
            .with_latent_dim(4)
            .with_num_gnn_layers(1)
            .init(&device)
    }

    fn dummy_inputs() -> (Tensor<B, 2>, Vec<[usize; 2]>, Tensor<B, 2>) {
        let device = Default::default();
        let nodes = Tensor::<B, 2>::ones([5, 8], &device);
        let edge_index = vec![[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 4]];
        let edges = Tensor::<B, 2>::ones([6, 3], &device);
        (nodes, edge_index, edges)
    }

    #[test]
    fn test_vgae_encode_shape() {
        let vgae = make_vgae();
        let (nodes, edge_index, edges) = dummy_inputs();
        let (mu, log_var) = vgae.encode(nodes, &edge_index, edges);
        assert_eq!(mu.dims(), [1, 4]);
        assert_eq!(log_var.dims(), [1, 4]);
    }

    #[test]
    fn test_vgae_reparameterize_shape() {
        let vgae = make_vgae();
        let device = Default::default();
        let mu = Tensor::<B, 2>::zeros([1, 4], &device);
        let log_var = Tensor::<B, 2>::zeros([1, 4], &device);
        let z = vgae.reparameterize(mu, log_var);
        assert_eq!(z.dims(), [1, 4]);
    }

    #[test]
    fn test_vgae_decode_shape() {
        let vgae = make_vgae();
        let device = Default::default();
        let z = Tensor::<B, 2>::zeros([1, 4], &device);
        let out = vgae.decode(z);
        // Decoder outputs [1, node_feature_dim=8]
        assert_eq!(out.dims(), [1, 8]);
    }

    #[test]
    fn test_vgae_decode_to_nodes_shape() {
        let vgae = make_vgae();
        let device = Default::default();
        let z = Tensor::<B, 2>::zeros([1, 4], &device);
        let out = vgae.decode_to_nodes(z, 5);
        assert_eq!(out.dims(), [5, 8]);
    }

    #[test]
    fn test_vgae_forward_shapes() {
        let vgae = make_vgae();
        let (nodes, edge_index, edges) = dummy_inputs();
        let output = vgae.forward(nodes, &edge_index, edges, 5);
        assert_eq!(output.mu.dims(), [1, 4]);
        assert_eq!(output.log_var.dims(), [1, 4]);
        assert_eq!(output.z.dims(), [1, 4]);
        assert_eq!(output.reconstructed.dims(), [5, 8]);
    }

    #[test]
    fn test_vgae_embed_shape() {
        let vgae = make_vgae();
        let (nodes, edge_index, edges) = dummy_inputs();
        let mu = vgae.embed(nodes, &edge_index, edges);
        assert_eq!(mu.dims(), [1, 4]);
    }

    #[test]
    fn test_vgae_latent_dim() {
        let vgae = make_vgae();
        assert_eq!(vgae.latent_dim(), 4);
    }

    #[test]
    fn test_vgae_loss_positive() {
        let device = Default::default();
        let recon = Tensor::<B, 2>::ones([5, 8], &device);
        let target = Tensor::<B, 2>::zeros([5, 8], &device);
        let mu = Tensor::<B, 2>::ones([1, 4], &device);
        let log_var = Tensor::<B, 2>::zeros([1, 4], &device);

        let loss = vgae_loss(recon, target, mu, log_var, 0.001);
        let loss_val: f32 = loss.into_scalar();
        assert!(loss_val > 0.0, "Loss should be positive");
    }

    #[test]
    fn test_vgae_loss_zero_reconstruction() {
        let device = Default::default();
        let tensor = Tensor::<B, 2>::ones([3, 4], &device);
        let mu = Tensor::<B, 2>::zeros([1, 2], &device);
        let log_var = Tensor::<B, 2>::zeros([1, 2], &device);

        let loss = vgae_loss(tensor.clone(), tensor, mu, log_var, 0.001);
        let loss_val: f32 = loss.into_scalar();
        // Reconstruction loss should be ~0, only KL term remains
        assert!(loss_val.abs() < 0.1, "Loss should be near zero for perfect reconstruction: {}", loss_val);
    }

    #[test]
    fn test_train_config_defaults() {
        let config = TrainConfig::default();
        assert_eq!(config.num_epochs, 20);
        assert!((config.kl_weight - 0.001).abs() < 1e-6);
        assert!((config.learning_rate - 1e-3).abs() < 1e-10);
    }
}
