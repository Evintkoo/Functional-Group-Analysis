/// Graph Attention Network (GAT) layers implemented with Burn.
/// Uses a simplified message-passing approach compatible with Burn's tensor API.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

/// Configuration for a single GAT layer.
#[derive(Config, Debug)]
pub struct GatLayerConfig {
    pub in_features: usize,
    pub out_features: usize,
    #[config(default = "9")]
    pub edge_feature_dim: usize,
}

/// A single Graph Attention layer with edge features.
/// Uses additive attention: score = LeakyReLU(a^T [Wh_src || Wh_dst || We_edge])
#[derive(Module, Debug)]
pub struct GatLayer<B: Backend> {
    w_node: Linear<B>,
    w_edge: Linear<B>,
    attn: Linear<B>,
}

impl GatLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GatLayer<B> {
        GatLayer {
            w_node: LinearConfig::new(self.in_features, self.out_features)
                .with_bias(true)
                .init(device),
            w_edge: LinearConfig::new(self.edge_feature_dim, self.out_features)
                .with_bias(false)
                .init(device),
            // Attention: takes concatenated [h_src, h_dst, e_edge] of dim 3*out
            attn: LinearConfig::new(self.out_features * 3, 1)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> GatLayer<B> {
    /// Forward pass for GAT layer using manual message passing.
    ///
    /// node_features: [num_nodes, in_features]
    /// edge_index: pairs [src, dst]
    /// edge_features: [num_edges, edge_feature_dim]
    ///
    /// Returns: [num_nodes, out_features]
    pub fn forward(
        &self,
        node_features: Tensor<B, 2>,
        edge_index: &[[usize; 2]],
        edge_features: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let device = node_features.device();
        let num_nodes = node_features.dims()[0];
        let out_dim = self.w_node.forward(
            Tensor::<B, 2>::zeros([1, node_features.dims()[1]], &device)
        ).dims()[1];

        // Project all nodes: [num_nodes, out_features]
        let h = self.w_node.forward(node_features);

        if edge_index.is_empty() {
            return h;
        }

        // Project edge features: [num_edges, out_features]
        let e = self.w_edge.forward(edge_features);

        // Group edges by destination
        let mut dst_groups: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for (i, &[_, dst]) in edge_index.iter().enumerate() {
            if dst < num_nodes {
                dst_groups[dst].push(i);
            }
        }

        // Build output node-by-node via attention-weighted aggregation
        let h_data: Vec<f32> = h.to_data().to_vec().unwrap();
        let e_data: Vec<f32> = e.to_data().to_vec().unwrap();

        let mut output_data = vec![0.0f32; num_nodes * out_dim];

        for dst in 0..num_nodes {
            let edges = &dst_groups[dst];
            if edges.is_empty() {
                // Copy original features for isolated nodes
                for d in 0..out_dim {
                    output_data[dst * out_dim + d] = h_data[dst * out_dim + d];
                }
                continue;
            }

            // Compute attention scores
            let mut scores = Vec::with_capacity(edges.len());
            let mut messages = Vec::with_capacity(edges.len());

            for &ei in edges {
                let src = edge_index[ei][0];
                // Concatenate [h_src, h_dst, e_edge] for attention
                let mut concat = Vec::with_capacity(out_dim * 3);
                for d in 0..out_dim {
                    concat.push(h_data[src * out_dim + d]);
                }
                for d in 0..out_dim {
                    concat.push(h_data[dst * out_dim + d]);
                }
                for d in 0..out_dim {
                    concat.push(e_data[ei * out_dim + d]);
                }

                // Compute attention score via linear layer
                let concat_t = Tensor::<B, 2>::from_floats(
                    &concat[..], &device,
                ).reshape([1, out_dim * 3]);
                let score: f32 = self.attn.forward(concat_t)
                    .into_data().to_vec::<f32>().unwrap()[0];

                // LeakyReLU
                let score = if score >= 0.0 { score } else { 0.2 * score };
                scores.push(score);

                // Message = h_src + e_edge
                let mut msg = vec![0.0f32; out_dim];
                for d in 0..out_dim {
                    msg[d] = h_data[src * out_dim + d] + e_data[ei * out_dim + d];
                }
                messages.push(msg);
            }

            // Softmax over scores
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            // Weighted sum
            for (k, msg) in messages.iter().enumerate() {
                let alpha = exp_scores[k] / sum_exp;
                for d in 0..out_dim {
                    output_data[dst * out_dim + d] += alpha * msg[d];
                }
            }
        }

        Tensor::<B, 1>::from_floats(&output_data[..], &device)
            .reshape([num_nodes, out_dim])
    }
}

/// Multi-layer GAT encoder configuration.
#[derive(Config, Debug)]
pub struct GatEncoderConfig {
    pub node_feature_dim: usize,
    pub edge_feature_dim: usize,
    #[config(default = "64")]
    pub hidden_dim: usize,
    #[config(default = "32")]
    pub output_dim: usize,
    #[config(default = "3")]
    pub num_layers: usize,
}

/// Multi-layer GAT encoder for molecular graphs.
#[derive(Module, Debug)]
pub struct GatEncoder<B: Backend> {
    input_proj: Linear<B>,
    layers: Vec<GatLayer<B>>,
    output_proj: Linear<B>,
}

impl GatEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GatEncoder<B> {
        let mut layers = Vec::new();

        for _ in 0..self.num_layers {
            layers.push(
                GatLayerConfig::new(self.hidden_dim, self.hidden_dim)
                    .with_edge_feature_dim(self.edge_feature_dim)
                    .init(device),
            );
        }

        GatEncoder {
            input_proj: LinearConfig::new(self.node_feature_dim, self.hidden_dim).init(device),
            layers,
            output_proj: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
        }
    }
}

impl<B: Backend> GatEncoder<B> {
    /// Encode node features through multi-layer GAT.
    /// Returns node-level embeddings [num_nodes, output_dim]
    pub fn forward(
        &self,
        node_features: Tensor<B, 2>,
        edge_index: &[[usize; 2]],
        edge_features: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mut h = self.input_proj.forward(node_features);

        for layer in &self.layers {
            let h_new = layer.forward(h.clone(), edge_index, edge_features.clone());
            // Residual connection + ReLU
            h = burn::tensor::activation::relu(h_new + h.clone());
        }

        self.output_proj.forward(h)
    }
}

/// Global attention pooling: aggregate node embeddings into a single graph-level embedding.
#[derive(Module, Debug)]
pub struct GlobalAttentionPool<B: Backend> {
    gate: Linear<B>,
}

#[derive(Config, Debug)]
pub struct GlobalAttentionPoolConfig {
    pub feature_dim: usize,
}

impl GlobalAttentionPoolConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GlobalAttentionPool<B> {
        GlobalAttentionPool {
            gate: LinearConfig::new(self.feature_dim, 1).init(device),
        }
    }
}

impl<B: Backend> GlobalAttentionPool<B> {
    /// Pool node embeddings to graph-level vector via attention.
    /// node_embeddings: [num_nodes, feature_dim]
    /// Returns: [1, feature_dim]
    pub fn forward(&self, node_embeddings: Tensor<B, 2>) -> Tensor<B, 2> {
        let gate_scores = self.gate.forward(node_embeddings.clone()); // [N, 1]
        let attention = burn::tensor::activation::softmax(gate_scores, 0); // [N, 1]
        let weighted = node_embeddings * attention; // [N, feature_dim]
        weighted.sum_dim(0) // [1, feature_dim]
    }
}
