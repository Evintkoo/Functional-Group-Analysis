/// Graph Attention Network (GAT) layers implemented with Burn.
/// Uses batched message-passing for GPU-accelerated computation via Metal/wgpu.

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
            attn: LinearConfig::new(self.out_features * 3, 1)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> GatLayer<B> {
    /// Forward pass for GAT layer using batched message passing.
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

        // Project all nodes and edges in batch: stays on GPU
        let h = self.w_node.forward(node_features); // [N, out_dim]

        if edge_index.is_empty() {
            return h;
        }

        let e = self.w_edge.forward(edge_features); // [E, out_dim]

        // Group edges by destination for scatter
        let mut dst_groups: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for (i, &[_, dst]) in edge_index.iter().enumerate() {
            if dst < num_nodes {
                dst_groups[dst].push(i);
            }
        }

        // Batch: gather [h_src || h_dst || e_edge] for ALL edges at once
        let num_edges = edge_index.len();
        let mut src_indices = Vec::with_capacity(num_edges);
        let mut dst_indices = Vec::with_capacity(num_edges);
        for &[s, d] in edge_index {
            src_indices.push(s);
            dst_indices.push(d);
        }

        // Gather source and destination node features for all edges
        let h_src = gather_rows::<B>(&h, &src_indices, out_dim, &device);  // [E, out_dim]
        let h_dst = gather_rows::<B>(&h, &dst_indices, out_dim, &device);  // [E, out_dim]

        // Concatenate [h_src, h_dst, e] → [E, 3*out_dim], then compute attention in batch
        let concat = Tensor::cat(vec![h_src.clone(), h_dst, e.clone()], 1); // [E, 3*out_dim]
        let scores_raw = self.attn.forward(concat); // [E, 1]

        // LeakyReLU on scores
        let scores = Tensor::max_pair(scores_raw.clone(), scores_raw.mul_scalar(0.2)); // LeakyReLU

        // Extract scores and messages to CPU for scatter-softmax
        // (scatter-softmax per destination node is not easily batched on GPU)
        let scores_data: Vec<f32> = scores.to_data().to_vec().unwrap();

        // Compute messages: h_src + e (all on GPU, then extract)
        let messages_t = h_src.add(e); // [E, out_dim]
        let messages_data: Vec<f32> = messages_t.to_data().to_vec().unwrap();
        let h_data: Vec<f32> = h.to_data().to_vec().unwrap();

        // Scatter: aggregate messages per destination with softmax attention
        let mut output_data = vec![0.0f32; num_nodes * out_dim];

        for dst in 0..num_nodes {
            let edges = &dst_groups[dst];
            if edges.is_empty() {
                for d in 0..out_dim {
                    output_data[dst * out_dim + d] = h_data[dst * out_dim + d];
                }
                continue;
            }

            // Softmax over this node's incoming edge scores
            let max_s = edges.iter().map(|&ei| scores_data[ei]).fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = edges.iter().map(|&ei| (scores_data[ei] - max_s).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            for (k, &ei) in edges.iter().enumerate() {
                let alpha = exp_scores[k] / sum_exp;
                for d in 0..out_dim {
                    output_data[dst * out_dim + d] += alpha * messages_data[ei * out_dim + d];
                }
            }
        }

        Tensor::<B, 1>::from_floats(&output_data[..], &device)
            .reshape([num_nodes, out_dim])
    }
}

/// Gather rows from a 2D tensor by indices. Returns [len(indices), cols].
fn gather_rows<B: Backend>(
    tensor: &Tensor<B, 2>,
    indices: &[usize],
    cols: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    if indices.is_empty() {
        return Tensor::<B, 2>::zeros([0, cols], device);
    }
    let idx_data: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let idx_tensor = Tensor::<B, 1, burn::tensor::Int>::from_ints(&idx_data[..], device);
    tensor.clone().select(0, idx_tensor)
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_gat_layer_output_shape() {
        let device = Default::default();
        let layer = GatLayerConfig::new(8, 4)
            .with_edge_feature_dim(3)
            .init::<TestBackend>(&device);

        let nodes = Tensor::<TestBackend, 2>::zeros([5, 8], &device);
        let edges = Tensor::<TestBackend, 2>::zeros([4, 3], &device);
        let edge_index = vec![[0, 1], [1, 2], [2, 3], [3, 4]];

        let out = layer.forward(nodes, &edge_index, edges);
        assert_eq!(out.dims(), [5, 4]);
    }

    #[test]
    fn test_gat_layer_no_edges() {
        let device = Default::default();
        let layer = GatLayerConfig::new(8, 4)
            .with_edge_feature_dim(3)
            .init::<TestBackend>(&device);

        let nodes = Tensor::<TestBackend, 2>::zeros([3, 8], &device);
        let edges = Tensor::<TestBackend, 2>::zeros([0, 3], &device);
        let edge_index: Vec<[usize; 2]> = vec![];

        let out = layer.forward(nodes, &edge_index, edges);
        assert_eq!(out.dims(), [3, 4]);
    }

    #[test]
    fn test_gat_layer_single_node() {
        let device = Default::default();
        let layer = GatLayerConfig::new(4, 2)
            .with_edge_feature_dim(3)
            .init::<TestBackend>(&device);

        let nodes = Tensor::<TestBackend, 2>::ones([1, 4], &device);
        let edges = Tensor::<TestBackend, 2>::zeros([0, 3], &device);

        let out = layer.forward(nodes, &[], edges);
        assert_eq!(out.dims(), [1, 2]);
    }

    #[test]
    fn test_gat_encoder_output_shape() {
        let device = Default::default();
        let encoder = GatEncoderConfig::new(8, 3)
            .with_hidden_dim(16)
            .with_output_dim(6)
            .with_num_layers(2)
            .init::<TestBackend>(&device);

        let nodes = Tensor::<TestBackend, 2>::zeros([5, 8], &device);
        let edges = Tensor::<TestBackend, 2>::zeros([4, 3], &device);
        let edge_index = vec![[0, 1], [1, 0], [1, 2], [2, 1]];

        let out = encoder.forward(nodes, &edge_index, edges);
        assert_eq!(out.dims(), [5, 6]);
    }

    #[test]
    fn test_global_attention_pool_output_shape() {
        let device = Default::default();
        let pool = GlobalAttentionPoolConfig::new(8).init::<TestBackend>(&device);

        let nodes = Tensor::<TestBackend, 2>::ones([10, 8], &device);
        let out = pool.forward(nodes);
        assert_eq!(out.dims(), [1, 8]);
    }

    #[test]
    fn test_global_attention_pool_single_node() {
        let device = Default::default();
        let pool = GlobalAttentionPoolConfig::new(4).init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let out = pool.forward(input.clone());
        assert_eq!(out.dims(), [1, 4]);
        // With single node, attention weight = 1.0 (softmax of single value)
        let out_data: Vec<f32> = out.to_data().to_vec().unwrap();
        let in_data: Vec<f32> = input.to_data().to_vec().unwrap();
        for (o, i) in out_data.iter().zip(in_data.iter()) {
            assert!((o - i).abs() < 1e-5, "Single node pool should preserve values");
        }
    }

    #[test]
    fn test_gat_encoder_deterministic() {
        let device = Default::default();
        let encoder = GatEncoderConfig::new(4, 2)
            .with_hidden_dim(8)
            .with_output_dim(4)
            .with_num_layers(1)
            .init::<TestBackend>(&device);

        let nodes = Tensor::<TestBackend, 2>::ones([3, 4], &device);
        let edges = Tensor::<TestBackend, 2>::ones([2, 2], &device);
        let edge_index = vec![[0, 1], [1, 2]];

        let out1 = encoder.forward(nodes.clone(), &edge_index, edges.clone());
        let out2 = encoder.forward(nodes, &edge_index, edges);

        let d1: Vec<f32> = out1.to_data().to_vec().unwrap();
        let d2: Vec<f32> = out2.to_data().to_vec().unwrap();
        assert_eq!(d1, d2, "Same input should produce same output");
    }
}
