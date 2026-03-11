/// Feature extraction from molecular graphs.
/// Produces fixed-size node feature vectors and edge feature vectors for GNN input.

use crate::smiles::{
    Bond, BondStereo, BondType, Chirality, Element, Hybridization, MolGraph,
};
use petgraph::graph::NodeIndex;

/// Dimension of node (atom) feature vector
pub const NODE_FEATURE_DIM: usize =
    Element::NUM_TYPES       // atom type one-hot (14)
    + 1                      // degree (normalized)
    + 1                      // formal charge (normalized)
    + Hybridization::NUM_TYPES // hybridization one-hot (7)
    + 1                      // is_aromatic
    + 1                      // is_in_ring
    + 1                      // atomic mass (normalized)
    + 3;                     // chirality one-hot (none, CW, CCW)
// Total: 14 + 1 + 1 + 7 + 1 + 1 + 1 + 3 = 29

/// Dimension of edge (bond) feature vector
pub const EDGE_FEATURE_DIM: usize =
    BondType::NUM_TYPES      // bond type one-hot (4)
    + 1                      // is_conjugated
    + 1                      // is_in_ring
    + BondStereo::NUM_TYPES; // stereo one-hot (3)
// Total: 4 + 1 + 1 + 3 = 9

/// Infer hybridization from atom connectivity and bond types in the graph.
pub fn infer_hybridization(graph: &MolGraph, node: NodeIndex) -> Hybridization {
    let atom = &graph[node];
    let mut double_count = 0u8;
    let mut triple_count = 0u8;
    let mut aromatic_count = 0u8;
    let degree = graph.edges(node).count();

    for edge in graph.edges(node) {
        match edge.weight().bond_type {
            BondType::Double => double_count += 1,
            BondType::Triple => triple_count += 1,
            BondType::Aromatic => aromatic_count += 1,
            _ => {}
        }
    }

    if triple_count > 0 {
        Hybridization::SP
    } else if aromatic_count > 0 || (double_count == 1 && degree <= 3) {
        Hybridization::SP2
    } else if double_count >= 2 {
        Hybridization::SP
    } else if degree <= 4 {
        match atom.element {
            Element::S | Element::P if degree > 2 => {
                if degree > 4 { Hybridization::SP3D } else { Hybridization::SP3 }
            }
            _ => Hybridization::SP3,
        }
    } else if degree == 5 {
        Hybridization::SP3D
    } else {
        Hybridization::SP3D2
    }
}

/// Check if atom is in a ring by examining ring-closure bonds.
pub fn is_in_ring(graph: &MolGraph, node: NodeIndex) -> bool {
    graph.edges(node).any(|e| e.weight().is_in_ring)
}

/// Extract feature vector for a single atom node.
pub fn atom_features(graph: &MolGraph, node: NodeIndex) -> Vec<f32> {
    let atom = &graph[node];
    let mut features = vec![0.0f32; NODE_FEATURE_DIM];
    let mut offset = 0;

    // Atom type one-hot
    let aidx = atom.element.index();
    if aidx < Element::NUM_TYPES {
        features[offset + aidx] = 1.0;
    }
    offset += Element::NUM_TYPES;

    // Degree (normalized by 6)
    let degree = graph.edges(node).count() as f32;
    features[offset] = degree / 6.0;
    offset += 1;

    // Formal charge (normalized)
    features[offset] = atom.formal_charge as f32 / 4.0;
    offset += 1;

    // Hybridization one-hot
    let hyb = infer_hybridization(graph, node);
    features[offset + hyb.index()] = 1.0;
    offset += Hybridization::NUM_TYPES;

    // Is aromatic
    features[offset] = if atom.is_aromatic { 1.0 } else { 0.0 };
    offset += 1;

    // Is in ring
    features[offset] = if is_in_ring(graph, node) { 1.0 } else { 0.0 };
    offset += 1;

    // Atomic mass (normalized by 200)
    features[offset] = atom.element.atomic_mass() / 200.0;
    offset += 1;

    // Chirality one-hot
    match atom.chirality {
        Chirality::None => features[offset] = 1.0,
        Chirality::Clockwise => features[offset + 1] = 1.0,
        Chirality::CounterClockwise => features[offset + 2] = 1.0,
    }

    features
}

/// Extract feature vector for a bond edge.
pub fn bond_features(bond: &Bond) -> Vec<f32> {
    let mut features = vec![0.0f32; EDGE_FEATURE_DIM];
    let mut offset = 0;

    // Bond type one-hot
    features[offset + bond.bond_type.index()] = 1.0;
    offset += BondType::NUM_TYPES;

    // Is conjugated
    features[offset] = if bond.is_conjugated { 1.0 } else { 0.0 };
    offset += 1;

    // Is in ring
    features[offset] = if bond.is_in_ring { 1.0 } else { 0.0 };
    offset += 1;

    // Stereo one-hot
    features[offset + bond.stereo.index()] = 1.0;

    features
}

/// Molecular-level graph features: adjacency, node features, edge features.
#[derive(Debug, Clone)]
pub struct MolecularFeatures {
    pub node_features: Vec<Vec<f32>>,   // [num_atoms x NODE_FEATURE_DIM]
    pub edge_features: Vec<Vec<f32>>,   // [num_bonds x EDGE_FEATURE_DIM]
    pub edge_index: Vec<[usize; 2]>,    // COO format edge list (bidirectional)
    pub num_atoms: usize,
    pub num_bonds: usize,
}

/// Extract all features from a molecular graph.
pub fn extract_features(graph: &MolGraph) -> MolecularFeatures {
    let num_atoms = graph.node_count();

    let node_feats: Vec<Vec<f32>> = graph
        .node_indices()
        .map(|n| atom_features(graph, n))
        .collect();

    let mut edge_feats = Vec::new();
    let mut edge_index = Vec::new();

    for edge in graph.edge_indices() {
        let (src, dst) = graph.edge_endpoints(edge).unwrap();
        let bf = bond_features(&graph[edge]);
        // Bidirectional edges for message passing
        edge_index.push([src.index(), dst.index()]);
        edge_index.push([dst.index(), src.index()]);
        edge_feats.push(bf.clone());
        edge_feats.push(bf);
    }

    MolecularFeatures {
        node_features: node_feats,
        edge_features: edge_feats,
        edge_index,
        num_atoms,
        num_bonds: graph.edge_count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smiles::parse_smiles;

    #[test]
    fn test_feature_dimensions() {
        let g = parse_smiles("CCO").unwrap();
        let feats = extract_features(&g);
        assert_eq!(feats.node_features.len(), 3);
        assert_eq!(feats.node_features[0].len(), NODE_FEATURE_DIM);
        // 2 bonds * 2 directions = 4 edge entries
        assert_eq!(feats.edge_index.len(), 4);
        assert_eq!(feats.edge_features[0].len(), EDGE_FEATURE_DIM);
    }

    #[test]
    fn test_carbon_features() {
        let g = parse_smiles("C").unwrap();
        let feats = extract_features(&g);
        let carbon = &feats.node_features[0];
        // Carbon index is 1 in Element enum
        assert_eq!(carbon[1], 1.0); // C one-hot
        assert!(carbon[Element::NUM_TYPES + 2 + Hybridization::NUM_TYPES + 2] > 0.0); // mass > 0
    }
}
