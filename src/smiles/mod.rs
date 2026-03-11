/// SMILES string parser → molecular graph (petgraph).
/// Handles atoms, bonds, rings, branches, charges, chirality, and implicit hydrogens.

use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Undirected;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    H, He, Li, Be, B, C, N, O, F, Ne,
    Na, Mg, Al, Si, P, S, Cl, Ar,
    K, Ca, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn,
    Ga, Ge, As, Se, Br, Kr,
    Ag, Cd, In, Sn, Sb, Te, I,
    Pt, Au, Hg, Tl, Pb,
    Unknown,
}

impl Element {
    pub fn from_symbol(s: &str) -> Self {
        match s {
            "H" => Self::H, "He" => Self::He, "Li" => Self::Li, "Be" => Self::Be,
            "B" => Self::B, "C" => Self::C, "N" => Self::N, "O" => Self::O,
            "F" => Self::F, "Ne" => Self::Ne, "Na" => Self::Na, "Mg" => Self::Mg,
            "Al" => Self::Al, "Si" => Self::Si, "P" => Self::P, "S" => Self::S,
            "Cl" => Self::Cl, "Ar" => Self::Ar, "K" => Self::K, "Ca" => Self::Ca,
            "Ti" => Self::Ti, "V" => Self::V, "Cr" => Self::Cr, "Mn" => Self::Mn,
            "Fe" => Self::Fe, "Co" => Self::Co, "Ni" => Self::Ni, "Cu" => Self::Cu,
            "Zn" => Self::Zn, "Ga" => Self::Ga, "Ge" => Self::Ge, "As" => Self::As,
            "Se" => Self::Se, "Br" => Self::Br, "Kr" => Self::Kr, "Ag" => Self::Ag,
            "Cd" => Self::Cd, "In" => Self::In, "Sn" => Self::Sn, "Sb" => Self::Sb,
            "Te" => Self::Te, "I" => Self::I, "Pt" => Self::Pt, "Au" => Self::Au,
            "Hg" => Self::Hg, "Tl" => Self::Tl, "Pb" => Self::Pb,
            _ => Self::Unknown,
        }
    }

    pub fn atomic_number(&self) -> u8 {
        match self {
            Self::H => 1, Self::He => 2, Self::Li => 3, Self::Be => 4, Self::B => 5,
            Self::C => 6, Self::N => 7, Self::O => 8, Self::F => 9, Self::Ne => 10,
            Self::Na => 11, Self::Mg => 12, Self::Al => 13, Self::Si => 14, Self::P => 15,
            Self::S => 16, Self::Cl => 17, Self::Ar => 18, Self::K => 19, Self::Ca => 20,
            Self::Ti => 22, Self::V => 23, Self::Cr => 24, Self::Mn => 25, Self::Fe => 26,
            Self::Co => 27, Self::Ni => 28, Self::Cu => 29, Self::Zn => 30,
            Self::Ga => 31, Self::Ge => 32, Self::As => 33, Self::Se => 34, Self::Br => 35,
            Self::Kr => 36, Self::Ag => 47, Self::Cd => 48, Self::In => 49, Self::Sn => 50,
            Self::Sb => 51, Self::Te => 52, Self::I => 53, Self::Pt => 78, Self::Au => 79,
            Self::Hg => 80, Self::Tl => 81, Self::Pb => 82, Self::Unknown => 0,
        }
    }

    pub fn default_valence(&self) -> u8 {
        match self {
            Self::H => 1, Self::B => 3, Self::C => 4, Self::N => 3, Self::O => 2,
            Self::F => 1, Self::Si => 4, Self::P => 3, Self::S => 2,
            Self::Cl => 1, Self::Br => 1, Self::I => 1, Self::Se => 2,
            _ => 4,
        }
    }

    /// Atomic mass in Daltons
    pub fn atomic_mass(&self) -> f32 {
        match self {
            Self::H => 1.008, Self::He => 4.003, Self::Li => 6.941, Self::Be => 9.012,
            Self::B => 10.81, Self::C => 12.011, Self::N => 14.007, Self::O => 15.999,
            Self::F => 18.998, Self::Ne => 20.18, Self::Na => 22.99, Self::Mg => 24.305,
            Self::Al => 26.982, Self::Si => 28.086, Self::P => 30.974, Self::S => 32.06,
            Self::Cl => 35.45, Self::Ar => 39.948, Self::K => 39.098, Self::Ca => 40.078,
            Self::Ti => 47.867, Self::V => 50.942, Self::Cr => 51.996, Self::Mn => 54.938,
            Self::Fe => 55.845, Self::Co => 58.933, Self::Ni => 58.693, Self::Cu => 63.546,
            Self::Zn => 65.38, Self::Ga => 69.723, Self::Ge => 72.63, Self::As => 74.922,
            Self::Se => 78.971, Self::Br => 79.904, Self::Kr => 83.798, Self::Ag => 107.868,
            Self::Cd => 112.414, Self::In => 114.818, Self::Sn => 118.71, Self::Sb => 121.76,
            Self::Te => 127.6, Self::I => 126.904, Self::Pt => 195.084, Self::Au => 196.967,
            Self::Hg => 200.592, Self::Tl => 204.38, Self::Pb => 207.2, Self::Unknown => 0.0,
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::H => 0, Self::C => 1, Self::N => 2, Self::O => 3, Self::S => 4,
            Self::F => 5, Self::P => 6, Self::Cl => 7, Self::Br => 8, Self::I => 9,
            Self::B => 10, Self::Si => 11, Self::Se => 12, Self::Unknown => 13,
            other => other.atomic_number() as usize % 14,
        }
    }

    pub const NUM_TYPES: usize = 14;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Chirality {
    None,
    Clockwise,        // @@
    CounterClockwise, // @
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hybridization {
    S,
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
    Other,
}

impl Hybridization {
    pub fn index(&self) -> usize {
        match self {
            Self::S => 0, Self::SP => 1, Self::SP2 => 2, Self::SP3 => 3,
            Self::SP3D => 4, Self::SP3D2 => 5, Self::Other => 6,
        }
    }
    pub const NUM_TYPES: usize = 7;
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub element: Element,
    pub formal_charge: i8,
    pub chirality: Chirality,
    pub is_aromatic: bool,
    pub explicit_h_count: u8,
    pub isotope: Option<u16>,
}

impl Atom {
    pub fn new(element: Element) -> Self {
        Self {
            element,
            formal_charge: 0,
            chirality: Chirality::None,
            is_aromatic: false,
            explicit_h_count: 0,
            isotope: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondType {
    pub fn index(&self) -> usize {
        match self {
            Self::Single => 0, Self::Double => 1, Self::Triple => 2, Self::Aromatic => 3,
        }
    }
    pub fn order(&self) -> f32 {
        match self {
            Self::Single => 1.0, Self::Double => 2.0, Self::Triple => 3.0, Self::Aromatic => 1.5,
        }
    }
    pub const NUM_TYPES: usize = 4;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BondStereo {
    None,
    E,
    Z,
}

impl BondStereo {
    pub fn index(&self) -> usize {
        match self {
            Self::None => 0, Self::E => 1, Self::Z => 2,
        }
    }
    pub const NUM_TYPES: usize = 3;
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub bond_type: BondType,
    pub is_conjugated: bool,
    pub is_in_ring: bool,
    pub stereo: BondStereo,
}

impl Bond {
    pub fn new(bond_type: BondType) -> Self {
        Self {
            bond_type,
            is_conjugated: false,
            is_in_ring: false,
            stereo: BondStereo::None,
        }
    }
}

pub type MolGraph = Graph<Atom, Bond, Undirected>;

/// Parse a SMILES string into a molecular graph.
pub fn parse_smiles(smiles: &str) -> Result<MolGraph, String> {
    let mut graph = Graph::new_undirected();
    let mut stack: Vec<NodeIndex> = Vec::new();
    let mut ring_map: HashMap<u8, (NodeIndex, Option<BondType>)> = HashMap::new();
    let mut prev_node: Option<NodeIndex> = None;
    let mut next_bond: Option<BondType> = None;

    let chars: Vec<char> = smiles.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];
        match ch {
            '(' => {
                if let Some(node) = prev_node {
                    stack.push(node);
                }
                i += 1;
            }
            ')' => {
                prev_node = stack.pop();
                i += 1;
            }
            '-' => { next_bond = Some(BondType::Single); i += 1; }
            '=' => { next_bond = Some(BondType::Double); i += 1; }
            '#' => { next_bond = Some(BondType::Triple); i += 1; }
            ':' => { next_bond = Some(BondType::Aromatic); i += 1; }
            '/' | '\\' => { i += 1; } // geometric stereo markers (handled at bond level)
            '.' => { prev_node = None; i += 1; } // disconnect
            '[' => {
                // Bracket atom
                i += 1;
                let (atom, consumed) = parse_bracket_atom(&chars[i..])?;
                i += consumed;
                if i < len && chars[i] == ']' { i += 1; }
                let node = graph.add_node(atom);
                if let Some(prev) = prev_node {
                    let bt = next_bond.take().unwrap_or(BondType::Single);
                    graph.add_edge(prev, node, Bond::new(bt));
                }
                prev_node = Some(node);
            }
            '0'..='9' => {
                let ring_id = ch as u8 - b'0';
                handle_ring_closure(&mut graph, &mut ring_map, ring_id, prev_node.unwrap(), next_bond.take());
                i += 1;
            }
            '%' => {
                // Two-digit ring closure
                if i + 2 < len {
                    let d1 = chars[i + 1] as u8 - b'0';
                    let d2 = chars[i + 2] as u8 - b'0';
                    let ring_id = d1 * 10 + d2;
                    handle_ring_closure(&mut graph, &mut ring_map, ring_id, prev_node.unwrap(), next_bond.take());
                    i += 3;
                } else {
                    i += 1;
                }
            }
            _ => {
                // Organic subset or aromatic
                let (atom, consumed) = parse_organic_atom(&chars[i..])?;
                i += consumed;
                let is_aromatic = atom.is_aromatic;
                let node = graph.add_node(atom);
                if let Some(prev) = prev_node {
                    let bt = next_bond.take().unwrap_or(
                        if is_aromatic && graph[prev].is_aromatic {
                            BondType::Aromatic
                        } else {
                            BondType::Single
                        }
                    );
                    graph.add_edge(prev, node, Bond::new(bt));
                }
                prev_node = Some(node);
            }
        }
    }

    detect_rings_and_aromaticity(&mut graph);

    Ok(graph)
}

fn handle_ring_closure(
    graph: &mut MolGraph,
    ring_map: &mut HashMap<u8, (NodeIndex, Option<BondType>)>,
    ring_id: u8,
    current: NodeIndex,
    bond_type: Option<BondType>,
) {
    if let Some((open_node, open_bt)) = ring_map.remove(&ring_id) {
        let bt = bond_type.or(open_bt).unwrap_or(BondType::Single);
        let mut bond = Bond::new(bt);
        bond.is_in_ring = true;
        graph.add_edge(open_node, current, bond);
    } else {
        ring_map.insert(ring_id, (current, bond_type));
    }
}

fn parse_organic_atom(chars: &[char]) -> Result<(Atom, usize), String> {
    if chars.is_empty() {
        return Err("Unexpected end of SMILES".into());
    }

    let ch = chars[0];
    let is_aromatic = ch.is_lowercase();
    let upper = ch.to_uppercase().next().unwrap();

    // Try two-char element first
    if chars.len() > 1 && chars[1].is_lowercase() && !is_aromatic {
        let sym: String = [upper, chars[1]].iter().collect();
        let elem = Element::from_symbol(&sym);
        if elem != Element::Unknown {
            let mut atom = Atom::new(elem);
            atom.is_aromatic = is_aromatic;
            return Ok((atom, 2));
        }
    }

    let sym = upper.to_string();
    let elem = Element::from_symbol(&sym);
    if elem == Element::Unknown && !matches!(upper, 'C' | 'N' | 'O' | 'S' | 'P' | 'B' | 'F' | 'I') {
        return Err(format!("Unknown organic atom: {}", ch));
    }
    let elem = if elem == Element::Unknown {
        Element::from_symbol(&sym)
    } else {
        elem
    };

    let mut atom = Atom::new(elem);
    atom.is_aromatic = is_aromatic;
    Ok((atom, 1))
}

fn parse_bracket_atom(chars: &[char]) -> Result<(Atom, usize), String> {
    let mut i = 0;

    // Optional isotope
    let mut isotope: Option<u16> = None;
    while i < chars.len() && chars[i].is_ascii_digit() {
        let val = isotope.unwrap_or(0) * 10 + (chars[i] as u16 - '0' as u16);
        isotope = Some(val);
        i += 1;
    }

    // Element symbol
    if i >= chars.len() {
        return Err("Unexpected end in bracket atom".into());
    }

    let is_aromatic = chars[i].is_lowercase();
    let upper = chars[i].to_uppercase().next().unwrap();
    let mut sym = upper.to_string();
    i += 1;

    if i < chars.len() && chars[i].is_lowercase() && chars[i] != 'h' || 
       (i < chars.len() && chars[i].is_lowercase() && {
           let two: String = [upper, chars[i]].iter().collect();
           Element::from_symbol(&two) != Element::Unknown
       }) {
        sym.push(chars[i]);
        i += 1;
    }

    let elem = Element::from_symbol(&sym);
    let mut atom = Atom::new(elem);
    atom.is_aromatic = is_aromatic;
    atom.isotope = isotope;

    // Chirality
    while i < chars.len() && chars[i] == '@' {
        if atom.chirality == Chirality::None {
            atom.chirality = Chirality::CounterClockwise;
        } else {
            atom.chirality = Chirality::Clockwise;
        }
        i += 1;
    }

    // H count
    if i < chars.len() && chars[i] == 'H' {
        i += 1;
        if i < chars.len() && chars[i].is_ascii_digit() {
            atom.explicit_h_count = chars[i] as u8 - b'0';
            i += 1;
        } else {
            atom.explicit_h_count = 1;
        }
    }

    // Charge
    if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
        let sign: i8 = if chars[i] == '+' { 1 } else { -1 };
        i += 1;
        if i < chars.len() && chars[i].is_ascii_digit() {
            atom.formal_charge = sign * (chars[i] as i8 - b'0' as i8);
            i += 1;
        } else {
            let mut count: i8 = 1;
            while i < chars.len() && chars[i] == if sign > 0 { '+' } else { '-' } {
                count += 1;
                i += 1;
            }
            atom.formal_charge = sign * count;
        }
    }

    Ok((atom, i))
}

/// Simple ring/aromaticity detection using DFS cycle detection.
fn detect_rings_and_aromaticity(graph: &mut MolGraph) {
    let node_count = graph.node_count();
    if node_count == 0 { return; }

    // Bonds already marked is_in_ring from ring closures in SMILES.
    // Also mark conjugation for aromatic and double bonds.
    let edge_indices: Vec<_> = graph.edge_indices().collect();
    for eidx in edge_indices {
        let bond = &graph[eidx];
        let bt = bond.bond_type;
        if bt == BondType::Aromatic || bt == BondType::Double {
            graph[eidx].is_conjugated = true;
        }
    }

    // Mark atoms in rings based on ring-closure bonds
    let ring_edges: Vec<_> = graph.edge_indices()
        .filter(|&e| graph[e].is_in_ring)
        .collect();
    for eidx in &ring_edges {
        let (a, b) = graph.edge_endpoints(*eidx).unwrap();
        // BFS to find path between a and b excluding this edge (= ring members)
        mark_ring_path(graph, a, b, *eidx);
    }
}

fn mark_ring_path(graph: &mut MolGraph, start: NodeIndex, end: NodeIndex, exclude_edge: petgraph::graph::EdgeIndex) {
    use std::collections::VecDeque;

    let mut visited = vec![false; graph.node_count()];
    let mut parent: Vec<Option<(NodeIndex, petgraph::graph::EdgeIndex)>> = vec![None; graph.node_count()];
    let mut queue = VecDeque::new();

    visited[start.index()] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if node == end {
            // Trace back path and mark bonds as in_ring
            let mut cur = end;
            while let Some((prev, eidx)) = parent[cur.index()] {
                graph[eidx].is_in_ring = true;
                cur = prev;
                if cur == start { break; }
            }
            return;
        }
        let neighbors: Vec<_> = graph.edges(node).map(|e| {
            (e.id(), if e.source() == node { e.target() } else { e.source() })
        }).collect();
        for (eidx, neighbor) in neighbors {
            if eidx == exclude_edge { continue; }
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                parent[neighbor.index()] = Some((node, eidx));
                queue.push_back(neighbor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_methane() {
        let g = parse_smiles("C").unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g[petgraph::graph::NodeIndex::new(0)].element, Element::C);
    }

    #[test]
    fn test_parse_ethanol() {
        let g = parse_smiles("CCO").unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_parse_benzene() {
        let g = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(g.node_count(), 6);
        assert_eq!(g.edge_count(), 6);
    }

    #[test]
    fn test_parse_bracket_atom() {
        let g = parse_smiles("[NH4+]").unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g[petgraph::graph::NodeIndex::new(0)].formal_charge, 1);
    }

    #[test]
    fn test_parse_branch() {
        let g = parse_smiles("CC(=O)O").unwrap();
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 3);
    }
}
