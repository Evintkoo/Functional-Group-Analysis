/// Functional group detection via substructure pattern matching on molecular graphs.
///
/// Detects 22 common functional groups by inspecting atom connectivity,
/// bond types, and ring membership in petgraph molecular graphs.

use petgraph::visit::EdgeRef;
use std::collections::HashMap;

use crate::smiles::{Atom, Bond, BondType, Element, MolGraph};

/// All functional groups we detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionalGroup {
    Hydroxyl,        // -OH (not in COOH)
    Carboxyl,        // -C(=O)OH
    PrimaryAmine,    // -NH2
    SecondaryAmine,  // >NH (not amide)
    TertiaryAmine,   // >N< (not in ring, not amide)
    Amide,           // -C(=O)N
    Ester,           // -C(=O)O-
    Ether,           // C-O-C (not ester, not epoxide)
    Ketone,          // >C=O (not COOH, not amide, not ester)
    Aldehyde,        // -CHO
    Nitro,           // -NO2
    Sulfonyl,        // -SO2-
    Sulfoxide,       // -SO-
    Thiol,           // -SH
    Thioether,       // C-S-C
    Nitrile,         // -C≡N
    Halide,          // C-F, C-Cl, C-Br, C-I
    Phenyl,          // 6-membered aromatic carbocycle
    Heterocycle,     // ring containing N, O, or S
    Imine,           // C=N (not in ring)
    Phosphate,       // P(=O)(O)
    Epoxide,         // 3-membered ring with O
}

impl FunctionalGroup {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hydroxyl => "Hydroxyl (-OH)",
            Self::Carboxyl => "Carboxyl (-COOH)",
            Self::PrimaryAmine => "Primary Amine (-NH₂)",
            Self::SecondaryAmine => "Secondary Amine (>NH)",
            Self::TertiaryAmine => "Tertiary Amine (>N<)",
            Self::Amide => "Amide (-CONH-)",
            Self::Ester => "Ester (-COO-)",
            Self::Ether => "Ether (C-O-C)",
            Self::Ketone => "Ketone (>C=O)",
            Self::Aldehyde => "Aldehyde (-CHO)",
            Self::Nitro => "Nitro (-NO₂)",
            Self::Sulfonyl => "Sulfonyl (-SO₂-)",
            Self::Sulfoxide => "Sulfoxide (-SO-)",
            Self::Thiol => "Thiol (-SH)",
            Self::Thioether => "Thioether (C-S-C)",
            Self::Nitrile => "Nitrile (-C≡N)",
            Self::Halide => "Halide (C-X)",
            Self::Phenyl => "Phenyl (aromatic ring)",
            Self::Heterocycle => "Heterocycle",
            Self::Imine => "Imine (C=N)",
            Self::Phosphate => "Phosphate (-PO₄)",
            Self::Epoxide => "Epoxide",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Hydroxyl => "OH",
            Self::Carboxyl => "COOH",
            Self::PrimaryAmine => "NH2",
            Self::SecondaryAmine => "NH",
            Self::TertiaryAmine => "N<",
            Self::Amide => "CONH",
            Self::Ester => "COO",
            Self::Ether => "C-O-C",
            Self::Ketone => "C=O",
            Self::Aldehyde => "CHO",
            Self::Nitro => "NO2",
            Self::Sulfonyl => "SO2",
            Self::Sulfoxide => "SO",
            Self::Thiol => "SH",
            Self::Thioether => "C-S-C",
            Self::Nitrile => "CN",
            Self::Halide => "C-X",
            Self::Phenyl => "Ph",
            Self::Heterocycle => "HetCyc",
            Self::Imine => "C=N",
            Self::Phosphate => "PO4",
            Self::Epoxide => "Epox",
        }
    }

    pub const ALL: &'static [FunctionalGroup] = &[
        Self::Hydroxyl, Self::Carboxyl, Self::PrimaryAmine, Self::SecondaryAmine,
        Self::TertiaryAmine, Self::Amide, Self::Ester, Self::Ether,
        Self::Ketone, Self::Aldehyde, Self::Nitro, Self::Sulfonyl,
        Self::Sulfoxide, Self::Thiol, Self::Thioether, Self::Nitrile,
        Self::Halide, Self::Phenyl, Self::Heterocycle, Self::Imine,
        Self::Phosphate, Self::Epoxide,
    ];
}

/// Result of functional group detection for a single molecule.
#[derive(Debug, Clone)]
pub struct FGProfile {
    /// Count of each functional group found.
    pub counts: HashMap<FunctionalGroup, usize>,
    /// Total number of functional groups detected.
    pub total: usize,
}

impl FGProfile {
    pub fn count(&self, fg: FunctionalGroup) -> usize {
        *self.counts.get(&fg).unwrap_or(&0)
    }

    pub fn has(&self, fg: FunctionalGroup) -> bool {
        self.count(fg) > 0
    }

    /// Return sorted list of (FG, count) pairs present in this molecule.
    pub fn present(&self) -> Vec<(FunctionalGroup, usize)> {
        let mut v: Vec<_> = self.counts.iter()
            .filter(|(_, &c)| c > 0)
            .map(|(&fg, &c)| (fg, c))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

// ═══════════════════════════════════════════════════
// Helper functions for querying atom neighborhoods
// ═══════════════════════════════════════════════════

struct AtomEnv<'a> {
    graph: &'a MolGraph,
}

impl<'a> AtomEnv<'a> {
    fn new(graph: &'a MolGraph) -> Self {
        Self { graph }
    }

    fn atom(&self, idx: petgraph::graph::NodeIndex) -> &Atom {
        &self.graph[idx]
    }

    fn element(&self, idx: petgraph::graph::NodeIndex) -> &Element {
        &self.graph[idx].element
    }

    fn degree(&self, idx: petgraph::graph::NodeIndex) -> usize {
        self.graph.edges(idx).count()
    }

    /// Number of heavy (non-H) neighbors
    fn heavy_degree(&self, idx: petgraph::graph::NodeIndex) -> usize {
        self.graph.edges(idx)
            .filter(|e| {
                let other = if e.source() == idx { e.target() } else { e.source() };
                self.graph[other].element != Element::H
            })
            .count()
    }

    /// Compute implicit hydrogen count from valence rules.
    /// Implicit H = default_valence - (sum of bond orders) - |formal_charge|
    fn implicit_h_count(&self, idx: petgraph::graph::NodeIndex) -> usize {
        let atom = self.atom(idx);
        let valence = atom.element.default_valence() as f32;
        let bond_order_sum: f32 = self.graph.edges(idx)
            .map(|e| e.weight().bond_type.order())
            .sum();
        let charge_adj = (atom.formal_charge as f32).abs();
        let implicit = valence - bond_order_sum - charge_adj + atom.explicit_h_count as f32;
        implicit.max(0.0) as usize
    }

    /// Total H count: explicit (in graph) + explicit_h_count (bracket) + implicit (from valence)
    fn total_h_count(&self, idx: petgraph::graph::NodeIndex) -> usize {
        let explicit_in_graph = self.degree(idx) - self.heavy_degree(idx);
        let atom = self.atom(idx);
        // For bracket atoms, explicit_h_count is set; for organic subset, compute implicit
        if atom.explicit_h_count > 0 || atom.isotope.is_some() || atom.formal_charge != 0 {
            // Bracket atom: trust explicit_h_count
            explicit_in_graph + atom.explicit_h_count as usize
        } else {
            // Organic subset: compute from valence
            explicit_in_graph + self.implicit_h_count(idx)
        }
    }

    fn neighbors_with_bonds(&self, idx: petgraph::graph::NodeIndex) -> Vec<(petgraph::graph::NodeIndex, &'a Bond)> {
        self.graph.edges(idx)
            .map(|e| {
                let other = if e.source() == idx { e.target() } else { e.source() };
                (other, e.weight())
            })
            .collect()
    }

    fn has_neighbor(&self, idx: petgraph::graph::NodeIndex, elem: &Element, bond: BondType) -> bool {
        self.graph.edges(idx).any(|e| {
            let other = if e.source() == idx { e.target() } else { e.source() };
            &self.graph[other].element == elem && e.weight().bond_type == bond
        })
    }

    fn count_neighbors(&self, idx: petgraph::graph::NodeIndex, elem: &Element, bond: BondType) -> usize {
        self.graph.edges(idx).filter(|e| {
            let other = if e.source() == idx { e.target() } else { e.source() };
            &self.graph[other].element == elem && e.weight().bond_type == bond
        }).count()
    }

    fn neighbor_of_type(&self, idx: petgraph::graph::NodeIndex, elem: &Element, bond: BondType) -> Option<petgraph::graph::NodeIndex> {
        self.graph.edges(idx).find_map(|e| {
            let other = if e.source() == idx { e.target() } else { e.source() };
            if &self.graph[other].element == elem && e.weight().bond_type == bond {
                Some(other)
            } else {
                None
            }
        })
    }
}

/// Detect all functional groups in a molecular graph.
pub fn detect_functional_groups(graph: &MolGraph) -> FGProfile {
    let env = AtomEnv::new(graph);
    let mut counts: HashMap<FunctionalGroup, usize> = HashMap::new();

    // Track which carbonyls are "claimed" by COOH/amide/ester/aldehyde
    let mut claimed_carbonyls: Vec<petgraph::graph::NodeIndex> = Vec::new();
    // Track which oxygens are "claimed" by COOH/ester
    let mut claimed_oxygens: Vec<petgraph::graph::NodeIndex> = Vec::new();
    // Track which nitrogens are "claimed" by amide/nitro
    let mut claimed_nitrogens: Vec<petgraph::graph::NodeIndex> = Vec::new();

    // ── Pass 1: Detect complex groups first (to claim atoms) ──

    for idx in graph.node_indices() {
        let atom = env.atom(idx);

        // --- Carboxyl: C(=O)(OH) ---
        if atom.element == Element::C {
            let has_double_o = env.has_neighbor(idx, &Element::O, BondType::Double);
            if has_double_o {
                // Check for -OH neighbor (single-bonded O with no other heavy neighbors besides this C)
                for (nbr, bond) in env.neighbors_with_bonds(idx) {
                    if env.element(nbr) == &Element::O
                        && bond.bond_type == BondType::Single
                        && env.heavy_degree(nbr) == 1
                    {
                        *counts.entry(FunctionalGroup::Carboxyl).or_insert(0) += 1;
                        claimed_carbonyls.push(idx);
                        claimed_oxygens.push(nbr);
                        // Also claim the =O
                        if let Some(o_double) = env.neighbor_of_type(idx, &Element::O, BondType::Double) {
                            claimed_oxygens.push(o_double);
                        }
                        break;
                    }
                }
            }
        }

        // --- Nitro: N(=O)(=O) or N+(=O)(-O-) ---
        if atom.element == Element::N {
            let double_o_count = env.count_neighbors(idx, &Element::O, BondType::Double);
            let single_o_count = env.count_neighbors(idx, &Element::O, BondType::Single);
            if double_o_count >= 2 || (double_o_count >= 1 && single_o_count >= 1 && atom.formal_charge > 0) {
                *counts.entry(FunctionalGroup::Nitro).or_insert(0) += 1;
                claimed_nitrogens.push(idx);
            }
        }

        // --- Amide: C(=O)N ---
        if atom.element == Element::C && !claimed_carbonyls.contains(&idx) {
            let has_double_o = env.has_neighbor(idx, &Element::O, BondType::Double);
            let has_n = env.has_neighbor(idx, &Element::N, BondType::Single);
            if has_double_o && has_n {
                // Check that the N is not a nitro
                if let Some(n_idx) = env.neighbor_of_type(idx, &Element::N, BondType::Single) {
                    if !claimed_nitrogens.contains(&n_idx) {
                        *counts.entry(FunctionalGroup::Amide).or_insert(0) += 1;
                        claimed_carbonyls.push(idx);
                        claimed_nitrogens.push(n_idx);
                        if let Some(o_double) = env.neighbor_of_type(idx, &Element::O, BondType::Double) {
                            claimed_oxygens.push(o_double);
                        }
                    }
                }
            }
        }

        // --- Ester: C(=O)O-C ---
        if atom.element == Element::C && !claimed_carbonyls.contains(&idx) {
            let has_double_o = env.has_neighbor(idx, &Element::O, BondType::Double);
            if has_double_o {
                for (nbr, bond) in env.neighbors_with_bonds(idx) {
                    if env.element(nbr) == &Element::O
                        && bond.bond_type == BondType::Single
                        && !claimed_oxygens.contains(&nbr)
                        && env.heavy_degree(nbr) == 2  // O bonded to this C and another C
                        && env.has_neighbor(nbr, &Element::C, BondType::Single)
                    {
                        *counts.entry(FunctionalGroup::Ester).or_insert(0) += 1;
                        claimed_carbonyls.push(idx);
                        claimed_oxygens.push(nbr);
                        if let Some(o_double) = env.neighbor_of_type(idx, &Element::O, BondType::Double) {
                            claimed_oxygens.push(o_double);
                        }
                        break;
                    }
                }
            }
        }

        // --- Nitrile: C≡N ---
        if atom.element == Element::C && env.has_neighbor(idx, &Element::N, BondType::Triple) {
            *counts.entry(FunctionalGroup::Nitrile).or_insert(0) += 1;
            if let Some(n_idx) = env.neighbor_of_type(idx, &Element::N, BondType::Triple) {
                claimed_nitrogens.push(n_idx);
            }
        }

        // --- Phosphate: P with =O and -O ---
        if atom.element == Element::P {
            let double_o = env.count_neighbors(idx, &Element::O, BondType::Double);
            let single_o = env.count_neighbors(idx, &Element::O, BondType::Single);
            if double_o >= 1 && single_o >= 1 {
                *counts.entry(FunctionalGroup::Phosphate).or_insert(0) += 1;
            }
        }

        // --- Sulfonyl: S(=O)(=O) ---
        if atom.element == Element::S {
            let double_o = env.count_neighbors(idx, &Element::O, BondType::Double);
            if double_o >= 2 {
                *counts.entry(FunctionalGroup::Sulfonyl).or_insert(0) += 1;
            } else if double_o == 1 {
                *counts.entry(FunctionalGroup::Sulfoxide).or_insert(0) += 1;
            }
        }
    }

    // ── Pass 2: Simpler groups (respecting claims) ──

    for idx in graph.node_indices() {
        let atom = env.atom(idx);

        // --- Aldehyde: C(=O)H where C has exactly 1 heavy neighbor besides the O ---
        if atom.element == Element::C && !claimed_carbonyls.contains(&idx) {
            let has_double_o = env.has_neighbor(idx, &Element::O, BondType::Double);
            if has_double_o && env.heavy_degree(idx) <= 2 {
                let h_count = env.total_h_count(idx);
                if h_count >= 1 {
                    *counts.entry(FunctionalGroup::Aldehyde).or_insert(0) += 1;
                    claimed_carbonyls.push(idx);
                }
            }
        }

        // --- Ketone: C(=O) not already claimed ---
        if atom.element == Element::C && !claimed_carbonyls.contains(&idx) {
            if env.has_neighbor(idx, &Element::O, BondType::Double) && env.heavy_degree(idx) >= 2 {
                *counts.entry(FunctionalGroup::Ketone).or_insert(0) += 1;
                claimed_carbonyls.push(idx);
            }
        }

        // --- Hydroxyl: O-H not in COOH ---
        if atom.element == Element::O && !claimed_oxygens.contains(&idx) {
            if env.heavy_degree(idx) == 1 {
                let has_only_single = env.neighbors_with_bonds(idx)
                    .iter()
                    .all(|(_, b)| b.bond_type == BondType::Single);
                let h_count = env.total_h_count(idx);
                if has_only_single && h_count >= 1 {
                    *counts.entry(FunctionalGroup::Hydroxyl).or_insert(0) += 1;
                    claimed_oxygens.push(idx);
                }
            }
        }

        // --- Ether: C-O-C (not ester, not epoxide, not claimed) ---
        if atom.element == Element::O && !claimed_oxygens.contains(&idx) && !atom.is_aromatic {
            if env.heavy_degree(idx) == 2 {
                let nbrs: Vec<_> = env.neighbors_with_bonds(idx);
                let all_single_c = nbrs.iter().filter(|(n, b)| {
                    env.element(*n) == &Element::C && b.bond_type == BondType::Single
                }).count();
                if all_single_c >= 2 {
                    *counts.entry(FunctionalGroup::Ether).or_insert(0) += 1;
                    claimed_oxygens.push(idx);
                }
            }
        }

        // --- Primary amine: N with 2+ H (heavy_degree == 1) ---
        if atom.element == Element::N && !claimed_nitrogens.contains(&idx) && !atom.is_aromatic {
            let heavy_deg = env.heavy_degree(idx);
            let h_total = env.total_h_count(idx);

            if heavy_deg <= 1 && h_total >= 2 {
                *counts.entry(FunctionalGroup::PrimaryAmine).or_insert(0) += 1;
                claimed_nitrogens.push(idx);
            } else if heavy_deg == 2 && h_total >= 1 {
                *counts.entry(FunctionalGroup::SecondaryAmine).or_insert(0) += 1;
                claimed_nitrogens.push(idx);
            } else if heavy_deg >= 3 && h_total == 0 {
                *counts.entry(FunctionalGroup::TertiaryAmine).or_insert(0) += 1;
                claimed_nitrogens.push(idx);
            }
        }

        // --- Halide: C bonded to F, Cl, Br, or I ---
        if atom.element == Element::C {
            for (nbr, bond) in env.neighbors_with_bonds(idx) {
                if bond.bond_type == BondType::Single {
                    match env.element(nbr) {
                        Element::F | Element::Cl | Element::Br | Element::I => {
                            *counts.entry(FunctionalGroup::Halide).or_insert(0) += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        // --- Thiol: S-H ---
        if atom.element == Element::S && env.heavy_degree(idx) == 1 {
            if env.total_h_count(idx) >= 1 {
                *counts.entry(FunctionalGroup::Thiol).or_insert(0) += 1;
            }
        }

        // --- Thioether: C-S-C ---
        if atom.element == Element::S && env.heavy_degree(idx) == 2 && !atom.is_aromatic {
            let c_count = env.count_neighbors(idx, &Element::C, BondType::Single);
            if c_count >= 2 {
                *counts.entry(FunctionalGroup::Thioether).or_insert(0) += 1;
            }
        }

        // --- Imine: C=N (not in ring, not nitrile, not claimed) ---
        if atom.element == Element::C && !atom.is_aromatic {
            for (nbr, bond) in env.neighbors_with_bonds(idx) {
                if env.element(nbr) == &Element::N
                    && bond.bond_type == BondType::Double
                    && !claimed_nitrogens.contains(&nbr)
                    && !bond.is_in_ring
                {
                    *counts.entry(FunctionalGroup::Imine).or_insert(0) += 1;
                }
            }
        }
    }

    // ── Pass 3: Ring-based groups ──

    // Phenyl: count aromatic rings (6-membered all-carbon)
    // Heterocycle: rings containing heteroatoms
    // We approximate by counting aromatic atoms
    let aromatic_carbons: Vec<_> = graph.node_indices()
        .filter(|&idx| graph[idx].is_aromatic && graph[idx].element == Element::C)
        .collect();
    let aromatic_heteroatoms: Vec<_> = graph.node_indices()
        .filter(|&idx| graph[idx].is_aromatic && graph[idx].element != Element::C && graph[idx].element != Element::H)
        .collect();

    // Approximate: every 6 aromatic C atoms ≈ 1 phenyl ring
    // (fused rings counted separately is complex; this is a reasonable approximation)
    if aromatic_carbons.len() >= 5 {
        // More precise: count connected components of aromatic C atoms
        let phenyl_count = count_aromatic_rings(graph, &aromatic_carbons);
        if phenyl_count > 0 {
            *counts.entry(FunctionalGroup::Phenyl).or_insert(0) += phenyl_count;
        }
    }

    if !aromatic_heteroatoms.is_empty() {
        *counts.entry(FunctionalGroup::Heterocycle).or_insert(0) += aromatic_heteroatoms.len();
    }

    // Epoxide: 3-membered ring containing O
    // (Detected via ring_member atoms - this is an approximation)
    let epoxide_count = detect_epoxides(graph);
    if epoxide_count > 0 {
        *counts.entry(FunctionalGroup::Epoxide).or_insert(0) += epoxide_count;
    }

    let total = counts.values().sum();
    FGProfile { counts, total }
}

/// Count approximate number of aromatic rings by grouping aromatic C atoms.
fn count_aromatic_rings(graph: &MolGraph, aromatic_cs: &[petgraph::graph::NodeIndex]) -> usize {
    if aromatic_cs.is_empty() { return 0; }

    // BFS to find connected components of aromatic C atoms
    let aromatic_set: std::collections::HashSet<_> = aromatic_cs.iter().copied().collect();
    let mut visited = std::collections::HashSet::new();
    let mut ring_count = 0;

    for &start in aromatic_cs {
        if visited.contains(&start) { continue; }

        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.push(node);
            for edge in graph.edges(node) {
                let other = if edge.source() == node { edge.target() } else { edge.source() };
                if aromatic_set.contains(&other) && !visited.contains(&other) {
                    visited.insert(other);
                    queue.push_back(other);
                }
            }
        }

        // Each connected aromatic component of size n contains ~(n/4) rings for fused systems
        // Single ring = 6 atoms, naphthalene = 10 atoms (2 rings), etc.
        let n = component.len();
        if n >= 5 {
            ring_count += ((n as f64) / 4.5).ceil() as usize;
        }
    }

    ring_count
}

/// Detect epoxide (oxirane) motifs: O in a 3-membered ring.
fn detect_epoxides(graph: &MolGraph) -> usize {
    let mut count = 0;
    for idx in graph.node_indices() {
        if graph[idx].element == Element::O {
            let neighbors: Vec<_> = graph.edges(idx)
                .map(|e| {
                    if e.source() == idx { e.target() } else { e.source() }
                })
                .collect();
            // Check if any pair of O's neighbors are bonded to each other
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if graph.find_edge(neighbors[i], neighbors[j]).is_some() {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// Aggregate functional group profiles across many molecules.
#[derive(Debug, Clone)]
pub struct FGCensus {
    /// How many molecules contain each FG (at least once).
    pub prevalence: HashMap<FunctionalGroup, usize>,
    /// Total count of each FG across all molecules.
    pub total_count: HashMap<FunctionalGroup, usize>,
    /// Mean count per molecule for each FG.
    pub mean_count: HashMap<FunctionalGroup, f64>,
    /// Number of molecules analyzed.
    pub num_molecules: usize,
}

impl FGCensus {
    pub fn from_profiles(profiles: &[FGProfile]) -> Self {
        let num_molecules = profiles.len();
        let mut prevalence: HashMap<FunctionalGroup, usize> = HashMap::new();
        let mut total_count: HashMap<FunctionalGroup, usize> = HashMap::new();

        for profile in profiles {
            for &fg in FunctionalGroup::ALL {
                let c = profile.count(fg);
                if c > 0 {
                    *prevalence.entry(fg).or_insert(0) += 1;
                }
                *total_count.entry(fg).or_insert(0) += c;
            }
        }

        let mean_count: HashMap<FunctionalGroup, f64> = FunctionalGroup::ALL.iter()
            .map(|&fg| {
                let total = *total_count.get(&fg).unwrap_or(&0) as f64;
                (fg, total / num_molecules.max(1) as f64)
            })
            .collect();

        Self { prevalence, total_count, mean_count, num_molecules }
    }

    pub fn prevalence_pct(&self, fg: FunctionalGroup) -> f64 {
        let p = *self.prevalence.get(&fg).unwrap_or(&0) as f64;
        p / self.num_molecules.max(1) as f64 * 100.0
    }

    /// Return FGs sorted by prevalence (descending).
    pub fn sorted_by_prevalence(&self) -> Vec<(FunctionalGroup, usize, f64)> {
        let mut v: Vec<_> = FunctionalGroup::ALL.iter()
            .map(|&fg| {
                let prev = *self.prevalence.get(&fg).unwrap_or(&0);
                let pct = self.prevalence_pct(fg);
                (fg, prev, pct)
            })
            .filter(|(_, prev, _)| *prev > 0)
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

/// Compute functional group enrichment for a cluster relative to the population.
/// Enrichment ratio = (cluster_prevalence%) / (population_prevalence%).
/// Values > 1.0 mean the FG is over-represented in this cluster.
pub fn fg_enrichment(
    cluster_census: &FGCensus,
    population_census: &FGCensus,
) -> Vec<(FunctionalGroup, f64)> {
    let mut enrichments: Vec<(FunctionalGroup, f64)> = FunctionalGroup::ALL.iter()
        .filter_map(|&fg| {
            let pop_pct = population_census.prevalence_pct(fg);
            if pop_pct < 1.0 { return None; } // Skip very rare FGs
            let cluster_pct = cluster_census.prevalence_pct(fg);
            Some((fg, cluster_pct / pop_pct))
        })
        .collect();
    enrichments.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    enrichments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smiles::parse_smiles;

    #[test]
    fn test_detect_hydroxyl() {
        let graph = parse_smiles("CCO").unwrap(); // ethanol
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::Hydroxyl));
    }

    #[test]
    fn test_detect_carboxyl() {
        let graph = parse_smiles("CC(=O)O").unwrap(); // acetic acid
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::Carboxyl));
        assert!(!profile.has(FunctionalGroup::Hydroxyl)); // OH is part of COOH
    }

    #[test]
    fn test_detect_amine() {
        let graph = parse_smiles("CCN").unwrap(); // ethylamine
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::PrimaryAmine));
    }

    #[test]
    fn test_detect_ketone() {
        let graph = parse_smiles("CC(=O)C").unwrap(); // acetone
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::Ketone));
    }

    #[test]
    fn test_detect_halide() {
        let graph = parse_smiles("CCCl").unwrap(); // chloroethane
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::Halide));
    }

    #[test]
    fn test_detect_aromatic() {
        let graph = parse_smiles("c1ccccc1").unwrap(); // benzene
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::Phenyl));
    }

    #[test]
    fn test_detect_amide() {
        let graph = parse_smiles("CC(=O)NC").unwrap(); // N-methylacetamide
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::Amide));
        assert!(!profile.has(FunctionalGroup::Ketone));
    }

    #[test]
    fn test_heterocycle() {
        let graph = parse_smiles("c1ccncc1").unwrap(); // pyridine
        let profile = detect_functional_groups(&graph);
        assert!(profile.has(FunctionalGroup::Heterocycle));
    }
}
