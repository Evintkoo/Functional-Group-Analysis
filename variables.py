atom_list = ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
hybridization_list = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
is_atom_in_ring = ["total_atom_in_ring"]
is_atom_aromatic = ["total_atom_in_aromatic_ring"]
atomic_mass = ["atomic_mass"]
total_chirality = ["total_CHI_UNSPECIFIED", "total_CHI_TETRAHEDRAL_CW", "total_CHI_TETRAHEDRAL_CCW", "total_CHI_OTHER"]
bond_type = ["total_single", "total_double", "total_triple", "total_aromatic"]
total_conjugated = ["total_conjugated_bond"]
total_ring = ["total_bond_in_ring"]
stereochemistry_bond = ["total_STEREOZ", "total_STEREOE", "toal_STEREOANY", "total_STEREONONE"]
feature_list = atom_list + hybridization_list + is_atom_in_ring + is_atom_aromatic + atomic_mass + total_chirality + bond_type + total_conjugated + total_ring + stereochemistry_bond
selected_feature = ['C',
 'N',
 'O',
 'S',
 'F',
 'P',
 'Cl',
 'Br',
 'I',
 'SP',
 'SP2',
 'SP3',
 'SP3D',
 'total_atom_in_ring',
 'total_atom_in_aromatic_ring',
 'atomic_mass',
 'total_CHI_UNSPECIFIED',
 'total_CHI_TETRAHEDRAL_CW',
 'total_CHI_TETRAHEDRAL_CCW',
 'total_single',
 'total_double',
 'total_triple',
 'total_aromatic',
 'total_conjugated_bond',
 'total_bond_in_ring',
 'total_STEREOZ',
 'total_STEREOE',
 'total_STEREONONE']