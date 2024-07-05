# import packages

# general tools
import numpy as np

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
import math
import matplotlib.pyplot as plt

# custom SOM
from SOM_plus_clustering.modules.som import SOM


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = False):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for (smiles, y_val) in tqdm(zip(x_smiles, y)):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))

    return data_list

def resize_and_center_tensor(tensor, target_rows, target_cols):
    # Get the dimensions of the original tensor
    original_rows, original_cols = tensor.shape
    
    # Calculate the padding needed to center the tensor
    pad_top = (target_rows - original_rows) // 2
    pad_bottom = target_rows - original_rows - pad_top
    pad_left = (target_cols - original_cols) // 2
    pad_right = target_cols - original_cols - pad_left
    
    # Pad the tensor with zeros to center it
    padding = (pad_left, pad_right, pad_top, pad_bottom)  # (left, right, top, bottom)
    centered_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
    
    return centered_tensor

# Define the ANN model for tokenizing with specific input and output sizes using Sequential
class TokenizerANN(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(TokenizerANN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),  # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),       # Third hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),       # Fourth hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),        # Fifth hidden layer
            nn.ReLU(),
            nn.Linear(64, output_size) ,
            nn.Softmax()# Encoder output layer
        )
       
        self.decoder = nn.Sequential(
            nn.Linear(output_size, 64),          # Decoder input layer
            nn.ReLU(),
            nn.Linear(64, 128),        # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),       # Second hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),       # Third hidden layer
            nn.ReLU(),
            nn.Linear(128, input_size)   # Decoder output layer
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class deep_som:
      def __init__(self) -> None:
            self.layers = [SOM(m=16, n=8, dim=tokenized.shape[1], 
                  initiate_method="SOM++", 
                  neighbour_rad=0.1, 
                  learning_rate=0.1, 
                  distance_function="euclidean"),
                        SOM(m=8, n=4, dim=tokenized.shape[1], 
                  initiate_method="SOM++", 
                  neighbour_rad=0.1, 
                  learning_rate=0.1, 
                  distance_function="euclidean"),
                        SOM(m=4, n=2, dim=tokenized.shape[1], 
                  initiate_method="SOM++", 
                  neighbour_rad=0.1, 
                  learning_rate=0.1, 
                  distance_function="euclidean"),
                        SOM(m=2, n=2, dim=tokenized.shape[1], 
                  initiate_method="SOM++", 
                  neighbour_rad=0.1, 
                  learning_rate=0.1, 
                  distance_function="euclidean")]
      
      def fit(self, X, epoch=10):
            data = X
            for layer in self.layers:
                  layer.fit(data, epoch=epoch)
                  data = layer.cluster_center_
      
      def predict(self, X):
            return self.layers[-1].predict(X)
      
      @property
      def cluster_center_(self):
            return [layer.cluster_center_ for layer in self.layers]

if __name__ == "__main__":
    # edges of each groups
    edge_list = [0.39941986, 0.51981407, 0.69371681, 0.81411102]
    
    # read the data
    df = pd.read_csv("C:/Users/Evint/Documents/Projects/Functional-Group-Analysis/250k_rndm_zinc_drugs_clean_3.csv")
    
    limit = []
    for i in range(len(edge_list)):
        lower_limit = edge_list[i-1] if i != 0 else 0
        upper_limit = 1 if i == len(edge_list) else edge_list[i] 
        limit.append((lower_limit,upper_limit))
    limit.append((limit[-1][1],1))
    counter = 0
    for lower_limit, upper_limit in tqdm(limit):
        filtered_df = df[(df['qed'] >= lower_limit) & (df['qed'] < upper_limit)]
        print("start batch", counter, "size", filtered_df.shape[0])
        
        # read the smiles
        smiles = filtered_df['smiles']
        y = filtered_df["qed"]
        
        #create geometric graph
        geometric_graph = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles,y)
        
        target_size = (38, 74)
        
        for data in geometric_graph:
            data.x = resize_and_center_tensor(data.x, target_size[0], target_size[1])
            
        X = np.array([np.array(data.x).reshape(-1) for data in geometric_graph])
        
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        
        # Create a new instance of the model
        loaded_model = TokenizerANN(input_size = X.shape[1],
                            output_size=4).to(device)
        
        # Load the saved state dictionary into the new model instance
        loaded_model.load_state_dict(torch.load("tokenizer_ann_with_decoder.pth"))
        tensor_X = torch.from_numpy(X).to(device)
        
        # tokenize data
        tokenized = loaded_model.encoder(torch.tensor(tensor_X, dtype=torch.float32)).to("cpu").detach().numpy()
        
        # cluster the data
        som_model = deep_som()
        som_model.fit(X=tokenized, epoch = 50)
        predict = som_model.predict(tokenized)
        
        filtered_df['label'] = predict
        
        # save the cluster center
        np.save(f"data/experiment_2/group_{counter}/cluster_center.npy", np.array(som_model.cluster_center_[-1]))
        # save the labeled data
        filtered_df.to_csv(f"data/experiment_2/group_{counter}/labeled_data.csv", index=False)
        # save the matrix data
        pd.DataFrame(data).to_csv(f"data/experiment_2/group_{counter}/matrix_data.csv", index=False)
        # save the tokenized data
        pd.DataFrame(tokenized).to_csv(f"data/experiment_2/group_{counter}/tokenized_data.csv", index=False)
        counter += 1