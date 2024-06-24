import pandas as pd
import numpy as np
import pysmiles
import networkx as nx
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
from SOM_plus_clustering.modules.som import SOM


fn_group_dict = {
        "amines" : [('N', 'C', 1), ('N', 'C', 1)],
        "alcohol" : [('O', 'C', 1)],
        "ether" : [('O', 'C', 1), ('O', 'C', 1)],
        "alkyl halide": [('C', 'C', 1), ('C', 'O', 2), ('C', 'C', 1)],
        "thiol": [('S', 'C', 1)],
        "aldehyde": [('C', 'O', 2)],
        "aldehyde2": [('O', 'C', 2)],
        "ketone": [('O', 'C', 1), ('O', 'C', 1)],
        "amides": [('N', 'C', 1), ('N', "O", "2"), ('N', 'C', 1)],
        "sulfide": [('S', 'C', 1), ('S', 'C', 1)],
        "amines2" : [('N', 'C', 1), ('N', 'C', 1), ('N', 'C', 1)],
        "amines3" : [('N', 'C', 1)],
        "carbodiimides" : [('C', 'N', 2), ('C', 'N', 2)],
        "nitrates": [('N', 'O', 1), ('N', 'O', 1), ('N', 'O', 2)],
        "esters": [('C','C', 1), ('C', 'O', 2), ('C', 'O', 1)],
        "haloalkene": [ ('F', 'C', 1)], 
        "haloalkene2": [('Cl', 'C', 1)], 
        "haloalkene3": [('Br', 'C', 1)], 
        "haloalkene4": [('I', 'C', 1)],
        "imine": [('N', 'C', 2), ('N', 'N', 1)],
        "nitro": [('N', 'C', 1), ('N', 'O', 2), ('N', 'O', 1)],
        "cyanide": [('N', 'C', 3)],
        "isocyanate": [('C', 'N', 2), ('C', 'O', 2)],
        "azocompound": [('N', 'N', 2), ('N', 'C', 1)],
        "sulfoxide":[('O', 'S', 2)],
        "azido": [('N', 'N', 2), ('N', 'N', 2)],
        "nitroso": [('N', 'C', 1), ('N', 'O', 2)],
        "phospate": [('P', 'O', 1), ('P', 'O', 1), ('P', 'O', 1), ('P', 'O', 2)],
        "phospite": [('P', 'O', 1), ('P', 'O', 1), ('P', 'O', 1)],
        "isothio": [('C', 'N', 2), ('C', 'S', 2)],
        "thioamide": [('C', 'N', 1), ('C', 'S', 2)],
        "nitro2": [('N', 'C', 1), ('N', 'O', 2), ('N', 'O', 2)]
    }


# Define the ANN model for tokenizing with specific input and output sizes using Sequential
class TokenizerANN(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(TokenizerANN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),  # First hidden layer
            nn.ReLU(),
            nn.Linear(512, 256),       # Third hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),       # Fourth hidden layer
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
            nn.Linear(128, 256),       # Second hidden layer
            nn.ReLU(),
            nn.Linear(256, 512),       # Third hidden layer
            nn.ReLU(),
            nn.Linear(512, input_size)   # Decoder output layer
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
        
def find_list_fn_group(smile):
    # make list of bond functional group connection in digraph
    fn_group_dict = {
        "amines" : [('N', 'C', 1), ('N', 'C', 1)],
        "alcohol" : [('O', 'C', 1)],
        "ether" : [('O', 'C', 1), ('O', 'C', 1)],
        "alkyl halide": [('C', 'C', 1), ('C', 'O', 2), ('C', 'C', 1)],
        "thiol": [('S', 'C', 1)],
        "aldehyde": [('C', 'O', 2)],
        "aldehyde2": [('O', 'C', 2)],
        "ketone": [('O', 'C', 1), ('O', 'C', 1)],
        "amides": [('N', 'C', 1), ('N', "O", "2"), ('N', 'C', 1)],
        "sulfide": [('S', 'C', 1), ('S', 'C', 1)],
        "amines2" : [('N', 'C', 1), ('N', 'C', 1), ('N', 'C', 1)],
        "amines3" : [('N', 'C', 1)],
        "carbodiimides" : [('C', 'N', 2), ('C', 'N', 2)],
        "nitrates": [('N', 'O', 1), ('N', 'O', 1), ('N', 'O', 2)],
        "esters": [('C','C', 1), ('C', 'O', 2), ('C', 'O', 1)],
        "haloalkene1": [ ('F', 'C', 1)], 
        "haloalkene2": [('Cl', 'C', 1)], 
        "haloalkene3": [('Br', 'C', 1)], 
        "haloalkene4": [('I', 'C', 1)],
        "imine": [('N', 'C', 2), ('N', 'N', 1)],
        "nitro": [('N', 'C', 1), ('N', 'O', 2), ('N', 'O', 1)],
        "cyanide": [('N', 'C', 3)],
        "isocyanate": [('C', 'N', 2), ('C', 'O', 2)],
        "azocompound": [('N', 'N', 2), ('N', 'C', 1)],
        "sulfoxide":[('O', 'S', 2)],
        "azido": [('N', 'N', 2), ('N', 'N', 2)],
        "nitroso": [('N', 'C', 1), ('N', 'O', 2)],
        "phospate": [('P', 'O', 1), ('P', 'O', 1), ('P', 'O', 1), ('P', 'O', 2)],
        "phospite": [('P', 'O', 1), ('P', 'O', 1), ('P', 'O', 1)],
        "isothio": [('C', 'N', 2), ('C', 'S', 2)],
        "thioamide": [('C', 'N', 1), ('C', 'S', 2)],
        "nitro2": [('N', 'C', 1), ('N', 'O', 2), ('N', 'O', 2)]
    }
    sorted_fn_group_dict = {i:sorted(fn_group_dict[i]) for i in fn_group_dict}
    elements = smile.nodes(data='element')
    order = list(smile.edges(data='order'))
    order_dict = {(i,j):k for i,j,k in order}
    mol_fn_group = list()
    for nodes in list(smile):
        #print(list(smile.edges(nodes)))
        list_edges = list(smile.edges(nodes))
        list_edge_with_bond_order = list()
        for i, j in list_edges:
            try:
                list_edge_with_bond_order.append((elements[i],elements[j],order_dict[(i,j)]))
            except: 
                list_edge_with_bond_order.append((elements[i],elements[j],order_dict[(j,i)]))
        list_edges = sorted(list_edge_with_bond_order)
        if list_edges in list(sorted_fn_group_dict.values()):
            mol_fn_group.append(list(sorted_fn_group_dict)[list(sorted_fn_group_dict.values()).index(list_edges)])
    return mol_fn_group

def find_total_aromatic(smile):
    elements = smile.nodes(data='element')
    order = list(smile.edges(data='order'))
    order_dict = {(i,j):k for i,j,k in order}
    cycles = nx.cycle_basis(smile)
    is_aromatic = list()
    for paths in cycles:
        #print(paths)
        list_edge_with_bond_order = list()
        for i in range(len(paths)):
            if i == len(paths)-1:
                a, b = paths[i], paths[0]
            else :
                a, b = paths[i], paths[i+1]
            #print((a,b))
            try:
                list_edge_with_bond_order.append((elements[a],elements[b],order_dict[(a,b)]))
            except: 
                list_edge_with_bond_order.append((elements[a],elements[b],order_dict[(b,a)]))
        is_aromatic.append(all([i[2] == 1.5 for i in list_edge_with_bond_order]))
    return sum(is_aromatic)

def find_total_cycle_non_aromatic(smile):
    elements = smile.nodes(data='element')
    order = list(smile.edges(data='order'))
    order_dict = {(i,j):k for i,j,k in order}
    cycles = nx.cycle_basis(smile)
    is_aromatic = list()
    for paths in cycles:
        #print(paths)
        list_edge_with_bond_order = list()
        for i in range(len(paths)):
            if i == len(paths)-1:
                a, b = paths[i], paths[0]
            else :
                a, b = paths[i], paths[i+1]
            #print((a,b))
            try:
                list_edge_with_bond_order.append((elements[a],elements[b],order_dict[(a,b)]))
            except: 
                list_edge_with_bond_order.append((elements[a],elements[b],order_dict[(b,a)]))
        is_aromatic.append(all([i[2] == 1.5 for i in list_edge_with_bond_order]))
    return sum([not i for i in is_aromatic])

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
    for lower_limit, upper_limit in limit:
        filtered_df = df[(df['qed'] >= lower_limit) & (df['qed'] < upper_limit)]
        print("start batch", counter, "size", filtered_df.shape[0])
        
        # read the smiles
        filtered_df['smiles'] = filtered_df['smiles'].apply(lambda s: s.replace('\n', ''))
        smiles = filtered_df["smiles"].values
        
        # create molecule graph
        mol_graph = [pysmiles.read_smiles(smile, explicit_hydrogen=True, reinterpret_aromatic=True) for smile in smiles]
        
        # create list of molecule nodes
        element_list_mols = [smile.nodes(data='element') for smile in mol_graph]
        
        # find all of the unique atom
        all_elements = np.unique([x for y in [list([j for i, j in elements]) for elements in element_list_mols] for x in y])
        
        # accumulator of element
        list_element = list()
        
        # give the result in list_element
        for smile in mol_graph:
            elements = smile.nodes(data='element')
            element_list = list([j for i, j in elements])
            list_num_element = [element_list.count(i) for i in all_elements]
            list_element.append(list_num_element)
        
        # convert list_element to the dataframe
        list_element_df = pd.DataFrame(list_element, columns = all_elements)
        
        # get all of functional group, total aromatic molecule, cycle non aromatic
        list_fn_group_in_mols = [find_list_fn_group(i) for i in mol_graph]
        list_total_aromatic_mols = [find_total_aromatic(i) for i in mol_graph]
        list_cycle_non_aromatic = [find_total_cycle_non_aromatic(i) for i in mol_graph]
        
        # count all of functional group, total aromatic molecule, cycle non aromatic
        x = [[i.count(j) for j in list(fn_group_dict.keys())] for i in list_fn_group_in_mols]
        functional_group_df = pd.DataFrame(x, columns=list(fn_group_dict.keys()))
        functional_group_df["aromatic"] = list_total_aromatic_mols
        functional_group_df["cycle"] = list_cycle_non_aromatic
        
        frames = [list_element_df, functional_group_df]
        result = pd.concat(frames, axis=1)
        
        data = result

        # result.to_csv("data/matrix_data.csv")
        target_size = 3
        device = "cuda"
        
        # Create a new instance of the model
        loaded_model = TokenizerANN(input_size = data.shape[1],
                            output_size=4).to(device)
        
        # Load the saved state dictionary into the new model instance
        loaded_model.load_state_dict(torch.load("tokenizer_ann_with_decoder.pth"))
        tensor_X = torch.from_numpy(data.values).to(device)
        
        # tokenize data
        tokenized = loaded_model.encoder(torch.tensor(tensor_X, dtype=torch.float32)).to("cpu").detach().numpy()
        
        # cluster the data
        som_model = deep_som()
        som_model.fit(X=tokenized, epoch = 50)
        predict = som_model.predict(tokenized)
        
        
        filtered_df['label'] = predict
        
        # save the cluster center
        np.save(f"data/experiment_1/group_{counter}/cluster_center.npy", np.array(som_model.cluster_center_[-1]))
        # save the labeled data
        filtered_df.to_csv(f"data/experiment_1/group_{counter}/labeled_data.csv", index=False)
        # save the matrix data
        pd.DataFrame(data).to_csv(f"data/experiment_1/group_{counter}/matrix_data.csv", index=False)
        # save the tokenized data
        pd.DataFrame(tokenized).to_csv(f"data/experiment_1/group_{counter}/tokenized_data.csv", index=False)
        counter += 1
