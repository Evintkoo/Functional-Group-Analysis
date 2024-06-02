import pandas as pd
import numpy as np
import pysmiles
from tqdm import tqdm
import networkx as nx
import json
import numpy as np
import os
from multiprocessing import Pool

def reshape_2d_array(array:np.array, target_size:int):
        """reshape any array with the same dimension to the same size

        Args:
            array (np.array): the unsorted and not standardized array (m,n)
            target_size (int): the shape of the array after reshaped

        Returns:
            array: the reshaped (padded or trunctacted) array with shape of (target_size, target_size)
        """
        
        # reshape the row of the array
        reshaped_row = []
        for vector in array:
            if len(vector) < target_size:
                    reshaped_vector = np.pad(vector, (0, target_size - len(vector)), 'constant')
            else:
                reshaped_vector = vector[:target_size]
            reshaped_row.append(reshaped_vector)
        reshaped_row = np.transpose(reshaped_row)
        
        # reshape the collumn of the array
        reshaped_collumn = []
        for vector in reshaped_row:
            if len(vector) < target_size:
                    reshaped_vector = np.pad(vector, (0, target_size - len(vector)), 'constant')
            else:
                reshaped_vector = vector[:target_size]
            reshaped_collumn.append(reshaped_vector)
            
        return np.transpose(reshaped_collumn)

# function for batching data
def batch_data(data, start_index, batch_size):
    """
    Split data into batches of specified size.
    
    Args:
    - data: list of data points (e.g., vectors)
    - batch_size: size of each batch
    
    Returns:
    - List of batches, where each batch is a list of data points
    """
    return [data[i:i + batch_size] for i in range(start_index, len(data), batch_size)]

if __name__ == "__main__":
    batch_size = 1024
    data_path = "C:/Users/Evint/Documents/Projects/Functional-Group-Analysis/250k_rndm_zinc_drugs_clean_3.csv"
    
    # read the csv data using pandas
    df = pd.read_csv(data_path)
    
    # take the collumn named "smiles"
    df['smiles'] = df['smiles'].apply(lambda s: s.replace('\n', ''))
    
    # take the string
    list_smiles_str = df["smiles"].values
    
    try:
        with open("data/batching_state.json", 'r') as f:
            batch = json.load(f)
            batch_number = batch["current_batch_number"]
            old_batch_size = batch["current_batch_size"]
            input_size = batch["current_input_size"]
    except:
        batch_number = 0
        old_batch_size = batch_size
        input_size = 64
        
    # batch the strings
    batches_smiles_str = batch_data(list_smiles_str, 
                                    start_index=(batch_number-1)*old_batch_size, 
                                    batch_size=batch_size) # list of list of string
    
        
    print("current batch", batch_number)
    
    # convert the data using batching system
    for batch_smile_str in batches_smiles_str:
        # convert the list of smiles to graph networkx
        list_mol_graph = [pysmiles.read_smiles(smile_str, explicit_hydrogen=True, 
                                            reinterpret_aromatic=True, 
                                            zero_order_bonds=True) for smile_str in batch_smile_str]
        
        # convert the networkx graph to adjacency matrix
        list_adj_matrix = [nx.adjacency_matrix(mol_graph).todense() for mol_graph in list_mol_graph]
        
        # reshape the array into standardized shape
        reshaped_adj_matrix = np.array([reshape_2d_array(array = adj_matrix,
                                                        target_size=input_size) for adj_matrix in list_adj_matrix])
        
        batching_progress = {"current_batch_number": batch_number,
                             "current_batch_size": batch_size,
                             "current_input_size": input_size}
        json.dump(batching_progress, open("data/batching_state.json", "w"))
        batch_number += 1
        json.dump(reshaped_adj_matrix.tolist(), open(f'data/vector_batches/batching_state_{batch_number}.json', "w"))
