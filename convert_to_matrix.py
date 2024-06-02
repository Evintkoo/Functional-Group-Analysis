import pandas as pd
import numpy as np
import pysmiles
import networkx as nx

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
    # read the data
    df = pd.read_csv("C:/Users/Evint/Documents/Projects/Functional-Group-Analysis/250k_rndm_zinc_drugs_clean_3.csv")
    
    # read the smiles
    df['smiles'] = df['smiles'].apply(lambda s: s.replace('\n', ''))
    smiles = df["smiles"].values
    
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
    
    result.to_csv("data/matrix_data.csv")