### Project Title : "Functional Group and Bonding Characteristic of High Drug-likeness Compound using Self Organizing Matrix Algorithm with K-Means++ as Initiator Algorithm".

**Authors:** Evint Leovonzko, Callixta Cahyaningrum, Rachmania Ulwani

**Year:** 2023

### Source Code Explanation

These are the stages of the methodology in this paper:

#### 1. Data gathering
***Brief Expalantion***

**Convert molecules into graph, and collect the necessary data.**

**Steps**
1. Interpret the molecules with high druglikeness score in a graph data structures
2. Extract the node, edges, and the structure, and count the number of atoms, bonds, and the functional group
3. Save the data into a table

**Method Theorem**: Graph (node, edges), Algorithm (Depth First Search)

**Data Source**: https://www.kaggle.com/code/basu369victor/drug-molecule-generation-with-vae/notebook

**Files**

250k_rndm_zinc_drugs_clean_3.csv

Atomic-Bond/1-data_gathering.ipynb

Elements/1-data_gathering.ipynb

Functional-Group/1-data_gathering.ipynb

Atomic-Bond/Datas/encoded_data.csv

#### 2. Data Cleaning
***Brief Expalantion***

**Reformat the table after the data was gathered, make the data to be ready to analize**

**Steps**
1. Remove the duplicate of the data
2. Remove empty or 0 data

**Method Theorem**: -

**Files**
Atomic-Bond/2-data_cleaning.ipynb
Elements/1-data_gathering.ipynb
Functional-Group/1-data_gathering.ipynb
Atomic-Bond/Datas/cleared_data.csv
Elements/Datas/cleared_encoded_data.csv
Functional-Group/Datas/cleared_encoded_data.csv

#### 3. Feature Selection
***Brief Expalantion***

**Select the columns that is reasonable to be further analyzed**

**Steps**
1. Find the value of the variance in each columns
2. Pick the features (columns) that has high variance 
**Method Theorem**: Variance

**Files**
Atomic-Bond/3-feature_selection.ipynb
Elements/2-feature_selection.ipynb
Functional-Group/2-feature_selection.ipynb
Atomic-Bond/Datas/selected_feature_data.csv
Elements/Datas/selected_data.csv
Functional-Group/Datas/selected_data.csv

#### 4. Clustering
***Brief Expalantion***

**Implement Self Organizing Matrix with K-Means++ as initiator to cluster the data**

**Steps**
1. Convert the data (.csv file extention) into a data digital data structure (multidimensonal list)
2. Train the Self Organizing Matrix Model
3. Save the value of the neuron inside the model's matrix in table form

**Files**

Atomic-Bond/4-clustering.ipynb

Elements/3-clustering.ipynb

Functional-Group/3-clustering.ipynb

Atomic-Bond/Datas/Stats/clustercenters.xlsx

Atomic-Bond/Datas/Stats/clustercenters.csv

Elements/Datas/Stats/clustercenters.xlsx

Elements/Datas/Stats/clustercenters.csv

Functional-Group/Datas/Stats/clustercenters.xlsx

Functional-Group/Datas/Stats/clustercenters.csv

#### 5. Qualitative analysis
***Brief Expalantion***

**Calculate the statistics value of the collected data, and plot the feature's distribution**

**Steps**
1. Load the cleaned dataset
2. Plot the histogram of each features
3. Save the plot of the data
4. Save heatmap of correlation data

**Files**

Atomic-Bond/5-data_analysis.ipynb

Elements/4-data_analysis.ipynb

Functional-Group/4-data_analysis.ipynb

Atomic-Bond/Datas/Stats/Distribution

Atomic-Bond/Datas/Stats/correlation_heatmap.png

Atomic-Bond/Datas/Stats/variance_data.csv

Atomic-Bond/Datas/Stats/variance_data.xlsx

Elements/Datas/Stats/Distribution

Elements/Datas/Stats/correlation_heatmap.png

Elements/Datas/Stats/variance_data.csv

Elements/Datas/Stats/variance_data.xlsx

Functional-Group/Datas/Stats/Distribution

Functional-Group/Datas/Stats/correlation_heatmap.png

Functional-Group/Datas/Stats/variance_data.csv

Functional-Group/Datas/Stats/variance_data.xlsx


#### 6. Quantitative analysis
***Brief Expalantion***

**Analyze the data based on theorem and other literature reviews**

**LIST OF THINGS THAT COULD BE ANALIZED**
1. Statistics analysis:

    a. Why several bond has low variance?

    b. What is the distribution of each bond data?

    c. What is the characteristic of each cluster?

2. Bonding/Inorganic Chemistry analysis:

    a. What is the bonding characteristic and atribute in these selected bond? ; strength, etc

    b. What relation for each cluster to the molecular weight of compound?

    c. What is the characteristic of the atoms that highly involves in the clusters?

3. Organic Chemistry analysis:

    a. What functional group that might be represented by the selected bond?

    b. What is the characteristic or atribute of these functional group?