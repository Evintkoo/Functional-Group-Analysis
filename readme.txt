Method analysis:
1. Data gathering 
    explanation: collect and filter data that want to be use. 
        take top 25% data based on its druglikeness score
    method theorem: graph, smiles
    steps:
        load data
        filter data
        count edges
        make table
        save table
    file: 
        source code: data_gathering.ipynb
        table: 250k_rndm_zinc_drugs_clean_3.csv
    src:
        data: https://www.kaggle.com/code/basu369victor/drug-molecule-generation-with-vae/notebook
2. Data cleaning
    explanation: clean data.
        removes duplicates, and manipulate data to be ready for next step
    Steps:
        remove duplicates
        add the fliped edge Data
        remove useless data
    file: 
        source code: data_cleaning.ipynb
        table: cleared_data.csv
3. Feature selection
    explanation: select the bond that has high variance
        using variance formula for each bond, select the variance that greater than 1
    theorem: variance
    steps: 
        count each bond variance
        remove the data that has variance less than 1
    file: 
        source code: feature_selection.ipynb
        table: selected_data.csv
4. Clustering
    explanation: group the data, and find each group's characteristics.
        clustering using SOM algorithm with kernel density estimator optimization, and save the cluster center (characteristics) in data
    theorem: Self Organizing matrix
    steps:
        load cleaned data
        load self organizing matrix model
        fit the model
        find the cluster center
        save cluster center
    file: 
        source code: clustering.ipynb
        table: clustercenters.csv, clustercenters.xlsx
5. Cluster and Data analysis:
    explanation: analyze the cluster and data characteristics
    theorem: correlation analysis
    steps:
        load selected_data.csv
        analyze the correlation
        variance analysis
        save all data in "Datas" files

What to Discuss:
    1. Statistics analysis:
        a. Why several bond has low variance?
        b. What is the distribution of each bond data?
        c. What is the characteristic of each cluster?
    2. Bonding/Inorganic Chemistry analysis:
        a. What is the bonding characteristic and atribute in these selected bond? ; strength, etc
        b. What relation for each cluster to the molecular weight of compound?
    3. Organic Chemistry analysis:
        a. What functional group that might be represented by the selected bond?
        b. What is the characteristic or atribute of these functional group?