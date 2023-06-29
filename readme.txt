Several data that might be interesting:
 1. CH bond
 2. CC single bond 
 3. CC 1.5 bond 
 4. CN single bond
 5. NH single bond
 6. CO single bond
?7. CN 1.5 bond
 7. 10 highest data correlation

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
        source code: data_analysis.ipynb
        table: data_analysis.ipynb
3. Clustering
    explanation: group the data, and find each group's characteristics.
        clustering using SOM algorithm, and save the cluster center (characteristics) in data
    theorem: Self Organizing matrix
    steps:
        load cleaned data
        load self organizing matrix model
        fit the model
        find the cluster center
        save cluster center
    file: 
        source code: clustering.ipynb
        table: 
4. Cluster and Data analysis:
    explanation: analyze the cluster and data characteristics
    theorem: correlation analysis