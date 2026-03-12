Title
Characteristic of High Drug-likeness Compound using Self Organizing Matrix Algorithm with K-Means++ as Initiator Algorithm
Abstract
The discovery and development of drug-like compunds remain central to advancing pharmaceutical therapies, with drug-likeness as a crucial metric in predicting a compound's potential efficacy and bioavailability. Traditional approaches, such as Lipinski’s Rule of Five and ADMET scoring, have guided drug discovery by evaluating key molecular properties. However, advanced machine-learning techniques are now essential for identifying meaningful patterns in molecular structures. Self-organizing matrix (SOM) algorithms and optimized clustering methods like K-Means++ offer a robust framework for analyzing high-dimensional molecular data. This study implemented an Artificial Neural Network tokenizer and a Self-Organizing Map to cluster compounds based on their drug-likeness scores. This approach identified distinct molecular characteristics for each cluster, encompassing atomic elements, ring and aromatic properties, stereochemistry and chirality, and bond types. Correlation analysis between these molecular variables and the Quantitative Estimate of Druglikeness (QED) revealed that specific structural attributes can influence a compound’s overall drug quality. Additionally, qualitative comparisons with prior research confirmed the consistency of these findings and reinforced their relevance in guiding drug development efforts.
Introduction
The quality of a drug is important as it influences treatment efficacy and safety. Moreover, quality depends on correct manufacturing and storage; high-quality drugs are available when using rational buying procedures and when suppliers are reliable [1]. In addition, Health Canada is committed to ensuring timely access to safe, effective, and high-quality drugs [2].
Additionally, the synthesizability of a drug is also important since it is essential for the drug discovery process. Recently, MIT researchers have developed a machine learning model that proposes new molecules for the drug discovery process while ensuring the molecules it suggests can actually be synthesized in a laboratory [3]. This is crucial because if a chemist cannot actually make the molecule, its disease-fighting properties cannot be tested. Additionally, Generative Adversarial Networks (GANs) provide a valuable tool for exploring chemical space and optimizing known compounds for a desired functionality [4].
Drug likeness is a qualitative concept used in drug design for assessing how "druglike" a substance is with respect to factors like bioavailability [5]. It is estimated from the molecular structure before the substance is even synthesized and tested. Moreover, a druglike molecule exhibits properties such as solubility in both water and fat, potency at the biological target, ligand efficiency, lipophilic efficiency, and molecular weight. A traditional method to evaluate drug-likeness is to check compliance with Lipinski’s Rule of Five, which covers the number of hydrophilic groups, molecular weight, and hydrophobicity [5]. 
Regarding Lipinski’s rule of five, it is a rule of thumb to evaluate drug-likeness or determine if a chemical compound with a certain pharmacological or biological activity has chemical properties and physical properties that would likely make it an orally active drug in humans [6]. The rule describes molecular properties important for a drug’s pharmacokinetics in the human body, including their absorption, distribution, metabolism, and excretion (ADME). However, the rule does not predict if a compound is pharmacologically active [7].
In addition to Lipinski’s Rule of Five, there are several methods to measure the quality of a compound for being a drug. One such method is the Drug Score, which combines drug-likeness, cLogP, logS, molecular weight, and toxicity risks in one handy value that may be used to judge the compound’s overall potential to qualify for a drug [8]. Another method is the ADMET score, which evaluates the drug-likeness of compounds based on 18 weighted ADMET properties [9]. The ADMET score was able to distinguish withdrawn drugs from approved drugs with statistically significant accuracy.
Equally important, graph representations of molecules are commonly used in drug discovery and AI-driven drug discovery [10]. These representations are based on the idea that a molecule can be represented as a graph, where atoms are nodes and bonds are edges. Graph-based molecular representation learning (MRL) is a key step to building the connection between machine learning and chemical science, as it encodes molecules as numerical vectors preserving the molecular structures and features. On top of these representations, downstream tasks such as property prediction can be performed [11].
Additionally, another way to analyze molecular bonding is through electrostatic potential maps, also known as electrostatic potential energy maps or molecular electrical potential surfaces. These maps illustrate the charge distributions of molecules three-dimensionally and allow us to visualize variably charged regions of a molecule [12]. Knowledge of the charge distributions can be used to determine how molecules interact with one another. 
Furthermore, a Self-Organizing Map (SOM) is a type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional representation of the input space of the training samples, called a map [13]. SOMs are useful for visualizing high-dimensional data and for clustering data based on similarity [14]. Moreover, they can be used in combination with other clustering methods to increase efficiency [15].
This paper implemented molecular attributes to represent molecular data with a self-organizing matrix to extract the characteristics of the data that was gathered. This paper hypothesizes that there would be several patterns of atoms, atomic bonds, and functional groups that uniquely appear inside the molecule of high drug-like compounds. The literature review section explains the different ideas and concepts that were used in this study. The methodology section explains the experiment procedure of this study. In the result section, there are several explanations of the data that was collected and highlights its important trend. The discussion section explains the reasoning behind the result and the meaning of the result. 
Literature Review
Drug Likeness Score
Drug-likeness is one of the biochemical properties of the molecules that are used to predict the chance of the compounds becoming drugs and can be used to remove compounds that will fail in clinical trials [16]. This virtual screening of drug candidates is important since it can enhance the probability for a drug to become successful and also depress the cost caused by drug development [17; 18; 16]. One of the quantitative approaches to drug-likeness is the quantitative estimate of drug-likeness (QED) score that is determined from eight molecular properties, ranging between zero (all properties unfavorable) and one (all properties favorable) [19]. However, the Synthetic accessibility (SA) score approach can also be used in determining the drug-likeness qualitatively, which ranges from 1 (very easy) to 10 (very difficult) [20; 21] 
Synthetic Accessibility Score
Synthetic accessibility is a method to measure how easy or difficult it is to synthesize a molecule. This score usually was expressed as a score from 1 to 10, which represents the higher the score the harder is the molecule to synthesize. There are several factors to determine the score of Synthetic Accessibility of a given molecule, which are the molecular structure, the availability of starting materials and reagents, the number and type of synthetic steps required, and the overall yield and efficiency of the synthesis [20].
Self Organizing Matrix
Self Organizing Matrix is an unsupervised learning that was introduced by Kohonen that is based on competitive learning [22]. As a clustering method, SOM is capable of exploring the data and projecting a high-dimensional pattern into a low-dimensional topology map. SOM was represented in a matrix, which consisted of a one or 2-dimensional grid of nodes called neurons. In competitive learning, neuron activation is a function of the distance between neuron weight and the input data. A neuron would be activated if its weight distance was the smallest among the other distances. The activated neuron would learn the most and its weights were modified which causes a neuron to learn repeatedly. This concept was further developed into Rival Penalized Controlled Competitive Learning. Hence, SOM preserves the topology of input data by assigning each datum to a neuron with the highest similarity [23].
In order to reduce the flaw, several methods and optimizations were proposed and published to modify the Serf Organizing Matrix Clustering. One of the methods is K-Means++ as the initiator value, which is an algorithm that selects only the first center uniformly random from the data. Each subsequent center is selected with a probability proportional to its contribution to the overall error given the previous selection. The total running time of the K-Means++ algorithm is O(nkd) [24].
Molecular Bonding
Bond formation is attributed to the decrease in energy, primarily due to electrons residing between the nuclei of the bonded atoms, shielding internuclear repulsions and maintaining atomic cohesion [25].
The Valence Bond Theory proposes that in a bond, one electron from each atom interacts with both nuclei, resulting in bond energies consistent with experimental values. This bonding involves electrons from the valence shells, with each atom contributing one electron, leading to a localized electron pair between adjacent atoms. The most widely used bonding model in organic compounds derives from the Valence Bond Theory (VBT) and hybridization procedures. Sigma bonds result from the overlap of hybrid orbitals or s orbitals of hydrogen on adjacent atoms, while pi bonds arise from the overlap of two adjacent p orbitals. The combination of these orbitals creates bonding and antibonding orbitals, with bonding orbitals having lower energy than antibonding ones. Lone pairs of electrons occupy nonbonding orbitals. The number of bonds between atoms is termed bond order, and this concept originates from molecular orbital theory [25].
In polar covalent bonds, there is an unequal sharing of bonding electrons due to differences in electronegativity. When an atom with higher electronegativity bonds to carbon, it gains a partial negative charge, while carbon possesses a partial positive charge. This charge separation leads to polar covalent bonds [25]. Complex molecules may have multiple polar bonds with varying degrees of polarity. The collective effect of these polar bonds can be understood through electrostatic potential surface plots [25]
Functional groups containing electronegative atoms often exhibit electron-withdrawing behavior through sigma bonds. This phenomenon, known as the inductive effect, results in bond polarizations, polarizations within molecules, and bond and molecular dipole moments [25]. A bond dipole represents the local moment associated with a polar covalent bond, wherein one end of the bond is δ+ and the other end is δ- [25].
In some cases, single Lewis structures cannot adequately describe a molecule. Multiple Lewis structures are drawn, and the actual molecule is a hybrid or mixture of these resonance structures [25].
Organic Functional Group
Functional groups are the units of connected atoms that are determined by the arrangement of the specific bonding of the atoms in organic compounds [26] and have relatively constant characteristics, although they are connected to different structures [27]. The functional groups have their own properties and behaviors, including the overall water and lipid solubility, route of administration, ability to interact with specific biological targets, mechanism of action, and route of metabolism and elimination [28]. Due to this, functional groups are an important aspect that is effectively used to characterize drug molecules [29], including drug-likeness.
Methodology 
Data Acquisition and Preparation
The molecular data used in this research was obtained from the National Center for Biotechnology Information (NCBI). Each molecule's data includes its molecular formula represented in SMILES (Simplified Molecular Input Line Entry System) format and its corresponding drug-likeness value. This dataset forms the basis for converting molecular representations into numerical feature vectors, enabling the analysis and prediction of molecular properties using machine learning techniques. The aim is to leverage these detailed molecular encodings to enhance the understanding of drug-like characteristics and facilitate the development of predictive models in cheminformatics.
Segmentation Methodology
The segmentation methodology begins with importing the necessary libraries, including NumPy for numerical operations and SciPy for signal processing. The input data, precisely the drug-likeness values from a labeled DataFrame, is extracted and converted into a NumPy array. A histogram of the "qed" values is then computed using 125 bins and normalizing the density of the histogram to reflect the probability density function.
The histogram data is inverted to identify local minima in the histogram, which represents potential segment boundaries. This inversion transforms peaks in the original histogram into valleys in the inverted data. A SciPy function detects these valleys in the inverted histogram data. The indices of the detected valleys correspond to the local minima in the original histogram, which can be used to segment the data based on the "qed" value distribution. This process allows for effectively identifying natural groupings within the data based on the distribution of drug-likeness scores.
Artificial Neural Network for Tokenization
An artificial neural network (ANN) model was developed for tokenizing molecular data using an encoder-decoder architecture implemented with the PyTorch library. The architecture consists of two main components: the encoder and the decoder.
The encoder network is constructed using a sequential stack of layers. It begins with a linear transformation mapping the input size to 256 neurons, followed by a ReLU activation function. Two additional hidden layers, each with 256 neurons and ReLU activations, follow this. A subsequent layer reduces the dimensionality to 128 neurons. The final layer in the encoder maps this 128-dimensional space to the output size with a Softmax activation function, ensuring the output is a probability distribution.
The decoder network employs a sequential structure. It starts with a linear transformation from the output size to 128 neurons, followed by a ReLU activation. Then, another hidden layer with 128 neurons and a ReLU activation follows. The subsequent two layers increase the dimensionality to 256 neurons, each followed by ReLU activations. The final layer maps this 256-dimensional space back to the input size, reconstructing the original input data.
The forward pass through the network involves passing the input through the encoder to obtain the encoded representation, which is then passed through the decoder to reconstruct the original input. This architecture effectively balances the reduction of dimensionality and the preservation of essential information through the decoding process, making it particularly suitable for tasks requiring high fidelity in data reconstruction, such as molecular tokenization.

Tokenizer Model Training Flow
The model training process involves iteratively updating the weights of the ANN to minimize the loss function over a specified number of epochs. The model is trained during each epoch using batches of input data and corresponding target values. The optimizer's gradients are reset at the start of each batch, and the model processes the inputs to generate outputs. The loss between the predicted outputs and actual targets is calculated using a loss function, followed by backpropagation to update the model's weights. This process ensures that the model gradually improves its predictions by minimizing the loss.
The average loss is computed and stored at the end of each epoch. Additionally, the model's performance is evaluated on a validation dataset to calculate the validation loss, which is also recorded. These losses are appended to history lists for further analysis. The minimum loss encountered during training is tracked, and the model's state is saved whenever a new minimum loss is achieved. This approach ensures that the best-performing model is preserved, allowing for robust model training and validation. The progress of each epoch, including the training and validation losses, is printed to monitor the training process.
Self-Organizing Map (Deep SOM) Model
The Self-Organizing Map model is a machine-learning algorithm for organizing and clustering high-dimensional data through a series of learning episodes. The model implements competitive learning to learn the data and find the best fit for it, facilitating a better representation of the data.
Model Architecture
The SOM model is initialized with 100 neurons for each group and a grid dimension of 10 by 10. The input dimension matches the number of output layers of the autoencoder model. The weight initiation was using SOM++, the neighborhood radius was set to 7.81, and the learning rate was 0.5. 
Model Training
The training method involves fitting the SOM model using the input data over a specified number of epochs. The training process involves sequentially fitting each SOM layer to the data, starting from the first layer. After fitting a layer, the data is updated to the cluster centers obtained from the current layer before being passed to the next layer. This hierarchical training process enables the model to capture and refine the underlying structure of the data progressively. In our study, the Deep SOM model was instantiated and trained on the tokenized molecular data with an epoch count of 50.
Model Prediction
The prediction method assigns clusters to the input data using the final SOM layer in the sequence. This method leverages the hierarchical clustering achieved by the previous layers to provide the final clustering output. After training the model, predictions were made on the same tokenized data to determine the cluster assignments.

Comprehensive Workflow
The workflow begins by defining an edge list for drug-likeness values. The dataset, containing molecular information, is read from a CSV file. The drug-likeness values are segmented into groups based on the specified edge list, creating limits for each group with the drug-likeness score limit, in which the drug-likeness score range for groups 0, 1, 2, 3, and 4 was 0-0.4, 0.4-0.51, 0.51-0.69, and 0.69-0.18, respectively. Then, the 100 clusters were taken from every group and averaged. The data is filtered for each segment to include only the molecules within the specified drug-likeness range. The SMILES strings of the filtered molecules are then vectorized using an appropriate function.
The vectorized data is transferred to a RTX 3060 GPU, ensuring efficient processing. A pre-trained TokenizerANN model, capable of encoding and decoding the molecular data, is loaded and used to tokenize the data, converting it into a numerical format suitable for clustering. The tokenized data is then clustered using a Deep SOM model, which hierarchically organizes the data into meaningful clusters.
Each segment's clustered data is labeled and saved, along with the cluster centers, the original matrix data, and the tokenized data. This process is repeated for each drug-likeness group, resulting in a comprehensive dataset with labeled clusters for further analysis. The workflow ensures systematic data processing, tokenization, and clustering, facilitating the study of molecular properties and their drug-likeness characteristics.
Figure 1. Tokenizer Model Training 



Figure 2. Tokenizer Model Training 

Result
Table 1. Characteristics of  Atomic Elements and Hybridization in Each Averaged Cluster Group 
Averaged
Cluster Group
C
N 
O
F
Sp2 
 Hybridization
Sp3
 Hybridization
0
17
2.34
2.14
0.83
15.77
7.29
1
17.85
2.9
1.95
0.92
16.06
8.15
2
18.14
2.22
2.61
0.87
17
7.48
3
17.89
1.67
3.23
0.85
17.9
6.73
4
17.7
1.42
3.37
0.79
18.43
5.02

*The value for P, Cl, S, and Br was omitted due to less than 0.5 
Table 1 shows that Carbon (C) is the most dominant element across all averaged cluster groups, with the highest value found in averaged cluster group 2 (18.14). The Nitrogen (N) content was highest in the averaged cluster group 1 (2.9), while the Oxygen (O) was highest in the averaged cluster 4. However, Fluorine (F) remained relatively low across the averaged cluster groups. The Sp2 hybridization shows the highest value in the averaged cluster group 4 (18.43), while the highest Sp3 hybridization value is found in the averaged cluster group 2 (8.15). 
Table 2. Characteristics of  Ring and Aromatic Properties in Each Averaged Cluster Group 
Averaged Cluster Group
Total Atom in Ring
Total Atom in Aromatic Ring
Total Bond in Ring 
0
15.09
10.77
16.38
1
16.43
11.88
17.79
2
16.14
11.73
17.45
3
14.76
11.2
15.91
4
14.86
11.51
16.08

Table 2 shows that the averaged cluster group 1 has the highest number of total atoms in rings (16.43), total bonds in rings (11.88), and total atoms in aromatic rings (11.88). It is also notable that all the cluster groups have similar amounts of total atoms in aromatic rings ranging from 10.77 - 11.88.   

Table 3. Characteristics of  Stereochemistry and Chirality in Each Averaged Cluster Group 
Averaged
Cluster Group
Unspecified Chirality
Clockwise
Chirality
Counter- clockwise Chirality 
No Stereochemistry
0
22.38
0.38
0.41
25.91
1
23.3
0.67
0.61
27.61
2
23.72
0.91
0.56
28.25
3
23.76
0.62
0.64
27.86
4
22.51
0.13
1.02
26.54

*The data for the E/Z stereochemistry table was omitted because the value was below 0.05 
Table 3 shows that the molecules with unspecified chirality and no stereochemistry dominate all of the averaged cluster groups, with averaged cluster group 3 having the highest value (24.25). However, all of the averaged cluster groups exhibit minimal clockwise and counter-clockwise chirality.
Table 4. Characteristics of Bond Types in Each Averaged Cluster Group 
Averaged
Cluster Group
Total Single Bond
Total Double Bond
Total Aromatic Bond 
Total Conjugated Bond
0
12.71
1.79
11.47
16.25
1
13.46
1.51
12.61
16.64
2
13.54
1.99
12.4
17.13
3
13.04
2.51
11.8
17.52
4
10.89
2.6
12.09
18.29

Table 4 shows that the averaged cluster group 4 has the highest total conjugated bonds (18.29) and total double bonds (2.6) but has the lowest amount of total single bonds (10.89). Additionally, the averaged cluster group 0 has the lowest total aromatic bonds (11.47) compared to the other 4 clusters. 
Figure 5. Correlation Coefficients Between Molecular Descriptors and QED (Quantitative Estimate Druglikeness) Values 

As depicted in Figure 5, several molecular descriptors demonstrated significant correlations with QED values. Notably, descriptors such as Sp2 hybridization (–0.4474), total conjugated bonds (–0.4525), total aromatic bonds (–0.3840), and unspecified chirality (–0.3780) were strongly negatively correlated with QED, indicating that an increase in these structural features tends to reduce a molecule's drug-likeness. Conversely, descriptors like Sp3 hybridization (0.2585) and stereochemical features, including clockwise (0.1216) and counterclockwise chirality (0.1297), showed moderate positive correlations, suggesting their potential to enhance drug-like properties. Additionally, the influence of halogen-specific descriptors (F, Cl, Br, I), total triple bonds, and Sp3d hybridization on QED was minimal, as evidenced by their near-zero correlation coefficients. Collectively, these findings underscore the critical role of unsaturation, conjugation, aromaticity, and stereochemistry in modulating drug-likeness, offering valuable insights for the rational design and optimization of novel drug candidates.
Discussion
This study aims to determine each drug group's molecular characteristics based on the drug-likeness score. The results show five different groups based on the distribution of the drug-likeness score. The results also describe the correlation between each variable and the drug-likeness score based on the compound's molecular properties.
Table 1 shows that group 0, which has the lowest drug-likeness score, shows a low average total carbon atom in the molecule. The number of carbon atoms might represent the aliphatic degrees of a molecule to predict solubility and improve the features and permeability of the compound [30]. The result suggests that carbon saturation might be the lowest and decreases the compound's solubility, leading to a low score of drug-likeness, which supports past research. This result is also supported by Figure 5, which shows that the correlation of the C atom is relatively high compared to other bonds. With the presented result, this study shows that the existence and number of C atoms in a molecule are highly correlated to the drug-likeness of a compound.
The existence of nitrogen atoms in every compound, with the average number of atoms in each compound mostly having a decreasing trend (Table 1). This might show that the nitrogen atom might be involved indirectly in the drug-likeness score, which might be impacted by minor changes to the drug-likeness score due to its position relative to the compound. The existence of nitrogen with a decreasing trend also supports past research that showed the impact of nitrogen on a drug compound is always cotext-dependent [31]. Since the position of the nitrogen atom is essential, it might show that the position of the nitrogen atom on the higher score of the group is close to or inside the compound’s ring.  This study shows that the existence of a nitrogen atom affects the drug-likeness of a compound, which might be caused by the position of the nitrogen in the molecule, which could be further researched.
Table 1 highlighted the hybridizations SP2 and SP3 might have a high influence on the drug-likeness of a compound. The average number of SP2 increases while SP3 shows a decreasing trend, which shows that the influence of SP2 might improve the quality of a drug. At the same time, the existence of SP3 might reduce the quality and cause the drug-likeness score of a compound to decrease. This result shows the significance of a molecule's structure to the drug-likeness score, which shows that the molecule's planar structure would increase the drug's quality, making the drug more reactive than the one with a more tetrahedral structure. Based on the correlation analysis, the SP2 and SP3 bond has a relatively high correlation to the drug-likeness score, which shows that SP2 and SP3 hybridization might have a high influence on the drug-likeness of a compound. This study's result suggests a strong influence between several hybridizations on a compound, which might directly impact the structure and shape of the compound itself.
According to Table 2, the total number of atoms in the ring tends to decline as the drug-likeness score rises. The number of atoms in a ring represents a molecule's complexity and the compound's molecular weight. The number of atoms in the ring also means the size of the ring itself, and the presence of bulky substituents at a reaction might increase the speed of the response on the C7 compared to C6 [32]. This might correlate to the number of atoms in the aromatic ring and the total bonds in the compound ring. These results show that the number of atoms in the aromatic ring should be the least to produce a high drug-likeness score compound. 
The number of total bonds in a ring and aromatic ring has a trend of decreasing along with the increase of drug-likeness score (Table 2). The number of atoms in an aromatic ring might represent the number of aromatic rings in a compound, which is an essential contributor to the developability parameter. The aromatic component's existence also affects the compound's solubility, and it is suggested that the carbo aromatic component be replaced with rings with heteroaromatic congeners to increase developability [33]. The correlation of this study also supports past research, which shows that the correlation of the average number of aromatic atoms in the ring and the total bond in the ring are negatively correlated with the drug-likeness score. 
Furthermore, the dominant value of the unspecified chirality (Table 3) and its negative correlation with QED (Figure 5) suggest that the molecules with poorly defined or absent stereochemical features are less likely to exhibit high drug-likeness. However, although the overall presence of chirality is limited, cluster 2 and cluster 4 show higher values for clockwise chirality and counterclockwise chirality (Table 3), respectively, with a positive correlation with the QED (Figure 5), suggesting that the chirality might play a critical role in drug efficacy by enabling selective interaction with biological targets [34]. However, the weak value of the correlation with the QED occurred since the chirality's value is primarily seen in target-specific interactions rather than influencing properties like solubility or absorption, which QED mainly measures​ [35]. The averaged cluster group 0, the average group cluster with the lowest drug-likeness score, has the highest proportion of non-stereochemical molecules (Table 3) and shows a relatively low QED trend. This supports the idea that the molecules that lack stereochemical stereochemistry (primarily found in complex molecules) negatively impact the overall drug-likeness since it will reduce specificity and have weaker interaction with the biological target [36]. 
Table 4 shows that the averaged cluster group 4 exhibits the highest total conjugated bonds (18.29) and total double bonds (2.6) but has the lowest amount of total single bonds (10.89). This may suggest that molecules in this cluster group potentially possess a higher degree of unsaturation and conjugation, potentially influencing their chemical reactivity and physical properties. 
The total aromatic bonds do not follow a clear linear trend across the cluster groups. The averaged cluster group 0 exhibits the lowest total aromatic bonds (11.47) compared to the other four clusters. These findings could potentially suggest that molecules in cluster group 0 may have less aromatic character, which might influence their stability and reactivity in certain chemical environments.
In organic chemistry, more substituted radicals (e.g., tertiary radicals) are generally more stable thermodynamically but can exhibit higher reactivity kinetically [37]. This apparent paradox arises from the difference between thermodynamic stability and kinetic lability. Additionally, Natural Bond Orbital (NBO) analysis reveals that the stability of some molecular structures is largely due to electronic delocalization effects, which can influence reactivity [38]. For instance, in NMP compounds, electronic delocalization extends from the conjugated bonds of the carbazole ring to the lone pairs of electronegative neighboring atoms, affecting their chemical properties [38].
The variation in bond types across the averaged cluster groups may have implications for the overall molecular structure and properties of the compounds within each group. For instance, the higher number of conjugated and double bonds in cluster group 4 might contribute to increased planarity and rigidity of the molecules, potentially affecting their interactions with biological targets or their behavior in different chemical processes. Cluster properties strongly depend on their geometry and size [37]. As the cluster size increases, properties evolve non-monotonically, with every atom potentially influencing the overall characteristics.
Figure 5 shows the correlation coefficients between molecular descriptors and Quantitative Estimate of Druglikeness (QED) values. Sp2 hybridization shows a strong negative correlation (-0.4473746544) with QED, while sp3 hybridization exhibits a positive correlation (0.2584741244). Total atoms in aromatic rings (-0.3792255113), total aromatic bonds (-0.3839731124), and total conjugated bonds (-0.4524639275) all show substantial negative correlations with QED. Most elements, including carbon (-0.2659795961), nitrogen (-0.1892791272), and oxygen (-0.2312073706), display negative correlations with QED. Fluorine is a notable exception with a slight positive correlation (0.03207767272). both clockwise (0.1216485582) and counter-clockwise (0.1296595366), show positive correlations with QED, while unspecified chirality (-0.3779756585) correlates negatively. Total single bonds correlate positively (0.1127563341) with QED, whereas double bonds (-0.267609662) and bonds in rings (-0.2670792633) show negative correlations.
The difference in results between this study and previous research could be attributed to several factors. One potential cause is the focus on the prepared and processed data, where some loss of information may have occurred during data processing. Additionally, this study did not incorporate 3D geometry reinterpretation, which might have led to some degree of inaccuracy in the interpretation of molecular structures, potentially contributing to misinterpretation of the sampled data. Furthermore, the sampling method used in this study, which involved a Self-Organizing Matrix to optimize and simplify data sampling, may require further investigation to assess the quality and reliability of the sampling approach.
In future research, researchers can focus on the discoveries of the actual drug-likeness score based on lab experiments and lab data. Since this experiment is only based on the available data, it is essential to research different methods further to analyze the correlation between drug-likeness score and the molecular attribute itself. On top of that, it is important to find several different approaches to interpreting the drug-likeness score to expand the understanding of the causal relationship between the drug-likeness score and molecular attributes. Lastly, further research could extend this study by implementing a deeper analysis of each drug-likeness group and the compound to better implement the presented result with other variables such as molecular shape. 
Conclusion
In conclusion, this study has demonstrated the effectiveness of using a Self Organizing Matrix (SOM) with K-Means++ as an initiator algorithm to analyze the molecular characteristics influencing drug-likeness. The findings indicate that several molecular attributes, such as carbon content, nitrogen positioning, and hybridization patterns (notably Sp2 and Sp3), significantly affect the drug-likeness. A higher degree of SP2 hybridization and fewer aromatic ring atoms were linked to improved drug-likeness, suggesting the importance of planar molecular structures. Additionally, the presence of chirality, though limited, plays a role in drug efficacy by enhancing interactions with biological targets. These insights provide valuable guidance for predicting drug-likeness in early drug discovery phases. However, further studies should focus more on experimental validation and further exploration of stereochemistry and molecular shape to improve the understanding of drug-likeness predictors.


References

American Society of Health-System Pharmacists. (2023). Functional Groups Characteristics and Roles. https://www.ashp.org/  
Anslyn, E. V., & Dougherty, D. A. (2006). Modern physical organic chemistry. University Science
Bertucci, C., Pistolozzi, M., & De Simone, A. (2010). Circular dichroism in drug discovery and development: An abridged review. Analytical and Bioanalytical Chemistry, 398(1), 155–166. https://doi.org/10.1007/s00216-010-3959-2  
Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012). Quantifying the chemical beauty of drugs. Nature chemistry, 4(2), 90–98. https://doi.org/10.1038/nchem.1243
Blanchard, A. E., Stanley, C., & Bhowmik, D. (2021). Using Gans with adaptive training data to search for new molecules. Journal of Cheminformatics, 13(1). https://doi.org/10.1186/s13321-021-00494-3
Cai, C., Lin, H., Wang, H., Xu, Y., Ouyang, Q., Lai, L., & Pei, J. (2022). Midruglikeness: Subdivisional drug-likeness prediction models using Active Ensemble Learning Strategies. Biomolecules, 13(1), 29. https://doi.org/10.3390/biom13010029 
Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons. Alexandria Engineering Journal, 53(4), 827-831. https://doi.org/10.1016/j.aej.2014.09.007
Crowley, D., Collins, C., Delargy, I., Laird, E., & Van Hout, M. C. (2017). Irish general practitioner attitudes toward decriminalisation and medical use of cannabis: Results from a national survey. Harm Reduction Journal, 14(1). https://doi.org/10.1186/s12954-016-0129-7  
David, L., Thakkar, A., Mercado, R., & Engkvist, O. (2020). Molecular representations in AI-Driven Drug Discovery: A review and practical guide. Journal of Cheminformatics, 12(1). https://doi.org/10.1186/s13321-020-00460-5 
Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of cheminformatics, 1, 1-11.
Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Cheminformatics, 1(1), 8-8. https://doi.org/10.1186/1758-2946-1-8
Guo, Z., Guo, K., Nan, B., Tian, Y., Iyer, R. G., Ma, Y., Wiest, O., Zhang, X., Wang, W., Zhang, C., & Chawla, N. V. (2023, April 20). Graph-based molecular representation learning. arXiv.org. https://arxiv.org/abs/2207.04869 
He, Z., Zhang, J., Shi, X. H., Hu, L. L., Kong, X., Cai, Y. D., & Chou, K. C. (2010). Predicting drug-target interaction networks based on functional groups and biological features. PloS one, 5(3), e9603. https://doi.org/10.1371/journal.pone.0009603 
Health Canada. (2018, June 22). Prescription drugs. Government of Canada . https://www.canada.ca/en/health-canada/services/health-care-system/pharmaceuticals.html
Kohonen, T., & Honkela, T. (2007). Kohonen network. Scholarpedia, 2(1), 1568. https://doi.org/10.4249/scholarpedia.1568 
Kotyrba, M., Volna, E., Jarusek, R., & Smolka, P. (2021). The use of conventional clustering methods combined with SOM to increase the efficiency. Neural Computing and Applications, 33(23), 16519–16531. https://doi.org/10.1007/s00521-021-06251-9 
Krämer, A., Hudson, P. S., Jones, M. R., & Brooks, B. R. (2020). Multi-phase boltzmann weighting: Accounting for local inhomogeneity in molecular simulations of water–octanol partition coefficients in the SAMPL6 challenge. Journal of Computer-Aided Molecular Design, 34(5), 471-483. https://doi.org/10.1007/s10822-020-00285-2
Kumar, R. (2021). Effects of stereoisomers on Drug Activity. American Journal of Biomedical Science &amp; Research, 13(3), 220–222. https://doi.org/10.34297/ajbsr.2021.13.001861  
Lee, K., Jang, J., Seo, S., Lim, J., & Kim, W. Y. (2022). Drug-likeness scoring based on unsupervised learning. Chemical Science, 13(2), 554-565.
Leeson, P. D., & Springthorpe, B. (2007). The influence of drug-like concepts on decision-making in medicinal chemistry. Nature reviews Drug discovery, 6(11), 881-890.
Libretexts. (2020, July 14). Functional Groups. Chemistry LibreTexts. https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_and_Chemical_Reactivity_(Kotz_et_al.)/10%3A_Carbon%3A_More_Than_Just_Another_Element/10.6%3A_Functional_Groups  
Libretexts. (2023, January 30). Electrostatic potential maps. Chemistry LibreTexts. https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_%28Physical_and_Theoretical_Chemistry%29/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Electrostatic_Potential_maps 
Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (2012). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. Advanced Drug Delivery Reviews, 64, 4–17. https://doi.org/10.1016/j.addr.2012.09.019 
Maslehat, S., Sardari, S., & Arjenaki, M. G. (2018). Frequency and Importance of Six Functional Groups that Play A Role in Drug Discovery. Biosciences Biotechnology Research Asia, 15(3), 541-548.
MathWorks. (n.d.). Neural net fitting. MathWorks. https://www.mathworks.com/help/deeplearning/gs/cluster-data-with-a-self-organizing-map.html 
MSF Medical Guidelines. (n.d.). Drug quality and storage. Drug quality and storage | MSF Medical Guidelines. https://medicalguidelines.msf.org/en/viewport/EssDr/english/drug-quality-and-storage-16688167.html
Pal, R., Poddar, A., & Chattaraj, P. K. (2021). Atomic Clusters: Structure, Reactivity, Bonding, and Dynamics. Frontiers in chemistry, 9, 730548. https://doi.org/10.3389/fchem.2021.730548
Pennington, L. D., Collier, P. N., & Comer, E. (2023). Harnessing the necessary nitrogen atom in chemical biology and drug discovery. Medicinal Chemistry Research, 32(7), 1278–1293. https://doi.org/10.1007/s00044-023-03073-3
Pradeepkiran, J. A., Sainath, S. B., & Shrikanya, K. V. L. (2021). In silico validation and ADMET analysis for the best lead molecules. Brucella Melitensis, 133–176. https://doi.org/10.1016/b978-0-323-85681-2.00008-2 
Rangel-Galván, M., Castro, M. E., Perez-Aguilar, J. M., Caballero, N. A., Rangel-Huerta, A., & Melendez, F. J. (2022). Theoretical Study of the Structural Stability, Chemical Reactivity, and Protein Interaction for NMP Compounds as Modulators of the Endocannabinoid System. Molecules, 27(2), 414. https://doi.org/10.3390/molecules27020414
Ranjith, D., & Ravikumar, C. (2019). SwissADME predictions of pharmacokinetics and drug-likeness properties of small molecules present in Ipomoea mauritiana Jacq. Journal of Pharmacognosy and Phytochemistry, 8(5), 2063-2073.
Ritchie, T. J., Macdonald, S. J. F., Young, R. J., & Pickett, S. D. (2011). The impact of aromatic ring count on compound developability: further insights by examining carbo- and hetero-aromatic and -aliphatic ring types. Drug Discovery Today, 16(3-4), 164–171. https://doi.org/10.1016/j.drudis.2010.11.014
Thieme WebCheminars. (n.d.). Drug score. Organic Chemistry Portal. https://www.organic-chemistry.org/prog/peo/drugScore.html 
TI, O., PD, L., SJ, T., & AM, D. (2001). Is there a difference between leads and drugs? A historical perspective. Journal of chemical information and computer sciences. https://pubmed.ncbi.nlm.nih.gov/11604031/ 
Ursu, O., Rayan, A., Goldblum, A., & Oprea, T. I. (2011). Wiley Interdiscip. Rev.: Comput. Mol. Sci, 1, 760-781.
Walker, B., Holling, C. S., Carpenter, S. R., & Kinzig, A. P. (2004). Resilience, Adaptability and Transformability in Social-ecological Systems. Ecology and Society, 9(2).
Wei, W., Cherukupalli, S., Jing, L., Liu, X., & Zhan, P. (2020). Fsp3: A new parameter for drug-likeness. Drug Discovery Today, 25(10), 1839–1845. https://doi.org/10.1016/j.drudis.2020.07.017
Zewe, A. (2022, April 26). A smarter way to develop new drugs. MIT News | Massachusetts Institute of Technology. https://news.mit.edu/2022/ai-molecules-new-drugs-0426  
 Bahmani, B., Moseley, B., Vattani, A., Kumar, R., and Vassilvitskii, S. (2012). Scalable k-means++. Proceedings of the VLDB Endowment, 5(7):622–633,2012

‌


‌


‌


Appendixes
Appendix 1: QED Distribution and QED Upper and Lower Bound for each label



Appendix 2. Heatmap of Correlation for Each Molecular Descriptors

S3 Figure. Scatterplot of Latent Space of the Autoencoder after UMAP

