# Introduction

## 1.1 Drug-Likeness and the Chemical Quality of Pharmaceuticals

The quality of a drug is a fundamental determinant of therapeutic efficacy and patient safety. Drug-likeness — the qualitative assessment of whether a molecule possesses physicochemical and structural properties consistent with oral bioavailability — has become a central organising principle in early-stage drug discovery (Lipinski et al., 1997; Leeson & Springthorpe, 2007). Computational estimation of drug-likeness from molecular structure alone, prior to synthesis and biological testing, enables the rapid virtual screening of compound libraries containing millions of candidates, dramatically reducing the cost and time required to identify viable leads (Ursu et al., 2011; Cai et al., 2022).

The most influential framework for assessing drug-likeness remains Lipinski's Rule of Five (Ro5), which defines acceptable ranges for molecular weight (≤500 Da), lipophilicity (logP ≤ 5), hydrogen bond donors (≤5), and hydrogen bond acceptors (≤10) based on the observation that most orally bioavailable drugs satisfy at least three of these four criteria (Lipinski et al., 1997). While Ro5 provides a useful filter, it is inherently binary — a molecule either passes or fails — and does not quantify the *degree* of drug-likeness. To address this limitation, Bickerton et al. (2012) introduced the Quantitative Estimate of Drug-likeness (QED), a continuous score from 0 to 1 that integrates eight molecular properties (molecular weight, logP, number of hydrogen bond donors and acceptors, polar surface area, number of rotatable bonds, number of aromatic rings, and the presence of structural alerts) using a desirability function framework. QED provides a more nuanced and quantitative measure of pharmaceutical quality than binary rule-based filters.

Complementing drug-likeness, the Synthetic Accessibility Score (SAS) quantifies how readily a molecule can be synthesised, ranging from 1 (trivially easy) to 10 (practically impossible) based on molecular complexity and the availability of known synthetic fragments (Ertl & Schuffenhauer, 2009). The interplay between drug-likeness and synthetic accessibility is a central tension in medicinal chemistry: structural modifications that improve pharmacological properties — such as adding stereocentres, saturated heterocycles, or complex ring fusions — often increase synthetic difficulty (Roughley & Jordan, 2011). Understanding this trade-off at the molecular level is essential for rational drug design.

## 1.2 Functional Groups as Determinants of Molecular Properties

Functional groups — the units of connected atoms defined by specific bonding arrangements within organic molecules — are the primary carriers of chemical reactivity, physicochemical properties, and biological activity (Anslyn & Dougherty, 2006). Each functional group imparts characteristic properties to its parent molecule: carboxylic acids introduce ionisability at physiological pH, amide bonds provide hydrogen bonding capacity and metabolic stability, aromatic rings contribute hydrophobicity and π-stacking interactions with protein binding sites, and halogen substituents modulate lipophilicity and metabolic resistance (Maslehat et al., 2018).

In medicinal chemistry, the strategic selection and placement of functional groups is the primary tool for optimising lead compounds. The prevalence of certain functional groups in approved drugs — phenyl rings, amide bonds, heterocyclic nitrogen — is not coincidental but reflects the convergence of synthetic accessibility, metabolic stability, and target binding requirements (He et al., 2010). However, the relationship between functional group composition and drug-likeness is not simply additive; the molecular context in which a functional group appears — its position relative to other substituents, the ring system to which it is attached, the three-dimensional conformation of the molecule — profoundly affects its contribution to overall molecular properties. This context dependence is particularly evident for nitrogen atoms, whose impact on drug-likeness is highly sensitive to their hybridisation state and position within the molecular scaffold (Pennington et al., 2023).

Despite the recognised importance of functional groups in drug design, systematic large-scale analyses of the relationship between functional group composition and drug-likeness scores remain limited. Most studies examine individual functional group–property relationships in isolation, missing the combinatorial patterns that emerge when multiple functional groups interact within a single molecular scaffold.

## 1.3 Graph-Based Molecular Representations and Learned Embeddings

Traditional molecular representations for machine learning — fingerprint vectors, descriptor tables, SMILES strings — encode molecules as fixed-dimensional feature vectors that discard topological information. A molecular graph, by contrast, preserves the full connectivity of atoms (nodes) and bonds (edges), enabling structure-aware learning through message-passing neural networks (David et al., 2020; Guo et al., 2023). Graph Attention Networks (GATs) extend this framework by learning attention weights over each atom's neighbourhood, allowing the model to prioritise chemically significant interactions — such as electronegative atoms adjacent to hydrogen bond donors — during feature aggregation.

Variational Graph Autoencoders (VGAEs) combine graph neural networks with variational inference to learn continuous, low-dimensional latent representations of molecular graphs. Unlike deterministic autoencoders, VGAEs impose a probabilistic prior on the latent space, encouraging smooth interpolation between molecules and enabling generative sampling of novel structures (Kipf & Welling, 2016). The latent embeddings produced by a VGAE capture structural and electronic properties that are distributed across the entire molecular graph, including long-range interactions that fragment-based methods cannot represent.

Previous work in this domain has primarily employed dense autoencoders operating on flat feature vectors, which discard bond topology and treat molecular descriptors as independent variables. Our prior study used a feed-forward autoencoder to tokenise molecular features followed by SOM clustering, identifying trends in atomic composition, hybridisation, and bond types across drug-likeness strata (see §1.5). While those results demonstrated that unsupervised clustering could reveal structure–drug-likeness relationships, the flat representation limited the analysis to aggregate molecular statistics (total carbon count, total aromatic bonds) without capturing the functional group context that determines chemical behaviour.

## 1.4 Self-Organising Maps for Chemical Space Analysis

Self-Organising Maps (SOMs), introduced by Kohonen (1982), are unsupervised neural networks that project high-dimensional data onto a low-dimensional (typically two-dimensional) grid while preserving topological relationships — nearby points in the input space map to nearby neurons on the grid. This topology-preserving property makes SOMs particularly well-suited for visualising and navigating chemical space, where the notion of molecular similarity is inherently continuous and high-dimensional (Kohonen & Honkela, 2007).

In cheminformatics, SOMs have been applied to compound library design, activity landscape analysis, and ADMET property prediction (Chaudhary et al., 2014; Kotyrba et al., 2021). The combination of SOMs with optimised initialisation methods such as K-Means++ (Bahmani et al., 2012) improves convergence and cluster quality by selecting initial neuron weights that better span the data distribution. The U-matrix (unified distance matrix) derived from trained SOMs reveals the topological structure of the chemical space, with low U-matrix values indicating smooth transitions between similar molecules and high values marking boundaries between distinct molecular subpopulations.

## 1.5 Relationship to Prior Work

This study extends and substantially revises our previous analysis, which applied a feed-forward autoencoder and Deep SOM to 249,455 ZINC15 molecules stratified by QED score. That analysis revealed several trends: (1) Sp2 hybridisation correlated negatively with drug-likeness (r = −0.45) while Sp3 correlated positively (r = +0.26); (2) total conjugated and aromatic bond counts were negatively associated with QED; (3) carbon count, nitrogen positioning, and chirality showed stratum-dependent patterns; and (4) molecules with unspecified stereochemistry concentrated in low-QED strata.

However, the prior study suffered from several methodological limitations. The flat feature vector representation discarded bond topology and functional group context. The autoencoder was deterministic, producing a non-regularised latent space unsuitable for interpolation or generation. The analysis examined aggregate atomic statistics (total carbon, total nitrogen) rather than chemically meaningful functional groups. And the SOM configuration was fixed rather than optimised per stratum.

The present study addresses each of these limitations:

| Aspect | Prior Study | Present Study |
|---|---|---|
| Molecular representation | Flat 28-dim feature vector | Full molecular graph (atoms + bonds) |
| Feature learning | Dense autoencoder (28→16→28) | 3-layer Graph Attention Network (VGAE) |
| Latent model | Deterministic | Variational (KL-regularised) |
| Topology awareness | None (bag of atoms) | Message passing preserves bond connectivity |
| Functional group analysis | None (aggregate atom counts) | 22-type substructure detection + enrichment |
| SOM configuration | Fixed 10×10 grid | Autotuned per stratum (up to 30×30) |
| Cluster characterisation | Averaged atomic statistics | FG signatures, enrichment ratios, representatives |

## 1.6 Study Objectives

This study aims to characterise the functional group composition of drug-like molecules from the ZINC15 database and to elucidate how specific functional group patterns relate to drug-likeness (QED), lipophilicity (logP), and synthetic accessibility (SAS). Specifically, we address the following questions:

1. **What is the functional group landscape of commercially available drug-like chemical space?** We quantify the prevalence of 22 functional group types across 249,455 molecules and identify co-occurrence patterns.

2. **How do individual functional groups correlate with drug-likeness properties?** We compute point-biserial correlations between functional group presence and QED, logP, and SAS to identify groups that enhance or diminish pharmaceutical quality.

3. **Does drug-like chemical space exhibit internal structure at the functional group level?** Using VGAE-derived embeddings and stratified SOM clustering, we identify molecular subpopulations defined by characteristic functional group signatures and assess whether these subpopulations correspond to recognised pharmacophore patterns.

4. **What is the relationship between aromatic character and drug-likeness in contemporary chemical libraries?** We examine whether the aromatic dominance of current drug-like space is a chemical necessity or a historical artefact of synthetic methodology, and identify viable non-aromatic drug-like scaffolds.

5. **How do drug-likeness and synthetic accessibility trade off at the functional group level?** We quantify the QED–SAS relationship across strata and identify functional group motifs that simultaneously optimise both metrics.
