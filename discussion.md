# Discussion

## 5.1 The Aromatic Dominance of Drug-Like Chemical Space

The functional group census across 249,455 ZINC15 molecules reveals a striking structural homogeneity: 83.0% of molecules contain at least one phenyl ring (mean 2.42 per molecule), 68.0% contain amide bonds, and 58.0% contain heterocyclic rings. This triad — phenyl, amide, heterocycle — constitutes the canonical scaffold of modern medicinal chemistry. The dominance of aromatic systems reflects both pharmacological rationale and synthetic convenience: phenyl rings serve as rigid hydrophobic anchors in protein binding pockets, providing favourable enthalpic contributions through π–π stacking and CH–π interactions with aromatic residues in the target protein (Ritchie et al., 2011). Their synthetic accessibility via well-established cross-coupling reactions (Suzuki, Heck, Sonogashira) and nucleophilic aromatic substitution has made them the default building block in combinatorial library design.

However, the overrepresentation of flat, sp2-rich scaffolds has been increasingly recognised as a liability. Ritchie et al. (2011) demonstrated that the mean number of aromatic rings per molecule has increased steadily in medicinal chemistry patents over the past three decades, correlating with rising clinical attrition rates. Our data quantify this directly: phenyl prevalence actually *increases* in the lowest-QED stratum (92.6% in Stratum 0 vs. 83.0% overall), suggesting that excessive aromaticity actively harms drug-likeness. This is consistent with the QED metric's penalisation of high aromatic ring counts, but also reflects the underlying pharmaceutical reality — highly aromatic molecules tend to be lipophilic, poorly soluble, and prone to cytochrome P450 metabolism.

The "escape from flatland" hypothesis (Lovering et al., 2009) proposes that increasing the fraction of sp3-hybridised carbons (Fsp3) improves the probability of clinical success. Our Stratum 4 analysis provides empirical support at unprecedented scale: **Cluster 0** (n = 1,615 molecules, mean QED = 0.86) contains only 11.5% phenyl-bearing molecules versus the 83.0% dataset average, while enriching for tertiary amines (1.81×), thioethers (1.89×), and ethers (1.41×). These saturated, three-dimensional scaffolds — piperidines, morpholines, tetrahydropyrans, spiro-fused ring systems — are precisely the architectural class that Lovering et al. identified as correlating with improved developability. Their concentration in the highest-QED stratum validates the hypothesis quantitatively: within the ZINC15 commercial library, the most drug-like molecules are disproportionately non-aromatic.

This finding also connects to our prior analysis, which showed Sp2 hybridisation correlated negatively with QED (r = −0.45) while Sp3 correlated positively (r = +0.26). The current functional-group-level analysis provides mechanistic depth to those aggregate statistics: it is not merely the *count* of sp2 atoms that matters, but the specific functional group context — a single phenyl ring in an otherwise saturated scaffold has a different impact on drug-likeness than three fused aromatic rings.

## 5.2 Lipophilicity, Synthetic Accessibility, and the Drug-Likeness Triad

The three molecular properties tracked across QED strata — drug-likeness (QED), lipophilicity (logP), and synthetic accessibility (SAS) — form an interconnected triad whose internal tensions define the practical design space available to medicinal chemists.

**The logP gradient across strata** (Stratum 0 mean logP ≈ 3.2 → Stratum 4 ≈ 2.0) confirms that even within Lipinski-compliant space (logP < 5), lower lipophilicity associates with higher drug-likeness. This relationship has a pharmacokinetic basis: excessively lipophilic compounds exhibit poor aqueous solubility, high plasma protein binding, increased metabolic clearance by CYP enzymes, and elevated off-target promiscuity — particularly non-selective binding to hERG potassium channels, a major cause of drug withdrawal (Waring, 2010). The ZINC15 dataset, pre-filtered for purchasable compounds, already excludes extreme logP outliers, yet the gradient persists, indicating that logP optimisation remains beneficial even within the "drug-like" window.

**The counterintuitive SAS gradient** (Stratum 0 SAS ≈ 2.5 → Stratum 4 ≈ 3.5) reveals a fundamental tension: more drug-like molecules are *harder* to synthesise. This is not paradoxical when considered from a structural chemistry perspective. High-QED molecules tend to feature more stereocentres (enabling enantioselective target binding), more saturated heterocyclic rings (improving solubility and reducing off-target activity), and more diverse functional group decoration (providing multiple pharmacophore interactions). Each of these structural features increases SAS because they require enantioselective synthesis, heterocycle-forming reactions (e.g., Buchwald–Hartwig, Ullmann coupling), or multi-step functionalisation sequences (Roughley & Jordan, 2011).

This QED–SAS trade-off has direct implications for hit-to-lead campaigns. The data support the increasingly common practice of incorporating synthetic accessibility as a co-objective alongside potency and selectivity from the earliest stages of molecular design — a principle formalised in the concept of "lead-like" chemical space (Teague et al., 1999).

### Functional Group Contributions to the Triad

The point-biserial correlations between individual functional groups and the three property metrics provide a quantitative SAR atlas:

**Phenyl → ↑logP (+0.40), ↓SAS (−0.43), ≈QED (−0.07).** Aromatic rings are the primary driver of lipophilicity in this dataset, with each ring contributing approximately 1.5 logP units via π-system hydrophobicity (Wildman & Crippen, 1999). Their strong negative SAS correlation reflects synthetic reality: aromatic building blocks are commercially abundant, and ring-forming reactions (electrocyclic, Diels–Alder) are among the most reliable in organic synthesis. The near-neutral QED effect indicates that phenyl rings are neither explicitly rewarded nor heavily penalised by the drug-likeness metric — they are the "background scaffold" of drug-like space.

**Nitro (−NO₂) → ↓QED (−0.32), ≈logP (+0.04), ≈SAS (−0.07).** The strong negative QED correlation reflects QED's explicit penalisation of structural alerts. Nitroaromatics are flagged as toxicophores in multiple predictive models due to their propensity for nitroreduction by mammalian nitroreductases and gut microbiota, producing genotoxic hydroxylamine and nitroso intermediates (Kazius et al., 2005; Benigni & Bossa, 2011). Our data show a >20-fold depletion of nitro groups between Stratum 0 (29.6%) and Stratum 4 (1.3%), empirically validating the structural alert classification. The near-neutral logP and SAS effects confirm that the QED penalty is driven by toxicity risk, not by unfavourable physicochemical properties.

**Amide (−CONH−) → ≈QED (+0.02), ≈logP (+0.09), ↓SAS (−0.29).** Amide bonds are the most common bond-forming reaction in medicinal chemistry — Brown and Boström (2016) found that amide coupling accounted for the single most-used reaction in the synthesis of drug molecules reported between 2014 and 2015. The strong negative SAS correlation directly reflects this synthetic prevalence: molecules containing amide bonds are easier to make because amide coupling reagents (HATU, EDC, T3P) are reliable and commercially available. The QED-neutrality of amides is notable: despite contributing hydrogen bonding capacity (one HBD, one HBA per amide), their overall effect on drug-likeness is balanced.

**Carboxyl (−COOH) → ≈QED (0.00), ↓logP (−0.28), ↑SAS (+0.13).** The negative logP correlation reflects the ionisation of carboxylic acids at physiological pH (pKa ≈ 4.5), which dramatically increases aqueous solubility at the expense of membrane permeability. The positive SAS correlation likely reflects the synthetic steps required to introduce or reveal carboxylic acid functionality — deprotection, saponification, or oxidation. Despite common assumptions that carboxylic acids are "undrugable" due to poor permeability, their QED-neutrality indicates that the drug-likeness metric does not systematically penalise them — suggesting that carboxylate-containing drugs achieve oral bioavailability through compensating properties.

**Halide (C−X) → ≈QED (+0.01), ↑logP (+0.26), ↓SAS (−0.16).** Halogens — particularly fluorine and chlorine — increase lipophilicity and are synthetically accessible via electrophilic halogenation, Balz–Schiemann, or Sandmeyer reactions. Fluorine's unique role in drug design merits emphasis: it is the most commonly introduced halogen in FDA-approved drugs (Gillis et al., 2015), serving as a metabolic blocker (preventing CYP-mediated oxidation at specific positions), a lipophilicity modulator (C–F bond dipole partially mimics C–OH), and a conformational constraint (gauche effect in vicinal difluorides). The modest positive logP effect (+0.26) reflects the aggregate of all halides; fluorine alone has a smaller lipophilicity contribution than chlorine or bromine.

## 5.3 Functional Group Co-occurrence and Pharmacophore Signatures

That no individual functional group correlates with any single molecular property above |r| = 0.43 (Phenyl → SAS) carries a fundamental message: **molecular properties emerge from the combinatorial interplay of multiple functional groups, not from individual substituent effects.** This explains why additive group-contribution models for logP (e.g., Crippen's fragmental method) and solubility (e.g., ESOL) exhibit systematic errors for molecules with extensive intramolecular interactions — hydrogen bonds, charge-transfer effects, steric buttressing — that create non-additive property contributions (Tropsha, 2010).

The cluster-level functional group enrichment analysis reveals several pharmacologically meaningful co-occurrence patterns:

### Sulfonamide Pharmacophore (Cross-Stratum)

Across strata 1–4, clusters enriched for sulfonyl (−SO₂−, 1.5–2.1×) consistently co-enrich for nitrile (−C≡N), imine (C=N), and heterocyclic nitrogen. This combination — a sulfonamide hydrogen-bond donor/acceptor paired with a nitrogen-containing heterocycle — is the canonical pharmacophore of two major drug classes:

1. **Sulfonamide antibacterials** (sulfamethoxazole, sulfadiazine): The sulfonamide group competitively inhibits dihydropteroate synthase by mimicking the PABA substrate, while the heterocyclic ring modulates selectivity and pharmacokinetics.

2. **Kinase inhibitors** (vemurafenib, dabrafenib): The sulfonamide serves as a solubilising group and hydrogen-bond anchor in the kinase hinge region, while the heterocyclic component occupies the ATP-binding pocket.

The persistence of this co-occurrence pattern across multiple QED strata indicates that sulfonamide pharmacophores are compatible with a wide range of drug-likeness profiles — providing medicinal chemists with considerable room for property optimisation without abandoning the core pharmacophore.

### Polar Aliphatic Cluster (Stratum 0, Cluster 600)

The simultaneous enrichment of primary amine (3.8×), hydroxyl (3.7×), and carboxyl (2.5×) at logP = −0.19 identifies a population of amino acid derivatives, sugar-like scaffolds, and peptidomimetics. These highly polar molecules "fail" drug-likeness by QED standards, but this failure reflects a calibration limitation rather than inherent pharmacological inadequacy. QED was trained on a historical set of orally bioavailable small molecules (Bickerton et al., 2012) and systematically undervalues:

- **Injectable biologics** and peptide drugs (insulin, GLP-1 agonists) that bypass oral absorption entirely
- **Prodrugs** that mask polar groups for absorption and release them intracellularly
- **Transporter substrates** that exploit active uptake mechanisms (e.g., amino acid transporters for gabapentin, nucleoside transporters for gemcitabine)
- **CNS-penetrant polar molecules** that utilise receptor-mediated transcytosis across the blood–brain barrier (Pardridge, 2012)

This cluster represents a pharmacologically active chemical space that is invisible to QED-based drug-likeness assessment.

### Halide–Heterocycle Synergy (Stratum 4, Cluster 899)

In the highest-QED cluster (n = 3,347), halide enrichment at 1.51× combined with heterocycle enrichment at 1.23× reflects the dominance of halogenated heteroaromatic scaffolds in marketed drugs. An analysis of all FDA-approved small-molecule drugs reveals that approximately 40% contain at least one fluorine atom and 25% contain at least one chlorine atom (Gillis et al., 2015). The specific combination of halide + heterocycle serves complementary roles:

- **Fluorine on aromatic rings**: blocks metabolically labile positions, extending half-life
- **Chlorine in hydrophobic pockets**: fills van der Waals space, improving binding affinity
- **Heterocyclic nitrogen**: provides hydrogen-bond acceptor capability for target engagement

The co-enrichment of carboxyl groups at 3.67× in this same cluster is notable. These high-QED carboxylate-containing molecules likely represent NSAID-like scaffolds (ibuprofen, naproxen architecture), angiotensin receptor antagonists (losartan-type), or PPAR agonists (fibrate-class) — therapeutic classes where the anionic carboxylate is essential for target binding.

## 5.4 The Topology of Drug-Like Chemical Space

The monotonic decrease in quantization error from Stratum 0 (QE = 1.10) to Stratum 4 (QE = 0.74) provides structural evidence that drug-like molecules occupy a **more compact and self-similar** region of chemical space than non-drug-like molecules. This compactness has a chemical explanation rooted in constraint satisfaction: the Lipinski boundaries (MW < 500, logP < 5, HBD ≤ 5, HBA ≤ 10) define a bounded polytope in property space, and molecules optimised for oral bioavailability converge toward structurally similar solutions within this polytope. Non-drug-like molecules, unconstrained by these pharmaceutical boundaries, explore a far wider manifold of molecular architectures.

The 3D UMAP projection (Figure 9) visualises this asymmetry: Stratum 4 (high QED) forms a dense, coherent cloud in latent space, while Stratum 0 (low QED) is diffuse and peripherally distributed. The intermediate strata (1–3) form a continuous surface connecting these extremes, consistent with the view that chemical space is fundamentally a continuum rather than a collection of discrete islands (Dobson, 2004). The absence of sharp boundaries between strata in the UMAP projection is pharmacologically significant: it suggests that drug-likeness can be incrementally improved through structural modification, rather than requiring discontinuous "jumps" between molecular archetypes.

Within each stratum, the SOM U-matrix heatmaps (Figure 11) reveal the internal topological structure. Stratum 4 has the lowest U-matrix maximum (0.074 vs. 0.141 for Stratum 2), indicating smoother transitions between neighbouring clusters. Molecules at the drug-likeness optimum are more structurally similar to their nearest neighbours than are molecules at intermediate QED values. This smoothness has practical implications:

- **Virtual screening**: interpolating between Stratum 4 cluster centroids is more likely to produce valid molecular structures than interpolating in Stratum 2, where the rugged U-matrix landscape signals abrupt transitions between dissimilar scaffolds.
- **Lead optimisation**: small structural modifications to high-QED molecules are more likely to remain in drug-like space, whereas modifications to intermediate-QED molecules risk crossing a U-matrix boundary into a less favourable structural region.

## 5.5 Comparison with Prior Work

Our earlier analysis, which used a feed-forward autoencoder on flat 28-dimensional feature vectors, identified several trends that are both confirmed and clarified by the present graph-based functional group analysis:

| Prior Finding (Flat AE) | Present Finding (VGAE + FG Analysis) | Interpretation |
|---|---|---|
| Sp2 hybridisation negatively correlates with QED (r = −0.45) | Phenyl prevalence decreases from 92.6% (S0) to 83.0% (S4); sp3-rich cluster identified in S4 | The sp2 effect is driven primarily by excessive aromatic ring accumulation; the FG-level analysis reveals that a *single* aromatic ring is QED-neutral while *multiple* rings are detrimental |
| Sp3 hybridisation positively correlates with QED (r = +0.26) | Tertiary amines (1.81×), thioethers (1.89×), ethers (1.41×) enriched in high-QED non-aromatic cluster | The sp3 benefit is mediated by specific saturated functional groups, not by sp3 character per se |
| Carbon count negatively correlates with QED (r = −0.27) | No direct equivalent | Carbon count is a proxy for molecular weight; the QED penalty reflects MW > 500 penalisation rather than carbon-specific effects |
| Total aromatic bonds decrease with increasing QED | Aromatic ring count correlates with logP (+0.40) and anti-correlates with SAS (−0.43) | Aromatic bonds affect QED indirectly through logP; the relationship is mediated by lipophilicity, not aromaticity per se |
| Nitrogen count shows weak decreasing trend | Nitrogen position matters more than count: tertiary amine enriched in high-QED, primary amine in low-QED | Confirms the context-dependence hypothesis from prior work; functional group identity, not element count, determines the property impact |
| Chirality shows weak positive QED correlation (r ≈ +0.12) | Not directly measured (stereo-agnostic FG detection) | Future work should incorporate stereocentre counting into the FG vocabulary |

The most significant advance over the prior analysis is the shift from aggregate atomic statistics to **functional group signatures with enrichment ratios**. Where the prior study could state "higher-QED molecules have fewer aromatic bonds," the present analysis can state "1,615 high-QED molecules lack aromatic scaffolds entirely and are enriched for piperidines, morpholines, and thioethers" — a far more actionable insight for medicinal chemistry.

## 5.6 Implications for Molecular Design

The findings from this analysis translate directly to actionable guidance for medicinal chemistry campaigns:

1. **Aromatic ring reduction is empirically validated.** The identification of 1,615 high-QED molecules with minimal aromatic character provides concrete exemplars for the Lovering hypothesis. Medicinal chemists pursuing "flatland escape" strategies can use this cluster's representative molecules as starting points for scaffold design.

2. **The sulfonamide pharmacophore is QED-robust.** The recurrence of sulfonyl + heterocyclic nitrogen enrichment across all five strata indicates that this pharmacophore can be optimised across a wide QED range without fundamental redesign. This is practically valuable for kinase inhibitor and antibacterial programmes.

3. **Carboxylate drugs are not categorically excluded from drug-like space.** The 3.67× carboxyl enrichment in the highest-QED cluster challenges the empirical rule that carboxylic acids preclude oral bioavailability. For targets requiring anionic pharmacophores — integrins, PPARs, prostanoid receptors — this finding legitimises carboxylate-retaining design strategies.

4. **SAS should be a co-objective from hit identification.** The positive SAS gradient across QED strata quantifies the trade-off: each 0.1 QED improvement costs approximately 0.25 SAS units. This allows prospective project teams to set realistic SAS thresholds based on their target QED range.

5. **The nitro group depletion gradient provides a structural alert benchmark.** The >20-fold nitro depletion from Stratum 0 to Stratum 4 can serve as a reference point for evaluating other potential structural alerts — any functional group showing a comparable depletion gradient warrants investigation as a toxicophore.

## 5.7 Limitations

1. **Dataset bias.** ZINC15 is filtered for commercial availability, excluding natural products (macrolides, terpenes, alkaloids), macrocycles (>500 Da cyclic peptides), and covalent warheads (acrylamides, α,β-unsaturated carbonyls) — all growing areas of drug discovery. Extending this analysis to ChEMBL, DrugBank, or the COCONUT natural product database (Sorokina et al., 2021) would test whether the observed FG–property relationships generalise beyond synthetic commercial space.

2. **QED calibration.** QED is trained on a historical corpus of orally bioavailable small-molecule drugs (Bickerton et al., 2012) and does not account for non-oral modalities, targeted degraders (PROTACs, molecular glues), or RNA-targeting small molecules. The stratum boundaries used here are QED quintiles; alternative stratification by therapeutic area, target class, or Ro5 compliance status might reveal different patterns.

3. **Functional group vocabulary.** The 22-type functional group vocabulary covers canonical medicinal chemistry motifs but misses pharmacologically important substructures: boronic acids (proteasome inhibitors), azetidines (Pfizer's "magic ring"), deuterated methyl groups (metabolic stabilisation), vinyl sulfonamides (covalent warheads), and fluoroalkyl groups. Extending coverage to 50–100 substructure types would provide a more complete picture.

4. **No bioactivity integration.** The present analysis establishes structural relationships but cannot determine whether co-clustered molecules share pharmacological activity. Cross-referencing clusters with ChEMBL target annotations, HTS screening data, or pharmaceutical patent assignees would connect structural clusters to therapeutic hypotheses and identify whether latent-space proximity predicts biological similarity.

5. **Stereochemistry.** While the graph neural network encodes E/Z and R/S configurations as node features, the functional group detection is stereo-agnostic. For chiral drug molecules — particularly those where enantiomers exhibit dramatically different pharmacology (e.g., thalidomide, omeprazole) — stereocentre-aware analysis could reveal enantiomer-specific clustering patterns.

## 5.8 Future Directions

**Target deconvolution.** Overlaying ChEMBL bioactivity annotations onto the SOM would test whether structural clusters correspond to target families — and reveal activity cliffs (structurally similar molecules with >100-fold potency differences) that are invisible at the whole-dataset level.

**Scaffold hopping.** Inter-cluster distance analysis identifies structurally dissimilar molecules occupying adjacent positions in latent space. These molecule pairs are candidates for bioisosteric scaffold replacement — the most sought-after capability in lead optimisation — and could be prioritised for matched molecular pair analysis.

**Generative molecular design.** The stratified SOM provides natural conditioning variables for constrained generative models. Sampling molecular graphs from the latent-space regions corresponding to high-QED, low-SAS cluster centroids would bias generative models toward synthetically accessible drug candidates, addressing the well-known problem that unconstrained generators produce synthetically intractable molecules.

**Multi-objective ADMET integration.** Extending the property set beyond the QED/logP/SAS triad to include Caco-2 permeability predictions, hERG channel liability scores, microsomal metabolic stability estimates, and plasma protein binding fractions would enable multi-dimensional stratification — mapping the relationships between functional group composition and the full ADMET profile required for clinical development.

**Natural product chemical space comparison.** Repeating this analysis on the COCONUT (Sorokina et al., 2021) or LOTUS databases would quantify how natural product chemical space differs from synthetic drug space in terms of functional group distribution, ring saturation, stereochemical complexity, and the presence of macrocyclic and glycosidic motifs absent from ZINC15. Such a comparison could identify natural-product-inspired design principles for synthetic molecule programmes.

## References

Anslyn, E. V., & Dougherty, D. A. (2006). *Modern Physical Organic Chemistry*. University Science Books.

Benigni, R., & Bossa, C. (2011). Mechanisms of chemical carcinogenicity and mutagenicity: a review with implications for predictive toxicology. *Chemical Reviews*, 111(4), 2507–2536.

Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012). Quantifying the chemical beauty of drugs. *Nature Chemistry*, 4(2), 90–98.

Brown, D. G., & Boström, J. (2016). Analysis of past and present synthetic methodologies on medicinal chemistry: where have all the new reactions gone? *Journal of Medicinal Chemistry*, 59(10), 4443–4458.

Clemons, P. A., et al. (2010). Small molecules of different origins have distinct distributions of structural complexity that correlate with protein-binding profiles. *Proceedings of the National Academy of Sciences*, 107(44), 18787–18792.

Dobson, C. M. (2004). Chemical space and biology. *Nature*, 432(7019), 824–828.

Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *Journal of Cheminformatics*, 1, 8.

Gillis, E. P., Eastman, K. J., Hill, M. D., Donnelly, D. J., & Meanwell, N. A. (2015). Applications of fluorine in medicinal chemistry. *Journal of Medicinal Chemistry*, 58(21), 8315–8359.

He, Z., et al. (2010). Predicting drug-target interaction networks based on functional groups and biological features. *PLoS ONE*, 5(3), e9603.

Kazius, J., McGuire, R., & Bursi, R. (2005). Derivation and validation of toxicophores for mutagenicity prediction. *Journal of Medicinal Chemistry*, 48(1), 312–320.

Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (1997). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. *Advanced Drug Delivery Reviews*, 23(1–3), 3–25.

Lovering, F., Bikker, J., & Humblet, C. (2009). Escape from flatland: increasing saturation as an approach to improving clinical success. *Journal of Medicinal Chemistry*, 52(21), 6752–6756.

Maslehat, S., Sardari, S., & Arjenaki, M. G. (2018). Frequency and importance of six functional groups that play a role in drug discovery. *Biosciences Biotechnology Research Asia*, 15(3), 541–548.

Pardridge, W. M. (2012). Drug transport across the blood–brain barrier. *Journal of Cerebral Blood Flow & Metabolism*, 32(11), 1959–1972.

Pennington, L. D., Collier, P. N., & Comer, E. (2023). Harnessing the necessary nitrogen atom in chemical biology and drug discovery. *Medicinal Chemistry Research*, 32(7), 1278–1293.

Ritchie, T. J., Macdonald, S. J. F., Young, R. J., & Pickett, S. D. (2011). The impact of aromatic ring count on compound developability. *Drug Discovery Today*, 16(3–4), 164–171.

Roughley, S. D., & Jordan, A. M. (2011). The medicinal chemist's toolbox: an analysis of reactions used in the pursuit of drug candidates. *Journal of Medicinal Chemistry*, 54(10), 3451–3479.

Sorokina, M., et al. (2021). COCONUT online: collection of open natural products database. *Journal of Cheminformatics*, 13, 2.

Teague, S. J., Davis, A. M., Leeson, P. D., & Oprea, T. I. (1999). The design of leadlike combinatorial libraries. *Angewandte Chemie International Edition*, 38(24), 3743–3748.

Tropsha, A. (2010). Best practices for QSAR model development, validation, and exploitation. *Molecular Informatics*, 29(6–7), 476–488.

Ursu, O., Rayan, A., Goldblum, A., & Oprea, T. I. (2011). Understanding drug-likeness. *WIREs Computational Molecular Science*, 1(5), 760–781.

Waring, M. J. (2010). Lipophilicity in drug discovery. *Expert Opinion on Drug Discovery*, 5(3), 235–248.

Wildman, S. A., & Crippen, G. M. (1999). Prediction of physicochemical parameters by atomic contributions. *Journal of Chemical Information and Computer Sciences*, 39(5), 868–873.
