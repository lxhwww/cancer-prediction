# Cancer Types Prediction Using Recurrent Type Networks

Large-scale cancer genomics data often comes in multiplatform and heterogeneous forms. These datasets impose great challenges in terms of the bioinformatics approach and computational algorithms. Numerous researchers have proposed to utilize this data to overcome several challenges, using classical machine learning algorithms as either the primary subject or a supporting element for cancer diagnosis and prognosis.

## Deep learning in cancer genomics
Biomedical informatics includes all techniques regarding the development of data analytics, mathematical modeling, and computational simulation for the study of biological systems. In recent years, we've witnessed huge leaps in biological computing that has resulted in large, information-rich resources being at our disposal. These cover domains such as anatomy, modeling (3D printers), genomics, and pharmacology, among others.

One of the most famous success stories of biomedical informatics is from the domain of genomics. The **Human Genome Project (HGP)** was an international research project with the objective of determining the full sequence of human DNA. This project has been one of the most important landmarks in computational biology and has been used as a base for other projects, including the Human Brain Project, which is determined to sequence the human brain. The data that was used in this thesis is also the indirect result of the HGP. 

The era of big data starts from the last decade or so, which was marked by an overflow of digital information in comparison to its analog counterpart. Just in the year 2016, 16.1 zettabytes of digital data were generated, and it is predicted to reach 163 ZB/year by 2025. As good a piece of news as this is, there are some problems lingering, especially of data storage and analysis. For the latter, simple machine learning methods that were used in normal-size data analysis won't be effective anymore and should be substituted by deep neural network learning methods. Deep learning is generally known to deal very well with these types of large and complex datasets.

Along with other crucial areas, the biomedical area has also been exposed to these big data phenomena. One of the main largest data sources is omics data such as genomics, metabolomics, and proteomics. Innovations in biomedical techniques and equipment, such as DNA sequencing and mass spectrometry, have led to a massive accumulation of -omics data.

Typically -omics data is full of veracity, variability and high dimensionality. These datasets are sourced from multiple, and even sometimes incompatible, data platforms. These properties make these types of data suitable for applying DL approaches. Deep learning analysis of -omics data is one of the main tasks in the biomedical sector as it has a chance to be the leader in personalized medicine. By acquiring information about a person's omics data, diseases can be dealt with better and treatment can be focused on preventive measures.

Cancer is generally known to be one of the deadliest diseases in the world, which is mostly due to its complexity of diagnosis and treatment. It is a genetic disease that involves multiple gene mutations. As the importance of genetic knowledge in cancer treatment is increasingly addressed, several projects to document the genetic data of cancer patients has emerged recently. One of the most well known is The **Cancer Genome Atlas (TCGA)** project, which is available on the TCGA research network: http://cancergenome.nih.gov/.

As mentioned before, there have been a number of deep learning implementations in the biomedical sector, including cancer research. For cancer research, most researchers usually use -omics or medical imaging data as inputs. Several research works have focused on cancer analysis. Some of them use either a histopathology image or a PET image as a source. Most of that research focuses on classification based on that image data with **convolutional neural networks (CNNs)**.

However, many of them use -omics data as their source. Fakoor et al<sup>1</sup> classified the various types of cancer using patients' gene expression data. Due to the different dimensionality of each data from each cancer type, they used **principal component analysis (PCA)<sup>2</sup>** first to reduce the dimensionality of microarray gene expression data.

Then they applied sparse and stacked autoencoders to classify various cancers, including acute myeloid leukemia, breast cancer, and ovarian cancer.

Ibrahim et al<sup>3</sup>, on the other hand, used miRNA expression data from six types of cancer genes/miRNA feature selection. They proposed a novel multilevel feature selection approach named **MLFS** (short for **Multilevel gene/miRNA feature selection**), which was based on **Deep Belief Networks (DBN)** and unsupervised active learning.

Finally, Liang et al<sup>4</sup> clustered ovarian and breast cancer patients using multiplatform genomics and clinical data. The ovarian cancer dataset contained gene expression, DNA methylation, and miRNA expression data across 385 patients, which were downloaded from **The Cancer Genome Atlas (TCGA)**.

The breast cancer dataset included GE data and corresponding clinical information, such as survival time and time to recurrence data, which was collected by the Netherlands Cancer Institute. To deal with this multiplatform data, they used **multimodal Deep Belief Networks (mDBN)**.

First, they implemented a DBN for each of those data to get their latent features. Then, another DBN used to perform the clustering is implemented using those latent features as the input. Apart from these researchers, much research work is going on to give cancer genomics, identification, and treatment a significant boost.


## Cancer genomics dataset description
Genomics data covers all data related to DNA on living things. Although in this thesis we will also use other types of data like transcriptomic data (RNA and miRNA), for convenience purposes, all data will be termed as genomics data. Research on human genetics found a huge breakthrough in recent years due to the success of the HGP (1984-2000) on sequencing the full sequence of human DNA.

One of the areas that have been helped a lot due to this is the research of all diseases related to genetics, including cancer. Due to various biomedical analyses done on DNA, there exist various types of -omics or genomics data. Here are some types of -omics data that were crucial to cancer analysis:

- **Raw sequencing data**: This corresponds to the DNA coding of whole
chromosomes. In general, every human has 24 types of chromosomes in each cell
of their body, and each chromosome consists of 4.6-247 million base pairs. Each
base pair can be coded in four different types, which are **adenine (A)**, **cytosine (C)**, **guanine (G)**, and **thymine (T)**. Therefore, raw sequencing data consists of
billions of base pair data, with each coded in one of these four different types.
- **Single-Nucleotide Polymorphism (SNP) data**: Each human has a different raw
sequence, which causes genetic mutation. Genetic mutation can cause an actual
disease, or just a difference in physical appearance (such as hair color), or
nothing at all. When this mutation happens only on a single base pair instead of a sequence of base pairs, it is called **Single-Nucleotide Polymorphism (SNP)**.
- **Copy Number Variation (CNV) data**: This corresponds to a genetic mutation
that happens in a sequence of base pairs. Several types of mutation can happen,
including deletion of a sequence of base pairs, multiplication of a sequence of
base pairs, and relocation of a sequence of base pairs into other parts of the
chromosome.
- **DNA methylation data**: Which corresponds to the amount of methylation
(methyl group connected to base pair) that happens to areas in the chromosome.
A large amount of methylation in promoter regions of a gene can cause gene
repression. DNA methylation is the reason each of our organs acts differently
even though all of them have the same DNA sequence. In cancer, this DNA
methylation is disrupted.
- **Gene expression data**: This corresponds to the number of proteins that were
expressed from a gene at a given time. Cancer happens either because of high
expression of an oncogene (that is, a gene that causes a tumor), low expression of a tumor suppressor gene (a gene that prevents a tumor), or both. Therefore, the analysis of gene expression data can help discover protein biomarkers in cancer.
- **miRNA expression data**: Corresponds to the amount of microRNA that was
expressed at a given time. miRNA plays a role in protein silencing at the mRNA
stage. Therefore, an analysis of gene expression data can help discover miRNA
biomarkers in cancer.

There are several databases of genomics datasets, where the aforementioned data can be found. Some of them focus on the genomics data of cancer patients. These databases include:
- [**The Cancer Genome Atlas (TCGA)**](https://cancergenome.nih.gov/)
- [**International Cancer Genome Consortium (ICGC)**](https://icgc.org/)
- [**Catalog of Somatic Mutations in Cancer (COSMIC)**](https://cancer.sanger.ac.uk/cosmic)

This genomics data is usually accompanied by clinical data of the patient. This clinical data can comprise general clinical information (for example, age or gender) and their cancer status (for example, cancer location or cancer stage). All of this genomics data itself has a characteristic of high dimensions. For example, the gene expression data for each patient is structured based on the gene ID, which reaches around 60,000 types.

Moreover, some of the data itself comes from more than one format. For example, 70% of the DNA methylation data is collected from breast cancer patients and the remaining 30% are curated from different platforms. Therefore, there are two different structures on in this dataset. Therefore, to analyze genomics data by dealing with the heterogeneity, researchers have often used powerful machine learning techniques or even deep neural networks.

## Dataset Details

I will be using [the gene expression cancer RNA-Seq dataset downloaded from the UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq).

This dataset is a random subset of another dataset reported in the following paper: Weinstein, John N., et al. _The cancer genome atlas pan-cancer analysis project. Nature Genetics 45.10 (2013): 1113-1120_. The preceding diagram shows the data collection pipeline for the pan-cancer analysis project.

The name of the project is The Pan-Cancer analysis project. It assembled data from thousands of patients with primary tumors occurring in different sites of the body. It covered 12 tumor types (see the upper-left panel in the preceding figure) including:
- Glioblastoma Multiform (GBM)
- Lymphoblastic acute myeloid leukemia (AML)
- Head and Neck Squamous Carcinoma (HNSC)
- Lung Adenocarcinoma (LUAD)
- Lung Squamous Carcinoma (LUSC)
- Breast Carcinoma (BRCA)
- Kidney Renal Clear Cell Carcinoma (KIRC)
- Ovarian Carcinoma (OV)
- Bladder Carcinoma (BLCA)
- Colon Adenocarcinoma (COAD)
- Uterine Cervical and Endometrial Carcinoma (UCEC)
- Rectal Adenocarcinoma (READ)

This collection of data is part of the RNA-Seq (HiSeq) PANCAN dataset. It is a random extraction of gene expressions of patients having different types of tumors: BRCA, KIRC, COAD, LUAD, and PRAD.

This dataset is a random collection of cancer patients from 801 patients, each having 20,531 attributes. Samples (instances) are stored row-wise. Variables (attributes) of each sampl are RNA-Seq gene expression levels measured by the illumina HiSeq platform. A dummy name (`gene_XX`) is given to each attribute. The attributes are ordered consistently with the original submission. For example, `gene_1` on `sample_0` is significantly and differentially expressed with a a value of `2.01720929003`.

Dataset Files:
- `data.csv`: Contains the gene expression data of each sample
- `labels.csv`: The labels associated with each sample


---------
<sup>1</sup> _Using deep learning to enhance cancer diagnosis and classification_ by R. Fakoor et al. in proceedings of the International Conference on Machine Learning, 2013.

<sup>2</sup> PCA is a statistical technique used to emphasize variation and extract the most significant patterns from a dataset; principal components are the simplest of the true eigenvector-based multivariate analyses. PCA is frequently used for making data exploration easy to visualize. Consequently, PCA is one of the most used algorithms in exploratory data analysis and for making predictive models.

<sup>3</sup> _Multilevel gene/miRNA feature selection using deep belief nets and active learning_ (R. Ibrahim, et al.) in Proceedings 36th annual International Conference Eng. Med. Biol. Soc. (EMBC), pp. 3957-3960, IEEE, 2014.

<sup>4</sup> _Integrative data analysis of multi-platform cancer data with a multimodal deep learning approach_ (by M. Liang et al.) in Molecular Pharmaceutics, vol. 12, pp. 928{937, IEEE/ACM Transaction Computational Biology and Bioinformatics, 2015.
