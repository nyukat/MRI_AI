# Improving breast cancer diagnostics with artificial intelligence for MRI
[![DOI](https://zenodo.org/badge/421948410.svg)](https://zenodo.org/badge/latestdoi/421948410)
## Introduction
This repository contains code that was used to train and evaluate deep learning models, as described in the article "_Improving breast cancer diagnostics with artificial intelligence for MRI_" by Jan Witowski et al. It includes model code, training loop, utilities, Jupyter notebooks and scripts used to evaluate results and generate plots. 

![Overview of the study](https://cdn.discordapp.com/attachments/992824619427954808/1007789851615178852/fig1.png "Overview of the study")

## TCGA-BRCA processing
This repository contains script for processing the [TCGA-BRCA](https://wiki.cancerimagingarchive.net/display/Public/TCGA-BRCA) breast MRI data set. Specifically, there are two key files in the repository:

### tcga_brca/tcga_brca_key.yaml
This file identifies all pre- and post-contrast sequences for all examinations in the data set. For example, `102: t1pre` indicates that series number 102 contains pre-contrast images. Additionally, this file recognizes different ways in which volumes are saved. In some MRI exams, left and right breasts are saved in separate series. Please refer to the file for more details.


### tcga_brca/Process_TCGA_BRCA.ipynb
This jupyter notebook utilizes information from the `tcga_brca_key.yaml` to do the following tasks:
* load DCE-MRI exams from the TCGA-BRCA data set,
* identify pre- and post-contrast sequences,
* resample them to the same anisotropic spacing,
* reorient them to LPS orientation,
* save them to a standardized nifti (.nii) format, so that for each breast MRI exam there are three files: `t1pre.nii.gz`, `t1c1.nii.gz`, and `t1c2.nii.gz`.

## Reference
```
@article{witowski2022improving,
   title = {Improving breast cancer diagnostics with artificial intelligence for MRI}, 
   author = {Jan Witowski and Laura Heacock and Beatriu Reig and Stella K. Kang and Alana Lewin and Kristine Pysarenko and Shalin Patel and Naziya Samreen and Wojciech Rudnicki and Elżbieta Łuczyńska and Tadeusz Popiela and Linda Moy and Krzysztof J. Geras},
   journal = {medRxiv:10.1101/2022.02.07.22270518}, 
   year = {2022}
}
```
