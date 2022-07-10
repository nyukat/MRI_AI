# Improving breast cancer diagnostics with deep learning for MRI  
[![DOI](https://zenodo.org/badge/421948410.svg)](https://zenodo.org/badge/latestdoi/421948410)

## Introduction

This repository contains script for processing the [TCGA-BRCA](https://wiki.cancerimagingarchive.net/display/Public/TCGA-BRCA) breast MRI data set. Specifically, there are two key files in the repository:

### tcga_brca_key.yaml
This file identifies all pre- and post-contrast sequences for all examinations in the data set. For example, `102: t1pre` indicates that series number 102 contains pre-contrast images. Additionally, this file recognizes different ways in which volumes are saved. In some MRI exams, left and right breasts are saved in separate series. Please refer to the file for more details.


### Process_TCGA_BRCA.ipynb
This jupyter notebook utilizes information from the `tcga_brca_key.yaml` to do the following tasks:
* load DCE-MRI exams from the TCGA-BRCA data set,
* identify pre- and post-contrast sequences,
* resample them to the same anisotropic spacing,
* reorient them to LPS orientation,
* save them to a standardized nifti (.nii) format, so that for each breast MRI exam there are three files: `t1pre.nii.gz`, `t1c1.nii.gz`, and `t1c2.nii.gz`.

## Reference
Files and scripts above have been used in our article, "_Improving breast cancer diagnostics with deep learning for MRI_" by Jan Witowski et al.

