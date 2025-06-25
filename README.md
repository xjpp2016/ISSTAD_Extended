# ISSTAD on VisA and MPDD Datasets

## Overview

This repository extends [ISSTAD](https://github.com/xjspin/ISSTAD/) to two additional datasets: **VisA** and **MPDD**, for supplementary experiments beyond the original MVTec AD evaluation.

## Datasets

- **VisA**: [https://github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)  
- **MPDD**: [https://github.com/stepanje/MPDD](https://github.com/stepanje/MPDD)

## Pretrained MAE Model

Please download the pretrained MAE model from the official release:  
[https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth)

## Environment Setup

The implementation has been tested with the following dependencies:

```bash
python==3.9.13  
matplotlib==3.6.0  
numpy==1.23.3  
opencv-python==4.6.0.66  
pandas==1.5.1  
pillow==9.2.0  
scikit-learn==1.1.2  
scipy==1.9.1  
six==1.16.0  
timm==0.3.2  
torch==1.12.1+cu116  
tqdm==4.64.1  
```

The code is executable on Windows systems, and if running on Linux, it requires execution on a disk with an NTFS file system. Otherwise, the results may degrade, especially for the localization result on the MVTec AD dataset.

### Run
VisA dataset 
```bash
python main_visa_ad.py
```
MPDD dataset
```bash
python main_mpdd_ad.py
```