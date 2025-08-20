# ISSTAD on VisA and MPDD Datasets

This repository extends [ISSTAD](https://github.com/xjspin/ISSTAD/) to two additional datasets: **VisA** and **MPDD**, for supplementary experiments beyond the original MVTec AD evaluation.

## Datasets

- **VisA**: [https://github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)  
- **MPDD**: [https://github.com/stepanje/MPDD](https://github.com/stepanje/MPDD)

The training and testing data for VisA and MPDD should be placed in the `./data/visa/VisA/` and `./data/mpdd/MPDD/` directories, respectively.

Alternatively, you can use our **pre-organized dataset collection**, hosted on Hugging Face for direct use:  
👉 [https://huggingface.co/datasets/xjpha/ISSTAD_Data/tree/main](https://huggingface.co/datasets/xjpha/ISSTAD_Data/tree/main)

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

The code is executable on Windows systems. When running on Linux, it must be executed on a disk with an NTFS file system; otherwise, the performance may degrade, particularly for localization tasks.

**Note**: The `timm` version must be **0.3.2**.  
If you encounter the error `No module named 'torch._six'`, modify the file:  
`site-packages/timm/models/layers/helpers.py` by:

1. Commenting out this line:
```python
from torch._six import container_abcs
```

2. Replacing it with:
```python
import collections.abc as container_abcs
```



## Training
VisA dataset 
```bash
python train_visa_ad.py
```
MPDD dataset
```bash
python train_mpdd_ad.py
```

## Generate Heatmaps with a Single Command
VisA dataset 
```bash
python get_result_visa.py
```
MPDD dataset
```bash
python get_result_mpdd.py
```

### Citation
If you find this work helpful, please consider citing:

```
@article{JIN_ISSTAD,
title = {Incremental self-supervised learning based on transformer for anomaly detection and localization},
journal = {Engineering Applications of Artificial Intelligence},
volume = {160},
pages = {111978},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.111978},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625019864},
author = {Wenping Jin and Fei Guo and Qi Wu and Li Zhu},
keywords = {Anomaly detection, Vision transformer, Masked autoencoder, Pixel-level self-supervised learning},
}
```

```
@article{jin2023isstad,
  title={ISSTAD: Incremental self-supervised learning based on transformer for anomaly detection and localization},
  author={Jin, Wenping and Guo, Fei and Zhu, Li},
  journal={arXiv preprint arXiv:2303.17354},
  year={2023}
}
```