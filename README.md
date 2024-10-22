# DiscoGNN
This is a Pytorch implementation of the DiscoGNN paper: 

## Installation
We used the following Python packages for core development. We tested on `Python 3.7`.
```
pytorch                   1.0.1
torch-cluster             1.2.4              
torch-geometric           1.0.3
torch-scatter             1.1.2 
torch-sparse              0.2.4
torch-spline-conv         1.0.6
rdkit                     2019.03.1.0
tqdm                      4.31.1
tensorboardx              1.6
```

## Dataset download
All the necessary data files can be downloaded from the following links.

For the chemistry dataset, download from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `dataset/`.

## Pre-training and fine-tuning
#### 1. Self-supervised pre-training
```
python pretrain_disco.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 2. Fine-tuning
```
python finetune.py --input_model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH
```
This will finetune pre-trained model specified in `INPUT_MODEL_PATH` using dataset `DOWNSTREAM_DATASET.` The result of fine-tuning will be saved to `OUTPUT_FILE_PATH.`

## Reproducing results in the paper
Our results in the paper can be reproduced using a random seed ranging from 0 to 9 with scaffold splitting. 

## Acknowledgment
[1] Strategies for Pre-training Graph Neural Networks (Hu et al., ICLR 2020)

## Citation
```
@inproceedings{
xia2023molebert,
title={Mole-{BERT}: Rethinking Pre-training Graph Neural Networks for Molecules},
author={Jun Xia and Chengshuai Zhao and Bozhen Hu and Zhangyang Gao and Cheng Tan and Yue Liu and Siyuan Li and Stan Z. Li},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=jevY-DtiZTR}
}
```

