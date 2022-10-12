# MaskGAE: Masked Graph Modeling Meets Graph Autoencoders
PyTorch implementation of the paper [MaskGAE: Masked Graph Modeling Meets Graph Autoencoders](https://arxiv.org/abs/2205.10053).

<p align="center"> <img src="framework.png" /> <p align="center"><em>Fig. 1. Masked Graph Autoencoders.</em></p>

# Requirements
Higher versions should be also available.

+ numpy==1.18.1
+ torch==1.12.1+cu102
+ torch-cluster==1.6.0
+ torch_geometric>=2.1.0
+ torch-scatter==2.0.9
+ torch-sparse==0.6.14
+ CUDA 10.2
+ CUDNN 7.6.0

# Installation

```bash
pip install -r requirements.txt
```

# Reproduction

## Link prediction
+ Cora
```bash
python train_linkpred.py --dataset Cora --bn

python train_linkpred.py --dataset Cora --bn --mask Edge
```

+ Citeseer
```bash
python train_linkpred.py --dataset Citeseer --bn

python train_linkpred.py --dataset Citeseer --bn --mask Edge
```

+ Pubmed
```bash
python train_linkpred.py --dataset Pubmed --bn --encoder_dropout 0.2

python train_linkpred.py --dataset Pubmed --bn --encoder_dropout 0.2 --mask Edge
```

## Node classification

+ Cora
```bash
# 84.30 ± 0.39
python train_nodeclas.py --dataset Cora --bn --l2_normalize --alpha 0.004

# 83.77 ± 0.33
python train_nodeclas.py --dataset Cora --bn --l2_normalize --alpha 0.003 --mask Edge --eval_period 10
```

+ Citeseer
```bash
# 73.80 ± 0.81
python train_nodeclas.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1 --alpha 0.001 --lr 0.02

# 72.94 ± 0.20
python train_nodeclas.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1 --alpha 0.001  --lr 0.02 --mask Edge  --eval_period 20
```

+ Pubmed
```bash
# 83.58 ± 0.45
python train_nodeclas.py --dataset Pubmed --bn --l2_normalize --alpha 0.001  --encoder_dropout 0.5 --decoder_dropout 0.5

# 82.42 ± 0.58
python train_nodeclas.py --dataset Pubmed --bn --l2_normalize --alpha 0.001  --encoder_dropout 0.5 --mask Edge
```

+ Photo
```bash
# 93.33 ± 0.13
python train_nodeclas.py --dataset Photo --bn --nodeclas_weight_decay 5e-3 --decoder_channels 128 --lr 0.005

# 93.30 ± 0.04
python train_nodeclas.py --dataset Photo --bn --nodeclas_weight_decay 5e-3 --decoder_channels 64 --mask Edge

```

+ Computers
```bash
# 89.54 ± 0.06
python train_nodeclas.py --dataset Computers --bn --encoder_dropout 0.5 --alpha 0.002 --encoder_channels 128 --hidden_channels 256 --eval_period 20

# 89.44 ± 0.11
python train_nodeclas.py --dataset Computers --bn --encoder_dropout 0.5 --alpha 0.003 --encoder_channels 128 --hidden_channels 256 --eval_period 10 --mask Edge
```

+ arxiv
```bash
# 71.16 ± 0.33
python train_nodeclas.py --dataset arxiv --bn --decoder_channels 128 --decoder_dropout 0. --decoder_layers 4 \
                          --encoder_channels 256 --encoder_dropout 0.2 --encoder_layers 4 \
                          --hidden_channels 512 --lr 0.0005 --nodeclas_weight_decay 0 --weight_decay 0.0001 --epochs 100  \
                          --eval_period 10 
                 
# 70.97 ± 0.29
python train_nodeclas.py --dataset arxiv --bn --decoder_channels 128 --decoder_dropout 0. --decoder_layers 4 \
                          --encoder_channels 256 --encoder_dropout 0.2 --encoder_layers 4 \
                          --hidden_channels 512 --lr 0.0005 --nodeclas_weight_decay 0 --weight_decay 0.0001 --epochs 100  \
                          --eval_period 10 --mask Edge
```

+ MAG
```bash

# 32.79 ± 0.32
python train_nodeclas.py --dataset mag --alpha 0.003 --bn --decoder_channels 128\
                         --encoder_channels 256 --encoder_dropout 0.7 --epochs 100 \
                         --hidden_channels 128 --nodeclas_weight_decay 1e-5 --weight_decay 5e-5 --eval_period 10                   
                          
# 32.75 ± 0.43
python train_nodeclas.py --dataset mag --alpha 0.003 --bn --decoder_channels 128\
                         --encoder_channels 256 --encoder_dropout 0.7 --epochs 100 \
                         --hidden_channels 128 --nodeclas_weight_decay 1e-5 --weight_decay 5e-5 --eval_period 10 --mask Edge   
```

# Graph Classification

+ MUTAG
```bash
# 89.47 ± 0.11
python train_graphclas.py --dataset MUTAG --alpha 0.001 --bn --pooling sum --hidden_channels 128

# 89.45 ± 0.08
python train_graphclas.py --dataset MUTAG --alpha 0.001 --bn --pooling sum --hidden_channels 128 --mask Edge
```

+ IMDB-BINARY
```bash
# 74.63 ± 0.05
python train_graphclas.py --dataset IMDB-BINARY --alpha 0.001 --bn --pooling mean --hidden_channels 512 --decoder_layers 2

# 75.23 ± 0.04
python train_graphclas.py --dataset IMDB-BINARY --alpha 0.001 --bn --pooling mean --encoder_activation relu --hidden_channels 256 --decoder_layers 4 --mask Edge

```

+ TODO