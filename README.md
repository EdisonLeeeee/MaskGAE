# What’s Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders
PyTorch implementation of What’s Behind the Mask: Understanding Masked Graph Modeling
for Graph Autoencoders

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
+ Collab
```bash
python train_linkpred_ogb.py
python train_linkpred_ogb.py --mask Edge
```

## Node classification

+ Cora
```bash
python train_nodeclas.py --dataset Cora --bn --l2_normalize --alpha 0.004
python train_nodeclas.py --dataset Cora --bn --l2_normalize --alpha 0.003 --mask Edge --eval_period 10
```
+ Citeseer
```bash
python train_nodeclas.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1 --alpha 0.001 --lr 0.02
python train_nodeclas.py --dataset Citeseer --bn --l2_normalize --nodeclas_weight_decay 0.1 --alpha 0.001  --lr 0.02 --mask Edge  --eval_period 20
```
+ Pubmed
```bash
python train_nodeclas.py --dataset Pubmed --bn --l2_normalize --alpha 0.001  --encoder_dropout 0.5 --decoder_dropout 0.5
python train_nodeclas.py --dataset Pubmed --bn --l2_normalize --alpha 0.001  --encoder_dropout 0.5 --mask Edge
```
+ Photo
```bash
python train_nodeclas.py --dataset Photo --bn --nodeclas_weight_decay 5e-3 --decoder_channels 128 --lr 0.005
python train_nodeclas.py --dataset Photo --bn --nodeclas_weight_decay 5e-3 --decoder_channels 64 --mask Edge
```
+ Computers
```bash
python train_nodeclas.py --dataset Computers --bn --encoder_dropout 0.5 --alpha 0.002 --encoder_channels 128 --hidden_channels 256 --eval_period 20
python train_nodeclas.py --dataset Computers --bn --encoder_dropout 0.5 --alpha 0.003 --encoder_channels 128 --hidden_channels 256 --eval_period 10 --mask Edge
```
+ arxiv
```bash
python train_nodeclas.py --dataset arxiv --bn --decoder_channels 128 --decoder_dropout 0. --decoder_layers 4 \
                          --encoder_channels 256 --encoder_dropout 0.2 --encoder_layers 4 \
                          --hidden_channels 512 --lr 0.0005 --nodeclas_weight_decay 0 --weight_decay 0.0001 --epochs 100  \
                          --eval_period 10         
python train_nodeclas.py --dataset arxiv --bn --decoder_channels 128 --decoder_dropout 0. --decoder_layers 4 \
                          --encoder_channels 256 --encoder_dropout 0.2 --encoder_layers 4 \
                          --hidden_channels 512 --lr 0.0005 --nodeclas_weight_decay 0 --weight_decay 0.0001 --epochs 100  \
                          --eval_period 10 --mask Edge
```
+ MAG
```bash
python train_nodeclas.py --dataset mag --alpha 0.003 --bn --decoder_channels 128\
                         --encoder_channels 256 --encoder_dropout 0.7 --epochs 100 \
                         --hidden_channels 128 --nodeclas_weight_decay 1e-5 --weight_decay 5e-5 --eval_period 10                                       
python train_nodeclas.py --dataset mag --alpha 0.003 --bn --decoder_channels 128\
                         --encoder_channels 256 --encoder_dropout 0.7 --epochs 100 \
                         --hidden_channels 128 --nodeclas_weight_decay 1e-5 --weight_decay 5e-5 --eval_period 10 --mask Edge   
```
