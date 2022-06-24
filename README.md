# MaskGAE
PyTorch implementation of the paper [MaskGAE: Masked Graph Modeling Meets Graph Autoencoders](https://arxiv.org/abs/2205.10053).

<p align="center"> <img src="framework.png" /> <p align="center"><em>Fig. 1. Masked Graph Autoencoders.</em></p>

# Requirements

- ogb == 1.3.3
- torch_sparse == 0.6.10
- torch_cluster == 1.5.9
- torch_geometric == 2.0.4
- torch == 1.9.0
- scipy == 1.7.3
- numpy == 1.18.5

# Installation

```bash
pip install -r requirements.txt
```

# Reproduce

For link prediction tasks, see `linkpred.ipynb` .

For node classification tasks, see `nodeclas.ipynb`.

# Cite

```bibtex
@article{li_maskgae,
  author    = {Jintang Li and
               Ruofan Wu and
               Wangbin Sun and
               Liang Chen and
               Sheng Tian and
               Liang Zhu and
               Changhua Meng and
               Zibin Zheng and
               Weiqiang Wang},
  title     = {MaskGAE: Masked Graph Modeling Meets Graph Autoencoders},
  journal   = {CoRR},
  volume    = {abs/2205.10053},
  year      = {2022}
}
```