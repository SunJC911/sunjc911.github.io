---
title: RecBole环境配置 + GSC + LightGCN（rb）
description: Titan X rb
date: 2022-09-05
categories:
 - RecBole
tags:
 - RecBole
excerpt_separator: <!--more--> 

---



<!--more--> 

# 环境配置

Python：3.8

Pytorch：1.12.1

CUDA：11.3

PyG：2.1.0

RecBole：1.0.1（conda安装的为0.2.0，从pycharm环境里升级为1.0.1）**可用**

faiss-gpu：1.7.2

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

conda install -c aibox recbole
conda install faiss-gpu -c pytorch
```

实测ECCV-2022-Generative Subgraph Contrast for Self-Supervised Graph Representation Learning（**GSC**）和WWW-2022-Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning（**NCL**）都可以跑。

## 要解决的问题

### 怎么生成GSC中的node_neighbor_cen？

![node_neighbor_cen](https://sunjc911.github.io/assets/images/rb/node_neighbor_cen.png)

```
node_neighbor_cen = func.sub_sam(nodes_batch, adj_lists, k1)
```

nodes_batch: 随机选择节点作为中心节点（会重复）

adj_lists: 元组，每个节点与谁交互

![adj_lists](https://sunjc911.github.io/assets/images/rb/adj_lists.png)

k1：每个子图BFS搜索到的节点数量（包括自己），设为k=15待调参

目前准备在NCL trainer 47行加入node_neighbor_cen

1.需要知道nodes，手动查ml-1m user6040，item3629，共9629个节点

ncl.py中def make_adj(self): 用来制作adj涉及问题：

**ncl.py 54行 coo_matrix怎么转list？制作一个函数**

进阶问题：每个batch多少个节点？每个中心节点找几个节点？

进阶模块：用户和物品分开（模糊思路）？BFS换一下？

### gen生成器怎么放入recbole框架下？