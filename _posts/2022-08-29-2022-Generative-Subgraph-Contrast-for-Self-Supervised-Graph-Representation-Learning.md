---
title: 2022-Generative Subgraph Contrast for Self-Supervised Graph Representation Learning
description:
date: 2022-08-29
categories:
 - ECCV
tags:
 - Graph
 - CL
 - SSL
 - RepresentationLearning
excerpt_separator: <!--more--> 


---

## 摘要

**（related work值得看一下那些文献）**发现的问题：扰动不能很好捕捉内在(intrinsic)(?)**局部图**的结构，基于向量内积的余弦相似度不能充分利用图的**局部结构**来很好地表征图的差异。提出一种新的基于**自适应子图生成的对比学习框架**，用于高效、鲁棒的自监督图表示学习，并利用最优传输距离作为子图之间的相似性度量。它的目的是通过捕捉图的内在结构来生成对比样本，同时根据子图的特征和结构来区分样本。具体来说，对于每个中心节点，通过自适应学习对应邻域节点的关系权值，我们首先开发一个网络来**生成插值子图**。然后，我们分别从相同的节点和不同的节点构造子图的正对和负对。最后，我们使用两种类型的最优传输距离(即Wasserstein距离和Gromov-Wasserstein距离)来构建结构化对比损失。实验为节点分类。<!--more-->

![title](https://sunjc911.github.io/assets/images/GSC/title.png)

## 介绍

手动扰动针对特定数据集，不能自适应，可能产生不同正样本。readout函数通常用于构造节点/图之间的矢量相似度度量，忽略了图的结构。因此，基于向量内积的相似度度量不能很好地表征图的差异。

先通过广度优先搜索（BFS）采样子图。然后，我们开发了一个子图生成网络，自适应地生成子图，其节点用学习到的权值插值到特征空间中。对于每个节点，我们可以对相邻节点分配不同的**注意权值**，从而得到加权节点，从而形成的子图可以捕捉图的内在几何结构。因此，我们用采样的子图和相同中心节点的生成子图构造正对，用不同中心节点的采样和生成子图构造负对。最后，基于构造的正/负子图，我们构造结构化的对比损失来学习具有Wasserstein距离和Gromov-Wasserstein距离的节点表示。**结构化对比损失可以使正子图之间的几何差异最小化，使负子图之间的几何差异最大化。**

## 方法

基于广度优先搜索的采样子图，我们首先自适应地生成对比子图来构造正/负样本。然后，我们使用最佳传输距离(即Wasserstein距离和Gromov-Wasserstein距离)来计算构建样本之间的对比损失。

![overall](https://sunjc911.github.io/assets/images/GSC/overall.png)

### Adaptive Subgraph Generation

扰动操作可能会丢失重要信息，甚至破坏图的固有结构。因此，构建的样本可能不够有识别力来训练对比学习模型。

为了构建更有效的对比样本，我们提出了一个可学习的子图生成模块来生成正/负子图样本。期望所生成的子图能很好地刻画图的固有局部结构。

![fig2](https://sunjc911.github.io/assets/images/GSC/fig2.png)

根据BFS抽样的子图生成对应节点，i为中心节点，j为邻居节点，aj为可学习的权重

![f1](https://sunjc911.github.io/assets/images/GSC/f1.png)

![f2](https://sunjc911.github.io/assets/images/GSC/f2.png)

![f3](https://sunjc911.github.io/assets/images/GSC/f3.png)

根据生成的节点生成边（邻接矩阵）

![f4](https://sunjc911.github.io/assets/images/GSC/f4.png)

![f4_2](https://sunjc911.github.io/assets/images/GSC/f4_2.png)

至此，我们得到了生成的包含节点特征和边的对比子图。本质上，我们使用**自适应生成**的样本来**取代基于扰动的样本**。与基于扰动的随机丢弃图信息的方法不同，所提出的生成模块能够**保持图的完整性**。我们的生成模块通过将学习到的注意权值分配给邻域节点，可以自适应地利用图的固有几何结构，生成更有效的对比样本。此外，由于图中相邻节点之间的相似性是一种固有属性，因此中心节点与其邻域之间存在很强的相关性。因此，**邻域插值生成的子图与原始子图具有内在的相似性**。将生成的子图作为正样本处理是合理有效的。

### OT（Optimal Transport） Distance Based Contrastive Learning

大多数图对比学习方法使用节点对或节点-子图对或子图对作为对比样本。特别的是，**子图的特征可以用读出函数来提取**。因此，这些方法主要采用矢量相似度度量来计算这些样本之间的相似度。然而，基于向量内积的相似性度量不能充分利用图的局部结构来很好地描述图的差异。本文使用**最优传输距离**(optimal transport distance (i.e., Wasserstein distance and Gromov-Wasserstein
distance) )作为对比子图的相似度度量。因此，我们可以**准确地描述子图之间的几何差异**(?)。

#### Wasserstein distance (WD)

WD通常用于匹配两个离散分布(例如，两组节点嵌入)[1]。它可以通过计算**两个子图中所有节点对**的差值来表示将一个子图转换为另一个子图的成本。在我们的设置中，WD被用来衡量子图节点之间的相似性。

与基于读出函数的方式相比，WD可以利用所有节点之间的相似信息，更有效地区分对比样本。因此，利用基于wd的对比损失，我们可以最大化正子图节点之间的相似性，最小化负子图节点之间的相似性。

#### Gromov-Wasserstein distance (GWD)

GWD[4,25]是在我们只能得到**每个子图内节点对之间的距离时使用**的。GWD可以用来计算子图中节点对之间的距离，以及测量子图之间这些距离的差异。也就是说，GWD可以测量每个子图中节点对之间的距离，并与对应子图中的节点对进行比较。因此，GWD可以用来捕获子图**边之间的相似度**。

基于gwd的对比损失算法可以最大化正子图边之间的相似性，最小化负子图边之间的相似性，从而获取子图之间的几何差异。

## 实验

节点分类任务

消融实验

![table](https://sunjc911.github.io/assets/images/GSC/table.png)

## 想法

采样正样本的策略，采样负样本的策略，全局信息，局部信息。数据增强的策略和扰动程度。

OT对均匀性和一致性的影响？

只有局部，加上考虑全局的方法试试？

## 参考文献

[1] Chen, L., Gan, Z., Cheng, Y., Li, L., Carin, L., Liu, J.: Graph optimal transport for cross-domain alignment. In: International Conference on Machine Learning. pp. 1542–1553. PMLR (2020)

