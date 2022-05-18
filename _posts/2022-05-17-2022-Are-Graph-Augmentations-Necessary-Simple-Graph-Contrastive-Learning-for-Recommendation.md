---
title: 2022-Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation
description:
date: 2022-05-17
categories:
 - SIGIR
tags:
 - CL
 - Graph
 - Rec
excerpt_separator: <!--more--> 
---

## 摘要

作者指出，在CL-based推荐系统中，**真正影响性能的是CL的损失，而不是原始图的数据增强**。基于此发现，作者提出SimGCL，未使用原始图数据增强，**在图embedding空间加入均匀分布噪声**(uniform noises)来创造对比视图。<!--more-->

![title](https://sunjc911.github.io/assets/images/SimGCL/title.png)

## 介绍

对比学习(Contrastive Learning, CL)是解决推荐系统数据集稀疏问题的好方法，因为它能从未标记数据中提取普遍的特征，并用自监督(Self-supervised Learning, SSL)方法进行正则化(regularize)数据的表示(representation)。CL一般采用数据增强方法构造原始图的对比视图，通过encoder最大化不同视图的一致性表示。下图以edge dropout为例。

![ED](https://sunjc911.github.io/assets/images/SimGCL/ED.png)

但有文献指出就算是edge dropout rate 0.9的对比视图也能有效果，而rate 0.9的视图已经损失了大量的信息并且拥有一个高度倾斜的结构，所以这是违反人们常理的。作者借此提出：***当CL和Recommendation结合时，我们真的需要图数据增强吗？***

实验结果表明，**CL的loss(如InfoNCE)才是性能的关键**。图数据增强也不是没有用处，它能帮助模型学习到图不受扰动因素影响的表示。然而图数据增强非常耗人工和时间。作者提出第二个问题：***是否有高效的数据增强方法？***

## GCL for recommendation

以SGL[1]为例子, 其使用节点和边的dropout进行数据增强。Loss为：

$$\mathcal{L}_{\text {joint }}=\mathcal{L}_{r e c}+\lambda \mathcal{L}_{c l}$$

包括$$\mathcal{L}_{\text {joint }}$$和$$\mathcal{L}_{c l}$$。$$\mathcal{L}_{c l}$$为InfoNCE：

$$\mathcal{L}_{c l}=\sum_{i \in \mathcal{B}}-\log \frac{\exp \left(\mathrm{z}_{i}^{\prime \top} \mathrm{z}_{i}^{\prime \prime} / \tau\right)}{\sum_{j \in \mathcal{B}} \exp \left(\mathrm{z}_{i}^{\prime \top} \mathrm{z}_{j}^{\prime \prime} / \tau\right)}$$

$$ \mathcal{B}$$为a sampled batch。**CL鼓励z'和z''的一致性。**

SGL以LightGCN[2]为backbone，其消息传递可写成矩阵形式：

$$\mathrm{E}=\frac{1}{1+L}\left(\mathrm{E}^{(0)}+\tilde{\mathrm{A}} \mathrm{E}^{(0)}+\ldots+\tilde{\mathrm{A}}^{L} \mathrm{E}^{(0)}\right)$$

其中：

$$\mathrm{E}^{(0)} \in \mathbb{R}^{|N| \times d}$$

是随机初始化的节点embedding；

$$\tilde{\mathrm{A}} \in \mathbb{R}^{|N| \times|N|}$$

是正则化无向邻接矩阵；

$$\mathrm{z}_{i}^{\prime}=\frac{\mathrm{e}_{i}^{\prime}}{\left\|\mathrm{e}_{i}^{\prime}\right\|_{2}}$$

其中$$\mathbf{e}_{i}^{\prime}$$是$$\mathrm{E}$$中$$\mathbf{e}_{i}$$的数据增强版本。

## Necessity of Graph Augmentation

作者的实验：对比SGL的不同变体性能

![SGLvariants](https://sunjc911.github.io/assets/images/SimGCL/SGLvariants.png)

ND为node dropout，ED为edge dropout，RW为random walk。WA为无数据增强，即$$\mathrm{Z}_{i}^{\prime}=\mathrm{Z}_{i}^{\prime \prime}=\mathrm{Z}_{i}$$,所以WA的$$\mathcal{L}_{c l}$$变为：

$$\mathcal{L}_{c l}=\sum_{i \in \mathcal{B}}-\log \frac{\exp (1 / \tau)}{\sum_{j \in \mathcal{B}} \exp \left(\mathrm{z}_{i}^{\top} \mathrm{z}_{j} / \tau\right)}$$

可以看到**ED最好，但是比WA只好一点**，说明**图增广的轻微扰动有用**。

## 参考文献

[1] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. **Self-supervised graph learning for recommendation**. In SIGIR.

[2] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** In SIGIR.



