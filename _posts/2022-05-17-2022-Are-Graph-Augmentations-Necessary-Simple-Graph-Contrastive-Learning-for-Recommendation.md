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

## InfoNCE Loss Influences More

根据文献[3],CL会使得正样本在距离上更近（**alignment**），使特征在超球面上分布更均匀（**uniformity**）。

作者在两个数据集上实验（只有uniformity可视化，**未展示alignment可视化图**）。

![uniformity](https://sunjc911.github.io/assets/images/SimGCL/uniformity.png)

可以看到**CL使得特征超球分布更加均匀，缓和特征聚集程度**。

作者解释LightGCN的高度聚集分布原因。1.LightGCN的消息传递机制使得节点embedding变得相似（**过平滑**）；2.数据流行度偏差（**长尾问题**）。

作者将InfoNCE重写为下式：

$$\mathcal{L}_{c l}=\sum_{i \in \mathcal{B}}-1 / \tau+\log \left(\exp (1 / \tau)+\sum_{j \in \mathcal{B} /\{i\}} \exp \left(\mathbf{z}_{i}^{\top} \mathbf{z}_{j} / \tau\right)\right)$$

可以看出**cl loss实际是最小化$$\mathbf{e}_{i}$$和$$\mathbf{e}_{j}$$的余弦相似度**，从而使它们在空间上远离，计算在rec loss的影响下，在超球上也会显得均匀（uniformity）。

这就说明**分布的均匀性是SGL中推荐性能的决定性影响的潜在因素，而不是基于drop out的图增广**。优化CL loss可以看作是隐式的去偏方法，因为均匀分布可提高泛化能力。但只追求cl loss最小化也不好。大白话：**uniformi可视化图要分布均匀，但是也要有点聚集**。

## SIMGCL: SIMPLE GRAPH CONTRASTIVELEARNING FOR RECOMMENDATION

受到上述分析的启发（**在一定范围内调整学习到的表征（learned representation）的uniformity**），作者提出SimGCL。

给图结构添加均匀分布非常耗时和人力，所以作者将注意力转到embedding空间。受到[4]的启发，作者**直接在表示中加入随机噪声构造对比视图**：

$$\mathrm{e}_{i}^{\prime}=\mathrm{e}_{i}+\Delta_{i}^{\prime}, \quad \mathbf{e}_{i}^{\prime \prime}=\mathrm{e}_{i}+\Delta_{i}^{\prime \prime}$$

其中噪声向量$$\Delta_{i}^{\prime}$$和$$\Delta_{i}^{\prime \prime}$$都遵循$$\|\Delta\|_{2}=\epsilon$$和$$\Delta=\bar{\Delta} \odot \operatorname{sign}\left(\mathbf{e}_{i}\right)$$, $$\bar{\Delta} \in \mathbb{R}^{d} \sim U(0,1)$$。第一个约束$$\Delta$$的大小，$$\Delta$$是半径为$$\epsilon$$的超球上的点。第二个约束使得$$\mathrm{e}_{i}$$,$$\Delta_{i}^{\prime}$$和$$\Delta_{i}^{\prime \prime}$$在一个hyperoctant上，这样加入噪声不会引起较大偏差使得正样本变少。可视化：

![noise](https://sunjc911.github.io/assets/images/SimGCL/noise.png)

由于旋转足够小，**增广表示保留了原始表示的大部分信息，同时也保留了一些方差**。注意，对于每个节点表示，添加的随机噪声是不同的。

矩阵形式：

$$\begin{array}{r}
\mathrm{E}^{\prime}=\frac{1}{L}\left(\left(\tilde{\mathrm{A}} \mathrm{E}^{(0)}+\Delta^{(1)}\right)+\left(\tilde{\mathrm{A}}\left(\tilde{\mathrm{A}} \mathrm{E}^{(0)}+\Delta^{(1)}\right)+\Delta^{(2)}\right)\right)+\ldots \\
\left.+\left(\tilde{\mathrm{A}}^{L} \mathrm{E}^{(0)}+\tilde{\mathrm{A}}^{L-1} \Delta^{(1)}+\ldots+\tilde{\mathrm{A}} \Delta^{(L-1)}+\Delta^{(L)}\right)\right)
\end{array}$$

注意，计算最终表示时没有加入初始embedding$$\mathrm{E}^{(0)}$$,因为作者说这样效果更好。如果没有CL任务，在LightGCN中这样做会导致性能下降。

**最终Loss：BPR + InfoNCE**

## EXPERIMENTAL RESULTS

![exp](https://sunjc911.github.io/assets/images/SimGCL/exp.png)

此外作者还对比了不同噪声方法：

![noisecomparison](https://sunjc911.github.io/assets/images/SimGCL/noisecomparison.png)

## 参考文献

[1] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. **Self-supervised graph learning for recommendation**. In SIGIR.

[2] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** In SIGIR.

[3] TongzhouWang and Phillip Isola. 2020. **Understanding contrastive representation learning through alignment and uniformity on the hypersphere.** In ICML.

[4] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014. **Explaining and harnessing adversarial examples (2014)**. arXiv preprint arXiv:1412.6572 (2014).

## Code

### 处理数据

### 构造稀疏二部邻接矩阵

```
tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
```

**sp.csr_matrix**：压缩稀疏矩阵

tmp_adj矩阵大概：

![tmp_adj](https://sunjc911.github.io/assets/images/SimGCL/tmp_adj.png)

```
adj_mat = tmp_adj + tmp_adj.T
```

adj_mat 矩阵大概：

![adj_mat](https://sunjc911.github.io/assets/images/SimGCL/adj_mat.png)

### 其他

exec() : 执行括号内的语句；
