---
title: 2022-Enhancing Sequential Recommendation with Graph Contrastive Learning
description:
date: 2022-06-01
categories:
 - IJCAI
tags:
 - CL
 - Graph
 - Rec
 - Sequential
excerpt_separator: <!--more--> 
---

## 摘要

顺序推荐系统捕捉用户的动态行为模式，预测用户的下一个交互行为。现有的大多数序列推荐方法仅利用单个交互序列的局部上下文信息，仅基于项目预测损失来学习模型参数。因此，他们通常无法学习适当的序列表示。本文提出GCL4SR，采用a Weighted Item Transition Graph (WITG)，基于所有用户的交互序列构建，为每次交互提供全局上下文信息，弱化序列数据中的噪声信息。此外,GCL4SR使用WITG的子图来增加每个交互序列的表示。提出了两个辅助学习目标，以最大化相同交互序列在WITG上增强表示之间的一致性，最小化在WITG上由全局上下文增强的表示与原始序列的局部表示之间的差异。在真实数据集上的大量实验证明了GCL4SR始终优于最先进的顺序推荐方法。<!--more-->

![title](https://sunjc911.github.io/assets/images/GCL4SR/title.png)

## 介绍

SOTA SR有不足：1、现有的方法对每个用户交互序列单独建模，只利用每个序列中的局部上下文。然而，它们通常忽略具有相似行为模式的用户之间的相关性(例如，具有相同的物品子序列)。2、用户行为数据非常稀疏。以往的方法通常只使用物品预测任务来训练推荐模型。他们往往遭受数据稀疏问题，不能学习适当的序列表示。3、序列推荐模型通常基于隐式反馈序列，其中可能包含噪声信息。

简单概述：只用自己的，数据稀疏问题，噪声。

为了解决上述问题，构造Weighted Item Transition Graph (WITG)，通过观察到的所有用户的交互序列来描述物品转换模式（item transition patterns）。此转换图可以为每个user-item交互提供全局上下文信息。为了减轻数据稀疏性的影响，WITG的邻域采样（neighborhood sampling）为每个交互序列建立增强图视图（augmented graph views）。然后，图表对比学习被用来学习用户交互序列的增强表示，这样，WITG上的全局上下文信息可以自然地纳入到增强表示中。此外，由于WITG使用转换频率来描述每个物品转换的重要性，在学习序列表示时，它可以帮助减弱用户交互序列中噪声交互的影响。

本文提出GCL4SR，利用从WITG采样的子图来利用不同序列的全局上下文信息。通过在WITG上增加序列视图来容纳全局上下文信息，改进了序列推荐任务。此外，我们还开发了两个辅助学习目标，以最大化相同交互序列在WITG上增强表示之间的一致性，最小化在WITG上由全局上下文增强的表示与原始序列的局部表示之间的差异。实验结果成为SOTA。

## 预备知识

常规定义

### 如何构建WITG

以一个序列S为例，对于每个S中的$$v_{t}$$，如果$$v_{t}$$和$$v_{t+k}$$之间存在边，则更新边的权重为$$w(v_{t},v_{t+k})←w(v_{t},v_{t+k})+1/k$$；否则构造一条边，边权重为1/k，其中k∈{1，2，3}（empirically）。这里的1/k表示在序列S中目标节点$$v_{t}$$对于k阶邻居$$v_{t+k}$$的重要性。灵感来源于LIghtGCN（?）。构造完所有user的S后，正则化边权值


$$
\widehat{w}(v_{t},v_{j})=w(v_{t},v_{j})(\frac{1}{deg(v_{i})}+\frac{1}{deg(v_{j})})
$$


deg(·)表示节点的度。WITG($$\mathcal{G}$$)是无向图。下图展示正则化前边权重的确定

![WITG](https://sunjc911.github.io/assets/images/GCL4SR/WITG.png)

## GCL4SR

![overall](https://sunjc911.github.io/assets/images/GCL4SR/overall.png)

### Graph Augmented Sequence Representation Learning

#### Graph-based Augmentation

构造完WITG($$\mathcal{G}$$)后，为S构造对比视图，使用邻居抽样（neighborhood sampling）[1]，从给定序列的大型转换图生成增强图视图。具体而言，对于每个节点，深度为2，每一步不考虑边权值采样20个，迭代采样所有节点。得到$$\mathcal{G}^{’}_{S}=(V^{’}_{S},E^{’}_{S},A^{’}_{S})$$和$$\mathcal{G}^{''}_{S}=(V^{''}_{S},E^{''}_{S},A^{''}_{S})$$。其中V，E，A表示节点，边，归一化后的边权重。

#### Shared Graph Neural Networks

GCN+GraphSage(mean pooling)

#### Graph Contrastive Learning Objective

相同序列的增广图为正对

$$\left\{(\mathcal{G}^{’}_{S},\mathcal{G}^{''}_{S})|S∈D\right\}$$

不同序列的增广图为负对

$$\left\{(\mathcal{G}^{’}_{S},\mathcal{G}^{''}_{K})|S,K∈D,S≠K\right\}$$

Loss:


$$
\mathcal{L}_{G C L}(S)=\sum_{S \in \mathcal{D}}-\log \frac{\exp \left(\cos \left(\mathbf{z}_{S}^{\prime}, \mathbf{z}_{S}^{\prime \prime}\right) / \tau\right)}{\sum_{K \in \mathcal{D}} \exp \left(\cos \left(\mathbf{z}_{S}^{\prime}, \mathbf{z}_{K}^{\prime \prime}\right) / \tau\right)}
$$


 $$\mathbf{z}_{S}^{\prime}$$和$$\mathbf{z}_{S}^{\prime \prime} \in \mathbb{R}^{1 \times d}$$是GNN输出的$$H^{'}_{S}$$和$$H^{"}_{S}\in \mathbb{R}^{n \times d}$$的mean pooling后的结果。

### User-specific Gating

每个用户都有自己喜欢的物品，所以全局上下文信息对于用户来说是user-specific。以下特定于用户的门机制用于捕获根据用户的个性化偏好定制的全局上下文信息，


$$
\alpha=\sigma\left(\mathbf{W}_{g 1} \mathbf{H}_{S}^{\prime \top}+\mathbf{W}_{g 2} \mathbf{p}_{u}^{\top}+b_{g}\right)
$$

$$
\mathbf{Q}_{S}^{\prime}=\alpha \otimes \mathbf{H}_{S}^{\prime}+(1-\alpha) \otimes \mathbf{p}_{u}
$$



$$\mathbf{W}_{g 1}$$和$$\mathbf{W}_{g 2} \in \mathbb{R}^{d \times d}, b_{g} \in \mathbb{R}^{d \times 1}$$，用户embedding$$\mathbf{p}_{u}$$描述用户的一般偏好。同样的对于$$\mathcal{G}^{''}_{S}$$,也要构造$$\mathbf{Q}_{S}^{\prime\prime}$$。

#### Representation Alignment Objective

最大均值差异（MMD）用来定义个性化全局上下文（$$\mathbf{Q}_{S}^{\prime}$$和$$\mathbf{Q}_{S}^{\prime\prime}$$）的表示和局部序列的表示（$$E^{(0)}_{S}$$）之间的距离。两个特征分布$$X \in \mathbb{R}^{m \times d}$$和$$Y \in \mathbb{R}^{\tilde{m} \times d}$$的MMD可以如下定义：


$$
M M D(\mathbf{X}, \mathbf{Y})=\frac{1}{m^{2}} \sum_{a=1}^{m} \sum_{b=1}^{m} \mathcal{K}\left(\mathbf{x}_{a}, \mathbf{x}_{b}\right)
+\frac{1}{\widetilde{m}^{2}} \sum_{a=1}^{\widetilde{m}} \sum_{b=1}^{\widetilde{m}} \mathcal{K}\left(\mathbf{y}_{a}, \mathbf{y}_{b}\right)-\frac{2}{m \widetilde{m}} \sum_{a=1}^{m} \sum_{b=1}^{\widetilde{m}} \mathcal{K}\left(\mathbf{x}_{a}, \mathbf{y}_{b}\right)
$$


K为卷积函数，xa，yb表示第a行的X和第b行的Y。这项工作中，K为带bandwidth的高斯核函数
$$
\mathcal{K}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=e^{-\frac{\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|^{2}}{2 \rho^{2}}}
$$


然后，最小化两者的距离


$$
\mathcal{L}_{M M}(S)=M M D\left(\mathbf{E}_{S}^{(0)}, \mathbf{Q}_{S}^{\prime}\right)+M M D\left(\mathbf{E}_{S}^{(0)}, \mathbf{Q}_{S}^{\prime \prime}\right)
$$


### Basic Sequence Encoder

出了增广对比，还用传统系列模型去编码用户交互序列。使用文献[2]作为骨架模型来构建用户交互序列。
$$
\mathbf{H}^{\ell}=\operatorname{FFN}\left(\right.$ Concat $\left(\right.$ head $_{1}, \ldots$, head $\left.\left._{h}\right) \mathbf{W}^{h}\right),
$$

$$
head_{i}=\operatorname{Attention}\left(\mathbf{H}^{\ell-1} \mathbf{W}_{i}^{Q}, \mathbf{H}^{\ell-1} \mathbf{W}_{i}^{K}, \mathbf{H}^{\ell-1} \mathbf{W}_{i}^{V}\right)
$$

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q K}^{\top}}{\sqrt{d}}\right) \mathbf{V}
$$



### Prediction Layer


$$
\mathbf{M}=\operatorname{AttNet}\left(\operatorname{Concat}\left(\mathbf{Q}_{S}^{\prime}, \mathbf{Q}_{S}^{\prime \prime}, \mathbf{H}^{\ell}\right) \mathbf{W}_{T}\right)
$$

$$
\hat{\mathbf{y}}^{(S)}=\operatorname{softmax}\left(\mathbf{M E}^{\top}\right),\hat{\mathbf{y}}^{(S)} \in \mathbb{R}^{1 \times|V|}
$$



### Multi Task Learning

将$$S_{u}=\left\{v_{u}^{1}, v_{u}^{2}, \cdots, v_{u}^{\left|S_{u}\right|}\right\}$$切片成一组子序列和目标标签：
$$
\left\{\left(S_{u}^{1: 1}, v_{u}^{2}\right),\left(S_{u}^{1: 2}, v_{u}^{3}\right), \cdots,\left(S_{u}^{1:\left|S_{u}\right|-1}, v_{u}^{\left|S_{u}\right|}\right)\right\}
$$
其中$$\left|S_{u}\right|$$表示长度，$$S_{u}^{1: k-1}=\left\{v_{u}^{1}, v_{u}^{2}, \cdots, v_{u}^{k-1}\right\}$$，$$v_{u}^{k}$$是$$S_{u}^{11: k-1}$$的目标标签。之后基于交叉熵得到我们的主loss：


$$
\mathcal{L}_{\text {main }}=-\sum_{S_{u} \in \mathcal{D}} \sum_{k=1}^{\left|S_{u}\right|-1} \log (\hat{\mathbf{y}}^{(S_{u}^{1: k})}(v^{k+1}_{u}))
$$


其中$$\hat{\mathbf{y}}^{\left(S_{u}^{1: k}\right)}\left(v_{u}^{k+1}\right)$$表示基于子序列$$S_{u}^{1: k}$$根据$$\hat{\mathbf{y}}^{(S)}$$的式子预测的$$v^{k+1}_{u}$$的交互概率。


$$
\mathcal{L}=\mathcal{L}_{\text {main }}+\sum_{S_{u} \in \mathcal{D}} \sum_{k=1}^{\left|S_{u}\right|-1} \lambda_{1} \mathcal{L}_{G C L}\left(S_{u}^{1: k}\right)+\lambda_{2} \mathcal{L}_{M M}\left(S_{u}^{1: k}\right)
$$


## 实验

![exp](https://sunjc911.github.io/assets/images/GCL4SR/exp.png)

## 参考文献

[1] William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NeurIPS’17.

[2] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In ICDM’18, pages 197–206. IEEE, 2018.