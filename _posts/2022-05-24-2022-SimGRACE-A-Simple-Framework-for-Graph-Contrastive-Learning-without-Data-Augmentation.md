---
title: 2022-SimGRACE:A Simple Framework for Graph Contrastive Learning without Data Augmentation
description: 
date: 2022-05-24
categories:
 - WWW
tags:
 - CL
 - Graph
excerpt_separator: <!--more--> 
---

## 摘要

数据增强耗时耗力要经验，很难保持语义（semantic），限制了GCL的效率和普适性。作者提出SimGRACE，它不要数据增强。输入原始图到GNN模型和该模型的扰动版本中，得到两个相关图用于对比用来更新参数。SimGRACE关注**graph-level**表示学习。**灵感来源：想要保持语义，编码器的扰动能很好保持原始图的语义。**升级版AT-SimGRACE增强了鲁棒性。<!--more-->

![title](https://sunjc911.github.io/assets/images/SimGRACE/title.png)

## 介绍

**对比学习(Contrastive Learning, CL)可以学习到对扰动不变的表示。**但是扰动可能会改变语义，所以需要耗时耗力定制。作者提问：***我们能把图对比学习从繁琐的手工试错、繁琐的搜索或昂贵的领域知识中解放出来吗?***

作者提出SimGRACE，两个GNN模型及其扰动版本为两个编码器，得到两个相关视图。然后，最大化这两个视图的一致性。同一mini-batch中，相同原始输入的embedding为正对，其余为负对。对比其他SOTA算法：

![table1](https://sunjc911.github.io/assets/images/SimGRACE/table1.png)

**鲁棒性需要以训练时间为代价，因为对每个图进行对抗性转换非常耗时。**作者提出AT-SimGRACE，减少训练时间，提高鲁棒性。

## SimGRACE

![overall](https://sunjc911.github.io/assets/images/SimGRACE/overall.png)

### Encoder perturbation

GNN encoder和它的扰动版本：


$$
\mathbf{h}=f(G ; \theta), \mathbf{h}^{\prime}=f\left(G ; \theta^{\prime}\right)
$$


扰动的公式：


$$
\theta_{l}^{\prime}=\theta_{l}+\eta \cdot \Delta \theta_{l} ; \quad \Delta \theta_{l} \sim \mathcal{N}\left(0, \sigma_{l}^{2}\right)
$$


$$\theta_{l}$$和$$\theta_{l}^{\prime}$$为L层的参数及其扰动版本。扰动项服从高斯分布。

### Projection head

文献[1]说，a non-linear transformation g(·) named projection head将表示映射到另一个潜在空间，可以提高性能。Sim-GRACE中，采用一个两层感知器(MLP)来获得z和z’：


$$
z=g(\mathrm{h}), z^{\prime}=g\left(\mathrm{h}^{\prime}\right)
$$

### Contrastive loss

使用normalized temperature-scaled cross entropy loss (**NT-Xent**)加强z和z’一致性。z和z’改写为$$z_{n}, z_{n}^{\prime}$$


$$
\ell_{n}=-\log \frac{\left.\exp \left(\operatorname{sim}\left(z_{n}, z_{n}^{\prime}\right)\right) / \tau\right)}{\sum_{n^{\prime}=1, n^{\prime} \neq n}^{N} \exp \left(\operatorname{sim}\left(z_{n}, z_{n^{\prime}}\right) / \tau\right)}
$$

$$
\operatorname{sim}\left(z, z^{\prime}\right)=z^{\top} z^{\prime} /\|z\|\left\|z^{\prime}\right\|
$$

**InfoNCE分母为正负样本的和，NT-Xent分母只有负样本。**

Why can SimGRACE work well？

根据[2]的alignment和uniformity，SimGRACE的alignment和uniformity的loss都比sota小。

![au](https://sunjc911.github.io/assets/images/SimGRACE/au.png)

## EXPERIMENTS

结果稍微好或者持平或者低一点。

## 参考文献

[1] Chen Ting, Kornblith Simon, Norouzi Mohammad, and Hinton Geoffrey. 2020. **A Simple Framework for Contrastive Learning of Visual Representations**. in ICML.

[2] Wang Tongzhou and Isola Phillip. 2020. **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**. in ICML.