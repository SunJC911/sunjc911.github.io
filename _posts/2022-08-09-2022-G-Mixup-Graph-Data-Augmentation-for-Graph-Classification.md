---
title: 2022-G-Mixup:Graph Data Augmentation for Graph Classification
description: ICML2022 Outstanding Paper
date: 2022-08-09
categories:
 - ICML
tags:
 - Graph
 - DataAugmentation
 - GraphClassification
excerpt_separator: <!--more--> 

---

## 摘要

Mixup通过在两个随机样本之间插入特征和标签来提高神经网络的泛化和鲁棒性。应用在图数据增强的挑战：不同数量的节点，不容易对齐，在非欧几里得空间中的类型学具有特殊性。为此，我们提出G-Mixup，通过插入不同类别图的生成器(即graphon)来扩充图的分类。具体来说，首先使用同一类中的图来估计一个 graphon。然后，在欧几里得空间中对不同类的 graphons 进行插值，得到混合的 graphons，合成图便是通过基于混合 graphons 的采样生成的。经实验评估，G-Mixup 显着提高了图神经网络的泛化性和鲁棒性。<!--more-->

![title](https://sunjc911.github.io/assets/images/Gmixup/title.png)

## 介绍

GNN在图分类表现很好，数据增强提高泛化能力。但是现存图数据增强方法是within-graph，修改单个图中边或节点获得数据增强图，这不能支持不同实例间的信息交换。between-graph有待开发。

**Mixup可提高图像识别、NLP中的泛化和鲁棒性。**Mixup的基本思想是对随机样本对的连续值进行线性插值，生成更多的合成训练数据。
$$
x_{new}=\lambda x_i+(1-\lambda)x_j\\y_{new}=\lambda y_i+(1-\lambda)y_j
$$
(xi,yi),(xj.yj)为两个随机样本，x为数据，y为one-hot标签。

作者提出问题：**mixup是否可以增强GNN的泛化和鲁棒性？**

Mixup要求原始数据实例在欧几里得空间中是规则且对齐良好的，比如图像数据和表数据。但**对graph数据使用有困难**，(1)图数据是不规则的，不同图中节点的数量通常是不同的;(2)图数据不对齐，图中的节点没有自然排序，很难匹配不同图之间的节点;(3)类之间的图拓扑是发散的，来自不同类的一对图的拓扑通常是不同的，而来自同一类的一对图的拓扑通常是相似的。因此不能直接用Mixup。

**为了解决上述问题**，提出**class-level**的**G-Mixup**，基于graphons生成图数据。一个类的图由一个graphon生成，混合不同graphons生成图。graphons可以被看成是概率矩阵$$W_G$$和$$W_H$$。$$W(i, j)$$表示节点i和j之间有边的概率。现实世界的图可以被看成是graphons生成的图。由于不同图的图形是规则的、对齐良好的，并且定义在欧几里德空间中，因此很容易和自然地混合这些图形，然后生成由此而来的合成图。基于此，只需要混合graphs。我们还提供了graphs mixup的理论分析，这保证了生成的图将保留两个原始类的关键特征。i.i.d表示独立同分布

![overview](https://sunjc911.github.io/assets/images/Gmixup/overview.png)

关键步骤：I)为每一类图估计一个图，ii)混合不同图类的图，iii)基于混合图生成合成图。

**贡献如下**:首先，我们提出了G-Mixup来扩充图分类的训练图。由于直接混合图是很难的，G-Mixup混合不同类型图的图形来生成合成图。其次，我们从理论上证明了合成图将是原始图的混合，其中源图的关键拓扑(即判别母图motifs)将混合在一起。第三，我们在各种图神经网络和数据集上证明了G-Mixup的有效性。大量的实验结果表明，G-Mixup在增强泛化和鲁棒性方面大大提高了图神经网络的性能。

## 预备知识（详见论文）

motifs：一个图可以包含一些频繁的子图，这些子图被称为母图motifs。

图同态，graphon，图分类

## 方法

图数据是nontrivial，因为图是irregular, unaligned and non-Euclidean数据。本文将证明graphons theory可以解决上述问题。具体来说，G-Mixup对不同的图生成器进行插值，得到一个新的混合图生成器。然后，基于混合图对合成图进行采样，实现数据扩充。从该生成器中采样的图部分具有原始图的性质。

![formula](https://sunjc911.github.io/assets/images/Gmixup/formula.png)

### 应用

#### Graphon Estimation and Mixup

估计graphon是使用G-Mixup先决条件。然而graphon是一个未知函数，没有真实世界图数据的封闭表达式。使用step function[参考文献]去近似graphons。step function可以看作一个矩阵W,其中Wij为存节点间存在边的概率。使用矩阵形式的step function。step function估计方法已经得到了很好的研究，该方法首先根据节点测量值(如度)对一组图中的节点进行对齐，然后从所有对齐的邻接矩阵中估计step function。

#### Synthetic Graphs Generation

一个 graphon W 提供一个分布来生成任意大小的图。合成图节点特征的生成包括两个步骤:1)基于原始节点特征构建graphon节点特征，2)基于graphon节点特征生成合成图节点特征。具体来说，在graphon估计阶段，我们在对齐邻接矩阵的同时对齐原始节点特征，因此我们对每个graphon都有一组对齐的原始节点特征，然后我们对对齐的原始节点特征进行池化(在我们的实验中是平均池化)，获得图形节点特征。生成的图的节点特征与graphon特征相同。

## 理论分析（详见论文）

## 实验

![exp-graphons](https://sunjc911.github.io/assets/images/Gmixup/exp-graphons.png)

表明不同图类的graphons不同

![casestudy](https://sunjc911.github.io/assets/images/Gmixup/casestudy.png)

一个可视化例子

![bong+g](https://sunjc911.github.io/assets/images/Gmixup/bong+g.png)

骨干为GCN和GIN+Gmixup

![pool+g](https://sunjc911.github.io/assets/images/Gmixup/pool+g.png)

不同池化方法+Gmixup

![robustness](https://sunjc911.github.io/assets/images/Gmixup/robustness.png)

标签破坏不同比例下的robustness

![epochloss](https://sunjc911.github.io/assets/images/Gmixup/epochloss.png)

loss总体降低

![toprobustness](https://sunjc911.github.io/assets/images/Gmixup/toprobustness.png)

## 结论

这项工作发展了一种新的图增广方法，称为G-Mixup。与图像数据不同，图数据是不规则的、未对齐的，并且处于非欧几里得空间中，因此很难混淆。然而，一个类中的图具有相同的生成器(即graphon)，它是正则的，对齐的，并且在欧几里得空间中。因此，我们转而混合不同类的图形来生成合成图。G-Mixup是对不同类型图的拓扑进行混合和插值。综合实验表明，G-Mixup训练的gnn具有更好的性能和泛化性能，提高了模型对噪声标签和损坏拓扑的鲁棒性。

## 代码

![code](https://sunjc911.github.io/assets/images/Gmixup/code.png)

https://github.com/ahxt/g-mixup

## 想法

通过聚类给节点上伪标签，通过EM算法进行迭代更新？

概率矩阵W可不可以做成有概率阈值的，大于多少保留不然切掉？

线性插值是否可以自适应？

~~二元分类是否可以转化为交互和未交互的分类？~~

**用于创建CL的图**

是否可以实验得出同一个数据集中，不同伪标签下类的graphon不同

