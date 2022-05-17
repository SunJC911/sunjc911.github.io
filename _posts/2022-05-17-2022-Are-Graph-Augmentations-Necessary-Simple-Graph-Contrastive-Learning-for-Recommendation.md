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
---

## 摘要

作者指出，在CL-based推荐系统中，真正影响性能的是CL的损失，而不是原始图的数据增强。基于此发现，作者提出SimGCL，未使用原始图数据增强，在图embedding空间加入均匀分布噪声(uniform noises)来创造对比视图。

<!--more-->

![title](https://sunjc911.github.io/assets/images/SimGCL/title.png)

## 介绍

对比学习(Contrastive Learning, CL)是解决推荐系统数据集稀疏问题的好方法，因为它能从未标记数据中提取普遍的特征，并用自监督(Self-supervised Learning, SSL)方法进行正则化(regularize)数据的表示(representation)。CL一般采用数据增强方法构造原始图的对比视图，通过encoder最大化不同视图的一致性表示。但有文献指出就算是edge dropout rate 0.9的对比视图也能有效果，而rate 0.9的视图已经损失了大量的信息并且拥有一个高度倾斜的结构，所以这是违反人们常理的。作者借此提出：*当CL和Recommendation结合时，我们真的需要图数据增强吗？*

实验结果表明，CL的loss(如InfoNCE)才是性能的关键。图数据增强也不是没有用处，它能帮助模型学习到图不受扰动因素影响的表示。然而图数据增强非常耗人工和时间。作者提出第二个问题：*是否有高效的数据增强方法？*

## GCL for recommendation

以SGL为例子







