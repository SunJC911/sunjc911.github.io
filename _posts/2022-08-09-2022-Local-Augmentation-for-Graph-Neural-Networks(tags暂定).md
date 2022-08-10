---
title: 2022-Local Augmentation for Graph Neural Networks(tags暂定)
description:
date: 2022-08-09
categories:
 - ICML
tags:
 - Graph
 - DataAugmentation
 - Supervised
excerpt_separator: <!--more--> 


---

## 摘要

GNN的邻域信息是否足够聚合，以学习邻域较少的节点表示，仍是一个有待解决的问题。为了解决这一问题，我们提出了一种简单有效的数据增强策略——局部增强策略，该策略以中心节点的特征为条件来学习邻居节点的节点特征分布，并通过生成的特征增强GNN的表达能力。可用于任意GNNbackbone。它从学习到的条件分布中采样与每个节点相关的特征向量，作为骨干模型在每次训练迭代时的额外输入。<!--more-->

![title](https://sunjc911.github.io/assets/images/LAGNN/title.png)

## 介绍

GNN采用一种消息传递机制，通过传递和聚合来自局部邻域的信息来生成信息表示。局部信息已经被研究。但有问题，**局部邻域信息是否足以获得有效的节点表示，特别是对于邻居数量有限的节点？**我们认为，局部邻域中有限的邻域数量限制了GNN的表达能力，阻碍了它们的性能，特别是在样本匮乏的情况下，一些节点只有很少的邻域。层数多可以多跳，但是过拟合。因此，我们着**重于丰富低度节点的局部信息，以获得有效的表示。**

**目的：通过数据增强为<u>局部</u>邻居生成更多特征。**现存研究很多为全局角度的topology-level and feature-level。Topology-level扰动度矩阵，改变图结构。 Feature-level给节点属性添加噪声提高泛化。

提出Local Augmentation for Graph Neural Networks(LA-GNNs). **Local Augmentation指的是通过基于局部结构和节点特征的生成模型生成邻域特征。**具体而言，我们提出的框架包括一个预训练步骤，该步骤通过生成模型学习给定一个中心节点特征的连接邻居节点特征的条件分布。如图所示，然后我们利用这个分布来生成与这个中心节点相关的特征向量，作为每次训练迭代的额外输入。此外，我们将生成模型的**预训练和下游GNN训练解耦**，允许我们的数据增强模型以即插即用的方式应用于任何GNN模型。

**实验无自监督。**

## 预备知识

符号定义与GNN概述

## Local Augmentation for Graph Neural Networks (LAGNN)



![overview](https://sunjc911.github.io/assets/images/LAGNN/overview.png)

## 想法

参考WWW-2022-Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning，有些用户交互就很少，利用多跳来聚合？