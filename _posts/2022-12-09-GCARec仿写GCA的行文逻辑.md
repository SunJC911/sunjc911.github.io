---
title: GCARec仿写GCA的行文逻辑
description: GCA，GCARec
date: 2022-12-09
categories:
 - Writing
tags:
 - Writing
excerpt_separator: <!--more--> 



---

## 摘要

GCN成为rec中的主流之一，能够捕捉高阶关系。但是受到监督信号和噪声交互。加入GCL。描述下GCL。但是现存方法忽略数据增强时边和节点的影响差异，将会降低性能。同时采用人工数据增强限制了泛化能力。我们认为数据增强应该是可学习的。提出GCARec。具体说。实验性能（SOTA）。

## 1 INTRODUCTION

rec解决信息过载。GCN成为rec主流方法。受到两个问题，稀疏监督信号，噪声交互。

为了解决问题，引入GCL，介绍GCL增强一致性。但是数据增强方案很好被探索。目前主要两种，随即增强，人工设计增强。这些增强有局限：首先，忽略增强时点和边的差异，降低表示质量。其次，根据领域知识进行人工增强不通用。因此，需要提出可学习的数据增强方案。可以保留重要的去掉不重要的，且使用更少资源。

to this end，提出GCARec。具体说...，Finally, we leverage the contrastive learning task (i.e., node
self-discrimination) as the auxiliary task to the recommendation task and jointly train them using the multi-task training strategy.

In summary, the contributions of our work are as follows:1.2.3。介绍每章干啥。

## 2 Preliminaries

符号定义，介绍GCN-based rec

## 3 Methodology

### 3.1CL框架

### 3.2自适应增强

#### 概率计算

#### 视图生成

### 3.3 CL

接着3.1，描述loss

### 3.4 多任务训练

## 4 EXP

为了评估我们提出的GCARec的有效性，我们通过回答以下问题进行了广泛的实验rq1 rq2 rq3

### 4.1 实验设置

#### datasets

画表及介绍

#### 验证指标

#### 对比方法

#### 参数设置

### 4.2 性能对比（rq1）

与传统CF比，GCF好；加入cl的SGL和GCARec比GCF好，说明CL好；总体而言，GCARec比SGL好，说明提出的点有效。

### 4.3 Further Study of GCARec

#### 数据稀疏性影响（rq2）

#### 噪声交互的鲁棒性（rq2）

#### 超参sensitivity（rq3）

## 5 Related work

### Graph-based Recommendation

### Self-supervised Learning in Recommender Systems

## 6 conclusion
