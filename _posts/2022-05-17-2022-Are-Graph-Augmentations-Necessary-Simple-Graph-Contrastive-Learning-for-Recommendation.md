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

作者指出，在CL-based推荐系统中，真正影响性能的是CL的损失，而不是原始图的数据增强。

基于此发现，作者提出SimGCL，未使用原始图数据增强，在图embedding空间加入均匀分布噪声(uniform noises)来创造对比视图。

