---
title: 2022-XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation
description:
date: 2022-09-14
categories:
 - TKDE
tags:
 - CL
 - Graph
 - Rec
excerpt_separator: <!--more--> 

---

## 摘要

SimGCL的进阶版，使用cross-layer contrast。看loss应该是受到NCL的启发。<!--more-->

![title](https://sunjc911.github.io/assets/images/XSimGCL/title.png)

SimGCL还不够好。需要做额外的向前/向后传播。实际上是CL Rec的普遍问题。导致计算复杂度是传统模型的三倍。

XSimGCL基于SimGCL，简化传播过程，从而降低复杂度。XSimGCL的推荐任务和对比任务在一个mini-batch中共享正向/向后传播，而不是拥有单独的管道。

SimGCL和XSimGCL使用相同的输入:初始嵌入和邻接矩阵。区别：SimGCL使用原始输入进行两次生成对比学习embedding，**XSimGCL使用cross-layer contrast**

![arc](https://sunjc911.github.io/assets/images/XSimGCL/arc.png)

## XSimGCL

提出想法：如果我们对比不同层的embedding会怎么样？

正如文献所建议的，使用时有一个甜罐，其中相关视图之间的互信息既不太高也不太低

它们共享一些共同的信息，但在聚集的邻居和添加的噪声方面有所不同，这符合甜罐理论。另外，考虑到添加噪声的大小足够小，我们可以直接使用摄动（perturbed）表示进行推荐任务。噪声类似于广泛使用的dropout技巧，只应用于训练。在测试阶段，模型切换到无噪声的普通模式。

只有一次向前/向后传播的过程

![loss](https://sunjc911.github.io/assets/images/XSimGCL/loss.png)

不同层之间对比的结果

![layer](https://sunjc911.github.io/assets/images/XSimGCL/layer.png)

![exp](https://sunjc911.github.io/assets/images/XSimGCL/exp.png)

![other](https://sunjc911.github.io/assets/images/XSimGCL/other.png)

文中结论：推广长尾项目的能力似乎与表征的一致性呈正相关

**关于LightGCN**：该算法采用SGC[1]算法去除了常规GCN算法中的变换矩阵和非线性激活函数等冗余操作(用GATConv对LightGCN后的embedding去生成子图是不是就不好？)

## 参考文献

[1] F. Wu, A. Souza, T. Zhang, C. Fifty, T. Yu, and K. Weinberger, “Simplifying graph convolutional networks,” in ICML, 2019, pp. 6861–6871.