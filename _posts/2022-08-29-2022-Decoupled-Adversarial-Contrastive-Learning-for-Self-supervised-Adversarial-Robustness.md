---
title: 2022-Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness
description:
date: 2022-08-29
categories:
 - ECCV
tags:
 - AdversarialTraining
 - CL
 - SSL
 - Pre-train
 - Robustness
excerpt_separator: <!--more--> 

---

## 摘要

对于无标签数据，对抗学习大多用SSL和AT结合，这样要优化一个复杂目标，要做到准确性和鲁棒性的平衡。本文采用**分而治之**思想，第一步预训练SSL生成target，第二部使用target进行AT。<!--more-->

![title](https://sunjc911.github.io/assets/images/DeACL/title.png)

## 内容

从求解耦合问题的最优组合策略到求解子问题的子解。

本文研究self-supervised robust representation learning.

分而治之使得可以弹性配置，且训练时间更少。

![overall](https://sunjc911.github.io/assets/images/DeACL/overall.png)

第一步SSL使用InfoNCE

![InfoNCE](https://sunjc911.github.io/assets/images/DeACL/InfoNCE.png)

第二部使用SSL获得的target来进行AT

![stage2](https://sunjc911.github.io/assets/images/DeACL/stage2.png)

阻止对比学习loss collapse方法：一种广泛使用的方法来缓解这一现象是引入对比分量，即同时最小化负样本之间的余弦相似度

## 实验

![exp](https://sunjc911.github.io/assets/images/DeACL/exp.png)

RA(robust)提高。

## 想法

将分而治之用到Rec的CL？