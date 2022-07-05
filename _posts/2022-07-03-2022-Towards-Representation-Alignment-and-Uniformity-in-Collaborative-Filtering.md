---
title: 2022-Towards Representation Alignment and Uniformity in Collaborative Filtering
description:
date: 2022-07-03
categories:
 - KDD
tags:
 - CF
 - Rec
excerpt_separator: <!--more--> 

---

## 摘要

CF在Rec很有用，BPR loss是主流。现存papers主要关注如何学习到更好的表示，即设计更好的encoder，很少去关注为什么CF好。作者根据CL的alignment和uniformity（AU）去对BPR的AU进行分析。并对主流CF方法进行AU分析。发现它们的AU各有所长。根据此，作者提出DirectAU Loss去直接优化CF的AU。实验效果SOTA。<!--more-->

![title](https://sunjc911.github.io/assets/images/DirectAU/title.png)

## 介绍

现存paper大多研究更强大的CF encoder，但又paper指出这样只能提升微乎其微的效果。事实上，设计目标函数即loss会有更强大的提升。

需要理解CF为何有效才能设计出好的loss。直观上，正对样本在空间上更接近，根据文献[1]研究CL提出的AU，作者想研究CF的AU。作者发现BPR能快速对齐（alignment）,之后主要是提升uniformity。其他方法也是去提高A或者U。基于此发现，提出DirectAU用于直接优化CF的AU。

## 预备知识

### Collaborative Filter（CF）

使用一个encoder$$f(·)$$将user和item映射到低维空间表示。比如矩阵分解（MF）,或者LightGCN等。最后dot product预测得分。大多数根据BPR loss训练模型。


$$
\mathcal{L}_{B P R}=\frac{1}{|\mathcal{R}|} \sum_{(u, i) \in \mathcal{R}}-\log \left[\operatorname{sigmoid}\left(s(u, i)-s\left(u, i^{-}\right)\right)\right]
$$


### Alignment and Uniformity（AU）

根据文献[1]\[2]，表示的质量取决于AU。给定数据分布$$p_{data}(·)$$和正样本对分布$$p_{pos}(·, ·)$$，alignment定义为正对的标准化embeddings之间的期望距离，~表示$$l_{2}$$正则化表示：


$$
l_{\text {align }} \triangleq \underset{\left(x, x^{+}\right) \sim p_{\text {pos }}}{\mathbb{E}}\left\|\tilde{f}(x)-\tilde{f}\left(x^{+}\right)\right\|^{2}
$$


uniformity定义为成对高斯函数的均值的对数：


$$
l_{\text {align }} \triangleq \underset{\left(x, x^{+}\right) \sim p_{\text {pos }}}{\mathbb{E}}\left\|\tilde{f}(x)-\tilde{f}\left(x^{+}\right)\right\|^{2}
$$


这两个指标与表征学习的目标非常一致：**正实例应该彼此靠近，而随机实例应该分散在超球面上**。

## ALIGNMENT AND UNIFORMITY IN COLLABORATIVE FILTERING

在本节中，我们首先从理论上证明了BPR损失有利于超球上的表示对齐和均匀性。然后，我们通过实验观察这两种特性在不同CF方法的训练过程中是如何演变的。

### Theoretical Analyses

具体看论文，没细看。

### Empirical Observations

下图为各个CF方法的AU可视化，说明CF方法和AU雀氏有关系（随着epoch在优化A、U）。

![figure2](https://sunjc911.github.io/assets/images/DirectAU/figure2.png)

## DIRECTLY OPTIMIZING ALIGNMENT AND UNIFORMITY (DIRECTAU)

上述分析表明AU对于CF也有着重要性，作者提出**直接优化AU来提升CF性能**。

输入为有交互的user-item对（正对），编码后进行L2归一化到超球面上。

![DirectAU](https://sunjc911.github.io/assets/images/DirectAU/DirectAU.png)

CF中AU的loss为：


$$
l_{\text {align }}=\underset{(u, i) \sim p_{\text {pos }}}{\mathbb{E}} \| \tilde{f}(u)-\tilde{f}(i) \|^{2},\\\\
l_{\text {uniform }}=\log \underset{u, u^{\prime} \sim p_{\text {user }}}{\mathbb{E}} e^{-2 \| \tilde{f}(u)-\tilde{f}\left(u^{\prime}\right) \|^{2}} / 2+\log \underset{i, i^{\prime} \sim p_{\text {item }}}{\mathbb{E}} e^{-2\left\|\tilde{f}(i)-\tilde{f}\left(i^{\prime}\right)\right\|^{2}} / 2 ,\\\\
\mathcal{L}_{\text {DirectAU }}=l_{\text {align }}+\gamma l_{\text {uniform }}
$$


DirectAU只用正样本对，批处理技术用于减少bias。

## EXPERIMENTS

数据详情

![data](https://sunjc911.github.io/assets/images/DirectAU/data.png)

实验结果

![exp](https://sunjc911.github.io/assets/images/DirectAU/exp.png)

AU结果

![expAU](https://sunjc911.github.io/assets/images/DirectAU/expAU.png)

用其他CF方法替换掉DirectAU的MF

![differentCFinDirectAU](https://sunjc911.github.io/assets/images/DirectAU/differentCFinDirectAU.png)

## 想法

1.减少bias能力如何？

2.对seq的潜在兴趣变量能否用AU分析？

## 核心代码

```
@staticmethod
def alignment(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()

@staticmethod
def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

def calculate_loss(self, user, item):
    user_e, item_e = self.encoder(user, item)  # [bsz, dim]
    align = self.alignment(user_e, item_e)
    uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
    loss = align + self.gamma * uniform
    return loss
```

## 参考文献

[1] TongzhouWang and Phillip Isola. 2020. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning. PMLR, 9929–9939.

[2] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. arXiv preprint arXiv:2104.08821 (2021).

