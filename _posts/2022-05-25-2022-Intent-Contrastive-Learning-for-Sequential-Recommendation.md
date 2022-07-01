---
title: 2022-Intent Contrastive Learning for Sequential Recommendation
description:
date: 2022-05-25
categories:
 - WWW
tags:
 - CL
 - Graph
 - Rec
 - Sequential
 - EM
excerpt_separator: <!--more--> 
---

## 摘要

用户的潜在意图（latent intents）对于序列推荐模型（SR）是个挑战。**作者调查潜在意图的好处并有效利用它们进行序列推荐，提出ICL，一种在序列模型中利用潜在意图变量的一般学习范式。**其核心思想是从未标记的用户行为序列中学习用户意图分布函数，并通过考虑学习到的意图，使用对比自监督学习(SSL)优化SR模型来改进推荐。具体来说，作者**引入一个潜在变量来表示用户意图**，并通过**聚类**来学习潜在变量的分布函数。作者建议通过对比SSL将学习到的意图利用到SR模型中，这将**最大化序列视图与其相应意图之间的一致性**。训练在意图表示学习和广义期望最大化(**EM**)框架内的SR模型优化步骤之间交替进行。将用户意图信息融合到SR中也提高了模型的鲁棒性。<!--more-->

![title](https://sunjc911.github.io/assets/images/ICL/title.png)

## 介绍

顺序推荐的目的是通过对用户过去的行为序列建模，准确地描述用户的动态兴趣。但是**消费行为可能会受到其他潜在因素影响。这促使我们挖掘用户共享的潜在意图，并使用学习到的意图来指导系统提供建议。**

现存的工作需要辅助信息（side information）来构建用户意图模型，但这些信息不能一直可用；或根据物品分类信息，但分类信息不能准确表达用户意图；或根据单个用户的意图训练，但忽视了不用用户潜在意图的相关性。

从用户行为中有效地建模潜在意图提出了**两个挑战**。首先，**没有用户意图的标签数据**，唯一数据就是交互数据，但不用行为可能具有相同的潜在意图。第二，需要潜在意图变量和序列embedding**正交**，不然会产生冗余信息。

**针对挑战，提出ICL，引入变量表示潜在意图，聚类该变量来学习潜在变量的分布函数。对比学习最大化序列和潜在意图的一致性和最大化数据增强的一致性。使用EM算法交替更新分布函数和参数。**

## 预备知识

### **EM算法**

https://mp.weixin.qq.com/s/Rk-F9QZxh-hCDDpdfw6k9g

### 问题定义

常规定义，序列长度多退少补。目的是预测next item。

### Deep SR Models for Next Item Prediction

为了不失去普适性，定义一个序列编码器$$f_{\theta}(\cdot)$$编码一个序列$$\mathrm{S}^{u}$$，输出用户全时刻兴趣表示$$\mathrm{H}^{u}=f_{\theta}\left(\mathrm{S}^{u}\right)$$。目标可定制为寻找最优编码器参数$$\theta$$最大化在全时刻上给定𝑁个序列下一项的对数似然函数：


$$
\theta^{*}=\underset{\theta}{\arg \max } \sum_{u=1}^{N} \sum_{t=2}^{T} \ln P_{\theta}\left(s_{t}^{u}\right)
$$


等价于最小化adapted二元交叉熵(BCE)损失：


$$
\begin{gathered}
\mathcal{L}_{\text {NextItem }}=\sum_{u=1}^{N} \sum_{t=2}^{T} \mathcal{L}_{\text {NextItem }}(u, t), \\
\\
\mathcal{L}_{\text {NextItem }}(u, t)=-\log \left(\sigma\left(\mathbf{h}_{t-1}^{u} \cdot \mathrm{s}_{t}^{u}\right)\right)-\sum_{n e g} \log \left(1-\sigma\left(\mathbf{h}_{t-1}^{u} \cdot \mathrm{s}_{n e g}^{u}\right)\right),
\end{gathered}
$$


负样本加权废算力，根据文献使用抽样softmax技术，在每个序列的每个时刻随机选一个负样本，$$\sigma$$是sigmoid。

### Contrastive SSL in SR


$$
\tilde{S}_{1}^{u}=g_{1}^{u}\left(S^{u}\right), \tilde{S}_{2}^{u}=g_{2}^{u}\left(S^{u}\right) \text {, s.t. } g_{1}^{u}, g_{2}^{u} \sim \mathcal{G}
$$


$$\mathcal{G}$$为数据转换函数集，g1和g2从中间随机选。将$$\tilde{S}_{1}^{u}$$和$$\tilde{S}_{2}^{u}$$通过$$f_{\theta}(\cdot)$$编码成 $$\tilde{\mathbf{H}}_{1}^{u}$$和$$\tilde{\mathbf{H}}_{2}^{u}$$并聚合成序列的向量表示$$\tilde{\mathbf{h}}_{1}^{u}$$和$$\tilde{\mathbf{h}}_{2}^{u}$$。通过InfoNCE优化$$\theta$$：


$$
\begin{gathered}
\mathcal{L}_{\text {SeqCL }}=\mathcal{L}_{\text {SeqCL }}\left(\tilde{\mathbf{h}}_{1}^{u}, \tilde{\mathbf{h}}_{2}^{u}\right)+\mathcal{L}_{\text {SeqCL }}\left(\tilde{\mathbf{h}}_{2}^{u}, \tilde{\mathbf{h}}_{1}^{u}\right) \\
\\
\mathcal{L}_{\text {SeqCL }}\left(\tilde{\mathbf{h}}_{1}^{u}, \tilde{\mathbf{h}}_{2}^{u}\right)=-\log \frac{\exp \left(\operatorname{sim}\left(\tilde{\mathbf{h}}_{1}^{u}, \tilde{\mathbf{h}}_{2}^{u}\right)\right)}{\sum_{n e g} \exp \left(\operatorname{sim}\left(\tilde{\mathbf{h}}_{1}^{u}, \tilde{\mathbf{h}}_{n e g}\right)\right)},
\end{gathered}
$$


### Latent Factor Modeling in SR

算法最主要的就是得到最佳$$\theta$$。假设有K个不同的用户意图（买礼物，买渔具等），那么意图变量可设为$$c=\left\{c\right\}_{i=1}^{K}$$，则每个用户可能和某一项交互的概率为：


$$
P_{\theta}\left(s^{u}\right)=\mathbb{E}_{(c)}\left[P_{\theta}\left(s^{u}, c\right)\right]
$$


用户意图是潜在的向量，因为c我们不能直接观察到。如果没有c，我们没法估计参数$$\theta$$，而没有$$\theta$$，我们就没法推断c。所以需要用EM算法。

## 方法

![overall](https://sunjc911.github.io/assets/images/ICL/overall.png)

E步更新分布函数Q(c)，M步更新$$\theta$$

首先讲怎么推导出目标函数用来将潜在变量c建模成SR模型，如何优化目标函数即$$\theta$$，并且在EM框架下估计c的分布函数。之后描述整体训练策略。最后讲细节分析。

### Intent Contrastive Learning

#### Modeling Latent Intent for SR

基于


$$
\theta^{*}=\underset{\theta}{\arg \max } \sum_{u=1}^{N} \sum_{t=2}^{T} \ln P_{\theta}\left(s_{t}^{u}\right)
\\\\
P_{\theta}\left(s^{u}\right)=\mathbb{E}_{(c)}\left[P_{\theta}\left(s^{u}, c\right)\right]
$$


重写目标函数为：


$$
\theta^{*}=\underset{\theta}{\arg \max } \sum_{u=1}^{N} \sum_{t=1}^{T} \ln \mathbb{E}_{(c)}\left[P_{\theta}\left(s_{t}^{u}, c_{i}\right)\right]
$$


假设c服从Q(c)分布，$$\sum_{c}Q(c_{i})=1$$并且$$Q(c_{i})≥1$$，就有：


$$
\begin{array}{r}
\sum_{u=1}^{N} \sum_{t=1}^{T} \ln \mathbb{E}_{(c)}\left[P_{\theta}\left(s_{t}^{u}, c_{i}\right)\right]=\sum_{u=1}^{N} \sum_{t=1}^{T} \ln \sum_{i=1}^{K} P_{\theta}\left(s_{t}^{u}, c_{i}\right) \\
=\sum_{u=1}^{N} \sum_{t=1}^{T} \ln \sum_{i=1}^{K} Q\left(c_{i}\right) \frac{P_{\theta}\left(s_{t}^{u}, c_{i}\right)}{Q\left(c_{i}\right)} .
\end{array}
$$


通过Jensen不等式转化为：


$$
\begin{gathered}
\geq \sum_{u=1}^{N} \sum_{t=1}^{T} \sum_{i=1}^{K} Q\left(c_{i}\right) \ln \frac{P_{\theta}\left(s_{t}^{u}, c_{i}\right)}{Q\left(c_{i}\right)} \\
\propto \sum_{u=1}^{N} \sum_{t=1}^{T} \sum_{i=1}^{K} Q\left(c_{i}\right) \cdot \ln P_{\theta}\left(s_{t}^{u}, c_{i}\right)
\end{gathered}
$$


当$$Q\left(c_{i}\right)=P_{\theta}\left(c_{i} \mid s_{t}^{u}\right)$$时为=号。为了简单起见，当优化下界的时候只关注最后一个时间步：


$$
\sum_{u=1}^{N} \sum_{i=1}^{K} Q\left(c_{i}\right) \cdot \ln P_{\theta}\left(S^{u}, c_{i}\right)
$$


其中$$Q\left(c_{i}\right)=P_{\theta}\left(c_{i} \mid S^{u}\right)$$。

这样就得到了目标函数的下界。但是这个式子不好直接优化因为Q(c)不知道。所以遵循EM算法去优化。

#### Intent Representation Learning

为了学习Q(c)，需要K个簇心，Q(c)分布函数如下：


$$
Q\left(c_{i}\right)=P_{\theta}\left(c_{i} \mid S^{u}\right)= \begin{cases}1 & \text { if } S^{u} \text { in cluster i} \\
0 & \text { else }\end{cases} \\
$$


在本文中，我们采用“聚合层”表示所有位置步骤上的平均池化操作。我们将其他先进的聚合方法(如基于注意力的方法)**留给未来的工作研究**。

#### Intent Contrastive SSL with FNM

Q(c)已经知道如何估计。为了最大化目标函数，我们还需要定义$$P_{\theta}\left(S^{u}, c_{i}\right)$$。假设先验意图遵循均匀分布。在给定c的条件下,$$\S^{u}$$的条件分布是l2归一化的各向同性高斯分布(球形分布，各个方向方差都一样的多维高斯分布)，则$$P_{\theta}\left(S^{u}, c_{i}\right)$$可重写为：


$$
P_{\theta}\left(S^{u}, c_{i}\right)=P_{\theta}\left(c_{i}\right) P_{\theta}\left(S^{u} \mid c_{i}\right)=\frac{1}{K} \cdot P_{\theta}\left(S^{u} \mid c_{i}\right) \\
\propto \frac{1}{K} \cdot \frac{\exp \left(-\left(\mathbf{h}^{u}-\mathbf{c}_{i}\right)^{2}\right)}{\sum_{j=1}^{K} \exp \left(-\left(\mathbf{h}_{i}^{u}-\mathbf{c}_{j}\right)^{2}\right)}\\
\propto \frac{1}{K} \cdot \frac{\exp \left(\mathbf{h}^{u} \cdot \mathbf{c}_{i}\right)}{\sum_{j=1}^{K} \exp \left(\mathbf{h}^{u} \cdot \mathbf{c}_{j}\right)}
$$


所以目标函数等价于最小化下面的损失函数：


$$
-\sum_{v=1}^{N} \log \frac{\exp \left(\operatorname{sim}\left(\mathbf{h}^{u}, \mathbf{c}_{i}\right)\right)}{\sum_{j=1}^{K} \exp \left(\operatorname{sim}\left(\mathbf{h}^{u}, \mathbf{c}_{j}\right)\right)}
$$


上式对比单个序列和潜在意图的一致性。之前SeqCL loss的数据增强是必须的，而ICL的数据增强是可选的，因为对比的是序列和潜在意图。本文加入数据扩充：


$$
\mathcal{L}_{\mathrm{ICL}}=\mathcal{L}_{\mathrm{ICL}}\left(\tilde{\mathbf{h}}_{1}^{u}, \mathbf{c}_{u}\right)+\mathcal{L}_{\mathrm{ICL}}\left(\tilde{\mathbf{h}}_{2}^{u}, \mathbf{c}_{u}\right) ,\\
\mathcal{L}_{\mathrm{ICL}}\left(\tilde{\mathbf{h}}_{1}^{u}, \mathbf{c}_{u}\right)=-\log \frac{\exp \left(\operatorname{sim}\left(\tilde{\mathbf{h}}_{1}^{u}, \mathbf{c}_{u}\right)\right)}{\sum_{n e g} \exp \left(\operatorname{sim}\left(\tilde{\mathbf{h}}_{1}^{u}, \mathbf{c}_{n e g}\right)\right)}
$$


$$c_{neg}$$是batch中所有的意图。直接优化上式可能会造成假阴性，因为在一个batch中用户可能有相同的意图。所以提出一种负样本抽样技术FNM：


$$
\mathcal{L}_{\mathrm{ICL}}\left(\tilde{\mathbf{h}}_{1}^{u}, \mathbf{c}_{u}\right)=-\log \frac{\exp \left(\operatorname{sim}\left(\tilde{\mathbf{h}}_{1}^{u}, \mathbf{c}_{u}\right)\right)}{\sum_{v=1}^{N} \mathbb{1}_{v \notin \mathcal{F}} \exp \left(\operatorname{sim}\left(\tilde{\mathbf{h}}_{1}, \mathbf{c}_{v}\right)\right)}
$$


其中F是一组和u有相同意图的用户。

### Multi-Task Learning


$$
\mathcal{L}=\mathcal{L}_{NextItem} + 𝜆·\mathcal{L}_{ICL}+𝛽·\mathcal{L}_{SeqCL}
$$


基于Transformer编码器建模成ICLRec。

## EXPERIMENTS

数据只是用‘5-core’，意思是用户或者项目至少有5个交互记录才会被选中。

结果：

![exp](https://sunjc911.github.io/assets/images/ICL/exp.png)

消融实验：

![ablation](https://sunjc911.github.io/assets/images/ICL/ablation.png)

## Code

### 处理数据

```
def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix

user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

args.item_size = max_item + 2
args.mask_id = max_item + 1

```

torch.utils.data.Dataset

https://blog.csdn.net/weixin_44211968/article/details/123744513

```
class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train", similarity_model_type="offline"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        # currently apply one transform, will extend to multiples
        self.augmentations = {
            "crop": Crop(tao=args.tao),
            "mask": Mask(gamma=args.gamma),
            "reorder": Reorder(beta=args.beta),
            "random": Random(tao=args.tao, gamma=args.gamma, beta=args.beta),
        }
        if self.args.augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views
        
# main调用上面的class
cluster_dataset = RecWithContrastiveLearningDataset(
    args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train"
)
cluster_sampler = SequentialSampler(cluster_dataset)
cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

train_dataset = RecWithContrastiveLearningDataset(
    args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train"
)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
```

```
cur_rec_tensors = (
    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
    torch.tensor(copied_input_ids, dtype=torch.long),
    torch.tensor(target_pos, dtype=torch.long),
    torch.tensor(target_neg, dtype=torch.long),
    torch.tensor(answer, dtype=torch.long),
) # tuple:5
```

![subsequent_mask](https://sunjc911.github.io/assets/images/ICL/subsequent_mask.png)
