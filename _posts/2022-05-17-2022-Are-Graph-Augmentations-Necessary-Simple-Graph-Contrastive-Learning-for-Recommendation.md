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
excerpt_separator: <!--more--> 
---

## 摘要

作者指出，在CL-based推荐系统中，**真正影响性能的是CL的损失，而不是原始图的数据增强**。基于此发现，作者提出SimGCL，未使用原始图数据增强，**在图embedding空间加入均匀分布噪声**(uniform noises)来创造对比视图。<!--more-->

![title](https://sunjc911.github.io/assets/images/SimGCL/title.png)

## 介绍

对比学习(Contrastive Learning, CL)是解决推荐系统数据集稀疏问题的好方法，因为它能从未标记数据中提取普遍的特征，并用自监督(Self-supervised Learning, SSL)方法进行正则化(regularize)数据的表示(representation)。CL一般采用数据增强方法构造原始图的对比视图，通过encoder最大化不同视图的一致性表示。下图以edge dropout为例。

![ED](https://sunjc911.github.io/assets/images/SimGCL/ED.png)

但有文献指出就算是edge dropout rate 0.9的对比视图也能有效果，而rate 0.9的视图已经损失了大量的信息并且拥有一个高度倾斜的结构，所以这是违反人们常理的。作者借此提出：***当CL和Recommendation结合时，我们真的需要图数据增强吗？***

实验结果表明，**CL的loss(如InfoNCE)才是性能的关键**。图数据增强也不是没有用处，它能帮助模型学习到图不受扰动因素影响的表示。然而图数据增强非常耗人工和时间。作者提出第二个问题：***是否有高效的数据增强方法？***

## GCL for recommendation

以SGL[1]为例子, 其使用节点和边的dropout进行数据增强。Loss为：

$$\mathcal{L}_{\text {joint }}=\mathcal{L}_{r e c}+\lambda \mathcal{L}_{c l}$$

包括$$\mathcal{L}_{\text {joint }}$$和$$\mathcal{L}_{c l}$$。$$\mathcal{L}_{c l}$$为InfoNCE：

$$\mathcal{L}_{c l}=\sum_{i \in \mathcal{B}}-\log \frac{\exp \left(\mathrm{z}_{i}^{\prime \top} \mathrm{z}_{i}^{\prime \prime} / \tau\right)}{\sum_{j \in \mathcal{B}} \exp \left(\mathrm{z}_{i}^{\prime \top} \mathrm{z}_{j}^{\prime \prime} / \tau\right)}$$

$$ \mathcal{B}$$为a sampled batch。**CL鼓励z'和z''的一致性。**

SGL以LightGCN[2]为backbone，其消息传递可写成矩阵形式：

$$\mathrm{E}=\frac{1}{1+L}\left(\mathrm{E}^{(0)}+\tilde{\mathrm{A}} \mathrm{E}^{(0)}+\ldots+\tilde{\mathrm{A}}^{L} \mathrm{E}^{(0)}\right)$$

其中：

$$\mathrm{E}^{(0)} \in \mathbb{R}^{|N| \times d}$$

是随机初始化的节点embedding；

$$\tilde{\mathrm{A}} \in \mathbb{R}^{|N| \times|N|}$$

是正则化无向邻接矩阵；

$$\mathrm{z}_{i}^{\prime}=\frac{\mathrm{e}_{i}^{\prime}}{\left\|\mathrm{e}_{i}^{\prime}\right\|_{2}}$$

其中$$\mathbf{e}_{i}^{\prime}$$是$$\mathrm{E}$$中$$\mathbf{e}_{i}$$的数据增强版本。

## Necessity of Graph Augmentation

作者的实验：对比SGL的不同变体性能

![SGLvariants](https://sunjc911.github.io/assets/images/SimGCL/SGLvariants.png)

ND为node dropout，ED为edge dropout，RW为random walk。WA为无数据增强，即$$\mathrm{Z}_{i}^{\prime}=\mathrm{Z}_{i}^{\prime \prime}=\mathrm{Z}_{i}$$,所以WA的$$\mathcal{L}_{c l}$$变为：

$$\mathcal{L}_{c l}=\sum_{i \in \mathcal{B}}-\log \frac{\exp (1 / \tau)}{\sum_{j \in \mathcal{B}} \exp \left(\mathrm{z}_{i}^{\top} \mathrm{z}_{j} / \tau\right)}$$

可以看到**ED最好，但是比WA只好一点**，说明**图增广的轻微扰动有用**。

## InfoNCE Loss Influences More

根据文献[3],CL会使得正样本在距离上更近（**alignment**），使特征在超球面上分布更均匀（**uniformity**）。

作者在两个数据集上实验（只有uniformity可视化，**未展示alignment可视化图**）。

![uniformity](https://sunjc911.github.io/assets/images/SimGCL/uniformity.png)

可以看到**CL使得特征超球分布更加均匀，缓和特征聚集程度**。

作者解释LightGCN的高度聚集分布原因。1.LightGCN的消息传递机制使得节点embedding变得相似（**过平滑**）；2.数据流行度偏差（**长尾问题**）。

作者将InfoNCE重写为下式：

$$\mathcal{L}_{c l}=\sum_{i \in \mathcal{B}}-1 / \tau+\log \left(\exp (1 / \tau)+\sum_{j \in \mathcal{B} /\{i\}} \exp \left(\mathbf{z}_{i}^{\top} \mathbf{z}_{j} / \tau\right)\right)$$

可以看出**cl loss实际是最小化$$\mathbf{e}_{i}$$和$$\mathbf{e}_{j}$$的余弦相似度**，从而使它们在空间上远离，计算在rec loss的影响下，在超球上也会显得均匀（uniformity）。

这就说明**分布的均匀性是SGL中推荐性能的决定性影响的潜在因素，而不是基于drop out的图增广**。优化CL loss可以看作是隐式的去偏方法，因为均匀分布可提高泛化能力。但只追求cl loss最小化也不好。大白话：**uniformi可视化图要分布均匀，但是也要有点聚集**。

## SIMGCL: SIMPLE GRAPH CONTRASTIVELEARNING FOR RECOMMENDATION

受到上述分析的启发（**在一定范围内调整学习到的表征（learned representation）的uniformity**），作者提出SimGCL。

给图结构添加均匀分布非常耗时和人力，所以作者将注意力转到embedding空间。受到[4]的启发，作者**直接在表示中加入随机噪声构造对比视图**：

$$\mathrm{e}_{i}^{\prime}=\mathrm{e}_{i}+\Delta_{i}^{\prime}, \quad \mathbf{e}_{i}^{\prime \prime}=\mathrm{e}_{i}+\Delta_{i}^{\prime \prime}$$

其中噪声向量$$\Delta_{i}^{\prime}$$和$$\Delta_{i}^{\prime \prime}$$都遵循$$\|\Delta\|_{2}=\epsilon$$和$$\Delta=\bar{\Delta} \odot \operatorname{sign}\left(\mathbf{e}_{i}\right)$$, $$\bar{\Delta} \in \mathbb{R}^{d} \sim U(0,1)$$。第一个约束$$\Delta$$的大小，$$\Delta$$是半径为$$\epsilon$$的超球上的点。第二个约束使得$$\mathrm{e}_{i}$$,$$\Delta_{i}^{\prime}$$和$$\Delta_{i}^{\prime \prime}$$在一个hyperoctant上，这样加入噪声不会引起较大偏差使得正样本变少。可视化：

![noise](https://sunjc911.github.io/assets/images/SimGCL/noise.png)

由于旋转足够小，**增广表示保留了原始表示的大部分信息，同时也保留了一些方差**。注意，对于每个节点表示，添加的随机噪声是不同的。

矩阵形式：

$$\begin{array}{r}
\mathrm{E}^{\prime}=\frac{1}{L}\left(\left(\tilde{\mathrm{A}} \mathrm{E}^{(0)}+\Delta^{(1)}\right)+\left(\tilde{\mathrm{A}}\left(\tilde{\mathrm{A}} \mathrm{E}^{(0)}+\Delta^{(1)}\right)+\Delta^{(2)}\right)\right)+\ldots \\
\left.+\left(\tilde{\mathrm{A}}^{L} \mathrm{E}^{(0)}+\tilde{\mathrm{A}}^{L-1} \Delta^{(1)}+\ldots+\tilde{\mathrm{A}} \Delta^{(L-1)}+\Delta^{(L)}\right)\right)
\end{array}$$

注意，计算最终表示时没有加入初始embedding$$\mathrm{E}^{(0)}$$,因为作者说这样效果更好。如果没有CL任务，在LightGCN中这样做会导致性能下降。

**最终Loss：BPR + InfoNCE**

## EXPERIMENTAL RESULTS

![exp](https://sunjc911.github.io/assets/images/SimGCL/exp.png)

此外作者还对比了不同噪声方法：

![noisecomparison](https://sunjc911.github.io/assets/images/SimGCL/noisecomparison.png)

## 参考文献

[1] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. **Self-supervised graph learning for recommendation**. In SIGIR.

[2] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** In SIGIR.

[3] TongzhouWang and Phillip Isola. 2020. **Understanding contrastive representation learning through alignment and uniformity on the hypersphere.** In ICML.

[4] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014. **Explaining and harnessing adversarial examples (2014)**. arXiv preprint arXiv:1412.6572 (2014).

## Code

### 处理数据

pass

### 构造稀疏二部邻接矩阵

```
tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
```

**sp.csr_matrix**：压缩稀疏矩阵

tmp_adj矩阵大概：

![tmp_adj](https://sunjc911.github.io/assets/images/SimGCL/tmp_adj.png)

```
adj_mat = tmp_adj + tmp_adj.T
```

adj_mat 矩阵大概：

![adj_mat](https://sunjc911.github.io/assets/images/SimGCL/adj_mat.png)

### 归一化矩阵

![normalizematrix](https://sunjc911.github.io/assets/images/SimGCL/normalizematrix.png)

```
def normalize_graph_mat(adj_mat):
    shape = adj_mat.get_shape()
    rowsum = np.array(adj_mat.sum(1))
    if shape[0] == shape[1]:
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
    else:
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
    return norm_adj_mat
```

### 交互矩阵

```
interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
```

### 初始化模型参数

```
def _init_model(self):
    initializer = nn.init.xavier_uniform_
    embedding_dict = nn.ParameterDict({
        'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
        'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
    })
    return embedding_dict
```

### 归一矩阵格式从csr转为coo并存入cuda

csr数据.tocoo()

csr是对coo的行压缩矩阵，coo不能矩阵运算。**可是为什么要转？**

### train

模型放入cuda

```
model = self.model.cuda()
```

初始化优化器

```
optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
```

epoch循环

```
for epoch in range(self.maxEpoch):
    for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
        user_idx, pos_idx, neg_idx = batch
        model.train()
        rec_user_emb, rec_item_emb = model()
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx])
        batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
        # Backward and optimize
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if n % 100==0:
            print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
    model.eval()
    with torch.no_grad():
        self.user_emb, self.item_emb = self.model()
    self.fast_evaluation(epoch)
self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
```

batch方法：

```
def next_batch_pairwise(data, batch_size):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            neg_item = choice(item_list)
            while neg_item in data.training_set_u[user]:
                neg_item = choice(item_list)
            j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx
```

### Loss

rec loss: BPR

```
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)
```

cl loss: InfoNCE，对比视图为增加噪声后的embedding

```
def cal_cl_loss(self, idx):
    u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
    i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
    user_view_1, item_view_1 = self.forward(perturbed=True)
    user_view_2, item_view_2 = self.forward(perturbed=True)
    user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.15)
    item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.15)
    return user_cl_loss + item_cl_loss
    
def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)
```

perturbed=True后噪声加入：

```
if perturbed:
    random_noise = torch.rand_like(ego_embeddings).cuda()
    ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
```

$$\mathrm{e}_{i}^{\prime}=\mathrm{e}_{i}+\Delta_{i}^{\prime}, \quad \mathbf{e}_{i}^{\prime \prime}=\mathrm{e}_{i}+\Delta_{i}^{\prime \prime}$$

其中噪声向量$$\Delta_{i}^{\prime}$$和$$\Delta_{i}^{\prime \prime}$$都遵循$$\|\Delta\|_{2}=\epsilon$$和$$\Delta=\bar{\Delta} \odot \operatorname{sign}\left(\mathbf{e}_{i}\right)$$, $$\bar{\Delta} \in \mathbb{R}^{d} \sim U(0,1)$$。

loss正则化参数

```
def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg
```

反向传播与优化（每次计算完总loss后都要）

```
optimizer.zero_grad()
batch_loss.backward()
optimizer.step()
```

每个epch后加model.eval()

model.eval() 作用等同于 self.train(False)。简而言之，就是评估模式。而非训练模式。在评估模式下，`batchNorm`层，`dropout`层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。在对模型进行评估时，应该配合使用`with torch.no_grad()` 与 `model.eval()`：

```
    loop:
        model.train()    # 切换至训练模式
        train……
        model.eval()
        with torch.no_grad():
            Evaluation
    end loop
```

eval：

precision 

```
prec = sum([hits[user] for user in hits])
return prec / (len(hits) * N)
```

recall

```
def recall(hits, origin):
    recall_list = [hits[user]/len(origin[user]) for user in hits]
    recall = sum(recall_list) / len(recall_list)
    return recall
```

F1

```
def F1(prec, recall):
    if (prec + recall) != 0:
        return 2 * prec * recall / (prec + recall)
    else:
        return 0
```

NDCG

```
def NDCG(origin,res,N):
    sum_NDCG = 0
    for user in res:
        DCG = 0
        IDCG = 0
        #1 = related, 0 = unrelated
        for n, item in enumerate(res[user]):
            if item[0] in origin[user]:
                DCG+= 1.0/math.log(n+2)
        for n, item in enumerate(list(origin[user].keys())[:N]):
            IDCG+=1.0/math.log(n+2)
        sum_NDCG += DCG / IDCG
    return sum_NDCG / len(res)
```

### 其他

exec() : 执行括号内的语句；
