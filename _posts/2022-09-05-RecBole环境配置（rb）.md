---
title: RecBole环境配置 + GSC + LightGCN + NCL（rb
description: Titan X rb 新模型为bpr+reg+OT（暂定，实验中）
date: 2022-09-05
categories:
 - RecBole
tags:
 - RecBole
excerpt_separator: <!--more--> 

---

目前进度：将GSC粗暴加入NCL，可以运行产生loss，循环报错,**loss is nan实验证明是参数初始化导致**

问题：trainer.py的nodes_batch: 随机选择节点作为中心节点（会重复），**怎么自动获取用户数和物品数？**；loss数量与参数正则化的loss怎么搞？；GATConv和LightGCN层数问题？；how参数初始化？etc.

调参：k1：每个子图BFS搜索到的节点数量（包括自己），设为k=15待调参；Model的tau手动设为0.5，待调参（可以先不动）etc.

进阶问题：每个batch多少个节点？每个中心节点找几个节点？etc.

进阶模块：用户和物品分开（模糊思路）？BFS换一下？etc.

<!--more--> 

## 环境配置

Python：3.8

Pytorch：1.12.1

CUDA：11.3

PyG：2.1.0

RecBole：1.0.1（conda安装的为0.2.0，从pycharm环境里升级为1.0.1）**可用**

faiss-gpu：1.7.2

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

conda install -c aibox recbole
conda install faiss-gpu -c pytorch
```

实测ECCV-2022-Generative Subgraph Contrast for Self-Supervised Graph Representation Learning（**GSC**）和WWW-2022-Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning（**NCL**）都可以跑。

## 要解决的问题

### 怎么生成GSC中的node_neighbor_cen？（可运行，有小问题）

![node_neighbor_cen](https://sunjc911.github.io/assets/images/rb/node_neighbor_cen.png)

```
node_neighbor_cen = func.sub_sam(nodes_batch, adj_lists, k1)
```

trainer.py的nodes_batch: 随机选择节点作为中心节点（会重复），**怎么自动获取用户数和物品数？**

adj_lists: **元组**，每个节点与谁交互

![adj_lists](https://sunjc911.github.io/assets/images/rb/adj_lists.png)

k1：每个子图BFS搜索到的节点数量（包括自己），设为**k=15待调参**

目前准备在NCL trainer 47行加入node_neighbor_cen（目前可以运行该模块）

1.需要知道nodes，手动查ml-1m user6040，item3629，共9669个节点

ncl.py中def make_adj(self): 用来制作adj涉及问题：ncl.py 54行 coo_matrix怎么转元组（每个值是个list）？方法：制作一个函数

```
def make_adj(self, A):
    a = A.row  # ndarray(1349348)
    b = A.col  # ndarray(1349348)
    c = torch.LongTensor([a, b])
    adj_lists = defaultdict(set)
    for i in range(c.size(1)):
        # 记录每个点和其交互的点（包括自身）
        adj_lists[c[0][i].item()].add(c[1][i].item())
    f = adj_lists[3]
    return adj_lists
```

出现问题：adj_lists {defaultset:9639} 少了30个节点。

原因：缺少自环，需添加自环from torch_geometric.utils import add_self_loops。9669不少啦~~哈啊哈哈我尼玛就是个天才，科研能手卧槽哈哈哈哈哈~~

进阶问题：每个batch多少个节点？每个中心节点找几个节点？

进阶模块：用户和物品分开（模糊思路）？BFS换一下？

### gen生成器怎么放入recbole框架下？~~（待解决）~~

尝试1：把gen模型放入class NEW（GeneralRecommender）中

先了解RecBole的GeneralRecommender

![general_recommender](https://sunjc911.github.io/assets/images/rb/general_recommender.png)

```
class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config['device']
```

了解其父类AbstractRecommender

```
class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """

    def __init__(self):
        self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, 'other_parameter_name'):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'
```

main.py 16行改为NEW（无报错，但是得仔细研究下）

main.py 35行改为NEW（还未改完）

Model的tau手动设为0.5，待解决

~~需要去self_loop的边集合，~~之前的adj不包括自环！

进军_train_epoch!

报错：RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper__index_select)

原因：应该是z_g = self.activation(self.conv(lightgcn_all_embeddings, self.adj2))有东西没进显卡。

应该是需要加上.to(self.device)。目前排查到adj2，就是它

目前为止能正常运行

**开始放loss**

trainer.py 129 行加载loss

trainer.py 143行计算loss

ncl.py def calculate_loss 调用forward

往def calculate_loss里放OT loss

运行报错：RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

猜想原因：特征维度1024变为了64导致（不对）

实际原因：GSC进loss的adj为（nodes，nodes），我搞成了(2，xxxx）边矩阵

方法：用A制作Tensor（9669，9669）的adj

**.t()是他妈转置不是todevice的缩写傻逼**

```
self.A_Tensor = torch.from_numpy(sp.coo_matrix(self.A).toarray()).to(self.device)
```

## Loss is NAN

```
File "/home/sunjiecheng/workspace/rb/NCL-master/trainer.py", line 154, in _train_epoch
    self._check_nan(loss)
  File "/home/sunjiecheng/.conda/envs/rb/lib/python3.8/site-packages/recbole/trainer/trainer.py", line 264, in _check_nan
    raise ValueError('Training loss is nan')
ValueError: Training loss is nan
```

查看self._check_nan(loss)

```
def _check_nan(self, loss):
    if torch.isnan(loss):
        raise ValueError('Training loss is nan')
```

可能原因：初始化问题(ot的参数初始化导致ncl的loss也是nan) etc.

方法试验：删除ncl的loss部分，初始化ot的参数

开始实验：先注释ncl的部分

发现：_calculate_loss里，**OT的参数应该也要放进去，放哪些？**OT没有放参数正则化loss，也没有参数初始化

```
# 用来进行参数正则化，OT的参数应该也要放进去
reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)  # Emb
```

self初始化中apply用于参数初始化

```
self.apply(xavier_uniform_initialization)
# using xavier_normal_ in PyTorch to initialize the parameters in nn.Embedding and nn.Linear layers. For bias in nn.Linear layers, using constant 0 to initialize.
```

目前：losses: 0 bpr+reg; 1 ssl; 2 ot; 3 proto

**目标：**需要loss 为bpr+reg + ot（目前是这样）(是否需要参数正则化？何时需要参数正则化？)

### 临时设置

eopch改为20次方便运行

参数一直是623104

NCL参数是618816

### 实验一：loss只剩 bpr+reg

没有loss is nan 并且loss降低，性能提高。初步**判断为加入的OT的问题**

### 实验二：loss只剩 ot +reg

没有loss is nan 但是loss很大且不下降，性能很低且不提升。

### 实验三：loss只剩ot

没有loss is nan 但是loss很大且不下降，性能很低且不提升。

### 实验四：loss为bpr+reg+ot（loss is nan）

### 实验五：loss为bpr + ot（loss is nan）

### 判断为参数初始化问题导致的

### 尝试方法一：参数初始化

#### ncl不参数初始化

数据集ml-1m

先跑ncl原封不动一遍，看实机性能(与论文一致)，记得截图（main_ncl_all.py在pycharm上跑）

再跑ncl没有消融实验的部分：

不初始化参数

不加reg

不初始化参数和不加reg

## 实验消融表格

| model                                      | r10    | n10    | r20    | n20    | r50    | n50    |
| :----------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| ncl_all                                    | 0.2057 | 0.2732 | 0.3037 | 0.2843 | 0.4686 | 0.3300 |
| LightGCN                                   | 0.1876 | 0.2514 | 0.2796 | 0.2620 | 0.4469 | 0.3091 |
| ncl_wo_init_pam                            | 0.1647 | 0.2248 | 0.2503 | 0.2351 | 0.4049 | 0.2783 |
| ncl_wo_loss_reg                            | 0.205  | 0.2725 | 0.3032 | 0.2841 | 0.4682 | 0.3301 |
| ncl_wo_init_pam_and_loss_reg               | 0.1647 | 0.2248 | 0.2503 | 0.2351 | 0.4049 | 0.2783 |
| new_wo_init_pam_w_bpr_ot                   | 0.1559 | 0.2116 | 0.2375 | 0.2223 | 0.3894 | 0.2656 |
| new_wo_init_pam_w_bpr_reg_ot               | 0.1567 | 0.2129 | 0.2396 | 0.2234 | 0.392  | 0.2669 |
| new_wo_init_pam_w_bpr_reg_ot_layer0        | 0.1559 | 0.2122 | 0.2381 | 0.2226 | 0.3898 | 0.2657 |
| new_wo_init_pam_w_bpr_reg_ot_layer2        | 0.1569 | 0.213  | 0.2395 | 0.2236 | 0.3915 | 0.2667 |
| new_wo_init_pam_w_bpr_reg_ot_layer**3**    | 0.1573 | 0.2126 | 0.2426 | 0.2245 | 0.3936 | 0.268  |
| new_w_gscinit_pam_bpr_reg_ot_mean          | 0.1559 | 0.2125 | 0.2375 | 0.2225 | 0.3893 | 0.2658 |
| new_w_init_pam_bpr_reg_ot_0.0001(小10倍)lr | 0.0609 | 0.0956 | 0.1005 | 0.0994 | 0.1784 | 0.12   |
| new_w_init_pam_bpr_reg_ot                  | nan    |        |        |        |        |        |
| new_w_init_pam_ot                          | 0.0749 | 0.1093 | 0.1186 | 0.1127 | 0.2028 | 0.1359 |

wd从0.69变成0.849，gwd变为nan

grad_fn的几种backward什么意思

**参数初始化和loss_reg都有用**

K的个数（15）是否太少？

## z_g的GATConv层数和LightGCN的层数问题

### GATConv

查询pyg.nn的[GATConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GATConv#torch_geometric.nn.conv.GATConv)

+ **调用一次就卷一层**

+ 输入输出维度可调参（GAT默认64，GSC默认64）

+ heads是否可设置为多头（GAT默认8，GSC默认1）

+ add_self_loops是否可以设置为True（GAT默认True，GSC默认False（输入为去自环的adj））

+ 是否可以升级为[GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GATConv#torch_geometric.nn.conv.GATv2Conv)

NCL的LightGCN的embedding为1层原始embedding+**3 layer**聚合的embedding

z_g = self.activation(self.conv(lightgcn_all_embeddings, self.adj2))使用LIghtGCN的embedding是否有问题？

### GCNConv

查询pyg.nn的[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GATConv#torch_geometric.nn.conv.GCNConv)

+ 不像LightGCN会取每层的平均

查看GSC的encoder：两层GCNConv（每层GCNConv+SumAggregation+Linear），**没有层数相加然后平均**

对应LightGCN的encoder：3层Light Graph Convolution（LGC），**层数相加然后平均**

z_g = self.activation(self.conv(lightgcn_all_embeddings, self.adj2))使用LIghtGCN的embedding是否有问题？中**lightgcn_all_embeddings换成最后一层试试？**

## 参数初始化？

```
self.apply(xavier_uniform_initialization)
```

### 了解xavier_uniform_initialization

recbole.model.init.xavier_uniform_initialization(*module*)

在PyTorch中使用[xavier_uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_)初始化nn中的参数（**nn.embedding** and **nn.Linear** layers）。对于nn.Linear中的bias，使用常量0初始化。

例子：self.apply(xavier_uniform_initialization)

```
def xavier_uniform_initialization(module):
	if isinstance(module, nn.Embedding):
		xavier_uniform_(module.weight.data)
	elif isinstance(module, nn.Linear):
		xavier_uniform_(module.weight.data)
		if module.bias is not None:
			constant_(module.bias.data, 0)
```

了解self.apply()

loss = bpr+reg+ot

**epoch1的batch1全部loss等于nan**

分析原因：

epoch1的batch0的ego_embedding第一行不是全0，之后第一层开始全0（查看没有ot的epoch0的batch0的第一行，一样）（查看有ot的epoch0的batch1的第一行）**这个应该没问题**

z_g的结果也很稀疏，很多0（查看GSC源代码生成的z_g,确实非常稀疏，很多0）**这个应该没问题**

**epoch1的batch1的user_embedding和item_embedding除了第一行都是nan**

有**没有可能**是self.apply(xavier_uniform_initialization)放在了GATConv前面导致没初始化GATConv？

(效果还不如不初始化)放入GSC的参数初始化模块，禁用self.apply(xavier_uniform_initialization)

```
for m in self.modules():
    self.weights_init(m)
def weights_init(self, m):
    if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight.data)
    if m.bias is not None:
    m.bias.data.fill_(0.0)
```

**（未做）**把pyg的GATConv直接放入model（感觉不太好拿过来，可以拿recbole其他带有gat的模型那搬过来试试）

**（recall20低点其他都好）**先试试**lightgcn不用stack和mean的最后一层数据**

（不行）减小ot_loss：1e-7 * ot_loss

## 先将GAT放入Model中试试（放不了）

## 会不会因为ot_loss是做分类的，多用了什么分类函数导致的（不懂）

没参数初始化，可以放一起。ncl参数初始化后，不能放一起

ot_loss比ncl的对比loss多了log()

NCL的对比loss

![ncl_loss](https://sunjc911.github.io/assets/images/rb/ncl_loss.png)

OT的对比loss

![wd_loss](https://sunjc911.github.io/assets/images/rb/wd_loss.png)

![gwd_loss](https://sunjc911.github.io/assets/images/rb/gwd_loss.png)

simgcl的变式对比loss

![simgcl_loss](https://sunjc911.github.io/assets/images/rb/simgcl_loss.png)

ncl358用了relu

## 调参

## 其他

debug 参数不加载 妈的！莫名其妙好了，妈的！

自己添加的loss怎么放进INFO里打出来？