---
title: RecBole环境配置 + GSC + LightGCN + NCL（rb）
description: Titan X rb
date: 2022-09-05
categories:
 - RecBole
tags:
 - RecBole
excerpt_separator: <!--more--> 

---

目前进度：将GSC粗暴加入NCL，可以运行产生loss，循环到epoch4报错

问题：trainer.py的nodes_batch: 随机选择节点作为中心节点（会重复），**怎么自动获取用户数和物品数？**；loss数量与参数正则化的loss怎么搞？；GATConv和LightGCN层数问题？；how参数初始化？etc.

调参：k1：每个子图BFS搜索到的节点数量（包括自己），设为k=15待调参；Model的tau手动设为0.5，待调参（可以先不动）etc.

进阶问题：每个batch多少个节点？每个中心节点找几个节点？etc.

进阶模块：用户和物品分开（模糊思路）？BFS换一下？etc.

<!--more--> 

# 环境配置

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

~~**可以运行啦**~~（运行报错）

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

目前：losses: 0 bpr+reg; 1 ssl; 2 ot; 3 proto

**目标：**需要loss 为bpr+reg + ot(是否需要参数正则化？何时需要参数正则化？)

## z_g的GATConv层数和LightGCN的层数问题

## 参数初始化？

## 调参

## 其他

debug 参数不加载 妈的！莫名其妙好了，妈的！

自己添加的loss怎么放进INFO里打出来？