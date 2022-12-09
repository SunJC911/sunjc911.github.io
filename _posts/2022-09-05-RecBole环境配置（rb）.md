---
title: RecBole环境配置 + GSC + LightGCN + NCL（rb
description: Titan X rb 新模型为bpr+reg+OT（暂定，实验中）
date: 2022-09-05
categories:
 - exp
tags:
 - RecBole
 - exp
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

SELFRec

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

conda install -c aibox recbole
conda install faiss-gpu -c pytorch
```

实测ECCV-2022-Generative Subgraph Contrast for Self-Supervised Graph Representation Learning（**GSC**）和WWW-2022-Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning（**NCL**）都可以跑。

## 整理思路

### main

backbone: lightGCN

loss: BPR

### z_1和z_g

z_1:[layer0(initial), layer1, layer2, layer3, stack_mean_0_3(sm03)];

z_g:[gat(layer3), layer4, layer2, lg(sm03), simgcl]

### sample

numpy.random抽样user和item；

1.根据邻接矩阵抽样k1和k2得到子图的节点

2.根据二阶可达矩阵抽样k2得到子图的节点

### gwd的z_1的边

1.使用代价函数计算XX

2.使用邻接矩阵来获得z_1的邻接矩阵subg1_adj进行代价函数计算

3.使用可达矩阵来获得z_1的邻接矩阵subg1_adj进行代价函数计算

### gwd的T还是T_wd

### ot_loss

1.b_xnet

2.infonce

### default

```
print("**************")
print("default: xu_init, lightGCN, BPR, reg, 1e-7ot")
print("z1, z2: sm03, lg(list[-1])")
print("sample: 55b, 3k1, 2k2, 6040")
print("sample_adj, subg1_adj: A, A")
print("ot_loss: 55ui, fuck, no_norm, Cs_adj")
print("**************")
```

| model                                                        | r10               | n10               | r20               | n20           | r50               | n50           |
| ------------------------------------------------------------ | ----------------- | ----------------- | ----------------- | ------------- | ----------------- | ------------- |
| ncl_all                                                      | 0.2057            | 0.2732            | 0.3037            | 0.2843        | 0.4686            | 0.3300        |
| SGL                                                          | 0.1888            | 0.2526            | 0.2848            | 0.2649        | 0.4487            | 0.3111        |
| LightGCN                                                     | 0.1876            | 0.2514            | 0.2796            | 0.2620        | 0.4469            | 0.3091        |
| default(wg32a)[dontchange]                                   | <u>**0.1889**</u> | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | <u>**0.4493**</u> | <u>0.3108</u> |
| wg32x                                                        | <u>**0.1889**</u> | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | <u>**0.4493**</u> | <u>0.3108</u> |
| wg02a                                                        | **<u>0.189</u>**  | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | **<u>0.4493</u>** | <u>0.3108</u> |
| wg02x                                                        | **<u>0.189</u>**  | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | **<u>0.4493</u>** | <u>0.3108</u> |
| wg32a_z1:-2_z2:lg(list[-1])                                  | <u>**0.1889**</u> | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | <u>**0.4493**</u> | <u>0.3108</u> |
| wg32x_z1:-2_z2:lg(list[-1])                                  | <u>**0.1889**</u> | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | <u>**0.4493**</u> | <u>0.3108</u> |
| wg02a_z1:-2_z2:lg(list[-1])                                  | **<u>0.189</u>**  | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | **<u>0.4493</u>** | <u>0.3108</u> |
| wg02x_z1:-2_z2:lg(list[-1])                                  | **<u>0.189</u>**  | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | **<u>0.4493</u>** | <u>0.3108</u> |
| default_only_wd_3k1_2k2                                      | **<u>0.189</u>**  | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | **<u>0.4493</u>** | <u>0.3108</u> |
| default_only_wd_0k1_2k2                                      | **<u>0.189</u>**  | <u>0.2522</u>     | **0.2835**        | **0.2639**    | **<u>0.4493</u>** | <u>0.3108</u> |
| default_only_gwd_3k1_2k2_csadj                               | **<u>0.1889</u>** | 0.2521            | <u>0.2834</u>     | **0.2639**    | <u>0.**4493**</u> | <u>0.3108</u> |
| default_only_gwd_3k1_2k2_csxx                                | <u>**0.1889**</u> | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | <u>**0.4493**</u> | <u>0.3108</u> |
| default_only_gwd_0k1_2k2_csadj                               | <u>**0.1889**</u> | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | <u>**0.4493**</u> | <u>0.3108</u> |
| default_only_gwd_0k1_2k2_csxx                                | <u>**0.1889**</u> | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | <u>**0.4493**</u> | <u>0.3108</u> |
| default_only_wd_3k1_2k2_wobpr                                | 0.0078            | 0.012             | 0.0141            | 0.0129        | 0.0279            | 0.0178        |
| default_only_wd_0k1_2k2_wobpr                                | 0.0088            | 0.0085            | 0.0145            | 0.0104        | 0.0271            | 0.0146        |
| default_only_gwd_3k1_2k2_csadj_wobpr                         | 0.0658            | 0.0959            | 0.1138            | 0.1038        | 0.2059            | 0.1291        |
| default_only_gwd_3k1_2k2_csxx_wobpr                          | 0.076             | 0.1169            | 0.1291            | 0.1227        | 0.2175            | 0.1457        |
| default_only_gwd_0k1_2k2_csadj_wobpr                         | 0.0738            | 0.0939            | 0.1209            | 0.1022        | 0.2113            | 0.1305        |
| default_only_gwd_0k1_2k2_csxx_wobpr                          | 0.0783            | 0.1177            | 0.1299            | 0.1232        | 0.2223            | 0.148         |
| defalut_only_gwd_0k1_2k2_csxx_bpr{detail2}dwd                | 0.1848            | 0.2505            | 0.2763            | 0.2608        | 0.444             | 0.3078        |
| defalut_only_wd_0k1_2k2_csxx_bpr{detail2}dwd                 |                   |                   |                   |               |                   |               |
| default_Cs_xx{detail14}                                      | 0.1848            | 0.2505            | 0.2763            | 0.2608        | 0.444             | 0.3078        |
| z2_gat[dontchange]                                           | **<u>0.1889</u>** | <u>0.2522</u>     | **0.2835**        | **0.2639**    | **<u>0.4493</u>** | <u>0.3108</u> |
| 0k1_Cs_xx                                                    | <u>**0.189**</u>  | <u>0.2522</u>     | <u>0.2834</u>     | **0.2639**    | 0.4               | <u>0.3108</u> |
| default + A_2, xx; 0k1_30k2{detail14}                        | 0.1862            | 0.2511            | 0.2788            | 0.2621        | 04443             | 0.3083        |
| **default +kmean**                                           |                   |                   |                   |               |                   |               |
| z1_layer3_z2_lg(list[-1])                                    | 0.1848            | 0.2505            | 0.2763            | 0.2608        | 0.444             | 0.3708        |
| z1_sm13_z2_simgcl(都没加inital emb){detail6}                 | 0.1848            | 0.2505            | 0.2763            | 0.2608        | 0.444             | 0.3078        |
| z1_sm03_z2_simgcl(都加inital emb)                            | 0.1752            | 0.2416            | 0.2633            | 0.2511        | 0.4248            | 0.2958        |
| z1_sm03_z2_simgcl(没加inital emb){detail7}                   | 0.1848            | 0.2505            | 0.2763            | 0.2608        | 0.444             | 0.3078        |
| sample_adj, subg1_adj: A_2, A_2; 0k1_9k2                     | 0.1856            | 0.2506            | 0.278             | 0.2612        | 0.4431            | 0.3073        |
| sample_adj, subg1_adj: (A_2, A_2)noguiyi; 0k1_9k2            | 0.1856            | 0.2506            | 0.278             | 0.2612        | 0.4431            | 0.3073        |
| sample_adj, subg1_adj: A_2, A_2; 0k1_9k2; 1515b{detail1}     | 0.1783            | 0.2448            | 0.2678            | 0.2542        | 0.4302            | 0.2991        |
| sample_adj, subg1_adj: A_2, A_2; 0k1_9k2; 1515b；1e-8ot      | 0.1783            | 0.2448            | 0.2678            | 0.2542        | 0.4302            | 0.2991        |
| sample_adj, subg1_adj: A_2, A_2; 0k1_6k2                     | 0.1806            | 0.2465            | 0.27              | 0.2557        | 0.431             | 0.3004        |
| sample_adj, subg1_adj: A_2, A_2; 0k1_15k2                    | 0.1872            | 0.2518            | 0.2793            | 0.2624        | 0.4469            | 0.3094        |
| sample_adj, subg1_adj: A_2, xx; 0k1_15k2{tmux：15xx}         | 0.187             | 0.2521            | 0.2792            | 0.2627        | 0.4437            | 0.309         |
| sample_adj, subg1_adj: A_2, A_2; 0k1_20k2{detail9}           | 0.1842            | 0.2497            | 0.274             | 0.2592        | 0.4389            | 0.3053        |
| sample_adj, subg1_adj: A_2, A_2; 0k1_25k2{detail10}          | 0.1842            | 0.2497            | 0.274             | 0.2592        | 0.4389            | 0.3053        |
| sample_adj, subg1_adj: A_2, A_2; 0k1_30k2                    | 0.1866            | 0.2523            | 0.281             | 0.2635        | 0.4468            | 0.3101        |
| <u>sample_adj, subg1_adj: A_2, xx; 0k1_30k2{detail9}</u>{tmux：30xx} | 0.1877            | **0.2524**        | 0.2818            | <u>0.2637</u> | 0.4487            | **0.3109**    |
| sample_adj, subg1_adj: A_2, xx; 0k1_2k2_only_gwd_wobpr{detail9} |                   |                   |                   |               |                   |               |
| sample_adj, subg1_adj: A_2, A_2; 0k1_35k2{detail10}          | 0.185             | 0.2509            | 0.2764            | 0.261         | 0.441             | 0.307         |
| sample_adj, subg1_adj: A_2, A_2; 0k1_50k2                    | 0.1809            | 0.2469            | 0.2695            | 0.2558        | 0.4321            | 0.3012        |
| ncl_all                                                      | 0.2057            | 0.2732            | 0.3037            | 0.2843        | 0.4686            | 0.3300        |
| SGL                                                          | 0.1888            | 0.2526            | 0.2848            | 0.2649        | 0.4487            | 0.3111        |
| LightGCN                                                     | 0.1876            | 0.2514            | 0.2796            | 0.2620        | 0.4469            | 0.3091        |
| wg04A2x_z1:-2_z2:lg(list[-1])                                | **<u>0.19</u>**   | **<u>0.2531</u>** | **0.2844**        | **0.2643**    | <u>**0.449**</u>  | <u>0.3109</u> |
| wg04A2a_z1:-2_z2:lg(list[-1])                                | **<u>0.19</u>**   | **<u>0.2531</u>** | **0.2844**        | **0.2643**    | <u>**0.449**</u>  | <u>0.3109</u> |
| wg06A2x_z1:-2_z2:lg(list[-1])                                | **<u>0.19</u>**   | **<u>0.2531</u>** | **0.2844**        | **0.2643**    | <u>**0.449**</u>  | <u>0.3109</u> |
| wg06A2a_z1:-2_z2:lg(list[-1])                                | **<u>0.19</u>**   | **<u>0.2531</u>** | **0.2844**        | **0.2643**    | <u>**0.449**</u>  | <u>0.3109</u> |
| wg06A2x_z1:-2_z2:lg(list[-1])_负样本1(其他子图的z1)          | **<u>0.19</u>**   | **<u>0.2531</u>** | **0.2844**        | **0.2643**    | <u>**0.449**</u>  | <u>0.3109</u> |
| wg06A2x_z1:-2_z2:lg(list[-1])_负样本2(其他子图的z2)          | 0.1871            | 0.2517            | 0.2808            | 0.2628        | 0.4441            | 0.3087        |
| wg06A2a_z1:01_z2:0123(a6,moba3)                              | 0.19              | 0.2531            | 0.2844            | 0.2643        | 0.449             | 0.3109        |
| wg06A2a_z1:gat1_z2:gat3(a6,moba6)                            | 0.19              | 0.2531            | 0.2844            | 0.2643        | 0.4489            | 0.3109        |
| wg06A2a_z1:-3_z2:gat3(a6,moba7)                              | <u>**0.1901**</u> | <u>**0.2532**</u> | 0.2844            | 0.2643        | 0.4489            | 0.3109        |
| wg06A2a_z1:-2_z2:gat4(a6,moba8)                              | 0.1871            | 0.2517            | 0.2808            | 0.2628        | 0.4442            | 0.3087        |
| wg06A2a_z1:-2_z2:lg(list[-1])+ncl_kmeans(感觉得调调参)       | 0.1884            | 0.2528            | 0.281             | 0.2635        | 0.4452            | 0.3096        |
| wg06A2a_z1:-2_z2:lg(list[-1])+0.1*ncl_kmeans（moba2.）       | 0.19              | 0.253             | 0.2843            | 0.2642        | 0.4488            | 0.3109        |
| wg06A2a_z1:-3_z2:-1（a5，moba4.）                            | **<u>0.1901</u>** | **<u>0.2532</u>** | 0.2844            | 0.2643        | 0.449             | 0.3109        |
| wg06A2a_z1:-3_z2:-1_warmupall（a5，moba4.）                  | 0.1871            | 0.2517            | 0.2808            | 0.2628        | 0.4441            | 0.3087        |
| wg06A2a_z1:-3_z2:-1+ncl_kmeans                               | 0.1884            | 0.2527            | 0.2809            | 0.2635        | 0.4452            | 0.3096        |
| wg06A2a_z1:-3_z2:-1+0.1*ncl_kmeans                           | 0.19              | 0.253             | 0.2843            | 0.2642        | 0.4488            | 0.3109        |
| wg06A2a_z1:-3_z2:-1+0.01*ncl_kmeans(moba2                    | 0.1901            | 0.2531            | **<u>0.2845</u>** | 0.2643        | 0.449             | 0.3109        |
| wg06A2a_z1:-3_z2:-1+0.01*ncl_kmeans_warmupall(a,moba3)       | 0.1901            | 0.2531            | 0.2845            | 0.2643        | 0.449             | 0.3109        |
| wg06A2a_z1:-3_z2:-1+0.001*ncl_kmeans(moba3                   | 0.19              | 0.2531            | 0.2844            | 0.2643        | 0.449             | 0.3109        |
| wg09A2x_z1:-2_z2:lg(list[-1])                                | 0.1871            | 0.2517            | 0.2808            | 0.2628        | 0.4441            | 0.3087        |
| wg09A2a_z1:-2_z2:lg(list[-1])                                | 0.1871            | 0.2517            | 0.2808            | 0.2628        | 0.4441            | 0.3087        |
| wg030A2x_z1:-2_z2:lg(list[-1])                               | 0.1874            | 0.2523            | 0.2805            | 0.2633        | 0.4463            | 0.3099        |
| wg030A2a_z1:-2_z2:lg(list[-1])                               | 0.1874            | 0.2523            | 0.2805            | 0.2633        | 0.4463            | 0.3099        |
| wg06A2wx_z1:-2_z2:lg(list[-1])                               | 报错              |                   |                   |               |                   |               |
| wg06A2wa_z1:-2_z2:lg(list[-1])                               | 报错              |                   |                   |               |                   |               |
| sample_adj, subg1_adj:kmean, A_2;55c_5k2{detail5}            | 0.1843            | 0.2495            | 0.2769            | 0.2604        | 0.4395            | 0.306         |
| sample_adj, subg1_adj:kmean, xx;55c_5k2{detail11}            | 0.1852            | 0.2508            | 0.2764            | 0.261         | 0.4429            | 0.3076        |
| sample_adj, subg1_adj:kmean, A_2;55c_10k2                    | 0.1827            | 0.2494            | 0.2759            | 0.2599        | 0.4426            | 0.3066        |
| sample_adj, subg1_adj:kmean, xx;55c_10k2                     | 0.1827            | 0.2491            | 0.276             | 0.2599        | 0.4397            | 0.3056        |
| sample_adj, subg1_adj:kmean, A_2;55c_15k2                    |                   |                   |                   |               |                   |               |
| sample_adj, subg1_adj:kmean, xx;55c_15k2                     |                   |                   |                   |               |                   |               |
| wg500c07A2x_z1:-2_z2:lg(list[-1])                            | 0.1843            | 0.2501            | 0.2778            | 0.261         | 0.4407            | 0.3064        |
| wg500c07A2a_z1:-2_z2:lg(list[-1])                            | 报错              |                   |                   |               |                   |               |

## 用ncl的聚类然后抽样获得子图{detail5}



## 最新想法

### 1.子图emb z能不能用ncl的跳一层,侧重获得user或者item的emb？（detail3）

~~z1：center_emb(list[0])；z2：context_emb(list[2])~~

z1:LG_mean；z2：LG(layer[-1])(recall@20 少一点)，其他一样

**（未完待续）**

### 2.GSC用input_adj作为XX的边，合理，因为是现成的（**能不能二阶邻接矩阵**）（detail2）

没有做到user，item分开，因为邻接矩阵的1为user和item的交互

exp1. z_g用lg(list[-1])，sample后不加入k1节点，用xx代替矩阵（add no k1,use xx）（效果不如加上k1）

**（未完待续）**

### 2.1 构造二阶矩阵，然后去构造input_adj (detail1)

self.A_tensor * self.A_tensor 得到可达矩阵，再使得对角线为0，<u>大于1的全为1</u>（也许不用？说明更相关？）；

根据可达矩阵进行抽样，再使用input_adj构造xx

**ps**: input_adj几乎全交互，带点user不交互会不会好点？

### 3.fuckloss分母是不是多了四个正例？（detail1）

对结果没影响

### 4.gat不带activate行不行

### 5.T和T_wd得到的结果没区别，最好探究一下

### 6.fuck_loss是不是有问题啊？（）

### 7.random walk可以用来sample吗

### 8.用simgcl生成z_g呢？



## 探究GOT，GSC的代码差异细节

++++++++++++++++++++++++++

根据

- 初始化：xv
- loss：mf_loss + self.reg_weight * reg_loss, self.ssl_reg * ot_loss
- ui_55b_3k1_2k2_6040
- nc_f

++++++++++++++++++++++

### 1.(还要继续)加上xy正则化，然后再用得到的cos_dis减去max试

如果<u>可以出结果(不如不加)</u>，那么减去max是否可以替换成其他的方法？

<u>去掉减max，0epoch 88batchsize loss没问题，**bp后emb为nan**</u>

### 2.(再想想)GOT与GSC的cos_dis（detail1）

GOT：cos_dis = 1 - cos_dis

GSC：cos_dis = torch.exp(- cos_dis / tau)

我：cos_dis = torch.exp((- cos_dis / tau) - (- cos_dis / tau).max())

如果我要**加上或者改成**GOT的形式，必须要<u>normXY</u>

1.~~new 我~~：cos_dis = torch.exp((1 - cos_dis) / tau)（<u>0e0b ot:nan</u>）

2.~~new 我~~：cos_dis = 1 - cos_dis(效果不好)

### 3.GOT与GSC里wd除了上述2.没有区别

### 4.gwd多了输入参数T_wd, input_adj, tau

- T-wd: wd求出来的transport plan T
- input_adj: 子图的邻接矩阵
- tau: 温度

### 5.GW_distance差异巨大

```
# GOT
Cs = cos_batch_torch(X, X).float().cuda()
Ct = cos_batch_torch(Y, Y).float().cuda()
# GSC
cos_dis = torch.exp(- input_adj / tau).cuda()
beta = 0.1
min_score = cos_dis.min()
max_score = cos_dis.max()
threshold = min_score + beta * (max_score - min_score)
res = cos_dis - threshold
Cs = torch.nn.functional.relu(res.transpose(2, 1))
Ct = self.cos_batch(Y, Y, tau).float().cuda()

# 共同
bs = Cs.size(0)
m = Ct.size(2)
n = Cs.size(2)
T, Cst = self.GW_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)

# GOT
temp = torch.bmm(torch.transpose(Cst,1,2), T)
# GSC
temp = torch.bmm(torch.transpose(Cst, 1, 2), T_wd)

# 共同
distance = self.batch_trace(temp, m, bs)
return distance
```

区别：GSC用input_adj作为XX的边，合理，因为是现成的（**能不能二阶？因为是个二部图**）。用T和T_wd不清楚，可能是试出来的。

exp1.T_wd换成T

exp2.XX替换res

exp3.1＋2

## loss is nan

搞定"**w**_init_pam_bpr_reg_1e-7ot_l3_150150b_55ui_5k1_10k2_6040"这个，应该就能超过了，加油！

### Train 0 39/165 <u>ot_loss:nan</u>( (改为fuckloss后也为train 0 39/165))

其中loss_user is nan, loss_item还能计算出值（我估计再epoch就nan了，因为和user用一个算法）

loss_user中loss1可以计算，<u>loss2的b_xnet算出来是nan</u> 

### 去掉gwd运行试试

**Train1 57/165 全部loss为nan**(改为fuckloss后也为train 0 39/165)), 从56开始看起，估计出错在bp导致embedding为nan

all_embedding(9669,64)只有（1，64）不为nan，其全为nan。寻找下36的embedding多少。

改为fuck还出错说明self.wd出现了bp为nan的算子

### 调查self.wd

xy正则化有影响，bce有影响

#### 用bce

注释掉def cost_matrix_batch里的x与y的正则化后上面两个情况不报错，等跑完再加上gwd看看效果。（如果保留正则化，1e-12是不是有问题呢？）：<u>13 76/165 all loss nan</u>

同时跑一下加上gwd的（cos_batch 有正则化，未注释）：<u>0 39/165 nan</u>

跑一下加上gwd的（tmux gwd）（def cos_batch注释掉正则化）：<u>0 98/165 gwd nan</u>

#### 用fuck（目前在这）

注释掉def cost_matrix_batch里的x与y的正则化后上面两个情况不报错，等跑完再加上gwd看看效果。（pyc）（如果保留正则化，1e-12是不是有问题呢？）：<u>35 58/165 item loss nan</u> 从57开始看起

这：<u>把采样都改小，定位到ctrl+1, temp2有使得exp为inf的值</u>

https://blog.csdn.net/jump882/article/details/121371018

改为cos_dis = torch.exp((- cos_dis / tau) - (- cos_dis / tau).max())



同时跑一下加上gwd的

不注释＋fuck呢？

### 添加代码得到反向传播的错误信息

import torch.autograd as autograd

with autograd.detect_anomaly():

​            loss.backward()

什么算子会导致bp为loss

产生的疑问：

torch.exp输入什么值会nan

记录：

self.wd()开始对embedding动手了

cos_dist= torch.nn.functional.relu(cos_distance - threshold)

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

| model                                                        | r10           | n10           | r20           | n20           | r50           | n50           |
| :----------------------------------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ncl_all                                                      | 0.2057        | 0.2732        | 0.3037        | 0.2843        | 0.4686        | 0.3300        |
| SGL                                                          | 0.1888        | 0.2526        | 0.2848        | 0.2649        | 0.4487        | 0.3111        |
| LightGCN                                                     | 0.1876        | 0.2514        | 0.2796        | 0.2620        | 0.4469        | 0.3091        |
| ncl_wo_init_pam                                              | 0.1647        | 0.2248        | 0.2503        | 0.2351        | 0.4049        | 0.2783        |
| ncl_wo_loss_reg                                              | 0.205         | 0.2725        | 0.3032        | 0.2841        | 0.4682        | 0.3301        |
| ncl_wo_init_pam_and_loss_reg                                 | 0.1647        | 0.2248        | 0.2503        | 0.2351        | 0.4049        | 0.2783        |
| new_wo_init_pam_w_bpr_ot                                     | 0.1559        | 0.2116        | 0.2375        | 0.2223        | 0.3894        | 0.2656        |
| new_wo_init_pam_w_bpr_reg_ot                                 | 0.1567        | 0.2129        | 0.2396        | 0.2234        | 0.392         | 0.2669        |
| new_wo_init_pam_w_bpr_reg_ot_layer0                          | 0.1559        | 0.2122        | 0.2381        | 0.2226        | 0.3898        | 0.2657        |
| new_wo_init_pam_w_bpr_reg_ot_layer2                          | 0.1569        | 0.213         | 0.2395        | 0.2236        | 0.3915        | 0.2667        |
| new_wo_init_pam_w_bpr_reg_ot_**layer3****                    | 0.1573        | 0.2126        | 0.2426        | 0.2245        | 0.3936        | 0.268         |
| new_w_gscinit_pam_bpr_reg_ot_layer3_150150                   | 0.1548        | 0.2124        | 0.2378        | 0.2229        | 0.3902        | 0.2658        |
| new_w_newgscinit_pam_bpr_reg_ot_layer3_150150                | nan           |               |               |               |               |               |
| main_wo_init_pam_w_bpr_reg_ot_l3_gat2                        | 0.1546        | 0.2102        | 0.2364        | 0.2212        | 0.3887        | 0.2647        |
| main_wo_init_pam_w_bpr_reg_ot_l3_300batch                    | 0.1549        | 0.2125        | 0.239         | 0.2236        | 0.3895        | 0.266         |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150batch                 | 0.1548        | 0.212         | 0.2392        | 0.223         | 0.3912        | 0.2658        |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150batch_k30             | 0.1537        | 0.2117        | 0.2377        | 0.2223        | 0.3857        | 0.264         |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150batch_5k1_5k2         | 0.1669        | 0.2237        | 0.2529        | 0.2342        | 0.4045        | 0.2771        |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150b_**5**k1\_**10**k2   | 0.1647        | 0.2228        | 0.2534        | 0.2347        | 0.4049        | 0.2774        |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150b_5k1_10k2_**6040**   | 0.1582        | 0.2167        | 0.2407        | 0.2265        | 0.3924        | 0.2688        |
| main_wo_init_pam_w_bpr_reg_**1e-7**ot_l3_150150b_5k1_10k2_6040 | 0.1647        | 0.2216        | 0.2548        | 0.2346        | 0.408         | 0.2782        |
| main_wo_init_pam_w_bpr_reg_**1e-7**ot_l3_150150b_ui_5k1_10k2_6040 | 0.1656        | 0.2215        | 0.2524        | 0.2334        | 0.4075        | 0.2772        |
| <u>ncl_all</u>                                               | <u>0.2057</u> | <u>0.2732</u> | <u>0.3037</u> | <u>0.2843</u> | <u>0.4686</u> | <u>0.3300</u> |
| <u>SGL</u>                                                   | <u>0.1888</u> | <u>0.2526</u> | <u>0.2848</u> | <u>0.2649</u> | <u>0.4487</u> | <u>0.3111</u> |
| <u>LightGCN</u>                                              | <u>0.1876</u> | <u>0.2514</u> | <u>0.2796</u> | <u>0.2620</u> | <u>0.4469</u> | <u>0.3091</u> |
| main_w_init_pam_w_bpr_reg_**1e-7**ot_l3_150150b_ui_5k1_10k2_6040_wd_f_nc(new_cos_dis) | 0.1733        | 0.2394        | 0.2629        | 0.2495        | 0.4173        | 0.292         |
| w_init_pam_w_bpr_reg_**1e-7**ot_l3\_**5050**b_ui_5k1_10k2_6040_wd_f_nc(50) | 0.187         | 0.2522        | 0.2797        | 0.263         | 0.444         | 0.3093        |
| 1e-7ot_55b_3k1_2k2_6040_nc_f(f-4相同)                        | **0.1889**    | **0.2522**    | **0.2835**    | **0.2639**    | **0.4493**    | **0.3108**    |
| 1e-7ot_55b_3k1_2k2_6040_nc_f_normxy                          | 0.1882        | 0.252         | 0.281         | 0.263         | 0.4468        | 0.3098        |
| 1e-7ot_55b_3k1_2k2_6040_nc_f_normxy_1-cd                     | 0.1741        | 0.2395        | 0.2622        | 0.2494        | 0.4204        | 0.2929        |
| 1e-7ot_55b_3k1_2k2_6040_nc_f_z10_z22                         | 0.1794        | 0.2454        | 0.2717        | 0.2562        | 0.4333        | 0.301         |
| 1e-7ot_55b_4k1_2k2_6040_nc_f                                 | 0.1873        | 0.2524        | 0.2807        | 0.2635        | 0.4449        | 0.3094        |
| 1e-6ot_55b_3k1_2k2_6040_nc_f                                 | 0.1795        | 0.2454        | 0.2715        | 0.2561        | 0.4333        | 0.301         |
| 1e-8ot_55b_3k1_2k2_6040_nc_f                                 | 0.1889        | 0.2522        | 0.2834        | 0.2639        | 0.4493        | 0.3108        |
| 1e-7ot_55b_3k1_2k2_6040_nc_bce                               | 0.1889        | 0.2522        | 0.2834        | 0.2639        | 0.4493        | 0.3108        |
| 1e-7ot_55b_3k1_2k2_6040_nc_f_z1mean_z2lg(list[-1])           | **0.1889**    | **0.2522**    | **0.2834**    | **0.2639**    | **0.4493**    | **0.3108**    |
| 1e-7ot_55b_0k1_2k2_6040_nc_f_z1mean_z2lg(list[-1])_xx        | **0.189**     | 0.2522        | 0.2834        | 0.2639        | 0.4           | 0.3108        |
| 1e-7ot_55b_0k1_3k2_6040_nc_f_z1mean_z2lg(list[-1])_xx        |               |               |               |               |               |               |
|                                                              |               |               |               |               |               |               |
| w_init_w_bpr_reg_**1e-7**ot_l3_150150b_ui_5k1_10k2_6040_wdbce_nc(wdbce) | 0.1731        | 0.2393        | 0.2629        | 0.2495        | 0.4179        | 0.2922        |
|                                                              |               |               |               |               |               |               |
| main\_**w**_init_pam_bpr_reg_1e-7ot_l3_150150b_55ui_5k1_10k2_6040 |               |               |               |               |               |               |
| wo_init_pam_w_bpr_reg_**1e-7**ot_l3_150150b_55ui_5k1_10k2_6040 | 0.1656        | 0.2215        | 0.2524        | 0.2334        | 0.4075        | 0.2772        |
| wo_init_pam_w_bpr_reg_**1e-7**ot_l3_150150b_55ui_5k1_10k2_6040_f | **0.1663**    | **0.2221**    | 0.2531        | 0.2336        | 0.4079        | 0.2776        |
| wo_init_pam_w_bpr_reg_**1e-7**ot_l3_150150b_55ui_5k1_10k2_6040_2f | 0.1656        | 0.2215        | 0.2524        | 0.2334        | 0.4075        | 0.2772        |
| main_wo_init_pam_w_bpr_reg_**1e-7**ot_l3_200100b_5k1_10k2_6040 | 0.1659        | 0.2221        | 0.2531        | 0.2343        | 0.4058        | 0.2774        |
| main_wo_init_pam_w_bpr_reg_**1e-7**ot_l3_512512b_5k1_10k2_6040 | 0.1651        | 0.2221        | 0.2524        | 0.2337        | 0.4053        | 0.277         |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150b_6040_0k1_10k2       | 0.1597        | 0.211         | 0.241         | 0.2225        | 0.392         | 0.266         |
| main_wo_init_pam_w_bpr_reg_ot_l3_300b_5k1_10k2_6040          | 0.1642        | 0.2223        | 0.2522        | 0.2339        | 0.4043        | 0.2767        |
| new_wo_init_w_bpr_reg_ot_l3_300300b_6040_5k1_10k2            | 0.1607        | 0.2172        | 0.2465        | 0.2289        | 0.3977        | 0.2716        |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150b_10k1_5k2            | 0.1633        | 0.2207        | 0.2506        | 0.232         | 0.4016        | 0.2746        |
| main_wo_init_pam_w_bpr_reg_ot_l3_150150b_10k1_10k2           | 0.161         | 0.2203        | 0.2471        | 0.2307        | 0.3995        | 0.2732        |
| new_w_gscinit_pam_bpr_reg_ot_mean                            | 0.1559        | 0.2125        | 0.2375        | 0.2225        | 0.3893        | 0.2658        |
| new_w_init_pam_bpr_reg_ot_0.0001(小10倍)lr                   | 0.0609        | 0.0956        | 0.1005        | 0.0994        | 0.1784        | 0.12          |
| new_w_init_pam_bpr_reg_ot                                    | nan           |               |               |               |               |               |
| new_w_init_pam_ot                                            | 0.0749        | 0.1093        | 0.1186        | 0.1127        | 0.2028        | 0.1359        |

**wd从0.69变成0.849，gwd变为nan 搞不懂啊！**

会不会是没分物品和用户的loss导致的

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

## （感觉不行）新想法：用NCL的聚类获得伪标签指导GSC的分类Loss

来源：GSC的test需要标签

可是recall和ndcg不需要标签吧？

rec所谓的标签应该就是交互和不交互的二分类吧

### 先搞清楚rebole的valid和test的逻辑

## (未实现)新想法：放弃BCE，分别对item和user进行infonce

## （实验中）新想法：BFS改成1阶拿5个，2阶拿几个

### **关于LightGCN**

该算法采用SGC[1]算法**去除**了常规GCN算法中的**变换矩阵和非线性激活函数等冗余操作**(用GATConv对LightGCN后的embedding去生成子图是不是就不好？)

### 每个batch用bfs寻找节点还是每个epoch？node_batch用随机选取会不会有问题？

首先，放到batch得到最好结果

实验放到epoch得到什么，没有userid和itemid

## 其他

debug 参数不加载 妈的！莫名其妙好了，妈的！

自己添加的loss怎么放进INFO里打出来？