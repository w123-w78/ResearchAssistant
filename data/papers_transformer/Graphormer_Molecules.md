# Graphormer：图 Transformer 用于分子性质预测

## Abstract
本文将 Transformer 适配到图结构数据，提出 Graphormer，
通过结构编码显式注入图的拓扑与空间信息。
在 QM9 的 HOMO 能级预测上 MAE 降至 0.0325 eV，
在 PCQM4M-LSC 大规模基准上取得当时最优结果。

## Method
三项结构编码设计：
1. 中心性编码：按节点度数为每个节点分配可学习向量，注入节点重要性
2. 空间编码：用最短路径距离 D_ij 作为注意力偏置，可学习标量
3. 边编码：沿最短路径对边特征加权平均，加入注意力计算

模型为 12 层 Transformer，隐藏维度 768，32 个注意力头。
virtual node 用于聚合全图表示以做图级预测。
训练使用 Adam，学习率 2e-4，batch size 256，训练 100 万步。

## Experiments
数据集：QM9（13.4 万分子）、PCQM4M-LSC（380 万分子）、
MoleculeNet 的 BBBP 与 HIV。
QM9 HOMO 预测 MAE：GCN 0.0512，GAT 0.0478，
MPNN 0.0401，Graphormer 0.0325 eV。
BBBP 的 ROC-AUC 为 0.934。
消融显示，去掉空间编码后 MAE 恶化至 0.0455 eV；
去掉边编码后恶化 0.0031 eV。

## Conclusion
结构编码是图 Transformer 成功的关键，
空间（距离）信息对量子化学性质预测贡献最大。
