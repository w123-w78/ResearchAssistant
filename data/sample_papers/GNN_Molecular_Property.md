# 图神经网络用于分子性质预测的研究

## Abstract
本文系统比较了 GCN、GAT、MPNN 与 Graphormer 四类图神经网络在
分子性质预测任务上的表现。在 QM9 与 MoleculeNet 两个基准上评测，
结果表明引入 3D 空间信息的模型在量子化学性质预测上优势明显。

## Method
分子被建模为图：原子为节点，化学键为边。
节点特征包含原子序数、电荷、杂化方式等 11 维特征；
边特征包含键型、是否成环等 4 维特征。

对比的四种架构：
- GCN：谱域卷积，2 层，隐藏维度 128
- GAT：注意力聚合，4 头，隐藏维度 128
- MPNN：消息传递，迭代 3 步
- Graphormer：图 Transformer，6 层，引入空间距离编码

训练使用 Adam 优化器，学习率 1e-3，batch size 64，训练 200 个 epoch。

## Experiments
数据集：QM9（13.4 万分子，12 个量子化学性质）、
MoleculeNet 的 BBBP（2039 个分子，血脑屏障穿透性）与 HIV（41127 个分子）。

在 QM9 的 HOMO 能级预测上，MAE 分别为：
GCN 0.0512 eV、GAT 0.0478 eV、MPNN 0.0401 eV、Graphormer 0.0325 eV。
在 BBBP 分类上，ROC-AUC 分别为 0.881、0.902、0.917、0.934。
消融显示，移除 3D 距离编码后 Graphormer 的 MAE 恶化至 0.0455 eV。

## Conclusion
Graphormer 在两个基准上均取得最优结果，3D 空间信息对
量子化学性质预测贡献最大（贡献约 28% 的性能提升）。
