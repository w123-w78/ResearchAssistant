# SparseTS：稀疏注意力长序列时间序列预测

## Abstract
针对标准 Transformer 在长序列预测中 O(n^2) 复杂度过高的问题，
本文提出 SparseTS，一种稀疏注意力时间序列预测模型。
预测长度 720 步时 MSE 相比 Informer 降低 19.3%，推理速度提升 2.8 倍。

## Method
SparseTS 的三项设计：
1. 稀疏注意力：每个查询只与 Top-32 相似键计算注意力，复杂度降为 O(n log n)
2. 序列分解：输入拆分为趋势项与季节项，分别建模后相加
3. 蒸馏压缩：每两层之间做一次 2 倍下采样

输入编码包含标量值投影、位置编码与时间特征（小时、星期、月份）。
损失函数为 MSE，优化器 Adam，学习率 1e-4，训练 100 个 epoch。

## Experiments
数据集：ETTm2（69680 条）、Electricity（321 个用户用电量）、
Traffic（862 个传感器路况）。
预测长度 720 时 MSE：Informer 0.412，Autoformer 0.387，
FEDformer 0.361，SparseTS 0.292。
推理耗时：Informer 单次 1.42 秒，SparseTS 0.51 秒。
消融显示，移除序列分解后 MSE 恶化至 0.348；
稀疏度从 32 降到 8 时 MSE 上升 0.021。

## Conclusion
序列分解对长程预测的贡献大于稀疏注意力本身，
二者的组合在长序列场景下效果最好。
