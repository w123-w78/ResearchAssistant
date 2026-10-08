# TransUNet：Transformer 与 U-Net 结合的医学图像分割

## Abstract
本文将 Transformer 引入 U-Net 架构，提出 TransUNet，
同时利用 CNN 的局部细节与 Transformer 的全局上下文。
在 Synapse 多器官分割上 Dice 达到 77.48%，相比 U-Net 提升 3.9 个点。

## Method
编码器：先用 ResNet-50 提取多尺度特征，
在最低分辨率（14x14）处接入 12 层 Transformer 编码器，
patch 大小 1x1，隐藏维度 768，注意力头数 12。
解码器：沿用 U-Net 的跳跃连接结构，逐级上采样并与编码器特征拼接。

损失函数为 Dice 损失与交叉熵损失之和，权重各 0.5。
优化器 SGD，学习率 0.01，动量 0.9，训练 300 个 epoch。

## Experiments
数据集：Synapse 多器官 CT（30 例，8 个器官）、
ACDC 心脏 MRI（150 例）。
在 Synapse 上平均 Dice 为 77.48%，平均 HD95 为 31.69mm；
U-Net 基线为 73.55% 与 36.87mm。
在 ACDC 上平均 Dice 为 89.71%。
消融显示，Transformer 层数从 12 减到 6 时 Dice 下降 2.4 个点；
在 1/4 分辨率接入 Transformer 效果最优。

## Conclusion
Transformer 作为编码器的瓶颈层能有效建模器官间的长程依赖，
但纯 Transformer 编码器在小数据集上不如混合架构。
