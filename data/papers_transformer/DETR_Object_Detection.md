# DETR：基于 Transformer 的端到端目标检测

## Abstract
本文提出 DETR，将目标检测重构为集合预测问题，
彻底去掉锚框与非极大值抑制（NMS）等手工组件。
在 COCO 上达到 42.0 AP，与优化良好的 Faster R-CNN 基线持平。

## Method
DETR 由 CNN 骨干、Transformer 编解码器与预测头组成。
输入图像经 ResNet-50 提取特征后展平，加上位置编码送入 6 层编码器；
解码器输入 100 个可学习的 object query，6 层，并行输出所有预测。

训练采用匈牙利匹配：在预测与真值之间做一对一最优匹配，
匹配代价由分类损失、L1 框损失与 GIoU 损失加权构成（权重 1:5:2）。
优化器 AdamW，学习率 1e-4，batch size 64，训练 500 个 epoch。

## Experiments
数据集：COCO 2017（11.8 万训练图，80 类）。
结果：DETR 为 42.0 AP；引入多尺度特征与 300 epoch 训练的
改进版 DETR-DC5 为 43.3 AP；Faster R-CNN 基线为 42.0 AP。
对小目标 AP 仅 20.5（低于 Faster R-CNN 的 24.2），
大目标 AP 达 61.1，明显优于基线。
消融显示，去掉 Transformer 编码器后 AP 下降 3.9。

## Conclusion
DETR 验证了端到端集合预测的可行性，
但小目标性能不足与训练收敛慢（需 500 epoch）是主要短板。
