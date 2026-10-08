# Swin Transformer：层次化视觉骨干网络

## Abstract
针对 ViT 计算复杂度随图像尺寸平方增长、且不适配密集预测任务的问题，
本文提出 Swin Transformer，采用移位窗口注意力构建层次化特征图。
在 COCO 目标检测上达到 58.7 box AP，在 ADE20K 语义分割上达到 53.5 mIoU。

## Method
核心设计为移位窗口（shifted window）自注意力：
将特征图划分为不重叠的 MxM 局部窗口（默认 M=7），
只在窗口内计算自注意力，复杂度从 O(n^2) 降为线性。
相邻层之间窗口位置偏移 M/2，实现跨窗口信息交互。

层次化结构共 4 个 stage，通道数依次为 96、192、384、768，
各 stage 的 patch 数依次减少，形成类似 CNN 的多尺度特征。
模型规模：Swin-T（0.29 亿参数）、Swin-B（0.88 亿）、Swin-L（1.97 亿）。

## Experiments
数据集：ImageNet-1K（分类）、COCO（检测）、ADE20K（分割）。
ImageNet-1K Top-1：Swin-B 为 84.5%，Swin-L 为 87.3%。
COCO 检测 box AP：Swin-L 为 58.7，mask AP 为 51.1。
ADE20K 分割 mIoU：Swin-L 为 53.5。
消融显示，去掉移位操作后 ImageNet 准确率下降 1.1 个点。

## Conclusion
移位窗口机制在保持线性复杂度的同时实现了跨窗口建模，
使 Transformer 首次成为通用的视觉骨干网络。
