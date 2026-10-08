# Vision Transformer 在图像分类中的综述

## Abstract
本文系统综述了 Vision Transformer (ViT) 在图像分类任务上的研究进展。
我们在 ImageNet-1K、CIFAR-100 与 ADE20K 三个数据集上对比了 12 种代表性方法。
实验表明，当训练数据量超过 1 亿张时，ViT 的性能稳定超越同等规模的 CNN 基线。

## Introduction
自 2020 年 Dosovitskiy 等人提出 ViT 以来，Transformer 架构在计算机视觉领域
迅速普及。与卷积神经网络不同，ViT 将图像切分为 16x16 的 patch 序列，
通过自注意力机制建模全局依赖关系。

## Method
本文采用 DeiT 的训练策略，包含三项关键改进：
1. 强数据增强：RandAugment + MixUp + CutMix 组合
2. 知识蒸馏：引入 CNN 教师模型提供软标签监督
3. 随机深度（Stochastic Depth）：以 0.1 的概率随机丢弃层，缓解过拟合

损失函数为带标签平滑的交叉熵，平滑系数取 0.1。
优化器使用 AdamW，学习率 5e-4，权重衰减 0.05，训练 300 个 epoch。

## Experiments
在 ImageNet-1K 上，ViT-Base 达到 84.5% 的 Top-1 准确率，
ViT-Large 达到 87.1%，而同等参数量的 ResNet-152 基线为 82.3%。
在 CIFAR-100 数据集上，ViT-Base 取得 91.2% 的 Top-1 准确率。
消融实验显示，去除数据增强后准确率下降 4.7 个百分点，
去除知识蒸馏后下降 1.8 个百分点。

## Conclusion
本文验证了 ViT 在大规模数据下的性能优势，并指出数据增强是
影响最终性能的最关键因素。未来工作将探索更高效的注意力机制。
