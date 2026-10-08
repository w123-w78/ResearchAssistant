# Vision Transformer 在图像分类中的应用

## Abstract
本文将 Transformer 直接应用于图像分类，提出 Vision Transformer (ViT)。
在 ImageNet-1K 上，ViT-Large 达到 87.1% 的 Top-1 准确率，
但需要先在 JFT-300M 大规模数据上预训练。

## Method
将 224x224 图像切分为 16x16 的 patch，共 196 个 patch，
每个 patch 线性投影为 768 维 token，加上可学习的位置编码后送入标准 Transformer 编码器。
ViT-Base 为 12 层、隐藏维度 768、12 个注意力头；
ViT-Large 为 24 层、隐藏维度 1024、16 个头。

训练使用 AdamW，学习率 3e-3，权重衰减 0.3，batch size 4096，预训练 30 万步。
微调阶段采用余弦学习率衰减与 RandAugment 数据增强。

## Experiments
数据集：ImageNet-1K（128 万张，1000 类）、CIFAR-100、VTAB。
在 ImageNet-1K 上，ViT-Base 为 84.5%，ViT-Large 为 87.1%，
同等规模的 ResNet-152 基线为 82.3%。
消融显示，位置编码改为 2D 感知式并未带来明显提升（+0.1%）。

## Conclusion
ViT 证明了纯 Transformer 在大数据量下可超越 CNN，
但其对预训练数据规模的依赖是主要局限。
