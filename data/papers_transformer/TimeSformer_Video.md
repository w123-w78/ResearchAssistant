# TimeSformer：时空注意力视频理解模型

## Abstract
本文提出 TimeSformer，一个无卷积的纯 Transformer 视频分类模型，
对比了五种时空注意力方案。在 Kinetics-400 上达到 78.0% 的 Top-1 准确率，
训练成本仅为 3D CNN 的 1/10。

## Method
输入为 8 帧、每帧 224x224 的片段，每帧切分为 16x16 patch，
共 8x196=1568 个 token，加时空位置编码后送入 12 层 Transformer。

对比的五种注意力方案：
1. 联合时空注意力：所有 token 两两计算，准确率最高但复杂度 O((nt)^2)
2. 分离时空注意力：先时间后空间，复杂度降为 O(n^2 + t^2)，精度损失 0.4%
3. 稀疏局部全局注意力
4. 轴向注意力
5. 仅空间注意力（基线，无时序建模）

模型采用 ViT-Base 结构，在 ImageNet-21K 上预训练。

## Experiments
数据集：Kinetics-400（24 万视频，400 类）、Kinetics-600、
Something-Something-v2、Diving-48。
Kinetics-400 Top-1：联合注意力 78.0%，分离注意力 77.6%，
仅空间注意力 74.9%。
训练成本：I3D 需 8.2 GPU-天，TimeSformer 仅需 0.8 GPU-天。
消融显示，帧数从 8 增到 96 时准确率提升 3.1 个点。

## Conclusion
分离时空注意力在几乎不损失精度的前提下大幅降低计算量，
是视频 Transformer 的实用选择。
