# ViLT：极简视觉语言预训练模型

## Abstract
针对视觉语言模型中目标检测模块开销过大的问题，
本文提出 ViLT，用线性投影替代区域特征提取，
在 VQAv2 上达到 71.26 分，推理速度比区域特征类方法快 60 倍。

## Method
视觉侧：图像切分为 32x32 patch，线性投影为 token，
不使用 CNN 或目标检测器，直接送入 Transformer。
文本侧：WordPiece 分词后加位置编码。
两侧 token 拼接后输入单层模态融合 Transformer（12 层，隐藏维度 768）。

预训练目标有两个：
1. 图文匹配（ITM）：二分类判断图文是否配对
2. 掩码语言建模（MLM）：以 0.15 的概率遮盖文本 token 并预测

训练使用 AdamW，学习率 1e-4，batch size 4096，预训练 20 万步。

## Experiments
数据集：VQAv2、NLVR2、MSCOCO（检索）、Flickr30K。
VQAv2 得分：ViLT 为 71.26；基于区域特征的 ViLBERT 为 70.55，
VisualBERT 为 70.80。
COCO 图文检索 R@1：图到文 56.5，文到图 40.4。
消融显示，加入 masked patch prediction 后 VQA 下降 0.6 分；
去掉 ITM 目标后检索 R@1 下降 8.2。

## Conclusion
去掉视觉侧重度特征提取可行，但 ViLT 在检索类任务上
仍弱于使用区域特征的方法，说明细粒度视觉特征仍有价值。
