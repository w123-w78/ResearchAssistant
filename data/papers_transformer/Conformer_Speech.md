# Conformer：卷积增强的语音识别 Transformer

## Abstract
针对 Transformer 擅长全局建模但局部特征提取弱的问题，
本文提出 Conformer，将卷积模块嵌入 Transformer 块。
在 LibriSpeech 上词错误率（WER）降至 2.1%/4.3%（test-clean/test-other），
相对 Transformer 基线相对降低 17%。

## Method
每个 Conformer 块依次包含四个模块：
1. 前馈模块（半步残差，缩放 0.5）
2. 多头自注意力（8 头，隐藏维度 512）
3. 卷积模块：深度可分离卷积，核大小 31，带 GLU 门控
4. 第二个前馈模块（半步残差）

编码器共 17 层，参数量 1.18 亿。
前端为 80 维梅尔频谱，经两层卷积下采样 4 倍。
训练采用 SpecAugment 数据增强与 Adam 优化器，学习率峰值 5e-4，warmup 10 万步。

## Experiments
数据集：LibriSpeech（960 小时）、AISHELL-1（中文 178 小时）。
LibriSpeech WER：Transformer 基线为 2.5%/5.6%，
Conformer 为 2.1%/4.3%。
AISHELL-1 字错误率（CER）为 4.5%。
消融显示，去掉卷积模块后 WER 恶化至 2.4%/5.0%；
卷积核从 31 减到 3 时 WER 上升 0.2 个点。

## Conclusion
卷积与自注意力的互补组合在语音任务上优于任一单独模块，
Conformer 已成为语音识别的主流编码器结构。
