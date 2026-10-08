# -*- coding: utf-8 -*-
"""
生成【同研究方向】评测语料：Transformer 系列方法（10 篇）

为什么需要"同方向"语料：
  课程要求"选择一个具体研究方向，收集 10-20 篇相关论文"。
  若语料分属完全不同领域（如扩散分割 vs 芯片布局），主题差异过大，
  检索几乎必然命中，Hit@5/MRR 全部饱和为 1.0，无法体现
  混合检索与重排序的真实增益。
  同一方向（都是 Transformer 变体）的论文术语高度重叠，
  才是对检索系统真正的考验，也让评测数据有说服力。

用法：python scripts/make_transformer_papers.py
"""
from __future__ import annotations

from pathlib import Path

PAPERS = {
    "ViT_Image_Classification.md": """# Vision Transformer 在图像分类中的应用

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
""",
    "Swin_Transformer_Hierarchical.md": """# Swin Transformer：层次化视觉骨干网络

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
""",
    "DETR_Object_Detection.md": """# DETR：基于 Transformer 的端到端目标检测

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
""",
    "TransUNet_Medical_Seg.md": """# TransUNet：Transformer 与 U-Net 结合的医学图像分割

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
""",
    "TimeSformer_Video.md": """# TimeSformer：时空注意力视频理解模型

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
""",
    "Conformer_Speech.md": """# Conformer：卷积增强的语音识别 Transformer

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
""",
    "ViLT_Multimodal.md": """# ViLT：极简视觉语言预训练模型

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
""",
    "PointTransformer_3D.md": """# Point Transformer：三维点云分析

## Abstract
本文提出 Point Transformer，将自注意力引入三维点云处理，
设计适合无序点集的局部注意力机制。
在 S3DIS 语义分割上 mIoU 达到 70.4%，在 ModelNet40 分类上达到 93.7%。

## Method
点云被表示为无序坐标集合，无法直接套用图像的位置编码。
本文提出向量自注意力：注意力权重由位置编码向量调制，
位置编码为两点坐标差经 MLP 映射得到的向量。

网络为 U-Net 式编码器-解码器结构，5 个 stage，
每 stage 内做最远点采样下采样，邻域大小 k=16。
注意力采用减法关系：gamma = MLP(p_i - p_j)，
相比标量点积注意力在点云上 mIoU 提升 2.1 个点。

训练使用 AdamW，学习率 5e-4，batch size 16，训练 100 个 epoch。

## Experiments
数据集：S3DIS（6 个区域，13 类）、ModelNet40（12311 个 CAD 模型）、
ShapeNetPart（部件分割）。
S3DIS Area5 mIoU：Point Transformer 为 70.4%，
PointNet++ 为 53.5%，KPConv 为 67.1%。
ModelNet40 总体准确率 93.7%。
消融显示，邻域 k 从 16 降到 8 时 mIoU 下降 1.3；
去掉位置编码调制后 mIoU 暴跌至 61.2%。

## Conclusion
向量形式的相对位置注意力是点云 Transformer 的关键，
局部邻域注意力比全局注意力更适合三维数据。
""",
    "SparseTS_TimeSeries.md": """# SparseTS：稀疏注意力长序列时间序列预测

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
""",
    "Graphormer_Molecules.md": """# Graphormer：图 Transformer 用于分子性质预测

## Abstract
本文将 Transformer 适配到图结构数据，提出 Graphormer，
通过结构编码显式注入图的拓扑与空间信息。
在 QM9 的 HOMO 能级预测上 MAE 降至 0.0325 eV，
在 PCQM4M-LSC 大规模基准上取得当时最优结果。

## Method
三项结构编码设计：
1. 中心性编码：按节点度数为每个节点分配可学习向量，注入节点重要性
2. 空间编码：用最短路径距离 D_ij 作为注意力偏置，可学习标量
3. 边编码：沿最短路径对边特征加权平均，加入注意力计算

模型为 12 层 Transformer，隐藏维度 768，32 个注意力头。
virtual node 用于聚合全图表示以做图级预测。
训练使用 Adam，学习率 2e-4，batch size 256，训练 100 万步。

## Experiments
数据集：QM9（13.4 万分子）、PCQM4M-LSC（380 万分子）、
MoleculeNet 的 BBBP 与 HIV。
QM9 HOMO 预测 MAE：GCN 0.0512，GAT 0.0478，
MPNN 0.0401，Graphormer 0.0325 eV。
BBBP 的 ROC-AUC 为 0.934。
消融显示，去掉空间编码后 MAE 恶化至 0.0455 eV；
去掉边编码后恶化 0.0031 eV。

## Conclusion
结构编码是图 Transformer 成功的关键，
空间（距离）信息对量子化学性质预测贡献最大。
""",
}


def generate(output_dir: str = "./data/papers_transformer") -> list:
    """生成同方向语料，返回文件路径列表。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in PAPERS.items():
        p = out / name
        p.write_text(content, encoding="utf-8")
        paths.append(str(p))
    print(f"已生成 {len(paths)} 篇 Transformer 方向论文到 {out.resolve()}")
    return paths


if __name__ == "__main__":
    generate()
