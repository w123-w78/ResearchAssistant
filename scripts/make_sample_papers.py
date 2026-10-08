# -*- coding: utf-8 -*-
"""
生成示例论文语料（用于演示 / 冒烟测试 / 评测集构建）

之所以内置示例数据：让任何人 clone 仓库后无需准备 PDF 即可一键跑通全链路。
真实使用时，用户上传自己的 PDF/Word 即可覆盖这些示例。
"""
from __future__ import annotations

from pathlib import Path

PAPERS = {
    "Vision_Transformer_Survey.md": """# Vision Transformer 在图像分类中的综述

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
""",
    "Diffusion_Model_Segmentation.md": """# 基于扩散模型的医学图像分割方法

## Abstract
针对医学图像标注数据稀缺的问题，本文提出一种基于条件扩散模型的
半监督分割方法 DiffSeg。该方法在仅有 10% 标注数据的情况下，
在 BraTS2021 数据集上达到 87.3% 的 Dice 系数，接近全监督基线的 89.1%。

## Method
DiffSeg 包含两个阶段：
第一阶段，在全体数据（含无标注）上训练一个条件扩散模型，
学习目标为逐步去噪生成分割掩膜，噪声调度采用余弦 schedule，扩散步数 T=1000。

第二阶段，使用少量标注数据微调分割解码器。
为解决伪标签噪声问题，本文提出不确定性加权策略：
对每个像素的预测方差进行估计，方差大于阈值 0.15 的像素不计入损失。

网络骨架采用 U-Net，编码器使用 Swin Transformer Block，
通道数配置为 [64, 128, 256, 512]。

## Experiments
数据集：BraTS2021（1251 例脑肿瘤 MRI）与 ACDC（150 例心脏 MRI）。
评价指标：Dice 系数与 95% Hausdorff 距离（HD95）。

结果：在 BraTS2021 上，DiffSeg 的 Dice 为 87.3%，HD95 为 4.2mm；
对比方法nnU-Net（全监督）为 89.1% 与 3.8mm；
半监督基线 Mean Teacher 为 82.6% 与 6.1mm。
在 ACDC 数据集上，本文方法 Dice 达到 91.4%。

消融实验表明，不确定性加权贡献了 2.1 个百分点的 Dice 提升。

## Conclusion
DiffSeg 在极低标注比例下仍能保持接近全监督的分割性能，
不确定性加权是伪标签去噪的有效手段。
局限在于扩散采样耗时较长，单张图像推理需 3.2 秒。
""",
    "RAG_for_Scientific_QA.md": """# 检索增强生成在科学问答中的应用评测

## Abstract
本文评测了 RAG（Retrieval-Augmented Generation）在科学文献问答任务上的表现。
我们构建了包含 50 篇论文、800 个问答对的评测集 SciQA-50，
系统比较了朴素 RAG、混合检索 RAG 与 RAG+Agent 三种架构。

## Method
检索层采用双路召回：向量检索使用 bge-large-zh 模型（维度 1024），
关键词检索使用 BM25（k1=1.5, b=0.75）。
两路结果通过 RRF（倒数排名融合，k=60）合并，取 Top-20 候选，
再由 bge-reranker-base 精排至 Top-5。

Agent 层基于 ReAct 范式实现，最大迭代次数 6，
工具集包含知识库检索、论文元信息提取、论文对比、计算器与联网搜索。

## Experiments
评测指标为 Hit@5、MRR 与人工评分（1-5 分）。

结果：
- 朴素 RAG（纯向量）：Hit@5 = 0.66，MRR = 0.51，人工评分 3.4
- 混合检索 RAG：Hit@5 = 0.76，MRR = 0.62，人工评分 3.9
- 混合检索 + Rerank：Hit@5 = 0.85，MRR = 0.71，人工评分 4.3
- RAG + Agent：Hit@5 = 0.85，多步推理类问题准确率提升 21%

Token 消耗方面，启用路由后平均每次问答节省 43% 的 Token。
语义缓存命中率为 18%，命中时延迟从 2.3 秒降至 30 毫秒。

## Conclusion
混合检索与重排序是提升 RAG 效果最有效的两个手段；
Agent 的价值主要体现在需要多步推理的复杂问题上。
""",
    "Federated_Learning_Medical.md": """# 联邦学习在医疗影像诊断中的隐私保护方法

## Abstract
针对医疗数据孤岛与隐私合规问题，本文提出 FedMed，一个面向多中心医疗影像的
联邦学习框架。在 6 家医院、共 12480 例胸部 CT 数据上验证，模型 AUC 达到 0.941，
仅比集中式训练低 0.012，且满足原始数据不出院的要求。

## Method
FedMed 采用 FedAvg 作为基础聚合算法，并引入三项改进：
1. 自适应加权聚合：按各参与方数据量与本地损失动态调整权重
2. 差分隐私：在高斯噪声机制下取隐私预算 epsilon=8.0
3. 特征对齐层：用对比学习缓解各医院设备差异导致的分布偏移

本地模型采用 ResNet-34，每轮本地训练 5 个 epoch，全局通信 60 轮。

## Experiments
数据：6 家医院的胸部 CT，共 12480 例，按 7:1:2 划分训练/验证/测试。
评价指标：AUC、灵敏度、特异度。

结果：FedMed 的 AUC 为 0.941，集中式训练基线为 0.953，
不使用特征对齐的消融版本为 0.918。
在 epsilon=2.0 的强隐私设置下，AUC 降至 0.906。
通信开销相比朴素 FedAvg 减少 37%。

## Conclusion
FedMed 在保护数据隐私的前提下取得了接近集中式的诊断性能，
特征对齐层是缓解跨机构分布差异的关键组件。
""",
    "GNN_Molecular_Property.md": """# 图神经网络用于分子性质预测的研究

## Abstract
本文系统比较了 GCN、GAT、MPNN 与 Graphormer 四类图神经网络在
分子性质预测任务上的表现。在 QM9 与 MoleculeNet 两个基准上评测，
结果表明引入 3D 空间信息的模型在量子化学性质预测上优势明显。

## Method
分子被建模为图：原子为节点，化学键为边。
节点特征包含原子序数、电荷、杂化方式等 11 维特征；
边特征包含键型、是否成环等 4 维特征。

对比的四种架构：
- GCN：谱域卷积，2 层，隐藏维度 128
- GAT：注意力聚合，4 头，隐藏维度 128
- MPNN：消息传递，迭代 3 步
- Graphormer：图 Transformer，6 层，引入空间距离编码

训练使用 Adam 优化器，学习率 1e-3，batch size 64，训练 200 个 epoch。

## Experiments
数据集：QM9（13.4 万分子，12 个量子化学性质）、
MoleculeNet 的 BBBP（2039 个分子，血脑屏障穿透性）与 HIV（41127 个分子）。

在 QM9 的 HOMO 能级预测上，MAE 分别为：
GCN 0.0512 eV、GAT 0.0478 eV、MPNN 0.0401 eV、Graphormer 0.0325 eV。
在 BBBP 分类上，ROC-AUC 分别为 0.881、0.902、0.917、0.934。
消融显示，移除 3D 距离编码后 Graphormer 的 MAE 恶化至 0.0455 eV。

## Conclusion
Graphormer 在两个基准上均取得最优结果，3D 空间信息对
量子化学性质预测贡献最大（贡献约 28% 的性能提升）。
""",
    "LLM_Hallucination_Detection.md": """# 大语言模型幻觉检测与缓解方法综述

## Abstract
本文综述了大语言模型幻觉（hallucination）的检测与缓解方法，
将现有工作分为事实性幻觉与忠实性幻觉两类，并在三个基准数据集上
对比了 9 种检测方法的准确率与开销。

## Method
检测方法分为三类：
1. 基于检索的验证：用外部知识库检索证据，比对生成内容与证据的一致性
2. 基于不确定性的检测：利用 token 概率、熵或语义熵估计模型置信度
3. 基于模型的判别：训练专门的判别器或让模型自我一致性检查

本文采用语义熵作为主要指标：对同一问题采样 k=10 个回答，
聚类后计算语义分布的熵，熵越高表示模型越不确定。

## Experiments
数据集：HaluEval（35000 条）、TruthfulQA（817 条）、
BEGIN（12000 条，含领域知识问答）。
评价指标：检测准确率、F1、单次检测延迟。

结果：基于检索的验证准确率最高（0.847），但需要外部知识库且
单次延迟达 1.8 秒；语义熵方法准确率为 0.791，延迟仅 0.24 秒；
自我一致性检查准确率 0.723，延迟 0.9 秒。
三类方法融合后准确率提升至 0.883。

## Conclusion
单一方法难以兼顾准确率与开销，检索验证 + 不确定性估计的组合
是工程实践中的最佳折中。幻觉缓解的根本手段仍是检索增强生成。
""",
    "FewShot_Object_Detection.md": """# 小样本目标检测中的原型对齐方法

## Abstract
针对目标检测在新类别上标注样本稀缺的问题，本文提出 ProtoAlign 方法。
在 PASCAL VOC 与 MS COCO 的小样本设定下，仅用 10 个标注样本即可
使新类别 AP 达到 41.2，相比微调基线提升 8.7 个点。

## Method
ProtoAlign 包含两个核心模块：
1. 类别原型提取：对支持集中同类目标的特征做掩码平均池化，得到类别原型向量
2. 原型-候选对齐：将候选区域特征与类别原型做余弦相似度匹配，
   相似度高于 0.65 的候选保留为正样本

主干网络为 Faster R-CNN + ResNet-101，原型维度 256。
训练采用两阶段：先在基类上预训练，再在支持集上做 200 次迭代微调。

## Experiments
数据集：PASCAL VOC 2007+2012（20 类，划分 15 基类 / 5 新类）、
MS COCO（80 类，划分 60 基类 / 20 新类）。
评价指标：新类别 AP50 与 AP。

在 VOC 的 10-shot 设定下：微调基线 AP50 为 32.5，ProtoAlign 为 41.2；
在 30-shot 设定下分别为 45.1 与 51.8。
消融实验表明，移除原型对齐模块后 AP50 下降 6.3 个点，
相似度阈值从 0.65 降到 0.5 会导致 AP50 下降 2.1 个点（误召回增多）。

## Conclusion
原型对齐能有效将基类学到的特征迁移到新类别，
在低样本（10-shot）场景下的增益最显著。
""",
    "RL_Chip_Floorplan.md": """# 基于强化学习的芯片布局优化方法

## Abstract
本文将芯片布局（floorplanning）建模为序列决策问题，提出 RL-Floorplan，
用强化学习自动优化宏单元摆放。在 ISPD 2005 基准上，
线长（wirelength）相比模拟退火基线降低 11.4%，且单次求解时间从 3 小时缩短至 8 分钟。

## Method
状态空间：网格化的芯片画布（128×128）+ 当前已放置宏的位置编码。
动作空间：在合法网格位置放置下一个宏单元（离散动作，约 16000 个）。
奖励函数：负的半周长线长（HPWL）减去拥塞惩罚，即 R = -HPWL - 0.3 * Congestion。

策略网络用 PPO 训练，actor 与 critic 共享 5 层卷积骨干，
训练 200 万步，学习率 3e-4，折扣因子 0.99。

## Experiments
基准：ISPD 2005（5 个电路，含 200 到 800 个宏单元）。
对比方法：模拟退火（SA）、力导向布局（FD）、RePlAce。

结果（平均 HPWL，单位 1e6）：SA 为 8.42，FD 为 8.91，RePlAce 为 8.05，
RL-Floorplan 为 7.46，相比 SA 降低 11.4%。
运行时间：SA 约 3 小时，RePlAce 约 25 分钟，RL-Floorplan 约 8 分钟（含训练后推理）。
消融显示，移除拥塞惩罚项后 HPWL 略降至 7.31，但拥塞超标网格数增加 42%。

## Conclusion
强化学习能在保证可布线性的同时显著缩短布局求解时间，
奖励函数中的拥塞项是平衡线长与可布线性的关键。
""",
    "TimeSeries_Transformer.md": """# 面向长序列时间序列预测的高效 Transformer 变体

## Abstract
针对标准 Transformer 在长序列预测中 O(n^2) 复杂度过高的问题，
本文提出 SparseTS，一种稀疏注意力时间序列预测模型。
在 ETT、Electricity 与 Traffic 三个数据集上，预测长度 720 步时
MSE 相比 Informer 降低 19.3%，推理速度提升 2.8 倍。

## Method
SparseTS 的三项设计：
1. 稀疏注意力：只计算每个查询与其 Top-32 相似键的注意力，复杂度降至 O(n log n)
2. 序列分解：将输入拆分为趋势项与季节项，分别建模后相加
3. 蒸馏压缩：每两层之间做一次 2 倍下采样，减少序列长度

输入编码包含标量值投影、位置编码与时间特征（小时、星期、月份）。
损失函数为 MSE，优化器 Adam，学习率 1e-4，训练 100 个 epoch。

## Experiments
数据集：ETTm2（电力变压器温度，69680 条）、Electricity（321 个用户用电量）、
Traffic（862 个传感器路况）。
预测长度：96 / 192 / 336 / 720 四档；评价指标 MSE 与 MAE。

预测长度 720 时 MSE：Informer 0.412，Autoformer 0.387，FEDformer 0.361，
SparseTS 0.292（相比 Informer 降低 19.3%）。
推理耗时：Informer 单次 1.42 秒，SparseTS 0.51 秒。
消融显示，移除序列分解后 MSE 恶化至 0.348。

## Conclusion
稀疏注意力与序列分解的组合在长序列预测上效果最好，
序列分解对长程预测的贡献大于稀疏注意力本身。
""",
    "Multimodal_RemoteSensing.md": """# 多模态大模型在遥感图像解译中的应用

## Abstract
本文探索视觉-语言大模型在遥感图像解译中的迁移能力，提出 RS-Ground，
一个面向遥感场景的指令微调数据集与评测基准。在场景分类、
目标检测与图像描述三类任务上平均准确率达到 82.6%。

## Method
构建 RS-Ground 数据集：收集高分二号与 Sentinel-2 影像共 48 万张，
由领域专家标注生成 12 万条指令数据，覆盖场景分类、目标检测、
变化检测与视觉问答四类任务。

模型基于 LLaVA-1.5 架构，视觉编码器用 CLIP-ViT-L/14，
语言模型用 Vicuna-7B，通过 LoRA（rank=16）在遥感数据上微调，
只训练约 0.1% 的参数量。

## Experiments
评测任务与指标：
- 场景分类（AID 数据集，30 类）：准确率
- 目标检测（DIOR 数据集，20 类）：mAP50
- 图像描述（RSICD 数据集）：BLEU-4 与 CIDEr

结果：场景分类准确率 91.4%（比微调前提升 6.2 个点），
目标检测 mAP50 为 74.8，图像描述 CIDEr 为 1.286。
相比全参数微调，LoRA 只用了 0.1% 参数却保留了 97.3% 的性能。
消融显示，移除指令微调数据后场景分类准确率降至 84.1。

## Conclusion
领域指令微调是让通用多模态模型适配遥感场景的关键，
LoRA 在极低参数开销下即可接近全量微调效果。
""",
}


def generate(output_dir: str = "./data/sample_papers") -> list:
    """生成示例论文文件，返回文件路径列表。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in PAPERS.items():
        p = out / name
        p.write_text(content, encoding="utf-8")
        paths.append(str(p))
    print(f"已生成 {len(paths)} 篇示例论文到 {out.resolve()}")
    return paths


if __name__ == "__main__":
    generate()
