# 检索增强生成在科学问答中的应用评测

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
