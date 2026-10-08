# -*- coding: utf-8 -*-
"""
检索层统一数据模型与基类
------------------------
RetrievedChunk 是"检索层 -> 生成层"之间流转的标准结构，
携带完整的打分链路（向量分 / BM25分 / RRF分 / 精排分），
便于模块四做可观测性展示与模块五做 Bad Case 归因。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..chunking.base import Chunk


@dataclass
class RetrievedChunk:
    """检索结果（含多级打分明细）。"""
    chunk: Chunk
    score: float = 0.0              # 最终用于排序的分数
    rank: int = 0                   # 最终排名（从 1 开始）
    source: str = ""                # 命中来源: vector / bm25 / both / fusion / rerank
    vector_score: float = 0.0
    vector_rank: int = 0
    bm25_score: float = 0.0
    bm25_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    # 便捷代理，让上层可以像用 Chunk 一样直接用
    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def citation(self) -> str:
        return self.chunk.citation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.chunk.doc_title,
            "page": self.chunk.page,
            "text": self.text[:300],
            "score": round(self.score, 6),
            "source": self.source,
            "vector_score": round(self.vector_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "rerank_score": round(self.rerank_score, 6),
            "citation": self.citation,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RetrievedChunk({self.chunk_id[:12]}, score={self.score:.4f}, "
                f"src={self.source}, doc={self.doc_id[:8]}p{self.chunk.page})")


class BaseRetriever:
    """检索器基类。"""

    name: str = "base"

    def build(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError

    def add(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> int:
        raise NotImplementedError

    @property
    def size(self) -> int:
        return 0


def reciprocal_rank_fusion(rank_lists: List[List[str]], k: int = 60,
                           weights: Optional[List[float]] = None) -> Dict[str, float]:
    """
    RRF（Reciprocal Rank Fusion）倒数排名融合。

        score(d) = Σ_i  w_i / (k + rank_i(d))

    为什么用 RRF 而不是分数加权：
      向量相似度（0~1 余弦）与 BM25 分数（0~20+）量纲不同，直接加权需要归一化
      且对参数敏感；RRF 只用"名次"，对量纲不敏感，鲁棒性更好，是业界主流做法。

    :param rank_lists: 多个检索器返回的 chunk_id 有序列表（越靠前排名越高）
    :param k:          平滑系数，默认 60（原始论文经验值）
    :param weights:    各路检索器的权重
    """
    weights = weights or [1.0] * len(rank_lists)
    fused: Dict[str, float] = {}
    for w, rank_list in zip(weights, rank_lists):
        for rank, cid in enumerate(rank_list, start=1):
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank)
    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))


def dedup_by_text(chunks: List[RetrievedChunk], threshold: float = 0.9) -> List[RetrievedChunk]:
    """按文本相似度去重（同一段落被多次召回时只保留分数最高的）。"""
    seen: Dict[str, RetrievedChunk] = {}
    out: List[RetrievedChunk] = []
    for c in chunks:
        key = c.text.strip()[:120]
        if key in seen:
            continue
        seen[key] = c
        out.append(c)
    return out
