# -*- coding: utf-8 -*-
"""
模块一：文本分块策略统一入口

提供 4 种切分策略 + 对比实验工具（产出《分块策略对比实验报告》）
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .academic import AcademicChunker
from .base import BaseChunker, Chunk, split_paragraphs, split_sentences
from .fixed_chunk import FixedSizeChunker
from .recursive_chunk import RecursiveCharacterChunker
from .semantic_chunk import SemanticChunker

try:
    from ..utils.logger import get_logger

    logger = get_logger("research_assistant.chunking")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("chunking")

__all__ = [
    "Chunk", "BaseChunker", "FixedSizeChunker", "RecursiveCharacterChunker",
    "SemanticChunker", "AcademicChunker", "get_chunker", "list_chunkers",
    "compare_chunkers", "split_sentences", "split_paragraphs",
]

# 策略名 -> 类
_CHUNKER_REGISTRY: Dict[str, type] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveCharacterChunker,
    "semantic": SemanticChunker,
    "academic": AcademicChunker,
}


def list_chunkers() -> List[str]:
    return list(_CHUNKER_REGISTRY)


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """按名称构建分块器。

    strategy: fixed | recursive | semantic | academic
    """
    name = (strategy or "recursive").lower()
    if name not in _CHUNKER_REGISTRY:
        raise ValueError(f"未知分块策略: {name}，可选 {list(_CHUNKER_REGISTRY)}")
    chunker = _CHUNKER_REGISTRY[name](**kwargs)
    logger.info("初始化分块器: %s(size=%s, overlap=%s)",
                name, chunker.chunk_size, chunker.chunk_overlap)
    return chunker


def compare_chunkers(docs: List[Any],
                     strategies: Optional[List[str]] = None,
                     chunk_sizes: Optional[List[int]] = None,
                     **common_kwargs) -> List[Dict[str, Any]]:
    """
    5.1.2 要求的分块策略对比实验。

    统计每个 (策略, chunk_size) 组合：
      - 块数、平均/最大/最小块长、耗时
      - 过短块占比（< min_chunk_size，衡量碎片化程度）
      - 块长标准差（衡量长度均匀性）

    返回结构化结果，可直接转 DataFrame 出图，填入《分块策略对比实验报告》。
    """
    strategies = strategies or ["fixed", "recursive", "semantic", "academic"]
    chunk_sizes = chunk_sizes or [512]

    rows: List[Dict[str, Any]] = []
    for strategy in strategies:
        for size in chunk_sizes:
            try:
                chunker = get_chunker(strategy, chunk_size=size,
                                      chunk_overlap=min(64, size // 8), **common_kwargs)
            except Exception as exc:  # noqa: BLE001
                rows.append({"strategy": strategy, "chunk_size": size, "error": str(exc)})
                continue
            start = time.perf_counter()
            chunks = chunker.split_documents(docs)
            elapsed = time.perf_counter() - start
            if not chunks:
                rows.append({"strategy": strategy, "chunk_size": size, "num_chunks": 0,
                             "error": "未产生任何块"})
                continue
            lens = [c.char_count for c in chunks]
            mean_len = sum(lens) / len(lens)
            short_ratio = sum(1 for l in lens if l < chunker.min_chunk_size) / len(lens)
            std = (sum((l - mean_len) ** 2 for l in lens) / len(lens)) ** 0.5
            rows.append({
                "strategy": strategy,
                "chunk_size": size,
                "num_chunks": len(chunks),
                "avg_len": round(mean_len, 1),
                "max_len": max(lens),
                "min_len": min(lens),
                "std_len": round(std, 1),
                "short_ratio": round(short_ratio, 4),
                "elapsed_s": round(elapsed, 4),
                "avg_tokens": round(sum(c.token_count for c in chunks) / len(chunks), 1),
            })
    return rows


def merge_short_chunks(chunks: List[Chunk], min_size: int = 80) -> List[Chunk]:
    """后处理：把同一文档内的过短块合并到相邻块，降低碎片率。"""
    if not chunks:
        return []
    out: List[Chunk] = [chunks[0]]
    for c in chunks[1:]:
        prev = out[-1]
        if len(c.text) < min_size and prev.doc_id == c.doc_id and prev.page == c.page:
            prev.text = prev.text + "\n" + c.text
            prev.end_char = c.end_char
            prev.metadata["merged"] = True
        else:
            out.append(c)
    return out
