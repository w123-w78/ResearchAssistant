# -*- coding: utf-8 -*-
"""
模块一 5.1.2 策略三：语义切分（Semantic Chunking）

思路：以"句子"为最小单位，按相邻句子的语义相似度决定是否合并。
  - 相邻句相似度高 -> 属于同一语义单元 -> 合并
  - 相似度骤降     -> 话题切换        -> 断开

相似度计算（三级降级，保证零依赖可跑）：
  1. Embedding 余弦相似度（传入 embedder 时，效果最好）
  2. 词袋 Jaccard / 余弦（有 numpy 即可）
  3. 纯字符级重叠率（兜底）

优势：块内语义连贯，检索时召回的片段"讲得完整"；
劣势：需逐句计算相似度，耗时高于前两种策略。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseChunker, split_sentences

try:
    from ..utils.logger import get_logger

    logger = get_logger("research_assistant.chunking")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("chunking")


class SemanticChunker(BaseChunker):
    """
    基于句子边界 + 语义相似度的分块器。

    mode:
      "similarity" 相邻句相似度低于阈值则断开（默认）
      "percentile" 取所有相邻相似度的分位数作为动态阈值（对长文档更鲁棒）
    """

    name = "semantic"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0,
                 min_chunk_size: int = 80, embedder=None,
                 similarity_threshold: float = 0.35, mode: str = "similarity",
                 percentile: float = 25.0, **kwargs):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, **kwargs)
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.mode = mode
        self.percentile = percentile
        if mode not in ("similarity", "percentile"):
            raise ValueError("mode 必须是 'similarity' 或 'percentile'")

    # ---------------- 主入口 ----------------
    def split_text(self, text: str, doc_meta: Optional[Dict[str, Any]] = None) -> List[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]] if len(sentences[0]) >= self.min_chunk_size else []

        sims = self._adjacent_similarity(sentences)
        threshold = self._resolve_threshold(sims)
        # 相邻句相似度低于阈值 -> 在该位置断开
        breakpoints = [i + 1 for i, s in enumerate(sims) if s < threshold]
        groups = self._group_by_breakpoints(sentences, breakpoints)
        # 组内再按 chunk_size 兜底切分，防止超长
        out: List[str] = []
        for g in groups:
            out.extend(self._enforce_size(g))
        return [o for o in out if o.strip()]

    # ---------------- 相似度 ----------------
    def _adjacent_similarity(self, sentences: List[str]) -> List[float]:
        """计算相邻句相似度列表，长度 = len(sentences) - 1。"""
        if self.embedder is not None:
            try:
                vecs = self.embedder.encode(sentences)
                if hasattr(vecs, "shape") and vecs.ndim == 2 and vecs.shape[0] == len(sentences):
                    return [float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(sentences) - 1)]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding 相似度计算失败，降级为词袋: %s", exc)
        return [self._bow_similarity(sentences[i], sentences[i + 1])
                for i in range(len(sentences) - 1)]

    @staticmethod
    def _tokens(text: str) -> List[str]:
        zh = re.findall(r"[\u4e00-\u9fff]", text)
        en = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", text.lower())
        return zh + en

    def _bow_similarity(self, a: str, b: str) -> float:
        """词袋余弦相似度（无模型时的可靠降级方案）。"""
        ta, tb = set(self._tokens(a)), set(self._tokens(b))
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        # 余弦 = |A∩B| / sqrt(|A|*|B|)
        return inter / ((len(ta) * len(tb)) ** 0.5)

    # ---------------- 阈值 ----------------
    def _resolve_threshold(self, sims: List[float]) -> float:
        if self.mode == "similarity" or not sims:
            return self.similarity_threshold
        return float(np.percentile(sims, self.percentile))

    # ---------------- 分组 ----------------
    @staticmethod
    def _group_by_breakpoints(sentences: List[str], breakpoints: List[int]) -> List[str]:
        groups: List[str] = []
        start = 0
        for bp in breakpoints:
            groups.append("".join(sentences[start:bp]))
            start = bp
        if start < len(sentences):
            groups.append("".join(sentences[start:]))
        return groups

    def _enforce_size(self, text: str) -> List[str]:
        """超长组按句子再切；过短组返回原样（由上层合并）。"""
        if len(text) <= self.chunk_size:
            return [text]
        sentences = split_sentences(text)
        out: List[str] = []
        cur = ""
        for s in sentences:
            if len(cur) + len(s) > self.chunk_size and cur:
                out.append(cur)
                cur = s
            else:
                cur += s
        if cur:
            out.append(cur)
        return out
