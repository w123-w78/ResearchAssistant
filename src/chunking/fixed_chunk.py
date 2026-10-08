# -*- coding: utf-8 -*-
"""
模块一 5.1.2 策略一：固定大小切分（Fixed-size Chunking）

最朴素的基线方案：按固定字符数滑窗切分。
用途：作为对比实验的 baseline，验证"语义边界感知"带来的收益。

优点：O(n) 速度、长度严格可控、便于 batch 向量化
缺点：会在句子甚至单词中间切断，破坏语义完整性
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """固定大小 + 重叠滑窗切分。"""

    name = "fixed"

    def split_text(self, text: str, doc_meta: Optional[Dict[str, Any]] = None) -> List[str]:
        if not text:
            return []
        size, overlap = self.chunk_size, self.chunk_overlap
        step = size - overlap
        pieces: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            piece = text[start:end]
            # 若不是最后一块，尽量回退到最近的空白，减少单词被切断
            if end < n:
                last_space = piece.rfind(" ")
                if last_space > size * 0.6:
                    end = start + last_space + 1
                    piece = text[start:end]
            if piece.strip():
                pieces.append(piece)
            if end >= n:
                break
            start += step
        return self._merge_tiny(pieces)

    def _merge_tiny(self, pieces: List[str]) -> List[str]:
        """把过短碎片合并到上一块，避免出现 20 字的"孤儿块"。"""
        merged: List[str] = []
        for p in pieces:
            if merged and len(p) < self.min_chunk_size:
                merged[-1] = merged[-1] + p
            else:
                merged.append(p)
        return merged
