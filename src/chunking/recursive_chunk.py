# -*- coding: utf-8 -*-
"""
模块一 5.1.2 策略二：递归字符切分（Recursive Character Splitting）

手写实现 LangChain RecursiveCharacterTextSplitter 的核心逻辑（不依赖框架），
便于理解"按分隔符层级递归降级"的思想：

    段落 \n\n  ->  换行 \n  ->  中文句号 。  ->  空格  ->  字符

流程：先用高层分隔符切，若某段仍超长，则递归用下一层分隔符再切，
最后把小于 chunk_size 的相邻片段做贪心合并（保证块尽量"装满"，减少碎片）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseChunker

# 分隔符优先级：从"语义边界最强"到"最弱"
DEFAULT_SEPARATORS = [
    "\n\n",                      # 段落
    "\n",                        # 换行
    "。", "！", "？", ". ", "! ", "? ", "；", ";",   # 句子
    "，", ",", " ",              # 短语 / 词
    "",                          # 兜底：按字符硬切
]


class RecursiveCharacterChunker(BaseChunker):
    """递归字符分块器（工业界默认方案，本项目默认策略）。"""

    name = "recursive"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 min_chunk_size: int = 80, separators: Optional[List[str]] = None,
                 keep_separator: bool = True, **kwargs):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, **kwargs)
        self.separators = separators or DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

    # ---------------- 主入口 ----------------
    def split_text(self, text: str, doc_meta: Optional[Dict[str, Any]] = None) -> List[str]:
        if not text:
            return []
        raw = self._recursive_split(text, self.separators)
        merged = self._merge_splits(raw)
        return [m for m in merged if m.strip()]

    # ---------------- 递归切分 ----------------
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """按分隔符层级递归切分，直到所有片段都 <= chunk_size。"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for i, sep in enumerate(separators):
            if sep == "":
                break  # 最后一层：按字符硬切
            if sep not in text:
                continue
            parts = self._split_by_sep(text, sep)
            if len(parts) == 1:
                continue  # 该分隔符没起作用，换下一层
            # 对仍然超长的片段，用更细的分隔符继续递归
            result: List[str] = []
            for p in parts:
                if len(p) > self.chunk_size:
                    result.extend(self._recursive_split(p, separators[i + 1:]))
                elif p.strip():
                    result.append(p)
            if result:
                return result

        # 兜底：按 chunk_size 硬切
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _split_by_sep(self, text: str, sep: str) -> List[str]:
        if self.keep_separator:
            parts = text.split(sep)
            out = [p + sep for p in parts[:-1]] + [parts[-1]]
            return [p for p in out if p]
        return [p for p in text.split(sep) if p]

    # ---------------- 贪心合并 ----------------
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        把小片段贪心合并到接近 chunk_size，并维护 chunk_overlap 的重叠。
        这是递归切分"好用"的关键：切完之后还要再拼回来。
        """
        if not splits:
            return []
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        for s in splits:
            s_len = len(s)
            if s_len > self.chunk_size:
                # 单段本身就超长：先落盘当前缓冲，再硬切该段
                if cur:
                    chunks.append("".join(cur))
                    cur, cur_len = [], 0
                chunks.extend(self._hard_split(s))
                continue
            if cur_len + s_len > self.chunk_size and cur:
                chunk_text = "".join(cur)
                chunks.append(chunk_text)
                # 保留尾部 overlap：把末尾内容按字符数回退带入下一块
                cur, cur_len = self._overlap_tail(chunk_text)
            cur.append(s)
            cur_len += s_len

        if cur:
            chunks.append("".join(cur))
        return chunks

    def _overlap_tail(self, text: str):
        """计算重叠窗口的初始内容。"""
        if self.chunk_overlap <= 0:
            return [], 0
        tail = text[-self.chunk_overlap:]
        return [tail], len(tail)

    def _hard_split(self, text: str) -> List[str]:
        step = max(self.chunk_size - self.chunk_overlap, 1)
        return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]
