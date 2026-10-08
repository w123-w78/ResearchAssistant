# -*- coding: utf-8 -*-
"""
文本工具：Token 估算 / 关键词抽取 / 句子切分
（纯函数，无任何模型依赖）
"""
from __future__ import annotations

import re
from typing import Dict, List

_TIKTOKEN = None

# 中文停用词（关键词抽取时用）
STOPWORDS = {
    "这个", "那个", "什么", "如何", "为什么", "哪些", "请问", "一下", "可以", "我们", "你们",
    "有没有", "是不是", "怎么", "多少", "以及", "关于", "对于", "一个", "没有", "就是",
    "the", "and", "for", "with", "that", "this", "from", "what", "how", "why", "are", "you",
}


def estimate_tokens(text: str) -> int:
    """
    Token 估算（中英混排）。
    优先使用 tiktoken(cl100k_base)；不可用时用启发式：
      中文 1 字 ≈ 0.7 token，英文 1 词 ≈ 1.3 token
    """
    global _TIKTOKEN
    if _TIKTOKEN is None:
        try:
            import tiktoken  # type: ignore

            _TIKTOKEN = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _TIKTOKEN = False
    if _TIKTOKEN:
        try:
            return len(_TIKTOKEN.encode(text))
        except Exception:  # noqa: BLE001
            pass
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]+", text))
    return max(1, int(zh * 0.7 + en * 1.3))


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """
    轻量关键词抽取（中文 2-6 字词组 + 英文单词，按词频排序）。

    用途：keyword_extract 工具、Agent 查询改写前的关键词识别。
    """
    en = re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text)
    zh = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    freq: Dict[str, int] = {}
    for w in en:
        w = w.lower()
        if len(w) > 2 and w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    for w in zh:
        if 2 <= len(w) <= 6 and w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
    return [w for w, _ in items[:top_k]]


def split_sentences(text: str) -> List[str]:
    """中英文句子切分。"""
    if not text.strip():
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p and p.strip()]


def truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """按字符截断（保留语义完整性的简单版本）。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix
