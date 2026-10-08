# -*- coding: utf-8 -*-
"""
模块一 5.1.2：文本分块基类与统一 Chunk 模型
-------------------------------------------
Chunk 是检索层的最小单位，也是引用溯源的最小单位（保留 doc_id + page）。

分块策略对比（学术论文场景经验值）：
  固定大小   速度最快，实现最简单；缺点是切断句子/公式，检索召回率最低
  递归字符   优先按段落→句子→词递归切分，语义完整性好；工业界默认方案
  语义切分   按句子边界 + 相邻句相似度合并，语义最连贯；耗时中等
  学术分块   识别 Abstract/Method/Experiment 章节，公式与表格不切断；论文场景最优
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..utils.text_utils import estimate_tokens


@dataclass
class Chunk:
    """统一文本块模型。"""
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    page: int = 1
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    source: str = ""            # 原始文件路径
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def token_count(self) -> int:
        if "token_count" not in self.metadata:
            self.metadata["token_count"] = estimate_tokens(self.text)
        return int(self.metadata["token_count"])

    @property
    def citation(self) -> str:
        """引用标识，如：Attention Is All You Need (p.3)"""
        title = self.doc_title or self.doc_id[:8]
        if len(title) > 40:
            title = title[:40] + "..."
        return f"{title} (p.{self.page})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(**d)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Chunk({self.chunk_id[:10]}, doc={self.doc_id[:8]}, p{self.page}, {self.char_count}字)"


class BaseChunker:
    """分块器基类。子类实现 split_document(doc) / split_text(text)。"""

    name: str = "base"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 min_chunk_size: int = 80, **kwargs):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.extra = kwargs

    # -------- 子类实现 --------
    def split_text(self, text: str, doc_meta: Optional[Dict[str, Any]] = None) -> List[str]:
        raise NotImplementedError

    # -------- 对外接口 --------
    def split_document(self, doc) -> List[Chunk]:
        """把 Document 切成 Chunk 列表（保留页码与字符偏移）。"""
        from ..data_loader.base import Document  # 延迟导入避免循环

        if not isinstance(doc, Document):
            raise TypeError("split_document 需要传入 Document 对象")
        chunks: List[Chunk] = []
        global_idx = 0
        for page in doc.pages:
            if not page.text.strip():
                continue
            pieces = self.split_text(page.text, doc_meta=doc.metadata)
            offset = 0
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                # 逐块推进偏移量定位字符位置（不回退，避免重复命中同一子串）
                start = page.text.find(piece[:30], offset) if len(piece) >= 30 else offset
                if start < 0:
                    start = offset
                cid = f"{doc.doc_id}#{global_idx}"
                chunks.append(Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=piece,
                    page=page.page,
                    start_char=start,
                    end_char=start + len(piece),
                    chunk_index=global_idx,
                    source=doc.source,
                    metadata={
                        "file_name": doc.metadata.get("file_name", ""),
                        "chunker": self.name,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    },
                ))
                global_idx += 1
                offset = start + max(len(piece) - self.chunk_overlap, 1)
        return chunks

    def split_documents(self, docs) -> List[Chunk]:
        out: List[Chunk] = []
        for d in docs:
            out.extend(self.split_document(d))
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(size={self.chunk_size}, overlap={self.chunk_overlap})"


# ------------------------------------------------------------------
# 通用句子切分（中英混排）
# ------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> List[str]:
    """中英文句子切分：优先中文句号/问号/感叹号，兼容英文句点。"""
    if not text.strip():
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def split_paragraphs(text: str) -> List[str]:
    """段落切分（保留 Markdown 标题独立成段）。"""
    if not text.strip():
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
