# -*- coding: utf-8 -*-
"""
模块一：文档加载器基类与统一数据模型
------------------------------------
统一数据模型（全系统流通的标准结构）：
    Document  ->  Page  ->  (分块后) Chunk  ->  (检索后) RetrievedChunk

设计原则：加载层只负责"把文件变成 Document"，不做分块、不做清洗决策，
保证职责单一，便于后续替换 / 测试。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Page:
    """一页（或一段）文本。非 PDF 文档统一 page=1。"""
    page: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Document:
    """统一文档模型。"""
    doc_id: str
    title: str
    source: str                      # 原始文件路径
    file_type: str                   # pdf / docx / txt / md
    pages: List[Page]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -------- 便捷属性 --------
    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.pages)

    def page_text(self, page_no: int) -> str:
        for p in self.pages:
            if p.page == page_no:
                return p.text
        return ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        pages = [Page(**p) for p in d.get("pages", [])]
        return cls(
            doc_id=d["doc_id"], title=d.get("title", ""), source=d.get("source", ""),
            file_type=d.get("file_type", ""), pages=pages, metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Document(id={self.doc_id[:8]}, title={self.title!r}, pages={self.num_pages}, chars={self.char_count})"


def make_doc_id(path: str) -> str:
    """基于绝对路径 + 文件大小生成稳定 doc_id（同一文件重复导入不会重复入库）。"""
    ap = os.path.abspath(path)
    size = os.path.getsize(ap) if os.path.exists(ap) else 0
    raw = f"{ap}|{size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------
# 文本清洗（学术论文场景）
# ------------------------------------------------------------------
_LIGATURE_MAP = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
    "…": "...", " ": " ",
}


def clean_text(text: str, keep_formula: bool = True) -> str:
    """
    学术论文文本清洗：
      1. PDF 连字符 / 换行粘连修复
      2. 全角半角与特殊连字还原
      3. 页眉页脚噪声（纯数字行、"arXiv:xxxx" 等）剔除
      4. 多余空白压缩（可选保留公式块）
    """
    if not text:
        return ""
    for k, v in _LIGATURE_MAP.items():
        text = text.replace(k, v)
    # 单词内换行粘连：trans-\nformer -> transformer
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 单行换行 -> 空格（段落内换行），保留空行作为段落分隔
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 去除页眉页脚类噪声行
    noise_patterns = [
        r"^\s*\d{1,4}\s*$",                              # 纯页码
        r"^arXiv:\d{4}\.\d{4,5}.*$",                     # arXiv 水印
        r"^(Preprint|Under review|Accepted at).{0,60}$",  # 预印本水印
    ]
    lines = []
    for line in text.split("\n"):
        if any(re.match(p, line.strip(), re.IGNORECASE) for p in noise_patterns):
            continue
        lines.append(line)
    text = "\n".join(lines)
    # 压缩连续空格（不破坏公式中的多空格排版）
    if keep_formula:
        text = re.sub(r"[ \t]{3,}", "  ", text)
    else:
        text = re.sub(r"[ \t]{2,}", " ", text)
    # 压缩 3 个以上空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------------
# 加载器基类
# ------------------------------------------------------------------
class BaseLoader:
    """文档加载器基类。子类实现 load()。"""

    supported_exts: List[str] = []

    def __init__(self, clean: bool = True, keep_formula: bool = True):
        self.clean = clean
        self.keep_formula = keep_formula

    def load(self, path: str) -> Document:
        raise NotImplementedError

    def _postprocess(self, pages: List[Page]) -> List[Page]:
        """清洗 + 丢弃空页。"""
        out: List[Page] = []
        for p in pages:
            text = clean_text(p.text, self.keep_formula) if self.clean else p.text
            if text.strip():
                out.append(Page(page=p.page, text=text, metadata=p.metadata))
        return out

    def can_load(self, path: str) -> bool:
        return Path(path).suffix.lower().lstrip(".") in self.supported_exts


class LoaderResult:
    """批量导入的单条结果（状态追踪用）。"""

    def __init__(self, path: str, success: bool, doc: Optional[Document] = None,
                 error: str = "", skipped: bool = False):
        self.path = path
        self.success = success
        self.doc = doc
        self.error = error
        self.skipped = skipped

    def __repr__(self) -> str:  # pragma: no cover
        if self.skipped:
            return f"[SKIP] {self.path}"
        if self.success:
            return f"[OK]   {self.path} -> {self.doc}"
        return f"[FAIL] {self.path} : {self.error}"


def build_metadata(path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """生成文档元信息（导入时间 / 文件大小 / 后缀）。"""
    p = Path(path)
    meta: Dict[str, Any] = {
        "file_name": p.name,
        "file_size": p.stat().st_size if p.exists() else 0,
        "file_ext": p.suffix.lower().lstrip("."),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    return meta
