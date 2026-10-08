# -*- coding: utf-8 -*-
"""
模块一 5.1.1：纯文本加载器（.txt / .md / .markdown / .jsonl）

特点：
  - 自动编码探测（utf-8 -> gbk -> latin-1），解决中文 txt 乱码问题
  - Markdown 保留标题层级（# / ## / ###），供语义分块识别章节边界
  - JSONL 视为"一行一篇"，适合批量导入摘要数据
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import BaseLoader, Document, Page, build_metadata, make_doc_id

try:
    from ..utils.logger import get_logger

    logger = get_logger("research_assistant.loader.text")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("text_loader")


class TextLoader(BaseLoader):
    """TXT / Markdown 加载器。"""

    supported_exts = ["txt", "md", "markdown"]

    def __init__(self, clean: bool = True, keep_formula: bool = True,
                 encoding: Optional[str] = None, split_lines: int = 0):
        """
        split_lines: >0 时每 N 行切成一个逻辑页（便于长文本定位引用位置）
        """
        super().__init__(clean=clean, keep_formula=keep_formula)
        self.encoding = encoding
        self.split_lines = split_lines

    def load(self, path: str) -> Document:
        raw = self._read(path)
        ext = Path(path).suffix.lower().lstrip(".")
        pages = self._to_pages(raw)
        pages = self._postprocess(pages)
        meta = build_metadata(path, {"parser": "text", "encoding": self.encoding or "auto"})
        return Document(
            doc_id=make_doc_id(path),
            title=self._guess_title(raw, path),
            source=str(Path(path).resolve()),
            file_type=ext if ext else "txt",
            pages=pages,
            metadata=meta,
        )

    def _read(self, path: str) -> str:
        if self.encoding:
            with open(path, "r", encoding=self.encoding, errors="ignore") as f:
                return f.read()
        # 编码探测顺序：utf-8 -> utf-8-sig -> gbk -> gb18030 -> latin-1
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                self.encoding = enc
                return content
            except (UnicodeDecodeError, LookupError):
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _to_pages(self, text: str) -> List[Page]:
        if self.split_lines and self.split_lines > 0:
            lines = text.split("\n")
            pages = []
            for i in range(0, len(lines), self.split_lines):
                chunk = "\n".join(lines[i:i + self.split_lines])
                pages.append(Page(page=i // self.split_lines + 1, text=chunk))
            return pages
        return [Page(page=1, text=text)]

    @staticmethod
    def _guess_title(text: str, path: str) -> str:
        for line in text.strip().split("\n")[:10]:
            line = line.strip().lstrip("#").strip()
            if len(line) > 4:
                return line[:200]
        return Path(path).stem


class MarkdownLoader(TextLoader):
    """Markdown 加载器：额外抽取标题结构注入 metadata，供学术分块使用。"""

    supported_exts = ["md", "markdown"]

    def load(self, path: str) -> Document:
        doc = super().load(path)
        headings = [l.strip() for l in doc.full_text.split("\n") if l.strip().startswith("#")]
        doc.metadata["headings"] = headings[:30]
        doc.metadata["parser"] = "markdown"
        return doc


def load_text(path: str, **kwargs) -> Document:
    """便捷函数。"""
    return TextLoader(**kwargs).load(path)
