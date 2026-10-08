# -*- coding: utf-8 -*-
"""
模块一：文档处理与检索层 —— 加载器统一入口

对外只暴露三个函数/类：
    load_document(path)       单文件加载
    DocumentImporter          批量导入（进度回调 + 失败重试 + 状态追踪）
    SUPPORTED_EXTS            支持的扩展名清单
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import (BaseLoader, Document, LoaderResult, Page,
                   build_metadata, clean_text, make_doc_id)
from .docx_loader import DocxLoader
from .pdf_loader import PDFLoader, compare_pdf_engines
from .text_loader import MarkdownLoader, TextLoader, load_text

try:
    from ..utils.logger import get_logger

    logger = get_logger("research_assistant.loader")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("loader")

__all__ = [
    "Document", "Page", "BaseLoader", "LoaderResult", "PDFLoader", "DocxLoader",
    "TextLoader", "MarkdownLoader", "load_document", "load_directory",
    "DocumentImporter", "compare_pdf_engines", "SUPPORTED_EXTS", "clean_text",
]

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".markdown"}

# 扩展名 -> 加载器类
_LOADER_MAP = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
}


def get_loader(path: str, **kwargs) -> BaseLoader:
    """按扩展名返回对应加载器实例。"""
    ext = Path(path).suffix.lower()
    if ext not in _LOADER_MAP:
        raise ValueError(f"不支持的文件类型: {ext}，当前支持 {sorted(SUPPORTED_EXTS)}")
    return _LOADER_MAP[ext](**kwargs)


def load_document(path: str, **kwargs) -> Document:
    """加载单个文档。"""
    return get_loader(path, **kwargs).load(path)


def load_directory(dir_path: str, recursive: bool = True, **kwargs) -> List[Document]:
    """批量加载目录下所有支持的文件（跳过失败项，不中断整体流程）。"""
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    pattern = "**/*" if recursive else "*"
    docs: List[Document] = []
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            try:
                docs.append(load_document(str(p), **kwargs))
            except Exception as exc:  # noqa: BLE001
                logger.error("加载失败 %s: %s", p, exc)
    return docs


class DocumentImporter:
    """
    批量文档导入器（模块一要求的"批量导入与状态追踪"）。

    能力：
      - 进度回调：on_progress(current, total, path, status)
      - 失败重试：max_retries 次，指数退避
      - 去重：同一文件（路径+大小）已导入则跳过
      - 状态追踪：返回 LoaderResult 列表，含成功/失败/跳过与耗时
    """

    ProgressCallback = Callable[[int, int, str, str], None]

    def __init__(self, max_retries: int = 2, skip_duplicates: bool = True, **loader_kwargs):
        self.max_retries = max_retries
        self.skip_duplicates = skip_duplicates
        self.loader_kwargs = loader_kwargs
        self._seen: set = set()

    def import_files(self, paths: List[str],
                     on_progress: Optional[ProgressCallback] = None) -> List[LoaderResult]:
        results: List[LoaderResult] = []
        total = len(paths)
        for i, path in enumerate(paths, 1):
            # 去重检查
            if self.skip_duplicates:
                key = self._file_key(path)
                if key in self._seen:
                    results.append(LoaderResult(path, False, skipped=True))
                    self._notify(on_progress, i, total, path, "skip")
                    continue
                self._seen.add(key)

            doc, err = self._load_with_retry(path)
            if doc is not None:
                results.append(LoaderResult(path, True, doc=doc))
                self._notify(on_progress, i, total, path, "ok")
            else:
                results.append(LoaderResult(path, False, error=err))
                self._notify(on_progress, i, total, path, "fail")
        return results

    def import_directory(self, dir_path: str, recursive: bool = True,
                         on_progress: Optional[ProgressCallback] = None) -> List[LoaderResult]:
        root = Path(dir_path)
        pattern = "**/*" if recursive else "*"
        paths = [str(p) for p in sorted(root.glob(pattern))
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        return self.import_files(paths, on_progress=on_progress)

    # ---------------- 内部 ----------------
    def _load_with_retry(self, path: str):
        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                start = time.perf_counter()
                doc = load_document(path, **self.loader_kwargs)
                logger.info("导入成功 %s (%.2fs, %s 页)", Path(path).name,
                            time.perf_counter() - start, doc.num_pages)
                return doc, ""
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("导入失败(%s/%s) %s: %s", attempt + 1, self.max_retries + 1, path, last_err)
                time.sleep(min(2 ** attempt, 4))
        return None, last_err

    @staticmethod
    def _file_key(path: str) -> str:
        p = Path(path)
        return f"{p.resolve()}|{p.stat().st_size if p.exists() else 0}"

    @staticmethod
    def _notify(cb: Optional[ProgressCallback], cur: int, total: int, path: str, status: str) -> None:
        if cb:
            try:
                cb(cur, total, path, status)
            except Exception:  # noqa: BLE001 - 回调异常不影响导入主流程
                pass

    @staticmethod
    def summary(results: List[LoaderResult]) -> Dict[str, Any]:
        """汇总导入结果（前端展示 / 报告用）。"""
        ok = [r for r in results if r.success]
        fail = [r for r in results if not r.success and not r.skipped]
        skip = [r for r in results if r.skipped]
        return {
            "total": len(results),
            "success": len(ok),
            "failed": len(fail),
            "skipped": len(skip),
            "documents": [r.doc for r in ok],
            "errors": [{"path": r.path, "error": r.error} for r in fail],
            "total_chars": sum(r.doc.char_count for r in ok if r.doc),
        }
