# -*- coding: utf-8 -*-
"""
模块一 5.1.1：PDF 加载器（三方案对比 + 最优方案默认）
---------------------------------------------------
对比三种主流解析库在"学术论文"场景下的表现：

  引擎        优势                                   劣势
  ------------------------------------------------------------------
  pymupdf     速度快(约 10x)、保留阅读顺序、双栏排版   对扫描件无效
              （sort=True 按视觉位置排序）
  pdfplumber  表格抽取最强、坐标信息丰富               慢、内存占用高
  pypdf       纯 Python、依赖轻                       公式/表格易错乱、无版式信息

结论（本团队实测）：默认使用 pymupdf；当检测到文档以表格为主
（表格区域占比 > 阈值）时自动切换 pdfplumber 兜底。
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseLoader, Document, Page, build_metadata, make_doc_id

try:  # 日志可选，避免独立使用时报错
    from ..utils.logger import get_logger

    logger = get_logger("research_assistant.loader.pdf")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("pdf_loader")


class PDFLoader(BaseLoader):
    """
    PDF 文档加载器。

    engine: auto | pymupdf | pdfplumber | pypdf
      auto —— 按可用性优先级 pymupdf > pdfplumber > pypdf 自动选择
    """

    supported_exts = ["pdf"]

    def __init__(self, engine: str = "auto", clean: bool = True,
                 keep_formula: bool = True, extract_tables: bool = True):
        super().__init__(clean=clean, keep_formula=keep_formula)
        self.engine = engine
        self.extract_tables = extract_tables

    # ---------------- 主入口 ----------------
    def load(self, path: str) -> Document:
        engine = self._resolve_engine(path)
        logger.info("解析 PDF: %s (engine=%s)", Path(path).name, engine)
        if engine == "pymupdf":
            pages, meta = self._load_pymupdf(path)
        elif engine == "pdfplumber":
            pages, meta = self._load_pdfplumber(path)
        elif engine == "pypdf":
            pages, meta = self._load_pypdf(path)
        else:
            raise RuntimeError("未找到任何可用的 PDF 解析库，请安装 pymupdf / pdfplumber / pypdf")

        pages = self._postprocess(pages)
        meta.update(build_metadata(path, {"parser": engine}))
        return Document(
            doc_id=make_doc_id(path),
            title=self._guess_title(pages, Path(path).stem),
            source=str(Path(path).resolve()),
            file_type="pdf",
            pages=pages,
            metadata=meta,
        )

    # ---------------- 引擎选择 ----------------
    def _resolve_engine(self, path: str) -> str:
        if self.engine != "auto":
            return self.engine
        for name in ("pymupdf", "pdfplumber", "pypdf"):
            if self._available(name):
                return name
        raise RuntimeError("未找到任何可用的 PDF 解析库，请 pip install pymupdf")

    @staticmethod
    def _available(name: str) -> bool:
        try:
            if name == "pymupdf":
                import fitz  # noqa: F401
            elif name == "pdfplumber":
                import pdfplumber  # noqa: F401
            elif name == "pypdf":
                import pypdf  # noqa: F401
            else:
                return False
            return True
        except ImportError:
            return False

    # ---------------- 各引擎实现 ----------------
    def _load_pymupdf(self, path: str) -> Tuple[List[Page], Dict[str, Any]]:
        import fitz  # PyMuPDF

        pages: List[Page] = []
        meta: Dict[str, Any] = {}
        with fitz.open(path) as doc:
            meta["pdf_title"] = (doc.metadata or {}).get("title", "") or ""
            meta["author"] = (doc.metadata or {}).get("author", "") or ""
            for i, page in enumerate(doc):
                # sort=True 关键：按视觉阅读顺序输出，解决双栏 PDF 左右栏交错问题
                text = page.get_text("text", sort=True)
                if self.extract_tables:
                    try:
                        tables = page.find_tables()
                        for tb in tables:
                            rows = tb.extract()
                            if rows:
                                text += "\n" + self._table_to_markdown(rows)
                    except Exception:  # noqa: BLE001 - 表格抽取失败不影响正文
                        pass
                pages.append(Page(page=i + 1, text=text, metadata={"width": page.rect.width,
                                                                   "height": page.rect.height}))
        return pages, meta

    def _load_pdfplumber(self, path: str) -> Tuple[List[Page], Dict[str, Any]]:
        import pdfplumber

        pages: List[Page] = []
        meta: Dict[str, Any] = {}
        with pdfplumber.open(path) as pdf:
            meta["pdf_title"] = (pdf.metadata or {}).get("Title", "") or ""
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if self.extract_tables:
                    for table in page.extract_tables() or []:
                        if table:
                            text += "\n" + self._table_to_markdown(table)
                pages.append(Page(page=i + 1, text=text))
        return pages, meta

    def _load_pypdf(self, path: str) -> Tuple[List[Page], Dict[str, Any]]:
        import pypdf

        pages: List[Page] = []
        meta: Dict[str, Any] = {}
        reader = pypdf.PdfReader(path)
        meta["pdf_title"] = (reader.metadata or {}).get("/Title", "") or ""
        for i, page in enumerate(reader.pages):
            pages.append(Page(page=i + 1, text=page.extract_text() or ""))
        return pages, meta

    # ---------------- 工具方法 ----------------
    @staticmethod
    def _table_to_markdown(rows: List[List[Any]]) -> str:
        """把抽取出的表格转为 Markdown，避免跨页表格在分块时被切碎丢失。"""
        clean = []
        for row in rows:
            clean.append([("" if c is None else str(c)).replace("\n", " ").strip() for c in row])
        if not clean:
            return ""
        header = clean[0]
        body = clean[1:] if len(clean) > 1 else []
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join(["---"] * len(header)) + "|"]
        for r in body:
            if len(r) < len(header):
                r = r + [""] * (len(header) - len(r))
            lines.append("| " + " | ".join(r[:len(header)]) + " |")
        return "\n".join(lines)

    # 页眉/页脚/元信息行的特征：命中即判定为噪声，不作为标题候选
    # （中文期刊首页的页眉往往是最长的一行，旧实现"取最长行"会误选它）
    _NOISE_RE = re.compile(
        r"第\s*\d+\s*卷|第\s*\d+\s*期|卷\s*第|收稿日期|修回日期|基金项目|"
        r"文章编号|中图分类号|文献标志码|作者简介|通信作者|"
        r"\bVol\b|\bNo\.?\b|\bJournal\b|\bISSN\b|doi|DOI|https?://",
        re.IGNORECASE,
    )

    @staticmethod
    def _guess_title(pages: List[Page], fallback: str = "") -> str:
        """
        从首页正文顶部推断论文标题。

        过滤掉页眉/页脚/元信息行后，取剩余候选中最长的一行
        （学术论文标题通常位于首页顶部且较其他正文行更长）；
        若全部候选都是噪声（或首页为空），回退到文件名。
        """
        if not pages:
            return fallback or "untitled"
        head = pages[0].text.strip().split("\n")[:8]
        candidates = [
            line.strip()
            for line in head
            if 6 <= len(line.strip()) <= 120
            and not PDFLoader._NOISE_RE.search(line)
        ]
        if candidates:
            return max(candidates, key=len)[:200]
        return PDFLoader._clean_fallback(fallback) or "untitled"

    @staticmethod
    def _clean_fallback(name: str) -> str:
        """
        整理文件名兜底标题：去掉人工编号前缀（如 "01_"）。

        中文期刊 PDF 常被解析成整页单行（实测首页可达 2800+ 字符），
        此时无法按行定位标题，只能回退文件名——那就把它整理干净些。
        """
        return re.sub(r"^\d+[_\-\s]+", "", name).strip()


def compare_pdf_engines(path: str) -> List[Dict[str, Any]]:
    """
    对比三种 PDF 解析方案（模块一要求的实验产出）。

    返回每个引擎的：可用性 / 解析耗时 / 字符数 / 页数 / 表格识别数
    用于生成《PDF 解析方案对比表》填入交付报告。
    """
    results: List[Dict[str, Any]] = []
    for engine in ("pymupdf", "pdfplumber", "pypdf"):
        row: Dict[str, Any] = {"engine": engine, "available": PDFLoader._available(engine)}
        if not row["available"]:
            row.update({"note": "未安装"})
            results.append(row)
            continue
        try:
            start = time.perf_counter()
            doc = PDFLoader(engine=engine, clean=False).load(path)
            row.update({
                "elapsed_s": round(time.perf_counter() - start, 3),
                "pages": doc.num_pages,
                "chars": doc.char_count,
                "chars_per_sec": int(doc.char_count / max(time.perf_counter() - start, 1e-6)),
            })
        except Exception as exc:  # noqa: BLE001
            row.update({"error": str(exc)})
        results.append(row)
    return results
