# -*- coding: utf-8 -*-
"""
模块一 5.1.2 策略四：学术论文结构感知分块（Academic Chunker）

针对论文的三大结构特征做专门优化（本项目的差异化点）：

  1. 章节感知：识别 Abstract / Introduction / Method / Experiments / Conclusion
     等章节标题，保证一个 chunk 不跨章节（避免"方法"和"实验"混在一起）
  2. 公式保护：LaTeX 公式（$...$ / $$...$$ / \\begin{equation}）与
     独立公式块整体保留，绝不在公式中间切断
  3. 表格保护：Markdown 表格行整块保留，不被拆散

实现：在递归字符切分的基础上叠加"章节切分 -> 保护块切分 -> 递归切分"三段式。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseChunker, split_paragraphs
from .recursive_chunk import RecursiveCharacterChunker

# 常见论文章节标题（中英双语）
SECTION_PATTERNS = [
    r"^\s*\d{0,2}\.?\s*(abstract|摘要)\s*$",
    r"^\s*\d{0,2}\.?\s*(introduction|引言|简介|绪论)\s*$",
    r"^\s*\d{0,2}\.?\s*(related work|related works|相关工作)\s*$",
    r"^\s*\d{0,2}\.?\s*(method|methodology|approach|method[s]?|方法|模型方法)\s*$",
    r"^\s*\d{0,2}\.?\s*(experiments?|experimental setup|results?|实验|实验结果)\s*$",
    r"^\s*\d{0,2}\.?\s*(discussion|讨论|分析)\s*$",
    r"^\s*\d{0,2}\.?\s*(conclusion|conclusions|结论|总结)\s*$",
    r"^\s*\d{0,2}\.?\s*(references|参考文献)\s*$",
    r"^\s*\d+(\.\d+){0,2}\s+\S.{0,60}$",      # 编号标题：3.2 Proposed Method
]
_SECTION_RE = [re.compile(p, re.IGNORECASE) for p in SECTION_PATTERNS]

# 公式环境
_FORMULA_ENV_RE = re.compile(
    r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{\1\*?\}",
    re.DOTALL,
)
_INLINE_FORMULA_RE = re.compile(r"\$\$.+?\$\$|\\\(.+?\\\)|(?<!\$)\$(?!\$)[^$\n]{1,200}\$(?!\$)", re.DOTALL)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


class AcademicChunker(BaseChunker):
    """面向学术论文的结构感知分块器（本系统默认推荐策略）。"""

    name = "academic"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 min_chunk_size: int = 80, keep_formula: bool = True,
                 section_keywords: Optional[List[str]] = None, **kwargs):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, **kwargs)
        self.keep_formula = keep_formula
        self.section_keywords = [k.lower() for k in (section_keywords or [])]
        # 内部复用递归切分器处理"章节内的长文本"
        self._inner = RecursiveCharacterChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=min_chunk_size
        )

    # ---------------- 主入口 ----------------
    def split_text(self, text: str, doc_meta: Optional[Dict[str, Any]] = None) -> List[str]:
        if not text.strip():
            return []
        # Step 1: 按章节切分为若干 section（保留标题在各自 section 开头）
        sections = self._split_by_section(text)
        out: List[str] = []
        for title, body in sections:
            prefix = f"{title}\n" if title else ""
            # Step 2: 章节内按"保护块"切分（公式/表格不切断）
            for block in self._split_protected(body):
                # Step 3: 普通文本走递归切分
                if self._is_protected(block):
                    out.append(prefix + block if prefix and not out else (block if not prefix else prefix + block))
                    prefix = ""
                else:
                    for piece in self._inner.split_text(block):
                        out.append(prefix + piece)
                        prefix = ""
        return [o.strip() for o in out if o.strip()]

    # ---------------- Step 1: 章节切分 ----------------
    def _split_by_section(self, text: str) -> List[Tuple[str, str]]:
        """返回 [(章节标题, 章节正文), ...]；无章节时返回 [("", text)]。"""
        lines = text.split("\n")
        sections: List[Tuple[str, str]] = []
        cur_title = ""
        cur_body: List[str] = []
        for line in lines:
            if self._is_section_heading(line):
                if cur_body:
                    sections.append((cur_title, "\n".join(cur_body)))
                cur_title = line.strip()
                cur_body = []
            else:
                cur_body.append(line)
        if cur_body:
            sections.append((cur_title, "\n".join(cur_body)))
        return sections or [("", text)]

    def _is_section_heading(self, line: str) -> bool:
        s = line.strip()
        if not s or len(s) > 80:
            return False
        if any(rx.match(s) for rx in _SECTION_RE):
            return True
        # 用户在配置里自定义的关键词
        low = s.lower().strip(" .:：")
        return any(low.startswith(k) for k in self.section_keywords) if self.section_keywords else False

    # ---------------- Step 2: 保护块切分 ----------------
    def _split_protected(self, text: str) -> List[str]:
        """把公式/表格从普通文本中剥离出来，作为不可分割的独立块。"""
        if not self.keep_formula:
            return split_paragraphs(text)
        blocks: List[str] = []
        rest = text
        # 公式环境 \begin{equation}...\end{equation}
        def _take(m):
            blocks.append(m.group(0))
            return "\n@@PROTECTED_%d@@\n" % (len(blocks) - 1)

        rest = _FORMULA_ENV_RE.sub(_take, rest)
        # 行内/独立公式 $...$ $$...$$
        for m in list(_INLINE_FORMULA_RE.finditer(rest)):
            seg = m.group(0)
            if "\n" in seg:  # 跨行公式块，单独保护
                blocks.append(seg)
                rest = rest.replace(seg, "\n@@PROTECTED_%d@@\n" % (len(blocks) - 1))
        # 表格：连续以 | 开头的行
        lines = rest.split("\n")
        buf: List[str] = []
        result_lines: List[str] = []
        for line in lines:
            if _TABLE_ROW_RE.match(line):
                buf.append(line)
            else:
                if buf:
                    blocks.append("\n".join(buf))
                    result_lines.append("@@PROTECTED_%d@@" % (len(blocks) - 1))
                    buf = []
                result_lines.append(line)
        if buf:
            blocks.append("\n".join(buf))
            result_lines.append("@@PROTECTED_%d@@" % (len(blocks) - 1))
        rest = "\n".join(result_lines)

        # 还原：把占位符替换回保护块内容，其余按段落切分
        out: List[str] = []
        for para in split_paragraphs(rest):
            if re.fullmatch(r"@@PROTECTED_\d+@@", para.strip()):
                idx = int(re.search(r"\d+", para.strip()).group())
                out.append(blocks[idx])
            elif "@@PROTECTED_" in para:
                parts = re.split(r"(@@PROTECTED_\d+@@)", para)
                cur = ""
                for p in parts:
                    m = re.fullmatch(r"@@PROTECTED_(\d+)@@", p.strip())
                    if m:
                        if cur.strip():
                            out.append(cur.strip())
                            cur = ""
                        out.append(blocks[int(m.group(1))])
                    else:
                        cur += p
                if cur.strip():
                    out.append(cur.strip())
            else:
                out.append(para)
        return out

    @staticmethod
    def _is_protected(block: str) -> bool:
        b = block.strip()
        return (
            b.startswith("\\begin{") or b.startswith("$$")
            or (_TABLE_ROW_RE.match(b.split("\n")[0]) if b else False)
        )
