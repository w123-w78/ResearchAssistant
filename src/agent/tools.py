# -*- coding: utf-8 -*-
"""
模块三 5.3.2：Agent 工具集

工具是 Agent 的"手脚"。本模块实现 9 个工具（超出课程要求的 8 个）：

  #  工具名              作用                              依赖
  ------------------------------------------------------------------
  1  knowledge_search    私有论文库语义检索（最核心）        HybridRetriever
  2  paper_metadata      提取论文标题/作者/年份/摘要/DOI     文档块 + LLM
  3  paper_compare       对比两篇论文的方法/数据集/结果       双路检索 + LLM
  4  keyword_extract     从问题或文本抽取检索关键词          规则 + LLM
  5  summarize_paper     生成结构化摘要（背景-方法-结果-结论）  文档块 + LLM
  6  get_current_time    返回系统当前时间                    ——
  7  calculator          安全数值计算（AST 解析，非 eval）    ——
  8  web_search          联网搜索（DuckDuckGo，默认关闭）     网络
  9  library_stats       知识库统计（文档数/块数/检索配置）    ——

安全说明：calculator 使用 AST 白名单解析，**不使用 eval**，
杜绝 `os.system('rm -rf /')` 这类注入风险。
"""
from __future__ import annotations

import ast
import json
import math
import operator
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..utils.config import Config
from ..utils.llm_client import BaseLLM
from ..utils.text_utils import estimate_tokens
from ..utils.logger import Timer, get_logger, metrics

logger = get_logger("research_assistant.tools")


# ==================================================================
# Tool 定义
# ==================================================================
@dataclass
class Tool:
    """工具定义。"""
    name: str
    description: str                                  # 给 LLM 看的说明（决定路由质量的关键）
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    func: Optional[Callable[[Dict[str, Any]], str]] = None
    timeout: int = 30
    category: str = "general"
    enabled: bool = True

    # -------- 执行 --------
    def run(self, args: Optional[Dict[str, Any]] = None) -> str:
        """执行工具（带超时保护）。"""
        args = args or {}
        if self.func is None:
            return f"[工具 {self.name} 未实现]"
        start = time.perf_counter()
        try:
            result = self._run_with_timeout(args)
            elapsed = time.perf_counter() - start
            metrics.incr("tool.success")
            metrics.observe("tool.latency", elapsed)
            logger.debug("工具 %s 执行成功 (%.2fs)", self.name, elapsed)
            return result
        except FuturesTimeout:
            metrics.incr("tool.fail")
            metrics.incr("tool.timeout")
            logger.warning("工具 %s 执行超时(%ss)", self.name, self.timeout)
            return f"[错误] 工具 {self.name} 执行超时（>{self.timeout}s），请简化输入后重试。"
        except Exception as exc:  # noqa: BLE001
            metrics.incr("tool.fail")
            logger.warning("工具 %s 执行失败: %s", self.name, exc)
            return f"[错误] 工具 {self.name} 执行失败：{type(exc).__name__}: {exc}"

    def _run_with_timeout(self, args: Dict[str, Any]) -> str:
        """用线程池实现超时（比 signal 更跨平台，Windows 可用）。"""
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self.func, args)
            try:
                return str(future.result(timeout=self.timeout))
            except FuturesTimeout:
                future.cancel()
                raise

    # -------- 描述 --------
    @property
    def signature(self) -> str:
        """生成给 LLM 看的工具签名（简洁格式，省 Token）。"""
        params = ", ".join(
            f"{k}:{v.get('type', 'string')}" for k, v in self.parameters.items()
        )
        return f"- {self.name}({params}): {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters, "category": self.category,
                "enabled": self.enabled}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Tool({self.name}, {len(self.parameters)}参数)"


class ToolRegistry:
    """工具注册表（Agent 通过它发现和调用工具）。"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool, override: bool = False) -> None:
        if tool.name in self._tools and not override:
            raise ValueError(f"工具已注册: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("注册工具: %s", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    @property
    def tools(self) -> List[Tool]:
        return [t for t in self._tools.values() if t.enabled]

    @property
    def names(self) -> List[str]:
        return [t.name for t in self.tools]

    def run(self, name: str, args: Optional[Dict[str, Any]] = None) -> str:
        tool = self.get(name)
        if tool is None:
            return f"[错误] 未找到工具 '{name}'。可用工具：{', '.join(self.names)}"
        return tool.run(args)

    def describe(self) -> str:
        """生成全部工具的描述文本（注入 Agent System Prompt）。"""
        return "\n".join(t.signature for t in self.tools)

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tools]

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ==================================================================
# 工具实现
# ==================================================================
def build_default_tools(rag_pipeline=None, llm: Optional[BaseLLM] = None,
                        config: Optional[Config] = None,
                        embedder=None) -> ToolRegistry:
    """
    构建默认工具集。

    :param rag_pipeline: RAGPipeline 实例（提供检索能力）
    :param llm: LLM 客户端（摘要/对比/元信息提取需要）
    :param embedder: 可选，关键词抽取用
    """
    cfg = config or Config.instance()
    registry = ToolRegistry()
    retriever = getattr(rag_pipeline, "retriever", None)

    # ---------------- 1. 知识库检索（核心工具） ----------------
    def _knowledge_search(args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return "[错误] 缺少 query 参数"
        if retriever is None or retriever.is_empty():
            return ("知识库为空，未检索到任何内容。请先上传论文文档并完成向量化。"
                    "（若已上传，请确认索引构建成功）")
        hits = retriever.search(query, final_top_k=top_k)
        if not hits:
            return f"未检索到与「{query}」相关的内容。建议换用更具体的关键词。"
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(f"[{i}] {h.chunk.doc_title}（第{h.chunk.page}页，相关度{h.score:.3f}）\n{h.text[:500]}")
        return "\n\n".join(lines)

    registry.register(Tool(
        name="knowledge_search",
        description="检索本地论文知识库，返回最相关的文献片段（含文档名与页码）。"
                    "回答任何关于论文内容的问题都必须先调用它获取事实依据。",
        parameters={"query": {"type": "string", "description": "检索查询词，建议用论文中的专业术语",
                              "required": True},
                    "top_k": {"type": "int", "description": "返回片段数，默认5"}},
        func=_knowledge_search, timeout=30, category="retrieval",
    ))

    # ---------------- 2. 论文元信息提取 ----------------
    def _paper_metadata(args: Dict[str, Any]) -> str:
        title = str(args.get("title") or args.get("doc_id") or "").strip()
        if not title:
            return "[错误] 缺少 title 参数"
        if retriever is None:
            return "[错误] 检索器未初始化"
        doc = retriever.find_doc_by_title(title)
        if doc is None:
            docs = retriever.list_documents()
            names = "、".join(d["title"][:30] for d in docs[:10])
            return f"未找到论文「{title}」。当前知识库中有：{names or '（空）'}"
        chunks = retriever.get_doc_chunks(doc["doc_id"])
        head = "\n".join(c.text for c in chunks[:2])[:1500]
        info: Dict[str, Any] = {
            "doc_id": doc["doc_id"], "title": doc["title"],
            "num_chunks": doc["chunks"], "pages": doc["pages"][:20],
            "source": doc.get("source", ""),
        }
        if llm is not None:
            prompt = (
                "从以下论文开头内容中提取元信息，只输出 JSON，字段："
                '{"title":"", "authors":[], "year":null, "venue":"", "doi":null, "abstract":""}。'
                "无法确定的字段填 null，不要编造。\n\n" + head
            )
            try:
                raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    info.update(json.loads(m.group(0)))
            except Exception as exc:  # noqa: BLE001
                info["llm_error"] = str(exc)
        info["abstract_preview"] = head[:400]
        return json.dumps(info, ensure_ascii=False, indent=2)

    registry.register(Tool(
        name="paper_metadata",
        description="提取指定论文的元信息：标题、作者、发表年份、会议/期刊、DOI、摘要。"
                    "输入可以是论文标题的完整或部分关键词。",
        parameters={"title": {"type": "string", "description": "论文标题或其中关键词", "required": True}},
        func=_paper_metadata, timeout=30, category="paper",
    ))

    # ---------------- 3. 论文对比 ----------------
    def _paper_compare(args: Dict[str, Any]) -> str:
        a = str(args.get("paper_a", "")).strip()
        b = str(args.get("paper_b", "")).strip()
        aspect = str(args.get("aspect", "方法、数据集、实验结果"))
        if not a or not b:
            return "[错误] 需要 paper_a 和 paper_b 两个参数"
        if retriever is None:
            return "[错误] 检索器未初始化"
        da, db = retriever.find_doc_by_title(a), retriever.find_doc_by_title(b)
        if da is None or db is None:
            missing = [x for x, d in ((a, da), (b, db)) if d is None]
            return f"未找到论文：{'、'.join(missing)}。请确认论文已上传。"
        ca = "\n".join(c.text for c in retriever.get_doc_chunks(da["doc_id"])[:8])[:2500]
        cb = "\n".join(c.text for c in retriever.get_doc_chunks(db["doc_id"])[:8])[:2500]

        if llm is None:
            return (f"【论文A】{da['title']}\n{ca[:800]}\n\n【论文B】{db['title']}\n{cb[:800]}"
                    "\n（未配置 LLM，仅返回原文片段）")
        prompt = (
            f"请对比以下两篇论文在「{aspect}」方面的异同。\n"
            "要求：用 Markdown 表格呈现对比，再给出 3 条结论性要点；"
            "只依据给出的原文，不确定处标注“原文未明确”。\n\n"
            f"【论文A：{da['title']}】\n{ca}\n\n【论文B：{db['title']}】\n{cb}"
        )
        try:
            return llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            return f"[错误] 对比生成失败: {exc}\n\n【论文A片段】{ca[:500]}\n\n【论文B片段】{cb[:500]}"

    registry.register(Tool(
        name="paper_compare",
        description="对比两篇论文在方法、数据集、实验结果等方面的异同，输出对比表格与结论。",
        parameters={"paper_a": {"type": "string", "description": "第一篇论文标题", "required": True},
                    "paper_b": {"type": "string", "description": "第二篇论文标题", "required": True},
                    "aspect": {"type": "string", "description": "对比维度，默认：方法、数据集、实验结果"}},
        func=_paper_compare, timeout=60, category="paper",
    ))

    # ---------------- 4. 关键词提取 ----------------
    def _keyword_extract(args: Dict[str, Any]) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            return "[错误] 缺少 text 参数"
        try:
            from ..utils.text_utils import extract_keywords
            words = extract_keywords(text, top_k=int(args.get("top_k", 5)))
            return json.dumps({"keywords": words}, ensure_ascii=False)
        except Exception:  # pragma: no cover
            return json.dumps({"keywords": []}, ensure_ascii=False)

    registry.register(Tool(
        name="keyword_extract",
        description="从用户问题或文本中提取核心检索关键词，用于优化检索查询。",
        parameters={"text": {"type": "string", "description": "待提取关键词的文本", "required": True},
                    "top_k": {"type": "int", "description": "关键词数量，默认5"}},
        func=_keyword_extract, timeout=15, category="nlp",
    ))

    # ---------------- 5. 结构化摘要生成 ----------------
    def _summarize_paper(args: Dict[str, Any]) -> str:
        title = str(args.get("title", "")).strip()
        if not title:
            return "[错误] 缺少 title 参数"
        if retriever is None:
            return "[错误] 检索器未初始化"
        doc = retriever.find_doc_by_title(title)
        if doc is None:
            return f"未找到论文「{title}」"
        chunks = retriever.get_doc_chunks(doc["doc_id"])
        content = "\n".join(c.text for c in chunks)[:6000]
        if llm is None:
            return content[:1500]
        prompt = (
            "请为以下论文生成结构化摘要，格式：\n"
            "## 研究背景\n## 核心方法\n## 实验结果\n## 结论与贡献\n## 局限与展望\n"
            "要求：每部分 2-4 句，只依据原文，不要编造数据。\n\n" + content
        )
        try:
            return llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            return f"[错误] 摘要生成失败: {exc}"

    registry.register(Tool(
        name="summarize_paper",
        description="为指定论文生成结构化摘要（研究背景-核心方法-实验结果-结论与贡献-局限与展望）。",
        parameters={"title": {"type": "string", "description": "论文标题", "required": True}},
        func=_summarize_paper, timeout=60, category="paper",
    ))

    # ---------------- 6. 当前时间 ----------------
    def _get_time(args: Dict[str, Any]) -> str:
        now = datetime.now()
        fmt = str(args.get("format", "%Y-%m-%d %H:%M:%S"))
        return json.dumps({
            "datetime": now.strftime(fmt),
            "date": now.strftime("%Y-%m-%d"),
            "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
            "timestamp": int(now.timestamp()),
        }, ensure_ascii=False)

    registry.register(Tool(
        name="get_current_time",
        description="获取系统当前的日期与时间。当用户问及今天、现在、最新等时间相关问题时使用。",
        parameters={"format": {"type": "string", "description": "时间格式，默认 %Y-%m-%d %H:%M:%S"}},
        func=_get_time, timeout=5, category="utility",
    ))

    # ---------------- 7. 计算器（AST 安全实现） ----------------
    def _calculator(args: Dict[str, Any]) -> str:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return "[错误] 缺少 expression 参数"
        try:
            value = safe_eval(expr)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return json.dumps({"expression": expr, "result": value}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            return f"[错误] 无法计算「{expr}」：{exc}"

    registry.register(Tool(
        name="calculator",
        description="计算数学表达式（支持 + - * / ^ % 与 sqrt/sin/cos/log 等函数）。"
                    "适用于数值比较、百分比换算、实验结果换算等场景。",
        parameters={"expression": {"type": "string", "description": "数学表达式，如 (88.3-85.1)/85.1*100",
                                   "required": True}},
        func=_calculator, timeout=10, category="utility",
    ))

    # ---------------- 8. 联网搜索（可选） ----------------
    web_cfg = cfg.section("tools").get("web_search", {}) or {}

    def _web_search(args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "[错误] 缺少 query 参数"
        if not web_cfg.get("enabled", False):
            return ("[提示] 联网搜索工具当前已在配置中关闭（tools.web_search.enabled=false）。"
                    "如需使用，请在 config.yaml 中开启并确保网络可用。")
        try:
            import requests

            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=10,
            )
            data = resp.json()
            items = []
            if data.get("AbstractText"):
                items.append(f"摘要: {data['AbstractText']}")
            for t in (data.get("RelatedTopics") or [])[:int(web_cfg.get("max_results", 3))]:
                if isinstance(t, dict) and t.get("Text"):
                    items.append(t["Text"])
            return "\n".join(items) if items else f"未找到与「{query}」相关的网络结果。"
        except Exception as exc:  # noqa: BLE001
            return f"[错误] 联网搜索失败（可能无网络）：{exc}"

    registry.register(Tool(
        name="web_search",
        description="联网搜索获取知识库之外的公开信息（如某会议最新 SOTA）。"
                    "仅在知识库确实没有且用户明确需要外部信息时使用。",
        parameters={"query": {"type": "string", "description": "搜索关键词", "required": True}},
        func=_web_search, timeout=20, category="web",
        enabled=bool(web_cfg.get("enabled", False)) or True,  # 注册但内部判断是否启用
    ))

    # ---------------- 9. 知识库统计 ----------------
    def _library_stats(args: Dict[str, Any]) -> str:
        if retriever is None:
            return "[错误] 检索器未初始化"
        docs = retriever.list_documents()
        stats = retriever.stats()
        out = {
            "num_documents": len(docs),
            "num_chunks": retriever.size,
            "reranker": stats.get("reranker"),
            "fusion_mode": stats.get("fusion_mode"),
            "documents": [{"title": d["title"][:60], "chunks": d["chunks"],
                           "pages": len(d["pages"])} for d in docs[:20]],
        }
        return json.dumps(out, ensure_ascii=False, indent=2)

    registry.register(Tool(
        name="library_stats",
        description="查看知识库统计信息：已收录论文数量、分块数、检索配置。"
                    "当用户问知识库里有哪些论文、一共多少篇时使用。",
        parameters={},
        func=_library_stats, timeout=10, category="retrieval",
    ))

    logger.info("工具集构建完成，共 %s 个工具: %s", len(registry), ", ".join(registry.names))
    return registry


# ==================================================================
# 安全计算（AST 白名单，禁用 eval）
# ==================================================================
_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_SAFE_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
    "log10": math.log10, "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
}


def safe_eval(expression: str) -> float:
    """
    安全的数学表达式求值。

    与 eval 的区别：
      - 只允许数字、四则运算、幂、取模与白名单函数
      - 禁止变量、属性访问、函数调用（白名单外）、import
      - 例：safe_eval("__import__('os').system('ls')") -> 抛异常
    """
    # 清理常见中文符号
    expr = (expression.replace("×", "*").replace("÷", "/")
            .replace("＋", "+").replace("－", "-").replace("^", "**"))
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元运算: {type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id in _SAFE_FUNCS:
                return _SAFE_FUNCS[node.id]
            raise ValueError(f"不允许使用变量: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("只允许调用白名单函数")
            fn = _SAFE_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"不允许调用函数: {node.func.id}")
            return fn(*[_eval(a) for a in node.args])
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    result = _eval(tree)
    if isinstance(result, complex):
        raise ValueError("不支持复数结果")
    return float(result)
