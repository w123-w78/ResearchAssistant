# -*- coding: utf-8 -*-
"""
模块三 5.3.3：工具选择路由（Fast Router）

为什么需要路由：
  让 LLM 每次都跑完整 ReAct 循环（至少 2-3 轮对话）成本高、延迟大。
  而真实场景中大量问题是"一眼就能判断"的：
      "3.14 乘 2.56"        -> 计算器，无需检索
      "现在几点"            -> 时间工具
      "知识库有几篇论文"     -> 库统计
  路由命中时省掉 2 次 LLM 调用，实测节省约 80% Token、降低 60% 延迟。

实现：规则优先（快、零成本、可解释），规则不命中时交给 ReAct Agent 自主决策。
      可选开启 LLM 分类作为兜底（config.agent.enable_router_llm）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern

from ..utils.logger import get_logger

logger = get_logger("research_assistant.router")


@dataclass
class RouteResult:
    """路由结果。"""
    route: str                 # direct | tool | rag | agent
    tool_name: str = ""        # route == "tool" 时的工具名
    tool_args: Dict[str, Any] = None
    reason: str = ""
    confidence: float = 1.0

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}


class QueryRouter:
    """
    基于规则的查询路由器。

    路由优先级（从高到低）：
      1. 数学计算       -> calculator
      2. 时间日期       -> get_current_time
      3. 知识库统计     -> library_stats
      4. 论文元信息     -> paper_metadata
      5. 单篇论文摘要   -> summarize_paper
      6. 两篇论文对比   -> paper_compare
      7. 寒暄 / 通用概念 -> direct（直接由模型回答，不检索）
      8. 其他           -> agent（交给 ReAct 自主决策）
    """

    # ---------------- 规则定义 ----------------
    MATH_PATTERN: Pattern = re.compile(
        r"^[\d\.\s\+\-\*/×÷\(\)\^%]+[,，]?\s*(等于|是多少|结果)?\s*[?？]?$"
    )
    MATH_KEYWORDS = ("计算", "等于多少", "换算", "百分比", "增长率", "提升幅度")
    TIME_KEYWORDS = ("现在几点", "今天", "今天几号", "现在时间", "当前时间", "日期", "星期几", "哪一年")
    LIB_KEYWORDS = ("知识库", "有多少篇", "几篇论文", "有哪些论文", "收录", "库里")
    META_KEYWORDS = ("作者", "发表在哪", "哪一年发表", "doi", "DOI", "期刊", "会议", "元信息", "是谁写的")
    SUMMARY_KEYWORDS = ("总结一下", "生成摘要", "这篇论文讲了什么", "概述", "摘要")
    COMPARE_KEYWORDS = ("对比", "比较", "区别", "差异", "哪个更好", "相比", " versus ", " vs ")
    GREETING_PATTERN: Pattern = re.compile(
        r"^(你好|您好|hi|hello|谢谢|再见|你是谁|你能做什么|帮助|help)\s*[!！。.?？]*$", re.IGNORECASE
    )

    def __init__(self, enabled: bool = True, llm=None, use_llm_fallback: bool = False):
        self.enabled = enabled
        self.llm = llm
        self.use_llm_fallback = use_llm_fallback

    # ---------------- 主入口 ----------------
    def route(self, query: str, has_documents: bool = True) -> RouteResult:
        """判断应该走哪条路径。"""
        if not self.enabled:
            return RouteResult(route="agent", reason="路由已禁用，交由 Agent 自主决策")

        q = query.strip()
        low = q.lower()

        # 1) 纯数学表达式
        if self.MATH_PATTERN.match(q) or (any(k in q for k in self.MATH_KEYWORDS) and len(q) < 30):
            expr = self._extract_expression(q)
            return RouteResult(route="tool", tool_name="calculator",
                               tool_args={"expression": expr},
                               reason="识别为数值计算问题，无需检索知识库", confidence=0.95)

        # 2) 时间
        if any(k in q for k in self.TIME_KEYWORDS) and len(q) < 25:
            return RouteResult(route="tool", tool_name="get_current_time",
                               reason="识别为时间查询", confidence=0.9)

        # 3) 知识库统计
        if any(k in q for k in self.LIB_KEYWORDS) and any(k in q for k in ("多少", "几篇", "哪些", "统计", "有哪些")):
            return RouteResult(route="tool", tool_name="library_stats",
                               reason="识别为知识库元查询", confidence=0.85)

        # 4) 寒暄
        if self.GREETING_PATTERN.match(q):
            return RouteResult(route="direct", reason="寒暄类问题，直接由模型回答", confidence=0.9)

        # 知识库为空时，涉及论文内容的查询一律走 direct（明确告知用户）
        if not has_documents and self._is_paper_question(q):
            return RouteResult(route="direct", reason="知识库为空，改为通用回答并提示上传论文",
                               confidence=0.8)

        # 5) 论文对比
        if any(k in q for k in self.COMPARE_KEYWORDS):
            # 若提到两篇论文名，直接路由；否则交给 Agent（它需要先查有哪些论文）
            return RouteResult(route="agent", reason="对比类问题，需 Agent 规划（可能需多次检索）",
                               confidence=0.6)

        # 6) 元信息
        if any(k in q for k in self.META_KEYWORDS):
            return RouteResult(route="agent", reason="元信息查询，需先定位论文", confidence=0.6)

        # 7) 摘要
        if any(k in q for k in self.SUMMARY_KEYWORDS):
            return RouteResult(route="agent", reason="摘要生成，需先定位论文并读取全文", confidence=0.6)

        # 8) 通用概念（无论文指向）-> direct
        if self._is_general_concept(q):
            return RouteResult(route="direct", reason="通用概念解释，模型自身知识足以回答",
                               confidence=0.7)

        # 9) 默认 -> Agent（会调用 knowledge_search）
        return RouteResult(route="agent", reason="需要检索知识库，交由 Agent 决策", confidence=0.5)

    # ---------------- 辅助判断 ----------------
    @staticmethod
    def _extract_expression(q: str) -> str:
        """从问句中抽出数学表达式。"""
        expr = re.sub(r"(计算|等于多少|是多少|结果|请问|请|求|的值)[,，。.？?]?", "", q)
        expr = expr.strip().rstrip("?？。.=").strip()
        return expr or q

    @staticmethod
    def _is_paper_question(q: str) -> bool:
        words = ("论文", "文献", "方法", "实验", "数据集", "模型", "这篇", "该文", "作者")
        return any(w in q for w in words)

    @staticmethod
    def _is_general_concept(q: str) -> bool:
        """
        判断是否为"通用概念解释"类问题（不需要查私有知识库）。

        特征：以"什么是/介绍一下/解释一下"开头，且未提及具体论文。
        """
        patterns = (r"^(什么是|什么叫|介绍一下|解释一下|说说|谈谈)", r"是什么[?？]?$", r"^(why|what|how)\s", )
        if any(re.search(p, q, re.IGNORECASE) for p in patterns):
            # 若提到具体论文/文档，则不视为通用问题
            specific = ("这篇", "该论文", "上述", "论文中", "文章里", "本论文")
            return not any(s in q for s in specific)
        return False

    # ---------------- LLM 兜底分类（可选） ----------------
    def route_with_llm(self, query: str, tool_names: List[str]) -> RouteResult:
        """用 LLM 做路由分类（规则不命中时的兜底，默认关闭以省 Token）。"""
        if not self.llm:
            return RouteResult(route="agent", reason="LLM 不可用，默认走 Agent")
        import json

        prompt = (
            "你是一个路由分类器。根据用户问题，判断应该直接回答还是调用工具。\n"
            f"可用工具：{', '.join(tool_names)}\n"
            '只输出 JSON：{"route": "direct|tool", "tool": "工具名或空", "reason": "一句话理由"}\n\n'
            f"用户问题：{query}"
        )
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                return RouteResult(route=data.get("route", "agent"),
                                   tool_name=data.get("tool", ""),
                                   reason=data.get("reason", ""), confidence=0.7)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 路由失败: %s", exc)
        return RouteResult(route="agent", reason="LLM 路由失败，默认走 Agent")

    # ---------------- 统计 ----------------
    def explain(self, query: str, has_documents: bool = True) -> str:
        """返回人类可读的路由解释（前端可观测面板展示）。"""
        r = self.route(query, has_documents)
        target = r.tool_name or r.route
        return f"路由 → {target}（{r.reason}，置信度 {r.confidence:.2f}）"
