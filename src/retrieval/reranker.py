# -*- coding: utf-8 -*-
"""
模块一 5.1.4：重排序（Reranker）

为什么需要 Rerank：
  双塔检索（向量 / BM25）为了效率把 query 和 doc 分别独立编码，
  无法建模二者的细粒度交互。Rerank 模型把 (query, doc) 拼在一起打分，
  能捕捉更精细的相关性，代价是不能离线预计算、只能在线跑。

  工程实践：先用双塔召回 Top-20（快、召回率高），
            再对这 20 条做精排取 Top-5（慢，但只对 20 条跑）。

实现方式（按 config.retrieval.reranker.provider 选择）：
  dashscope  调用通义千问 gte-rerank-v2 在线精排（推荐，真实交叉编码）
  heuristic  纯本地启发式打分（零网络开销，离线/降级可用）
  none       不重排（对照组实验用）

健壮性：dashscope 调用失败（网络/额度/接口变更）时自动降级为 heuristic，
        任何情况下都不会让检索链路中断。
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..utils.config import Config
from ..utils.logger import Timer, get_logger, metrics

from .base import RetrievedChunk

logger = get_logger("research_assistant.reranker")

# 注意：旧路径 /api/v1/services/rerank/text-similarity/rerank 已废弃（返回 url error）
RERANK_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class Reranker:
    """重排序器。"""

    def __init__(self, provider: str = "dashscope", model: str = "qwen3.7-text-rerank",
                 top_n: int = 5, timeout: int = 30, api_key: str = "",
                 config: Optional[Config] = None):
        self.config = config or Config.instance()
        self.provider = (provider or "dashscope").lower()
        self.model_name = model
        self.top_n = top_n
        self.timeout = timeout
        llm_cfg = self.config.section("llm")
        emb_cfg = self.config.section("embedding")
        from ..utils.env import ensure_loaded, get_api_key
        ensure_loaded()
        self.api_key = (api_key
                        or (emb_cfg.get("api_key") or "").strip()
                        or (llm_cfg.get("api_key") or "").strip()
                        or get_api_key("DASHSCOPE_API_KEY")).strip()
        self._active_provider = self.provider if self.provider == "heuristic" else "none"
        self._fallback_reason = ""
        if self.provider == "dashscope" and self.api_key:
            self._active_provider = "dashscope"

    @property
    def active_provider(self) -> str:
        """实际生效的精排方式。"""
        return self._active_provider

    @property
    def fallback_reason(self) -> str:
        return self._fallback_reason

    # ---------------- 主入口 ----------------
    def rerank(self, query: str, candidates: List[RetrievedChunk],
               top_n: Optional[int] = None) -> List[RetrievedChunk]:
        """对候选片段重排序，返回 top_n 个。"""
        if not candidates:
            return []
        if self.provider == "none":
            for i, c in enumerate(candidates, 1):
                c.rank = i
                c.source = c.source or "fusion"
            return candidates

        n = top_n or self.top_n
        with Timer() as t:
            if self._active_provider == "dashscope":
                scored, ok = self._dashscope_score(query, candidates)
                if not ok:  # 在线精排失败 -> 降级
                    logger.warning("在线 Rerank 失败，降级为启发式: %s", self._fallback_reason)
                    self._active_provider = "heuristic"
                    scored, _ = self._heuristic_score(query, candidates), True
            else:
                scored, _ = self._heuristic_score(query, candidates), True
        metrics.observe("rerank.latency", t.elapsed)
        metrics.incr("rerank.calls")

        for c, s in scored:
            c.rerank_score = float(s)
            c.score = float(s)
            c.source = "rerank"
        result = [c for c, _ in scored][:n]
        for i, c in enumerate(result, 1):
            c.rank = i
        return result

    # ---------------- DashScope 在线精排 ----------------
    def _dashscope_score(self, query: str, candidates: List[RetrievedChunk]
                         ) -> Tuple[List[Tuple[RetrievedChunk, float]], bool]:
        """调用 gte-rerank-v2；返回 (打分结果, 是否成功)。"""
        docs = [c.text[:2000] for c in candidates]  # 控制请求体大小
        payload = {
            "model": self.model_name,
            "input": {"query": query, "documents": docs},
            "parameters": {"top_n": min(self.top_n, len(docs)), "return_documents": False},
        }
        try:
            resp = requests.post(
                RERANK_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload, timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            results = ((data.get("output") or {}).get("results")) or []
            if not results:
                self._fallback_reason = f"接口未返回结果: {str(data)[:200]}"
                return [], False
            scored: List[Tuple[RetrievedChunk, float]] = []
            for item in results:
                idx = int(item.get("index", -1))
                if 0 <= idx < len(candidates):
                    scored.append((candidates[idx], float(item.get("relevance_score", 0.0))))
            if not scored:
                self._fallback_reason = "返回索引无法映射回候选"
                return [], False
            scored.sort(key=lambda x: x[1], reverse=True)
            metrics.incr("rerank.dashscope_calls")
            return scored, True
        except Exception as exc:  # noqa: BLE001
            self._fallback_reason = f"{type(exc).__name__}: {exc}"
            return [], False

    # ---------------- 启发式精排（降级方案） ----------------
    def _heuristic_score(self, query: str, candidates: List[RetrievedChunk]
                         ) -> List[Tuple[RetrievedChunk, float]]:
        """
        本地启发式精排：综合融合分、查询词覆盖率、词项紧密度与长度惩罚。

        虽然不如交叉编码精准，但零依赖、零延迟、完全可解释，
        作为降级方案保证系统在任何环境下都有"第三层精排"。
        """
        q_terms = self._terms(query)
        base = [c.rrf_score or c.score for c in candidates]
        bmax, bmin = max(base), min(base)
        span = (bmax - bmin) or 1.0

        scored: List[Tuple[RetrievedChunk, float]] = []
        for c, b in zip(candidates, base):
            norm_base = (b - bmin) / span
            terms = self._terms(c.text)
            coverage = (len(q_terms & terms) / len(q_terms)) if q_terms else 0.0
            density = self._density(c.text, q_terms)
            length_penalty = 1.0 - math.exp(-len(c.text) / 300.0)
            score = 0.55 * norm_base + 0.30 * coverage + 0.10 * density + 0.05 * (1 - length_penalty)
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        metrics.incr("rerank.heuristic_calls")
        return scored

    # ---------------- 工具 ----------------
    @staticmethod
    def _terms(text: str) -> set:
        en = set(re.findall(r"[a-z][a-z0-9\-]{1,}", text.lower()))
        zh: set = set()
        for seg in re.findall(r"[\u4e00-\u9fff]+", text):
            zh |= ({seg[i:i + 2] for i in range(len(seg) - 1)} or {seg})
        return en | zh

    @staticmethod
    def _density(text: str, q_terms: set) -> float:
        """关键词在文本中的聚集度（0~1）。"""
        if not q_terms or not text:
            return 0.0
        positions: List[int] = []
        for t in q_terms:
            start = 0
            while len(positions) < 40:
                idx = text.find(t, start)
                if idx < 0:
                    break
                positions.append(idx)
                start = idx + 1
        if len(positions) < 2:
            return 1.0 if positions else 0.0
        mean = sum(positions) / len(positions)
        std = (sum((p - mean) ** 2 for p in positions) / len(positions)) ** 0.5
        return float(max(0.0, 1.0 - std / max(len(text), 1)))

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Reranker(model={self.model_name}, active={self.active_provider}, "
                f"top_n={self.top_n})")
