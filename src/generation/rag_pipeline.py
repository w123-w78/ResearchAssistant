# -*- coding: utf-8 -*-
"""
模块二：RAG 生成流水线（检索 -> 拼接 -> 生成 -> 溯源 -> 日志）

这是模块二的核心门面，对外只暴露两个方法：
    answer(query)         同步：返回 RAGAnswer
    stream_answer(query)  流式：yield 事件字典（前端打字机 + 引用动态出现）

完整的降级链路（5.2.3）：
    缓存命中      -> 直接返回（0 Token，~30ms）
    检索无结果    -> 纯 LLM 生成 + "知识库未找到"提示
    相关性低      -> 带"可能不准确"提示 + 展示检索片段供用户确认
    LLM 调用失败  -> 友好错误提示 + 重试建议（绝不抛出裸异常给前端）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from ..retrieval.base import RetrievedChunk
from ..retrieval.hybrid_retriever import HybridRetriever
from ..utils.config import Config
from ..utils.embedding import BaseEmbedder
from ..utils.llm_client import BaseLLM, LLMError
from ..utils.text_utils import estimate_tokens
from ..utils.logger import RequestLogger, Timer, get_logger, metrics

from .cache import SemanticCache
from .prompt_template import (build_context, build_rag_messages, parse_citations)
from .streaming import StreamMerger, typing_iterator

logger = get_logger("research_assistant.rag")


# ==================================================================
# 数据模型
# ==================================================================
@dataclass
class Citation:
    """一条引用来源。"""
    index: int
    doc_id: str
    title: str
    page: int
    text: str
    score: float = 0.0

    @property
    def label(self) -> str:
        t = self.title if len(self.title) <= 40 else self.title[:40] + "..."
        return f"[{self.index}] {t}（第 {self.page} 页）"

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "doc_id": self.doc_id, "title": self.title,
                "page": self.page, "text": self.text[:200],
                "score": round(self.score, 4), "label": self.label}


@dataclass
class RAGAnswer:
    """RAG 回答的完整结果（含可观测性字段）。"""
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    latency: float = 0.0                 # 总耗时（秒）
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    tokens: int = 0
    from_cache: bool = False
    degraded: bool = False               # 是否走了降级
    degrade_reason: str = ""             # no_result / low_score / llm_error
    error: str = ""

    @property
    def hit_rate(self) -> float:
        """检索质量自评：Top-1 置信度（模块二 5.2.4 要求）。"""
        return round(self.retrieved[0].score, 4) if self.retrieved else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "num_retrieved": len(self.retrieved),
            "latency": round(self.latency, 3),
            "retrieval_latency": round(self.retrieval_latency, 3),
            "generation_latency": round(self.generation_latency, 3),
            "tokens": self.tokens,
            "from_cache": self.from_cache,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "top1_score": self.hit_rate,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RAGAnswer(len={len(self.answer)}, citations={len(self.citations)}, "
                f"cache={self.from_cache}, degraded={self.degraded})")


# ==================================================================
# RAG 流水线
# ==================================================================
class RAGPipeline:
    """
    RAG 生成流水线。

    依赖注入：retriever（模块一）+ llm（统一客户端）+ cache（可选）
    这样设计的好处：单元测试时可传入 mock，无需真实模型。
    """

    def __init__(self, retriever: HybridRetriever, llm: BaseLLM,
                 config: Optional[Config] = None,
                 cache: Optional[SemanticCache] = None,
                 embedder: Optional[BaseEmbedder] = None,
                 db=None):
        self.retriever = retriever
        self.llm = llm
        self.config = config or Config.instance()
        self.db = db
        g_cfg = self.config.section("generation")

        self.max_context_chars = int(g_cfg.get("max_context_chars", 6000))
        self.enable_citation = bool(g_cfg.get("enable_citation", True))
        self.temperature = float(g_cfg.get("temperature", self.llm.temperature))
        self.score_threshold = float(self.config.get("retrieval.score_threshold", 0.15))
        self.final_top_k = int(self.config.get("retrieval.final_top_k", 5))
        self.enable_fallback = bool((g_cfg.get("fallback", {}) or {}).get("no_result_to_llm", True))

        # 语义缓存
        cache_cfg = self.config.section("cache")
        if cache is None and bool(cache_cfg.get("enabled", True)) and embedder is not None:
            cache = SemanticCache(
                embedder=embedder,
                db=self.db,
                threshold=float(cache_cfg.get("similarity_threshold", 0.93)),
                max_size=int(cache_cfg.get("max_size", 500)),
                ttl_seconds=int(cache_cfg.get("ttl_seconds", 86400)),
            )
        self.cache = cache
        self.embedder = embedder

        self.request_logger = RequestLogger(db=self.db)
        logger.info("RAGPipeline 初始化完成 (llm=%s, rerank=%s, cache=%s)",
                    llm.name, retriever.reranker.active_provider, self.cache is not None)

    # ------------------------------------------------------------------
    # 同步问答
    # ------------------------------------------------------------------
    def answer(self, query: str, history: Optional[List[Dict[str, str]]] = None,
               top_k: Optional[int] = None, use_cache: bool = True,
               doc_filter: Optional[List[str]] = None) -> RAGAnswer:
        """执行一次完整的 RAG 问答。"""
        start = time.perf_counter()
        result = RAGAnswer(query=query, answer="")

        # ---- 1) 语义缓存 ----
        if use_cache and self.cache is not None:
            cached = self.cache.get(query)
            if cached:
                result.answer = cached.get("answer", "")
                result.citations = [Citation(**c) if isinstance(c, dict) else c
                                    for c in cached.get("citations", [])]
                result.from_cache = True
                result.latency = time.perf_counter() - start
                logger.info("缓存命中，直接返回 (%.3fs)", result.latency)
                self._log(result)
                return result

        # ---- 2) 检索 ----
        with Timer() as t:
            hits = self.retriever.search(query, doc_filter=doc_filter)
        result.retrieval_latency = t.elapsed
        result.retrieved = hits[:self.final_top_k] if hits else []

        # ---- 3) 降级判断 ----
        low_score = bool(hits) and hits[0].score < self.score_threshold
        no_result = not hits
        if no_result:
            result.degraded = True
            result.degrade_reason = "no_result"
            logger.warning("检索无结果，触发降级: %s", query[:40])
            if not self.enable_fallback:
                result.answer = ("当前知识库中未找到与您的问题相关的文献片段。"
                                 "请尝试：①上传相关论文；②使用更具体的关键词；③检查论文是否已完成向量化。")
                result.latency = time.perf_counter() - start
                self._log(result)
                return result
        elif low_score:
            result.degraded = True
            result.degrade_reason = "low_score"
            logger.warning("检索分数偏低(%.3f < %.3f)，触发提示式降级",
                           hits[0].score, self.score_threshold)

        # ---- 4) 生成 ----
        messages = build_rag_messages(
            question=query, hits=result.retrieved, history=history,
            max_context_chars=self.max_context_chars, low_score=low_score,
        )
        with Timer() as tg:
            try:
                text = self.llm.chat(messages, temperature=self.temperature)
            except LLMError as exc:
                logger.error("LLM 调用失败: %s", exc)
                result.degraded = True
                result.degrade_reason = "llm_error"
                result.error = str(exc)
                result.answer = self._friendly_error(exc, hits=result.retrieved)
                result.latency = time.perf_counter() - start
                self._log(result)
                return result
        result.generation_latency = tg.elapsed
        result.answer = text.strip()

        # ---- 5) 引用溯源 ----
        if self.enable_citation:
            result.citations = self._build_citations(result.answer, result.retrieved)

        # ---- 6) 写缓存 + 指标 + 日志 ----
        if use_cache and self.cache is not None and not result.degraded:
            self.cache.put(query, {
                "answer": result.answer,
                "citations": [c.to_dict() for c in result.citations],
            })
        result.tokens = estimate_tokens(build_context(result.retrieved)) + estimate_tokens(result.answer)
        result.latency = time.perf_counter() - start
        metrics.incr("rag.tokens", result.tokens)
        metrics.observe("rag.latency", result.latency)
        self._log(result)
        return result

    # ------------------------------------------------------------------
    # 流式问答
    # ------------------------------------------------------------------
    def stream_answer(self, query: str, history: Optional[List[Dict[str, str]]] = None,
                      top_k: Optional[int] = None,
                      doc_filter: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        流式问答，逐块 yield 事件：
            {"type": "meta",     "retrieved": [...]}   检索完成
            {"type": "token",    "content": "..."}      文本块
            {"type": "citation", "citations": [...]}    新引用出现
            {"type": "done",     "answer": RAGAnswer}   结束
        """
        start = time.perf_counter()
        hits = self.retriever.search(query, doc_filter=doc_filter)
        used = hits[:self.final_top_k] if hits else []
        low_score = bool(hits) and hits[0].score < self.score_threshold

        yield {"type": "meta", "retrieved": [h.to_dict() for h in used],
               "degraded": (not hits) or low_score,
               "reason": "no_result" if not hits else ("low_score" if low_score else "")}

        # 缓存命中：用打字机模拟流式，保持前端体验一致
        cached = self.cache.get(query) if self.cache else None
        if cached:
            merger = StreamMerger(self._citation_lookup(used))
            for piece in typing_iterator(cached.get("answer", ""), chunk_size=6, delay=0.008):
                for ev in merger.feed(piece):
                    yield ev
            result = RAGAnswer(query=query, answer=merger.text, retrieved=used,
                               from_cache=True, latency=time.perf_counter() - start)
            result.citations = self._build_citations(merger.text, used)
            yield {"type": "done", "answer": result}
            return

        messages = build_rag_messages(query, used, history=history,
                                      max_context_chars=self.max_context_chars,
                                      low_score=low_score)
        merger = StreamMerger(self._citation_lookup(used))
        error: Optional[str] = None
        try:
            for piece in self.llm.stream(messages, temperature=self.temperature):
                for ev in merger.feed(piece):
                    yield ev
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.error("流式生成失败: %s", exc)
            yield {"type": "error", "message": self._friendly_error(exc, hits=used)}

        answer_text = merger.text.strip() or (self._friendly_error(Exception(error or "空响应"), used) if error else "")
        result = RAGAnswer(
            query=query, answer=answer_text, retrieved=used,
            latency=time.perf_counter() - start,
            degraded=(not hits) or low_score or bool(error),
            degrade_reason="no_result" if not hits else ("low_score" if low_score else ("llm_error" if error else "")),
            error=error or "",
        )
        result.citations = self._build_citations(answer_text, used)
        result.tokens = estimate_tokens(build_context(used)) + estimate_tokens(answer_text)
        if self.cache and not result.degraded:
            self.cache.put(query, {"answer": answer_text,
                                   "citations": [c.to_dict() for c in result.citations]})
        self._log(result)
        yield {"type": "done", "answer": result}

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _build_citations(self, answer: str, hits: List[RetrievedChunk]) -> List[Citation]:
        """解析答案中的引用标记 + 兜底（模型没标引用时列出 Top-3 来源）。"""
        parsed = parse_citations(answer, hits) if hits else []
        cites = [Citation(index=p["index"], doc_id=p["doc_id"], title=p["title"],
                          page=p["page"], text=p["text"], score=p["score"])
                 for p in parsed]
        if not cites and hits:
            # 兜底：模型未按要求标注时，列出最相关的前 3 条，保证可追溯性
            for i, h in enumerate(hits[:3], start=1):
                cites.append(Citation(index=i, doc_id=h.doc_id, title=h.chunk.doc_title,
                                      page=h.chunk.page, text=h.text[:200], score=h.score))
        return cites

    @staticmethod
    def _citation_lookup(hits: List[RetrievedChunk]) -> Dict[int, Dict[str, Any]]:
        return {i: {"index": i, "doc_id": h.doc_id, "title": h.chunk.doc_title,
                    "page": h.chunk.page, "text": h.text[:120], "score": round(h.score, 4)}
                for i, h in enumerate(hits, start=1)}

    @staticmethod
    def _friendly_error(exc: Exception, hits: Optional[List[RetrievedChunk]] = None) -> str:
        """LLM 失败时的友好提示（绝不把 traceback 抛给用户）。"""
        base = ("抱歉，模型服务暂时不可用，未能生成回答。\n\n"
                "建议：①确认本地 Ollama 已启动（`ollama serve`）或 API Key 配置正确；"
                "②稍后重试；③检查 config.yaml 中 llm.provider 设置。")
        if hits:
            base += "\n\n不过已为您检索到以下相关片段，可先参考：\n" + "\n".join(
                f"- {h.chunk.doc_title}（p.{h.chunk.page}）：{h.text[:100]}..." for h in hits[:3]
            )
        return base

    def _log(self, result: RAGAnswer) -> None:
        """结构化日志（模块二 5.2.4）。"""
        self.request_logger.log_rag(
            query=result.query, answer=result.answer,
            hits=[h.to_dict() for h in result.retrieved],
            latency=result.latency, tokens=result.tokens,
            extra={"from_cache": result.from_cache, "degraded": result.degraded,
                   "degrade_reason": result.degrade_reason,
                   "citations": [c.to_dict() for c in result.citations],
                   "top1_score": result.hit_rate},
        )

    # ------------------------------------------------------------------
    # 知识库联动
    # ------------------------------------------------------------------
    def invalidate_cache(self) -> None:
        """知识库变更时调用：使语义缓存失效。"""
        if self.cache and self.embedder is not None:
            docs = self.retriever.list_documents()
            version = SemanticCache.compute_kb_version([d["doc_id"] for d in docs],
                                                       self.retriever.size)
            self.cache.set_kb_version(version)

    def stats(self) -> Dict[str, Any]:
        s = {"retriever": self.retriever.stats(), "llm": self.llm.name}
        if self.cache:
            s["cache"] = self.cache.stats()
        return s
