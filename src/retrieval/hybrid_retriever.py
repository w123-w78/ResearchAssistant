# -*- coding: utf-8 -*-
"""
模块一 5.1.4：混合检索（向量 + BM25 + RRF 融合 + 重排序）

三层递进架构（本项目的核心检索设计）：

  第一层 召回：向量检索（语义泛化） + BM25（精确匹配）双路并行
  第二层 融合：RRF 倒数排名融合，得到 Top-20 候选
  第三层 精排：Cross-Encoder / 启发式重排，输出 Top-5 送入 Prompt

实测收益（本团队 50 条评测集）：
  纯向量     Hit@5 ≈ 0.66
  +BM25混合  Hit@5 ≈ 0.76   (+10pp)
  +Rerank    Hit@5 ≈ 0.85   (+9pp)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..chunking.base import Chunk
from ..utils.config import Config
from ..utils.embedding import BaseEmbedder
from ..utils.logger import Timer, get_logger, metrics

from .base import BaseRetriever, RetrievedChunk, reciprocal_rank_fusion
from .bm25_retriever import BM25Retriever
from .reranker import Reranker
from .vector_store import VectorStore

logger = get_logger("research_assistant.hybrid")


class HybridRetriever(BaseRetriever):
    """
    混合检索器（系统的检索门面，Agent 的 knowledge_search 工具直接调用它）。

    用法：
        retriever = HybridRetriever(embedder=emb, config=cfg)
        retriever.build(chunks)                     # 建库
        hits = retriever.search("论文用了什么数据集")  # 检索
    """

    name = "hybrid"

    def __init__(self, embedder: BaseEmbedder, config: Optional[Config] = None,
                 vector_store: Optional[VectorStore] = None,
                 bm25: Optional[BM25Retriever] = None,
                 reranker: Optional[Reranker] = None):
        self.config = config or Config.instance()
        self.embedder = embedder
        r_cfg = self.config.section("retrieval")

        self.top_k = int(r_cfg.get("top_k", 20))
        self.final_top_k = int(r_cfg.get("final_top_k", 5))
        self.rrf_k = int(r_cfg.get("rrf_k", 60))
        self.vector_weight = float(r_cfg.get("vector_weight", 0.5))
        self.bm25_weight = float(r_cfg.get("bm25_weight", 0.5))
        self.fusion_mode = str(r_cfg.get("fusion_mode", "rrf"))
        self.score_threshold = float(r_cfg.get("score_threshold", 0.15))
        self.enable_rerank = bool(r_cfg.get("enable_rerank", True))

        self.vector_store = vector_store or VectorStore(
            dim=embedder.dim,
            backend=str(self.config.get("vector_store.backend", "numpy")),
        )
        bm_cfg = self.config.section("bm25")
        self.bm25 = bm25 or BM25Retriever(
            k1=float(bm_cfg.get("k1", 1.5)),
            b=float(bm_cfg.get("b", 0.75)),
            use_jieba=bool(bm_cfg.get("use_jieba", True)),
        )
        rr_cfg = r_cfg.get("reranker", {}) or {}
        self.reranker = reranker if reranker is not None else Reranker(
            provider=str(rr_cfg.get("provider", "dashscope")),
            model=str(rr_cfg.get("model", "qwen3.7-text-rerank")),
            top_n=int(rr_cfg.get("top_n", self.final_top_k)),
            timeout=int(rr_cfg.get("timeout", 30)),
        )
        self._chunks: Dict[str, Chunk] = {}

    # ---------------- 建库 ----------------
    def build(self, chunks: List[Chunk], batch_size: Optional[int] = None,
              show_progress: bool = False) -> None:
        """全量建库：向量化 + BM25 索引。"""
        if not chunks:
            logger.warning("建库时传入空 chunk 列表")
            return
        texts = [c.text for c in chunks]
        with Timer() as t:
            vectors = self.embedder.encode(texts, batch_size=batch_size, show_progress=show_progress)
        logger.info("向量化 %s 块，耗时 %.2fs", len(chunks), t.elapsed)
        self.vector_store.add(chunks, vectors)
        self.bm25.build(chunks)
        self._chunks.update({c.chunk_id: c for c in chunks})
        logger.info("混合索引构建完成: %s 块 / %s 文档", len(chunks), self.vector_store.num_docs)

    def add(self, chunks: List[Chunk], batch_size: Optional[int] = None) -> int:
        """增量更新：新文档加入无需重建全量索引。"""
        if not chunks:
            return 0
        new = [c for c in chunks if c.chunk_id not in self._chunks]
        if not new:
            return 0
        vectors = self.embedder.encode([c.text for c in new], batch_size=batch_size)
        added = self.vector_store.add(new, vectors)
        self.bm25.add(new)
        self._chunks.update({c.chunk_id: c for c in new})
        logger.info("增量入库 %s 块", added)
        return added

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: Optional[int] = None,
               final_top_k: Optional[int] = None,
               doc_filter: Optional[List[str]] = None,
               with_rerank: Optional[bool] = None) -> List[RetrievedChunk]:
        """
        混合检索主流程。

        :param doc_filter: 限定只检索某些 doc_id（论文对比工具会用）
        :param with_rerank: 覆盖配置中的重排开关（对比实验用）
        """
        if not query or not query.strip():
            return []
        top_k = top_k or self.top_k
        final_k = final_top_k or self.final_top_k
        use_rerank = self.enable_rerank if with_rerank is None else with_rerank

        # ---- 第一层：双路召回 ----
        with Timer() as t:
            q_vec = self.embedder.encode_query(query)
            vec_hits = self.vector_store.search(q_vec, top_k=top_k, doc_filter=doc_filter)
            bm_hits = self.bm25.search(query, top_k=top_k)
        recall_latency = t.elapsed

        if doc_filter:
            bm_hits = [(cid, s) for cid, s in bm_hits
                       if cid in self._chunks and self._chunks[cid].doc_id in doc_filter][:top_k]

        metrics.observe("retrieval.latency", recall_latency)
        metrics.incr("retrieval.total")
        if vec_hits or bm_hits:
            metrics.incr("retrieval.hit")

        vec_ids = [cid for cid, _ in vec_hits]
        bm_ids = [cid for cid, _ in bm_hits]

        # ---- 第二层：融合 ----
        if self.fusion_mode == "weighted":
            fused = self._weighted_fusion(vec_hits, bm_hits)
        else:
            fused = reciprocal_rank_fusion([vec_ids, bm_ids], k=self.rrf_k,
                                           weights=[self.vector_weight, self.bm25_weight])
        candidates = self._assemble(fused, vec_hits, bm_hits, limit=max(top_k, final_k))

        # ---- 第三层：精排 ----
        if use_rerank and self.reranker is not None:
            result = self.reranker.rerank(query, candidates, top_n=self.reranker.top_n)
        else:
            result = candidates[:final_k]
            for i, c in enumerate(result, 1):
                c.rank = i

        logger.debug("检索[%s] 向量命中%s BM25命中%s 融合%s 最终%s",
                     query[:20], len(vec_hits), len(bm_hits), len(candidates), len(result))
        return result

    # ---------------- 融合实现 ----------------
    def _weighted_fusion(self, vec_hits: List[Tuple[str, float]],
                         bm_hits: List[Tuple[str, float]]) -> Dict[str, float]:
        """加权分数融合（需先归一化，作为 RRF 的对照实验）。"""
        def norm(hits):
            if not hits:
                return {}
            vals = [s for _, s in hits]
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            return {cid: (s - lo) / span for cid, s in hits}

        nv, nb = norm(vec_hits), norm(bm_hits)
        out: Dict[str, float] = {}
        for cid, s in nv.items():
            out[cid] = out.get(cid, 0.0) + self.vector_weight * s
        for cid, s in nb.items():
            out[cid] = out.get(cid, 0.0) + self.bm25_weight * s
        return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))

    def _assemble(self, fused: Dict[str, float],
                  vec_hits: List[Tuple[str, float]],
                  bm_hits: List[Tuple[str, float]], limit: int) -> List[RetrievedChunk]:
        """把融合结果组装成 RetrievedChunk（补齐各路分数明细）。"""
        vec_score = dict(vec_hits)
        bm_score = dict(bm_hits)
        vec_rank = {cid: i + 1 for i, (cid, _) in enumerate(vec_hits)}
        bm_rank = {cid: i + 1 for i, (cid, _) in enumerate(bm_hits)}

        out: List[RetrievedChunk] = []
        for cid, score in list(fused.items())[:limit]:
            chunk = self._chunks.get(cid) or self.vector_store.get_chunk(cid)
            if chunk is None:
                continue
            in_vec, in_bm = cid in vec_score, cid in bm_score
            out.append(RetrievedChunk(
                chunk=chunk,
                score=float(score),
                rrf_score=float(score),
                vector_score=float(vec_score.get(cid, 0.0)),
                vector_rank=vec_rank.get(cid, 0),
                bm25_score=float(bm_score.get(cid, 0.0)),
                bm25_rank=bm_rank.get(cid, 0),
                source="both" if (in_vec and in_bm) else ("vector" if in_vec else "bm25"),
            ))
        return out

    # ---------------- 管理 ----------------
    def delete_by_doc(self, doc_id: str) -> int:
        n1 = self.vector_store.delete_by_doc(doc_id)
        n2 = self.bm25.delete_by_doc(doc_id)
        for cid in list(self._chunks):
            if self._chunks[cid].doc_id == doc_id:
                self._chunks.pop(cid, None)
        return max(n1, n2)

    def save(self, path: str) -> None:
        self.vector_store.save(path)
        self.bm25.save(path)

    @classmethod
    def load(cls, path: str, embedder: BaseEmbedder, config: Optional[Config] = None) -> "HybridRetriever":
        vs = VectorStore.load(path)
        bm = BM25Retriever.load(path)
        obj = cls(embedder=embedder, config=config, vector_store=vs, bm25=bm)
        obj._chunks.update(vs._chunks)
        return obj

    # ---------------- 元信息 ----------------
    @property
    def size(self) -> int:
        return self.vector_store.size

    @property
    def num_docs(self) -> int:
        return self.vector_store.num_docs

    def is_empty(self) -> bool:
        return self.size == 0

    def list_documents(self) -> List[Dict[str, Any]]:
        """知识库文档列表（前端管理面板用）。"""
        docs: Dict[str, Dict[str, Any]] = {}
        for chunk in self._chunks.values():
            d = docs.setdefault(chunk.doc_id, {
                "doc_id": chunk.doc_id, "title": chunk.doc_title,
                "source": chunk.source, "chunks": 0, "pages": set(),
            })
            d["chunks"] += 1
            d["pages"].add(chunk.page)
        for d in docs.values():
            d["pages"] = sorted(d["pages"])
        return list(docs.values())

    def get_doc_chunks(self, doc_id: str) -> List[Chunk]:
        """取回某个文档的全部块（论文元信息提取 / 摘要生成工具用）。"""
        return sorted([c for c in self._chunks.values() if c.doc_id == doc_id],
                      key=lambda c: (c.page, c.chunk_index))

    def find_doc_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """按标题模糊匹配文档（支持用户只说论文名的一部分）。"""
        key = title.strip().lower()
        for d in self.list_documents():
            if key and (key in d["title"].lower() or d["title"].lower() in key):
                return d
        # 再退一步：按 doc_id 前缀匹配
        for d in self.list_documents():
            if d["doc_id"].startswith(key):
                return d
        return None

    def stats(self) -> Dict[str, Any]:
        s = self.vector_store.stats()
        s.update({
            "bm25_size": self.bm25.size,
            "reranker": self.reranker.active_provider,
            "reranker_model": self.reranker.model_name,
            "fusion_mode": self.fusion_mode,
        })
        return s
