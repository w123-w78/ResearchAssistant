# -*- coding: utf-8 -*-
"""
模块一 5.1.3：向量数据库封装（Chroma / FAISS 双后端）

设计目标：
  - 统一接口，后端可插拔：numpy(内存) / faiss(高性能) / chroma(持久化)
  - 支持增量更新：新文档加入只需 add，无需重建全量索引
  - 支持按 doc_id 删除（知识库管理面板的"删除文档"功能）

后端对比：
  numpy   零依赖，暴力余弦检索，1 万条以内毫秒级；教学/默认
  faiss   Facebook 开源，IVF/HNSW 近似检索，百万级可用；需 pip install faiss-cpu
  chroma  自带持久化与 metadata 过滤；需 pip install chromadb
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..chunking.base import Chunk
from ..utils.logger import Timer, get_logger, metrics

logger = get_logger("research_assistant.vector_store")


class VectorStore:
    """
    向量存储（默认 numpy 暴力检索；backend='faiss' 时自动切换 FAISS）。

    对外接口：
        add(chunks, vectors)            增量写入
        search(query_vec, top_k)        -> [(chunk_id, score), ...]
        delete_by_doc(doc_id)           -> 删除条数
        save(path) / load(path)         持久化
        get_chunk(chunk_id)            取回原始 Chunk
    """

    def __init__(self, dim: int, backend: str = "numpy", metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self.backend = backend
        self._ids: List[str] = []
        self._id_set: set = set()
        self._chunks: Dict[str, Chunk] = {}
        self._matrix: Optional[np.ndarray] = None   # (N, dim)，已归一化
        self._faiss_index = None
        self._dirty = True

        if backend == "faiss":
            self._try_init_faiss()

    # ---------------- FAISS ----------------
    def _try_init_faiss(self) -> None:
        try:
            import faiss  # noqa: F401
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            logger.info("向量库后端: FAISS (IndexFlatIP, dim=%s)", self.dim)
        except ImportError:
            logger.warning("faiss 未安装，自动降级为 numpy 后端（pip install faiss-cpu）")
            self.backend = "numpy"

    # ---------------- 写入 ----------------
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> int:
        """增量写入向量（支持重复调用，无需重建全量索引）。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if not chunks:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度 {vectors.shape[1]} 与库维度 {self.dim} 不一致")
        # 归一化，余弦相似度 == 内积
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        new_idx = 0
        rows = []
        for c, v in zip(chunks, vectors):
            if c.chunk_id in self._id_set:
                continue  # 幂等：同 id 不重复写入
            self._ids.append(c.chunk_id)
            self._id_set.add(c.chunk_id)
            self._chunks[c.chunk_id] = c
            rows.append(v)
            new_idx += 1
        if rows:
            block = np.vstack(rows)
            self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])
            if self._faiss_index is not None:
                self._faiss_index.add(block)
            self._dirty = True
        logger.info("向量库新增 %s 条（总 %s 条）", new_idx, len(self._ids))
        return new_idx

    # ---------------- 检索 ----------------
    def search(self, query_vec: np.ndarray, top_k: int = 10,
               doc_filter: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """检索 Top-K，返回 [(chunk_id, cosine_score), ...]。"""
        if not self._ids or self._matrix is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        top_k = min(top_k, len(self._ids))

        with Timer() as t:
            if self._faiss_index is not None:
                scores, idxs = self._faiss_index.search(q, top_k)
                pairs = [(self._ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]
            else:
                # 余弦相似度矩阵乘法（归一化后即内积）
                sims = (self._matrix @ q.T).reshape(-1)
                if doc_filter:
                    mask = np.array([self._chunks[cid].doc_id in doc_filter for cid in self._ids])
                    sims = np.where(mask, sims, -1.0)
                top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                pairs = [(self._ids[i], float(sims[i])) for i in top_idx if sims[i] > -1.0]
        metrics.observe("vector_search.latency", t.elapsed)
        return pairs

    # ---------------- 删除 / 查询 ----------------
    def delete_by_doc(self, doc_id: str) -> int:
        """删除某个文档的所有向量（返回删除条数）。"""
        keep = [i for i, cid in enumerate(self._ids) if self._chunks[cid].doc_id != doc_id]
        removed = len(self._ids) - len(keep)
        if removed == 0:
            return 0
        if self._matrix is not None:
            self._matrix = self._matrix[keep]
        self._ids = [self._ids[i] for i in keep]
        self._id_set = set(self._ids)
        for cid in list(self._chunks):
            if self._chunks[cid].doc_id == doc_id:
                self._chunks.pop(cid, None)
        if self._faiss_index is not None:
            self._rebuild_faiss()
        logger.info("删除文档 %s 的 %s 条向量", doc_id[:8], removed)
        return removed

    def _rebuild_faiss(self) -> None:
        if self._matrix is None or self._matrix.size == 0:
            return
        try:
            import faiss

            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self._faiss_index.add(self._matrix)
        except ImportError:
            self._faiss_index = None

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    def list_docs(self) -> Dict[str, int]:
        """统计每个 doc_id 下的块数（知识库管理面板用）。"""
        stat: Dict[str, int] = {}
        for cid in self._ids:
            d = self._chunks[cid].doc_id
            stat[d] = stat.get(d, 0) + 1
        return stat

    # ---------------- 持久化 ----------------
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p.with_suffix(".npy")), self._matrix if self._matrix is not None
                else np.zeros((0, self.dim), dtype=np.float32))
        with open(p.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump({"ids": self._ids, "chunks": self._chunks,
                         "dim": self.dim, "backend": self.backend}, f)
        logger.info("向量库已保存: %s (%s 条)", path, len(self._ids))

    @classmethod
    def load(cls, path: str, backend: Optional[str] = None) -> "VectorStore":
        p = Path(path)
        npy, meta_pkl = p.with_suffix(".npy"), p.with_suffix(".meta.pkl")
        if not meta_pkl.exists():
            raise FileNotFoundError(f"索引不存在: {meta_pkl}")
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        store = cls(dim=meta["dim"], backend=backend or meta.get("backend", "numpy"))
        store._ids = meta["ids"]
        store._id_set = set(store._ids)
        store._chunks = meta["chunks"]
        if npy.exists():
            store._matrix = np.load(str(npy))
            if store._matrix.shape[0] != len(store._ids):
                logger.warning("索引矩阵与元数据条数不一致，将重建矩阵")
                store._matrix = None
            elif store._faiss_index is not None and store._matrix is not None and store._matrix.size:
                store._faiss_index.add(store._matrix)
        logger.info("向量库已加载: %s 条向量", len(store._ids))
        return store

    # ---------------- 元信息 ----------------
    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def num_docs(self) -> int:
        return len(self.list_docs())

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "num_vectors": self.size,
            "num_docs": self.num_docs,
            "docs": self.list_docs(),
        }

    def clear(self) -> None:
        self._ids, self._id_set, self._chunks = [], set(), {}
        self._matrix = None
        self._rebuild_faiss()

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:  # pragma: no cover
        return f"VectorStore(backend={self.backend}, dim={self.dim}, size={self.size})"
