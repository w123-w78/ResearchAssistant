# -*- coding: utf-8 -*-
"""
模块二 5.2.3：语义缓存（Semantic Cache，持久化到 SQLite）

问题：科研问答重复提问率高（"这篇论文用了什么数据集"会反复问），
      每次都跑"检索 + LLM 生成"既慢又费钱。

方案：把 (问题向量 -> 答案) 存起来；新问题的向量与历史问题比对，
      余弦相似度超过阈值（默认 0.93）直接返回旧答案，跳过整条 RAG 链路。

收益：命中时延迟从 ~2s 降到 ~50ms，Token 消耗为 0。

存储：SQLite 的 cache 表（key / query / answer / citations / vector / kb_version）
      - 启动时从库里加载向量到内存工作集，命中与淘汰后在库里同步
      - 知识库版本（kb_version）变化时整体失效，避免"上传新论文后返回旧答案"
"""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import Config
from ..utils.embedding import BaseEmbedder
from ..utils.logger import get_logger, metrics

logger = get_logger("research_assistant.cache")


class SemanticCache:
    """基于 Embedding 相似度的语义缓存（LRU + TTL + SQLite 持久化）。"""

    def __init__(self, embedder: BaseEmbedder, db=None, threshold: float = 0.93,
                 max_size: int = 500, ttl_seconds: int = 86400,
                 enabled: bool = True, config: Optional[Config] = None):
        self.config = config or Config.instance()
        self.embedder = embedder
        self.db = db
        self.threshold = threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.enabled = enabled and bool(self.config.get("storage.enable_cache", True))

        self._store: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._kb_version: str = "init"
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

        self._load()

    # ---------------- 知识库版本 ----------------
    def set_kb_version(self, version: str) -> None:
        """知识库变更时调用，使旧缓存整体失效（一致性保障）。"""
        if version != self._kb_version:
            with self._lock:
                logger.info("知识库版本变更 %s -> %s，清空语义缓存",
                            self._kb_version[:8], version[:8])
                self._kb_version = version
                self._store.clear()
                self._vectors.clear()
                if self.db is not None:
                    self.db.cache_clear()
        elif not self._store and self.db is not None:
            self._kb_version = version

    @staticmethod
    def compute_kb_version(doc_ids: List[str], num_chunks: int) -> str:
        """根据文档集合计算知识库指纹。"""
        raw = "|".join(sorted(doc_ids)) + f"#{num_chunks}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    # ---------------- 读写 ----------------
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """查询缓存；命中返回 {"answer", "citations", "cache_score"}。"""
        if not self.enabled or not self._store:
            self.misses += 1
            return None
        qv = self.embedder.encode_query(query)
        with self._lock:
            best_key, best_score = self._best_match(qv)

        if best_key is None or best_score < self.threshold:
            self.misses += 1
            metrics.incr("cache.miss")
            return None

        entry = self._store[best_key]
        if self.ttl > 0 and (time.time() - entry["created_at"]) > self.ttl:
            with self._lock:
                self._store.pop(best_key, None)
                self._vectors.pop(best_key, None)
                if self.db is not None:
                    self.db.cache_delete(best_key)
            self.misses += 1
            metrics.incr("cache.miss")
            return None

        with self._lock:
            entry["created_at"] = time.time()   # LRU：刷新访问时间
            if self.db is not None:
                try:
                    self.db.cache_hit(best_key)
                except Exception:  # noqa: BLE001
                    pass
        self.hits += 1
        metrics.incr("cache.hit")
        logger.debug("语义缓存命中 (%.4f >= %.2f): %s", best_score, self.threshold, query[:30])
        return {**entry["value"], "cache_score": float(best_score), "from_cache": True}

    def put(self, query: str, value: Dict[str, Any]) -> None:
        """写入缓存（内存 + SQLite）。"""
        if not self.enabled:
            return
        key = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()[:16]
        try:
            vec = self.embedder.encode_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("缓存写入失败（向量化异常）: %s", exc)
            return

        with self._lock:
            self._store[key] = {"query": query, "value": value, "created_at": time.time()}
            self._vectors[key] = vec
            # LRU 淘汰
            while len(self._store) > self.max_size:
                oldest = min(self._store, key=lambda k: self._store[k]["created_at"])
                self._store.pop(oldest, None)
                self._vectors.pop(oldest, None)
                if self.db is not None:
                    self.db.cache_delete(oldest)
            if self.db is not None:
                try:
                    self.db.cache_put(
                        key=key, query=query,
                        answer=value.get("answer", ""),
                        citations=value.get("citations", []),
                        vector=vec.astype(np.float32).tobytes(),
                        kb_version=self._kb_version,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("缓存持久化失败: %s", exc)

    def _best_match(self, qv: np.ndarray) -> Tuple[Optional[str], float]:
        """在缓存工作集中找最相似条目。"""
        best_key, best_score = None, -1.0
        for key, vec in self._vectors.items():
            score = float(np.dot(qv, vec))
            if score > best_score:
                best_score, best_key = score, key
        return best_key, best_score

    # ---------------- 加载 / 管理 ----------------
    def _load(self) -> None:
        """启动时从 SQLite 恢复缓存（向量以 BLOB 存储）。"""
        if self.db is None or not self.enabled:
            return
        try:
            rows = self.db.cache_all()
        except Exception as exc:  # noqa: BLE001
            logger.warning("缓存加载失败: %s", exc)
            return
        n = 0
        for r in rows:
            blob = r.get("vector")
            if not blob:
                continue
            try:
                vec = np.frombuffer(blob, dtype=np.float32)
            except Exception:  # noqa: BLE001
                continue
            key = r["key"]
            self._vectors[key] = vec
            self._store[key] = {
                "query": r.get("query", ""),
                "value": {"answer": r.get("answer", ""),
                          "citations": _safe_json(r.get("citations"))},
                "created_at": r.get("updated_at") or r.get("created_at") or time.time(),
            }
            n += 1
        if n:
            logger.info("已从 SQLite 恢复 %s 条语义缓存", n)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._vectors.clear()
            if self.db is not None:
                self.db.cache_clear()
        logger.info("语义缓存已清空")

    def prune(self) -> int:
        """清理过期与超量条目。"""
        if self.db is None:
            return 0
        return self.db.cache_prune(max_size=self.max_size, ttl_seconds=self.ttl)

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "size": len(self._store),
            "db_size": self.db.cache_count() if self.db is not None else 0,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "threshold": self.threshold,
            "kb_version": self._kb_version[:8],
        }

    def __len__(self) -> int:
        return len(self._store)


def _safe_json(raw) -> Any:
    """缓存里的 citations 可能已是 dict 或 JSON 字符串。"""
    import json

    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return []
