# -*- coding: utf-8 -*-
"""
Embedding 客户端 —— 阿里云 DashScope text-embedding-v3
-------------------------------------------------------
统一入口：build_embedder() / DashScopeEmbedder

接口约定：
    emb.encode(texts) -> np.ndarray  (N, dim)，已 L2 归一化
    emb.encode_query(text) -> np.ndarray (dim,)
    emb.dim -> int

模型说明（text-embedding-v3）：
  - 维度 1024（可选 1024 / 768 / 512，需在 config.yaml 的 embedding.dim 指定）
  - 单次请求最多 10 条文本 -> 自动分批（config.embedding.batch_size，默认 10）
  - 支持中英双语，中文语义检索表现优秀

工程特性：
  - 自动分批 + 失败重试
  - 输出统一做 L2 归一化，使余弦相似度 == 向量内积，检索层可直接矩阵乘法
  - 向量维度首次调用时自动探测，避免配置与实际不一致
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from .config import Config
from .env import ensure_loaded, get_api_key
from .logger import Timer, get_logger, metrics

logger = get_logger("research_assistant.embedding")

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class EmbeddingError(RuntimeError):
    """Embedding 调用异常。"""


class BaseEmbedder:
    """Embedding 基类：统一接口 + 分批 + 归一化。"""

    def __init__(self, config: Optional[Config] = None, **kwargs):
        self.config = config or Config.instance()
        emb_cfg = self.config.section("embedding")
        self.model_name: str = kwargs.get("model", emb_cfg.get("model", "text-embedding-v3"))
        self.batch_size: int = int(kwargs.get("batch_size", emb_cfg.get("batch_size", 10)))
        self.normalize: bool = bool(kwargs.get("normalize", emb_cfg.get("normalize", True)))
        self._dim: int = int(kwargs.get("dim", emb_cfg.get("dim", 1024)))
        self.max_retries: int = int(kwargs.get("max_retries", emb_cfg.get("max_retries", 2)))

    # -------- 子类实现 --------
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    # -------- 对外接口 --------
    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Any, batch_size: Optional[int] = None,
               show_progress: bool = False) -> np.ndarray:
        """批量编码；自动分批 + 归一化。"""
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if t else " " for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        bs = max(1, batch_size or self.batch_size)
        vecs: List[np.ndarray] = []
        with Timer() as t:
            for i in range(0, len(texts), bs):
                vecs.append(self._encode_batch(texts[i:i + bs]))
                if show_progress:
                    logger.debug("向量化进度 %s/%s", min(i + bs, len(texts)), len(texts))
        out = np.vstack(vecs).astype(np.float32)
        if self.normalize:
            out = _l2_normalize(out)
        metrics.incr("embedding.calls")
        metrics.incr("embedding.texts", len(texts))
        metrics.observe("embedding.latency", t.elapsed)
        return out

    def encode_query(self, text: str) -> np.ndarray:
        """单条查询编码。"""
        return self.encode([text])[0]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度。"""
        a, b = a.reshape(-1), b.reshape(-1)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.model_name}, dim={self.dim})"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class DashScopeEmbedder(BaseEmbedder):
    """阿里云 DashScope text-embedding-v3（OpenAI 兼容 /embeddings 接口）。"""

    def __init__(self, config: Optional[Config] = None, **kwargs):
        super().__init__(config, **kwargs)
        emb_cfg = self.config.section("embedding")
        llm_cfg = self.config.section("llm")
        self.base_url: str = (
            kwargs.get("base_url")
            or emb_cfg.get("base_url")
            or llm_cfg.get("base_url")
            or DASHSCOPE_BASE_URL
        ).rstrip("/")
        # Embedding 默认复用 LLM 的 Key（同一个 DashScope 账号）
        ensure_loaded()   # 确保 .env 已加载
        self.api_key: str = (
            kwargs.get("api_key")
            or (emb_cfg.get("api_key") or "").strip()
            or (llm_cfg.get("api_key") or "").strip()
            or get_api_key("DASHSCOPE_API_KEY")
        ).strip()
        self.timeout: int = int(kwargs.get("timeout", emb_cfg.get("timeout", 60)))
        if not self.api_key:
            raise EmbeddingError(
                "未配置 DashScope API Key（Embedding 与 LLM 共用）。请在项目根目录 .env 写入 "
                "DASHSCOPE_API_KEY=sk-xxxxxx，或设置环境变量，或填写 config.yaml 的 llm.api_key。"
            )
        self._dim_detected = False

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                payload: Dict[str, Any] = {"model": self.model_name, "input": texts}
                # text-embedding-v3 支持指定维度（1024/768/512）
                if self._dim in (1024, 768, 512):
                    payload["dimension"] = self._dim
                resp = requests.post(f"{self.base_url}/embeddings", headers=self._headers(),
                                     json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("error"):
                    raise EmbeddingError(f"DashScope 返回错误: {data['error']}")
                items = data.get("data") or []
                if not items:
                    raise EmbeddingError(f"DashScope 返回空向量: {data}")
                items.sort(key=lambda x: x.get("index", 0))
                arr = np.asarray([d["embedding"] for d in items], dtype=np.float32)
                # 首次调用用真实维度校正配置
                if not self._dim_detected:
                    self._dim_detected = True
                    if arr.shape[1] != self._dim:
                        logger.warning("Embedding 实际维度 %s 与配置 %s 不一致，已自动校正",
                                       arr.shape[1], self._dim)
                        self._dim = int(arr.shape[1])
                return arr
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("Embedding 调用失败(%s/%s): %s", attempt + 1, self.max_retries + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
        raise EmbeddingError(f"Embedding 调用失败（已重试 {self.max_retries} 次）: {last_err}")

    def health(self) -> Dict[str, Any]:
        """探活：编码一条短文本。"""
        try:
            t0 = time.perf_counter()
            v = self.encode(["健康检查"])
            return {"status": "ok", "model": self.model_name, "dim": int(v.shape[1]),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "model": self.model_name, "error": str(exc)}


def build_embedder(config: Optional[Config] = None, **kwargs) -> DashScopeEmbedder:
    """构建 Embedding 客户端（本项目统一使用 DashScope text-embedding-v3）。"""
    cfg = config or Config.instance()
    emb = DashScopeEmbedder(config=cfg, **kwargs)
    logger.info("初始化 Embedding: DashScope / model=%s", emb.model_name)
    return emb
