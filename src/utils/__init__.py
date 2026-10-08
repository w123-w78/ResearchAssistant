# -*- coding: utf-8 -*-
"""通用工具包：配置 / 日志 / LLM / Embedding / 文本处理 / 指标采集。"""
from .config import Config
from .embedding import BaseEmbedder, DashScopeEmbedder, EmbeddingError, build_embedder
from .llm_client import BaseLLM, LLMError, DashScopeLLM, build_llm
from .logger import MetricsCollector, RequestLogger, Timer, get_logger, metrics, setup_logging
from .text_utils import estimate_tokens, extract_keywords, split_sentences, truncate

__all__ = [
    "Config", "setup_logging", "get_logger", "RequestLogger", "Timer", "MetricsCollector", "metrics",
    "BaseLLM", "DashScopeLLM", "LLMError", "build_llm",
    "BaseEmbedder", "DashScopeEmbedder", "EmbeddingError", "build_embedder",
    "estimate_tokens", "extract_keywords", "split_sentences", "truncate",
]
