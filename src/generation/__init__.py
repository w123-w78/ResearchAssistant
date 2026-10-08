# -*- coding: utf-8 -*-
"""模块二：RAG 生成层统一入口。"""
from .cache import SemanticCache
from .prompt_template import (LOW_SCORE_TEMPLATE, NO_CONTEXT_TEMPLATE, RAG_TEMPLATE,
                              SYSTEM_ROLE, build_context, build_rag_messages,
                              build_summary_messages, parse_citations)
from .rag_pipeline import RAGAnswer, RAGPipeline, Citation
from .streaming import (SSEFormatter, StreamMerger, stream_to_text, typing_iterator)

__all__ = [
    "RAGPipeline", "RAGAnswer", "Citation", "SemanticCache",
    "SYSTEM_ROLE", "RAG_TEMPLATE", "NO_CONTEXT_TEMPLATE", "LOW_SCORE_TEMPLATE",
    "build_context", "build_rag_messages", "build_summary_messages", "parse_citations",
    "SSEFormatter", "StreamMerger", "typing_iterator", "stream_to_text",
]
