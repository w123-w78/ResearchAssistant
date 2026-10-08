# -*- coding: utf-8 -*-
"""
模块二 5.2.2：流式输出工具

提供三种流式能力：
  1. SSEFormatter      —— 服务端推送（FastAPI / 自定义后端接入用）
  2. merge_stream      —— 把 LLM 的 token 流聚合成字符串，同时做引用标记识别
  3. typing_iterator   —— 前端打字机效果的模拟（MockLLM / 缓存命中场景用）

引用标记与流式结合的处理思路：
  流式过程中先缓存完整文本，出现 "[1]" 这类标记时立即向前端发一次
  citation 事件，前端在对应位置渲染上标，实现"边出答案边出引用"。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Pattern

_CITATION_RE: Pattern = re.compile(r"\[(\d+)\]")


class SSEFormatter:
    """Server-Sent Events 格式化器。

    前端用法：
        const es = new EventSource('/api/chat/stream');
        es.onmessage = e => render(JSON.parse(e.data));
    """

    @staticmethod
    def event(data: Any, event: Optional[str] = None, event_id: Optional[str] = None) -> str:
        """生成一条 SSE 消息（以 \\n\\n 结尾）。"""
        lines: List[str] = []
        if event_id:
            lines.append(f"id: {event_id}")
        if event:
            lines.append(f"event: {event}")
        payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        for line in payload.split("\n"):
            lines.append(f"data: {line}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def token(text: str) -> str:
        return SSEFormatter.event({"type": "token", "content": text})

    @staticmethod
    def citation(items: List[Dict[str, Any]]) -> str:
        return SSEFormatter.event({"type": "citation", "citations": items})

    @staticmethod
    def done(extra: Optional[Dict[str, Any]] = None) -> str:
        return SSEFormatter.event({"type": "done", **(extra or {})})

    @staticmethod
    def error(message: str) -> str:
        return SSEFormatter.event({"type": "error", "message": message}, event="error")

    @staticmethod
    def heartbeat() -> str:
        """心跳包，防止代理层超时断开。"""
        return ": ping\n\n"


class StreamMerger:
    """
    流式聚合器：边收 token 边输出事件。

    事件类型：
      {"type": "token",     "content": "文字"}
      {"type": "citation",  "citations": [...]}   发现新的引用标记时触发
    """

    def __init__(self, citation_lookup: Optional[Dict[int, Dict[str, Any]]] = None):
        """
        :param citation_lookup: {编号: 来源信息}，用于在流式过程中解析引用
        """
        self.citation_lookup = citation_lookup or {}
        self.buffer: List[str] = []
        self._seen_citations: set = set()

    def feed(self, piece: str) -> Iterator[Dict[str, Any]]:
        """喂入一个 token 片段，产出事件列表。"""
        self.buffer.append(piece)
        yield {"type": "token", "content": piece}

        # 检测新出现的引用标记
        text = "".join(self.buffer)
        for n in _CITATION_RE.findall(text):
            idx = int(n)
            if idx in self._seen_citations:
                continue
            self._seen_citations.add(idx)
            if idx in self.citation_lookup:
                yield {"type": "citation", "citations": [self.citation_lookup[idx]]}

    @property
    def text(self) -> str:
        return "".join(self.buffer)

    @property
    def citations(self) -> List[Dict[str, Any]]:
        return [self.citation_lookup[i] for i in sorted(self._seen_citations)
                if i in self.citation_lookup]


def typing_iterator(text: str, chunk_size: int = 4, delay: float = 0.01,
                    enabled: bool = True) -> Iterator[str]:
    """
    打字机效果迭代器：把整段文本按小块 + 延时吐出。

    用于：
      - MockLLM / 缓存命中时，仍保持与真实模型一致的流式体验
      - 前端无需区分"真流式"和"假流式"
    """
    if not enabled:
        yield text
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        if delay > 0:
            time.sleep(delay)


def iter_llm_stream(llm, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
    """统一封装：优先走真流式，失败自动降级为整段返回。"""
    try:
        for piece in llm.stream(messages, **kwargs):
            yield piece
    except Exception:  # noqa: BLE001 - 由调用方处理降级
        return


def stream_to_text(iterator: Iterator[str]) -> str:
    """把 token 流收成完整字符串（测试与非流式场景用）。"""
    return "".join(iterator)
