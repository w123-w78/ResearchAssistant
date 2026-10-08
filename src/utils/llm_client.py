# -*- coding: utf-8 -*-
"""
LLM 客户端 —— 阿里云 DashScope（通义千问，OpenAI 兼容接口）
-------------------------------------------------------------
统一入口：build_llm() / DashScopeLLM

接口约定：
    llm.chat(messages) -> str
    llm.stream(messages) -> Iterator[str]
    llm.count_tokens(text) -> int

接入说明：
  - 兼容接口地址：https://dashscope.aliyuncs.com/compatible-mode/v1
  - 常用模型：qwen-plus（均衡）/ qwen-turbo（快、便宜）/ qwen-max（最强）
  - API Key：优先读环境变量 DASHSCOPE_API_KEY，其次读 config.yaml 的 llm.api_key
    （不要把 Key 提交到 Git！建议配 .env 或系统环境变量）

工程特性：
  - 指数退避重试（最多 max_retries 次）
  - Token 与耗时指标自动采集（模块四可观测性）
  - 流式输出失败自动降级为整段返回
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

from .config import Config
from .env import ensure_loaded, get_api_key
from .logger import Timer, get_logger, metrics
from .text_utils import estimate_tokens

logger = get_logger("research_assistant.llm")

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class LLMError(RuntimeError):
    """LLM 调用异常（供上层降级策略捕获）。"""


class BaseLLM:
    """LLM 基类：统一接口 + 重试 + 指标采集。"""

    def __init__(self, config: Optional[Config] = None, **kwargs):
        self.config = config or Config.instance()
        llm_cfg = self.config.section("llm")
        self.model: str = kwargs.get("model", llm_cfg.get("model", "qwen-plus"))
        self.temperature: float = float(kwargs.get("temperature", llm_cfg.get("temperature", 0.2)))
        self.top_p: float = float(kwargs.get("top_p", llm_cfg.get("top_p", 0.85)))
        self.max_tokens: int = int(kwargs.get("max_tokens", llm_cfg.get("max_tokens", 2048)))
        self.timeout: int = int(kwargs.get("timeout", llm_cfg.get("timeout", 120)))
        self.max_retries: int = int(kwargs.get("max_retries", llm_cfg.get("max_retries", 2)))

    # -------- 子类实现 --------
    def _chat_once(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    def _stream_once(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        raise NotImplementedError

    # -------- 对外接口 --------
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """带重试与指标采集的同步对话。"""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with Timer() as t:
                    text = self._chat_once(messages, **kwargs)
                prompt_text = "\n".join(m.get("content", "") for m in messages)
                tokens = self.count_tokens(prompt_text) + self.count_tokens(text)
                metrics.incr("llm.calls")
                metrics.incr("llm.tokens", tokens)
                metrics.observe("llm.latency", t.elapsed)
                return text
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("LLM 调用失败(%s/%s): %s", attempt + 1, self.max_retries + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 5))
        raise LLMError(f"LLM 调用失败（已重试 {self.max_retries} 次）: {last_err}")

    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """流式输出；失败时自动降级为一次性返回后分片输出。"""
        try:
            for chunk in self._stream_once(messages, **kwargs):
                yield chunk
            metrics.incr("llm.calls")
        except Exception as exc:  # noqa: BLE001
            logger.warning("流式输出失败，降级为普通输出: %s", exc)
            text = self.chat(messages, **kwargs)
            for i in range(0, len(text), 16):
                yield text[i:i + 16]

    def count_tokens(self, text: str) -> int:
        """Token 估算（中英混排）。"""
        return estimate_tokens(text)

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.model})"

    def __repr__(self) -> str:  # pragma: no cover
        return self.name


class DashScopeLLM(BaseLLM):
    """
    阿里云 DashScope（通义千问）OpenAI 兼容接口客户端。

    环境变量：
        DASHSCOPE_API_KEY=sk-xxxx
    """

    def __init__(self, config: Optional[Config] = None, **kwargs):
        super().__init__(config, **kwargs)
        llm_cfg = self.config.section("llm")
        self.base_url: str = (
            kwargs.get("base_url") or llm_cfg.get("base_url") or DASHSCOPE_BASE_URL
        ).rstrip("/")
        ensure_loaded()   # 确保 .env 已加载
        self.api_key: str = (
            kwargs.get("api_key")
            or (llm_cfg.get("api_key") or "").strip()
            or get_api_key("DASHSCOPE_API_KEY")
        ).strip()
        if not self.api_key:
            raise LLMError(
                "未配置 DashScope API Key。任选一种方式配置：\n"
                "  1) 项目根目录创建 .env 并写入：DASHSCOPE_API_KEY=sk-xxxxxx（推荐）\n"
                "  2) 设置环境变量：set DASHSCOPE_API_KEY=sk-xxxxxx\n"
                "  3) 在 config.yaml 填写 llm.api_key: sk-xxxxxx\n"
                "（通义千问控制台 -> API-KEY 管理 获取，切勿提交到代码仓库）"
            )

    # -------- 请求构造 --------
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def _payload(self, messages: List[Dict[str, str]], stream: bool, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if stream:
            # 通义千问流式建议开启 usage 统计，便于 Token 核算
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    # -------- 调用实现 --------
    def _chat_once(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = requests.post(self._endpoint(), headers=self._headers(),
                             json=self._payload(messages, False, kwargs),
                             timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise LLMError(f"DashScope 返回错误: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"DashScope 返回空 choices: {data}")
        # 记录真实 Token 消耗（若接口返回 usage）
        usage = data.get("usage") or {}
        if usage.get("total_tokens"):
            metrics.incr("llm.tokens.real", int(usage["total_tokens"]))
        return (choices[0].get("message") or {}).get("content", "")

    def _stream_once(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        with requests.post(self._endpoint(), headers=self._headers(),
                           json=self._payload(messages, True, kwargs),
                           stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise LLMError(f"DashScope 流式错误: {chunk['error']}")
                # usage 分片没有 choices
                if chunk.get("usage"):
                    metrics.incr("llm.tokens.real", int(chunk["usage"].get("total_tokens", 0)))
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                piece = delta.get("content", "")
                if piece:
                    yield piece

    def health(self) -> Dict[str, Any]:
        """探活：发一个极短请求确认 Key 与网络可用。"""
        try:
            t0 = time.perf_counter()
            self._chat_once([{"role": "user", "content": "ping"}], max_tokens=4)
            return {"status": "ok", "model": self.model,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "model": self.model, "error": str(exc)}


def build_llm(config: Optional[Config] = None, **kwargs) -> DashScopeLLM:
    """构建 LLM 客户端（本项目统一使用 DashScope）。"""
    cfg = config or Config.instance()
    llm = DashScopeLLM(config=cfg, **kwargs)
    logger.info("初始化 LLM: DashScope / model=%s", llm.model)
    return llm
