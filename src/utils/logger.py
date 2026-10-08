# -*- coding: utf-8 -*-
"""
日志与可观测性基础设施
----------------------
两类记录：
  1. app.log         —— 常规运行日志（分级、滚动文件、控制台）
  2. request_logs 表 —— 结构化请求日志（SQLite）
                        记录：问题 / 命中片段 / Agent 轨迹 / Token / 耗时 / 降级原因
                        模块五评测时可直接用 SQL 聚合与 Bad Case 筛选

另外提供 MetricsCollector，用于进程内统计 Token、检索命中率、工具成功率。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config

_LOG_LOCK = threading.Lock()
_INITIALIZED = False


def setup_logging(config: Optional[Config] = None, log_dir: Optional[str] = None) -> logging.Logger:
    """初始化根 logger（幂等）。"""
    global _INITIALIZED
    with _LOG_LOCK:
        cfg = config or Config.instance()
        level = str(cfg.get("logging.level", "INFO")).upper()
        console = bool(cfg.get("logging.console", True))
        file_path = log_dir or cfg.get("logging.file", "./logs/app.log")
        max_bytes = int(cfg.get("logging.max_bytes", 10 * 1024 * 1024))
        backup = int(cfg.get("logging.backup_count", 5))

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        root.setLevel(getattr(logging, level, logging.INFO))
        for h in list(root.handlers):
            root.removeHandler(h)

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            root.addHandler(ch)

        _INITIALIZED = True
        return root


def get_logger(name: str = "research_assistant") -> logging.Logger:
    """获取模块级 logger；未初始化则先初始化。"""
    if not _INITIALIZED:
        try:
            setup_logging()
        except Exception:  # pragma: no cover - 日志失败绝不能阻断业务
            logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


class RequestLogger:
    """
    结构化请求日志（写入 SQLite request_logs 表）。

    设计要点：
      - 数据库实例由系统层注入（db 参数），避免模块间循环依赖
      - 未注入时惰性获取全局 Database 单例
      - 写库失败只告警不抛异常，绝不影响问答主流程
    """

    def __init__(self, db=None, enabled: bool = True, config: Optional[Config] = None):
        self.config = config or Config.instance()
        self.enabled = enabled and bool(self.config.get("observability.log_requests", True))
        self._db = db
        self._logger = get_logger("research_assistant.request")

    # ---------------- 数据库（惰性，避免循环导入） ----------------
    @property
    def db(self):
        if self._db is None:
            try:
                from ..storage.database import get_database

                self._db = get_database()
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("请求日志数据库不可用，将跳过记录: %s", exc)
                self.enabled = False
        return self._db

    def set_db(self, db) -> None:
        self._db = db

    # ---------------- 写入 ----------------
    def log(self, record: Dict[str, Any]) -> None:
        """通用写入接口。"""
        if not self.enabled:
            return
        try:
            self.db.log_request(
                type_=record.get("type", "rag"),
                query=record.get("query", ""),
                answer=record.get("answer", ""),
                hits=record.get("hits", []),
                steps=record.get("steps", []),
                citations=record.get("citations", []),
                latency_ms=record.get("latency_ms", 0.0),
                tokens=record.get("tokens", 0),
                route=record.get("route", ""),
                degraded=bool(record.get("degraded", False)),
                degrade_reason=record.get("degrade_reason", ""),
                from_cache=bool(record.get("from_cache", False)),
                top1_score=record.get("top1_score", 0.0),
                session_id=record.get("session_id", ""),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("写入请求日志失败: %s", exc)

    def log_rag(self, query: str, answer: str, hits: List[Dict[str, Any]],
                latency: float, tokens: int, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录一次 RAG 问答（latency 单位为秒）。"""
        extra = extra or {}
        self.log({
            "type": "rag", "query": query, "answer": answer, "hits": hits,
            "citations": extra.get("citations", []),
            "latency_ms": round(latency * 1000, 2), "tokens": tokens,
            "route": extra.get("route", "rag"),
            "degraded": extra.get("degraded", False),
            "degrade_reason": extra.get("degrade_reason", ""),
            "from_cache": extra.get("from_cache", False),
            "top1_score": extra.get("top1_score", 0.0),
            "session_id": extra.get("session_id", ""),
        })

    def log_agent(self, query: str, answer: str, steps: List[Dict[str, Any]],
                  latency: float, tokens: int, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录一次 Agent 推理（含完整 Thought/Action/Observation 轨迹）。"""
        extra = extra or {}
        self.log({
            "type": "agent", "query": query, "answer": answer, "steps": steps,
            "citations": extra.get("citations", []),
            "latency_ms": round(latency * 1000, 2), "tokens": tokens,
            "route": extra.get("route", "agent"),
            "degraded": extra.get("degraded", False),
            "degrade_reason": extra.get("degrade_reason", ""),
            "top1_score": extra.get("top1_score", 0.0),
            "session_id": extra.get("session_id", ""),
        })

    # ---------------- 查询（评测 / Bad Case 分析） ----------------
    def read_all(self, type_: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            return self.db.query_logs(type_=type_, limit=limit)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("读取请求日志失败: %s", exc)
            return []

    def bad_cases(self, limit: int = 100) -> List[Dict[str, Any]]:
        """筛出降级/失败的请求，用于模块五 Bad Case 分析。"""
        return self.read_all(limit=limit) and [
            r for r in self.read_all(limit=limit)
            if r.get("degraded") or not r.get("answer")
        ]

    def stats(self) -> Dict[str, Any]:
        try:
            return self.db.log_stats() if self.enabled else {}
        except Exception:  # noqa: BLE001
            return {}


class Timer:
    """上下文管理器计时器：with Timer() as t: ... ; t.elapsed（秒）"""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.elapsed = time.perf_counter() - self._start
        return False

    @property
    def ms(self) -> float:
        return round(self.elapsed * 1000, 2)


class MetricsCollector:
    """进程内指标采集器（模块四可观测性）。"""

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def incr(self, key: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[key] += value

    def observe(self, key: str, value: float) -> None:
        with self._lock:
            self._samples[key].append(value)
            self._counters[key + ".count"] += 1
            self._counters[key + ".sum"] += value

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = dict(self._counters)
            for key, values in self._samples.items():
                if values:
                    out[key + ".avg"] = round(sum(values) / len(values), 4)
                    out[key + ".max"] = round(max(values), 4)
            total = out.get("retrieval.total", 0)
            out["retrieval.hit_rate"] = round(out.get("retrieval.hit", 0) / total, 4) if total else 0.0
            tool_total = out.get("tool.success", 0) + out.get("tool.fail", 0)
            out["tool.success_rate"] = round(out.get("tool.success", 0) / tool_total, 4) if tool_total else 0.0
        return out

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._samples.clear()


metrics = MetricsCollector()
