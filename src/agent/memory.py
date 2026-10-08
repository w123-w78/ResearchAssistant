# -*- coding: utf-8 -*-
"""
模块三 5.3.4：多轮对话记忆管理

三大机制：
  1. 会话隔离：每个 session_id 一份独立的 Memory，多用户并发互不干扰
  2. 滑动窗口：按 Token 预算自动丢弃最早的消息（保留 system + 最近 N 轮）
  3. 摘要压缩：对话超过阈值轮次时，把早期历史压缩为一段摘要，
               用摘要替换原始消息，在省 Token 的同时保住上下文连贯性

为什么必须做：
  模型上下文窗口有限（7B 模型通常 4k-32k），长对话会直接报错或被截断。
  而科研问答天然是多轮的（"这篇论文的方法是什么" -> "和另一篇比呢"）。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.llm_client import BaseLLM
from ..utils.text_utils import estimate_tokens
from ..utils.logger import get_logger

try:
    from ..generation.prompt_template import build_summary_messages
except Exception:  # pragma: no cover
    build_summary_messages = None  # type: ignore

logger = get_logger("research_assistant.memory")


@dataclass
class Message:
    """一条对话消息。"""
    role: str                 # system / user / assistant / tool
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        if "tokens" not in self.metadata:
            self.metadata["tokens"] = estimate_tokens(self.content)
        return int(self.metadata["tokens"])

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content,
                "timestamp": self.timestamp, "metadata": self.metadata}

    def to_api(self) -> Dict[str, str]:
        """转成 LLM API 需要的格式。"""
        return {"role": self.role, "content": self.content}


class SessionMemory:
    """单个会话的记忆。"""

    def __init__(self, session_id: str, title: str = "", max_tokens: int = 3000,
                 max_turns: int = 20, summarize_threshold: int = 8):
        self.session_id = session_id
        self.title = title or "新对话"
        self.messages: List[Message] = []
        self.summary: str = ""
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.summarize_threshold = summarize_threshold
        self.created_at = time.time()
        self.updated_at = time.time()
        self.metadata: Dict[str, Any] = {}

    # ---------------- 增删 ----------------
    def add(self, role: str, content: str, **metadata) -> Message:
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.updated_at = time.time()
        # 首条用户消息自动作为会话标题
        if role == "user" and self.title in ("新对话", "") and content:
            self.title = content[:20].strip()
        return msg

    def add_user(self, content: str, **meta) -> Message:
        return self.add("user", content, **meta)

    def add_assistant(self, content: str, **meta) -> Message:
        return self.add("assistant", content, **meta)

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    # ---------------- 查询 ----------------
    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages) + estimate_tokens(self.summary)

    @property
    def turns(self) -> int:
        """用户提问轮次。"""
        return sum(1 for m in self.messages if m.role == "user")

    def get_recent(self, n: int = 6) -> List[Dict[str, str]]:
        """取最近 n 条消息的 API 格式（不含 system）。"""
        msgs = [m for m in self.messages if m.role != "system"]
        return [m.to_api() for m in msgs[-n:]]

    def get_history_text(self, n: int = 10) -> str:
        """把历史拼成纯文本（用于摘要或调试展示）。"""
        msgs = self.messages[-n:]
        return "\n".join(f"{m.role}: {m.content}" for m in msgs)

    # ---------------- 滑动窗口 ----------------
    def apply_sliding_window(self, max_tokens: Optional[int] = None,
                             max_turns: Optional[int] = None) -> int:
        """
        Token 级滑动窗口：从最早的消息开始丢弃，直到满足预算。
        永远保留最后一条用户消息（否则当前问题会丢失）。

        :return: 被丢弃的消息数
        """
        budget = max_tokens or self.max_tokens
        max_turn = max_turns or self.max_turns
        dropped = 0

        # 按轮次裁剪
        while self.turns > max_turn:
            self._drop_earliest_turn()
            dropped += 1

        # 按 Token 裁剪
        while self.total_tokens > budget and len(self.messages) > 2:
            # 保护最后一条消息
            if len(self.messages) <= 2:
                break
            self.messages.pop(0)
            dropped += 1
        if dropped:
            logger.debug("会话 %s 滑动窗口丢弃 %s 条消息", self.session_id[:8], dropped)
        return dropped

    def _drop_earliest_turn(self) -> None:
        """丢弃最早的一轮（user + 其后的 assistant/tool）。"""
        for i, m in enumerate(self.messages):
            if m.role == "user":
                end = i + 1
                while end < len(self.messages) and self.messages[end].role in ("assistant", "tool"):
                    end += 1
                del self.messages[i:end]
                return
        if self.messages:
            self.messages.pop(0)

    # ---------------- 序列化 ----------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id, "title": self.title,
            "summary": self.summary, "created_at": self.created_at,
            "updated_at": self.updated_at, "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs) -> "SessionMemory":
        obj = cls(session_id=d["session_id"], title=d.get("title", ""), **kwargs)
        obj.summary = d.get("summary", "")
        obj.created_at = d.get("created_at", time.time())
        obj.updated_at = d.get("updated_at", time.time())
        obj.metadata = d.get("metadata", {})
        for m in d.get("messages", []):
            obj.messages.append(Message(role=m["role"], content=m["content"],
                                        timestamp=m.get("timestamp", time.time()),
                                        metadata=m.get("metadata", {})))
        return obj

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SessionMemory({self.session_id[:8]}, {len(self.messages)}条, {self.total_tokens}tokens)"


class MemoryManager:
    """
    多会话记忆管理器（会话隔离 + 滑动窗口 + 摘要压缩 + SQLite 持久化）。

    存储分层：
      SQLite(messages 表) = 完整历史，永不裁剪，供 UI 展示与复盘
      SessionMemory(内存)  = 工作集，按窗口裁剪后用于拼 Prompt
    """

    def __init__(self, llm: Optional[BaseLLM] = None, config=None, db=None):
        from ..utils.config import Config

        self.config = config or Config.instance()
        m_cfg = self.config.section("memory")
        self.max_tokens = int(m_cfg.get("max_tokens", 3000))
        self.max_turns = int(m_cfg.get("max_turns", 20))
        self.summarize_threshold = int(m_cfg.get("summarize_threshold", 8))
        self.llm = llm
        self.db = db

        self._sessions: Dict[str, SessionMemory] = {}
        self._lock = threading.RLock()
        if self.db is not None:
            self._load_index_from_db()

    # ---------------- 数据库惰性获取（避免循环导入） ----------------
    def _ensure_db(self):
        if self.db is None:
            try:
                from ..storage.database import get_database

                self.db = get_database()
            except Exception:  # noqa: BLE001
                self.db = False
        return self.db or None

    def set_db(self, db) -> None:
        self.db = db

    # ---------------- 会话管理 ----------------
    def create_session(self, session_id: Optional[str] = None, title: str = "") -> SessionMemory:
        sid = session_id or uuid.uuid4().hex[:16]
        with self._lock:
            mem = SessionMemory(sid, title=title, max_tokens=self.max_tokens,
                                max_turns=self.max_turns,
                                summarize_threshold=self.summarize_threshold)
            self._sessions[sid] = mem
        db = self._ensure_db()
        if db is not None:
            try:
                db.upsert_session(sid, title=mem.title, summary="")
            except Exception as exc:  # noqa: BLE001
                logger.warning("会话持久化失败: %s", exc)
        logger.info("创建会话 %s", sid[:8])
        return mem

    def get_session(self, session_id: str, create: bool = True) -> SessionMemory:
        with self._lock:
            mem = self._sessions.get(session_id)
            if mem is None:
                mem = self._restore_session(session_id)
                if mem is not None:
                    self._sessions[session_id] = mem
            if mem is None and create:
                mem = self.create_session(session_id)
            return mem

    def _restore_session(self, session_id: str) -> Optional[SessionMemory]:
        """从 SQLite 恢复会话（含全部历史消息）。"""
        db = self._ensure_db()
        if db is None:
            return None
        try:
            row = db.get_session(session_id)
            if not row:
                return None
            mem = SessionMemory(session_id, title=row.get("title", ""),
                                max_tokens=self.max_tokens, max_turns=self.max_turns,
                                summarize_threshold=self.summarize_threshold)
            mem.summary = row.get("summary", "") or ""
            mem.created_at = row.get("created_at", time.time())
            mem.updated_at = row.get("updated_at", time.time())
            for m in db.get_messages(session_id, limit=300):
                meta = m.get("meta")
                mem.messages.append(Message(
                    role=m["role"], content=m["content"],
                    timestamp=m.get("created_at", time.time()),
                    metadata=meta if isinstance(meta, dict) else {},
                ))
            logger.debug("从数据库恢复会话 %s（%s 条消息）", session_id[:8], len(mem.messages))
            return mem
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话恢复失败: %s", exc)
            return None

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        db = self._ensure_db()
        if db is not None:
            db.delete_session(session_id)
        logger.info("删除会话 %s", session_id[:8])
        return existed

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """会话列表（优先读库，含消息条数统计）。"""
        db = self._ensure_db()
        if db is not None:
            try:
                return db.list_sessions(limit=limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning("会话列表读取失败: %s", exc)
        with self._lock:
            items = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
        return [{"session_id": s.session_id, "title": s.title, "turns": s.turns,
                 "num_messages": len(s.messages), "tokens": s.total_tokens,
                 "updated_at": s.updated_at} for s in items]

    def get_messages(self, session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """完整历史（UI 展示用，读库全量，不受滑动窗口影响）。"""
        db = self._ensure_db()
        if db is not None:
            return db.get_messages(session_id, limit=limit)
        mem = self._sessions.get(session_id)
        return [m.to_dict() for m in (mem.messages if mem else [])]

    # ---------------- 记忆处理主入口 ----------------
    def add_message(self, session_id: str, role: str, content: str, **meta) -> Message:
        """写入消息（内存 + SQLite），并自动执行滑动窗口与摘要压缩。"""
        mem = self.get_session(session_id)
        msg = mem.add(role, content, **meta)
        db = self._ensure_db()
        if db is not None:
            try:
                db.append_message(session_id, role, content, tokens=msg.tokens, meta=meta)
            except Exception as exc:  # noqa: BLE001
                logger.warning("消息持久化失败: %s", exc)
        mem.apply_sliding_window()
        if mem.turns > self.summarize_threshold and self.llm is not None:
            self.maybe_summarize(session_id)
        return msg

    def get_context(self, session_id: str, max_messages: int = 8) -> List[Dict[str, str]]:
        """
        取出送给 LLM 的上下文：
          摘要（若有） + 最近若干条消息
        """
        mem = self.get_session(session_id, create=False)
        if mem is None:
            return []
        ctx: List[Dict[str, str]] = []
        if mem.summary:
            ctx.append({"role": "system", "content": f"【此前的对话摘要】\n{mem.summary}"})
        ctx.extend(mem.get_recent(max_messages))
        return ctx

    # ---------------- 摘要压缩 ----------------
    def maybe_summarize(self, session_id: str, force: bool = False) -> Optional[str]:
        """
        当对话轮次超过阈值时，压缩早期历史。

        策略：保留最近 4 条消息，把更早的部分交给 LLM 生成摘要，
              摘要存入 mem.summary，原消息删除。
        """
        mem = self.get_session(session_id, create=False)
        if mem is None or self.llm is None:
            return None
        if not force and mem.turns <= self.summarize_threshold:
            return None

        keep = 4
        if len(mem.messages) <= keep + 1:
            return None
        old_msgs = mem.messages[:-keep]
        history_text = "\n".join(f"{m.role}: {m.content}" for m in old_msgs)
        if mem.summary:
            history_text = f"[已有摘要]\n{mem.summary}\n\n[新增对话]\n{history_text}"

        try:
            if build_summary_messages is not None:
                messages = build_summary_messages(history_text)
            else:  # pragma: no cover
                messages = [{"role": "user", "content": f"请压缩以下对话历史：\n{history_text}"}]
            summary = self.llm.chat(messages, temperature=0.1).strip()
        except Exception as exc:  # noqa: BLE001 - 摘要失败不能影响对话
            logger.warning("对话摘要生成失败，跳过压缩: %s", exc)
            return None

        mem.summary = summary
        mem.messages = mem.messages[-keep:]
        db = self._ensure_db()
        if db is not None:
            try:
                db.update_session_summary(session_id, summary)
            except Exception as exc:  # noqa: BLE001
                logger.warning("摘要持久化失败: %s", exc)
        logger.info("会话 %s 已压缩：%s 条 -> 摘要(%s字) + %s 条（数据库保留完整历史）",
                    session_id[:8], len(old_msgs), len(summary), keep)
        return summary

    # ---------------- 持久化（SQLite） ----------------
    def _load_index_from_db(self) -> int:
        """启动时把库中的会话元信息载入内存工作集（消息按需懒加载）。"""
        try:
            rows = self.db.list_sessions(limit=200)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话列表加载失败: %s", exc)
            return 0
        for r in rows:
            mem = SessionMemory(r["session_id"], title=r.get("title", ""),
                                max_tokens=self.max_tokens, max_turns=self.max_turns,
                                summarize_threshold=self.summarize_threshold)
            mem.summary = r.get("summary", "") or ""
            mem.created_at = r.get("created_at", time.time())
            mem.updated_at = r.get("updated_at", time.time())
            self._sessions[mem.session_id] = mem
        if rows:
            logger.info("已从 SQLite 恢复 %s 个会话", len(rows))
        return len(rows)

    def save_all(self) -> None:
        """兼容旧接口：SQLite 模式下消息已实时落库，无需额外操作。"""
        logger.debug("SQLite 持久化模式，save_all 无需执行")

    def load_all(self) -> int:
        """兼容旧接口：返回库中会话数。"""
        db = self._ensure_db()
        if db is None:
            return len(self._sessions)
        return len(db.list_sessions(limit=10000))

    def stats(self) -> Dict[str, Any]:
        db = self._ensure_db()
        if db is not None:
            try:
                return {
                    "num_sessions": db.query_one("SELECT COUNT(*) AS c FROM sessions")["c"],
                    "total_messages": db.query_one("SELECT COUNT(*) AS c FROM messages")["c"],
                    "storage": "sqlite",
                }
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            return {
                "num_sessions": len(self._sessions),
                "total_messages": sum(len(s.messages) for s in self._sessions.values()),
                "total_tokens": sum(s.total_tokens for s in self._sessions.values()),
                "storage": "memory",
            }

    def __len__(self) -> int:
        return len(self._sessions)
