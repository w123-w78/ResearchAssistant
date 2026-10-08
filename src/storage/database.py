# -*- coding: utf-8 -*-
"""
SQLite 持久化层
---------------
系统所有"需要落盘、需要查询、需要统计"的数据统一存入 SQLite：

  表名            用途                                   对应模块
  ------------------------------------------------------------------
  documents       已上传论文的元数据与索引状态             模块一 / 知识库管理
  chunks          文本块（引用溯源、摘要提取的数据源）      模块一
  sessions        会话（标题、摘要、时间戳）                模块三 记忆管理
  messages        对话消息（多轮上下文、滑动窗口）          模块三 记忆管理
  cache           语义缓存（问题向量 -> 答案）              模块二
  request_logs    请求日志（Token/耗时/命中率/Agent轨迹）    模块二 + 模块五评测

为什么选 SQLite：
  - 零部署：Python 标准库自带，无需额外服务，符合"轻量部署"定位
  - 可查询：评测阶段可直接写 SQL 统计 Bad Case（比翻 JSONL 高效得多）
  - 易迁移：表结构清晰，后续换 MySQL/PostgreSQL 只需替换本模块

线程安全：sqlite3 连接默认不允许跨线程共享，这里用
  check_same_thread=False + 全局 RLock 保证并发安全。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger("research_assistant.db")

# ==================================================================
# 建表语句
# ==================================================================
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

-- 文档元数据
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT,
    file_type   TEXT,
    num_pages   INTEGER DEFAULT 0,
    char_count  INTEGER DEFAULT 0,
    num_chunks  INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'indexed',      -- pending / indexing / indexed / failed
    error       TEXT,
    meta        TEXT,                         -- JSON
    created_at  REAL,
    updated_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- 文本块
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER,
    page        INTEGER DEFAULT 1,
    start_char  INTEGER DEFAULT 0,
    end_char    INTEGER DEFAULT 0,
    char_count  INTEGER DEFAULT 0,
    text        TEXT,
    created_at  REAL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- 会话
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    title       TEXT DEFAULT '新对话',
    summary     TEXT DEFAULT '',
    meta        TEXT,
    created_at  REAL,
    updated_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);

-- 对话消息
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,                -- user / assistant / system / tool
    content     TEXT,
    tokens      INTEGER DEFAULT 0,
    meta        TEXT,
    created_at  REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

-- 语义缓存
CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    answer      TEXT,
    citations   TEXT,                          -- JSON
    vector      BLOB,                          -- float32 向量
    kb_version  TEXT,
    hit_count   INTEGER DEFAULT 0,
    created_at  REAL,
    updated_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_cache_kb ON cache(kb_version);

-- 请求日志（评测与 Bad Case 分析的数据源）
CREATE TABLE IF NOT EXISTS request_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT,                          -- rag / agent
    query       TEXT,
    answer      TEXT,
    hits        TEXT,                          -- JSON: 命中片段
    steps       TEXT,                          -- JSON: Agent 推理轨迹
    citations   TEXT,                          -- JSON
    latency_ms  REAL,
    tokens      INTEGER,
    route       TEXT,
    degraded    INTEGER DEFAULT 0,
    degrade_reason TEXT,
    from_cache  INTEGER DEFAULT 0,
    top1_score  REAL,
    session_id  TEXT,
    created_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_logs_type ON request_logs(type);
CREATE INDEX IF NOT EXISTS idx_logs_created ON request_logs(created_at);
"""


class Database:
    """
    SQLite 数据库管理（系统唯一的数据落盘出口）。

    用法：
        db = Database("./data/assistant.db")
        db.upsert_document(doc, num_chunks=42)
        rows = db.list_documents()
    """

    _instance: Optional["Database"] = None
    _lock = threading.RLock()

    def __init__(self, db_path: Optional[str] = None, config: Optional[Config] = None):
        cfg = config or Config.instance()
        self.db_path = db_path or cfg.get("storage.db_path", "./data/assistant.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        logger.info("SQLite 已连接: %s", Path(self.db_path).resolve())

    # ---------------- 单例 ----------------
    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "Database":
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.commit()

    # ==================================================================
    # 通用执行
    # ==================================================================
    def execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executemany(self, sql: str, seq: Iterable) -> None:
        with self._lock:
            self.conn.executemany(sql, seq)
            self.conn.commit()

    def query(self, sql: str, params: Iterable = ()) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Iterable = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ==================================================================
    # 1. documents
    # ==================================================================
    def upsert_document(self, doc, num_chunks: int = 0, status: str = "indexed",
                        error: str = "") -> None:
        """插入或更新文档记录。"""
        now = time.time()
        self.execute(
            """INSERT INTO documents
               (doc_id, title, source, file_type, num_pages, char_count,
                num_chunks, status, error, meta, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 title=excluded.title, num_pages=excluded.num_pages,
                 char_count=excluded.char_count, num_chunks=excluded.num_chunks,
                 status=excluded.status, error=excluded.error,
                 meta=excluded.meta, updated_at=excluded.updated_at""",
            (doc.doc_id, doc.title, doc.source, doc.file_type, doc.num_pages,
             doc.char_count, num_chunks, status, error,
             json.dumps(doc.metadata, ensure_ascii=False), now, now),
        )

    def delete_document(self, doc_id: str) -> int:
        """删除文档（级联删除其 chunks）。"""
        cur = self.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.rowcount or 0

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return dict(row) if row else None

    def list_documents(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self.query("SELECT * FROM documents WHERE status=? ORDER BY created_at DESC",
                              (status,))
        else:
            rows = self.query("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def count_documents(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS c FROM documents")
        return int(row["c"]) if row else 0

    def update_doc_status(self, doc_id: str, status: str, error: str = "") -> None:
        self.execute("UPDATE documents SET status=?, error=?, updated_at=? WHERE doc_id=?",
                     (status, error, time.time(), doc_id))

    # ==================================================================
    # 2. chunks
    # ==================================================================
    def insert_chunks(self, chunks) -> int:
        """批量写入文本块（已存在则替换）。"""
        if not chunks:
            return 0
        now = time.time()
        rows = [(c.chunk_id, c.doc_id, c.chunk_index, c.page, c.start_char,
                 c.end_char, c.char_count, c.text, now) for c in chunks]
        self.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, chunk_index, page, start_char, end_char,
                char_count, text, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.execute("UPDATE documents SET num_chunks=?, updated_at=? WHERE doc_id=?",
                     (self.count_chunks(chunks[0].doc_id), now, chunks[0].doc_id))
        return len(rows)

    def delete_chunks_by_doc(self, doc_id: str) -> int:
        cur = self.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return cur.rowcount or 0

    def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        rows = self.query("SELECT * FROM chunks WHERE doc_id=? ORDER BY page, chunk_index",
                          (doc_id,))
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        return dict(row) if row else None

    def count_chunks(self, doc_id: Optional[str] = None) -> int:
        if doc_id:
            row = self.query_one("SELECT COUNT(*) AS c FROM chunks WHERE doc_id=?", (doc_id,))
        else:
            row = self.query_one("SELECT COUNT(*) AS c FROM chunks")
        return int(row["c"]) if row else 0

    def search_chunks(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        """全文关键词检索（SQL LIKE，辅助调试与知识库浏览）。"""
        rows = self.query(
            "SELECT c.*, d.title FROM chunks c JOIN documents d ON c.doc_id=d.doc_id "
            "WHERE c.text LIKE ? LIMIT ?", (f"%{keyword}%", limit))
        return [dict(r) for r in rows]

    # ==================================================================
    # 3. sessions / messages
    # ==================================================================
    def upsert_session(self, session_id: str, title: str = "", summary: str = "",
                       meta: Optional[Dict] = None) -> None:
        now = time.time()
        self.execute(
            """INSERT INTO sessions (session_id, title, summary, meta, created_at, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                 title=excluded.title, summary=excluded.summary,
                 meta=excluded.meta, updated_at=excluded.updated_at""",
            (session_id, title or "新对话", summary,
             json.dumps(meta or {}, ensure_ascii=False), now, now),
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM sessions WHERE session_id=?", (session_id,))
        return dict(row) if row else None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.query("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["num_messages"] = self.count_messages(d["session_id"])
            out.append(d)
        return out

    def delete_session(self, session_id: str) -> int:
        cur = self.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        return cur.rowcount or 0

    def update_session_summary(self, session_id: str, summary: str) -> None:
        self.execute("UPDATE sessions SET summary=?, updated_at=? WHERE session_id=?",
                     (summary, time.time(), session_id))

    # ---- messages ----
    def append_message(self, session_id: str, role: str, content: str,
                       tokens: int = 0, meta: Optional[Dict] = None) -> int:
        cur = self.execute(
            "INSERT INTO messages (session_id, role, content, tokens, meta, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, role, content, tokens,
             json.dumps(meta or {}, ensure_ascii=False), time.time()),
        )
        self.execute("UPDATE sessions SET updated_at=? WHERE session_id=?",
                     (time.time(), session_id))
        return int(cur.lastrowid or 0)

    def get_messages(self, session_id: str, limit: int = 100,
                     offset: int = 0) -> List[Dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id ASC LIMIT ? OFFSET ?",
            (session_id, limit, offset))
        return [dict(r) for r in rows]

    def get_recent_messages(self, session_id: str, n: int = 6) -> List[Dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM (SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?) "
            "ORDER BY id ASC", (session_id, n))
        return [dict(r) for r in rows]

    def count_messages(self, session_id: str) -> int:
        row = self.query_one("SELECT COUNT(*) AS c FROM messages WHERE session_id=?",
                             (session_id,))
        return int(row["c"]) if row else 0

    def delete_messages_before(self, session_id: str, before_id: int) -> int:
        """滑动窗口裁剪：删除早于某条 id 的历史消息。"""
        cur = self.execute("DELETE FROM messages WHERE session_id=? AND id < ?",
                           (session_id, before_id))
        return cur.rowcount or 0

    def clear_messages(self, session_id: str) -> int:
        cur = self.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        return cur.rowcount or 0

    # ==================================================================
    # 4. cache（语义缓存）
    # ==================================================================
    def cache_put(self, key: str, query: str, answer: str, citations: Any = None,
                  vector: Optional[bytes] = None, kb_version: str = "") -> None:
        # citations 允许传 list/dict，统一序列化为 JSON 字符串
        if citations is not None and not isinstance(citations, str):
            citations = json.dumps(citations, ensure_ascii=False)
        now = time.time()
        self.execute(
            """INSERT INTO cache (key, query, answer, citations, vector, kb_version,
                                  hit_count, created_at, updated_at)
               VALUES (?,?,?,?,?,?,0,?,?)
               ON CONFLICT(key) DO UPDATE SET
                 answer=excluded.answer, citations=excluded.citations,
                 vector=excluded.vector, kb_version=excluded.kb_version,
                 updated_at=excluded.updated_at""",
            (key, query, answer, citations, vector, kb_version, now, now),
        )

    def cache_all(self, kb_version: Optional[str] = None) -> List[Dict[str, Any]]:
        if kb_version:
            rows = self.query("SELECT * FROM cache WHERE kb_version=?", (kb_version,))
        else:
            rows = self.query("SELECT * FROM cache")
        return [dict(r) for r in rows]

    def cache_hit(self, key: str) -> None:
        self.execute("UPDATE cache SET hit_count=hit_count+1, updated_at=? WHERE key=?",
                     (time.time(), key))

    def cache_delete(self, key: str) -> None:
        self.execute("DELETE FROM cache WHERE key=?", (key,))

    def cache_clear(self, kb_version: Optional[str] = None) -> int:
        if kb_version:
            cur = self.execute("DELETE FROM cache WHERE kb_version=?", (kb_version,))
        else:
            cur = self.execute("DELETE FROM cache")
        return cur.rowcount or 0

    def cache_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS c FROM cache")
        return int(row["c"]) if row else 0

    def cache_prune(self, max_size: int = 500, ttl_seconds: int = 86400) -> int:
        """清理过期与超量缓存（LRU）：返回删除条数。"""
        removed = 0
        cur = self.execute("DELETE FROM cache WHERE ? - created_at > ?",
                           (time.time(), ttl_seconds))
        removed += cur.rowcount or 0
        row = self.query_one("SELECT COUNT(*) AS c FROM cache")
        total = int(row["c"]) if row else 0
        if total > max_size:
            cur = self.execute(
                "DELETE FROM cache WHERE key IN "
                "(SELECT key FROM cache ORDER BY updated_at ASC LIMIT ?)",
                (total - max_size,))
            removed += cur.rowcount or 0
        return removed

    # ==================================================================
    # 5. request_logs（评测数据源）
    # ==================================================================
    def log_request(self, type_: str, query: str, answer: str = "",
                    hits: Optional[List] = None, steps: Optional[List] = None,
                    citations: Optional[List] = None, latency_ms: float = 0.0,
                    tokens: int = 0, route: str = "", degraded: bool = False,
                    degrade_reason: str = "", from_cache: bool = False,
                    top1_score: float = 0.0, session_id: str = "") -> int:
        cur = self.execute(
            """INSERT INTO request_logs
               (type, query, answer, hits, steps, citations, latency_ms, tokens,
                route, degraded, degrade_reason, from_cache, top1_score, session_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (type_, query, answer,
             json.dumps(hits or [], ensure_ascii=False),
             json.dumps(steps or [], ensure_ascii=False),
             json.dumps(citations or [], ensure_ascii=False),
             latency_ms, tokens, route, int(degraded), degrade_reason,
             int(from_cache), top1_score, session_id, time.time()),
        )
        return int(cur.lastrowid or 0)

    def query_logs(self, type_: Optional[str] = None, limit: int = 100,
                   degraded_only: bool = False, since: Optional[float] = None
                   ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM request_logs WHERE 1=1"
        params: List[Any] = []
        if type_:
            sql += " AND type=?"
            params.append(type_)
        if degraded_only:
            sql += " AND degraded=1"
        if since:
            sql += " AND created_at>=?"
            params.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.query(sql, params)]

    def log_stats(self) -> Dict[str, Any]:
        """评测汇总：直接 SQL 聚合，省去手工处理 JSONL。"""
        row = self.query_one(
            """SELECT COUNT(*) AS total,
                      SUM(tokens) AS total_tokens,
                      AVG(latency_ms) AS avg_latency,
                      AVG(top1_score) AS avg_top1,
                      SUM(degraded) AS degraded_count,
                      SUM(from_cache) AS cache_count
               FROM request_logs""")
        d = dict(row) if row else {}
        by_type = {r["type"]: r["c"] for r in self.query(
            "SELECT type, COUNT(*) AS c FROM request_logs GROUP BY type")}
        by_route = {r["route"]: r["c"] for r in self.query(
            "SELECT route, COUNT(*) AS c FROM request_logs WHERE route!='' GROUP BY route")}
        return {
            "total_requests": d.get("total", 0) or 0,
            "total_tokens": d.get("total_tokens", 0) or 0,
            "avg_latency_ms": round(d.get("avg_latency") or 0, 2),
            "avg_top1_score": round(d.get("avg_top1") or 0, 4),
            "degraded_count": d.get("degraded_count", 0) or 0,
            "cache_hit_count": d.get("cache_count", 0) or 0,
            "by_type": by_type,
            "by_route": by_route,
        }

    def clear_logs(self) -> int:
        cur = self.execute("DELETE FROM request_logs")
        return cur.rowcount or 0

    # ==================================================================
    # 运维
    # ==================================================================
    def stats(self) -> Dict[str, Any]:
        """库整体统计（前端"系统状态"面板）。"""
        return {
            "db_path": str(Path(self.db_path).resolve()),
            "db_size_kb": round(Path(self.db_path).stat().st_size / 1024, 1)
            if Path(self.db_path).exists() else 0,
            "documents": self.count_documents(),
            "chunks": self.count_chunks(),
            "sessions": self.query_one("SELECT COUNT(*) AS c FROM sessions")["c"],
            "messages": self.query_one("SELECT COUNT(*) AS c FROM messages")["c"],
            "cache_entries": self.cache_count(),
            "logs": self.query_one("SELECT COUNT(*) AS c FROM request_logs")["c"],
        }

    def reset(self) -> None:
        """清空全部业务数据（保留表结构）。"""
        with self._lock:
            for t in ("request_logs", "cache", "messages", "sessions", "chunks", "documents"):
                self.conn.execute(f"DELETE FROM {t}")
            self.conn.commit()
        logger.warning("数据库已清空全部业务数据")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Database({self.db_path})"


def get_database(db_path: Optional[str] = None) -> Database:
    """获取全局数据库实例。"""
    return Database.get_instance(db_path)
