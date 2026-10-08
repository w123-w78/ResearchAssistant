# -*- coding: utf-8 -*-
"""
模块四 5.4.1：系统集成层（RAG + Agent 深度融合）

这一层是"把所有模块串起来"的胶水层，也是对外暴露的唯一门面：

    ResearchAssistantSystem
        ├── 模块一：DocumentImporter + Chunker + HybridRetriever
        ├── 模块二：RAGPipeline（生成 + 引用 + 缓存 + 降级）
        ├── 模块三：ReActAgent + ToolRegistry + MemoryManager
        └── 模块四：可观测性（metrics / request_log / health_check）

前端（Streamlit）、CLI、测试均只依赖这一个类，
保证"换前端不改核心"、"换 LLM 不改上层"。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from .agent import (MemoryManager, ReActAgent, ToolRegistry, build_default_tools)
from .chunking import Chunk, get_chunker
from .data_loader import Document, DocumentImporter, load_document
from .generation import RAGAnswer, RAGPipeline, SemanticCache
from .retrieval import HybridRetriever
from .storage import Database
from .utils import (build_embedder, build_llm, estimate_tokens, get_logger, metrics)

logger = get_logger("research_assistant.system")


class ResearchAssistantSystem:
    """
    智能科研助理系统（门面类）。

    典型用法：
        sys = ResearchAssistantSystem()
        sys.ingest_files(["paper1.pdf", "paper2.docx"])
        result = sys.ask("这篇论文用了什么数据集？", session_id="s1")
        print(result.answer)
    """

    def __init__(self, config=None, config_path: Optional[str] = None,
                 llm=None, embedder=None):
        from pathlib import Path

        from .utils.config import Config

        # 默认配置文件：项目根目录下的 config.yaml（与运行目录无关）
        # reload=True：确保门面层拿到的配置一定来自该文件。
        # 原因：日志模块在 import 阶段就会触发 Config.instance()，
        # 若那时 cwd 下没有 config.yaml，单例会用内置默认值创建，
        # 导致后续无论从哪里加载都返回错误的配置。
        default_cfg = Path(__file__).resolve().parent.parent / "config.yaml"
        self.config = config or Config.load(config_path or str(default_cfg), reload=True)
        setup_logging_if_needed(self.config)

        # ---- 基础组件 ----
        self.db = Database(db_path=self.config.get("storage.db_path"), config=self.config)
        self.llm = llm or build_llm(self.config)
        self.embedder = embedder or build_embedder(self.config)

        # ---- 模块一 ----
        self.chunker = get_chunker(
            self.config.get("chunking.strategy", "recursive"),
            chunk_size=int(self.config.get("chunking.chunk_size", 512)),
            chunk_overlap=int(self.config.get("chunking.chunk_overlap", 64)),
            min_chunk_size=int(self.config.get("chunking.min_chunk_size", 80)),
        )
        self.retriever = HybridRetriever(embedder=self.embedder, config=self.config)

        # ---- 模块二 ----
        self.rag = RAGPipeline(
            retriever=self.retriever, llm=self.llm,
            config=self.config, embedder=self.embedder, db=self.db,
        )

        # ---- 模块三 ----
        self.memory = MemoryManager(llm=self.llm, config=self.config, db=self.db)
        self.tools: ToolRegistry = build_default_tools(
            rag_pipeline=self.rag, llm=self.llm, config=self.config, embedder=self.embedder
        )
        self.agent = ReActAgent(
            llm=self.llm, tools=self.tools, memory=self.memory, config=self.config,
            has_documents_func=lambda: not self.retriever.is_empty(), db=self.db,
        )

        self.documents: Dict[str, Document] = {}
        self.chunks: List[Chunk] = []
        logger.info("系统初始化完成 | LLM=%s | Embedding=%s | 分块=%s",
                    self.llm.name, self.embedder.model_name, self.chunker.name)

        # 启动时自动加载已持久化的向量索引。
        # 不做这一步的话，每次重启进程索引都是空的：SQLite 里能看到文档列表，
        # 但检索不到任何内容，问答必然答不出（曾实际踩到）。
        self.load_index()

    # ==================================================================
    # 知识库：导入 / 删除
    # ==================================================================
    def ingest_files(self, paths: List[str],
                     on_progress: Optional[Callable[[int, int, str, str], None]] = None
                     ) -> Dict[str, Any]:
        """
        导入文档 -> 分块 -> 向量化建库（完整流水线）。

        :return: 导入汇总（成功/失败/跳过的数量与明细）
        """
        importer = DocumentImporter(max_retries=2)
        results = importer.import_files(paths, on_progress=on_progress)
        summary = DocumentImporter.summary(results)

        self._index_documents(summary["documents"])
        summary["total_chunks"] = self.retriever.size
        return summary

    def _index_documents(self, docs) -> int:
        """分块 -> 向量化入库 -> 元数据写入 SQLite（导入流程的公共部分）。"""
        if not docs:
            return 0
        for d in docs:
            self.documents[d.doc_id] = d
            try:
                self.db.upsert_document(d, num_chunks=0, status="indexing")
            except Exception as exc:  # noqa: BLE001
                logger.warning("文档元数据写入失败: %s", exc)
        new_chunks = self.chunker.split_documents(docs)
        self.chunks.extend(new_chunks)
        self.retriever.add(new_chunks)

        # 文本块与文档状态写入 SQLite
        for d in docs:
            n = self.retriever.get_doc_chunks(d.doc_id)
            try:
                self.db.upsert_document(d, num_chunks=len(n), status="indexed")
                if n:
                    self.db.insert_chunks(n)
            except Exception as exc:  # noqa: BLE001
                logger.warning("分块持久化失败(%s): %s", d.doc_id[:8], exc)
        # 知识库变化 -> 语义缓存失效
        self.rag.invalidate_cache()
        # 索引落盘：否则进程退出后向量全丢，重启只剩 SQLite 里的文档记录，
        # 表现为"列表有 10 篇、检索 0 块、问什么都答不出"
        try:
            self.save_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("索引持久化失败: %s", exc)
        logger.info("新增 %s 篇文档，%s 个文本块，知识库现有 %s 块",
                    len(docs), len(new_chunks), self.retriever.size)
        return len(new_chunks)

    def ingest_directory(self, dir_path: str, recursive: bool = True,
                         on_progress=None) -> Dict[str, Any]:
        """批量导入整个目录。"""
        importer = DocumentImporter(max_retries=2)
        results = importer.import_directory(dir_path, recursive=recursive, on_progress=on_progress)
        summary = DocumentImporter.summary(results)
        self._index_documents(summary["documents"])
        summary["total_chunks"] = self.retriever.size
        return summary

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """删除文档及其向量。"""
        removed = self.retriever.delete_by_doc(doc_id)
        self.documents.pop(doc_id, None)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self.db.delete_document(doc_id)          # 级联删除 chunks
        self.rag.invalidate_cache()
        return {"doc_id": doc_id, "removed_chunks": removed}

    def clear_knowledge_base(self) -> None:
        """清空知识库（向量索引 + SQLite 文档记录）。"""
        for row in self.db.list_documents():
            self.retriever.delete_by_doc(row["doc_id"])
            self.db.delete_document(row["doc_id"])
        self.documents.clear()
        self.chunks.clear()
        if self.rag.cache:
            self.rag.cache.clear()
        logger.info("知识库已清空")

    def list_documents(self) -> List[Dict[str, Any]]:
        """文档列表（SQLite 为准，附带向量库中的实际块数）。"""
        rows = self.db.list_documents()
        vec_stat = self.retriever.list_documents()
        vec_map = {d["doc_id"]: d["chunks"] for d in vec_stat}
        for r in rows:
            r["vectorized_chunks"] = vec_map.get(r["doc_id"], 0)
        return rows

    # ==================================================================
    # 问答：RAG 直答 / Agent 决策
    # ==================================================================
    def ask(self, query: str, session_id: str = "default",
            use_agent: bool = True, on_step=None,
            use_cache: bool = True) -> Dict[str, Any]:
        """
        统一问答入口。

        use_agent=True  -> 走 Agent（路由 + ReAct + 工具调用）
        use_agent=False -> 走纯 RAG（对比实验用）

        返回统一结构：{"answer", "citations", "steps", "meta": {...}}
        """
        start = time.perf_counter()
        if not use_agent:
            ans: RAGAnswer = self.rag.answer(query, use_cache=use_cache)
            return {
                "answer": ans.answer,
                "citations": [c.to_dict() for c in ans.citations],
                "steps": [],
                "meta": {**ans.to_dict(), "mode": "rag"},
            }

        result = self.agent.run(query, session_id=session_id, on_step=on_step)
        return {
            "answer": result.answer,
            "citations": self._citations_from_steps(result.steps),
            "steps": [s.to_dict() for s in result.steps],
            "meta": {**result.to_dict(), "mode": "agent",
                     "latency": round(time.perf_counter() - start, 3)},
        }

    def ask_stream(self, query: str, session_id: str = "default",
                   use_agent: bool = True) -> Iterator[Dict[str, Any]]:
        """
        流式问答（前端打字机 + 推理轨迹实时展示）。

        yield 事件：
          {"type": "step",  "step": {...}}    Agent 每一步推理
          {"type": "token", "content": "..."} 答案文本块
          {"type": "done",  "meta": {...}}    完成
        """
        if not use_agent:
            for ev in self.rag.stream_answer(query):
                if ev["type"] == "token":
                    yield {"type": "token", "content": ev["content"]}
                elif ev["type"] == "done":
                    ans = ev["answer"]
                    yield {"type": "done", "meta": {**ans.to_dict(), "mode": "rag"}}
            return

        def on_step(step):
            yield_ = None  # 回调无法直接 yield，用闭包收集
            steps_buffer.append(step)

        steps_buffer: List[Any] = []
        result = self.agent.run(query, session_id=session_id,
                                on_step=lambda s: steps_buffer.append(s))
        # 先推送推理轨迹
        for s in steps_buffer:
            yield {"type": "step", "step": s.to_dict()}
        # 再流式吐出答案
        from .generation.streaming import typing_iterator

        for piece in typing_iterator(result.answer, chunk_size=6,
                                     delay=float(self.config.get("frontend.typing_speed", 0.01))):
            yield {"type": "token", "content": piece}
        yield {"type": "done",
               "meta": {**result.to_dict(), "mode": "agent",
                        "citations": self._citations_from_steps(result.steps)}}

    @staticmethod
    def _citations_from_steps(steps: List[Any]) -> List[Dict[str, Any]]:
        """从 Agent 的 Observation 中提取引用（简化版：解析 [n] 来源行）。"""
        import re

        cites: List[Dict[str, Any]] = []
        for s in steps:
            if not getattr(s, "observation", ""):
                continue
            for m in re.finditer(r"\[(\d+)\]\s*([^\n（(]{2,80}?)（第(\d+)页",
                                 s.observation):
                idx = int(m.group(1))
                if any(c["index"] == idx for c in cites):
                    continue
                cites.append({"index": idx, "title": m.group(2).strip(),
                              "page": int(m.group(3)), "text": ""})
        return cites[:8]

    # ==================================================================
    # 可观测性（模块四 5.4.2）
    # ==================================================================
    def health_check(self) -> Dict[str, Any]:
        """健康检查：LLM 服务 / 向量库 / 工具集状态。"""
        checks: Dict[str, Any] = {}

        # LLM
        try:
            t0 = time.perf_counter()
            self.llm.chat([{"role": "user", "content": "ping"}], max_tokens=8)
            checks["llm"] = {"status": "ok", "provider": self.llm.name,
                             "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            checks["llm"] = {"status": "error", "provider": self.llm.name, "error": str(exc)}

        # 向量库
        checks["vector_store"] = {"status": "ok" if self.retriever.size >= 0 else "error",
                                  **self.retriever.stats()}
        # Embedding
        try:
            t0 = time.perf_counter()
            self.embedder.encode_query("健康检查")
            checks["embedding"] = {"status": "ok", "model": self.embedder.model_name,
                                   "dim": self.embedder.dim,
                                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            checks["embedding"] = {"status": "error", "error": str(exc)}

        checks["database"] = {"status": "ok", **self.db.stats()}
        checks["tools"] = {"count": len(self.tools), "names": self.tools.names}
        checks["memory"] = self.memory.stats()
        checks["overall"] = "ok" if all(
            v.get("status") in ("ok", None) for v in checks.values() if isinstance(v, dict)
        ) else "degraded"
        return checks

    def stats(self) -> Dict[str, Any]:
        """系统统计快照（前端监控面板）。"""
        s = self.rag.stats()
        s.update({
            "documents": self.db.count_documents(),
            "chunks": self.retriever.size,
            "sessions": self.memory.stats(),
            "metrics": metrics.snapshot(),
            "agent": self.agent.stats(),
            "database": self.db.stats(),
            "request_logs": self.db.log_stats(),
        })
        return s

    def save_index(self, path: Optional[str] = None) -> None:
        """持久化索引。"""
        p = path or self.config.get("vector_store.index_path", "./data/index/vector_store")
        self.retriever.save(p)
        self.memory.save_all()
        logger.info("索引已保存至 %s", p)

    def load_index(self, path: Optional[str] = None) -> bool:
        """加载已有索引。"""
        p = path or self.config.get("vector_store.index_path", "./data/index/vector_store")
        try:
            loaded = HybridRetriever.load(p, embedder=self.embedder, config=self.config)
            self.retriever = loaded
            self.rag.retriever = loaded
            # 重建工具（工具闭包持有旧 retriever 引用）
            self.tools = build_default_tools(rag_pipeline=self.rag, llm=self.llm,
                                             config=self.config, embedder=self.embedder)
            self.agent.tools = self.tools
            self.memory.load_all()
            logger.info("索引已加载：%s 块", self.retriever.size)
            return True
        except FileNotFoundError:
            logger.info("未发现已有索引，将从空库启动")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("索引加载失败: %s", exc)
            return False


def setup_logging_if_needed(config) -> None:
    """幂等初始化日志。"""
    try:
        from .utils.logger import setup_logging

        setup_logging(config)
    except Exception:  # noqa: BLE001
        pass
