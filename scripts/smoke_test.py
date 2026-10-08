# -*- coding: utf-8 -*-
"""
冒烟测试：不依赖 API Key，验证 SQLite 存储 / 分块 / 检索 / 路由 / ReAct 解析

用法： python scripts/smoke_test.py
说明：本脚本只测"不需要调用 DashScope 接口"的部分；
      端到端问答需要配置 DASHSCOPE_API_KEY 后运行 scripts/cli_demo.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import get_chunker                      # noqa: E402
from src.data_loader import Document, Page                # noqa: E402
from src.storage import Database                          # noqa: E402

OK, FAIL = "✓", "✗"
_results = {"pass": 0, "fail": 0}


def check(name: str, cond: bool, detail: str = "") -> None:
    _results["pass" if cond else "fail"] += 1
    print(f"  {OK if cond else FAIL} {name}" + (f"  ({detail})" if detail else ""))


def make_doc(title: str, text: str) -> Document:
    return Document(doc_id=f"doc_{abs(hash(title)) % 10**8}", title=title,
                    source=f"./{title}.md", file_type="md",
                    pages=[Page(page=1, text=text)], metadata={"file_name": title})


# ==================================================================
def test_database() -> Database:
    print("\n[1] SQLite 存储层")
    tmp = Path(tempfile.mkdtemp()) / "smoke.db"
    db = Database(str(tmp))
    check("建库成功", tmp.exists())

    doc = make_doc("测试论文", "本文提出一种基于扩散模型的分割方法。" * 20)
    db.upsert_document(doc, num_chunks=3, status="indexed")
    check("写入 documents", db.count_documents() == 1)
    check("读取 documents", db.get_document(doc.doc_id)["title"] == "测试论文")

    chunks = get_chunker("recursive", chunk_size=200, chunk_overlap=20).split_document(doc)
    db.insert_chunks(chunks)
    check("写入 chunks", db.count_chunks(doc.doc_id) == len(chunks), f"{len(chunks)} 块")
    check("全文检索 chunks", len(db.search_chunks("扩散模型")) > 0)

    db.upsert_session("s1", title="会话一")
    db.append_message("s1", "user", "这篇论文用了什么方法？", tokens=8)
    db.append_message("s1", "assistant", "用了扩散模型。", tokens=6)
    check("写入 messages", db.count_messages("s1") == 2)
    check("会话列表带条数", db.list_sessions()[0]["num_messages"] == 2)

    db.cache_put("k1", "问题A", "答案A", [], vector=np.zeros(4, np.float32).tobytes())
    check("写入 cache", db.cache_count() == 1)

    db.log_request("rag", "问题A", "答案A", hits=[{"chunk_id": "c1"}],
                   latency_ms=123.4, tokens=50, top1_score=0.87)
    db.log_request("agent", "问题B", "答案B", steps=[{"iteration": 1}],
                   latency_ms=200.0, tokens=80, route="agent", degraded=True)
    st = db.log_stats()
    check("写入 request_logs", st["total_requests"] == 2)
    check("日志聚合(按类型)", st["by_type"].get("rag") == 1)
    check("降级请求统计", st["degraded_count"] == 1)

    check("库统计可用", db.stats()["documents"] == 1)
    db.delete_document(doc.doc_id)
    check("级联删除 chunks", db.count_chunks(doc.doc_id) == 0)
    return db


# ==================================================================
def test_chunking() -> None:
    print("\n[2] 分块策略")
    doc = make_doc("长文档", "。".join(f"这是第{i}句测试内容，包含一些中文与 English 混排" for i in range(60)))
    for name in ("fixed", "recursive", "semantic", "academic"):
        chunks = get_chunker(name, chunk_size=300, chunk_overlap=30).split_document(doc)
        ok = len(chunks) > 0 and all(c.text.strip() for c in chunks)
        check(f"{name} 切分", ok, f"{len(chunks)} 块")


# ==================================================================
class StubEmbedder:
    """确定性伪向量（不调用 API），用于离线验证检索链路。"""

    def __init__(self, dim: int = 32):
        self._dim = dim
        self.model_name = "stub"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts, batch_size=None, show_progress=False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t:
                out[i, ord(ch) % self._dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def test_retrieval() -> None:
    print("\n[3] 检索层（BM25 + 向量 + RRF 融合）")
    from src.retrieval import BM25Retriever, HybridRetriever, Reranker
    from src.retrieval.evaluate import hit_at_k, mrr

    docs = [make_doc("ViT论文", "Vision Transformer 在 ImageNet 上达到 84.5% 的准确率。"),
            make_doc("扩散模型论文", "扩散模型用于医学图像分割，Dice 系数为 87.3%。")]
    chunks = get_chunker("recursive", chunk_size=120, chunk_overlap=10).split_documents(docs)

    retr = HybridRetriever(embedder=StubEmbedder(), reranker=Reranker(provider="none"))
    retr.build(chunks)
    check("建库", retr.size == len(chunks), f"{retr.size} 块")

    hits = retr.search("扩散模型的 Dice 系数是多少", with_rerank=False)
    check("混合检索有结果", len(hits) > 0)
    check("Top1 命中正确文档", hits and "扩散" in hits[0].chunk.doc_title,
          hits[0].chunk.doc_title if hits else "")
    check("Hit@1 计算", hit_at_k(hits, [docs[1].doc_id], k=1) == 1.0)
    check("MRR 计算", mrr(hits, [docs[1].doc_id]) == 1.0)

    bm = BM25Retriever()
    bm.build(chunks)
    check("BM25 检索", len(bm.search("ImageNet 准确率", top_k=3)) > 0)
    check("两路融合字段完整", hits[0].rrf_score > 0 and hits[0].source in ("both", "vector", "bm25"))

    rr = Reranker(provider="heuristic")
    top = rr.rerank("扩散模型 Dice", retr.search("扩散模型 Dice", with_rerank=False), top_n=2)
    check("启发式重排", len(top) == 2 and top[0].rerank_score >= top[-1].rerank_score)


# ==================================================================
def test_router_agent() -> None:
    print("\n[4] 路由 与 ReAct 解析")
    from src.agent import QueryRouter, ReActAgent, ToolRegistry, Tool

    r = QueryRouter()
    check("计算类路由", r.route("(88.3-85.1)/85.1*100").tool_name == "calculator")
    check("时间类路由", r.route("今天几号").tool_name == "get_current_time")
    check("寒暄直答", r.route("你好").route == "direct")
    check("论文问题走 Agent", r.route("这篇论文用了什么数据集").route == "agent")

    reg = ToolRegistry()
    reg.register(Tool(name="calculator", description="计算", parameters={},
                      func=lambda a: "42"))
    agent = ReActAgent.__new__(ReActAgent)   # 只测解析，不走完整构造
    cases = [
        ("Thought: 需要计算\nAction: calculator\nAction Input: {\"expression\": \"1+1\"}", "calculator"),
        ("Thought: 思考\nAction: [\"calculator\"]\nAction Input: [{}]", "calculator"),
        ("Thought: 完成\nFinal Answer: 答案是2", ""),
    ]
    for text, expect in cases:
        _, actions, final = agent._parse_output(text)
        ok = (actions[0][0] == expect) if expect else (final == "答案是2")
        check(f"解析: {text.splitlines()[1][:38]}", ok)

    from src.agent.tools import safe_eval
    check("安全计算 1+2*3", safe_eval("1+2*3") == 7.0)
    try:
        safe_eval("__import__('os').system('echo hi')")
        check("注入被拦截", False)
    except Exception:
        check("注入被拦截", True)


# ==================================================================
def main() -> int:
    print("=" * 62)
    print(" 智能科研助理 · 离线冒烟测试（无需 API Key）")
    print("=" * 62)
    try:
        test_database()
        test_chunking()
        test_retrieval()
        test_router_agent()
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n{FAIL} 测试中断: {exc}")
        return 1
    p, f = _results["pass"], _results["fail"]
    print("\n" + "=" * 62)
    print(f" 结果：通过 {p} 项，失败 {f} 项")
    print("=" * 62)
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

