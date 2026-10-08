# -*- coding: utf-8 -*-
"""
模块一产出：检索性能基准测试 + 分块策略对比实验
------------------------------------------------
产出两组数据（模块五报告与答辩 PPT 直接引用）：
  1. 检索三档对比：纯向量 → 混合检索(BM25+RRF) → 混合+重排序
     指标：Hit@5 / MRR / 平均延迟
  2. 分块策略对比：fixed / recursive / semantic / academic × chunk_size
     指标：块数 / 平均块长 / 过短块占比 / 切分耗时

用法：
    python scripts/benchmark_retrieval.py                 # 用示例语料
    python scripts/benchmark_retrieval.py --corpus ./data/uploads
    python scripts/benchmark_retrieval.py --stub          # 无 API Key 时用伪向量跑通流程

结果写入 reports/性能评测数据.json（CSV 版同名 .csv）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import compare_chunkers, get_chunker          # noqa: E402
from src.data_loader import load_directory                      # noqa: E402
from src.retrieval import HybridRetriever, Reranker             # noqa: E402
from src.retrieval.evaluate import evaluate_retrieval           # noqa: E402
from src.utils.config import Config                             # noqa: E402

# 每个文档自动生成 3 类问题：事实性 / 方法性 / 结果性
QUESTION_TEMPLATES = [
    "{} 这篇论文使用了哪些数据集？",
    "{} 的核心方法是什么？",
    "{} 的实验结果和主要指标是多少？",
]


def load_eval_set(docs, path: str = "./reports/评测集.json"):
    """
    读取 build_evalset.py 生成的评测集，按标题映射到本次导入的 doc_id。
    返回 None 表示不可用（回退到自动生成）。
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    title2id = {d.title: d.doc_id for d in docs}
    out = []
    for q in data.get("questions", []):
        ids = [title2id[t] for t in q.get("relevant_titles", []) if t in title2id]
        if not ids:
            continue      # 标题对不上（语料换了）就跳过
        out.append({**q, "relevant_doc_ids": ids})
    return out or None


def auto_eval_set(docs) -> list:
    """按实际文档自动构造"问题 -> doc_id"评测对（无需手填 doc_id）。"""
    out = []
    for d in docs:
        for tpl in QUESTION_TEMPLATES:
            out.append({
                "question": tpl.format(d.title),
                "relevant_doc_ids": [d.doc_id],
                "type": "自动构造",
            })
    return out


def run(corpus: str, stub: bool, out_dir: str, chunk_size: int = 256) -> dict:
    cfg = Config.load(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    docs = load_directory(corpus)
    if not docs:
        raise SystemExit(f"语料为空：{corpus}（可先运行 scripts/make_sample_papers.py）")

    emb = _embedder(cfg, stub)
    # 优先用正式评测集（题目不含标题，有区分度）；否则回退自动生成
    eval_set = load_eval_set(docs) or auto_eval_set(docs)
    print(f"语料 {len(docs)} 篇，评测 {len(eval_set)} 条")

    # ---- 1) 检索三档对比 ----
    rows = []
    last_result: dict = {}
    for name, use_bm25, rerank in (("纯向量", False, None),
                                   ("混合检索(RRF)", True, None),
                                   ("混合+重排序", True, "on")):
        chunks = get_chunker(cfg.get("chunking.strategy", "recursive"),
                             chunk_size=chunk_size,
                             chunk_overlap=max(32, chunk_size // 8)).split_documents(docs)
        retr = HybridRetriever(embedder=emb, config=cfg,
                               reranker=Reranker(provider="heuristic" if rerank else "none",
                                                top_n=int(cfg.get("retrieval.final_top_k", 5))))
        retr.build(chunks)
        if not use_bm25:      # 纯向量档：把 BM25 权重置 0
            retr.vector_weight, retr.bm25_weight = 1.0, 0.0
            retr.fusion_mode = "weighted"
        r = evaluate_retrieval(retr, eval_set, k=5, with_rerank=bool(rerank))
        last_result = r
        rows.append({"档位": name,
                     "Hit@1": r.get("hit@1", 0), "Hit@3": r.get("hit@3", 0),
                     "Hit@5": r["hit@k"], "MRR": r["mrr"],
                     "Recall@5": r["recall@k"], "块数": len(chunks)})
        print(f"  {name:<14} Hit@1={r.get('hit@1', 0):.3f}  Hit@3={r.get('hit@3', 0):.3f}"
              f"  Hit@5={r['hit@k']:.3f}  MRR={r['mrr']:.3f}")

    # ---- 2) 分块策略对比 ----
    chunk_rows = compare_chunkers(docs, chunk_sizes=[256, 512, 1024])
    print(f"分块对比：{len(chunk_rows)} 组")

    result = {"corpus": corpus, "num_docs": len(docs), "eval_size": len(eval_set),
              "chunk_size": chunk_size, "retrieval": rows, "chunking": chunk_rows,
              "bad_cases": last_result.get("bad_cases", [])[:20] if rows else []}
    _save(result, out_dir)
    return result


def _embedder(cfg, stub: bool):
    if stub:
        sys.path.insert(0, str(Path(__file__).resolve().parent / ".."))
        from tests.conftest import StubEmbedder
        print("使用 StubEmbedder（离线模式，数据仅用于验证流程）")
        return StubEmbedder()
    from src.utils.embedding import build_embedder
    return build_embedder(cfg)


def _save(result: dict, out_dir: str) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "性能评测数据.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(p / "性能评测数据.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["档位", "Hit@1", "Hit@3", "Hit@5", "MRR", "Recall@5", "块数"])
        w.writeheader()
        for r in result["retrieval"]:
            w.writerow(r)
    print(f"\n结果已写入 {p / '性能评测数据.json'} / .csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="./data/sample_papers")
    ap.add_argument("--stub", action="store_true", help="无 API Key 时用伪向量跑通流程")
    ap.add_argument("--out", default="./reports")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="评测用分块大小，默认 256（块数更多，指标区分度更好）")
    a = ap.parse_args()
    run(a.corpus, a.stub, a.out, a.chunk_size)
