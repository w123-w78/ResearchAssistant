# -*- coding: utf-8 -*-
"""
检索质量评估指标（模块五 5.5.2 使用）
-------------------------------------
实现 IR 领域四个标准指标：
  Hit@K     : Top-K 中是否至少命中一条相关文档（最直观的"召回成功与否"）
  Recall@K  : Top-K 命中的相关文档数 / 全部相关文档数
  MRR       : 第一条相关结果的排名倒数的平均（衡量"好结果排得够不够前"）
  Precision@K: Top-K 中相关文档占比（衡量"噪声多不多"）

评测集格式（reports/评测集.json）：
    [{"question": "...", "relevant_doc_ids": ["xxx", ...], "type": "事实性", ...}, ...]
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .base import RetrievedChunk


def hit_at_k(retrieved: Sequence[RetrievedChunk], relevant: Sequence[str], k: int = 5) -> float:
    """Hit@K：Top-K 命中即 1，否则 0。"""
    top = [r.doc_id for r in retrieved[:k]]
    return 1.0 if any(d in top for d in relevant) else 0.0


def recall_at_k(retrieved: Sequence[RetrievedChunk], relevant: Sequence[str], k: int = 5) -> float:
    """Recall@K。"""
    if not relevant:
        return 0.0
    top = [r.doc_id for r in retrieved[:k]]
    return len(set(top) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[RetrievedChunk], relevant: Sequence[str], k: int = 5) -> float:
    """Precision@K。"""
    top = [r.doc_id for r in retrieved[:k]]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def mrr(retrieved: Sequence[RetrievedChunk], relevant: Sequence[str]) -> float:
    """MRR（Mean Reciprocal Rank）：1 / 第一条相关结果的排名。"""
    for i, r in enumerate(retrieved, start=1):
        if r.doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(retriever, eval_set: List[Dict[str, Any]],
                       k: int = 5, with_rerank: bool = True,
                       verbose: bool = False, ks=(1, 3, 5)) -> Dict[str, Any]:
    """
    在评测集上跑检索并汇总指标。

    :param retriever: HybridRetriever 实例
    :param eval_set:  评测集，每项需含 question / relevant_doc_ids
    :return: 汇总字典 + 逐条明细（便于 Bad Case 分析）
    """
    hits, recalls, precs, mrrs = [], [], [], []
    hit_by_k = {kk: [] for kk in ks}
    details: List[Dict[str, Any]] = []
    for item in eval_set:
        q = item.get("question", "")
        relevant = item.get("relevant_doc_ids") or item.get("relevant_docs") or []
        results = retriever.search(q, top_k=max(max(ks) * 4, 20), with_rerank=with_rerank)
        h = hit_at_k(results, relevant, k)
        r = recall_at_k(results, relevant, k)
        p = precision_at_k(results, relevant, k)
        m = mrr(results, relevant)
        hits.append(h); recalls.append(r); precs.append(p); mrrs.append(m)
        for kk in ks:
            hit_by_k[kk].append(hit_at_k(results, relevant, kk))
        details.append({
            "question": q,
            "type": item.get("type", ""),
            "hit": h, "recall": round(r, 4), "precision": round(p, 4), "mrr": round(m, 4),
            "hit@1": hit_by_k[1][-1] if 1 in ks else None,
            "top1": results[0].doc_id if results else None,
            "top1_score": round(results[0].score, 4) if results else 0.0,
            "relevant": relevant,
            "is_bad_case": h == 0,
        })
        if verbose:
            print(f"[{'HIT ' if h else 'MISS'}] {q[:40]}")

    n = max(len(eval_set), 1)
    out = {
        "k": k,
        "num_cases": len(eval_set),
        "hit@k": round(sum(hits) / n, 4),
        "recall@k": round(sum(recalls) / n, 4),
        "precision@k": round(sum(precs) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "bad_cases": [d for d in details if d["is_bad_case"]],
        "details": details,
    }
    # 同时给出多个 k 的命中率（小语料下 Hit@5 易饱和，Hit@1 更有区分度）
    for kk in ks:
        out[f"hit@{kk}"] = round(sum(hit_by_k[kk]) / n, 4)
    return out
