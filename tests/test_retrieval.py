# -*- coding: utf-8 -*-
"""检索层单元测试（模块一）"""
from __future__ import annotations

import numpy as np
import pytest

from src.chunking import get_chunker
from src.retrieval import (BM25Retriever, HybridRetriever, Reranker, VectorStore)
from src.retrieval.base import reciprocal_rank_fusion
from src.retrieval.evaluate import evaluate_retrieval, hit_at_k, mrr, recall_at_k


def test_all_chunkers_produce_chunks(sample_docs):
    for strategy in ("fixed", "recursive", "semantic", "academic"):
        chunks = get_chunker(strategy, chunk_size=200, chunk_overlap=20).split_documents(sample_docs)
        assert len(chunks) > 0, f"{strategy} 未产生分块"
        assert all(c.text.strip() for c in chunks)
        assert all(c.doc_id and c.chunk_id for c in chunks)


def test_recursive_respects_chunk_size(sample_docs):
    chunks = get_chunker("recursive", chunk_size=150, chunk_overlap=10).split_documents(sample_docs)
    assert all(c.char_count <= 200 for c in chunks)   # 允许少量超长兜底


def test_vector_store_add_and_search(sample_chunks, stub_embedder):
    store = VectorStore(dim=stub_embedder.dim)
    vecs = stub_embedder.encode([c.text for c in sample_chunks])
    store.add(sample_chunks, vecs)
    assert store.size == len(sample_chunks)

    hits = store.search(stub_embedder.encode_query("扩散模型"), top_k=2)
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]          # 分数降序

    removed = store.delete_by_doc("doc_a")
    assert removed > 0 and store.size < len(sample_chunks)


def test_bm25_retrieval(sample_chunks):
    bm = BM25Retriever()
    bm.build(sample_chunks)
    hits = bm.search("扩散模型 Dice 系数", top_k=3)
    assert hits and hits[0][1] > 0
    ids = [cid for cid, _ in hits]
    assert any(cid.startswith("doc_b") for cid in ids), "BM25 未召回正确文档"


def test_rrf_fusion():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert fused["b"] > fused["a"]            # 两路都命中的排第一
    assert fused["a"] > fused["c"]
    assert abs(fused["a"] - (1 / 61)) < 1e-9  # RRF 公式校验


def test_hybrid_retriever_end_to_end(sample_chunks, stub_embedder):
    retr = HybridRetriever(embedder=stub_embedder, reranker=Reranker(provider="none"))
    retr.build(sample_chunks)
    assert retr.size == len(sample_chunks)

    hits = retr.search("扩散模型的 Dice 系数", with_rerank=False)
    assert len(hits) > 0
    assert hits[0].chunk.doc_id == "doc_b"
    assert hits[0].source in ("both", "vector", "bm25")
    assert hits[0].rrf_score > 0

    docs = retr.list_documents()
    assert len(docs) == 2
    assert retr.delete_by_doc("doc_b") > 0


def test_reranker_heuristic(sample_chunks, stub_embedder):
    retr = HybridRetriever(embedder=stub_embedder, reranker=Reranker(provider="none"))
    retr.build(sample_chunks)
    candidates = retr.search("扩散模型", with_rerank=False)
    rr = Reranker(provider="heuristic")
    top = rr.rerank("扩散模型 Dice", candidates, top_n=2)
    assert len(top) == 2
    assert top[0].rerank_score >= top[-1].rerank_score
    assert all(c.source == "rerank" for c in top)


def test_evaluate_metrics(sample_chunks, stub_embedder):
    retr = HybridRetriever(embedder=stub_embedder, reranker=Reranker(provider="none"))
    retr.build(sample_chunks)
    eval_set = [
        {"question": "扩散模型的 Dice 系数", "relevant_doc_ids": ["doc_b"], "type": "事实性"},
        {"question": "Vision Transformer 准确率", "relevant_doc_ids": ["doc_a"], "type": "事实性"},
        {"question": "完全不相关的问题xyz", "relevant_doc_ids": ["doc_none"], "type": "干扰项"},
    ]
    result = evaluate_retrieval(retr, eval_set, k=3, with_rerank=False)
    assert result["num_cases"] == 3
    assert result["hit@k"] >= 2 / 3
    assert 0.0 <= result["mrr"] <= 1.0
    assert len(result["bad_cases"]) >= 1       # 干扰项应为 Bad Case


def test_metrics_functions(sample_chunks, stub_embedder):
    retr = HybridRetriever(embedder=stub_embedder, reranker=Reranker(provider="none"))
    retr.build(sample_chunks)
    hits = retr.search("扩散模型", with_rerank=False)
    assert hit_at_k(hits, ["doc_b"], k=1) == 1.0
    assert mrr(hits, ["doc_b"]) == 1.0
    assert recall_at_k(hits, ["doc_b"], k=5) == 1.0
