# -*- coding: utf-8 -*-
"""
pytest 公共夹具
---------------
测试原则：用 Stub 替换 DashScope 的 LLM / Embedding，
使单元测试完全离线、可重复，不需要 API Key。
需要真实模型的端到端测试标记为 @pytest.mark.integration，默认跳过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import get_chunker                    # noqa: E402
from src.data_loader import Document, Page              # noqa: E402


class StubEmbedder:
    """确定性伪向量：含相同字符越多越相似，足以验证检索链路。"""

    def __init__(self, dim: int = 48):
        self._dim = dim
        self.model_name = "stub-embedder"

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


class StubLLM:
    """可控的假 LLM：按脚本依次返回预设回复。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []
        self.model = "stub-llm"
        self.temperature = 0.2

    @property
    def name(self) -> str:
        return "StubLLM"

    def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        if not self.replies:
            return "Final Answer: 这是桩模型的默认回答"
        return self.replies.pop(0)

    def stream(self, messages, **kwargs):
        yield self.chat(messages, **kwargs)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 2)


PAPER_A = "Vision Transformer 在 ImageNet-1K 上达到 84.5% 的 Top-1 准确率，" \
          "使用 AdamW 优化器，训练 300 个 epoch。"
PAPER_B = "DiffSeg 基于扩散模型进行医学图像分割，在 BraTS2021 上 Dice 系数为 87.3%。"


@pytest.fixture
def sample_docs():
    return [
        Document(doc_id="doc_a", title="ViT论文", source="./a.md", file_type="md",
                 pages=[Page(page=1, text=PAPER_A * 3)], metadata={"file_name": "a.md"}),
        Document(doc_id="doc_b", title="扩散模型论文", source="./b.md", file_type="md",
                 pages=[Page(page=1, text=PAPER_B * 3)], metadata={"file_name": "b.md"}),
    ]


@pytest.fixture
def sample_chunks(sample_docs):
    return get_chunker("recursive", chunk_size=150, chunk_overlap=15).split_documents(sample_docs)


@pytest.fixture
def stub_embedder():
    return StubEmbedder()


@pytest.fixture
def stub_llm():
    return StubLLM()
