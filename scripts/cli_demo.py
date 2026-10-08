# -*- coding: utf-8 -*-
"""
命令行演示入口（无需 Streamlit，便于服务器 / CI 环境验证）

用法：
    # 默认 mock 模式（无需任何模型，离线可跑）
    python scripts/cli_demo.py

    # 指定真实模型
    python scripts/cli_demo.py --llm ollama --model qwen2.5:7b

    # 纯 RAG 模式（不走 Agent）
    python scripts/cli_demo.py --no-agent

    # 交互式提问
    python scripts/cli_demo.py -i
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.system import ResearchAssistantSystem  # noqa: E402
from src.utils.config import Config  # noqa: E402


def build_system(args) -> ResearchAssistantSystem:
    from pathlib import Path

    cfg_path = str(Path(__file__).resolve().parent.parent / "config.yaml")
    cfg = Config.load(args.config or cfg_path)
    if args.model:
        cfg.set("llm.model", args.model)
    if args.db:
        cfg.set("storage.db_path", args.db)
    if args.no_rerank:
        cfg.set("retrieval.enable_rerank", False)
        cfg.set("retrieval.reranker.provider", "none")
    if args.heuristic_rerank:
        cfg.set("retrieval.reranker.provider", "heuristic")
    return ResearchAssistantSystem(config=cfg)


def ingest_demo_data(system: ResearchAssistantSystem, corpus_dir: str) -> None:
    """导入示例语料；若目录不存在则自动生成。"""
    from scripts.make_sample_papers import generate

    p = Path(corpus_dir)
    if not p.exists() or not list(p.glob("*.md")):
        print("[初始化] 未发现示例语料，自动生成 3 篇示例论文...")
        generate(corpus_dir)

    files = [str(f) for f in sorted(Path(corpus_dir).glob("*"))
             if f.suffix.lower() in (".md", ".txt", ".pdf", ".docx")]

    def progress(cur, total, path, status):
        icon = {"ok": "✓", "fail": "✗", "skip": "⊘"}.get(status, "·")
        print(f"  {icon} [{cur}/{total}] {Path(path).name}")

    print(f"\n[导入] 共 {len(files)} 个文件")
    summary = system.ingest_files(files, on_progress=progress)
    print(f"[导入] 成功 {summary['success']} / 失败 {summary['failed']} / 跳过 {summary['skipped']}"
          f"，共 {summary['total_chunks']} 个文本块\n")


DEMO_QUESTIONS = [
    "Vision Transformer 在 ImageNet 上的 Top-1 准确率是多少？",
    "DiffSeg 方法在 BraTS2021 数据集上的 Dice 系数是多少？",
    "混合检索相比朴素 RAG 的 Hit@5 提升了多少个百分点？",
    "知识库里一共有几篇论文？",
    "(88.3-85.1)/85.1*100 等于多少？",
]


def main():
    parser = argparse.ArgumentParser(description="智能科研助理 - 命令行演示")
    parser.add_argument("--corpus", default="./data/sample_papers", help="语料目录")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    parser.add_argument("--model", default=None, help="通义千问模型名，默认 qwen-plus")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径")
    parser.add_argument("--no-agent", action="store_true", help="不走 Agent，纯 RAG")
    parser.add_argument("--no-rerank", action="store_true", help="关闭重排序")
    parser.add_argument("--heuristic-rerank", action="store_true",
                        help="重排序改用本地启发式（不调用在线 Rerank）")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互式提问")
    args = parser.parse_args()

    system = build_system(args)
    print("=" * 70)
    print(" 智能科研助理 —— 基于 RAG + Agent 的论文知识库问答系统")
    print(f" LLM: {system.llm.name} | Embedding: {system.embedder.model_name}"
          f" | 分块: {system.chunker.name}")
    print("=" * 70)

    ingest_demo_data(system, args.corpus)

    # 健康检查
    health = system.health_check()
    print(f"[健康检查] 整体状态: {health['overall']} | LLM: {health['llm']['status']}"
          f" | 向量库: {health['vector_store']['num_vectors']} 条\n")

    questions = DEMO_QUESTIONS
    if args.interactive:
        print("进入交互模式（输入 q 退出）")
        while True:
            q = input("\n你: ").strip()
            if q.lower() in ("q", "quit", "exit"):
                break
            if not q:
                continue
            ask_and_print(system, q, use_agent=not args.no_agent)
        return

    for q in questions:
        ask_and_print(system, q, use_agent=not args.no_agent)

    print_stats(system)


def ask_and_print(system: ResearchAssistantSystem, q: str, use_agent: bool = True) -> None:
    print("-" * 70)
    print(f"问: {q}")
    result = system.ask(q, session_id="cli", use_agent=use_agent)
    mode = result["meta"].get("mode", "agent")
    if mode == "agent":
        print(f"[路由] {result['meta'].get('route', '-')} | "
              f"迭代 {result['meta'].get('iterations', 0)} 轮 | "
              f"工具调用 {result['meta'].get('tool_calls', 0)} 次")
        for s in result["steps"]:
            if s.get("action"):
                print(f"  · Action: {s['action']}")
                obs = s.get("observation", "")[:120].replace("\n", " ")
                print(f"    └─ {obs}...")
    print(f"答: {result['answer'][:600]}")
    if result["citations"]:
        print("引用: " + "; ".join(f"[{c['index']}] {c.get('title','')[:30]}"
                                   for c in result["citations"][:4]))
    print()


def print_stats(system: ResearchAssistantSystem) -> None:
    print("=" * 70)
    print("系统统计")
    s = system.stats()
    print(f"  文档数: {s['documents']} | 文本块: {s['chunks']}")
    print(f"  检索层: {s['retriever']['backend']} / reranker={s['retriever']['reranker']}"
          f" / fusion={s['retriever']['fusion_mode']}")
    if "cache" in s:
        c = s["cache"]
        print(f"  语义缓存: 命中 {c['hits']} 次 / 未命中 {c['misses']} 次，命中率 {c['hit_rate']:.1%}")
    m = s["metrics"]
    print(f"  指标: LLM 调用 {m.get('llm.calls', 0)} 次，Token {m.get('llm.tokens', 0)}，"
          f"工具成功率 {m.get('tool.success_rate', 0):.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
